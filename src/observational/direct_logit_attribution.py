from typing import List, Union, Literal, Optional
import plotly.express as px
import numpy as np
from fancy_einsum import einsum
import einops
import torch
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float
from transformer_lens import HookedTransformer, ActivationCache

import sys
sys.path.append("../")
from src.patching.act_patching import run_model, hook_save_head_output, hook_save_attention_output, hook_save_mlp_output, hook_save_layer_output, get_top_k_strings

def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    per_prompt: bool = False
) -> Float[Tensor, "*batch"]:
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


def residual_stack_to_logit_diff(
            residual_stack: Float[Tensor, "components batch d_model"],
            act_cache: ActivationCache,
            logit_diff_directions: Float[Tensor, "batch d_model"],
            prompts: List[str]
        ) -> float:
            # Applies layer normalization scaling to the residual stack (division by norm) returns residual_stack / scale 
            # which is a global property of the Transformer
            scaled_residual_stack = act_cache.apply_ln_to_stack(
                residual_stack, layer=-1, pos_slice=-1
            )
            return einsum(
                "... batch d_model, batch d_model -> ...",
                scaled_residual_stack,
                logit_diff_directions,
            ) / len(prompts)


def direct_logit_attribution(
            tl_model: HookedTransformer, 
            prompts: List[str], 
            act_cache: ActivationCache, 
            answer_tokens_ids: List[List[int]], 
            decomposition: Union[None, Literal['residual_stream', 'layer_blocks', 'attention_heads']] = None        
            ) -> float:
        assert decomposition in ['residual_stream', 'layer_blocks', 'attention_heads'], \
            f"Decomposition type must be one of [None, 'residual_stream', 'layer_blocks', 'attention_heads'], but got {decomposition}."
        
        # Returns a stack of mapped answer tokens (correct and wrong) to a tensor with the unembedding vector for those tokens 
        # (W_U[:, token] of shape d_model)
        answer_residual_directions = tl_model.tokens_to_residual_directions(answer_tokens_ids) # shape (batch_size, 2, d_model), where the 2nd dim is the correct and wrong answer
        # print("Answer residual directions shape:", answer_residual_directions.shape)
        
        # Calculate the difference between the logits of the two answers
        logit_diff_directions = (
            answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
        )
        
        # Accumulate the residuals for the last layer
        accumulated_residual, labels = act_cache.accumulated_resid(
            layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
        )

        if decomposition == "residual_stream":
            # Decompose the residual stream input to layer L into the sum of the outputs of previous layers
            logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, act_cache, logit_diff_directions, prompts)

        elif decomposition == "layer_blocks":
            # Decompose the residual stream input to layer L into the sum of the outputs of previous compoments (plus W_E and W_pos_emb)
            per_layer_residual, labels = act_cache.decompose_resid(
            layer=-1, pos_slice=-1, return_labels=True
            )
            logit_lens_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, act_cache, logit_diff_directions, prompts)
        
        elif decomposition == "attention_heads": 
            per_head_residual, labels = act_cache.stack_head_results(
                layer=-1, pos_slice=-1, return_labels=True
            )

            logit_lens_logit_diffs = residual_stack_to_logit_diff(per_head_residual, act_cache, logit_diff_directions, prompts)
            logit_lens_logit_diffs = einops.rearrange(
                logit_lens_logit_diffs,
                "(layer head_index) -> layer head_index",
                layer=tl_model.cfg.n_layers,
                head_index=tl_model.cfg.n_heads,
            )
        return logit_lens_logit_diffs, labels



def visual_logit_lens(
    tl_model: HookedTransformer,
    prompts: Union[str, List[str]],
    decomposition: Union[None, Literal['resid_post', 'attention_out', 'mlp_out', 'attention_heads']] = None,
    k: int = 3,
    logits_or_probs: Literal['logits', 'probs'] = None,
    titles: Optional[List[str]] = None
):
    device = tl_model.cfg.device
    
    if isinstance(prompts, str):
        prompts = [prompts]

    # Tokenize
    tokens = tl_model.to_tokens(prompts, prepend_bos=True).to(device)
    correct_ids = tokens[:, 1:]  # Shifted next-token labels
    tokens = tokens[:, :-1]      # Predict next token from each input

    n_tokens = tokens.shape[1]
    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads
    vocab_size = tl_model.cfg.d_vocab

    assert correct_ids.max() < vocab_size
    assert correct_ids.min() >= 0

    unembed = torch.nn.Sequential(
        tl_model.ln_final,
        tl_model.unembed,
    )

    # Hook setup
    head_out_arrs = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    out_arrs = [[] for _ in range(n_layers)]

    hooks = []
    if decomposition == 'attention_heads':
        for l in range(n_layers):
            for h in range(n_heads):
                hooks.append(hook_save_head_output(tl_model, l, h, head_out_arrs[l][h], include_bias=False))
    if decomposition in ['attention_out', 'mlp_out', 'resid_post']:
        for l in range(n_layers):
            hooks.append(hook_save_attention_output(tl_model, l, out_arrs[l], include_bias=False))
            hooks.append(hook_save_mlp_output(tl_model, l, out_arrs[l], include_bias=False))
            hooks.append(hook_save_layer_output(l, out_arrs[l]))

    # Forward pass
    _ = run_model(tl_model, tokens, hooks, split_qkv=False)

    # Initialize display matrices
    n_heads_or_1 = n_heads if decomposition == 'attention_heads' else 1
    str_matrix = np.empty((n_tokens, n_layers, n_heads_or_1), dtype=object)
    prob_matrices = torch.zeros(n_tokens, n_layers, n_heads_or_1, k, device=device)
    logits_matrices = torch.zeros(n_tokens, n_layers, n_heads_or_1, k, device=device)
    correct_rank_matrix = torch.full(
            (n_tokens, n_layers, n_heads_or_1),
            fill_value=-1,  # use -1 for padding / not found
            dtype=torch.long,
            device=device
        )

    for l in range(n_layers):
        for tok_idx in range(n_tokens):
            if decomposition == 'attention_heads':
                for h in range(n_heads):
                    out = head_out_arrs[l][h][0]
                    logits = unembed(out)[0, tok_idx, :]
                    probs = F.softmax(logits, dim=-1)
                    
                    sorted_indices = torch.argsort(probs, descending=True)
                    correct_token_id = correct_ids[0, tok_idx]

                    # Get rank of the correct token
                    rank = (sorted_indices == correct_token_id).nonzero(as_tuple=True)[0].item()
                    # Store rank
                    correct_rank_matrix[tok_idx, l, h] = rank  # or h=0 for component mode
                    
                    top_toks, (logit_tops, prob_tops) = get_top_k_strings(tl_model, logits.unsqueeze(0), k=k, probs=probs.unsqueeze(0), use_br=True)

                    str_matrix[tok_idx, l, h] = top_toks[0]
                    prob_matrices[tok_idx, l, h] = prob_tops[0]    # shape: (k,)
                    logits_matrices[tok_idx, l, h] = logit_tops[0] # shape: (k,)
                    
            else:
                out_attn = out_arrs[l][0]
                out_mlp = out_arrs[l][1]
                out_resid_post = out_arrs[l][2] 
                out = None
                if decomposition == 'attention_out':
                    out = out_attn
                elif decomposition == 'mlp_out':
                    out = out_mlp
                elif decomposition == 'resid_post':
                    out = out_resid_post

                logits = unembed(out)[0, tok_idx, :]
                probs = F.softmax(logits, dim=-1)
                
                sorted_indices = torch.argsort(probs, descending=True)
                correct_token_id = correct_ids[0, tok_idx]

                # Get rank of the correct token
                rank = (sorted_indices == correct_token_id).nonzero(as_tuple=True)[0].item()
                # Store rank
                correct_rank_matrix[tok_idx, l, 0] = rank  # or h=0 for component mode

                top_toks, (logit_tops, prob_tops) = get_top_k_strings(tl_model, logits.unsqueeze(0), k=k, probs=probs.unsqueeze(0), use_br=True)

                str_matrix[tok_idx, l, 0] = top_toks[0]
                prob_matrices[tok_idx, l, 0] = prob_tops[0]    # shape: (k,)
                logits_matrices[tok_idx, l, 0] = logit_tops[0] # shape: (k,)

    # Plot all token transitions
    for token_idx in range(n_tokens):
        # Build a new string matrix with all k values per cell
        plot_text = np.empty((n_layers, n_heads_or_1), dtype=object)

        for l in range(n_layers):
            for h in range(n_heads_or_1):
                token_strs = str_matrix[token_idx, l, h]         # a string with <br /> (e.g. "the<br />of<br />and<br />")
                token_logits = logits_matrices[token_idx, l, h]  # shape: (k,)
                token_probs = prob_matrices[token_idx, l, h]     # shape: (1,) or (k,)
                # Split string into list of tokens using <br /> to align with logits
                split_tokens = token_strs.strip().split("<br />")
                split_tokens = [t for t in split_tokens if t != ""]  # remove empty entries
                split_tokens = ['EOT' if tok == '<|endoftext|>' else tok for tok in split_tokens]
                
                # --- Format display string per mode ---
                lines = []
                for t, logit, prob, in zip(split_tokens, token_logits, token_probs):
                    if logits_or_probs == 'logits':
                        value = f"{logit:.2f}"
                    elif logits_or_probs == 'probs':
                        value = f"{prob:.2%}"
                    else:
                        value = "?"
                    lines.append(f"{t}: {value}")
                plot_text[l, h] = "<br>".join(lines)


        # Plot
        heatmap_values = -correct_rank_matrix[token_idx].detach().cpu().T  # shape: (h, l)
        # color_scale = px.colors.sequential.Cividis_r
        color_scale = [(0, "#1c1c1c"), (0.5, "#661010"), (1.0, "#cc3333")]

        fig = px.imshow(
            heatmap_values,
            color_continuous_scale=color_scale, 
            origin="lower",
            height=1100 if decomposition == 'attention_heads' else 300,
            width=130 * n_layers,
            labels=dict(x="Layer", y="Head" if decomposition == 'attention_heads' else "Component")
        )

        fig.update_traces(
            # text=str_matrix[token_idx].T,
            text=plot_text.T.tolist(),
            texttemplate="%{text}",
            textfont=dict(family="Courier New", size=11, color="white")
        )

        title = titles[token_idx] if titles else f"Token transition {token_idx}"
        fig.update_layout(title_text=title, title_x=0.5)
        fig.update_layout(coloraxis_colorbar=dict(title="–Rank of correct token"))


        fig.update_xaxes(
            tickmode="array",
            tickvals=np.arange(n_layers),
            ticktext=[f"Layer {i}" for i in range(n_layers)],
            title_text="Layer"
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=np.arange(n_heads) if decomposition == 'attention_heads' else [0],
            ticktext=[f"Head {i}" for i in range(n_heads)] if decomposition == 'attention_heads' else ["Output"],
            title_text="Head" if decomposition == 'attention_heads' else "Component"
        )

        fig.show()
