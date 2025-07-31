from typing import List, Tuple, Optional, Union
from typing_extensions import Literal
from jaxtyping import Float
import plotly.express as px
import plotly.graph_objects as go
import re
from fancy_einsum import einsum
from circuitsvis.attention import attention_heads

import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils


def find_token_positions(words, highlights):
    """Robust substring-based matching for highlight words."""
    highlights_set = set(h.lower() for h in highlights)
    token_hits = []
    for i, tok in enumerate(words):
        clean_tok = tok.strip("Ġ▁").lower()
        if any(h in clean_tok for h in highlights_set):
            token_hits.append(i)
    return token_hits


def show_attention_patterns(
    model: HookedTransformer,
    heads: List[Tuple[int, int]],
    prompts: Union[str, List[str]],
    precomputed_cache: Optional[ActivationCache] = None,
    mode: Union[Literal["val", "pattern", "scores", "ov"]] = None,
    effective_ov: bool = False,
    highlight_words: list = None,
    show_all_tokens: bool = True,
    title_suffix: Optional[str] = "",
    return_fig: bool = False,
    return_mtx: bool = False,
):
    """
    Visualizes the different types of attention for the specified heads in the model. This function is adapted from [Easy-Transformer repo](https://github.com/redwoodresearch/Easy-Transformer/blob/main/easy_transformer/ioi_utils.py#L147),
    a companion to the IOI paper. 

    Args:

    model (torch.nn.Module): Model to visualize.
    heads (List[Tuple[int, int]]): List of tuples specifying the layer and head indices to visualize.
    prompt (Union[str, List[str]]): Prompt used for visualization.
    precomputed_cache (Dict[str, torch.Tensor], optional): Precomputed activations cache.
    mode (str): Visualization mode ('pattern', 'val', 'scores').
    effective_ov (bool): Whether to use effective OV circuit. If False, compute the vanilla OV circuit.
    title_suffix (str): Suffix to append to the plot title.
    return_fig (bool): Whether to return the plotly figure.
    return_mtx (bool): Whether to return the attention matrices.

    Returns:
    - If return_fig=True and return_mtx=False, returns the plotly figure.
    - If return_fig=False and return_mtx=True, returns the attention matrices.
    - If return_fig=False and return_mtx=False, displays the attention patterns.

    Info:
    - 'scores': Visualizes the attention scores pre-softmax, 
    - 'pattern': Visualizes the attention patterns post-softmax, or attention probabilities,
    - 'val-weighted': Visualizes the value-weighted attention patterns,
    - 'ov': Visualizes the OV circuit

    Note: 
    ! More about the types of activations on:  https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html
    """
    assert mode in [
        "pattern",
        "scores",
        "val-weighted",
        "ov",
    ] 
    assert len(heads) == 1 or not (return_fig or return_mtx)

    # for (layer, head) in heads:
    #     cache = {}
    #     good_names = []
    #     good_names.append(f"blocks.{layer}.attn.hook_v")  # (batch, pos, head_index, d_head)
    #     good_names.append(f"blocks.{layer}.attn.hook_pattern")  # (batch, head_index, query_pos, key_pos)
    #     good_names.append(f"blocks.{layer}.attn.hook_attn_scores")  # (batch, head_index, query_pos, key_pos)
    #     good_names.append(f"blocks.{layer}.attn.hook_z")  #  (batch, pos, head_index, d_head)
    #     good_names.append(f"blocks.{layer}.hook_attn_out") 
    #     good_names.append(f"blocks.0.hook_resid_pre")   
    #     good_names.append(f"blocks.0.hook_mlp_out")  # (batch, pos, d_mlp), where d_mlp = 4 * d_model
        
    #     if precomputed_cache is None:
    #         cache = {}
    #         def hook_fn(activation: torch.Tensor, hook: HookPoint, name: str = "activation"):
    #             cache[name] = activation
    #             return activation

    #         fwd_hooks = [
    #             (good_names[0], partial(hook_fn, name=good_names[0])), 
    #             (good_names[1], partial(hook_fn, name=good_names[1])), 
    #             (good_names[2], partial(hook_fn, name=good_names[2])), 
    #             (good_names[3], partial(hook_fn, name=good_names[3])), 
    #             (good_names[4], partial(hook_fn, name=good_names[4])), 
    #             (good_names[5], partial(hook_fn, name=good_names[5])), 
    #             (good_names[6], partial(hook_fn, name=good_names[6])), 
    #         ]
    #         model.run_with_hooks(prompts, fwd_hooks=fwd_hooks)
    #     else:
            # cache = precomputed_cache
    if precomputed_cache is not None:
        cache = precomputed_cache
    else:
        _, cache = model.run_with_cache(prompts)

    toks = model.to_tokens(prompts)
    words = model.to_str_tokens(prompts)
    if isinstance(words[0], list):
        words = words[0]
    words = ['EOT' if word == '<|endoftext|>' else word for word in words]

    current_length = len(words)

    attn_results = torch.zeros(
        size=(len(prompts), current_length, current_length), dtype=torch.float16
    )

    layer, head = map(int, re.findall(r'\d+', str(heads[0])))
    attn_pattern = cache[utils.get_act_name('pattern', layer, 'a')][0, head, :, :].detach().cpu().squeeze(0)
    attn_scores = cache[utils.get_act_name('attn_scores', layer, 'a')][0, head, :, :].detach().cpu().squeeze(0)

    if not show_all_tokens:
        highlight_words = highlight_words or []
        highlight_ids = find_token_positions(words, highlight_words)
        step = max(current_length // 4, 1)
        regular_ticks = list(range(0, current_length, step))
        tickvals = sorted(set(highlight_ids + regular_ticks))
        ticktext = [f'{i}: "{words[i]}"' if i in highlight_ids else str(i) for i in tickvals]
    else:
        tickvals = list(range(current_length))
        ticktext = [f"{i}: {tok}" for i, tok in enumerate(words)]

    if mode == "val-weighted":
        ## FIXME: add support for models without GQA, keeping this as comment will work but uncomment for GQA models
        # if getattr(model.cfg, "ungroup_grouped_query_attention", True):
        #     # n_heads == n_key_value_heads
        #     v = cache[utils.get_act_name('v', layer, 'a')].detach().cpu()[0, :, head, :].norm(p=2,dim=-1)
        #     cont = attn_pattern * v.unsqueeze(0)

        # else:
        #     # use GQA kv_head grouping
        #     kv_group_size = model.cfg.n_heads // model.cfg.n_key_value_heads
        #     kv_head = head // kv_group_size
        v = cache[utils.get_act_name('v', layer, 'a')].detach().cpu()[0, :, head, :].norm(p=2,dim=-1)
        cont = attn_pattern * v.unsqueeze(0)

    labels={"y": "Queries", "x": "Keys"}

    if mode == "ov":
        # Declare weights
        # v = cache[utils.get_act_name('v', layer, 'a')][0, head, :, :].squeeze(0)  # (seq, d_head)
        W_V = model.W_V[layer, head]  
        W_O = model.W_O[layer, head]  
        W_OV = W_V @ W_O # (d_model, d_model)             
        W_E = model.W_E  # (vocab_size, d_model)        
        W_U = model.W_U  # (d_model, vocab_size)

        resid_mid = cache[utils.get_act_name('resid_mid', layer)]  # (1, seq, d_model)
        resid_pre = cache[utils.get_act_name('resid_pre', layer)]  # (1, seq, d_model)
        # NOTE: using the MLP output from the activation cache yields a slightly different ov plot, but the main diagonal is more or less preserved
        # However, if mlp_out from cache and model.0.mlp(normalized_resid_mid) are computed similarly, then the result should be the same, which it isn't.
        # torch.equal returns False, so the two activations are different
        mlp_out = cache[utils.get_act_name('mlp_out', layer)]  # (1, seq, d_model)

        # === Vanilla OV circuit ===
        if not effective_ov:
            ov_logits = W_E @ (W_OV) @ W_U   # (d_model, vocab_size)

        # === Effective OV circuit ===
        # EE = Resid pre -> LayerNorm -> MLP 
        # this is taken from https://github.com/callummcdougall/SERI-MATS-2023-Streamlit-pages/blob/main/transformer_lens/rs/callum2/utils.py#L249
        else:
            resid_mid = resid_pre            
            normalized_resid_mid = model.blocks[0].ln2(resid_mid)  
            # NOTE: applying the LN operation to the residual stream before Attention and using that as input to the MLP is different 
            # from using the cached MLP, because we effectively skip the Attention. 
            # assert torch.equal(mlp_out, model.blocks[0].mlp(model.blocks[0].ln2(resid_mid))), "mlp_out and model.blocks[0].mlp(normalized_resid_mid) are not identical"
            mlp_out = model.blocks[0].mlp(normalized_resid_mid)
            
            W_EE = resid_mid + mlp_out  # (1, seq, d_model)
            W_EE = W_EE.squeeze(0)  # (seq, d_model)

            # Effective logits from OV projection
            ov_logits = W_EE @ W_OV @ W_U  # (seq, vocab)

        # === Gather logit values for each (i,j) token pair ===
        if toks.ndim == 2:
            toks = toks[0]  # flatten from [1, seq_len] to [seq_len]
        toks = toks.tolist()  # convert to Python list of ints

        # Compute OV bigram scores
        cont_tmp = torch.zeros((len(toks), len(toks)), device=ov_logits.device)
        for i in range(len(toks)):
            for j in range(len(toks)):
                cont_tmp[i, j] = ov_logits[i, toks[j]]
        cont = cont_tmp.cpu()
        
    if return_fig and current_length < 100: 
        fig = px.imshow(
            attn_pattern if mode == "pattern" else attn_scores if mode == "scores" else cont,
            title=f"{layer}.{head} Attention" + " " + title_suffix if mode != "ov" else f"{layer}.{head} OV Logits",
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu" if mode != "ov" else "Viridis",
            labels={"y": "Output Token", "x": "Source Token"} if mode == "ov" else labels,
            height=600,
        )

        if mode == "ov":
            tickvals = list(range(current_length))
            ticktext = [f"{i}: {tok}" for i, tok in enumerate(words)]
            
            fig.update_layout(
            title=f"{layer}.{head} OV Logits" + " " + title_suffix,
            xaxis_title="Source Token",
            yaxis_title="Output Token",
            xaxis=dict(
                side="top",
                tickangle=45,
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=11),
                constrain='domain',
            ),
            yaxis=dict(
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=11),
                autorange="reversed",
                scaleanchor="x",  # ensures square aspect ratio
            ),
            width=1000,
            height=750,
            margin=dict(t=130, l=150)
            )
        else:
            fig.update_layout(
                xaxis=dict(
                    side="top", tickangle=45,
                    ticktext=ticktext,
                    tickvals=tickvals,
                    tickfont=dict(size=11),
                ),
                yaxis=dict(
                    ticktext=ticktext,
                    tickvals=tickvals,
                    tickfont=dict(size=11),
                ),
                width=1000,
                height=750,
                margin=dict(t=130, l=150)
            )

    if return_fig and not return_mtx:
        # NOTE: this was temporary for when trying to plot a big seq ~ ctx_len
        # if toks.shape[1] > 100:
        #     print(f"Sequence length = {toks.shape[1]} too large for interactive plot. Saving figure...")
        #     fig_path = f"figures/attention_layer{layer}_head{head}.png"
        #     if hasattr(fig, "write_image"):
        #         fig.write_image(fig_path)
        #     else:
        #         fig.savefig(fig_path)
        #         plt.close(fig)
        #     print(f"Saved to: {os.path.abspath(fig_path)}")
        # else:
            fig.show()

    elif return_mtx and not return_fig:
        # Return raw matrix
        if mode == "val-weighted":
            return cont
        elif mode == "pattern":
            attn_results[:, :current_length, :current_length] = (
                attn_pattern[:current_length, :current_length].clone().cpu()
            )
        elif mode == "scores":
            attn_results[:, :current_length, :current_length] = (
                attn_scores[:current_length, :current_length].clone().cpu()
            )
        elif mode == "ov":
            attn_results[:, :current_length, :current_length] = (
                cont[:current_length, :current_length].clone().cpu()
            )
        return attn_results
        

def show_attention_patterns_circuitsvis(
    model: HookedTransformer,
    heads: Union[List[int], int, Float[torch.Tensor, "heads"]],
    local_cache: ActivationCache,
    local_tokens: torch.Tensor,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
) -> str:
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: List[str] = []
    patterns: List[Float[torch.Tensor, "dest_pos src_pos"]] = []

    # Assume we have a single batch item
    batch_index = 0

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        patterns.append(local_cache["attn", layer][batch_index, head_index])

    # Convert the tokens to strings (for the axis labels)
    str_tokens = model.to_str_tokens(local_tokens)

    # Combine the patterns into a single tensor
    patterns: Float[torch.Tensor, "head_index dest_pos src_pos"] = torch.stack(
        patterns, dim=0
    )

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"

def avg_attention_positions(
    model: HookedTransformer, 
    prompts: List[str], 
    heads: List[Tuple[int, int]],
    position_dicts: List[dict], 
    title_text: str = None,
) -> None:

    tokens = model.to_tokens(prompts if isinstance(prompts, str) else prompts[0])
    average_attention = {}
    
    fig = go.Figure()
    for head in heads:
        average_attention[head] = {}
        cur_ys = []
        cur_stds = []
    
        att = torch.zeros(size=(len(prompts), tokens.shape[1], tokens.shape[1]))
        att += show_attention_patterns(model, [head], prompts, return_fig=False, return_mtx=True, mode="pattern")
        att /= len(head)
        
        for key in position_dicts[0].keys(): 
            end_to_target = []

            for i, pos_dict in enumerate(position_dicts):
                end_idx = pos_dict["end"].item()
                target_idx = pos_dict[key].item()
                end_to_target.append(att[i, end_idx, target_idx].item())

            end_to_target = torch.tensor(end_to_target)
            cur_ys.append(end_to_target.mean().item())
            cur_stds.append(end_to_target.std().item())
            average_attention[head][key] = end_to_target.mean().item()

        fig.add_trace(
            go.Bar(
                x=list(position_dicts[0].keys()),
                y=cur_ys,
                error_y=dict(type="data", array=cur_stds),
                name=str(head),
            )
        )

    fig.update_layout(title_text=title_text)
    fig.show()