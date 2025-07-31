from typing import List, Tuple, Dict
from jaxtyping import Float, Int
from functools import partial
from typing import Literal
import einops
import plotly.express as px

import torch
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

from data.ioi.ioi_dataset import NAMES


# ------- Attention probs & name embedding directions -------
def scatter_embedding_vs_attn(
    attn_from_end_to_io: torch.FloatTensor,
    attn_from_end_to_s: torch.FloatTensor,
    projection_in_io_dir: torch.FloatTensor,
    projection_in_s_dir: torch.FloatTensor,
    layer: int,
    head: int
):
    x = torch.concat([attn_from_end_to_io, attn_from_end_to_s], dim=0).cpu()
    y = torch.concat([projection_in_io_dir, projection_in_s_dir], dim=0).cpu()
    # color=["IO"] * N + ["S"] * N,
    color = ["IO"] * len(attn_from_end_to_io) + ["S"] * len(attn_from_end_to_s)

    fig = px.scatter(
        x=x,
        y=y,
        color=color,
        title=f"Projection of the output of {layer}.{head} along the name<br>embedding vs attention probability on name",
        labels={"x": "Attn prob on name", "y": "Dot w Name Embed", "color": "Name type"},
        color_discrete_sequence=["#72FF64", "#C9A5F7"],
        width=650
    )

    fig.show()

def calculate_and_show_scatter_embedding_vs_attn(
    model: HookedTransformer,
    layer: int,
    head: int,
    cache: ActivationCache,
    tokens: torch.Tensor,
    position_dicts: List[Dict[str, torch.Tensor]],
):
    """
    Computes and displays a scatter plot of the relationship between:
    - attention from END → IO/S1
    - projection of the output at END along IO/S1 unembedding directions

    Args:
        model (HookedTransformer): The TransformerLens model.
        layer (int): Target layer.
        head (int): Target attention head.
        cache (ActivationCache): Precomputed model activations.
        tokens (torch.Tensor): Token IDs of all prompts.
        position_dicts (List[Dict[str, Tensor]]): Positional index dictionaries for each prompt.
    """
    N = tokens.shape[0]  # Number of prompts
    z = cache[utils.get_act_name("z", layer)][:, :, head]  # shape: (batch, seq, d_head)
    output = z @ model.W_O[layer, head]  # (batch, seq, d_model)

    # Extract output vector at END token
    output_on_end_token = torch.stack([
        output[i, position_dicts[i]["end"].item()] for i in range(N)
    ])  # shape: (batch, d_model)

    # Extract the unembedding vectors for IO and S1 tokens
    io_unembedding = torch.stack([
        model.W_U.T[tokens[i, position_dicts[i]["IO"].item()]] for i in range(N)
    ])  # (batch, d_model)
    s_unembedding = torch.stack([
        model.W_U.T[tokens[i, position_dicts[i]["S1"].item()]] for i in range(N)
    ])  # (batch, d_model)

    # Project residual outputs onto those directions
    projection_in_io_dir = (output_on_end_token * io_unembedding).sum(-1)  # (batch,)
    projection_in_s_dir = (output_on_end_token * s_unembedding).sum(-1)    # (batch,)

    # Attention scores from END → IO/S1
    attn_probs = cache["pattern", layer][:, head]  # (batch, seq, seq)
    attn_from_end_to_io = torch.stack([
        attn_probs[i, position_dicts[i]["end"].item(), position_dicts[i]["IO"].item()]
        for i in range(N)
    ])  # (batch,)
    attn_from_end_to_s = torch.stack([
        attn_probs[i, position_dicts[i]["end"].item(), position_dicts[i]["S1"].item()]
        for i in range(N)
    ])  # (batch,)

    # Plot
    scatter_embedding_vs_attn(
        attn_from_end_to_io,
        attn_from_end_to_s,
        projection_in_io_dir,
        projection_in_s_dir,
        layer,
        head
    )

# ------- Copying circuits -------
def check_copy_circuit(model, layer, head, prompts, position_dicts, verbose=False, neg=False):
    """
    Tests whether a head copies the correct token into the output stream by
    projecting its OV circuit into the vocab space and checking top-k accuracy.

    Args:
        model: HookedTransformer
        layer: Layer index of the head
        head: Head index in the layer
        prompts: List of strings
        position_dicts: List[dict] containing positions of S1, IO, S2 for each prompt
        verbose: Whether to print failed predictions
        neg: Whether to flip sign of the output (to test negative heads)

    Returns:
        float: Top-k accuracy as a percentage
    """
    cache = {}

    def hook_fn(activation: torch.Tensor, hook: HookPoint, name: str = "activation"):
        cache[name] = activation
        return activation

    fwd_hooks: List[Tuple[str, callable]] = [
        ("blocks.0.hook_resid_post", partial(hook_fn, name="l0_hook_resid_post"))
    ]

    model.run_with_hooks(model.to_tokens(prompts), fwd_hooks=fwd_hooks)

    z_0 = cache["l0_hook_resid_post"]
    sign = -1 if neg else 1

    # Compute the OV output
    v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
    v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)
    o = sign * torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
    logits = model.unembed(model.ln_final(o))

    k = 5
    n_right = 0

    for seq_idx, prompt in enumerate(prompts):
        for word in ["IO", "S1", "S2"]:
            pos = position_dicts[seq_idx][word].item()
            pred_tokens = [
                model.tokenizer.decode(token)
                for token in torch.topk(logits[seq_idx, pos], k).indices
            ]
            target_tok_str = model.to_str_tokens(prompt)[position_dicts[seq_idx][word]]
            if target_tok_str in pred_tokens:
                n_right += 1
            elif verbose:
                print("-------")
                print("Seq: " + prompt["full_string"])
                print("Target: " + prompt[target_tok_str])
                print(
                    " ".join(
                        [
                            f"({i+1}):{model.tokenizer.decode(token)}"
                            for i, token in enumerate(torch.topk(logits[seq_idx, pos], k).indices)
                        ]
                    )
                )

    percent_right = (n_right / (len(prompts) * 3)) * 100
    print(f"Copy circuit for head {layer}.{head} (sign={sign}) : Top {k} accuracy: {percent_right:.2f}%")
    return percent_right


def get_copying_scores(
    model: HookedTransformer,
    k: int = 5,
    names: list = NAMES
) -> Float[Tensor, "2 layer-1 head"]:
    '''
    Gets copying scores (both positive and negative) as described in page 6 of the IOI paper, for every (layer, head) pair in the model.
    Returns these in a 3D tensor (the first dimension is for positive vs negative).
    '''
    results = torch.zeros((2, model.cfg.n_layers, model.cfg.n_heads), device="cuda")

    # Define components from our model (for typechecking, and cleaner code)
    embed: Embed = model.embed
    mlp0: MLP = model.blocks[0].mlp
    ln0: LayerNorm = model.blocks[0].ln2
    unembed: Unembed = model.unembed
    ln_final: LayerNorm = model.ln_final

    # Get embeddings for the names in our list
    name_tokens: Int[Tensor, "batch 1"] = model.to_tokens(names, prepend_bos=False)
    name_embeddings: Int[Tensor, "batch 1 d_model"] = embed(name_tokens)

    # Get residual stream after applying MLP
    resid_after_mlp1 = name_embeddings + mlp0(ln0(name_embeddings))

    # Loop over all (layer, head) pairs
    for layer in range(1, model.cfg.n_layers):
        for head in range(model.cfg.n_heads):

            # Get W_OV matrix
            W_OV = model.W_V[layer, head] @ model.W_O[layer, head]

            # Get residual stream after applying W_OV or -W_OV respectively
            # (note, because of bias b_U, it matters that we do sign flip here, not later)
            resid_after_OV_pos = resid_after_mlp1 @ W_OV
            resid_after_OV_neg = resid_after_mlp1 @ -W_OV

            # Get logits from value of residual stream
            logits_pos: Float[Tensor, "batch d_vocab"] = unembed(ln_final(resid_after_OV_pos)).squeeze()
            logits_neg: Float[Tensor, "batch d_vocab"] = unembed(ln_final(resid_after_OV_neg)).squeeze()

            # Check how many are in top k
            topk_logits: Int[Tensor, "batch k"] = torch.topk(logits_pos, dim=-1, k=k).indices
            in_topk = (topk_logits == name_tokens).any(-1)
            # Check how many are in bottom k
            bottomk_logits: Int[Tensor, "batch k"] = torch.topk(logits_neg, dim=-1, k=k).indices
            in_bottomk = (bottomk_logits == name_tokens).any(-1)

            # Fill in results
            results[:, layer-1, head] = torch.tensor([in_topk.float().mean(), in_bottomk.float().mean()])

    return results


# ------- IOI early heads hooks -------
def prev_token_hook(out_arr):
    def hook_fn(pattern, hook):
        layer = hook.layer()
        diagonal = pattern.diagonal(offset=1, dim1=-1, dim2=-2)
        out_arr[layer] = einops.reduce(diagonal, "batch head_index diagonal -> head_index", "mean")
    return hook_fn


def duplicate_token_hook(out_arr, seq_len: int):
    def hook_fn(pattern, hook):
        layer = hook.layer()
        diagonal = pattern.diagonal(offset=seq_len, dim1=-1, dim2=-2)
        out_arr[layer] = einops.reduce(diagonal, "batch head_index diagonal -> head_index", "mean")
    return hook_fn


def induction_hook(out_arr, seq_len: int):
    def hook_fn(pattern, hook):
        layer = hook.layer()
        diagonal = pattern.diagonal(offset=seq_len - 1, dim1=-1, dim2=-2)
        out_arr[layer] = einops.reduce(diagonal, "batch head_index diagonal -> head_index", "mean")
    return hook_fn


# -------  Validation of early heads -------
def generate_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
    batch: int = 1
) -> Float[Tensor, "batch 2*seq_len"]:
    '''
    Generates a sequence of repeated random tokens (no start token).
    '''
    rep_tokens_half = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    rep_tokens = torch.cat([rep_tokens_half, rep_tokens_half], dim=-1).to(model.cfg.device)
    return rep_tokens


def get_attn_scores(
    model: HookedTransformer,
    seq_len: int,
    batch: int,
    head_type: Literal["duplicate", "prev", "induction"]
):
    '''
    Returns attention scores for sequence of duplicated tokens, for every head.
    '''
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)

    _, cache = model.run_with_cache(
        rep_tokens,
        return_type=None,
        names_filter=lambda name: name.endswith("pattern")
    )

    # Get the right indices for the attention scores
    if head_type == "duplicate":
        src_indices = range(seq_len)
        dest_indices = range(seq_len, 2 * seq_len)
    elif head_type == "prev":
        src_indices = range(seq_len)
        dest_indices = range(1, seq_len + 1)
    elif head_type == "induction":
        dest_indices = range(seq_len, 2 * seq_len)
        src_indices = range(1, seq_len + 1)
    else:
        raise ValueError(f"Unknown head type {head_type}")

    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=torch.float32)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn_scores: Float[Tensor, "batch head dest src"] = cache["pattern", layer]
            avg_attn_on_duplicates = attn_scores[:, head, dest_indices, src_indices].mean().item()
            results[layer, head] = avg_attn_on_duplicates

    return results