from typing import List, Optional, Tuple, Literal, Union
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

from src.utils import parse_activation_identifier, format_percentage_position

def vocabulary_projection(
    prompt: str,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    method: Union[None, Literal['unembedding', 'top-k']] = None,
    top_k: int = 5,
    all_tok_pos: bool = True,
    specific_activation: Optional[str] = None,
    neuron: Optional[int] = None,
) -> Tuple[List[List[Tuple[str, float]]], torch.Tensor]:
    """
    Projects a particular activation of the Transformer onto the vocabulary (i.e. multiplying by the unembedding), showing the top-k most likely tokens at each position.

    Note: If using higher or lower dimensional activations compared to the hidden size (d_model), e.g. output of the MLP (`mlp.hook_post`) of shape `[seq, d_mlp]` decoding via the unembedding
    matrix (`model.W_U`) will result in a matmul error because we attempt to multiply our activation, a tensor of shape [seq_len, d_mlp] with a tensor of shape `[d_model, d_vocab]`.
    To avoid this, we can either use the `top-k` method which applies softmax directly to the activation and returns the top-k tokens.

    Args:
        prompt (str): The input text string to tokenize and run through the model.
        model (HookedTransformer): A model from TransformerLens with hooks and activation tracking.
        tokenizer: Tokenizer used to encode and decode text (should match the model).
        method (str): How to perform the projection. Options:
                      - "unembedding": Apply final layer norm and project using W_U.
                      - "top-k": Just apply softmax directly to the hidden state.
        top_k (int): Number of top predicted tokens to return at each position.
        all_tok_pos (bool): If True, return top-k tokens for every token in the prompt.
                            If False, return only for the final token.
        specific_activation (Optional[str]): Identifier for which module to hook into (e.g., "L9.MLP_Out").
        neuron (Optional[int]): If set, isolate the influence of a single neuron at that position.

    Returns:
        Tuple[List[List[Tuple[str, float]]], torch.Tensor]
            - Top-k tokens and probabilities at each position
            - Raw activation tensor captured (shape: [1, seq_len, hidden_dim])
    """
    model.reset_hooks()
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(model.cfg.device)

    # Activation cache
    act_cache = {}

    # Define hook to save activations
    def hook_fn(activation: torch.Tensor, hook: HookPoint):
        act_cache["activation"] = activation
        return activation

    # Resolve the module name for the hook
    module_name = parse_activation_identifier(specific_activation)
    hooks = [(module_name, hook_fn)]

    # Run model with hook
    _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

    # Check for captured activation
    if "activation" not in act_cache:
        raise ValueError(f"Activation for {specific_activation} not found. Check your identifier.")
    
    activation = act_cache["activation"]

    # Handle neuron projection
    if neuron is not None:
        if len(activation.shape) != 3:
            raise ValueError(f"Expected activation shape [1, seq_len, hidden_dim], got {activation.shape}")
        if not (0 <= neuron < activation.shape[-1]):
            raise ValueError(f"Neuron index {neuron} out of bounds for hidden size {activation.shape[-1]}.")

        new_activation = torch.zeros_like(activation)
        new_activation[:, :, neuron] = activation[:, :, neuron]
        activation = new_activation

    # Project to vocabulary space
    if method == "unembedding":
        activation_ln = model.ln_final(activation)
        logits = activation_ln @ model.W_U  # shape: [1, seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)

    elif method == "top-k":
        probs = F.softmax(activation, dim=-1)
        # if probs.shape[-1] != model.cfg.d_vocab:
        #     assert probs.shape[-1] == model.cfg.d_vocab, \
        #         f"Expected activation dimension {model.cfg.d_vocab}, got {probs.shape}"
        #     raise ValueError(f"Activation has incompatible dimension for vocab projection: {probs.shape[-1]}")
    else:
        raise ValueError(f"Invalid method: {method}. Use 'unembedding' or 'top-k'.")

    # Extract top-k predictions
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)

    results = []

    if all_tok_pos:
        for pos in range(tokens.shape[1]):
            top_tokens = [tokenizer.decode(tok_id.item()) for tok_id in top_k_indices[0, pos]]
            top_probs = top_k_probs[0, pos].tolist()
            results.append(list(zip(top_tokens, top_probs)))
    else:
        pos = -1
        top_tokens = [tokenizer.decode(tok_id.item()) for tok_id in top_k_indices[0, pos]]
        top_probs = top_k_probs[0, pos].tolist()
        results.append(list(zip(top_tokens, top_probs)))

    return results, activation


def inspect_ov_matrix(
        model: HookedTransformer, 
        tokenizer: AutoTokenizer, 
        words: List[str], 
        attn_head: Tuple[int, int], 
        pairwise: bool = False
) -> None: 
    """
    Inspects the OV matrix of a model for a specified Attention Head given sequence of tokens.
    Code taken primarily from: https://github.com/guy-dar/embedding-space

    Analyzes a single OV (W_V @ W_O) matrix from a specific attention head to measure
    how much an input token (via embedding) writes a particular output token (via unembedding)
    into the residual stream.

    The analysis computes:
        tmp[i, j] = ⟨ W_O W_V e_i , u_j ⟩
                  = how much token i writes token j via this attention head's OV circuit

    The OV matrix is a low-rank refactored matrix from einsum of W_V and W_O matrices.
    For more details check HookedTransformer.refactored_attn_matrices() method.

    Args:
        model (HookedTransformer): The model to inspect.
        tokenizer (AutoTokenizer): The tokenizer used to encode the tokens.
        words (torch.Tensor): The list of tokens to inspect.
        attn_head (Tuple[int, int]): The Attention Head to inspect.
        pairwise (bool): Whether to inspect pairwise token combinations. Default is False.
    
    Returns:
        Prints the token as entries in the OV matrix in descending order. Equivalent to a filtered top-k.
    """ 
    layer_idx = attn_head[0]
    head_idx = attn_head[1]

    # Extract W_V and W_O for the specific head
    W_V_head = model.W_V[layer_idx, head_idx, :]  
    W_O_head = model.W_O[layer_idx, head_idx] 

    # Get embedding and the unembedding matrices
    emb = model.embed
    emb_matrix = emb.W_E  # Shape: (d_vocab, d_model)
    emb_inv = emb_matrix.T  # Shape: (d_model, d_vocab)

    # Multiply embeddings with the OV matrix
    tmp = (emb_matrix @ (W_V_head @ W_O_head) @ emb_inv) # Shape: (d_vocab, d_vocab)
    tmp_desc = tmp.flatten() # Shape: (d_vocab * d_vocab)
    l = len(tmp_desc)

    if pairwise: 
        for word_1, word_2 in words:
            # Encode tokens
            token_1_encoded = tokenizer.encode(word_1)
            token_2_encoded = tokenizer.encode(word_2)

            # Get the matrix value for the token pair
            value = tmp[token_1_encoded, token_2_encoded]

            for v in value: 
                formatted_percentage, _ = format_percentage_position(v, tmp_desc, l)
                print(f"'{word_1}, {word_2}': {formatted_percentage}")
            # Free memory
        del tmp
        torch.cuda.empty_cache()
    else:
        for word in words:
            token_encoded = tokenizer.encode(word)
            
            value = tmp[token_encoded]
            
            for v in value: 
                formatted_percentage, _ = format_percentage_position(v, tmp_desc, l)
                print(f"'{word}': {formatted_percentage}")
        del tmp
        torch.cuda.empty_cache()
