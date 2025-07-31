import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import List
from jaxtyping import Bool
from transformer_lens import HookedTransformer, ActivationCache
from src.utils import COLOR_MAP


def plot_neuron_activation(
    model: HookedTransformer, clean_cache: ActivationCache, tokenizer, clean_tokens: torch.Tensor, 
    layer: int, neuron_idx: int, prompt_idx: int, target_roles: List[str], position_dict: dict, save: Bool = False
):
    """
    Extracts the activations of a specific neuron in the clean distribution 
    and plots its activation across all tokens in the prompt, supporting highlighting specific target tokens either by passing in a list of token IDs from 
    word_idx to token position dictionary or their string representations.

    Args:
        model: TransformerLens model.
        clean_cache (dict): Activation cache from the clean prompt.
        tokenizer: Tokenizer corresponding to the model.
        clean_tokens (torch.Tensor): Tokenized clean prompts.
        layer (int): MLP layer index to analyze.
        neuron_idx (int): Index of the MLP neuron to examine.
        prompt_idx (int): Index of the prompt to analyze.
        target_roles (List[str]): Tokens to highlight.
        position_dict (dict): Optional dictionary of label -> position for the prompt.
        save (bool): Whether to save the plot as an image.
    """
    model.reset_hooks()
    
    # Define the module name for the MLP output at the specified layer.
    module_name = f"blocks.{layer}.mlp.hook_post"
    
    if module_name not in clean_cache:
        raise ValueError(f"Activation {module_name} not found in the model (clean cache).")
    
    activation = clean_cache[module_name]  # Shape: (batch, seq_length, hidden_dim)

    # Extract activations for the specific neuron
    neuron_activation = activation[prompt_idx, :, neuron_idx]  # Shape: (seq_length,)

    # Decode tokens for x-axis labels
    token_labels = [tokenizer.decode([tok]) for tok in clean_tokens[prompt_idx]]
    target_indices = []

    for role in target_roles:
        if role not in position_dict:
            print(f"Role '{role}' not found in position_dict.")
            continue

        pos_tensor = position_dict[role]
        if isinstance(pos_tensor, torch.Tensor):
            pos = pos_tensor.item()
        elif isinstance(pos_tensor, int):
            pos = pos_tensor
        else:
            continue

        if not (0 <= pos < len(clean_tokens[prompt_idx])):
            print(f"Position {pos} for role '{role}' is out of range.")
            continue

        token = tokenizer.decode([clean_tokens[prompt_idx, pos]])
        activation_val = neuron_activation[pos].item()
        color = COLOR_MAP.get(role, "red")
        label = f'"{token}": {role}'

        print(f"Token '{token}' at position {pos} labeled as {role}. Activation: {activation_val:.4f}")
        target_indices.append((pos, label, color))

    if not target_indices:
        print("No target tokens found in the prompt.")
    
    # Plot neuron activation across all tokens 
    plt.figure(figsize=(10, 4))
    plt.plot(
        range(len(token_labels)),
        neuron_activation.cpu().numpy(),
        marker="o",
        linestyle="-",
        color="blue",
        label=f"Neuron {neuron_idx}"
    )

    # Highlight found target tokens with label-specific colors
    for idx, label, color in target_indices:
        plt.scatter(
            [idx],
            [neuron_activation[idx].item()],
            color=color,
            s=100,
            zorder=3,
            label=f"{label}"
        )

    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Finalize plot
    plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha="right")
    plt.xlabel("Token Position")
    plt.ylabel("Neuron Activation")
    plt.title(f"Neuron {neuron_idx} Activation Across Tokens")
    plt.tight_layout()
    plt.show()
    if save:
        # Create the directory if it doesn't exist
        output_dir = f"figures/mlp/firing/mlp_{layer}/neuron_{neuron_idx}"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"figures/mlp/firing/mlp_{layer}/neuron_{neuron_idx}/prompt_{prompt_idx}.png", bbox_inches='tight')
    

def mlp_neurons_by_abs_diff(
    model: HookedTransformer,
    clean_cache: ActivationCache,
    corr_cache: ActivationCache,
    clean_position_dicts: List[dict],
    corr_position_dicts: List[dict],
    layer: int,
    position_keys: List[str],
    threshold: float = None,
):
    """
    Computes the mean absolute difference in MLP neuron activations between clean and corrupted runs,
    at specified positions for each prompt, for a given model layer.

    Args:
        clean_cache: Activation cache from the clean (original) run. Should support indexing by module name.
        corr_cache: Activation cache from the corrupted run. Should support indexing by module name.
        clean_position_dicts (List[dict]): List of dicts mapping position keys (e.g., "S1", "IO", "S2") to token indices for each prompt in the clean run.
        corr_position_dicts (List[dict]): List of dicts mapping position keys to token indices for each prompt in the corrupted run.
        layer (int): The layer index to analyze.
        position_keys (List[str]): List of position keys to compare (e.g., ["S1", "IO", "S2"]).
        threshold (float, optional): If provided, only neurons with at least one position difference above this threshold are kept.

    Returns:
        pd.DataFrame: DataFrame indexed by neuron, with columns for each position key, containing the mean absolute activation difference.
                      If threshold is set, only neurons with at least one position above threshold are included.
    """
    module_name = f"blocks.{layer}.mlp.hook_post"
    act_clean = clean_cache[module_name]  # [batch, seq, d_mlp]
    act_corr  = corr_cache[module_name]   # [batch, seq, d_mlp]

    d_mlp = act_clean.shape[-1]
    assert d_mlp == model.cfg.d_mlp

    data = {pos_key: [] for pos_key in position_keys}

    for pos_key in position_keys:
        diffs = []
        for i in range(len(clean_position_dicts)):
            clean_pos = clean_position_dicts[i][pos_key].item()
            corr_pos  = corr_position_dicts[i][pos_key].item()

            clean_vec = act_clean[i, clean_pos]
            corr_vec  = act_corr[i, corr_pos]
            diff = (clean_vec - corr_vec).detach().cpu()  # shape: [d_mlp]
            diffs.append(diff.unsqueeze(0))
        diffs = torch.cat(diffs, dim=0)
        mean_abs_diff = diffs.abs().mean(dim=0)  # [d_mlp]
        data[pos_key] = mean_abs_diff.numpy()

    df = pd.DataFrame(data)
    df["neuron"] = range(d_mlp)
    df.set_index("neuron", inplace=True)

    if threshold is not None:
        df = df[(df.abs() >= threshold).any(axis=1)]

    return df
