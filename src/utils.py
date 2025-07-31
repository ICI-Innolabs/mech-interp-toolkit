import torch
from typing import List
import numpy as np


COLOR_MAP = {
    "IO": "green",
    "S1": "purple",
    "S2": "#4B0082",  # dark purple
    "IO+1": "lightgreen",
    "S1+1": "orchid",
    "S2+1": "#9370DB",
}

def build_position_dicts_from_token_ids(toks: torch.Tensor) -> List[dict]:
    """
    Builds a list of position dictionaries using known token positions in a fixed prompt template.

    Each dictionary contains the positional indices (not token IDs) of relevant semantic and syntactic entities relevant to the task.
    """
    position_dicts = []

    for i in range(toks.shape[0]):
        toks_list = toks[i].tolist()  # prompt i as list of token IDs

        # Extract token IDs from known template positions
        S1_token_id        = toks[i, 2]
        S1_plus_1_token_id = toks[i, 3]
        S1_minus_1_token_id = toks[i, 1]
        S2_token_id        = toks[i, 10]
        IO_token_id        = toks[i, 4]
        IO_plus_1_token_id = toks[i, 5]
        IO_minus_1_token_id = toks[i, 3]

        # Convert token IDs to positions in the sequence
        pos_dict = {
            "S1":     torch.tensor(toks_list.index(S1_token_id.item())),
            "S1+1":   torch.tensor(toks_list.index(S1_plus_1_token_id.item())),
            "S1-1":   torch.tensor(toks_list.index(S1_minus_1_token_id.item())),
            "S2":     torch.tensor(10),
            "IO":     torch.tensor(toks_list.index(IO_token_id.item())),
            "IO+1":   torch.tensor(toks_list.index(IO_plus_1_token_id.item())),
            "IO-1":   torch.tensor(toks_list.index(IO_minus_1_token_id.item())),
            "punct":  torch.tensor(9),  # fixed in prompt template
            "starts": torch.tensor(1),    # first word after BOS
            "end":    torch.tensor(toks.shape[1] - 1)  # last token
        }
        position_dicts.append(pos_dict)

    return position_dicts


def parse_activation_identifier(identifier: str) -> str:
    """
    Parses a simplified activation identifier and maps it to the corresponding module name.

    Supported formats:
    - Residual stream: L9.Resid_Pre, L9.Resid_Mid, L9.Resid_Post
    - MLP: L9.MLP_In, L9.MLP_Out
    - Attention output: L9.Attn.Output
    - Attention heads: L9.H9.Q, L9.H9.K, L9.H9.V, L9.H9.Z

    Args:
        identifier (str): Simplified activation identifier (e.g., 'L9.Resid_Pre').

    Returns:
        str: Corresponding module name in the model.
    """
    parts = identifier.split('.')
    if len(parts) < 2:
        raise ValueError(f"Invalid activation identifier format: {identifier}")

    layer_str, component = parts[0], '.'.join(parts[1:])

    # Validate and convert layer index
    if not (layer_str.startswith('L') and layer_str[1:].isdigit()):
        raise ValueError("Invalid layer specification. Expected format 'L{layer_number}'.")
    layer = int(layer_str[1:])

    # Map components to model module names
    if component.startswith("Resid_"):
        # Residual stream components
        resid_map = {
            "Resid_Pre": "hook_resid_pre",
            "Resid_Mid": "hook_resid_mid",
            "Resid_Post": "hook_resid_post",
        }
        if component not in resid_map:
            raise ValueError(f"Invalid residual stream component: {component}")
        return f"blocks.{layer}.{resid_map[component]}"

    elif component.startswith("MLP_"):
        # MLP components
        mlp_map = {
            "MLP_In": "hook_mlp_in",
            "MLP_Out": "mlp.hook_post",
            "MLP_Resid": "hook_mlp_out",
        }
        if component not in mlp_map:
            raise ValueError(f"Invalid MLP component: {component}")
        return f"blocks.{layer}.{mlp_map[component]}"

    elif component.startswith("H"):
        # Attention heads components
        head_parts = component.split('.')
        if len(head_parts) != 2:
            raise ValueError(f"Invalid attention head specification: {component}")

        head_str, subcomponent = head_parts
        if not (head_str.startswith('H') and head_str[1:].isdigit()):
            raise ValueError("Invalid attention head specification. Expected format 'H{head_number}'.")
        head = int(head_str[1:])

        attn_map = {
            "Q": "hook_q",
            "K": "hook_k",
            "V": "hook_v",
            "Z": "hook_z",
        }
        if subcomponent not in attn_map:
            raise ValueError(f"Invalid attention head subcomponent: {subcomponent}")
        return f"blocks.{layer}.attn.{attn_map[subcomponent]}"
    elif component == "Attn_Output":
        return f"blocks.{layer}.hook_attn_out"
    else:
        raise ValueError(f"Unknown component: {component}")

def format_percentage_position(value, tmp_desc, total_length):
        tmp_desc_np = tmp_desc.cpu().detach().numpy()
        position = len(np.where(tmp_desc_np >= value.cpu().detach().numpy())[0])
        percentage_position = position / total_length
        formatted_percentage = f'{percentage_position:.4%}'
        return formatted_percentage, percentage_position
