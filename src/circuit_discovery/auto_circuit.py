import os
from typing import Optional, List, Literal

import torch 
from transformer_lens import HookedTransformer

from auto_circuit.data import load_datasets_from_json
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores, Edge, Node, AblationType
from auto_circuit.visualize import node_name
from auto_circuit.utils.graph_utils import patchable_model, patch_mode
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.tensor_ops import (
    correct_answer_proportion, 
    correct_answer_greater_than_incorrect_proportion, 
    batch_avg_answer_diff, 
    batch_answer_diff_percents, 
    correct_answer_greater_than_incorrect_proportion
)
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.utils.graph_utils import patchable_model


def auto_circuit_experiment(
    model: HookedTransformer, device: str = "cuda", score_threshold: int = None, save_path: str = None
) -> None:
    """
    Runs a circuit discovery experiment using **Edge Patching** with default config (Resample Ablation and slicing logits on the last position) on a given `HookedTransformer` model.
    The function computes attribution scores for both "last" and "next" datasets, visualizes the resulting graphs, and saves the figures.
    Args:
        model (HookedTransformer): The transformer model to analyze. Must be an instance of HookedTransformer.
        score_threshold (int, optional): The minimum absolute value of the score for an edge to be be kept in the circuit visualzation. Defaults to None.
        device (str, optional): The device to run computations on (e.g., "cuda" or "cpu"). Defaults to "cuda".
        save_path (str, optional): The directory path where the resulting images will be saved. If None, images are saved in the project root.

    Note:
        The data used for the patching experiment should not be passed from the dataloaders used in training.

    TODO:
        - Add implementation for tokenwise circuits.

    """
    assert isinstance(
        model, HookedTransformer
    ), "Model must be an instance of HookedTransformer"
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # path to the JSON datasets
    base_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    path_last = repo_path_to_abs_path(
        os.path.join(base_dir, "data/succession_augmented_last_big.json")
    )
    path_next = repo_path_to_abs_path(
        os.path.join(base_dir, "data/succession_augmented_next_big.json")
    )

    train_loader_last, _ = load_datasets_from_json(
        model=model,
        path=path_last,
        device=device,
        prepend_bos=False,
        tail_divergence=False,
        batch_size=32,
        train_test_size=(256, 256),
    )
    train_loader_next, _ = load_datasets_from_json(
        model=model,
        path=path_next,
        device=device,
        prepend_bos=False,
        tail_divergence=False,
        batch_size=32,
        train_test_size=(256, 256),
    )

    auto_model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=True,
        device=device,
    )

    attribution_scores_last: PruneScores = mask_gradient_prune_scores(
        model=auto_model,
        dataloader=train_loader_last,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )

    attribution_scores_next: PruneScores = mask_gradient_prune_scores(
        model=auto_model,
        dataloader=train_loader_next,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )

    fig_last = draw_seq_graph(
        auto_model,
        attribution_scores_last,
        score_threshold=score_threshold,
        layer_spacing=True,
        orientation="v",
        display_ipython=False,
    )
    fig_next = draw_seq_graph(
        auto_model,
        attribution_scores_next,
        score_threshold=score_threshold,
        layer_spacing=True,
        orientation="v",
        display_ipython=False,
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    last_img_path = f"{save_path}/last.png"
    next_img_path = f"{save_path}/next.png"

    fig_last.write_image(last_img_path, scale=1)
    fig_next.write_image(next_img_path, scale=1)

    return {
        "attribution_scores_last": attribution_scores_last,
        "attribution_scores_next": attribution_scores_next,
    }


def ablation_experiment(
    model: HookedTransformer,
    device: str,
    phase: Literal['A', 'B'],
    edges_sorted: List[Edge],
    ablation_type: Literal['Zero, Resample'] = None,
    how_many: Optional[int] = None,
):
    """
    Runs an ablation experiment on a given HookedTransformer model by patching specified edges and evaluating the effect on model performance, compared with the un-patched model.

    Args:
        model (HookedTransformer): The transformer model to be ablated.
        device (str): The device to run computations on (e.g., 'cpu' or 'cuda').
        phase (Literal['A', 'B']): Phase of the experiment, either 'A' or 'B'.
        edges_sorted (List[Edge]): List of edges sorted by attribution score (descending).
        ablation_type (Literal['Zero', 'Resample'], optional): Type of ablation to perform. 
            'Zero' sets activations to zero, 'Resample' replaces activations with resampled values.
        how_many (Optional[int], optional): Number of top edges (by attribution) to keep unpatched; all others are patched. If None, all edges are patched.
    Returns:
        dict: Dictionary containing ablation metrics:
            - "correct_answer_proportion": Proportion of correct answers after ablation.
            - "correct_answer_greater_than_incorrect_proportion": Proportion where correct answer logits exceed incorrect ones.
            - "batch_avg_answer_diff": Average difference between correct and incorrect answer logits (ablated).
            - "batch_avg_answer_diff_clean": Average difference between correct and incorrect answer logits (clean/original model).
    
    Raises:
        AssertionError: 
            - If the model is not an instance of HookedTransformer or if the top edge is not found in the model's edge dictionary.
            - If `ablation_type` is not one of the specified types.
            - `how_many` must be a non-negative integer > 1 else it errors.
    """
    assert isinstance(
        model, HookedTransformer
    ), "Model must be an instance of HookedTransformer"
 
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # path to the JSON datasets
    base_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    path_last = repo_path_to_abs_path(
        os.path.join(base_dir, "data/succession_augmented_last_big.json")
    )
    path_next = repo_path_to_abs_path(
        os.path.join(base_dir, "data/succession_augmented_next_big.json")
    )
    if phase == 'A':
        train_loader, test_loader = load_datasets_from_json(
            model=model,
            path=path_last,
            device=device,
            prepend_bos=False,
            tail_divergence=False,
            batch_size=32,
            train_test_size=(256, 256),
        )
    elif phase == 'B':
        train_loader, test_loader = load_datasets_from_json(
            model=model,
            path=path_next,
            device=device,
            prepend_bos=False,
            tail_divergence=False,
            batch_size=32,
            train_test_size=(256, 256),
        )
    
    auto_model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=True,
        kv_caches=(train_loader.kv_cache, test_loader.kv_cache), # this is needed else, patching will not work
        device=device,
    )

    attribution_scores: PruneScores = mask_gradient_prune_scores(
        model=auto_model,
        dataloader=train_loader,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )

    for batch in test_loader:
        toks = batch.clean
        answers = batch.answers
        wrong_answers = batch.wrong_answers
        answers = [(answers[i], wrong_answers[i]) for i in range(len(answers))]
        answers = torch.tensor(answers, dtype=torch.long)
        answers = answers.to(device)

    if ablation_type == 'Zero':
        ablations = src_ablations(auto_model, toks, AblationType.ZERO)

    elif ablation_type == 'Resample':
        ablations = src_ablations(auto_model, toks, AblationType.RESAMPLE)
        
    # quick sanity check if an edge is present
    all_edge_strs = [str(edge) for edge in edges_sorted] # these are desceonding order by attribution score
    edge_str_to_obj = {str(edge): edge for edge in edges_sorted}

    found = any(
        all_edge_strs[0] in subdict
            for subdict in auto_model.edge_name_dict.values()
        )
    assert found, f"Edge '{all_edge_strs[0]}' not found in edge_name_dict"

    # patch all edges except the ones with the highest attribution scores, specified by `how_many`
    if how_many is not None:
        remaining_edge_strs = all_edge_strs[:how_many]
        patch_edges = [e for e in all_edge_strs if e not in remaining_edge_strs]

        remaining_edges_with_scores = []
        patched_edges_with_scores = []

        for edge_str in remaining_edge_strs:
            edge = edge_str_to_obj[edge_str]
            score = attribution_scores[edge.dest.module_name][edge.patch_idx].item()
            remaining_edges_with_scores.append((edge_str, score))

        for edge_str in patch_edges:
            edge = edge_str_to_obj[edge_str]
            score = attribution_scores[edge.dest.module_name][edge.patch_idx].item()
            patched_edges_with_scores.append((edge_str, score))
    
    with patch_mode(auto_model, ablations, patch_edges):
        patched_out = auto_model(toks)

    for batch in test_loader:
        clean_out = model(batch.clean)
    
    ablation_metrics = {
        "remaining_edges_with_scores": remaining_edges_with_scores if how_many is not None else None,
        "patched_edges_with_scores": patched_edges_with_scores if how_many is not None else None,
        "correct_answer_proportion": correct_answer_proportion(
            patched_out[:, -1, :], batch),
        "correct_answer_greater_than_incorrect_proportion": correct_answer_greater_than_incorrect_proportion(
            patched_out[:, -1, :], batch),
        "batch_avg_answer_diff": batch_avg_answer_diff(
            patched_out[:, -1, :], batch),
        "batch_avg_answer_diff_clean": batch_avg_answer_diff(
            clean_out[:, -1, :], batch),
    }   
    return ablation_metrics


def count_edges(scores):
    pruned_count = 0
    total_count = 0
    for key in scores:
        pruned_count += torch.sum(torch.isfinite(scores[key])).item()
        total_count += scores[key].numel()
        
    return pruned_count, total_count


def acdc_discovery(patched_model, dataloader, tau_exps=[-2], tau_bases=[1]):
    """
    Returns number of remaining edges after pruning and the scores of the edges
    """
    
    # Get the maximum scores at which each edge is pruned
    pruned_scores = acdc_prune_scores(patched_model, dataloader, official_edges=None, show_graphs=False, tao_exps=tau_exps, tao_bases=tau_bases)
    
    pruned_count, total_count = count_edges(pruned_scores)
    
    # Replace values with inf with 1
    # Store in plot_scores
    plot_scores = {}
    for key in pruned_scores:
        plot_scores[key] = torch.where(torch.isfinite(pruned_scores[key]), pruned_scores[key], torch.ones_like(pruned_scores[key]))
        
    return total_count, total_count - pruned_count, plot_scores

def prepare_model_acdc(model, device):
    hooked_model = model

    # Requirements mentioned in load_tl_model
    hooked_model.cfg.use_attn_result = True
    hooked_model.cfg.use_attn_in = True
    hooked_model.cfg.use_split_qkv_input = True
    hooked_model.cfg.use_hook_mlp_in = True
    hooked_model.eval()
    for param in hooked_model.parameters():
        param.requires_grad = False
        
    patched_model = patchable_model(hooked_model, factorized=True, slice_output="last_seq", separate_qkv=True, device=device)
        
    return patched_model