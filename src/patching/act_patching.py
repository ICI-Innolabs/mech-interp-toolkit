import torch
import torch.nn.functional as F

from functools import partial
from transformer_lens import HookedTransformer

@torch.no_grad()
def run_model(model: HookedTransformer, prompts, fwd_hooks, prepend_bos=True, split_qkv=True):

    model.set_use_attn_result(True)
    if split_qkv:
        model.set_use_split_qkv_input(True)
    else:
        model.set_use_split_qkv_input(False)

    model.reset_hooks()
    #print("PRO:", prompts)
    logits = model.run_with_hooks(
        prompts,
        fwd_hooks=fwd_hooks,
        prepend_bos=prepend_bos,
    )[:, -1, :] # (b, vocab_size)
    model.reset_hooks()

    probs = F.softmax(logits, dim=-1) # (b, vocab_size)

    return (logits, probs)

def get_top_k_strings(model: HookedTransformer, logits, k, probs=None, use_br=True):
    # logits: (b, vocab_size)

    logit_tops, top_idxs = torch.topk(logits, k, dim=-1) # each (b, k)

    prob_tops = None
    if probs is not None:
        prob_tops, _ = torch.topk(probs, k, dim=-1)

    top_toks = ["" for _ in range(logits.size(0))]
    
    for i in range(logits.size(0)):
        for tok_id in top_idxs[i]:
            top_toks[i] += f"{model.tokenizer.decode([tok_id])}"

            if use_br:
                top_toks[i] += "<br />"
            else:
                top_toks[i] += ","

    return top_toks, (logit_tops, prob_tops)

def hook_edit_pattern(layer, head, seq_idx):

    def edit_pattern(tensor, hook, head, seq_idx):
        # tensor: (b, n_heads, seq_len, seq_len)

        tensor[:, head, -1, :seq_idx] = 0.0
        tensor[:, head, -1, seq_idx+1:] = 0.0
        tensor[:, head, -1, seq_idx] = 1.0
        
        return tensor

    return (f"blocks.{layer}.attn.hook_pattern", partial(edit_pattern, head=head, seq_idx=seq_idx))

def hook_save_layer_input(layer, save_arr):

    def save_layer_input(tensor, hook, save_arr):
        # tensor: (b, seq_len, d_model)
        save_arr.append(tensor.clone())
        return tensor

    return (f"blocks.{layer}.hook_resid_pre", partial(save_layer_input, save_arr=save_arr))

def hook_save_head_output(model: HookedTransformer, layer, head, save_arr, include_bias):

    def save_head_output(tensor, hook, layer, head, save_arr, include_bias):
        # tensor: (b, seq_len, n_heads, d_model)
        if include_bias:
            save_arr.append((tensor[:, :, head, :] + model.blocks[layer].attn.b_O).clone())
        else:
            save_arr.append(tensor[:, :, head, :].clone())
        return tensor

    return (f"blocks.{layer}.attn.hook_result", partial(save_head_output, layer=layer, head=head, save_arr=save_arr, include_bias=include_bias))

def hook_save_attention_output(model: HookedTransformer, layer, save_arr, include_bias):

    def save_attention_output(tensor, hook, layer, save_arr, include_bias):
        # tensor: (b, seq_len,  d_model)
        if include_bias:
            save_arr.append((tensor[:, :, :] + model.blocks[layer].attn.b_O).clone())
        else:
            save_arr.append(tensor[:, :, :].clone())
        return tensor

    return (f"blocks.{layer}.hook_attn_out", partial(save_attention_output, layer=layer, save_arr=save_arr, include_bias=include_bias))

def hook_save_mlp_output(model: HookedTransformer, layer, save_arr, include_bias):

    def save_mlp_output(tensor, hook, layer, save_arr, include_bias):
        # tensor: (b, seq_len,  d_model)
        if include_bias:
            save_arr.append((tensor[:, :, :] + model.blocks[layer].attn.b_O).clone())
        else:
            save_arr.append(tensor[:, :, :].clone())
        return tensor

    return (f"blocks.{layer}.mlp.hook_post", partial(save_mlp_output, layer=layer, save_arr=save_arr, include_bias=include_bias))

def hook_save_layer_output(layer, save_arr):

    def save_layer_output(tensor, hook, save_arr):
        # tensor: (b, seq_len, d_model)
        save_arr.append(tensor.clone())
        return tensor
    
    return (f"blocks.{layer}.hook_resid_post", partial(save_layer_output, save_arr=save_arr))

def hook_save_head_pattern(layer, head, save_arr):

    def save_head_pattern(tensor, hook, head, save_arr):
        # tensor: (b, n_heads, seq_len, seq_len)
        save_arr.append(tensor[:, head, :, :].clone())
        return tensor

    return (f"blocks.{layer}.attn.hook_pattern", partial(save_head_pattern, head=head, save_arr=save_arr))

def hook_ablate_head(layer, head, means):
    # means: (b, n_layers, 1, seq_len, n_heads, d_model)
    # Replaces output of (layer, head) with means[:, layer, 0, :, head, :]

    def ablate_head(tensor, hook, layer, head, means):
        # tensor: (b, seq_len, n_heads, d_model)
        if means is None:
            tensor[:, :, head, :] = 0.0
        
        else:
            tensor[:, :, head, :] = means[:, layer, 0, :, head, :]

        return tensor

    return (f"blocks.{layer}.attn.hook_result", partial(ablate_head, layer=layer, head=head, means=means))


def hook_ablate_attn_out(layer, means): 
    # means: (b, n_layers, 1, seq_len, d_model)
    # Replaces output of (layer) with means[:, layer, 0, :]
    def ablate_attn_out(tensor, hook, layer, means):
        # tensor: (b, seq_len, d_model)
        if means is None:
            tensor = 0.0
        else:
            tensor[:, layer, :, :] = 0.

        return tensor

    return (f"blocks.{layer}.hook_attn_out", partial(ablate_attn_out, layer=layer, means=means))


def hook_ablate_attn_out_pos(layer, pos, means=None): 

    def ablate_attn_out(tensor, hook):
        # tensor: (B, seq_len, d_model)
        if means is None:
            if isinstance(pos, torch.Tensor):
                if pos.ndim == 0:
                    tensor[:, pos.item(), :] = 0.0
                else:
                    for i in range(tensor.size(0)):
                        tensor[i, pos[i].item(), :] = 0.0
            else:
                tensor[:, pos, :] = 0.0
        else:
            if isinstance(pos, torch.Tensor):
                if pos.ndim == 0:
                    tensor[:, pos.item(), :] = means
                else:
                    for i in range(tensor.size(0)):
                        tensor[i, pos[i].item(), :] = means[i]
            else:
                tensor[:, pos, :] = means
        return tensor

    return (f"blocks.{layer}.hook_attn_out", ablate_attn_out)


def hook_ablate_mlp_neuron_pos(layer, neuron, pos, means=None):

    def ablate_mlp_out(tensor, hook, neuron, pos, means):
        # tensor: (b, seq_len, d_mlp)
        if means is None:
            if isinstance(neuron, list):
                for n in neuron:
                    tensor[:, pos, n] = 0.0
            else:
                tensor[:, pos, neuron] = 0.0
        else:
            if isinstance(means, torch.Tensor):
                if isinstance(neuron, list):
                    for n in neuron:
                        tensor[:, pos, n] = means
                else:
                    tensor[:, pos, neuron] = means[neuron]

        return tensor

    return (f"blocks.{layer}.mlp.hook_post", partial(ablate_mlp_out, neuron=neuron, pos=pos, means=means))


def hook_set_mlp_neuron_pos(layer, neuron, pos, new_input):

    def set_single_neuron(tensor, hook, neuron, pos, new_input):
        # tensor: [batch, seq_len, d_mlp]
        # new_input: [batch, d_mlp] or [batch, seq_len, d_mlp]
        if isinstance(new_input, list):
            new_input = new_input[0]
        tensor[:, pos, neuron] = new_input[:, pos, neuron]
        return tensor

    return (f"blocks.{layer}.mlp.hook_post", partial(set_single_neuron, neuron=neuron, pos=pos, new_input=new_input))

def hook_set_layer_input(layer, new_input):
    # new_input: (b, seq_len, d_model)

    def set_layer_input(tensor, hook, new_input):
        # tensor: (b, seq_len, d_model)
        if isinstance(new_input, list):
            new_input = new_input[0]
        tensor.copy_(new_input)
        return tensor

    return (f"blocks.{layer}.hook_resid_pre", partial(set_layer_input, new_input=new_input))

def hook_set_layer_output(layer, new_output):
    # new_output: (b, seq_len, d_model)

    def set_layer_output(tensor, hook, new_output):
        # tensor: (b, seq_len, d_model)
        if isinstance(new_output, list):
            new_output = new_output[0]
        tensor.copy_(new_output)
        return tensor

    return (f"blocks.{layer}.hook_resid_post", partial(set_layer_output, new_output=new_output))