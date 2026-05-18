"""Parameter-level relevance aggregation for GPT-2 AttnLRP.

The main explanation file should stay easy to read. This module contains the
Part 3 details: enabling parameter gradients, collecting ``param * param.grad``,
and aggregating those scores by layer, component, and attention head.
"""
from __future__ import annotations

import re
from contextlib import contextmanager

import torch


_GPT2_LAYER_RE = re.compile(r"^transformer\.h\.(\d+)\.")
_COMPONENTS = ("attention", "mlp", "layernorm")


@contextmanager
def parameter_gradients_enabled(model, enabled: bool):
    """Temporarily enable parameter gradients when Part 3 attribution is needed."""
    if not enabled:
        yield
        return

    original_states = {param: param.requires_grad for param in model.parameters()}
    try:
        for param in model.parameters():
            param.requires_grad_(True)
        yield
    finally:
        for param, requires_grad in original_states.items():
            param.requires_grad_(requires_grad)
        model.zero_grad(set_to_none=True)


def _layer_index(parameter_name: str) -> int | None:
    match = _GPT2_LAYER_RE.match(parameter_name)
    return int(match.group(1)) if match else None


def _component(parameter_name: str) -> str | None:
    if ".attn." in parameter_name:
        return "attention"
    if ".mlp." in parameter_name:
        return "mlp"
    if ".ln_" in parameter_name:
        return "layernorm"
    return None


def _add_attention_head_relevance(
    model,
    parameter_name: str,
    relevance: torch.Tensor,
    attention_head_abs: torch.Tensor,
) -> None:
    """Aggregate one GPT-2 attention parameter tensor into [layer, head]."""
    layer_idx = _layer_index(parameter_name)
    if layer_idx is None or ".attn." not in parameter_name:
        return

    n_heads = int(model.config.n_head)
    head_dim = int(model.config.n_embd // n_heads)
    rel_abs = relevance.detach().abs()

    if parameter_name.endswith("attn.c_attn.weight"):
        # GPT-2 Conv1D stores [in_features, 3 * hidden]; split Q/K/V and heads.
        rel_abs = rel_abs.reshape(rel_abs.shape[0], 3, n_heads, head_dim)
        attention_head_abs[layer_idx] += rel_abs.sum(dim=(0, 1, 3)).cpu()
    elif parameter_name.endswith("attn.c_attn.bias"):
        rel_abs = rel_abs.reshape(3, n_heads, head_dim)
        attention_head_abs[layer_idx] += rel_abs.sum(dim=(0, 2)).cpu()
    elif parameter_name.endswith("attn.c_proj.weight"):
        # c_proj consumes concatenated heads; its input axis identifies heads.
        rel_abs = rel_abs.reshape(n_heads, head_dim, rel_abs.shape[1])
        attention_head_abs[layer_idx] += rel_abs.sum(dim=(1, 2)).cpu()


def collect_parameter_relevance(model, save_tensors: bool = False) -> dict:
    """Return Part 3 parameter relevance summaries after patched backward."""
    n_layers = len(model.transformer.h)
    n_heads = int(model.config.n_head)
    n_components = len(_COMPONENTS)
    component_to_idx = {name: idx for idx, name in enumerate(_COMPONENTS)}

    layer_component_signed = torch.zeros(n_layers, n_components)
    layer_component_abs = torch.zeros(n_layers, n_components)
    layer_component_numel = torch.zeros(n_layers, n_components, dtype=torch.int64)
    attention_head_abs = torch.zeros(n_layers, n_heads)
    module_records = []
    tensors = {} if save_tensors else None

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        relevance = (param.grad * param).detach().float()
        signed_sum = float(relevance.sum().cpu().item())
        abs_sum = float(relevance.abs().sum().cpu().item())
        layer_idx = _layer_index(name)
        component = _component(name)

        if layer_idx is not None and component is not None:
            comp_idx = component_to_idx[component]
            layer_component_signed[layer_idx, comp_idx] += signed_sum
            layer_component_abs[layer_idx, comp_idx] += abs_sum
            layer_component_numel[layer_idx, comp_idx] += param.numel()

        _add_attention_head_relevance(model, name, relevance, attention_head_abs)

        module_records.append(
            {
                "name": name,
                "shape": list(param.shape),
                "layer": layer_idx,
                "component": component or "global",
                "signed_sum": signed_sum,
                "abs_sum": abs_sum,
                "numel": int(param.numel()),
            }
        )
        if tensors is not None:
            tensors[name] = relevance.cpu()

    layer_component_abs_mean = torch.zeros(n_layers, n_components)
    valid_mask = layer_component_numel > 0
    layer_component_abs_mean[valid_mask] = (
        layer_component_abs[valid_mask].float()
        / layer_component_numel[valid_mask].float()
    )

    module_records.sort(key=lambda item: item["abs_sum"], reverse=True)
    summary = {
        "component_names": list(_COMPONENTS),
        "layer_component_signed": layer_component_signed,
        "layer_component_abs": layer_component_abs,
        "layer_component_abs_mean": layer_component_abs_mean,
        "attention_head_abs": attention_head_abs,
        "module_records": module_records,
    }

    result = {"parameter_summary": summary}
    if tensors is not None:
        result["parameter_relevance"] = tensors
    return result
