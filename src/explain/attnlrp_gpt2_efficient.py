"""Efficient AttnLRP explanation for GPT-2 using Input*Gradient.

The efficient LXT monkey patch rewires ``backward`` so gradients carry
AttnLRP relevance ratios instead of ordinary loss gradients. Token relevance
therefore comes from ``activation * activation.grad``; parameter relevance uses
the same allocation shortcut, ``parameter * parameter.grad``.
"""
from __future__ import annotations

import re

import torch


_LAYER_RE = re.compile(r"^transformer\.h\.(\d+)\.")
_COMPONENTS = ("attention", "mlp", "layernorm")


def _register_block_forward_hooks(model):
    """Capture GPT-2 block activations for activation * gradient relevance."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.requires_grad:
                hidden.retain_grad()
                captured[idx] = hidden
        return hook

    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_hook(i)))
    return captured, handles


def _set_parameter_grad_state(model, requires_grad: bool) -> dict[torch.nn.Parameter, bool]:
    states = {}
    for param in model.parameters():
        states[param] = param.requires_grad
        param.requires_grad_(requires_grad)
    return states


def _restore_parameter_grad_state(states: dict[torch.nn.Parameter, bool]) -> None:
    for param, requires_grad in states.items():
        param.requires_grad_(requires_grad)


def _component_for_parameter(name: str) -> str | None:
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if ".ln_" in name:
        return "layernorm"
    return None


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.match(name)
    return int(match.group(1)) if match else None


def _add_gpt2_head_relevance(
    model,
    name: str,
    relevance: torch.Tensor,
    head_relevance: torch.Tensor,
) -> None:
    """Aggregate GPT-2 attention parameter relevance to [layer, head]."""
    layer_idx = _layer_index(name)
    if layer_idx is None or ".attn." not in name:
        return

    n_heads = int(model.config.n_head)
    head_dim = int(model.config.n_embd // n_heads)
    rel_abs = relevance.detach().abs()

    if name.endswith("attn.c_attn.weight"):
        # GPT-2 Conv1D stores [in_features, 3 * hidden]; split q/k/v and heads.
        rel_abs = rel_abs.reshape(rel_abs.shape[0], 3, n_heads, head_dim)
        head_relevance[layer_idx] += rel_abs.sum(dim=(0, 1, 3)).cpu()
    elif name.endswith("attn.c_attn.bias"):
        rel_abs = rel_abs.reshape(3, n_heads, head_dim)
        head_relevance[layer_idx] += rel_abs.sum(dim=(0, 2)).cpu()
    elif name.endswith("attn.c_proj.weight"):
        # c_proj consumes concatenated heads, so the input axis identifies heads.
        rel_abs = rel_abs.reshape(n_heads, head_dim, rel_abs.shape[1])
        head_relevance[layer_idx] += rel_abs.sum(dim=(1, 2)).cpu()


def _collect_parameter_relevance(model, save_tensors: bool = False) -> dict:
    """Collect AttnLRP parameter relevance and compact layer/head summaries."""
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
        component = _component_for_parameter(name)

        if layer_idx is not None and component is not None:
            comp_idx = component_to_idx[component]
            layer_component_signed[layer_idx, comp_idx] += signed_sum
            layer_component_abs[layer_idx, comp_idx] += abs_sum
            layer_component_numel[layer_idx, comp_idx] += param.numel()

        _add_gpt2_head_relevance(model, name, relevance, attention_head_abs)

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


def _greedy_preview_token_ids(
    model,
    input_ids: torch.Tensor,
    device: str,
    max_new_tokens: int,
) -> list[int]:
    """Greedy argmax continuation (diagnostic only; independent of AttnLRP backward)."""
    generated: list[int] = []
    cur = input_ids.to(device).detach()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(cur, use_cache=False).logits
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            generated.append(next_id)
            cur = torch.cat(
                [cur, torch.tensor([[next_id]], device=device, dtype=cur.dtype)],
                dim=1,
            )
    return generated


def explain_sample(
    model,
    tokenizer,
    sample: dict,
    device: str = "cuda",
    parameter_attribution: bool = False,
    save_parameter_tensors: bool = False,
    decode_preview_tokens: int = 0,
) -> dict:
    """Run efficient GPT-2 AttnLRP with the contrastive GPT-2 seed."""
    grad_states = None
    if parameter_attribution:
        grad_states = _set_parameter_grad_state(model, True)

    enc = tokenizer(sample["prompt"], return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    embed = model.get_input_embeddings()(input_ids)
    embed.requires_grad_(True)
    embed.retain_grad()

    captured, handles = _register_block_forward_hooks(model)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(inputs_embeds=embed, use_cache=False).logits
        last_logits = logits[0, -1]
        target_id = int(last_logits.argmax(dim=-1).item())

        mask = torch.full_like(last_logits, -1.0 / last_logits.numel())
        mask[target_id] = 1.0
        last_logits.backward(mask)

        token_relevance = (
            (embed * embed.grad).float().sum(-1).squeeze(0).detach().cpu()
        )

        L = len(model.transformer.h)
        T = token_relevance.shape[0]
        layer_relevance = torch.zeros(L, T)
        for idx, hidden in captured.items():
            if hidden.grad is not None:
                layer_relevance[idx] = (
                    (hidden * hidden.grad).float().sum(-1).squeeze(0).detach().cpu()
                )

        parameter_result = (
            _collect_parameter_relevance(model, save_tensors=save_parameter_tensors)
            if parameter_attribution
            else {}
        )
    finally:
        for h in handles:
            h.remove()
        if grad_states is not None:
            _restore_parameter_grad_state(grad_states)
            model.zero_grad(set_to_none=True)

    preview_fields: dict = {}
    if decode_preview_tokens and decode_preview_tokens > 0:
        preview_ids = _greedy_preview_token_ids(
            model, input_ids, device, decode_preview_tokens
        )
        preview_fields = {
            "greedy_preview_token_ids": preview_ids,
            "greedy_preview_tokens": tokenizer.convert_ids_to_tokens(preview_ids),
            "greedy_preview_text": tokenizer.decode(
                preview_ids, skip_special_tokens=False
            ),
        }

    return {
        "tokens": tokens,
        "token_relevance": token_relevance,
        "layer_relevance": layer_relevance,
        "target_token": tokenizer.convert_ids_to_tokens([target_id])[0],
        "target_id": target_id,
        "input_ids": input_ids.detach().cpu(),
        **preview_fields,
        **parameter_result,
    }
