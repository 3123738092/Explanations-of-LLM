"""Efficient AttnLRP explanation for GPT-2 using Input*Gradient."""
from __future__ import annotations

import torch


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


def explain_sample(model, tokenizer, sample: dict, device: str = "cuda") -> dict:
    """Run efficient GPT-2 AttnLRP with the contrastive GPT-2 seed."""
    enc = tokenizer(sample["prompt"], return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    embed = model.get_input_embeddings()(input_ids)
    embed.requires_grad_(True)
    embed.retain_grad()

    captured, handles = _register_block_forward_hooks(model)
    try:
        logits = model(inputs_embeds=embed, use_cache=False).logits
        last_logits = logits[0, -1]
        target_id = int(last_logits.argmax(dim=-1).item())

        model.zero_grad(set_to_none=True)
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
    finally:
        for h in handles:
            h.remove()

    return {
        "tokens": tokens,
        "token_relevance": token_relevance,
        "layer_relevance": layer_relevance,
        "target_token": tokenizer.convert_ids_to_tokens([target_id])[0],
        "target_id": target_id,
        "input_ids": input_ids.detach().cpu(),
    }
