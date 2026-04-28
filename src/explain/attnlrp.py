"""AttnLRP explanation: per-token (input) and per-layer relevance scores.

Per-layer trace follows the official lxt recipe in
``docs/source/latent-feature-attribution.rst``: under ``attnlrp.register``
the autograd backward pass produces LRP relevance scores in place of
ordinary gradients, so the relevance at the residual-stream output of
transformer block ``l`` is just ``grad_output_l.sum(-1)``.
"""
from __future__ import annotations

import torch


def _register_block_backward_hooks(model):
    """Capture per-layer relevance via ``register_full_backward_hook``."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, _grad_input, grad_output):
            g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            if g is not None:
                captured[idx] = g.detach().float().cpu()
        return hook

    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_full_backward_hook(make_hook(i)))
    return captured, handles


def explain_sample(model, tokenizer, sample: dict, device: str = "cuda") -> dict:
    """Run AttnLRP on a single SQuAD_v2 sample.

    Returns
    -------
    dict with
      tokens           : list[str]           length T
      token_relevance  : torch.Tensor [T]    input-token relevance
      layer_relevance  : torch.Tensor [L, T] per-layer token relevance (residual stream)
      target_token     : str                 explained token (argmax next)
      target_id        : int
      input_ids        : torch.Tensor [1, T]
    """
    enc = tokenizer(sample["prompt"], return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    embed = model.get_input_embeddings()(input_ids)
    embed.requires_grad_(True)
    embed.retain_grad()

    captured, handles = _register_block_backward_hooks(model)
    try:
        logits = model(inputs_embeds=embed, use_cache=False).logits  # [1, T, V]
        last_logits = logits[0, -1]
        target_id = int(last_logits.argmax(dim=-1).item())
        target_logit = last_logits[target_id]

        model.zero_grad(set_to_none=True)
        # Initialize upstream relevance with 1.0 (NOT target_logit). On GPT-2
        # the argmax logit is typically large-negative (~-100) because LM-head
        # outputs are unnormalized, so passing it as gradient would flip all
        # signs and invert MoRF/LeRF.
        target_logit.backward()

        # Token-level: input × gradient form (same convention as lxt's vit_torch
        # example). Matches Blücher 2024 MoRF/LeRF expectations on GPT-2 SQuAD.
        token_relevance = (embed * embed.grad).float().sum(-1).squeeze(0).detach().cpu()

        L = len(model.transformer.h)
        T = token_relevance.shape[0]
        layer_relevance = torch.zeros(L, T)
        for idx, g in captured.items():
            layer_relevance[idx] = g.squeeze(0).sum(-1)
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
