"""AttnLRP explanation: per-token and per-layer relevance scores."""
from __future__ import annotations

import torch


def _register_block_hooks(model):
    """Hook each transformer block so we can read (activation, grad) after backward."""
    captured: list[tuple[int, torch.Tensor]] = []
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden.retain_grad()
            captured.append((idx, hidden))
        return hook

    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_hook(i)))
    return captured, handles


def explain_sample(model, tokenizer, sample: dict, device: str = "cuda") -> dict:
    """Run AttnLRP on a single SQuAD_v2 sample.

    Returns
    -------
    dict with
      tokens           : list[str]           length T
      token_relevance  : torch.Tensor [T]    input-token relevance
      layer_relevance  : torch.Tensor [L, T] per-layer token relevance
      target_token     : str                 explained token (argmax next)
      target_id        : int
      input_ids        : torch.Tensor [1, T]
    """
    enc = tokenizer(sample["prompt"], return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # feed embeddings so we can differentiate w.r.t. them
    embed = model.get_input_embeddings()(input_ids)
    embed.requires_grad_(True)
    embed.retain_grad()

    captured, handles = _register_block_hooks(model)
    try:
        logits = model(inputs_embeds=embed).logits  # [1, T, V]
        last_logits = logits[0, -1]
        target_id = int(last_logits.argmax(dim=-1).item())
        target_logit = last_logits[target_id]

        model.zero_grad(set_to_none=True)
        target_logit.backward()

        # R_i = (dy / dh_i) · h_i, summed over hidden dim  (AttnLRP input × gradient).
        token_relevance = (embed.grad * embed).float().sum(-1).squeeze(0).detach().cpu()

        L = len(captured)
        layer_relevance = torch.zeros(L, token_relevance.shape[0])
        for idx, hidden in captured:
            if hidden.grad is None:
                continue
            rel = (hidden.grad * hidden).float().sum(-1).squeeze(0).detach().cpu()
            layer_relevance[idx] = rel
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
