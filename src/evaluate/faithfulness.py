"""Faithfulness evaluation via MoRF / LeRF token flipping.

Follows the pixel/token-flipping protocol in Blücher et al. 2024: progressively
replace tokens (in descending relevance order for MoRF, ascending for LeRF)
with a mask/pad token and measure the target-class probability along the curve.
Lower AUC for MoRF (or higher for LeRF) means the attribution is more faithful.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _prob_of_target(model, input_ids: torch.Tensor, target_id: int) -> float:
    logits = model(input_ids=input_ids).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    return float(probs[target_id].item())


def faithfulness_score(
    model,
    tokenizer,
    explain_result: dict,
    device: str = "cuda",
    steps: int = 20,
    strategy: str = "morf",
) -> float:
    input_ids: torch.Tensor = explain_result["input_ids"].to(device)
    target_id: int = explain_result["target_id"]
    rel: torch.Tensor = explain_result["token_relevance"].clone()

    mask_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    T = input_ids.shape[1]
    descending = strategy == "morf"
    order = torch.argsort(rel, descending=descending)

    probs = [_prob_of_target(model, input_ids, target_id)]
    step_positions = torch.linspace(0, T, steps + 1).long().tolist()[1:]
    for k in step_positions:
        perturbed = input_ids.clone()
        perturbed[0, order[:k]] = mask_id
        probs.append(_prob_of_target(model, perturbed, target_id))

    probs_t = torch.tensor(probs)
    # mean probability over the curve = normalised AUC
    return float(torch.trapz(probs_t).item() / (len(probs_t) - 1))
