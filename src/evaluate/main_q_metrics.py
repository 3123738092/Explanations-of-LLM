"""Main-Q metric pipeline: argmax target + contrastive seed + faithfulness AUC."""
from __future__ import annotations

import torch
from faithfulness import faithfulness_score


def explain_prompt_main_q(model, tokenizer, prompt: str, device: str = "cuda", max_length: int = 512):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True)
    input_ids = enc.input_ids.to(device)
    input_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(inputs_embeds=input_embeds, use_cache=False).logits
    last_logits = logits[0, -1, :]
    target_id = int(torch.argmax(last_logits, dim=-1).item())
    target_logit = last_logits[target_id]
    target_logit.backward()
    token_relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
    return {"input_ids": input_ids.detach().cpu(), "token_relevance": token_relevance, "target_id": target_id}


def evaluate_prompts_main_q(model, tokenizer, prompts: list[str], device: str = "cuda",
                            max_length: int = 512, steps: int = 20, random_seed: int = 0):
    rng = torch.Generator().manual_seed(random_seed)
    auc_morf, auc_lerf, auc_random = [], [], []
    for prompt in prompts:
        with torch.enable_grad():
            explain = explain_prompt_main_q(model, tokenizer, prompt, device, max_length)
        auc_morf.append(faithfulness_score(model, tokenizer, explain, device, steps, "morf"))
        auc_lerf.append(faithfulness_score(model, tokenizer, explain, device, steps, "lerf"))
        rand_explain = {**explain, "token_relevance": torch.randn(explain["token_relevance"].shape, generator=rng)}
        auc_random.append(faithfulness_score(model, tokenizer, rand_explain, device, steps, "morf"))
    n = max(1, len(prompts))
    return {
        "n_samples": len(prompts),
        "auc_morf": float(sum(auc_morf) / n),
        "auc_lerf": float(sum(auc_lerf) / n),
        "auc_random": float(sum(auc_random) / n),
    }
