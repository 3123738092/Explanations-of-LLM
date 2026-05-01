#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import modeling_gpt2

from src.data.squad_loader import load_squad_v2_samples
from lxt.efficient import monkey_patch


def prob_of_target(model, input_ids: torch.Tensor, target_id: int) -> float:
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[target_id].item())


def faithfulness_auc(model, tokenizer, input_ids, token_rel, target_id, steps, strategy):
    mask_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    t_len = input_ids.shape[1]
    descending = strategy == "morf"
    order = torch.argsort(token_rel, descending=descending)

    probs = [prob_of_target(model, input_ids, target_id)]
    step_positions = torch.linspace(0, t_len, steps + 1).long().tolist()[1:]
    for k in step_positions:
        perturbed = input_ids.clone()
        perturbed[0, order[:k]] = mask_id
        probs.append(prob_of_target(model, perturbed, target_id))

    probs_t = torch.tensor(probs)
    return float(torch.trapz(probs_t).item() / (len(probs_t) - 1))


def compute_relevance(model, tokenizer, prompt: str, device: str, max_length: int):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True)
    input_ids = enc.input_ids.to(device)
    input_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(inputs_embeds=input_embeds, use_cache=False).logits
    last_logits = logits[0, -1, :]
    target_id = int(torch.argmax(last_logits, dim=-1).item())

    mask = torch.ones_like(last_logits) * (-1.0 / last_logits.numel())
    mask[target_id] = 1.0
    last_logits.backward(mask)

    token_rel = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
    return input_ids.detach(), token_rel, target_id


def evaluate_model(model, tokenizer, samples, device="cuda", max_length=512, steps=20, random_seed=0):
    rng = torch.Generator().manual_seed(random_seed)
    ms, ls, rms, rls = [], [], [], []
    for s in samples:
        with torch.enable_grad():
            input_ids, rel, target_id = compute_relevance(model, tokenizer, s["prompt"], device, max_length)
        ms.append(faithfulness_auc(model, tokenizer, input_ids, rel, target_id, steps, "morf"))
        ls.append(faithfulness_auc(model, tokenizer, input_ids, rel, target_id, steps, "lerf"))
        rand = torch.randn(rel.shape, generator=rng)
        rms.append(faithfulness_auc(model, tokenizer, input_ids, rand, target_id, steps, "morf"))
        rls.append(faithfulness_auc(model, tokenizer, input_ids, rand, target_id, steps, "lerf"))

    n = len(samples)
    return {
        "n_samples": n,
        "auc_morf": float(sum(ms)/n),
        "auc_lerf": float(sum(ls)/n),
        "auc_random_morf": float(sum(rms)/n),
        "auc_random_lerf": float(sum(rls)/n),
    }


def discover_models(outputs_dir: Path):
    out = []
    for p in sorted(outputs_dir.rglob("*")):
        if p.is_dir() and (p.name == "final" or p.name.startswith("checkpoint-")) and (p / "config.json").exists():
            out.append((f"{p.parent.name}/{p.name}", p))
    return out


def main():
    root = Path('/home/Feng/code/Explanations-of-LLM')
    base_model = Path('/home/Feng/code/LRP-eXplains-Transformers/model/gpt2-model')
    out_dir = root / 'outputs' / 'faithfulness_readme50_compare'
    out_dir.mkdir(parents=True, exist_ok=True)

    base_tok = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token

    samples = load_squad_v2_samples(
        num_samples=50,
        max_length=512,
        tokenizer=base_tok,
        split='validation',
    )

    model_specs = [('base_gpt2', base_model)] + discover_models(root / 'outputs')

    monkey_patch(modeling_gpt2, verbose=False)

    rows = []
    for model_id, model_path in model_specs:
        tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch.bfloat16).to('cuda')
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        m = evaluate_model(model, tok, samples, device='cuda', max_length=512, steps=20, random_seed=0)
        row = {'model_id': model_id, 'model_path': str(model_path), **m}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        del model
        torch.cuda.empty_cache()

    (out_dir / 'summary.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
