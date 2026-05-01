#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import modeling_gpt2
from lxt.efficient import monkey_patch


def load_sciq_samples(parquet_path: Path, tokenizer, num_samples: int = 50, max_length: int = 512):
    df = pd.read_parquet(parquet_path)
    samples = []
    for _, row in df.iterrows():
        q = str(row['question']).strip()
        c = str(row['support']).strip()
        a = str(row['correct_answer']).strip()
        prompt = f"Question: {q}\nContext: {c}\nAnswer: {a}"

        ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)['input_ids']
        if ids.shape[1] > max_length:
            continue

        samples.append({'prompt': prompt, 'question': q})
        if len(samples) >= num_samples:
            break
    return samples


def prob_of_target(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_id: int) -> float:
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[target_id].item())


def faithfulness_auc(model, tokenizer, input_ids, attention_mask, token_rel, target_id, steps, strategy):
    mask_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    t_len = input_ids.shape[1]
    descending = strategy == "morf"
    order = torch.argsort(token_rel, descending=descending)

    probs = [prob_of_target(model, input_ids, attention_mask, target_id)]
    step_positions = torch.linspace(0, t_len, steps + 1).long().tolist()[1:]
    for k in step_positions:
        perturbed = input_ids.clone()
        perturbed[0, order[:k]] = mask_id
        perturbed_mask = attention_mask.clone()
        probs.append(prob_of_target(model, perturbed, perturbed_mask, target_id))

    probs_t = torch.tensor(probs)
    return float(torch.trapz(probs_t).item() / (len(probs_t) - 1))


def compute_relevance(model, tokenizer, prompt: str, device: str, max_length: int):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    input_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(inputs_embeds=input_embeds, attention_mask=attention_mask, use_cache=False).logits
    last_logits = logits[0, -1, :]
    target_id = int(torch.argmax(last_logits, dim=-1).item())

    mask = torch.ones_like(last_logits) * (-1.0 / last_logits.numel())
    mask[target_id] = 1.0
    last_logits.backward(mask)

    token_rel = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
    return input_ids.detach(), attention_mask.detach(), token_rel, target_id


def evaluate_model(model, tokenizer, samples, device="cuda", max_length=512, steps=20, random_seed=0):
    rng = torch.Generator().manual_seed(random_seed)
    ms, ls, rms, rls = [], [], [], []
    for s in samples:
        with torch.enable_grad():
            input_ids, attention_mask, rel, target_id = compute_relevance(model, tokenizer, s["prompt"], device, max_length)
        ms.append(faithfulness_auc(model, tokenizer, input_ids, attention_mask, rel, target_id, steps, "morf"))
        ls.append(faithfulness_auc(model, tokenizer, input_ids, attention_mask, rel, target_id, steps, "lerf"))
        rand = torch.randn(rel.shape, generator=rng)
        rms.append(faithfulness_auc(model, tokenizer, input_ids, attention_mask, rand, target_id, steps, "morf"))
        rls.append(faithfulness_auc(model, tokenizer, input_ids, attention_mask, rand, target_id, steps, "lerf"))

    n = len(samples)
    return {
        "n_samples": n,
        "auc_morf": float(sum(ms) / n),
        "auc_lerf": float(sum(ls) / n),
        "auc_random_morf": float(sum(rms) / n),
        "auc_random_lerf": float(sum(rls) / n),
    }


def discover_models(outputs_dir: Path):
    out = []
    for p in sorted(outputs_dir.rglob("*")):
        if (
            p.is_dir()
            and (p.name == "final" or p.name.startswith("checkpoint-"))
            and (p / "config.json").exists()
            and ((p / "tokenizer_config.json").exists() or (p / "tokenizer.json").exists())
        ):
            out.append((f"{p.parent.name}/{p.name}", p))
    return out


def main():
    root = Path('/home/Feng/code/Explanations-of-LLM')
    base_model = Path('/home/Feng/code/LRP-eXplains-Transformers/model/gpt2-model')
    sciq_val = Path('/home/Feng/code/LRP-eXplains-Transformers/data/sciq/validation-00000-of-00001.parquet')
    out_dir = root / 'outputs' / 'faithfulness_sciq_fixed_compare'
    out_dir.mkdir(parents=True, exist_ok=True)

    monkey_patch(modeling_gpt2, verbose=False)

    base_tok = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token

    samples = load_sciq_samples(sciq_val, tokenizer=base_tok, num_samples=50, max_length=512)

    model_specs = [('base_gpt2', base_model)] + discover_models(root / 'outputs')

    rows = []
    skipped = []
    for model_id, model_path in model_specs:
        try:
            tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), local_files_only=True, torch_dtype=torch.bfloat16
            ).to('cuda')
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            metrics = evaluate_model(model, tok, samples, device='cuda', max_length=512, steps=20, random_seed=0)
            row = {'model_id': model_id, 'model_path': str(model_path), **metrics}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            skipped.append({"model_id": model_id, "model_path": str(model_path), "reason": str(e)})
            print(json.dumps({"skipped": model_id, "reason": str(e)}, ensure_ascii=False))

    (out_dir / 'summary.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'skipped.json').write_text(json.dumps(skipped, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
