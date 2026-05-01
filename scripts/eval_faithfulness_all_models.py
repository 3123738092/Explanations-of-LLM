#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_squad_texts(path: Path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    texts = []
    for article in obj.get('data', []):
        for para in article.get('paragraphs', []):
            context = para.get('context', '').strip()
            for qa in para.get('qas', []):
                question = qa.get('question', '').strip()
                if qa.get('is_impossible', False):
                    answer = 'unanswerable'
                else:
                    answers = qa.get('answers', [])
                    answer = answers[0]['text'].strip() if answers else 'unanswerable'
                texts.append(f"Question: {question}\nContext: {context}\nAnswer: {answer}")
    return texts


def load_sciq_texts(path: Path):
    df = pd.read_parquet(path)
    texts = []
    for _, row in df.iterrows():
        q = str(row['question']).strip()
        c = str(row['support']).strip()
        a = str(row['correct_answer']).strip()
        texts.append(f"Question: {q}\nContext: {c}\nAnswer: {a}")
    return texts


def prob_of_target(model, input_ids: torch.Tensor, target_id: int) -> float:
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[target_id].item())


def faithfulness_auc(model, tokenizer, input_ids, token_rel, target_id, steps, strategy):
    mask_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    t_len = input_ids.shape[1]
    descending = strategy == 'morf'
    order = torch.argsort(token_rel, descending=descending)

    probs = [prob_of_target(model, input_ids, target_id)]
    step_positions = torch.linspace(0, t_len, steps + 1).long().tolist()[1:]
    for k in step_positions:
        perturbed = input_ids.clone()
        perturbed[0, order[:k]] = mask_id
        probs.append(prob_of_target(model, perturbed, target_id))

    probs_t = torch.tensor(probs)
    return float(torch.trapz(probs_t).item() / (len(probs_t) - 1))


def compute_relevance(model, tokenizer, text: str, device: str, max_length: int):
    enc = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc.input_ids.to(device)
    input_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(inputs_embeds=input_embeds, use_cache=False).logits
    last_logits = logits[0, -1, :]
    target_id = int(torch.argmax(last_logits, dim=-1).item())

    # same contrastive seed as training-time faithfulness path
    mask = torch.ones_like(last_logits) * (-1.0 / last_logits.numel())
    mask[target_id] = 1.0
    last_logits.backward(mask)

    token_rel = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
    return input_ids.detach(), token_rel, target_id


def evaluate_model_on_texts(model, tokenizer, texts, device, max_length, steps, random_seed):
    rng = torch.Generator().manual_seed(random_seed)
    auc_morf = []
    auc_lerf = []
    auc_random_morf = []
    auc_random_lerf = []

    for text in texts:
        with torch.enable_grad():
            input_ids, token_rel, target_id = compute_relevance(model, tokenizer, text, device, max_length)

        auc_morf.append(faithfulness_auc(model, tokenizer, input_ids, token_rel, target_id, steps, 'morf'))
        auc_lerf.append(faithfulness_auc(model, tokenizer, input_ids, token_rel, target_id, steps, 'lerf'))

        rand_rel = torch.randn(token_rel.shape, generator=rng)
        auc_random_morf.append(faithfulness_auc(model, tokenizer, input_ids, rand_rel, target_id, steps, 'morf'))
        auc_random_lerf.append(faithfulness_auc(model, tokenizer, input_ids, rand_rel, target_id, steps, 'lerf'))

    n = max(1, len(texts))
    return {
        'n_cases': len(texts),
        'auc_morf': float(sum(auc_morf) / n),
        'auc_lerf': float(sum(auc_lerf) / n),
        'auc_random_morf': float(sum(auc_random_morf) / n),
        'auc_random_lerf': float(sum(auc_random_lerf) / n),
        'morf_better_than_random': float(sum(auc_morf) / n) < float(sum(auc_random_morf) / n),
        'lerf_better_than_random': float(sum(auc_lerf) / n) > float(sum(auc_random_lerf) / n),
    }


def discover_model_paths(outputs_dir: Path):
    paths = []
    for p in sorted(outputs_dir.rglob('*')):
        if p.is_dir() and (p.name == 'final' or p.name.startswith('checkpoint-')):
            if (p / 'config.json').exists():
                paths.append(p)
    return paths


def load_done_keys(path: Path):
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding='utf-8'))
    done = set()
    for r in rows:
        done.add((r['model_id'], r['dataset']))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_root', default='/home/Feng/code/Explanations-of-LLM')
    parser.add_argument('--data_root', default='/home/Feng/code/LRP-eXplains-Transformers/data')
    parser.add_argument('--base_model', default='/home/Feng/code/LRP-eXplains-Transformers/model/gpt2-model')
    parser.add_argument('--outputs_dir', default='outputs')
    parser.add_argument('--out_dir', default='outputs/faithfulness_all_models')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--max_cases_per_dataset', type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    data_root = Path(args.data_root)
    outputs_dir = repo_root / args.outputs_dir
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    squad_texts = load_squad_texts(data_root / 'SQuAD' / 'dev-v2.0.json')
    sciq_texts = load_sciq_texts(data_root / 'sciq' / 'validation-00000-of-00001.parquet')
    if args.max_cases_per_dataset > 0:
        squad_texts = squad_texts[:args.max_cases_per_dataset]
        sciq_texts = sciq_texts[:args.max_cases_per_dataset]

    datasets = {
        'squad_v2_dev': squad_texts,
        'sciq_validation': sciq_texts,
    }

    model_paths = discover_model_paths(outputs_dir)
    model_specs = [('base_gpt2', Path(args.base_model))] + [(f"{p.parent.name}/{p.name}", p) for p in model_paths]

    results_json = out_dir / 'summary.json'
    progress_jsonl = out_dir / 'progress.jsonl'
    done = load_done_keys(results_json)
    all_rows = json.loads(results_json.read_text(encoding='utf-8')) if results_json.exists() else []

    for model_id, model_path in model_specs:
        for ds_name, texts in datasets.items():
            key = (model_id, ds_name)
            if key in done:
                continue

            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch.bfloat16).to(args.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            metrics = evaluate_model_on_texts(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=args.device,
                max_length=args.max_length,
                steps=args.steps,
                random_seed=args.random_seed,
            )
            elapsed = time.time() - t0

            row = {
                'model_id': model_id,
                'model_path': str(model_path),
                'dataset': ds_name,
                'elapsed_sec': elapsed,
                **metrics,
            }
            all_rows.append(row)
            results_json.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding='utf-8')
            with progress_jsonl.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

            del model
            torch.cuda.empty_cache()

    # add random baseline table derived from existing runs if needed for convenient filtering
    # (random values are already included per row as auc_random_*)
    print(json.dumps({'out_dir': str(out_dir), 'num_rows': len(all_rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
