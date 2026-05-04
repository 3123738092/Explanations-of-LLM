#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluate.main_q_metrics import evaluate_prompts_main_q

ROOT = Path(__file__).resolve().parents[1]


def load_squad_samples(path: Path, num_samples: int, max_length: int, tokenizer):
    obj = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for article in obj.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "").strip()
            for qa in para.get("qas", []):
                q = qa.get("question", "").strip()
                prompt = f"Context: {context}\nQuestion: {q}\nAnswer:"
                ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
                if ids.shape[1] > max_length:
                    continue
                out.append({"id": qa.get("id", ""), "prompt": prompt})
                if len(out) >= num_samples:
                    return out
    return out


def load_sciq_samples(path: Path, num_samples: int, max_length: int, tokenizer):
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        q = str(row["question"]).strip()
        c = str(row["support"]).strip()
        prompt = f"Context: {c}\nQuestion: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[1] > max_length:
            continue
        out.append({"id": "", "prompt": prompt})
        if len(out) >= num_samples:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_tokenizer_path", default="/home/Feng/code/LRP-eXplains-Transformers/model/gpt2-model")
    parser.add_argument("--dataset", choices=["squad_v2", "sciq"], default="sciq")
    parser.add_argument("--squad_json", default="/home/Feng/code/LRP-eXplains-Transformers/data/SQuAD/dev-v2.0.json")
    parser.add_argument("--sciq_parquet", default="/home/Feng/code/LRP-eXplains-Transformers/data/sciq/validation-00000-of-00001.parquet")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    model_path = Path(args.model_path).resolve()
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(Path(args.base_tokenizer_path).resolve()), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == "squad_v2":
        samples = load_squad_samples(Path(args.squad_json), args.num_samples, args.max_length, tokenizer)
    else:
        samples = load_sciq_samples(Path(args.sciq_parquet), args.num_samples, args.max_length, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    metrics = evaluate_prompts_main_q(
        model=model,
        tokenizer=tokenizer,
        prompts=[s["prompt"] for s in samples],
        device=device,
        max_length=args.max_length,
        steps=args.steps,
        random_seed=args.random_seed,
    )
    summary = {
        "model_path": str(model_path),
        "dataset": args.dataset,
        "n_samples": len(samples),
        "mean_auc_morf": metrics["auc_morf"],
        "mean_auc_lerf": metrics["auc_lerf"],
        "mean_auc_random_morf": metrics["auc_random_morf"],
        "mean_auc_random_lerf": metrics["auc_random_lerf"],
    }

    if args.out_json:
        out_json = Path(args.out_json).resolve()
    else:
        out_dir = ROOT / "outputs" / "teammate_style_faithfulness"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"{args.dataset}_{model_path.parent.name}_{model_path.name}_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
