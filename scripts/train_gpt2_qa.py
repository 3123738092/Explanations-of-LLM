#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate.faithfulness import faithfulness_score


def format_squad_samples(path: Path, eos_token: str) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    samples = []
    for article in data:
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "").strip()
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "").strip()
                if qa.get("is_impossible", False):
                    answer = "unanswerable"
                else:
                    answers = qa.get("answers", [])
                    answer = answers[0]["text"].strip() if answers else "unanswerable"

                prompt = f"Question: {question}\nContext: {context}\nAnswer:"
                target = f" {answer}{eos_token}"
                samples.append(
                    {
                        "prompt": prompt,
                        "target": target,
                        "eval_target": answer,
                        "task": "squad_text",
                    }
                )
    return samples


def _shuffle_options(correct: str, distractors: List[str], seed: int) -> Tuple[List[str], str]:
    options = [correct] + distractors
    rng = random.Random(seed)
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    letter_map = {opt: letters[i] for i, opt in enumerate(options)}
    return options, letter_map[correct]


def format_sciq_samples(path: Path, eos_token: str, shuffle_seed: int) -> List[Dict[str, str]]:
    df = pd.read_parquet(path)
    samples = []
    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        context = str(row["support"]).strip()
        correct = str(row["correct_answer"]).strip()
        distractors = [
            str(row["distractor1"]).strip(),
            str(row["distractor2"]).strip(),
            str(row["distractor3"]).strip(),
        ]

        options, correct_letter = _shuffle_options(correct, distractors, shuffle_seed + int(idx))
        letters = ["A", "B", "C", "D"]
        option_lines = "\n".join([f"{letters[i]}. {options[i]}" for i in range(4)])

        prompt = f"Question: {question}\nContext: {context}\nOptions:\n{option_lines}\nAnswer:"
        target = f" {correct_letter}{eos_token}"
        samples.append(
            {
                "prompt": prompt,
                "target": target,
                "eval_target": correct_letter,
                "task": "sciq_letter",
            }
        )
    return samples


def build_datasets(
    data_dir: Path,
    dataset_name: str,
    eos_token: str,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    sciq_shuffle_seed: int = 42,
):
    if dataset_name == "squad_v2":
        train_samples = format_squad_samples(data_dir / "SQuAD" / "train-v2.0.json", eos_token=eos_token)
        dev_samples = format_squad_samples(data_dir / "SQuAD" / "dev-v2.0.json", eos_token=eos_token)
        # Use tail 1/3 of dev split for faster evaluation while keeping training set unchanged.
        start = (2 * len(dev_samples)) // 3
        eval_samples = dev_samples[start:]
    elif dataset_name == "sciq":
        train_samples = format_sciq_samples(
            data_dir / "sciq" / "train-00000-of-00001.parquet", eos_token=eos_token, shuffle_seed=sciq_shuffle_seed
        )
        eval_samples = format_sciq_samples(
            data_dir / "sciq" / "validation-00000-of-00001.parquet", eos_token=eos_token, shuffle_seed=sciq_shuffle_seed
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if max_train_samples > 0:
        train_samples = train_samples[:max_train_samples]
    if max_eval_samples > 0:
        eval_samples = eval_samples[:max_eval_samples]

    return Dataset.from_list(train_samples), Dataset.from_list(eval_samples)


def _prob_of_target(model, input_ids: torch.Tensor, target_id: int) -> float:
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[target_id].item())


def _compute_attnlrp_relevance(model, tokenizer, text: str, device: str, max_length: int):
    enc = tokenizer(
        text,
        return_tensors="pt",
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

    # Contrastive seed for GPT-2 explanation stability.
    mask = torch.ones_like(last_logits) * (-1.0 / last_logits.numel())
    mask[target_id] = 1.0
    last_logits.backward(mask)

    token_relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
    return input_ids.detach(), token_relevance, target_id


def build_compute_metrics_fn(tokenizer):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.array(logits)
        labels = np.array(labels)

        # With preprocess_logits_for_metrics enabled, preds are already token ids.
        # Depending on accelerate/transformers versions, shape may be flattened.
        if preds.ndim == labels.ndim + 1:
            preds = np.argmax(preds, axis=-1)
        if preds.shape != labels.shape and preds.size == labels.size:
            preds = preds.reshape(labels.shape)

        # CausalLM predicts next token: align prediction at t with label at t+1.
        if preds.ndim == 2 and labels.ndim == 2 and preds.shape[1] > 1 and labels.shape[1] > 1:
            preds = preds[:, :-1]
            labels = labels[:, 1:]

        mask = labels != -100
        if mask.sum() == 0:
            return {"eval_token_acc": 0.0, "eval_seq_acc": 0.0}

        token_acc = float((preds[mask] == labels[mask]).mean())

        seq_match = []
        for i in range(labels.shape[0]):
            m = mask[i]
            if not m.any():
                continue
            seq_match.append(bool(np.array_equal(preds[i][m], labels[i][m])))
        seq_acc = float(np.mean(seq_match)) if seq_match else 0.0

        return {"eval_token_acc": token_acc, "eval_seq_acc": seq_acc}

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    # Reduce cached eval tensors from [B, T, V] logits to [B, T] token ids.
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


class FaithfulnessTrainer(Trainer):
    def __init__(
        self,
        *args,
        tokenizer,
        eval_texts,
        faithfulness_eval_samples: int,
        faithfulness_steps: int,
        faithfulness_sample_seed: int,
        max_length: int,
        compute_faithfulness_on_eval: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.faith_tokenizer = tokenizer
        self.eval_texts = eval_texts
        self.faithfulness_eval_samples = faithfulness_eval_samples
        self.faithfulness_steps = faithfulness_steps
        self.faithfulness_sample_seed = faithfulness_sample_seed
        self.max_length = max_length
        self.compute_faithfulness_on_eval = compute_faithfulness_on_eval

    def _compute_faithfulness_metrics(self):
        if not self.compute_faithfulness_on_eval or self.faithfulness_eval_samples <= 0:
            return {}

        model = self.model
        model.eval()
        device = str(next(model.parameters()).device)

        n = min(self.faithfulness_eval_samples, len(self.eval_texts))
        if n == 0:
            return {}

        sample_rng = torch.Generator().manual_seed(self.faithfulness_sample_seed + int(self.state.global_step))
        perm = torch.randperm(len(self.eval_texts), generator=sample_rng).tolist()
        sampled_texts = [self.eval_texts[i] for i in perm[:n]]

        rng = torch.Generator().manual_seed(self.faithfulness_sample_seed)
        auc_morf, auc_lerf, auc_random_morf, auc_random_lerf = [], [], [], []

        for text in sampled_texts:
            with torch.enable_grad():
                input_ids, token_rel, target_id = _compute_attnlrp_relevance(
                    model, self.faith_tokenizer, text, device=device, max_length=self.max_length
                )

            explain = {
                "input_ids": input_ids,
                "token_relevance": token_rel,
                "target_id": target_id,
            }
            auc_morf.append(
                faithfulness_score(
                    model,
                    self.faith_tokenizer,
                    explain_result=explain,
                    device=device,
                    steps=self.faithfulness_steps,
                    strategy="morf",
                )
            )
            auc_lerf.append(
                faithfulness_score(
                    model,
                    self.faith_tokenizer,
                    explain_result=explain,
                    device=device,
                    steps=self.faithfulness_steps,
                    strategy="lerf",
                )
            )

            rand_rel = torch.randn(token_rel.shape, generator=rng)
            rand_explain = {
                "input_ids": input_ids,
                "token_relevance": rand_rel,
                "target_id": target_id,
            }
            auc_random_morf.append(
                faithfulness_score(
                    model,
                    self.faith_tokenizer,
                    explain_result=rand_explain,
                    device=device,
                    steps=self.faithfulness_steps,
                    strategy="morf",
                )
            )
            auc_random_lerf.append(
                faithfulness_score(
                    model,
                    self.faith_tokenizer,
                    explain_result=rand_explain,
                    device=device,
                    steps=self.faithfulness_steps,
                    strategy="lerf",
                )
            )

        return {
            "eval_auc_morf": float(sum(auc_morf) / len(auc_morf)),
            "eval_auc_lerf": float(sum(auc_lerf) / len(auc_lerf)),
            "eval_auc_random_morf": float(sum(auc_random_morf) / len(auc_random_morf)),
            "eval_auc_random_lerf": float(sum(auc_random_lerf) / len(auc_random_lerf)),
            "eval_faithfulness_samples": float(n),
        }

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        extra = self._compute_faithfulness_metrics()
        if extra:
            metrics.update(extra)
            self.log(extra)
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune local GPT-2 on SQuAD_v2 or SciQ using CausalLM")
    parser.add_argument("--dataset", choices=["squad_v2", "sciq"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--faithfulness_eval_samples", type=int, default=8)
    parser.add_argument("--faithfulness_steps", type=int, default=20)
    parser.add_argument("--faithfulness_sample_seed", type=int, default=42)
    parser.add_argument("--sciq_shuffle_seed", type=int, default=42)
    parser.add_argument("--disable_faithfulness_eval", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_ds, eval_ds = build_datasets(
        data_dir=data_dir,
        dataset_name=args.dataset,
        eos_token=tokenizer.eos_token,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        sciq_shuffle_seed=args.sciq_shuffle_seed,
    )

    # Faithfulness should be measured on model inputs, not prompt+gold target (prevents leakage).
    eval_texts = [x["prompt"] for x in eval_ds]

    def tokenize_fn(batch):
        input_ids_list = []
        attn_list = []
        labels_list = []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

            # Keep supervision tokens intact; truncate prompt first when too long.
            if len(target_ids) >= args.max_length:
                target_ids = target_ids[-args.max_length :]
                prompt_ids = []
            else:
                max_prompt_len = args.max_length - len(target_ids)
                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            attention_mask = [1] * len(input_ids)

            pad_len = args.max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                labels = labels + [-100] * pad_len

            input_ids_list.append(input_ids)
            attn_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_list,
            "labels": labels_list,
        }

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["prompt", "target", "eval_target", "task"])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["prompt", "target", "eval_target", "task"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = FaithfulnessTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics_fn(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        eval_texts=eval_texts,
        faithfulness_eval_samples=args.faithfulness_eval_samples,
        faithfulness_steps=args.faithfulness_steps,
        faithfulness_sample_seed=args.faithfulness_sample_seed,
        max_length=args.max_length,
        compute_faithfulness_on_eval=(not args.disable_faithfulness_eval),
    )

    meta = {
        "dataset": args.dataset,
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "faithfulness_eval_samples": args.faithfulness_eval_samples,
        "faithfulness_steps": args.faithfulness_steps,
        "faithfulness_sample_seed": args.faithfulness_sample_seed,
        "sciq_shuffle_seed": args.sciq_shuffle_seed,
        "compute_faithfulness_on_eval": (not args.disable_faithfulness_eval),
        "prompting": {
            "squad": "Question/Context/Answer with target answer+EOS; loss only on answer span",
            "sciq": "Question/Context/Options/Answer with target option-letter+EOS; loss only on answer span",
        },
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
