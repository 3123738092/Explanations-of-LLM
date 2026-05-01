#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def format_squad_samples(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    samples = []
    for article in data:
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "").strip()
                if qa.get("is_impossible", False):
                    answer = "unanswerable"
                else:
                    answers = qa.get("answers", [])
                    answer = answers[0]["text"].strip() if answers else "unanswerable"

                text = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
                samples.append({"text": text})
    return samples


def format_sciq_samples(path: Path) -> List[Dict[str, str]]:
    df = pd.read_parquet(path)
    samples = []
    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        context = str(row["support"]).strip()
        answer = str(row["correct_answer"]).strip()
        text = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
        samples.append({"text": text})
    return samples


def build_datasets(data_dir: Path, dataset_name: str, max_train_samples: int = 0, max_eval_samples: int = 0):
    if dataset_name == "squad_v2":
        train_samples = format_squad_samples(data_dir / "SQuAD" / "train-v2.0.json")
        eval_samples = format_squad_samples(data_dir / "SQuAD" / "dev-v2.0.json")
    elif dataset_name == "sciq":
        train_samples = format_sciq_samples(data_dir / "sciq" / "train-00000-of-00001.parquet")
        eval_samples = format_sciq_samples(data_dir / "sciq" / "validation-00000-of-00001.parquet")
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


def _faithfulness_auc(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    token_relevance: torch.Tensor,
    target_id: int,
    steps: int,
    strategy: str,
) -> float:
    mask_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    t_len = input_ids.shape[1]
    descending = strategy == "morf"
    order = torch.argsort(token_relevance, descending=descending)

    probs = [_prob_of_target(model, input_ids, target_id)]
    step_positions = torch.linspace(0, t_len, steps + 1).long().tolist()[1:]
    for k in step_positions:
        perturbed = input_ids.clone()
        perturbed[0, order[:k]] = mask_id
        probs.append(_prob_of_target(model, perturbed, target_id))

    probs_t = torch.tensor(probs)
    return float(torch.trapz(probs_t).item() / (len(probs_t) - 1))


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

        # Randomly sample N eval texts each evaluation; seed + global step keeps it reproducible.
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

            auc_morf.append(
                _faithfulness_auc(
                    model, self.faith_tokenizer, input_ids, token_rel, target_id, self.faithfulness_steps, "morf"
                )
            )
            auc_lerf.append(
                _faithfulness_auc(
                    model, self.faith_tokenizer, input_ids, token_rel, target_id, self.faithfulness_steps, "lerf"
                )
            )

            rand_rel = torch.randn(token_rel.shape, generator=rng)
            auc_random_morf.append(
                _faithfulness_auc(
                    model, self.faith_tokenizer, input_ids, rand_rel, target_id, self.faithfulness_steps, "morf"
                )
            )
            auc_random_lerf.append(
                _faithfulness_auc(
                    model, self.faith_tokenizer, input_ids, rand_rel, target_id, self.faithfulness_steps, "lerf"
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
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    eval_texts = [x["text"] for x in eval_ds]

    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        eval_strategy="steps",
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
        data_collator=data_collator,
        tokenizer=tokenizer,
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
        "compute_faithfulness_on_eval": (not args.disable_faithfulness_eval),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
