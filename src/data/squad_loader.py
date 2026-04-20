"""SQuAD_v2 loader: format each example as a GPT-2 QA prompt."""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset


def format_prompt(question: str, context: str) -> str:
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def load_squad_v2_samples(
    num_samples: int = 50,
    max_length: int = 512,
    tokenizer=None,
    split: str = "validation",
) -> list[dict]:
    ds = load_dataset("squad_v2", split=split)
    samples: list[dict] = []
    for ex in ds:
        prompt = format_prompt(ex["question"], ex["context"])
        if tokenizer is not None:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            if ids.shape[1] > max_length:
                continue
        answer: Optional[str] = (
            ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else None
        )
        samples.append(
            {
                "id": ex["id"],
                "prompt": prompt,
                "question": ex["question"],
                "context": ex["context"],
                "answer": answer,
            }
        )
        if len(samples) >= num_samples:
            break
    return samples
