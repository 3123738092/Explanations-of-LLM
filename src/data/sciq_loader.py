"""SciQ loader: format each example as a GPT-2 QA prompt (support as context)."""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset


def format_prompt(question: str, support: str) -> str:
    return f"Context: {support}\nQuestion: {question}\nAnswer:"


def load_sciq_samples(
    num_samples: int = 50,
    max_length: int = 512,
    tokenizer=None,
    split: str = "validation",
    dataset_name: str = "allenai/sciq",
) -> list[dict]:
    ds = load_dataset(dataset_name, split=split)
    samples: list[dict] = []
    for row_idx, ex in enumerate(ds):
        prompt = format_prompt(ex["question"], ex["support"])
        if tokenizer is not None:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            if ids.shape[1] > max_length:
                continue
        answer: Optional[str] = ex.get("correct_answer")
        samples.append(
            {
                "id": f"sciq-{split}-{row_idx}",
                "prompt": prompt,
                "question": ex["question"],
                "context": ex["support"],
                "answer": answer,
                "distractor1": ex["distractor1"],
                "distractor2": ex["distractor2"],
                "distractor3": ex["distractor3"],
            }
        )
        if len(samples) >= num_samples:
            break
    return samples
