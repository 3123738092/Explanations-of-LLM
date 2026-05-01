"""SQuAD_v2 loader: format each example as a GPT-2 QA prompt."""
from __future__ import annotations

import json
from pathlib import Path
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
    try:
        ds = load_dataset("squad_v2", split=split)
    except Exception:
        # Offline fallback: use local SQuAD files if HF dataset hub is unavailable.
        local_root = Path("/home/Feng/code/LRP-eXplains-Transformers/data/SQuAD")
        local_file = local_root / ("dev-v2.0.json" if split == "validation" else "train-v2.0.json")
        obj = json.loads(local_file.read_text(encoding="utf-8"))
        ds = []
        for article in obj.get("data", []):
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    ds.append(
                        {
                            "id": qa.get("id", ""),
                            "question": qa.get("question", ""),
                            "context": context,
                            "answers": {"text": [a.get("text", "") for a in qa.get("answers", [])]},
                        }
                    )
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
