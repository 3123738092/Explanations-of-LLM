"""SQuAD_v2 loader: format each example as a GPT-2 QA prompt."""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset

# ds = load_dataset("squad_v2", split="validation")
# ex = ds[0]
# ex 是 dict，键是列名，例如 SQuAD:
# ex["id"], ex["question"], ex["context"], ex["answers"], ...
# 我们的写法：for ex in ds:   # 按顺序从 0 到 len(ds)-1

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
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]  #tokenizer.encode(prompt)返回的是一个类似于字典的东西，有两个键 input_ids, attention_mask。 input_ids是token的id列表，attention_mask是注意力掩码列表。
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
