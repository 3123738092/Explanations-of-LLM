"""SciQ loader: format examples as GPT-2 QA prompts."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def format_prompt(question: str, context: str) -> str:
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def load_sciq_samples(
    num_samples: int = 50,
    max_length: int = 512,
    tokenizer=None,
    split: str = "validation",
    shuffle_seed: int = 42,
    dataset_name: str = "allenai/sciq",
    parquet_path: Path | None = None,
) -> list[dict]:
    # Offline-first: use local parquet when provided.
    if parquet_path is None:
        base = Path('/home/Feng/code/LRP-eXplains-Transformers/data/sciq')
        filename = 'validation-00000-of-00001.parquet' if split in ('validation', 'dev') else 'train-00000-of-00001.parquet'
        parquet_path = base / filename

    df = pd.read_parquet(parquet_path)

    samples: list[dict] = []
    for _, row in df.iterrows():
        question = str(row['question']).strip()
        context = str(row['support']).strip()
        answer = str(row['correct_answer']).strip()
        prompt = format_prompt(question, context)
        if tokenizer is not None:
            ids = tokenizer(prompt, return_tensors='pt')['input_ids']
            if ids.shape[1] > max_length:
                continue
        samples.append(
            {
                'id': '',
                'prompt': prompt,
                'question': question,
                'context': context,
                'answer': answer,
            }
        )
        if len(samples) >= num_samples:
            break
    return samples
