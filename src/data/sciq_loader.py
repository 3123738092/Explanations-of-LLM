"""SciQ loader with the same QA prompt format as the fine-tuning script.

Prompt template::
    Question: <question>\nContext: <context>\nOptions:\nA. <option>\n...\nAnswer:

The option order is shuffled deterministically with ``shuffle_seed + row_index``.
"""
from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset


def _shuffle_options(
    correct: str, distractors: list[str], seed: int
) -> tuple[list[str], str]:
    options = [correct] + distractors
    rng = random.Random(seed)
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    letter_map = {opt: letters[i] for i, opt in enumerate(options)}
    return options, letter_map[correct]


def format_sciq_prompt(
    question: str,
    context: str,
    options: list[str],
) -> str:
    letters = ["A", "B", "C", "D"]
    option_lines = "\n".join(f"{letters[i]}. {options[i]}" for i in range(4))
    return (
        f"Question: {question}\nContext: {context}\nOptions:\n{option_lines}\nAnswer:"
    )


def _rows_from_parquet(path: Path):
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Reading a SciQ parquet file requires pandas: pip install pandas"
        ) from e
    df = pd.read_parquet(path)
    for idx, row in df.iterrows():
        yield int(idx), row


def _rows_from_hf(dataset_name: str, split: str):
    ds = load_dataset(dataset_name, split=split)
    for row_idx, ex in enumerate(ds):
        yield row_idx, ex


def load_sciq_samples(
    num_samples: int = 50,
    max_length: int = 512,
    tokenizer=None,
    split: str = "validation",
    shuffle_seed: int = 42,
    dataset_name: str = "allenai/sciq",
    parquet_path: str | Path | None = None,
) -> list[dict]:
    """Load SciQ samples with prompts that match the fine-tuning prefix."""
    samples: list[dict] = []

    if parquet_path is not None:
        row_iter = _rows_from_parquet(Path(parquet_path))
    else:
        row_iter = _rows_from_hf(dataset_name, split)

    for row_idx, row in row_iter:
        question = str(row["question"]).strip()
        context = str(row["support"]).strip()
        correct = str(row["correct_answer"]).strip()
        distractors = [
            str(row["distractor1"]).strip(),
            str(row["distractor2"]).strip(),
            str(row["distractor3"]).strip(),
        ]

        options, correct_letter = _shuffle_options(
            correct, distractors, shuffle_seed + row_idx
        )
        prompt = format_sciq_prompt(question, context, options)

        if tokenizer is not None:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]  #tokenizer.encode(prompt)返回的是一个类似于字典的东西，有两个键 input_ids, attention_mask。 input_ids是token的id列表，attention_mask是注意力掩码列表。
            if ids.shape[1] > max_length:
                continue

        split_tag = Path(parquet_path).stem if parquet_path else split
        samples.append(
            {
                "id": f"sciq-{split_tag}-{row_idx}",
                "prompt": prompt,
                "question": question,
                "context": context,
                "answer": correct,
                "gold_letter": correct_letter,
                "distractor1": distractors[0],
                "distractor2": distractors[1],
                "distractor3": distractors[2],
            }
        )
        if len(samples) >= num_samples:
            break

    return samples
