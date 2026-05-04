"""SciQ loader: 与 ``train_gpt2_qa.format_sciq_samples`` 一致的 QA 提示格式。

提示模板::
    Question: …\\nContext: …\\nOptions:\\nA. …\\n…\\nAnswer:

选项顺序由 ``shuffle_seed + 行索引`` 固定随机打乱（与微调脚本一致）。
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
            "读取 SciQ parquet 需要安装 pandas：pip install pandas"
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
    """加载 SciQ 样本；``prompt`` 与微调时 CausalLM 前缀一致（不含答案）。"""
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
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
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
