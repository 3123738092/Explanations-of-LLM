"""在 SQuAD v2 提示上用模型贪心生成答案，并与数据集中的标准答案对照。

用法（与 main.py 相同配置）::

    cd Explanations-of-LLM
    python scripts/squad_answer_compare.py --config configs/gpt2_efficient_finetuned_squad.yaml

可选：只跑前 N 条、调整续写长度、把结果写入 JSONL。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import re
import string

import torch
import yaml

from src.data.squad_loader import load_squad_v2_samples
from src.models.gpt2_efficient_wrapper import load_gpt2_efficient_with_attnlrp


def _resolve_pretrained_local_path(name: str, cfg_path: Path) -> str:
    path = Path(name)
    if path.is_absolute():
        return str(path.resolve())
    repo_root = cfg_path.parent.parent if cfg_path.parent.name == "configs" else cfg_path.parent
    for base in (repo_root, Path.cwd()):
        candidate = (base / path).resolve()
        if candidate.exists():
            return str(candidate)
    return name


def _normalize_squad_answer(s: str) -> str:
    """与常见 SQuAD EM 预处理类似的轻量归一化（非官方实现，仅作粗匹配）。"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _generative_em(pred: str, gold: str | None) -> bool:
    """不可答样本 gold 为 None：若预测里出现明显拒答词则记为可能正确（启发式）。"""
    if gold is None:
        p = pred.lower()
        return any(
            x in p
            for x in ("unanswerable", "cannot answer", "no answer", "not in the context", "not found")
        )
    pn, gn = _normalize_squad_answer(pred), _normalize_squad_answer(gold)
    if not gn:
        return False
    return gn in pn or pn in gn or pn == gn


def _strip_generated_answer(raw: str) -> str:
    """截断到第一个换行或明显结束，避免把下一句 context 混进来。"""
    for sep in ("\n", "Question:", "Context:"):
        if sep in raw:
            raw = raw.split(sep, 1)[0]
    return raw.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="SQuAD greedy 生成 vs 标准答案")
    parser.add_argument("--config", default="configs/gpt2_efficient.yaml")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Answer: 之后续写的最大 token 数")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条；0 表示用配置里的 num_samples")
    parser.add_argument("--out", type=str, default="", help="可选：写入 JSONL 路径")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["model"]["name"] = _resolve_pretrained_local_path(cfg["model"]["name"], cfg_path)

    device = cfg["model"]["device"]
    dtype = cfg["model"].get("dtype", "bfloat16")
    num = args.limit if args.limit > 0 else int(cfg["data"]["num_samples"])

    model, tokenizer = load_gpt2_efficient_with_attnlrp(
        cfg["model"]["name"], device, dtype=dtype
    )

    samples = load_squad_v2_samples(
        num_samples=num,
        max_length=int(cfg["data"]["max_length"]),
        tokenizer=tokenizer,
        split=cfg["data"]["split"],
    )

    rows = []
    correct = 0
    for i, sample in enumerate(samples):
        enc = tokenizer(sample["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                enc["input_ids"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        pred_raw = full_text[len(sample["prompt"]) :]
        pred = _strip_generated_answer(pred_raw)
        gold = sample["answer"]
        em = _generative_em(pred, gold)
        if em:
            correct += 1

        row = {
            "idx": i,
            "id": sample["id"],
            "gold": gold,
            "pred": pred,
            "em_loose": em,
        }
        rows.append(row)

        gold_s = "(不可答)" if gold is None else repr(gold)
        print(f"[{i}] EM≈{em}  gold={gold_s}")
        print(f"    pred={repr(pred[:200])}{'...' if len(pred) > 200 else ''}")

    n = len(rows)
    print(f"\n粗匹配准确率（启发式 EM）: {correct}/{n} = {correct / max(1, n):.2%}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"已写入: {out_path.resolve()}")


if __name__ == "__main__":
    main()
