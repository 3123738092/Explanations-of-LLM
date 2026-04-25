"""Entry point for Part 1: AttnLRP on GPT-2 over SQuAD_v2.

For every sample we
  1. run AttnLRP and save token + per-layer relevance heatmaps
  2. evaluate faithfulness with MoRF and LeRF flipping
  3. evaluate the same two flips with a random-relevance baseline
  4. cache the relevance tensors as .pt for downstream analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.data.squad_loader import load_squad_v2_samples
from src.evaluate.faithfulness import faithfulness_score
from src.explain.attnlrp import explain_sample
from src.models.gpt2_wrapper import load_gpt2_with_attnlrp
from src.visualize.heatmap import save_layer_heatmap, save_token_heatmap


def _faith(model, tokenizer, result, device, steps, strategy):
    return faithfulness_score(
        model, tokenizer, result, device=device, steps=steps, strategy=strategy
    )


def _random_result(result: dict, generator: torch.Generator) -> dict:
    rand_rel = torch.randn(result["token_relevance"].shape, generator=generator)
    return {**result, "token_relevance": rand_rel}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    fig_dir = Path(cfg["output"]["figures_dir"])
    rel_dir = out_dir / "relevance"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel_dir.mkdir(parents=True, exist_ok=True)

    device = cfg["model"]["device"]
    steps = cfg["faithfulness"]["steps"]
    rng = torch.Generator().manual_seed(0)

    model, tokenizer = load_gpt2_with_attnlrp(cfg["model"]["name"], device)
    samples = load_squad_v2_samples(
        num_samples=cfg["data"]["num_samples"],
        max_length=cfg["data"]["max_length"],
        tokenizer=tokenizer,
        split=cfg["data"]["split"],
    )

    records = []
    for i, sample in enumerate(tqdm(samples, desc="AttnLRP")):
        result = explain_sample(model, tokenizer, sample, device=device)

        save_token_heatmap(
            result["tokens"], result["token_relevance"],
            out_path=fig_dir / f"sample_{i:03d}_tokens.png",
            title=f"Sample {i} — target='{result['target_token']}'",
        )
        save_layer_heatmap(
            result["tokens"], result["layer_relevance"],
            out_path=fig_dir / f"sample_{i:03d}_layers.png",
            title=f"Sample {i} — per-layer relevance",
        )

        torch.save(
            {
                "tokens": result["tokens"],
                "token_relevance": result["token_relevance"],
                "layer_relevance": result["layer_relevance"],
                "target_token": result["target_token"],
                "target_id": result["target_id"],
                "input_ids": result["input_ids"],
            },
            rel_dir / f"sample_{i:03d}.pt",
        )

        rand_result = _random_result(result, rng)
        records.append({
            "idx": i,
            "target": result["target_token"],
            "n_tokens": int(result["input_ids"].shape[1]),
            "auc_morf": _faith(model, tokenizer, result, device, steps, "morf"),
            "auc_lerf": _faith(model, tokenizer, result, device, steps, "lerf"),
            "auc_random_morf": _faith(model, tokenizer, rand_result, device, steps, "morf"),
            "auc_random_lerf": _faith(model, tokenizer, rand_result, device, steps, "lerf"),
        })

    keys = ["auc_morf", "auc_lerf", "auc_random_morf", "auc_random_lerf"]
    means = {k: sum(r[k] for r in records) / max(1, len(records)) for k in keys}

    summary = {
        "config": cfg,
        "n_samples": len(records),
        "mean": means,
        "per_sample": records,
    }
    summary_path = out_dir / "faithfulness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Part 1 results ===")
    print(f"Samples: {len(records)}")
    for k in keys:
        print(f"  mean {k}: {means[k]:.4f}")
    print("\nFaithfulness reading:")
    print("  AttnLRP is faithful if  auc_morf < auc_random_morf  AND  auc_lerf > auc_random_lerf")
    print(f"\nSummary written to: {summary_path}")
    print(f"Figures: {fig_dir}")
    print(f"Per-sample relevance tensors: {rel_dir}")


if __name__ == "__main__":
    main()
