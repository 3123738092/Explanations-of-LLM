"""Entry point for Part 1: AttnLRP on GPT-2 over SQuAD_v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from src.data.squad_loader import load_squad_v2_samples
from src.evaluate.faithfulness import faithfulness_score
from src.explain.attnlrp import explain_sample
from src.models.gpt2_wrapper import load_gpt2_with_attnlrp
from src.visualize.heatmap import save_layer_heatmap, save_token_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    fig_dir = Path(cfg["output"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gpt2_with_attnlrp(
        cfg["model"]["name"], cfg["model"]["device"]
    )
    samples = load_squad_v2_samples(
        num_samples=cfg["data"]["num_samples"],
        max_length=cfg["data"]["max_length"],
        tokenizer=tokenizer,
        split=cfg["data"]["split"],
    )

    records = []
    for i, sample in enumerate(tqdm(samples, desc="AttnLRP")):
        result = explain_sample(
            model, tokenizer, sample, device=cfg["model"]["device"]
        )
        save_token_heatmap(
            result["tokens"],
            result["token_relevance"],
            out_path=fig_dir / f"sample_{i:03d}_tokens.png",
            title=f"Sample {i} — target='{result['target_token']}'",
        )
        save_layer_heatmap(
            result["tokens"],
            result["layer_relevance"],
            out_path=fig_dir / f"sample_{i:03d}_layers.png",
            title=f"Sample {i} — per-layer relevance",
        )
        score = faithfulness_score(
            model,
            tokenizer,
            result,
            device=cfg["model"]["device"],
            steps=cfg["faithfulness"]["steps"],
            strategy=cfg["faithfulness"]["strategy"],
        )
        records.append({"idx": i, "faithfulness_auc": score, "target": result["target_token"]})

    mean = sum(r["faithfulness_auc"] for r in records) / max(1, len(records))
    summary_path = out_dir / "faithfulness_summary.json"
    summary_path.write_text(json.dumps({"mean": mean, "per_sample": records}, indent=2))
    print(f"\nMean faithfulness AUC ({cfg['faithfulness']['strategy']}): {mean:.4f}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
