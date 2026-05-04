"""Entry point for Part 1: AttnLRP over SQuAD_v2.

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

from src.data.sciq_loader import load_sciq_samples
from src.data.squad_loader import load_squad_v2_samples
from src.evaluate.faithfulness import faithfulness_score
from src.explain.attnlrp import explain_sample as explain_gpt2_sample
from src.explain.attnlrp_gpt2_efficient import (
    explain_sample as explain_gpt2_efficient_sample,
)
from src.models.gpt2_wrapper import load_gpt2_with_attnlrp
from src.visualize.heatmap import save_layer_heatmap, save_token_heatmap

try:
    from src.visualize.heatmap import (
        save_attention_head_heatmap,
        save_layer_param_line,
        save_layer_parameter_trend,
    )
except ImportError:
    save_attention_head_heatmap = None
    save_layer_param_line = None
    save_layer_parameter_trend = None


def _resolve_pretrained_local_path(name: str, cfg_path: Path) -> str:
    """Resolve local checkpoint dirs; leave Hugging Face hub ids (e.g. ``gpt2``) unchanged."""
    path = Path(name)
    if path.is_absolute():
        return str(path.resolve())
    repo_root = cfg_path.parent.parent if cfg_path.parent.name == "configs" else cfg_path.parent
    for base in (repo_root, Path.cwd()):
        candidate = (base / path).resolve()
        if candidate.exists():
            return str(candidate)
    return name


def _faith(model, tokenizer, result, device, steps, strategy):
    return faithfulness_score(
        model, tokenizer, result, device=device, steps=steps, strategy=strategy
    )


def _random_result(result: dict, generator: torch.Generator) -> dict:
    rand_rel = torch.randn(result["token_relevance"].shape, generator=generator)
    return {**result, "token_relevance": rand_rel}


def _load_model_and_explainer(model_cfg: dict):
    family = model_cfg.get("family", "gpt2")
    name = model_cfg["name"]
    device = model_cfg["device"]

    if family == "gpt2":
        tokenizer_name = model_cfg.get("tokenizer_name")
        return (*load_gpt2_with_attnlrp(name, device, tokenizer_name=tokenizer_name), explain_gpt2_sample)
    if family == "gpt2_efficient":
        from src.models.gpt2_efficient_wrapper import load_gpt2_efficient_with_attnlrp
        dtype = model_cfg.get("dtype", "bfloat16")
        tokenizer_name = model_cfg.get("tokenizer_name")
        return (
            *load_gpt2_efficient_with_attnlrp(
                name, device, dtype=dtype, tokenizer_name=tokenizer_name
            ),
            explain_gpt2_efficient_sample,
        )
    raise ValueError(f"Unsupported model family: {family}")


def _load_dataset_samples(cfg: dict, cfg_path: Path, tokenizer):
    data_cfg = cfg["data"]
    dataset = data_cfg.get("dataset", "squad_v2")
    common = {
        "num_samples": data_cfg["num_samples"],
        "max_length": data_cfg["max_length"],
        "tokenizer": tokenizer,
        "split": data_cfg.get("split", "validation"),
    }
    if dataset == "squad_v2":
        return load_squad_v2_samples(**common)
    if dataset == "sciq":
        parquet_raw = data_cfg.get("sciq_parquet_path")
        parquet_path = None
        if parquet_raw:
            parquet_path = Path(
                _resolve_pretrained_local_path(str(parquet_raw), cfg_path)
            )
        return load_sciq_samples(
            **common,
            shuffle_seed=int(data_cfg.get("sciq_shuffle_seed", 42)),
            dataset_name=data_cfg.get("hf_dataset", "allenai/sciq"),
            parquet_path=parquet_path,
        )
    raise ValueError(f"Unsupported data.dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2_efficient.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg["model"]["name"] = _resolve_pretrained_local_path(cfg["model"]["name"], cfg_path)
    out_dir = Path(cfg["output"]["dir"])
    fig_dir = Path(cfg["output"]["figures_dir"])
    rel_dir = out_dir / "relevance"
    param_dir = out_dir / "parameter_relevance"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_tokens = fig_dir / "tokens"
    fig_layers = fig_dir / "layers"
    fig_param_heads = fig_dir / "parameter_heads"
    fig_param_layers_line = fig_dir / "parameter_layers_line"
    fig_param_layers_block = fig_dir / "parameter_layers_block"
    for d in (fig_tokens, fig_layers, fig_param_heads,
              fig_param_layers_line, fig_param_layers_block):
        d.mkdir(parents=True, exist_ok=True)
    rel_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)

    device = cfg["model"]["device"]
    steps = cfg["faithfulness"]["steps"]
    rng = torch.Generator().manual_seed(0)
    param_cfg = cfg.get("parameter_attribution", {})
    param_enabled = bool(param_cfg.get("enabled", False))
    save_parameter_tensors = bool(param_cfg.get("save_tensors", False))
    top_parameter_modules = int(param_cfg.get("top_modules", 20))

    model, tokenizer, explain_sample = _load_model_and_explainer(cfg["model"])
    samples = _load_dataset_samples(cfg, cfg_path, tokenizer)

    records = []
    for i, sample in enumerate(tqdm(samples, desc="AttnLRP")):
        try:
            result = explain_sample(
                model,
                tokenizer,
                sample,
                device=device,
                parameter_attribution=param_enabled,
                save_parameter_tensors=save_parameter_tensors,
            )
        except TypeError:
            # Non-parameter explainers (e.g. standard GPT-2 AttnLRP) do not
            # accept parameter-attribution kwargs.
            result = explain_sample(model, tokenizer, sample, device=device)

        save_token_heatmap(
            result["tokens"], result["token_relevance"],
            out_path=fig_tokens / f"sample_{i:03d}_tokens.png",
            title=f"Sample {i} — target='{result['target_token']}'",
        )
        save_layer_heatmap(
            result["tokens"], result["layer_relevance"],
            out_path=fig_layers / f"sample_{i:03d}_layers.png",
            title=f"Sample {i} — per-layer relevance",
        )

        relevance_payload = {
            "tokens": result["tokens"],
            "token_relevance": result["token_relevance"],
            "layer_relevance": result["layer_relevance"],
            "target_token": result["target_token"],
            "target_id": result["target_id"],
            "input_ids": result["input_ids"],
        }
        torch.save(relevance_payload, rel_dir / f"sample_{i:03d}.pt")

        parameter_record = {}
        if param_enabled:
            parameter_summary = result["parameter_summary"]
            if save_attention_head_heatmap is not None:
                save_attention_head_heatmap(
                    parameter_summary["attention_head_abs"],
                    out_path=fig_param_heads / f"sample_{i:03d}_param_heads.png",
                    title=f"Sample {i} — attention-head parameter relevance",
                )
            if save_layer_param_line is not None:
                save_layer_param_line(
                    parameter_summary["layer_component_abs"],
                    parameter_summary["component_names"],
                    out_path=fig_param_layers_line / f"sample_{i:03d}_param_layers_line.png",
                    title=f"Sample {i} — layer-wise parameter relevance (line chart)",
                )
            if save_layer_parameter_trend is not None:
                save_layer_parameter_trend(
                    parameter_summary["layer_component_abs"],
                    parameter_summary["component_names"],
                    out_path=fig_param_layers_block / f"sample_{i:03d}_param_layers_block.png",
                    title=f"Sample {i} — layer-wise parameter relevance (block)",
                )
            parameter_payload = {
                "target_token": result["target_token"],
                "target_id": result["target_id"],
                "parameter_summary": parameter_summary,
            }
            if "parameter_relevance" in result:
                parameter_payload["parameter_relevance"] = result["parameter_relevance"]
            torch.save(parameter_payload, param_dir / f"sample_{i:03d}.pt")
            parameter_record = {
                "top_parameter_modules": parameter_summary["module_records"][
                    :top_parameter_modules
                ]
            }

        rand_result = _random_result(result, rng)
        records.append({
            "idx": i,
            "target": result["target_token"],
            "n_tokens": int(result["input_ids"].shape[1]),
            "auc_morf": _faith(model, tokenizer, result, device, steps, "morf"),
            "auc_lerf": _faith(model, tokenizer, result, device, steps, "lerf"),
            "auc_random_morf": _faith(model, tokenizer, rand_result, device, steps, "morf"),
            "auc_random_lerf": _faith(model, tokenizer, rand_result, device, steps, "lerf"),
            **parameter_record,
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
    print(
        f"Figures: {fig_dir} "
        f"(tokens/, layers/, parameter_heads/, parameter_layers_line/, parameter_layers_block/)"
    )
    print(f"Per-sample relevance tensors: {rel_dir}")
    if param_enabled:
        print(f"Per-sample parameter relevance summaries: {param_dir}")
        if save_parameter_tensors:
            print("Full parameter contribution tensors were saved.")
        else:
            print("Full parameter tensors were skipped; set parameter_attribution.save_tensors=true to save them.")


if __name__ == "__main__":
    main()
