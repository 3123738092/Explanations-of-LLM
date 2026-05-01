#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.gpt2 import modeling_gpt2

from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens


def load_model(model_path: Path, device: str):
    model = modeling_gpt2.GPT2LMHeadModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_squad_cases(squad_dev_path: Path, max_cases: int):
    obj = json.loads(squad_dev_path.read_text(encoding="utf-8"))
    cases = []
    case_idx = 0

    for article in obj["data"]:
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para.get("context", "").strip()
            qas = para.get("qas", [])
            for question_idx, qa in enumerate(qas):
                question = qa.get("question", "").strip()
                qa_id = qa.get("id", "")
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                cases.append(
                    {
                        "id": f"case_{case_idx:03d}",
                        "title": title,
                        "qa_id": qa_id,
                        "question_idx": question_idx,
                        "prompt": prompt,
                        "question": question,
                    }
                )
                case_idx += 1
                if len(cases) >= max_cases:
                    return cases
    return cases


@torch.no_grad()
def get_tokens(tokenizer, prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
    tokens = clean_tokens(tokenizer.convert_ids_to_tokens(input_ids))
    return tokens


def layerwise_attnlrp(model, tokenizer, prompt: str, device: str):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids.to(device)

    input_embeds = model.get_input_embeddings()(input_ids)
    input_embeds = input_embeds.detach().requires_grad_(True)

    outputs = model(
        inputs_embeds=input_embeds,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    logits = outputs.logits
    hidden_states = list(outputs.hidden_states)

    for h in hidden_states:
        h.retain_grad()

    last_logits = logits[0, -1, :]
    max_logit, max_index = torch.max(last_logits, dim=-1)

    # Contrastive mask for GPT-2 explanation stability
    mask = torch.ones_like(last_logits) * (-1.0 / last_logits.size(-1))
    mask[max_index] = 1.0
    last_logits.backward(mask)

    layer_scores = []
    for h in hidden_states:
        if h.grad is None:
            rel = torch.zeros(h.shape[:2], device=h.device, dtype=torch.float32)
        else:
            rel = (h.grad * h).float().sum(dim=-1)
        rel = rel.detach().cpu()[0]
        denom = rel.abs().max()
        if denom > 0:
            rel = rel / denom
        layer_scores.append(rel)

    tokens = clean_tokens(tokenizer.convert_ids_to_tokens(input_ids[0]))
    pred_token = tokenizer.decode([max_index.item()])

    return {
        "tokens": tokens,
        "layer_scores": layer_scores,
        "pred_token_id": int(max_index.item()),
        "pred_token": pred_token,
        "max_logit": float(max_logit.detach().cpu()),
    }


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    n = min(a.numel(), b.numel())
    a = a[:n].float()
    b = b[:n].float()
    if float(a.norm()) == 0.0 or float(b.norm()) == 0.0:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=0).item())


def render_case_heatmap(tokens, base_layers, tuned_layers, out_png: Path, title: str):
    num_layers = min(len(base_layers), len(tuned_layers))
    seq_len = min(len(tokens), min(x.numel() for x in base_layers[:num_layers]), min(x.numel() for x in tuned_layers[:num_layers]))

    mat_base = torch.stack([x[:seq_len] for x in base_layers[:num_layers]], dim=0).numpy()
    mat_tuned = torch.stack([x[:seq_len] for x in tuned_layers[:num_layers]], dim=0).numpy()

    fig_w = max(12, min(44, seq_len * 0.22))
    fig_h = max(7, min(18, num_layers * 0.7 + 3))
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

    im0 = axes[0].imshow(mat_base, cmap="bwr", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title(f"Base GPT-2 | {title}", fontsize=10)
    axes[0].set_ylabel("Layer")
    axes[0].set_yticks(range(num_layers))

    im1 = axes[1].imshow(mat_tuned, cmap="bwr", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title(f"Finetuned GPT-2 | {title}", fontsize=10)
    axes[1].set_ylabel("Layer")
    axes[1].set_yticks(range(num_layers))
    axes[1].set_xticks(range(seq_len))
    axes[1].set_xticklabels(tokens[:seq_len], rotation=90, fontsize=6)

    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def render_global_layer_plot(global_stats, out_png: Path):
    layers = [x["layer"] for x in global_stats]
    base_vals = [x["base_mean_abs"] for x in global_stats]
    tuned_vals = [x["finetuned_mean_abs"] for x in global_stats]
    delta_vals = [x["delta_mean_abs"] for x in global_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, base_vals, marker="o", label="base mean|rel|")
    ax.plot(layers, tuned_vals, marker="o", label="finetuned mean|rel|")
    ax.plot(layers, delta_vals, marker="o", label="delta (ft-base)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title("Layer-wise Relevance Comparison (Averaged Across Cases)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--finetuned_model", required=True)
    parser.add_argument("--squad_dev", default=str(root / "data" / "SQuAD" / "dev-v2.0.json"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_cases", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_case_heatmaps", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)

    monkey_patch(modeling_gpt2, verbose=False)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    base_model = load_model(Path(args.base_model), args.device)
    tuned_model = load_model(Path(args.finetuned_model), args.device)

    cases = load_squad_cases(Path(args.squad_dev), args.num_cases)

    case_summaries = []
    per_layer_rows = []
    global_base_abs = None
    global_tuned_abs = None
    layer_count = None

    for case in cases:
        cid = case["id"]
        prompt = case["prompt"]

        base = layerwise_attnlrp(base_model, tokenizer, prompt, args.device)
        tuned = layerwise_attnlrp(tuned_model, tokenizer, prompt, args.device)

        n_layers = min(len(base["layer_scores"]), len(tuned["layer_scores"]))
        if layer_count is None:
            layer_count = n_layers
            global_base_abs = torch.zeros(layer_count, dtype=torch.float64)
            global_tuned_abs = torch.zeros(layer_count, dtype=torch.float64)

        per_layer = []
        for li in range(n_layers):
            br = base["layer_scores"][li]
            tr = tuned["layer_scores"][li]
            n = min(br.numel(), tr.numel())
            br = br[:n]
            tr = tr[:n]

            base_abs = float(br.abs().mean())
            tuned_abs = float(tr.abs().mean())
            mean_abs_diff = float((br - tr).abs().mean())
            cos = cosine_sim(br, tr)

            global_base_abs[li] += base_abs
            global_tuned_abs[li] += tuned_abs

            row = {
                "case_id": cid,
                "qa_id": case["qa_id"],
                "layer": li,
                "base_mean_abs": base_abs,
                "finetuned_mean_abs": tuned_abs,
                "delta_mean_abs": tuned_abs - base_abs,
                "mean_abs_diff": mean_abs_diff,
                "cosine_similarity": cos,
            }
            per_layer_rows.append(row)
            per_layer.append(row)

        if args.save_case_heatmaps:
            render_case_heatmap(
                base["tokens"],
                base["layer_scores"],
                tuned["layer_scores"],
                out_dir / "cases" / f"{cid}_layerwise_base_vs_finetuned.png",
                title=f"{cid} | qa_id={case['qa_id']}",
            )

        case_json = {
            "case_id": cid,
            "qa_id": case["qa_id"],
            "title": case["title"],
            "question": case["question"],
            "prompt": prompt,
            "tokens": base["tokens"],
            "base_pred_token": base["pred_token"],
            "finetuned_pred_token": tuned["pred_token"],
            "base_layers": [x.tolist() for x in base["layer_scores"]],
            "finetuned_layers": [x.tolist() for x in tuned["layer_scores"]],
            "per_layer_metrics": per_layer,
        }
        (out_dir / "cases" / f"{cid}.json").write_text(
            json.dumps(case_json, ensure_ascii=False), encoding="utf-8"
        )

        case_summaries.append(
            {
                "case_id": cid,
                "qa_id": case["qa_id"],
                "question": case["question"],
                "base_pred_token": base["pred_token"],
                "finetuned_pred_token": tuned["pred_token"],
            }
        )

    total_cases = len(cases)
    global_stats = []
    for li in range(layer_count or 0):
        b = float(global_base_abs[li] / max(total_cases, 1))
        t = float(global_tuned_abs[li] / max(total_cases, 1))
        global_stats.append(
            {
                "layer": li,
                "base_mean_abs": b,
                "finetuned_mean_abs": t,
                "delta_mean_abs": t - b,
            }
        )

    with (out_dir / "per_layer_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "qa_id",
                "layer",
                "base_mean_abs",
                "finetuned_mean_abs",
                "delta_mean_abs",
                "mean_abs_diff",
                "cosine_similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(per_layer_rows)

    summary = {
        "num_cases": total_cases,
        "layer_count": layer_count,
        "global_layer_stats": global_stats,
        "cases": case_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    render_global_layer_plot(global_stats, out_dir / "global_layerwise_stats.png")

    print(json.dumps({"out_dir": str(out_dir), "num_cases": total_cases, "layer_count": layer_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
