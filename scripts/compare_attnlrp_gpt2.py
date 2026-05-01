#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
from transformers.models.gpt2 import modeling_gpt2

from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens


def render_compare_heatmap(tokens, rel_base, rel_tuned, out_png, title_top, title_bottom):
    n = min(len(tokens), len(rel_base), len(rel_tuned))
    tokens = tokens[:n]
    rel_base = rel_base[:n]
    rel_tuned = rel_tuned[:n]

    fig_w = max(12, min(42, n * 0.24))
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 4.8), sharex=True)

    for ax, vals, title in [
        (axes[0], rel_base, title_top),
        (axes[1], rel_tuned, title_bottom),
    ]:
        im = ax.imshow(vals.unsqueeze(0).numpy(), cmap="bwr", aspect="auto", vmin=-1, vmax=1)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(tokens, rotation=90, fontsize=7)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def compute_attnlrp(model, tokenizer, prompt, device):
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    input_embeds = model.get_input_embeddings()(input_ids)
    input_embeds = input_embeds.detach().requires_grad_(True)

    output_logits = model(inputs_embeds=input_embeds, use_cache=False).logits

    # contrastive explanation for GPT2 stability
    last_logits = output_logits[0, -1, :]
    max_logit, max_index = torch.max(last_logits, dim=-1)
    mask = torch.ones_like(last_logits) * (-1.0 / last_logits.size(-1))
    mask[max_index] = 1.0
    last_logits.backward(mask)

    relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
    denom = relevance.abs().max()
    if denom > 0:
        relevance = relevance / denom

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)
    pred_token = tokenizer.decode([max_index.item()])

    return tokens, relevance, max_index.item(), pred_token, float(max_logit.detach().cpu())


def load_model(model_path, device):
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


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--finetuned_model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--squad_dev", default=str(root / "data" / "SQuAD" / "dev-v2.0.json"))
    parser.add_argument("--num_cases", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monkey_patch(modeling_gpt2, verbose=False)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    base = load_model(args.base_model, args.device)
    tuned = load_model(args.finetuned_model, args.device)

    cases = load_squad_cases(Path(args.squad_dev), args.num_cases)

    summary = []

    for c in cases:
        cid = c["id"]
        prompt = c["prompt"]

        b_tokens, b_rel, b_idx, b_pred, b_logit = compute_attnlrp(base, tokenizer, prompt, args.device)
        t_tokens, t_rel, t_idx, t_pred, t_logit = compute_attnlrp(tuned, tokenizer, prompt, args.device)

        # keep equal length for fair quick stats (same tokenizer/prompt should already match)
        n = min(len(b_tokens), len(t_tokens))
        diff_l1 = float((b_rel[:n] - t_rel[:n]).abs().mean())

        render_compare_heatmap(
            b_tokens,
            b_rel,
            t_rel,
            out_dir / f"{cid}_base_vs_finetuned.png",
            f"{cid} | base gpt2 | pred={b_pred!r}",
            f"{cid} | finetuned gpt2 | pred={t_pred!r}",
        )

        summary.append(
            {
                "case_id": cid,
                "qa_id": c["qa_id"],
                "title": c["title"],
                "question": c["question"],
                "base_pred_token_id": b_idx,
                "base_pred_token": b_pred,
                "base_max_logit": b_logit,
                "finetuned_pred_token_id": t_idx,
                "finetuned_pred_token": t_pred,
                "finetuned_max_logit": t_logit,
                "mean_abs_relevance_diff": diff_l1,
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
