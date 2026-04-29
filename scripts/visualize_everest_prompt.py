#!/usr/bin/env python3
"""Standalone AttnLRP visualisation for the LXT quickstart-style Everest QA prompt.

Uses ``src.explain.attnlrp_gpt2_efficient.explain_sample`` and ``load_gpt2_efficient_with_attnlrp`` (efficient Input*Gradient / monkey_patch path), plus ``src.visualize.heatmap``.

Run from repo root::

    cd Explanations-of-LLM
    python scripts/visualize_everest_prompt.py
    python scripts/visualize_everest_prompt.py --out-dir outputs/everest_demo --model gpt2
    # PDF (needs TeX): add --pdf-backend xelatex for full Unicode; use --no-pdf to skip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Prefer vendored LXT so ``lxt.utils.pdf_heatmap`` matches third_party sources.
_LXT_SRC = _ROOT / "third_party" / "LRP-eXplains-Transformers"
if _LXT_SRC.is_dir() and str(_LXT_SRC) not in sys.path:
    sys.path.insert(0, str(_LXT_SRC))

import numpy as np
import torch

from lxt.utils import clean_tokens, pdf_heatmap

from src.explain.attnlrp_gpt2_efficient import explain_sample
from src.models.gpt2_efficient_wrapper import load_gpt2_efficient_with_attnlrp
from src.visualize.heatmap import save_layer_heatmap, save_token_heatmap

# Same text as LXT quickstart (GPT-2 contrastive section); line continuation matches the doc string.
PROMPT = """Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""


def main() -> None:
    parser = argparse.ArgumentParser(description="AttnLRP heatmaps for the Everest QA prompt.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/outputs/"),
        help="Directory for PNG figures (created if missing).",
    )
    parser.add_argument("--model", type=str, default="gpt2", help="HF model id (e.g. gpt2, gpt2-medium).")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Tokenizer truncation limit (GPT-2 n_ctx is 1024).",
    )
    parser.add_argument(
        "--pdf-backend",
        choices=("pdflatex", "xelatex"),
        default="pdflatex",
        help="LaTeX engine for lxt.utils.pdf_heatmap (xelatex: broader Unicode).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip token-level PDF (requires a LaTeX installation).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gpt2_efficient_with_attnlrp(args.model, device)

    # Truncate by tokens so we never exceed GPT-2 context.
    ids = tokenizer.encode(PROMPT, max_length=args.max_length, truncation=True)
    prompt = tokenizer.decode(ids)
    if len(ids) >= args.max_length:
        print(f"Warning: prompt truncated to {args.max_length} tokens.", file=sys.stderr)

    sample = {"prompt": prompt}
    result = explain_sample(model, tokenizer, sample, device=device)

    print(f"device={device} model={args.model}")
    print(f"explained next-token prediction: {result['target_token']!r} (id={result['target_id']})")
    print(f"n_tokens={result['input_ids'].shape[1]}")

    stem = "everest_prompt"
    save_token_heatmap(
        result["tokens"],
        result["token_relevance"],
        out_path=args.out_dir / f"{stem}_tokens.png",
        title=f"Token relevance — target next token '{result['target_token']}'",
    )
    save_layer_heatmap(
        result["tokens"],
        result["layer_relevance"],
        out_path=args.out_dir / f"{stem}_layers.png",
        title="Per-layer relevance (residual stream, row-wise norm)",
    )

    torch.save(
        {
            "tokens": result["tokens"],
            "token_relevance": result["token_relevance"],
            "layer_relevance": result["layer_relevance"],
            "target_token": result["target_token"],
            "target_id": result["target_id"],
            "input_ids": result["input_ids"],
            "prompt": prompt,
        },
        args.out_dir / f"{stem}_relevance.pt",
    )

    # Same normalization as LXT quickstart / pdf_heatmap contract: relevances in [-1, 1].
    if not args.no_pdf:
        rel = result["token_relevance"].float().numpy()
        scale = max(float(np.abs(rel).max()), 1e-9)
        rel_norm = np.clip(rel / scale, -1.0, 1.0)
        try:
            words = clean_tokens(list(result["tokens"]))
            pdf_path = args.out_dir / f"{stem}_tokens.pdf"
            pdf_heatmap(
                words,
                rel_norm,
                path=str(pdf_path),
                backend=args.pdf_backend,
            )
            print(f"Saved: {pdf_path}")
        except Exception as e:
            print(f"PDF export skipped or failed: {e}", file=sys.stderr)
            print(
                "Install TeX (e.g. texlive-xetex / texlive-latex-base) or pass --no-pdf.",
                file=sys.stderr,
            )

    print(f"Saved: {args.out_dir / (stem + '_tokens.png')}")
    print(f"Saved: {args.out_dir / (stem + '_layers.png')}")
    print(f"Saved: {args.out_dir / (stem + '_relevance.pt')}")


if __name__ == "__main__":
    main()
