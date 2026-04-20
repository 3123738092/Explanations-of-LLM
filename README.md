# Explanations of LLM — AttnLRP on GPT-2

Base codebase for the NLP course project **"Explanations of LLM"** (topic 7).
This repo implements **Part 1**: reproducing AttnLRP on SQuAD_v2 with GPT-2,
computing per-token / per-layer relevance, visualising them, and evaluating via
faithfulness scores.

## References

- [1] Bach et al., *Layer-wise Relevance Propagation*, PLOS ONE 2015.
- [2] Achtibat et al., *AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers*, ICML 2024.
- [3] Blücher et al., *Decoupling pixel flipping and occlusion strategy for consistent XAI benchmarks*, 2024.
- Reference implementation: <https://github.com/rachtibat/LRP-eXplains-Transformers>

## Project layout

```
code/
├── configs/default.yaml          # experiment config
├── main.py                       # entry point for Part 1
├── requirements.txt
├── scripts/run_part1.sh
└── src/
    ├── data/squad_loader.py      # SQuAD_v2 prompt builder
    ├── models/gpt2_wrapper.py    # load GPT-2 + register AttnLRP
    ├── explain/attnlrp.py        # per-token + per-layer relevance
    ├── evaluate/faithfulness.py  # MoRF / LeRF token-flipping AUC
    └── visualize/heatmap.py      # token & layer heatmaps
```

## Setup

```bash
# 1. Create a fresh env (example with conda)
conda create -n llm-xai python=3.10 -y
conda activate llm-xai

# 2. Install deps
pip install -r requirements.txt
```

> `lxt` is the PyPI name of **LRP-eXplains-Transformers** (Achtibat et al.).
> If your GPU stack needs a specific CUDA build of PyTorch, install that first.

## Usage

```bash
# run Part 1 end-to-end
bash scripts/run_part1.sh

# or directly
python main.py --config configs/default.yaml
```

Outputs land under `outputs/`:

- `outputs/figures/sample_*_tokens.png` — token-level relevance heatmap
- `outputs/figures/sample_*_layers.png` — per-layer relevance (layers × tokens)
- `outputs/faithfulness_summary.txt`    — per-sample + mean faithfulness AUC

## Part 1 checklist

- [x] Load GPT-2 base & tokenizer
- [x] Register AttnLRP rules via `lxt.models.gpt2.attnlrp`
- [x] Load SQuAD_v2 validation split and format as QA prompts
- [x] Compute per-token relevance `R(x_i)` from input embeddings
- [x] Capture per-layer relevance `R^(l)(x_i)` via block forward hooks
- [x] Visualise token / layer heatmaps
- [x] Faithfulness via MoRF token flipping (AUC of target-prob drop)

## Notes for Part 2 / Part 3 (later)

- Part 2 will fine-tune GPT-2 sequentially on SQuAD_v2 → SciQ, then re-run
  `src/explain/attnlrp.py` on identical inputs to diff pre/post-FT relevance.
- Part 3 will extend `src/explain/` to compute **parameter** relevance per
  layer (not input-token), then compare Model A (SciQ-FT) vs Model B (SQuAD-FT).
