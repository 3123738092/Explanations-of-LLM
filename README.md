# Explanations of LLM — AttnLRP on GPT-2

NLP course project (topic 7). **Part 1** reproduces AttnLRP [Achtibat et al., ICML 2024]
on SQuAD_v2 with GPT-2, computes per-token + per-layer relevance, visualises them,
and evaluates faithfulness via MoRF / LeRF token-flipping AUC [Blücher et al., 2024]
with a random-relevance baseline.

## References

- [1] Bach et al., *Layer-wise Relevance Propagation*, PLOS ONE 2015.
- [2] Achtibat et al., *AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers*, ICML 2024.
- [3] Blücher et al., *Decoupling pixel flipping and occlusion strategy for consistent XAI benchmarks*, 2024.
- Reference implementation: <https://github.com/rachtibat/LRP-eXplains-Transformers>

## Project layout

```
code/
├── configs/gpt2_efficient.yaml  # experiment config
├── main.py                      # Part 1 entry point
├── requirements.txt
├── scripts/run_part1.sh
├── third_party/
│   └── LRP-eXplains-Transformers/   # upstream repo (read-only)
└── src/
    ├── data/squad_loader.py     # SQuAD_v2 → QA prompt
    ├── models/gpt2_efficient_wrapper.py   # lxt efficient GPT-2 + monkey_patch
    ├── explain/attnlrp_gpt2_efficient.py  # Input*Gradient relevance under efficient rules
    ├── evaluate/faithfulness.py # MoRF / LeRF AUC
    └── visualize/heatmap.py     # token & layer heatmaps
```

## Setup

This repo uses the GPT-2 **efficient** AttnLRP path:

- `src/models/gpt2_efficient_wrapper.py`
- `src/explain/attnlrp_gpt2_efficient.py`

### Version notes (transformers)

- Use `model.family: gpt2_efficient` (see config below).
- Newer Transformers (e.g. `transformers==4.52.4`) should be used with this
  efficient path (`lxt.efficient.monkey_patch`).

```bash
conda create -n llm-xai-paper python=3.10 -y
conda activate llm-xai-paper

# torch 2.1 with CUDA 12.1
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# baseline deps (efficient path)
pip install transformers==4.52.4 accelerate tabulate matplotlib zennit \
            datasets pyyaml tqdm
pip install lxt   # = LRP-eXplains-Transformers
```

Verify the env:

```python
from lxt.efficient import monkey_patch  # must import cleanly
```

## Reproduction details (Part 1)

### Algorithm

We do **not** re-implement the LRP rules. We load GPT-2 and patch the Hugging Face
implementation via `lxt.efficient.monkey_patch`, then propagate relevance via
PyTorch autograd:

```python
# src/models/gpt2_efficient_wrapper.py
from transformers.models.gpt2 import modeling_gpt2
from lxt.efficient import monkey_patch
monkey_patch(modeling_gpt2, verbose=False)
```

Once patched, `loss.backward()` no longer computes ordinary gradients — the
patched autograd functions propagate **relevance**. So
`(input_embeds.grad * input_embeds).sum(-1)` gives the AttnLRP relevance of each
token (matches the reference example
`third_party/LRP-eXplains-Transformers/examples/paper/llama.py`).

Per-layer relevance comes from a forward-hook on each `model.transformer.h[l]`
that calls `hidden.retain_grad()`; after backward, `(hidden.grad * hidden).sum(-1)`
is the layer-`l` token relevance.

### Data

50 examples from `squad_v2[validation]`. Each is formatted as

```
Context: <passage>
Question: <q>
Answer:
```

We keep only prompts that fit `max_length=512` GPT-2 tokens.

### Target logit explained

For each prompt we compute the next-token logits at the final position, take
`argmax` as the predicted token, and run AttnLRP w.r.t. that scalar logit.

### Faithfulness (Blücher et al., 2024)

For every sample we run **token-flipping** with two strategies and compute the
target-probability AUC over `steps=20` flip levels:

- **MoRF** — flip Most-Relevant-First (descending relevance). Lower AUC ⇒ more faithful.
- **LeRF** — flip Least-Relevant-First (ascending relevance). Higher AUC ⇒ more faithful.

To occlude a token we replace its id with the pad/eos token id and re-run a
no-grad forward, recording `softmax(logits)[target_id]`.

### Random baseline

For each sample we draw `R_random ~ N(0, I)` of the same length and run the
same MoRF / LeRF pipeline on it (seed `0`). AttnLRP is faithful iff

```
auc_morf       <  auc_random_morf
auc_lerf       >  auc_random_lerf
```

## How to run

```bash
conda activate llm-xai-paper
python main.py --config configs/gpt2_efficient.yaml
```

### Run the efficient GPT-2 path

`main.py` selects model/explainer by `model.family`. Use:

```yaml
model:
  family: gpt2_efficient
  name: gpt2
  device: cuda
  dtype: bfloat16
```

Then run as usual:

```bash
python main.py --config configs/gpt2_efficient.yaml
```

and keep `model.family: gpt2_efficient`.

Run time: **~2 minutes** for 50 samples on a single CUDA device (gpt2-small,
fp32). Output tree:

```
outputs/
├── faithfulness_summary.json     # config + per-sample + mean AUCs
├── figures/
│   ├── sample_NNN_tokens.png     # 1 × T token-level heatmap
│   └── sample_NNN_layers.png     # L × T per-layer heatmap
└── relevance/
    └── sample_NNN.pt             # {tokens, token_relevance, layer_relevance,
                                  #  target_token, target_id, input_ids}
```

The cached `.pt` tensors are reused by Parts 2 / 3 so we never re-run AttnLRP
on identical inputs.

## Results (Part 1)

50 SQuAD_v2 validation samples, GPT-2 base (124 M params), fp32, single GPU,
`steps=20`, seed `0`.

| Metric | AttnLRP | Random baseline | Faithful? |
|---|---:|---:|---|
| `auc_morf` (lower better) | **0.0077** | 0.0499 | ✅ AttnLRP ≪ random |
| `auc_lerf` (higher better) | **0.0839** | 0.0567 | ✅ AttnLRP > random |

Interpretation:

- **MoRF**: removing the most-relevant tokens collapses the target probability
  ~6.5× faster under AttnLRP than under random ranking — relevance scores really
  do identify the tokens the model relies on.
- **LeRF**: removing the least-relevant tokens degrades the prediction much less
  under AttnLRP than under random — the bottom of the AttnLRP ranking is
  genuinely irrelevant to the prediction.

Both directions of the Blücher-2024 sanity check pass, so the reproduction is
working as intended.

Per-sample numbers and the full config are saved in
`outputs/faithfulness_summary.json`.

### Sample heatmaps

`outputs/figures/sample_000_tokens.png` shows the 1 × T token-relevance heatmap
for the first sample (target token `Normandy`); `sample_000_layers.png` shows
the same sample as an L × T grid (12 GPT-2 transformer blocks × prompt tokens).
Red = positive relevance, blue = negative.

## Part 1 checklist

- [x] Load GPT-2 base & tokenizer (lxt `GPT2LMHeadModel`)
- [x] Register AttnLRP rules via `attnlrp.register(model)`
- [x] Load SQuAD_v2 validation split and format as QA prompts
- [x] Compute per-token relevance `R(x_i)` from input embeddings
- [x] Capture per-layer relevance `R^(l)(x_i)` via block forward hooks
- [x] Visualise token / layer heatmaps
- [x] Faithfulness MoRF + LeRF AUC + random baseline

## Notes for Part 2 / Part 3

- **Part 2** — sequentially fine-tune GPT-2 on SQuAD_v2 → SciQ, then re-run
  `src/explain/attnlrp_gpt2_efficient.py` on identical prompts and diff pre / post-FT
  per-layer relevance. The `outputs/relevance/sample_NNN.pt` files cache the
  pre-FT tensors so the diff is just a load + subtract.
- **Part 3** — fine-tune two separate models (Model A on SciQ, Model B on
  SQuAD_v2), modify the AttnLRP propagation to attribute the explained logit
  to **parameter** tensors (not input tokens), and compare per-layer parameter
  relevance between A and B.
