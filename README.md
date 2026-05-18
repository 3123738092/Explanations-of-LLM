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
├── configs/
│   ├── gpt2_efficient.yaml                    # base GPT-2 / SQuAD config
│   ├── gpt2_efficient_finetuned_squad.yaml    # SQuAD fine-tuned config
│   └── gpt2_efficient_finetuned_sciq.yaml     # SciQ fine-tuned config
├── main.py                                    # direct Part 1 / Part 3 entry point
├── requirements.txt
├── RUNNING.md                                 # runnable commands for Parts 1-3
├── scripts/
│   ├── train_gpt2_qa.py                       # Part 2 fine-tuning
│   ├── squad_answer_compare.py                # SQuAD generation check
│   └── plot_finetune.py                       # report training-curve plots
├── third_party/
│   └── LRP-eXplains-Transformers/             # upstream repo fallback
└── src/
    ├── data/                                  # SQuAD / SciQ prompt loaders
    ├── models/gpt2_attnlrp_loader.py          # GPT-2 + lxt efficient monkey_patch
    ├── explain/attnlrp_gpt2_efficient.py      # token/layer AttnLRP flow
    ├── explain/parameter_attribution.py       # Part 3 parameter attribution
    ├── evaluate/faithfulness.py               # MoRF / LeRF AUC
    ├── evaluate/next_token_faithfulness.py    # Part 2 eval helper
    └── visualize/relevance_plots.py           # token, layer, parameter plots
```

## Setup

This repo uses the GPT-2 **efficient** AttnLRP path:

- `src/models/gpt2_attnlrp_loader.py`
- `src/explain/attnlrp_gpt2_efficient.py`

### Version notes (transformers)

- Use `model.family: gpt2_efficient` (see config below).
- Newer Transformers (e.g. `transformers==4.52.4`) should be used with this
  efficient path (`lxt.efficient.monkey_patch`).

```bash
conda create -n llm-xai-paper python=3.10 -y
conda activate llm-xai-paper
pip install -r requirements.txt
```

If Hugging Face or GitHub downloads are slow on the server, run:

```bash
source /etc/network_turbo
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
# src/models/gpt2_attnlrp_loader.py
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
outputs-gpt2-efficient/
├── faithfulness_summary.json
├── figures/
│   ├── tokens/sample_NNN_tokens.png
│   ├── layers/sample_NNN_layers.png
│   ├── parameter_heads/sample_NNN_param_heads.png
│   ├── parameter_layers_line/sample_NNN_param_layers_line.png
│   └── parameter_layers_block/sample_NNN_param_layers_block.png
├── relevance/sample_NNN.pt
└── parameter_relevance/sample_NNN.pt
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
`outputs-gpt2-efficient/faithfulness_summary.json`.

### Sample heatmaps

`outputs-gpt2-efficient/figures/tokens/sample_000_tokens.png` shows the 1 × T token-relevance heatmap
for the first sample (target token `Normandy`); `outputs-gpt2-efficient/figures/layers/sample_000_layers.png` shows
the same sample as an L × T grid (12 GPT-2 transformer blocks × prompt tokens).
Red = positive relevance, blue = negative.

## Part 1 checklist

- [x] Load GPT-2 base & tokenizer (lxt `GPT2LMHeadModel`)
- [x] Patch GPT-2 with `lxt.efficient.monkey_patch(modeling_gpt2)`
- [x] Load SQuAD_v2 validation split and format as QA prompts
- [x] Compute per-token relevance `R(x_i)` from input embeddings
- [x] Capture per-layer relevance `R^(l)(x_i)` via block forward hooks
- [x] Visualise token / layer heatmaps
- [x] Faithfulness MoRF + LeRF AUC + random baseline

## Notes for Part 2 / Part 3

- **Part 2** — fine-tune GPT-2 with `scripts/train_gpt2_qa.py`. The script supports
  SQuAD_v2 and SciQ, logs token accuracy, and can compute next-token MoRF/LeRF
  faithfulness during evaluation.
- **Part 3** — enable `parameter_attribution.enabled: true` in the YAML config and
  run `main.py`. Parameter relevance is computed from the patched backward pass as
  `param.grad * param`, then aggregated by layer, module, and attention head. The
  provided fine-tuned configs compare the SQuAD and SciQ checkpoints.

For exact commands, see `RUNNING.md`.
