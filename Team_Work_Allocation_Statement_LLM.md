# Team Work Allocation Statement

**Course:** AIAA 4051 — Natural Language Processing
**Project Option:** Topic 7 — Explanations of LLMs

---

## Team Members

| # | Name | Module Owned | Workload |
|---|---|---|---|
| 1 | Yuhan Chen | P1: AttnLRP base reproduction & visualization | **33%** |
| 2 | Feng Liang | P2: Model fine-tuning & training | **33%** |
| 3 | Han Qiang | P3: Algorithm improvement & visualization | **33%** |

Total: **100%**

---

## Detailed Contributions

### 1. Yuhan Chen — P1: AttnLRP Base Reproduction & Visualization (33%)

**Module:** Part 1 — Base GPT-2 AttnLRP

**Responsibilities:**
- Set up the `lxt` AttnLRP framework with the efficient monkey-patch implementation for GPT-2, establishing the coding pipeline for the entire project.
- Implemented token-level relevance extraction (Input×Gradient) and layer-wise relevance capture via retained forward hooks on GPT-2 transformer blocks.
- Designed and implemented three visualization tools: token-level heatmaps, layer×token heatmaps, and lxt-style PDF heatmaps with token text inside colored boxes.
- Conducted the initial reproducibility verification on base GPT-2 using SQuAD v2 and SciQ prompts, and authored the corresponding Part 1 results section of the paper.
- Built the MoRF/LeRF faithfulness evaluation pipeline based on Blücher et al. [4], including the random-relevance baseline, and produced the faithfulness and comprehensiveness/sufficiency tables.
- Implemented data loading and prompt formatting for both SQuAD v2 and SciQ datasets, with max-length filtering and answer-target extraction.
- Authored the Introduction, Experimental Setup, and contributed to the faithfulness analysis of the technical report.

---

### 2. Feng Liang — P2: Model Fine-Tuning & Training (33%)

**Module:** Part 2 — Supervised Fine-Tuning

**Responsibilities:**
- Designed and executed the supervised fine-tuning protocol: GPT-2 base fine-tuned separately on SQuAD v2 (2 epochs, ~36,400 steps) and SciQ (4 epochs, ~2,200 steps) with causal LM loss masked to answer tokens only.
- Configured and tuned training hyperparameters (learning rate 5×10⁻⁵, weight decay 0.01, warm-up ratio 0.03, batch size 16, max length 512) shared across both datasets.
- Saved dense checkpoint-level metrics (loss, token accuracy) approximately every 200 steps, producing the training dynamics curves.
- Evaluated AttnLRP faithfulness periodically during fine-tuning, producing the faithfulness co-evolution plots showing MoRF/LeRF AUC and comprehensiveness/sufficiency gaps over training steps.
- Re-ran AttnLRP on the fine-tuned models over the same prompts used in Part 1, enabling the before/after token-level comparison.
- Authored the fine-tuning methodology, the fine-tuning dynamics analysis, and the token-level comparison sections of the technical report.

---

### 3. Han Qiang — P3: Algorithm Improvement & Visualization (33%)

**Module:** Part 3 — Contrastive Masking & Parameter-Level Attribution

**Responsibilities:**
- Identified and diagnosed the GPT-2 logit sign-flipping artifact in AttnLRP, tracing its root cause to GPT-2's predominantly negative logit distribution arising from tied embeddings and large-vocabulary suppression.
- Designed and implemented the contrastive gradient masking initialization (target=1, others=−1/N), providing a mathematical derivation showing its equivalence to explaining a normalized softmax target.
- Extended AttnLRP from activation attribution to parameter-level attribution (R_θ = θ ⊙ ∇̃_θ · y_t), and designed the aggregation scheme by layer, architectural component (attention/MLP/LayerNorm), and attention head.
- Implemented the full parameter-level analysis pipeline: per-sample module records, layer-component heatmaps with row-wise normalization, and head-level attribution maps.
- Produced the task-specific parameter comparison (SQuAD v2 → late attention; SciQ → late MLP) and the head-level case studies distinguishing numeric-answer circuits from unanswerable-query signatures.
- Built the cohort-averaging pipeline (50 samples per configuration) and the appendix case-study figures with paired numeric/unanswerable samples sharing the same prompts.
- Authored the sign-fix methodology, the parameter attribution methodology, the Part 3 results analysis, and the Appendix.

---

## Shared / Joint Work

The following were jointly handled by all three members and are not double-counted in the percentages above (their effort is folded into each member's allocation):

- **Repository organization:** directory structure, environment configuration (`requirements.txt`, conda environment), checkpoint management, top-level `README.md`.
- **Poster design and preparation:** one poster jointly designed with each member contributing content for their respective module; layout and visual design done collaboratively.
- **Technical report writing:** each member drafted the sections corresponding to their module; integration, cross-referencing, English revision, and figure preparation were performed jointly.
- **Cross-module integration:** aligning the AttnLRP output format between Part 1, Part 2, and Part 3; ensuring consistent data loading, prompt formatting, and evaluation protocols across all experiments; end-to-end reproducibility verification.

---

*Signed (electronically, by commit history):*
- Yuhan Chen
- Feng Liang
- Han Qiang
