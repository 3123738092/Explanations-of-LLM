# Minimal Core Files (6-8)

This project is now standardized around `main_q.py` metric protocol.

## Keep These 8 Core Files
1. `main_q.py`
- Canonical experiment entrypoint (data loading, explanation, faithfulness summary).

2. `scripts/train_gpt2_qa.py`
- GPT-2 fine-tuning on SQuAD/SciQ; eval metrics aligned to main-q faithfulness logic.

3. `src/evaluate/main_q_metrics.py`
- Shared, canonical metric pipeline (argmax target + contrastive seed + random baseline).

4. `src/evaluate/faithfulness.py`
- Core MoRF/LeRF curve scoring function.

5. `src/explain/attnlrp.py`
- Standard AttnLRP explainer for GPT-2.

6. `src/data/squad_loader.py`
- SQuAD prompt/data loader.

7. `src/data/sciq_loader.py`
- SciQ prompt/data loader.

8. `src/models/gpt2_wrapper.py`
- GPT-2 model/tokenizer loading (with tokenizer fallback support).

## Compatibility File
- `main.py` now just forwards to `main_q.py` for backward-compatible commands.

## Dropped Legacy Evaluation Scripts
The following scripts were removed to avoid metric drift and confusion:
- `scripts/eval_outputs_on_readme50.py`
- `scripts/eval_outputs_on_sciq_fixed.py`
- `scripts/eval_faithfulness_all_models.py`

Use `main_q.py` or `scripts/eval_faithfulness_teammate_style.py` instead.
