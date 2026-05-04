# Faithfulness Evaluation Standard (Main-Q Aligned)

## Canonical Rule
- From now on, faithfulness metrics must follow `main_q.py` protocol.
- The canonical implementation entry is:
  - `src/evaluate/main_q_metrics.py`
  - `src/evaluate/faithfulness.py`

## Required Metric Protocol
1. Prompt-only input.
- Evaluate on model input prompt only.
- Do not append gold/reference answer to the prompt during evaluation.

2. Target token.
- Use final position next-token argmax (`target_id = argmax(logits[-1])`).

3. Relevance extraction.
- Use contrastive seed on final logits:
  - initialize vector as `-1/V`
  - set target position to `+1`
- Token relevance is `input_embeds * grad` summed over hidden dimension.

4. Faithfulness curves.
- Use `faithfulness_score(..., strategy="morf")` and `strategy="lerf"` from `src/evaluate/faithfulness.py`.
- Random baseline must use same input/target with random relevance vector from a fixed seed.

5. Comparison criterion.
- Better-than-random is defined as:
  - `auc_morf < auc_random_morf`
  - `auc_lerf > auc_random_lerf`

## Synced Scripts
The following scripts were synced to use the shared main-q metrics implementation:
- `scripts/eval_faithfulness_teammate_style.py`
- `scripts/eval_outputs_on_readme50.py`
- `scripts/eval_outputs_on_sciq_fixed.py`
- `scripts/eval_faithfulness_all_models.py`

## Common Pitfalls And Misunderstandings
1. Data/prompt mismatch.
- Comparing SciQ results with SQuAD results directly can be misleading.
- Prompt templates (`Context/Question/Answer` vs other orderings) can shift metrics a lot.

2. Gold answer leakage.
- Using `prompt + reference answer` during faithfulness evaluation usually inflates/changes behavior and breaks comparability.

3. Tokenizer mismatch.
- Some checkpoints do not store tokenizer files.
- Always provide a compatible fallback tokenizer path (for GPT-2 checkpoints, base GPT-2 tokenizer).

4. Different target definitions.
- Using argmax target vs fixed label token is a different task.
- Report target definition explicitly.

5. Different random seeds/sample subsets.
- AUC values can vary notably with small `num_samples`.
- Always record `num_samples`, `random_seed`, and dataset split.

6. Environment/version drift.
- Some efficient LXT paths depend on `transformers` versions with extra model modules.
- If that path cannot run, use the standard GPT-2 AttnLRP path but keep metric protocol identical.

## Minimal Reproducibility Checklist
- model path
- tokenizer source path
- dataset + split
- prompt template
- num_samples
- max_length
- steps
- random_seed
- device
- script path and git commit hash
