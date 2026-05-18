# Running the Experiments

Run every command from the repository root:

```bash
cd /root/autodl-tmp/Explanations-of-LLM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If Hugging Face or GitHub downloads are slow on the server, enable the academic network helper before running commands:

```bash
source /etc/network_turbo
```

## Part 1: Base GPT-2 AttnLRP

Part 1 runs AttnLRP on the base GPT-2 model with SQuAD v2 prompts. It saves token relevance, layer-token relevance, and MoRF/LeRF faithfulness results.

```bash
python main.py --config configs/gpt2_efficient.yaml
```

Main outputs:

```text
outputs-gpt2-efficient/faithfulness_summary.json
outputs-gpt2-efficient/relevance/
outputs-gpt2-efficient/figures/tokens/
outputs-gpt2-efficient/figures/layers/
```

## Part 2: Fine-Tuning GPT-2

The fine-tuning entry point is:

```bash
python scripts/train_gpt2_qa.py --help
```

Expected local data files:

```text
data/SQuAD/train-v2.0.json
data/SQuAD/dev-v2.0.json
data/sciq/train-00000-of-00001.parquet
data/sciq/validation-00000-of-00001.parquet
```

SQuAD fine-tuning:

```bash
python scripts/train_gpt2_qa.py \
  --dataset squad_v2 \
  --model_path /path/to/base_or_previous_gpt2_checkpoint \
  --data_dir data \
  --output_dir FT-model/modelA_squad \
  --max_steps 1600 \
  --max_length 512 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --logging_steps 20 \
  --eval_steps 200 \
  --save_steps 200
```

SciQ fine-tuning:

```bash
python scripts/train_gpt2_qa.py \
  --dataset sciq \
  --model_path /path/to/base_or_previous_gpt2_checkpoint \
  --data_dir data \
  --output_dir FT-model/modelB_sciq \
  --max_steps 1300 \
  --max_length 512 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --logging_steps 20 \
  --eval_steps 100 \
  --save_steps 100
```

Useful options:

```text
--max_train_samples N          limit training examples for debugging
--max_eval_samples N           limit evaluation examples for debugging
--resume_from_checkpoint PATH  continue from a saved checkpoint
--disable_faithfulness_eval    skip faithfulness evaluation during training
```

## Part 3: Parameter Attribution

Part 3 uses `main.py` with `parameter_attribution.enabled: true` in the config. The two provided fine-tuned configs already enable it.

SQuAD fine-tuned checkpoint:

```bash
python main.py --config configs/gpt2_efficient_finetuned_squad.yaml
```

SciQ fine-tuned checkpoint:

```bash
python main.py --config configs/gpt2_efficient_finetuned_sciq.yaml
```

Parameter-attribution outputs:

```text
<output.dir>/parameter_relevance/
<figures_dir>/parameter_heads/
<figures_dir>/parameter_layers_line/
<figures_dir>/parameter_layers_block/
```

Keep `parameter_attribution.save_tensors: false` unless full parameter tensors are explicitly needed, because full tensors are large.

## Config Notes

Each experiment is controlled by a YAML file under `configs/`.

Important fields:

```yaml
model:
  family: gpt2_efficient
  name: gpt2                         # Hugging Face model id or local checkpoint path
  tokenizer_name: gpt2               # optional; useful when a checkpoint has no tokenizer files
  device: cuda
  dtype: bfloat16

data:
  dataset: squad_v2                  # squad_v2 or sciq
  split: validation
  num_samples: 50
  max_length: 512

parameter_attribution:
  enabled: true
  save_tensors: false
  top_modules: 20

faithfulness:
  steps: 20
```

For SciQ, `sciq_shuffle_seed` should match the seed used during fine-tuning so that answer options are ordered consistently.
