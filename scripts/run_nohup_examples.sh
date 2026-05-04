#!/usr/bin/env bash
set -euo pipefail

# Run from repo root: Explanations-of-LLM
ENV_PY="/home/Feng/.conda/envs/lxt311/bin/python"
MODEL_PATH="model/gpt2-model"

mkdir -p logs outputs

# 1) Smoke test fine-tuning
nohup "$ENV_PY" scripts/train_gpt2_qa.py \
  --dataset squad_v2 \
  --model_path "$MODEL_PATH" \
  --data_dir data \
  --output_dir outputs/modelA_squad_gpt2_smoke \
  --max_steps 200 \
  --max_train_samples 2000 \
  --max_eval_samples 500 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_length 512 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  > logs/squad_smoke.log 2>&1 &

nohup "$ENV_PY" scripts/train_gpt2_qa.py \
  --dataset sciq \
  --model_path "$MODEL_PATH" \
  --data_dir data \
  --output_dir outputs/modelB_sciq_gpt2_smoke \
  --max_steps 200 \
  --max_train_samples 2000 \
  --max_eval_samples 500 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_length 512 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  > logs/sciq_smoke.log 2>&1 &

# 2) Full fine-tuning
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/Feng/.conda/envs/lxt311/bin/python -u scripts/train_gpt2_qa.py \
  --dataset squad_v2 --model_path /home/Feng/code/LRP-eXplains-Transformers/model/gpt2-model \
  --data_dir /home/Feng/code/LRP-eXplains-Transformers/data \
  --output_dir outputs/modelA_squad_gpt2_full_faith_fix \
  --num_train_epochs 2.0 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_length 512 \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 20 \
  --save_total_limit 1 \
  --faithfulness_eval_samples 16 \
  --faithfulness_steps 20 \
  --faithfulness_sample_seed 42 > logs/squad_full_faith.log 2>&1 &

nohup /home/Feng/.conda/envs/lxt311/bin/python -u scripts/train_gpt2_qa.py \
 --dataset sciq --model_path /home/Feng/code/LRP-eXplains-Transformers/model/gpt2-model \
 --data_dir /home/Feng/code/LRP-eXplains-Transformers/data \
 --output_dir outputs/modelB_sciq_gpt2_full_faith \
 --num_train_epochs 4.0 --per_device_train_batch_size 16 --per_device_eval_batch_size 8 \
 --gradient_accumulation_steps 1 --max_length 512 --save_steps 100 --eval_steps 100 \
 --logging_steps 20 --save_total_limit 1 --faithfulness_eval_samples 50 --faithfulness_steps 20 --faithfulness_sample_seed 42 > logs/sciq_full_faith.log 2>&1 &

# 3) Loss plots
nohup "$ENV_PY" scripts/plot_trainer_loss.py \
  --output_dir outputs/modelA_squad_gpt2 \
  --title "SQuAD GPT-2 Fine-tuning Loss Curve" \
  > logs/squad_loss_plot.log 2>&1 &

nohup "$ENV_PY" scripts/plot_trainer_loss.py \
  --output_dir outputs/modelB_sciq_gpt2 \
  --title "SciQ GPT-2 Fine-tuning Loss Curve" \
  > logs/sciq_loss_plot.log 2>&1 &

# 4) Token-level base vs finetuned comparison
nohup "$ENV_PY" scripts/compare_attnlrp_gpt2.py \
  --base_model model/gpt2-model \
  --finetuned_model outputs/modelA_squad_gpt2/checkpoint-8145 \
  --squad_dev data/SQuAD/dev-v2.0.json \
  --out_dir outputs/attnlrp_compare_50cases \
  --num_cases 50 \
  --device cuda \
  > logs/attnlrp_compare_50cases.log 2>&1 &

# 5) Layer-wise base vs finetuned comparison
nohup "$ENV_PY" scripts/compare_attnlrp_gpt2_layerwise.py \
  --base_model model/gpt2-model \
  --finetuned_model outputs/modelA_squad_gpt2/checkpoint-8145 \
  --squad_dev data/SQuAD/dev-v2.0.json \
  --out_dir outputs/attnlrp_layerwise_compare_50cases \
  --num_cases 50 \
  --device cuda \
  > logs/attnlrp_layerwise_50cases.log 2>&1 &

echo "Launched jobs. Use: ps -ef | grep -E 'train_gpt2_qa|compare_attnlrp|plot_trainer_loss'"
