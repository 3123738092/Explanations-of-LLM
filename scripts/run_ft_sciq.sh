#!/bin/bash
# SciQ fine-tuning on GPT-2 base
python train_gpt2_qa.py \
    --dataset sciq \
    --model_path /home/Feng/code/LRP-eXplains-Transformers/model/gpt2-model \
    --data_dir data \
    --output_dir outputs/modelB_sciq_ft \
    --max_length 512 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 50 \
    --eval_steps 50 \
    --faithfulness_eval_samples 8 \
    --faithfulness_steps 20
