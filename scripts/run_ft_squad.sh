#!/bin/bash
# SQuAD_v2 fine-tuning on GPT-2 base
python train_gpt2_qa.py \
    --dataset squad_v2 \
    --model_path /home/Feng/code/LRP-eXplains-Transformers/model/gpt2-model \
    --data_dir data \
    --output_dir outputs/modelA_squad_ft \
    --max_length 512 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --eval_steps 100 \
    --faithfulness_eval_samples 8 \
    --faithfulness_steps 20
