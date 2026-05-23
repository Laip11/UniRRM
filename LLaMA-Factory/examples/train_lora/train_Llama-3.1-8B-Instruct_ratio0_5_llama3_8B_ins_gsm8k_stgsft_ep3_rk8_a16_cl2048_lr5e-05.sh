#!/bin/bash
source /mnt/workspace/laip/miniconda3/etc/profile.d/conda.sh 
conda activate llama-factory
cd /mnt/workspace/laip/LLaMA-Factory

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH='/mnt/workspace/laip/model/meta/Llama-3.1-8B-Instruct'
OUTPUT_DIR='/mnt/workspace/laip/saves/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct_ratio0_5_llama3_8B_ins_gsm8k_stgsft_ep3_rk8_a16_cl2048_lr5e-05'

llamafactory-cli train \
    --model_name_or_path "${MODEL_PATH}" \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_target all \
    --dataset ratio0_5_llama3_8B_ins_gsm8k \
    --template llama3 \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_steps 10000 \
    --plot_loss \
    --overwrite_output_dir false \
    --save_only_model true \
    --report_to none \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --bf16 \
    --ddp_timeout 180000000
