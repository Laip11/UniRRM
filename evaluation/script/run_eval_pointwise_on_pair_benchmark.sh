#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REWARD_TYPE="${REWARD_TYPE:-genrm-scorer}"
TEMPLATE_NAME="${TEMPLATE_NAME:-UniRRM}"
TEMPERATURE="${TEMPERATURE:-0}"
DATASETS="${DATASETS:-judgebench_pro}"

model_ls=(
SUSTech-NLP/UniRRM-8B
)


for model in "${model_ls[@]}"; do
  python3 evaluation_pointwise_on_pair_benchmark.py \
    --model_name_or_path "${model}" \
    --datasets ${DATASETS} \
    --temperature "${TEMPERATURE}" \
    --reward_type "${REWARD_TYPE}" \
    --template_name "${TEMPLATE_NAME}" 
done
