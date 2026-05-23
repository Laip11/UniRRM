#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REWARD_TYPE="${REWARD_TYPE:-genrm-judge}"
TEMPLATE_NAME="${TEMPLATE_NAME:-UniRRM}"
DATASETS="${DATASETS:-allenai/reward-bench-2}"

model_ls=(
SUSTech-NLP/UniRRM-8B
)

for model in "${model_ls[@]}"; do
  python ./evaluation_listwise.py \
    --model_name_or_path "${model}" \
    --datasets ${DATASETS} \
    --reward_type "${REWARD_TYPE}" \
    --template_name "${TEMPLATE_NAME}" 
done
