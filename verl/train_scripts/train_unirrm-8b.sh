#!/bin/bash
set -e 
set +x

REWARD_PID=""

nohup python ./reward_part/reward_server.py --port 39853 > ./reward_log.txt 2>&1 &
REWARD_PID=$!
echo "reward_server started with PID $REWARD_PID"

export WANDB_API_KEY=your_wandb_api_key
wandb login

EXPERIMENT_NAME="UniRRM-8B"
PROJECT_NAME="UniRRM"
SFT_Model=''

echo "Starting training for project: ${PROJECT_NAME}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "=================================================="

train_files=train_files=./data/UniRRM-RL/train.parquet
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$train_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=3072 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$SFT_Model" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.max_num_batched_tokens=15000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.total_epochs=2 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.test_freq=-1 \
    trainer.save_freq=1000 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.logger="[console,wandb]" \
    trainer.default_local_dir='./saves/UniRRM-8B' \
    reward_model.launch_reward_fn_async=True \
    reward_model.reward_manager=naive \
    reward_model.reward_api=http://127.0.0.1:39853/latest_with_rubric_reward_qwen3max \
    $@

if ps -p $REWARD_PID > /dev/null; then
    echo "Stopping reward_server..."
    kill -9 $REWARD_PID
    wait $REWARD_PID 2>/dev/null || true
fi

echo "All done!"

