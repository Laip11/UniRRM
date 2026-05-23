# Discliamer: the model used in the script is only for academic purpose.
#!/bin/bash
#!/bin/bash
#!/bin/bash
CONDA_INSTALL_PATH="/mnt/workspace/laip/miniconda3"
ENV_IDENTIFIER="$CONDA_INSTALL_PATH/envs/verl"
set -e 
set +x

# 清理 Python 环境
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
unset PYTHONOPTIMIZE
unset PYTHONNOUSERSITE
unset PYTHONDEBUG
unset PYTHONDONTWRITEBYTECODE
unset PYTHONINSPECT
unset PYTHONIOENCODING
unset PYTHONFAULTHANDLER
unset PYTHONHASHSEED
unset PYTHONMALLOC
unset PYTHONPROFILEIMPORTTIME
unset PYTHONUSERSITE
unset PYTHONWARNINGS
unset PYTHONCASEOK
unset PYTHONDUMPREFS
unset PYTHONTHREADDEBUG
unset PYTHONVERBOSE
unset PYTHONWARNDEFAULTENCODING

source "$CONDA_INSTALL_PATH/etc/profile.d/conda.sh"
conda activate "$ENV_IDENTIFIER"

# =======================================================
# [核心修复] 设置分布式环境变量的默认值，防止 unbound variable 报错
# =======================================================
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# =======================================================

set -x

echo "当前使用的 Python 路径: $(which python)"
echo "Current Rank: $RANK, World Size: $WORLD_SIZE, Master Addr: $MASTER_ADDR"

############# 一些系统配置 #############
set -euxo pipefail # 注意：这里开启了 -u，所以上面必须先定义好 RANK

echo "pip下载源:"
pip config list

############# 相关路径配置 #############
NAS_ROOT="/mnt/workspace" 

HOME=/mnt/workspace/laip
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"


# prepare model ckpt
huggingface-cli download sfairXC/FsfairX-LLaMA3-RM-v0.1 --local-dir /mnt/workspace/laip/models/FsfairX-LLaMA3-RM-v0.1 &
wait

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="/mnt/workspace/laip/model/Qwen/Qwen3-1.7B" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="/mnt/workspace/laip/model/Qwen/Qwen3-1.7B" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.model.path="$HOME/models/FsfairX-LLaMA3-RM-v0.1" \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_example' \
    trainer.val_before_train=False \
    trainer.experiment_name='Qwen3-1.7B_hybrid_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=2 \
    trainer.total_epochs=15 $@
