#!/bin/bash

workdir=$(cd $(dirname $0); pwd)

res_dir=${workdir}/fsdp/$(date '+%m_%d_%H_%M_%S')
mkdir -p ${res_dir}
mkdir -p ${res_dir}/ckpt
mkdir -p ${res_dir}/tb
mkdir -p ${res_dir}/swan

node=4 #根据情况修改

set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export SWANLAB_MODE=local 
node_num=$node
per_device_bs=1
acc_steps=1
train_files=/***/train.parquet
test_files=/***/test.parquet
save_freq=100
test_freq=20
total_epochs=1
model_path=/***/Qwen2.5-Math-7B #实际的模型路径，根据情况修改
save_path=${res_dir}/ckpt #根据情况修改
#export TENSORBOARD_LOG_DIR=${11}
export TENSORBOARD_DIR=${res_dir}/tb  #根据情况修改
vllm_tp=4 #根据模型大小适当修改


echo $vllm_tp

ref_sync_steps=1

#num_worker=$(($node_num * 8))
num_worker=$(($node_num * 8 * 4))

ppo_mini_bs=$(($per_device_bs * $num_worker * $acc_steps))
interact_bs=$(($ppo_mini_bs))

max_prompt_length=1024
max_response_length=15360
max_num_batched_tokens=$(($max_prompt_length+$max_response_length))


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$interact_bs \
    data.val_batch_size=$interact_bs \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    data.shuffle=False \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=3.0e-06 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_bs \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$per_device_bs \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$per_device_bs \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$vllm_tp \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$per_device_bs \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.7 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_math_lighteval' \
    trainer.experiment_name='qwen2_5_7b_math_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$node_num \
    trainer.default_local_dir="$save_path" \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=False \
    trainer.balance_batch=False \