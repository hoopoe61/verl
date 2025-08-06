#!/usr/bin/env bash
set -xeuo pipefail

project_name='DPO'
exp_name='DPO-$$$3-4B'


TRAIN_FILE=/code/***/SFT/train-code/******_dpo/tmp_train.jsonl
TEST_FILE=/code/***/SFT/DeepSpeed-Chat/SFT_train_v1/DPO_data/dpo-resv10.9-32w_pair-1117_test.jsonl
max_length=null #一个sample的promt + chosen(promt + rejected)最大长度，不设置按照flush_left处理
max_prompt_length=512 #promt部分的最大长度，默认值是512
shuffle=True
seed=123
truncation="left"


# Paths
#MODEL_PATH="/code/******/verl/compare_dapo/Qwen3-4B"
actor_model_path="/code/******/verl/compare_grpo/$$$3b-thinking-0520"
ref_model_path="/code/***/SFT/train-code/******_dpo/dpo/0805/iter_0020500_hf"


# resource
NNODES=1 #这是节点的数量
n_gpus_per_node=8


train_bsz=8 #一个step对应的global batch size大小;
sp_size=4
per_actor_device_bsz=1 #一个dp对应的actor训练micro batch size大小，这个值可以根据显存情况设置，设置的越小显存占用越小，设置的过大会因为： dp * per_actor_device_bsz > train_bsz报错;
per_ref_device_bsz=1  #一个dp对应的ref前向micro batch size大小， 这个值可以根据显存情况设置，设置的越小显存占用越小，设置的过大会因为： dp * per_ref_device_bsz > train_bsz报错；ref模型大可以设置的小一些;


lr=2e-7
warmup_style="cosine"
beta=0.01
loss_type="hinge"
weight_decay=0.1
grad_clip=1.0


# Performance Related Parameter
use_dynamic_bsz=False #没有验证过True的场景，所以设置为False
use_fused_kernels=False #fused情况下无法对logp的计算定制，所以设置为False
offload=True #可以设置为False：会更快，但占用显存更多

save_freq=100
test_freq=100000000 #暂时还没有支持validate的功能，设置一个非常大的值；

workdir=$(cd $(dirname $0); pwd)
res_dir=${workdir}/run_record/$(date '+%m_%d_%H_%M_%S')
mkdir -p ${res_dir}
mkdir -p ${res_dir}/ckpt
mkdir -p ${res_dir}/tb
mkdir -p ${res_dir}/swan

CKPTS_DIR=${res_dir}/ckpt
export TENSORBOARD_DIR=${res_dir}/tb #设置tensorboard的地址


#默认加载：trainer/config/dpo_trainer.yaml的参数，优先级低于下面设置的参数；
python3 -m verl.trainer.main_dpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.truncation="$truncation" \
    data.shuffle=$shuffle \
    data.seed=$seed \
    data.train_batch_size=${train_bsz} \
    data.val_batch_size=${train_bsz} \
    data.max_length=${max_length} \
    data.max_prompt_length=${max_prompt_length} \
    actor_ref.model.use_remove_padding=True \
    actor_ref.model.use_fused_kernels=${use_fused_kernels} \
    actor_ref.model.enable_gradient_checkpointing=False \
    actor_ref.actor.path="$actor_model_path" \
    actor_ref.actor.beta=$beta \
    actor_ref.actor.loss_type="$loss_type" \
    actor_ref.actor.optim.lr=$lr \
    actor_ref.actor.optim.warmup_style="$warmup_style" \
    actor_ref.actor.optim.weight_decay=$weight_decay \
    actor_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_ref.actor.ppo_mini_batch_size=${train_bsz} \
    actor_ref.actor.ppo_micro_batch_size_per_gpu=$per_actor_device_bsz \
    actor_ref.actor.fsdp_config.param_offload=${offload} \
    actor_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_ref.actor.grad_clip=$grad_clip \
    actor_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_ref.ref.path="$ref_model_path" \
    actor_ref.ref.log_prob_micro_batch_size_per_gpu=$per_ref_device_bsz \
    actor_ref.ref.fsdp_config.param_offload=${offload} \
    actor_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=$test_freq \
    trainer.save_freq=$save_freq \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.balance_batch=False \