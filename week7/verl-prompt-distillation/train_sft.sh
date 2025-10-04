#!/bin/bash
# Supervised Fine-Tuning script for prompt distillation using verl
# This reproduces the tinker cookbook prompt distillation experiment
#
# Hyperparameters match the tinker implementation:
# - Learning rate: 1e-4
# - LoRA rank: 32
# - LoRA alpha: 16
# - Batch size: 128
# - Max length: 32768
# - Num epochs: 4
# - LR schedule: linear

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: train_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    echo "Example: bash train_sft.sh 2 ./models/prompt_distillation"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Note: Using Qwen3-4B-Instruct-2507 as student (smaller, more efficient)
# Teacher model (Qwen3-30B-A3B-Instruct-2507) is used for data generation
# All other hyperparameters match tinker exactly

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/prompt_distillation_lang.jsonl \
    data.val_files=./data/prompt_distillation_lang.jsonl \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=32768 \
    data.truncation=right \
    model.partial_pretrain=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=true \
    optim.lr=1e-4 \
    optim.lr_scheduler=linear \
    optim.warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=prompt-distillation \
    trainer.experiment_name=prompt-distillation-qwen3-4b-32rank-1e-4lr-128batch \
    trainer.total_epochs=4 \
    trainer.logger=console \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    use_remove_padding=true $@

# Note on hyperparameters:
# - train_batch_size=128: Same as tinker
# - lr=1e-4: Same as tinker  
# - lora_rank=32: Same as tinker
# - lora_alpha=16: Same as tinker (tinker doesn't specify, using verl default)
# - max_length=32768: Same as tinker
# - total_epochs=4: Same as tinker
# - lr_scheduler=linear: Same as tinker
# - Teacher: Qwen3-30B-A3B-Instruct-2507 (matches tinker's Qwen3-30B-A3B)
# - Student: Qwen3-4B-Instruct-2507 (smaller model for efficient deployment)

