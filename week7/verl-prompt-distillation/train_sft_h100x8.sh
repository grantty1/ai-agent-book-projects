#!/bin/bash
# Optimized training script for H100x8 GPU setup
# This configuration fully utilizes 8x H100 GPUs with 80GB VRAM each

set -x

# Use default save path if not provided
save_path=${1:-"./models/prompt_distillation"}

# Shift only if an argument was provided
if [ "$#" -ge 1 ]; then
    shift 1
fi

echo "============================================"
echo "H100x8 Training - Prompt Distillation"
echo "============================================"
echo "Save path: $save_path"
echo "GPUs: 8 (H100)"
echo "Batch size: 128 (16 per GPU)"
echo "============================================"
echo ""

# H100x8 Optimized Configuration
# ================================
# Total GPUs: 8
# Per-GPU VRAM: 80GB
# Total Compute: Massive!

# Batch size calculations:
# - Global batch size: 128 (matching tinker)
# - micro_batch_size_per_gpu: 16 (increased from 8)
# - Gradient accumulation steps: 128 / (8 GPUs * 16) = 1 (no accumulation needed!)

nproc_per_node=8  # Use all 8 H100 GPUs

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/prompt_distillation_lang.parquet \
    data.val_files=./data/prompt_distillation_lang.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=16 \
    data.max_length=32768 \
    data.truncation=right \
    model.partial_pretrain=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=false \
    optim.lr=1e-4 \
    optim.lr_scheduler=linear \
    optim.warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=prompt-distillation \
    trainer.experiment_name=prompt-distillation-qwen3-4b-h100x8 \
    trainer.total_epochs=4 \
    trainer.logger=console \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    use_remove_padding=true $@

# H100x8 Optimizations Applied:
# ==============================
# 1. nproc_per_node=8: Use all 8 H100 GPUs
# 2. micro_batch_size_per_gpu=16: Doubled from 8 (H100 has 80GB VRAM)
# 3. gradient_checkpointing=false: Disabled (plenty of memory, faster training)
# 4. No gradient accumulation needed: 8 GPUs * 16 batch = 128 global batch
# 
# Expected Training Time: ~15-20 minutes (vs 30-60 min on 2 GPUs)
# Expected Memory Usage: ~10-12GB per GPU (plenty of headroom)

