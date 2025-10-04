#!/bin/bash
# Quickstart script for prompt distillation experiment
# This runs the complete pipeline: data generation + training

set -e  # Exit on error

echo "=========================================="
echo "Prompt Distillation Quickstart"
echo "=========================================="
echo ""

# Configuration
NUM_GPUS=${1:-2}  # Default to 2 GPUs
SAVE_PATH=${2:-./models/prompt_distillation}

echo "Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Save path: $SAVE_PATH"
echo ""

# Step 1: Generate training data
echo "=========================================="
echo "Step 1: Generating training data..."
echo "=========================================="
echo ""

if [ -f "./data/prompt_distillation_lang.jsonl" ]; then
    echo "Training data already exists. Skipping generation."
    echo "To regenerate, delete ./data/prompt_distillation_lang.jsonl"
else
    python create_data.py \
        --input_file ../tinker-cookbook/example-data/multilingual.txt \
        --output_file ./data/prompt_distillation_lang.jsonl \
        --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --temperature 0.15 \
        --tensor_parallel_size 2
    
    if [ $? -ne 0 ]; then
        echo "Error: Data generation failed!"
        exit 1
    fi
fi

echo ""
echo "Data generation complete!"
echo ""

# Step 2: Train the model
echo "=========================================="
echo "Step 2: Training the student model..."
echo "=========================================="
echo ""

bash train_sft.sh $NUM_GPUS $SAVE_PATH

if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Model saved to: $SAVE_PATH"
echo ""
echo "To evaluate your model, run:"
echo "  python evaluate.py --model_path $SAVE_PATH/checkpoint-final"
echo ""

