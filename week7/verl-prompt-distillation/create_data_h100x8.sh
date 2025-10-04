#!/bin/bash
# Optimized data generation script for H100x8 GPU setup
# This configuration maximizes throughput for the 30B teacher model

set -e

echo "=========================================="
echo "Data Generation for H100x8 Setup"
echo "=========================================="
echo ""

# H100x8 Optimized Configuration for 30B Teacher Model
# ======================================================
# - Teacher: Qwen3-30B-A3B-Instruct-2507 (~60GB in FP16)
# - Strategy: Use tensor parallelism across 2-4 GPUs
# - Remaining GPUs: Can be used for higher batch processing

# Configuration
INPUT_FILE=${1:-"./example-data/multilingual.txt"}
OUTPUT_FILE=${2:-"./data/prompt_distillation_lang.jsonl"}
TENSOR_PARALLEL_SIZE=${3:-4}  # Use 4 GPUs for tensor parallelism (recommended for H100)

echo "Configuration:"
echo "  Input file: $INPUT_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE GPUs"
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    echo "⚠️  Output file already exists: $OUTPUT_FILE"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm "$OUTPUT_FILE"
fi

# Run data generation with optimized settings
python create_data.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --temperature 0.15 \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --max_tokens 1000

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Data generation complete!"
    echo "=========================================="
    echo "Output: $OUTPUT_FILE"
    echo ""
else
    echo ""
    echo "❌ Data generation failed!"
    exit 1
fi

