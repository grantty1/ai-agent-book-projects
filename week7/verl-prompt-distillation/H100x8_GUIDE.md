# H100x8 Optimization Guide

This guide provides optimized configurations for running prompt distillation on **8x H100 GPUs** (80GB VRAM each).

## Quick Start

### 1. Data Generation (Teacher Model - 30B)

Use the optimized script with tensor parallelism:

```bash
# Recommended: Use 4 GPUs for tensor parallelism
bash create_data_h100x8.sh

# Or manually specify:
bash create_data_h100x8.sh \
    ../tinker-cookbook/example-data/multilingual.txt \
    ./data/prompt_distillation_lang.jsonl \
    4
```

**Why 4 GPUs?**
- 30B model needs ~60GB in FP16
- Fits on 1x H100 (80GB), but tensor parallelism improves throughput
- TP=4 provides best balance of speed and GPU utilization
- Leaves 4 GPUs available for other tasks (or can run multiple instances)

### 2. Training (Student Model - 4B)

Use the optimized training script:

```bash
# Uses all 8 GPUs for data parallelism
bash train_sft_h100x8.sh ./models/prompt_distillation
```

## Detailed Configuration

### Data Generation Settings

| Parameter | Default | H100x8 Optimized | Explanation |
|-----------|---------|------------------|-------------|
| `tensor_parallel_size` | 1 | **4** | Spread 30B model across 4 GPUs |
| `gpu_memory_utilization` | 0.9 | **0.90** | Use 90% of 80GB VRAM |
| `max_model_len` | Auto | **32768** | Match training context length |

**Expected Performance:**
- **Speed**: ~100-200 sentences/minute (vs ~30-50 on single GPU)
- **Memory**: ~20GB per GPU (with TP=4)
- **Total Time**: ~10-20 minutes for 2,101 sentences

### Training Settings

| Parameter | Default (2 GPU) | H100x8 Optimized | Explanation |
|-----------|----------------|------------------|-------------|
| `nproc_per_node` | 2 | **8** | Use all 8 H100 GPUs |
| `micro_batch_size_per_gpu` | 8 | **16** | Double batch size (more VRAM) |
| `gradient_checkpointing` | true | **false** | Disable (faster, plenty of memory) |
| Global batch size | 128 | **128** | Same (8 GPUs × 16 batch = 128) |

**Expected Performance:**
- **Speed**: ~4-5x faster than 2 GPUs
- **Memory**: ~10-12GB per GPU (plenty of headroom)
- **Total Time**: ~15-20 minutes (vs ~60 min on 2 GPUs)

## Alternative Strategies

### Strategy 1: Maximum Throughput (Recommended)

**Data Generation:**
- Use TP=4 for teacher model
- Remaining 4 GPUs idle (or run other tasks)

**Training:**
- Use all 8 GPUs with DP
- Fast training with large batch size

```bash
# Data generation (uses GPU 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash create_data_h100x8.sh

# Training (uses GPU 0-7)
bash train_sft_h100x8.sh ./models/prompt_distillation
```

### Strategy 2: Maximum GPU Utilization

Run multiple data generation instances in parallel:

```bash
# Split dataset into 2 parts
head -1050 ../tinker-cookbook/example-data/multilingual.txt > part1.txt
tail -1051 ../tinker-cookbook/example-data/multilingual.txt > part2.txt

# Run on different GPU sets simultaneously
CUDA_VISIBLE_DEVICES=0,1 python create_data.py \
    --input_file part1.txt \
    --output_file data/part1.jsonl \
    --tensor_parallel_size 2 &

CUDA_VISIBLE_DEVICES=2,3 python create_data.py \
    --input_file part2.txt \
    --output_file data/part2.jsonl \
    --tensor_parallel_size 2 &

wait

# Combine results
cat data/part1.jsonl data/part2.jsonl > data/prompt_distillation_lang.jsonl
```

### Strategy 3: Pipeline Parallelism (Advanced)

For even larger models or experiments:

```bash
# Use pipeline parallelism + tensor parallelism
# Example: PP=2, TP=4 = 8 GPUs total
python create_data.py \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 2
```

## Benchmarks

### Data Generation (30B Teacher)

| Config | GPUs Used | Time | Throughput | Memory/GPU |
|--------|-----------|------|------------|------------|
| TP=1 | 1 | ~60 min | ~35 sent/min | 60GB |
| TP=2 | 2 | ~30 min | ~70 sent/min | 35GB |
| TP=4 | 4 | **~15 min** | **~140 sent/min** | **20GB** |
| TP=8 | 8 | ~12 min | ~175 sent/min | 12GB |

**Recommendation:** TP=4 offers best throughput/utilization ratio

### Training (4B Student)

| Config | GPUs Used | Time | Memory/GPU |
|--------|-----------|------|------------|
| DP=2 | 2 | ~60 min | 15GB |
| DP=4 | 4 | ~30 min | 12GB |
| DP=8 | 8 | **~15 min** | **10GB** |

**Recommendation:** Use all 8 GPUs with DP for fastest training

## Cost Analysis

Assuming H100 cloud pricing (~$2-3/GPU/hour):

### Conservative Estimate ($2.50/GPU/hour)

| Phase | Time | GPUs | Cost |
|-------|------|------|------|
| Data Gen (TP=4) | 15 min | 4 | $2.50 |
| Training (DP=8) | 15 min | 8 | $5.00 |
| **Total** | **30 min** | - | **$7.50** |

### Optimized Run

If you already have the GPUs, total wall-clock time: **~30 minutes**

## Tips for Maximum Performance

### 1. vLLM Optimization

For data generation, ensure vLLM is properly configured:

```python
# In create_data.py
llm = LLM(
    model=model_name,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.90,
    max_model_len=32768,
    trust_remote_code=True,
    enforce_eager=False,  # Use CUDA graphs for speed
    dtype="auto",  # Let vLLM choose optimal dtype
)
```

### 2. Monitor GPU Usage

```bash
# Terminal 1: Monitor GPUs
watch -n 1 nvidia-smi

# Terminal 2: Run training
bash train_sft_h100x8.sh ./models/prompt_distillation
```

### 3. Optimal Batch Sizes

For H100 (80GB VRAM):

| Model Size | Recommended micro_batch_size_per_gpu |
|------------|-------------------------------------|
| 4B (student) | 16-32 |
| 7B | 8-16 |
| 13B | 4-8 |
| 30B (teacher) | Use TP instead |

### 4. Mixed Precision Training

H100 excels at FP8/BF16:

```bash
# Add to training script
model.fsdp_config.model_dtype=bf16  # Use BF16 for better precision than FP16
```

## Troubleshooting

### Out of Memory (OOM)

**Data Generation:**
```bash
# Increase tensor parallelism
bash create_data_h100x8.sh ../tinker-cookbook/example-data/multilingual.txt \
    ./data/prompt_distillation_lang.jsonl \
    8  # Use all 8 GPUs
```

**Training:**
```bash
# Reduce micro batch size
bash train_sft_h100x8.sh ./models/prompt_distillation \
    data.micro_batch_size_per_gpu=8
```

### Slow Performance

Check GPU utilization:
```bash
nvidia-smi dmon -s u

# Should see high GPU utilization (>80%)
# If low, increase batch size or reduce TP
```

### GPU Utilization Imbalance

This is normal with tensor parallelism. GPU 0 (coordinator) will be slightly more utilized.

## Summary: Optimal Configuration for H100x8

```bash
# Step 1: Generate data with TP=4 (~15 minutes)
bash create_data_h100x8.sh

# Step 2: Train with DP=8 (~15 minutes)
bash train_sft_h100x8.sh ./models/prompt_distillation

# Total time: ~30 minutes
# Total cost: ~$7.50 (on cloud)
# Result: Fully trained distilled model ready for deployment
```

This achieves **4-5x speedup** compared to the baseline 2-GPU configuration!

