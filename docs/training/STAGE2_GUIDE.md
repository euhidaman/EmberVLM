# Stage 2: Multimodal Instruction Tuning

**Complete Training Guide for Instruction Following with Knowledge Distillation**

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Knowledge Distillation](#knowledge-distillation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## 🎯 Overview

Stage 2 builds upon the visual-language alignment from Stage 1 by teaching the model to follow multimodal instructions. This stage uses **knowledge distillation** from a larger teacher model (Qwen-VL-Chat) to transfer task-following capabilities efficiently.

### Objectives

1. **Instruction Following**: Teach the model to understand and execute multimodal instructions
2. **Knowledge Transfer**: Distill capabilities from Qwen-VL-Chat (7B) to EmberVLM (37M)
3. **Task Generalization**: Enable diverse task understanding (VQA, reasoning, etc.)
4. **Quality Improvement**: Improve response quality through teacher supervision

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| VQA Accuracy | >60% | Visual question answering accuracy |
| Instruction Following Score | >0.75 | Task completion quality |
| Distillation Loss | <1.0 | KL divergence from teacher |
| Response Coherence | >0.80 | Generated response quality |

## 🏗️ Architecture

### Training Framework

```
┌──────────────────────────────────────────────────────┐
│                   Stage 2 Training                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Teacher Model (Frozen)          Student Model       │
│  ┌───────────────────┐          ┌────────────────┐  │
│  │   Qwen-VL-Chat    │          │   EmberVLM     │  │
│  │      (7B)         │          │    (37M)       │  │
│  └─────────┬─────────┘          └────────┬───────┘  │
│            │                             │          │
│            │ Soft Targets                │          │
│            │ (Logits)                    │          │
│            └──────────────┬──────────────┘          │
│                           │                          │
│                           ▼                          │
│                  ┌─────────────────┐                 │
│                  │ Distillation    │                 │
│                  │ Loss (KL Div)   │                 │
│                  └─────────────────┘                 │
│                                                      │
│  Combined Loss:                                      │
│  L_total = α * L_distill + (1-α) * L_sft             │
│  where α = 0.3 (distillation), 1-α = 0.7 (SFT)      │
└──────────────────────────────────────────────────────┘
```

### Model Components

- **Vision Encoder**: RepViT (from Stage 1, partially frozen)
- **Projection Layer**: Trainable
- **Language Model**: TinyLLM (fully trainable)
- **Total Trainable**: ~23M parameters

### Loss Functions

```python
# Supervised Fine-Tuning Loss
L_sft = CrossEntropy(student_logits, ground_truth)

# Knowledge Distillation Loss
L_distill = KL_Divergence(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T²

# Combined Loss
L_total = 0.7 * L_sft + 0.3 * L_distill

# Temperature T = 2.0 (default)
```

## 🎓 Knowledge Distillation

### Why Distillation?

Knowledge distillation allows EmberVLM to learn from a much larger teacher model:

| Model | Parameters | Advantages |
|-------|------------|------------|
| **Teacher**: Qwen-VL-Chat | 7B | Strong reasoning, high quality |
| **Student**: EmberVLM | 37M | Fast inference, low memory |

### Distillation Process

1. **Teacher Inference**: Generate soft targets from Qwen-VL-Chat
2. **Temperature Softening**: Scale logits by temperature T=2.0
3. **KL Divergence**: Match student distribution to teacher
4. **Ground Truth**: Also learn from hard labels (SFT)

### Temperature Scaling

```python
def temperature_softmax(logits, temperature=2.0):
    """
    Higher temperature = softer probability distribution
    More information in "dark knowledge"
    """
    return F.softmax(logits / temperature, dim=-1)

# Example:
# T=1.0: [0.7, 0.2, 0.1] (sharp, less info)
# T=2.0: [0.5, 0.3, 0.2] (soft, more info)
```

### Distillation Benefits

- **Faster Convergence**: Learn from teacher's insights
- **Better Generalization**: Transfer knowledge beyond labels
- **Improved Quality**: Inherit teacher's response patterns
- **Efficient Training**: Fewer samples needed vs. training from scratch

## 📊 Datasets

### Training Datasets

#### 1. LLaVA-Instruct-150K

**Description**: High-quality instruction-following data generated by GPT-4

```json
{
  "image": "image_123.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat is unusual about this image?"
    },
    {
      "from": "gpt",
      "value": "The image shows a person water skiing while being pulled by a horse, which is highly unusual. Typically, water skiing involves being pulled by a motorboat, not a horse on land near the water."
    }
  ]
}
```

**Statistics**:
- Samples: 150,000
- Avg turns: 2.3 per conversation
- Avg response length: 45 tokens
- Domains: General vision, reasoning, detailed descriptions

#### 2. VQA-v2 (100K subset)

**Description**: Visual question answering with diverse questions

```json
{
  "image": "COCO_val2014_000000123456.jpg",
  "question": "What color is the umbrella?",
  "answer": "red",
  "answer_type": "color",
  "multiple_choice": ["red", "blue", "yellow", "green"]
}
```

**Statistics**:
- Samples: 100,000
- Question types: Yes/No (40%), Number (10%), Other (50%)
- Avg question length: 6.2 words
- Avg answer length: 1.5 words

#### 3. OK-VQA (50K subset)

**Description**: VQA requiring outside knowledge

```json
{
  "image": "image_456.jpg",
  "question": "What is this architectural style called?",
  "answer": "Gothic",
  "rationale": "The pointed arches and flying buttresses are characteristic of Gothic architecture."
}
```

**Statistics**:
- Samples: 50,000
- Knowledge domains: Science, history, culture, geography
- Avg question length: 7.8 words
- Requires reasoning: 85% of samples

### Dataset Statistics Summary

| Dataset | Samples | Avg Q Length | Avg A Length | Complexity |
|---------|---------|--------------|--------------|------------|
| LLaVA-Instruct | 150K | 15.2 words | 45 words | High |
| VQA-v2 | 100K | 6.2 words | 1.5 words | Medium |
| OK-VQA | 50K | 7.8 words | 2.3 words | High |
| **Total** | **300K** | **10.4 words** | **16.3 words** | **Mixed** |

### Data Format

All datasets are converted to a unified format:

```python
{
    "image": "path/to/image.jpg",
    "instruction": "What is in this image?",
    "response": "The image shows...",
    "teacher_logits": [...],  # Optional: from teacher model
    "metadata": {
        "source": "llava_instruct",
        "difficulty": "medium"
    }
}
```

## 🚀 Training

### Training Configuration

From `configs/training_stages.yaml`:

```yaml
stage2:
  name: "instruction_tuning"
  epochs: 5
  batch_size: 64
  learning_rate: 2.0e-4
  
  distillation:
    enabled: true
    teacher_model: "qwen-vl-chat"
    temperature: 2.0
    alpha: 0.3  # Distillation weight
    sft_weight: 0.7  # Supervised fine-tuning weight
  
  max_length: 512
```

### Training Commands

#### Basic Training

```bash
# With distillation (recommended)
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 2 \
    --resume_from outputs/stage1/checkpoint-final \
    --output_dir outputs \
    --wandb_project embervlm

# Without distillation (faster, lower quality)
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 2 \
    --resume_from outputs/stage1/checkpoint-final \
    --output_dir outputs \
    --disable_distillation
```

#### Advanced Options

```bash
torchrun --nproc_per_node=4 scripts/train_all.py \
    --stages 2 \
    --resume_from outputs/stage1/checkpoint-final \
    --output_dir outputs \
    --batch_size 64 \
    --learning_rate 2.0e-4 \
    --distillation_alpha 0.3 \
    --distillation_temperature 2.0 \
    --teacher_model qwen-vl-chat \
    --gradient_accumulation_steps 2 \
    --warmup_steps 1000 \
    --save_steps 500 \
    --eval_steps 250 \
    --fp16 \
    --wandb_project embervlm \
    --wandb_run_name stage2_instruct_tuning
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 64 | Per-device batch size |
| `--learning_rate` | 2.0e-4 | Peak learning rate |
| `--distillation_alpha` | 0.3 | Weight for distillation loss |
| `--distillation_temperature` | 2.0 | Temperature for soft targets |
| `--warmup_steps` | 1000 | Linear warmup steps |
| `--max_length` | 512 | Maximum sequence length |
| `--gradient_clip` | 1.0 | Gradient clipping norm |

### Teacher Model Setup

```bash
# Download teacher model (Qwen-VL-Chat)
python scripts/download_teacher.py --model qwen-vl-chat

# Or use HuggingFace cache
export HF_HOME=/path/to/cache
```

### Expected Training Time

| Hardware | Batch Size | With Distillation | Without Distillation |
|----------|------------|-------------------|----------------------|
| 1x V100 | 32 | ~8 hours/epoch | ~5 hours/epoch |
| 2x V100 | 64 | ~5 hours/epoch | ~3 hours/epoch |
| 4x A100 | 128 | ~2 hours/epoch | ~1.5 hours/epoch |
| 8x A100 | 256 | ~1 hour/epoch | ~45 mins/epoch |

**Total Training Time** (5 epochs):
- 2x V100: ~25 hours (with distillation), ~15 hours (without)
- 4x A100: ~10 hours (with distillation), ~7.5 hours (without)

### Memory Requirements

| Configuration | GPU Memory | Notes |
|---------------|------------|-------|
| Batch size 32 | ~20 GB | With teacher model |
| Batch size 64 | ~32 GB | Single A100 |
| FP16 training | ~60% less | Recommended |
| No distillation | ~12 GB | Much lower memory |

## 📈 Evaluation

### Evaluation Metrics

#### 1. VQA Accuracy

```python
# Exact match accuracy
accuracy = (predicted_answer == ground_truth).mean()

# Example output
VQA Accuracy: 62.5%
  Yes/No Questions: 78.3%
  Number Questions: 53.2%
  Other Questions: 58.7%
```

#### 2. Instruction Following Score

Custom metric measuring task completion:

```python
score = (
    0.4 * relevance_score +  # Response relevance
    0.3 * completeness_score +  # Task completion
    0.3 * coherence_score  # Response coherence
)

# Target: > 0.75
```

#### 3. Distillation Quality

```python
# KL divergence between student and teacher
kl_div = KL(P_student || P_teacher)

# Lower is better (target < 1.0)
```

### Evaluation Command

```bash
# Evaluate on multiple benchmarks
python scripts/evaluate.py \
    --stage 2 \
    --checkpoint outputs/stage2/checkpoint-final \
    --benchmarks vqa,okvqa,instruct \
    --split val \
    --output_file results/stage2_eval.json
```

### Sample Inference

```python
from embervlm import EmberVLM
from PIL import Image

model = EmberVLM.from_pretrained("outputs/stage2/checkpoint-final")
image = Image.open("test_image.jpg")

# VQA
response = model.generate(
    image=image,
    prompt="What is the person doing in this image?",
    max_length=100
)
print(f"Response: {response}")

# Instruction following
response = model.generate(
    image=image,
    prompt="Describe this image in detail, focusing on the main subject.",
    max_length=200
)
print(f"Response: {response}")
```

### Monitoring with W&B

```python
# Logged metrics
wandb.log({
    "stage2/sft_loss": sft_loss,
    "stage2/distill_loss": distill_loss,
    "stage2/total_loss": total_loss,
    "stage2/vqa_accuracy": vqa_acc,
    "stage2/instruct_score": instruct_score,
    "stage2/teacher_kl_div": kl_div,
    "stage2/learning_rate": current_lr,
})
```

## 💾 Checkpoints

### Checkpoint Structure

```
outputs/stage2/
├── checkpoint-500/
│   ├── pytorch_model.bin         # Student model weights
│   ├── optimizer.pt               # Optimizer state
│   ├── scheduler.pt               # LR scheduler
│   ├── config.json                # Model config
│   └── training_args.json         # Training config
├── checkpoint-1000/
├── checkpoint-final/              # Best checkpoint
├── teacher_outputs/               # Cached teacher logits
│   ├── llava_batch_0.pt
│   ├── llava_batch_1.pt
│   └── ...
└── metrics.json                   # Training metrics
```

### Caching Teacher Outputs

To speed up training, cache teacher model outputs:

```bash
# Pre-compute teacher logits
python scripts/cache_teacher_outputs.py \
    --teacher_model qwen-vl-chat \
    --dataset_path data/stage2 \
    --output_dir outputs/stage2/teacher_outputs \
    --batch_size 32

# Then train with cached outputs
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 2 \
    --use_cached_teacher_outputs outputs/stage2/teacher_outputs
```

**Benefits**:
- 3-5x faster training
- Lower GPU memory usage
- No need to load teacher model during training

### Loading from Stage 1

```bash
# Automatically loads Stage 1 checkpoint
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 2 \
    --resume_from outputs/stage1/checkpoint-final

# Or specify custom checkpoint
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 2 \
    --resume_from outputs/stage1/checkpoint-2000
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Teacher Model OOM

**Symptom**: `CUDA out of memory` when loading teacher

**Solutions**:
```bash
# Use cached teacher outputs
python scripts/cache_teacher_outputs.py

# Use smaller teacher model
--teacher_model qwen-vl-base  # Instead of chat

# Disable distillation temporarily
--disable_distillation
```

#### 2. High Distillation Loss

**Symptom**: KL divergence not decreasing

**Solutions**:
- Increase distillation weight: `--distillation_alpha 0.5`
- Lower temperature: `--distillation_temperature 1.5`
- Check teacher outputs are correct
- Verify student model loaded Stage 1 weights

#### 3. Poor VQA Performance

**Symptom**: VQA accuracy below 50%

**Solutions**:
- Train for more epochs: `--epochs 7`
- Increase learning rate: `--learning_rate 3.0e-4`
- Check VQA data preprocessing
- Verify answer format matching

#### 4. Incoherent Responses

**Symptom**: Generated text is repetitive or nonsensical

**Solutions**:
- Increase SFT weight: `--sft_weight 0.8`
- Use nucleus sampling: `--do_sample --top_p 0.95`
- Check max_length is sufficient
- Verify tokenizer configuration

### Debug Commands

```bash
# Test teacher model
python scripts/test_teacher.py --model qwen-vl-chat

# Verify distillation setup
python scripts/verify_distillation.py --stage 2

# Check data loading
python scripts/inspect_batch.py --stage 2 --num_batches 5

# Test student inference
python scripts/test_inference.py \
    --checkpoint outputs/stage2/checkpoint-500 \
    --image test.jpg \
    --prompt "Describe this image"
```

## 🎓 Advanced Topics

### Custom Teacher Models

Use different teacher models:

```python
# In training script
supported_teachers = {
    "qwen-vl-chat": "Qwen/Qwen-VL-Chat",
    "llava-1.5-7b": "liuhaotian/llava-v1.5-7b",
    "instructblip": "Salesforce/instructblip-vicuna-7b"
}

# Usage
--teacher_model llava-1.5-7b
```

### Distillation Strategies

Experiment with different strategies:

```yaml
# Hard distillation (more aggressive)
distillation:
  alpha: 0.5
  temperature: 1.0

# Soft distillation (more conservative)
distillation:
  alpha: 0.2
  temperature: 3.0

# Progressive distillation (change over time)
distillation:
  alpha_schedule: "linear"  # 0.5 -> 0.2 over training
```

### Multi-Teacher Distillation

Learn from multiple teachers:

```python
teachers = ["qwen-vl-chat", "llava-1.5-7b"]
weights = [0.6, 0.4]

L_distill = sum(
    w * KL(student, teacher)
    for w, teacher in zip(weights, teachers)
)
```

### Fine-Tuning Schedule

Customize learning rates for different components:

```python
param_groups = [
    {"params": vision_encoder.parameters(), "lr": 1.0e-5},
    {"params": projection.parameters(), "lr": 2.0e-4},
    {"params": language_model.parameters(), "lr": 2.0e-4},
]

optimizer = AdamW(param_groups)
```

## 📚 References

1. **Knowledge Distillation**: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
2. **LLaVA**: "Visual Instruction Tuning" (Liu et al., 2023)
3. **Qwen-VL**: "Qwen-VL: A Versatile Vision-Language Model" (2023)
4. **VQA-v2**: "Making the V in VQA Matter" (Goyal et al., 2017)

## 🔗 Next Steps

After completing Stage 2, proceed to:

- **[Stage 3: Robot Fleet Selection](STAGE3_GUIDE.md)**: Domain-specific training
- **[Stage 4: Advanced Reasoning](STAGE4_GUIDE.md)**: Chain-of-thought reasoning
- **[Deployment Guide](../deployment/DEPLOYMENT_GUIDE.md)**: Deploy your model

---

**Need Help?** Check our [FAQ](../FAQ.md) or open an [issue](https://github.com/yourusername/EmberVLM/issues).

