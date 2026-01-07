# EmberVLM Documentation

Comprehensive documentation for EmberVLM - A lightweight Vision-Language Model with Chain-of-Thought reasoning for robot fleet selection.

## 📚 Documentation Structure

### 🎯 Training Guides
Complete guides for all 4 training stages:

- **[Stage 1: Visual-Language Alignment](training/STAGE1_GUIDE.md)**
  - Aligning RepViT vision encoder with TinyLLM language model
  - Contrastive learning and image captioning
  - COCO, Flickr30k, CC3M datasets
  
- **[Stage 2: Multimodal Instruction Tuning](training/STAGE2_GUIDE.md)**
  - Task-following capabilities with distillation
  - LLaVA-Instruct, VQA-v2, OK-VQA datasets
  - Knowledge distillation from Qwen-VL-Chat
  
- **[Stage 3: Robot Fleet Selection Training](training/STAGE3_GUIDE.md)**
  - Robot selection based on task requirements
  - Chain-of-thought reasoning for selection
  - Multi-robot coordination and top-N selection
  
- **[Stage 4: Advanced Reasoning Integration](training/STAGE4_GUIDE.md)**
  - DeepSeek-R1 style reasoning
  - Two-phase training: reasoning heads + joint fine-tuning
  - Reasoning consistency and validation

### 🏗️ Architecture
- **[Model Architecture](architecture/MODEL_ARCHITECTURE.md)**: Complete model architecture overview
- **[Reasoning Module](architecture/REASONING_MODULE.md)**: Chain-of-thought reasoning implementation
- **[Vision Encoder](architecture/VISION_ENCODER.md)**: RepViT-based vision encoding

### 📊 Datasets
- **[Dataset Overview](datasets/DATASET_OVERVIEW.md)**: All datasets used in training
- **[Robot Selection Data](datasets/ROBOT_SELECTION_DATA.md)**: Robot fleet selection dataset format
- **[Data Augmentation](datasets/AUGMENTATION_STRATEGIES.md)**: Augmentation techniques

### 🚀 Deployment
- **[Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)**: Production deployment
- **[Quantization Guide](deployment/QUANTIZATION.md)**: Model quantization for edge devices
- **[API Reference](deployment/API_REFERENCE.md)**: Complete API documentation

## 🎓 Training Pipeline Overview

EmberVLM uses a **4-stage progressive training pipeline**:

```
┌─────────────────┐
│   Stage 1       │  Visual-Language Alignment
│   3 epochs      │  • Align vision and language spaces
│   330K samples  │  • Contrastive learning + Captioning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Stage 2       │  Multimodal Instruction Tuning
│   5 epochs      │  • Task-following capabilities
│   300K samples  │  • Knowledge distillation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Stage 3       │  Robot Fleet Selection
│   20 epochs     │  • Robot selection reasoning
│   1K samples    │  • Multi-robot coordination
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Stage 4       │  Advanced Reasoning
│   10 epochs     │  • DeepSeek-R1 style reasoning
│   50K samples   │  • Two-phase training
└─────────────────┘
```

## 🚦 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/EmberVLM.git
cd EmberVLM

# Install dependencies
pip install -r requirements.txt
```

### Training All Stages

```bash
# Train all 4 stages sequentially
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 1,2,3,4 \
    --wandb_project embervlm \
    --output_dir outputs
```

### Training Individual Stages

```bash
# Stage 1 only
torchrun --nproc_per_node=2 scripts/train_all.py --stages 1

# Stage 3 only (requires Stage 2 checkpoint)
torchrun --nproc_per_node=2 scripts/train_all.py --stages 3 \
    --resume_from outputs/stage2/checkpoint-789
```

### Inference

```python
from embervlm import EmberVLM

# Load trained model
model = EmberVLM.from_pretrained("outputs/final")

# Robot selection inference
result = model.select_robots(
    image="warehouse_scene.jpg",
    task="Pick and place boxes from conveyor to shelves",
    top_k=3,
    reasoning=True
)

print(f"Selected Robots: {result['robots']}")
print(f"Reasoning: {result['reasoning']}")
```

## 📁 Project Structure

```
EmberVLM/
├── configs/                    # Configuration files
│   └── training_stages.yaml   # Stage-specific configurations
├── docs/                      # Documentation (you are here)
│   ├── training/             # Training guides for all stages
│   ├── architecture/         # Model architecture docs
│   ├── deployment/          # Deployment guides
│   └── datasets/            # Dataset documentation
├── embervlm/                 # Main package
│   ├── models/              # Model implementations
│   ├── training/            # Training scripts
│   ├── data/                # Data loaders
│   └── monitoring/          # Logging and visualization
├── scripts/                  # Training and evaluation scripts
├── robot-selection-dataset/ # Robot selection data
└── outputs/                 # Training outputs
    ├── stage1/
    ├── stage2/
    ├── stage3/
    │   └── visualizations/  # Stage 3 visualizations
    └── stage4/
        └── visualizations/  # Stage 4 visualizations
```

## 📊 Visualizations

EmberVLM automatically generates comprehensive visualizations during training:

### Stage 3 Visualizations (Robot Selection)
- `outputs/stage3/visualizations/confusion_matrices/` - Robot selection confusion matrices
- `outputs/stage3/visualizations/radar_charts/` - Per-robot capability radar charts
- `outputs/stage3/visualizations/calibration_plots/` - Confidence calibration curves
- `outputs/stage3/visualizations/training_curves/` - Loss and accuracy over time

### Stage 4 Visualizations (Reasoning)
- `outputs/stage4/visualizations/reasoning_quality/` - Reasoning step analysis
- `outputs/stage4/visualizations/attention_maps/` - Attention heatmaps
- `outputs/stage4/visualizations/consistency_plots/` - Reasoning consistency metrics

## 🎯 Key Features

- **Lightweight**: 37M total parameters, 23M trainable
- **Efficient**: RepViT vision encoder (14M params) + TinyLLM (30M params)
- **Reasoning**: Chain-of-thought reasoning for robot selection
- **Multi-Robot**: Top-N robot selection with confidence scores
- **Visualizations**: Comprehensive training visualizations
- **Production-Ready**: Quantization support, efficient inference

## 📖 Reading Guide

### For Training
1. Start with [Stage 1 Guide](training/STAGE1_GUIDE.md) to understand the foundation
2. Progress through stages 2-4 sequentially
3. Refer to [Dataset Overview](datasets/DATASET_OVERVIEW.md) for data preparation
4. Check [Model Architecture](architecture/MODEL_ARCHITECTURE.md) for model details

### For Deployment
1. Review [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)
2. Consider [Quantization](deployment/QUANTIZATION.md) for edge devices
3. Use [API Reference](deployment/API_REFERENCE.md) for integration

### For Research
1. Study [Reasoning Module](architecture/REASONING_MODULE.md)
2. Review [Stage 4 Guide](training/STAGE4_GUIDE.md) for advanced reasoning
3. Examine [Robot Selection Data](datasets/ROBOT_SELECTION_DATA.md) format

## 🔧 Configuration

All training stages are configured in `configs/training_stages.yaml`. Key parameters:

| Stage | Epochs | Batch Size | Learning Rate | Samples |
|-------|--------|------------|---------------|---------|
| 1     | 3      | 128        | 2.0e-4        | 330K    |
| 2     | 5      | 64         | 2.0e-4        | 300K    |
| 3     | 20     | 32         | 1.0e-4        | 1K      |
| 4     | 10     | 32         | 1.0e-4/5.0e-5 | 50K     |

## 📝 Citation

If you use EmberVLM in your research, please cite:

```bibtex
@article{embervlm2026,
  title={EmberVLM: Lightweight Vision-Language Model with Chain-of-Thought Reasoning},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines in the main README.

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/EmberVLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/EmberVLM/discussions)
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- **RepViT**: Efficient vision encoder architecture
- **TinyLLM**: Lightweight language model
- **DeepSeek-R1**: Inspiration for reasoning capabilities
- **LLaVA**: Multimodal instruction tuning methodology

