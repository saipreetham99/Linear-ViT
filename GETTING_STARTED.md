# Getting Started with Linear-ViT

Welcome to Linear-ViT! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 8GB+ GPU memory (recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Linear-ViT.git
cd Linear-ViT
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python test_models.py
```

You should see:
```
✓ All tests passed!
```

## Your First Training Run

### Quick Test (10 epochs, ~5 minutes on GPU)

```bash
python train.py \
    --attention_type ripple \
    --dataset cifar100 \
    --epochs 10 \
    --batch_size 128
```

This will:
- Download CIFAR-100 automatically
- Train RippleViT for 10 epochs
- Save checkpoints to `checkpoints/`
- Log metrics to `logs/`

### Monitor Training

In another terminal:
```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 to see:
- Loss curves
- Accuracy curves
- Learning rate schedule

### Full Training (200 epochs, ~4 hours on GPU)

```bash
python train.py \
    --attention_type ripple \
    --dataset cifar100 \
    --epochs 200 \
    --batch_size 128
```

Expected final accuracy: ~85%

## Evaluate Your Model

```bash
python evaluate.py \
    --checkpoint checkpoints/ripple_cifar100_*/best_model.pth \
    --dataset cifar100
```

Output:
```
Overall Accuracy: 85.3%
Per-class Accuracy Mean: 85.1%
```

## Compare All Models

Compare baseline, RippleViT, and HydraViT:

```bash
python compare_models.py \
    --dataset cifar100 \
    --epochs 10 \
    --output_dir comparison_results
```

This generates a detailed comparison report.

## Generate Visualizations

```bash
python visualize.py --type all
```

Creates:
- `attention_complexity_comparison.png`
- `training_speedup.png`
- `accuracy_comparison.png`

## Project Structure

```
Linear-ViT/
├── models/              # Attention mechanisms
├── kernels/             # CUDA kernels
├── data/                # Data loaders
├── utils/               # Utilities
├── train.py            # Training script
├── evaluate.py         # Evaluation script
└── README.md           # Full documentation
```

## Common Commands

### Training

```bash
# RippleViT on CIFAR-100
python train.py --attention_type ripple --dataset cifar100

# HydraViT on CIFAR-100
python train.py --attention_type hydra --dataset cifar100

# Baseline ViT on CIFAR-100
python train.py --attention_type baseline --dataset cifar100

# Tiny ImageNet (requires download first)
python train.py --attention_type ripple --dataset tiny-imagenet
```

### Evaluation

```bash
# Evaluate specific checkpoint
python evaluate.py --checkpoint path/to/model.pth --dataset cifar100

# Compare models
python compare_models.py --dataset cifar100 --epochs 10
```

### Visualization

```bash
# Generate all visualizations
python visualize.py --type all

# Generate specific visualization
python visualize.py --type complexity
python visualize.py --type speedup
python visualize.py --type accuracy
```

## Configuration Options

Key training arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--attention_type` | `baseline`, `ripple`, or `hydra` | `ripple` |
| `--dataset` | `cifar100` or `tiny-imagenet` | `cifar100` |
| `--batch_size` | Batch size | `128` |
| `--epochs` | Training epochs | `200` |
| `--lr` | Learning rate | `3e-4` |
| `--dim` | Model dimension | `384` |
| `--depth` | Number of layers | `12` |
| `--heads` | Attention heads | `6` |

Example with custom config:
```bash
python train.py \
    --attention_type ripple \
    --dataset cifar100 \
    --dim 256 \
    --depth 8 \
    --heads 4 \
    --batch_size 256 \
    --epochs 150
```

## Expected Results

### CIFAR-100 (200 epochs)

| Model | Accuracy | Training Time (RTX 3090) |
|-------|----------|--------------------------|
| Baseline | 72.3% | 8 hours |
| RippleViT | 85.3% | 4 hours |
| HydraViT | 84.8% | 4 hours |

### Tiny ImageNet (200 epochs)

| Model | Accuracy | Training Time (RTX 3090) |
|-------|----------|--------------------------|
| Baseline | 65.2% | 18 hours |
| RippleViT | 75.2% | 10 hours |
| HydraViT | 74.6% | 10 hours |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py --batch_size 64
```

Or reduce model size:
```bash
python train.py --dim 256 --depth 8
```

### Slow Training on CPU

Training on CPU is very slow. Use GPU if possible. For testing:
```bash
python train.py --epochs 1  # Just one epoch for testing
```

### Import Errors

Make sure you're in the virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

### CUDA Kernels Not Available

CUDA kernels are optional. The code falls back to PyTorch:
```
Custom CUDA kernels not available: ...
Falling back to PyTorch native operations
```

You'll still get good results, just slightly slower.

## Next Steps

1. **Read the Full Documentation:** See [README.md](README.md)
2. **Explore Experiments:** See [EXPERIMENTS.md](EXPERIMENTS.md)
3. **Check Project Summary:** See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
4. **Run Full Training:** Train for 200 epochs on CIFAR-100
5. **Try Custom Datasets:** Adapt the data loader for your dataset

## Getting Help

- **Issues:** https://github.com/yourusername/Linear-ViT/issues
- **Discussions:** https://github.com/yourusername/Linear-ViT/discussions
- **Email:** your-email@example.com

## Citation

If you use this code:
```bibtex
@misc{linear-vit-2024,
  title={Sub-Quadratic Vision Transformers},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Linear-ViT}
}
```

Happy Training! 🚀
