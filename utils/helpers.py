import torch
import torch.nn as nn
import random
import numpy as np
import os
from datetime import datetime

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, filename):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0)

    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch} with best accuracy {best_acc:.2f}%")

    return epoch, best_acc

class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup and restarts.

    Args:
        optimizer: Wrapped optimizer
        first_cycle_steps: Number of steps in first cycle
        warmup_steps: Number of warmup steps
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        gamma: Decay factor for max_lr after each restart
    """
    def __init__(self, optimizer, first_cycle_steps, warmup_steps, min_lr=0, max_lr=1, gamma=1.0):
        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer)

        # Set initial learning rate
        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * self.step_in_cycle / self.warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1

            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = self.first_cycle_steps
                self.max_lr *= self.gamma
        else:
            if epoch >= self.first_cycle_steps:
                self.cycle = 1
                self.step_in_cycle = epoch - self.first_cycle_steps
            else:
                self.step_in_cycle = epoch

        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Non-trainable: {total - trainable:,}\n")

    return total, trainable

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0, mode='max'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

def log_model_info(model, config):
    """Log detailed model information"""
    print("\n" + "="*70)
    print("Model Configuration")
    print("="*70)
    print(f"Architecture: Vision Transformer")
    print(f"Attention Type: {config.attention_type}")
    print(f"Image Size: {config.img_size}x{config.img_size}")
    print(f"Patch Size: {config.patch_size}x{config.patch_size}")
    print(f"Number of Patches: {(config.img_size // config.patch_size) ** 2}")
    print(f"Embedding Dim: {config.dim}")
    print(f"Depth: {config.depth} layers")
    print(f"Attention Heads: {config.heads}")
    print(f"MLP Dim: {config.mlp_dim}")
    print(f"Dropout: {config.dropout}")

    if config.attention_type == 'ripple':
        print(f"Ripple Stages: {config.ripple_stages}")
    elif config.attention_type == 'hydra':
        print(f"Hydra Branches: {config.hydra_branches}")

    print("="*70)

    # Count parameters
    count_parameters(model)

    # Print complexity
    complexity = model.get_attention_complexity(config.img_size)
    print(f"Attention Complexity: {complexity}\n")
    print("="*70 + "\n")

def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device
