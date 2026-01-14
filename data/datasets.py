"""
Dataset utilities for CIFAR-100.

This module provides data loading, augmentation, and mixup functionality
for training Vision Transformers on CIFAR-100.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


def get_dataloaders(config):
    """
    Get CIFAR-100 train and validation dataloaders.

    Args:
        config: Configuration object with batch_size, num_workers, img_size, etc.

    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    # Load CIFAR-100 dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )

    return train_loader, val_loader


def mixup_data(x, y, alpha=1.0):
    """
    Apply mixup augmentation to a batch of data.

    Mixup creates virtual training examples by mixing pairs of examples and their labels.
    Reference: https://arxiv.org/abs/1710.09412

    Args:
        x: Input images (batch_size, channels, height, width)
        y: Target labels (batch_size,)
        alpha: Mixup hyperparameter (higher = more mixing)

    Returns:
        mixed_x: Mixed input images
        y_a: First set of target labels
        y_b: Second set of target labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for mixup training.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: First set of target labels
        y_b: Second set of target labels
        lam: Mixing coefficient from mixup_data

    Returns:
        loss: Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
