#!/usr/bin/env python3
"""
Evaluation script for Linear Vision Transformers.

This script evaluates trained models on test datasets and generates
detailed performance metrics including confusion matrices and per-class accuracies.
"""

import torch
import torch.nn as nn
import argparse
import os

from config import Config
from models.vit import build_vit
from data.datasets import get_dataloaders
from utils.metrics import evaluate_model, plot_confusion_matrix
from utils.helpers import set_seed, get_device

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Vision Transformer')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'tiny-imagenet'],
                        help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')

    return parser.parse_args()

def main():
    args = parse_args()

    # Set seed
    set_seed(42)

    # Get device
    device = get_device()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Fallback: create config manually
        config = Config()
        config.dataset = args.dataset
        config.data_dir = args.data_dir

    # Update dataset if specified
    if args.dataset != config.dataset:
        config.dataset = args.dataset
        if config.dataset == 'tiny-imagenet':
            config.img_size = 64
            config.num_classes = 200
        else:
            config.img_size = 32
            config.num_classes = 100

    config.batch_size = args.batch_size

    print(f"\nEvaluating on {config.dataset}")
    print(f"Attention type: {config.attention_type}\n")

    # Build model
    model = build_vit(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get data loader
    print("Loading test dataset...")
    _, test_loader = get_dataloaders(config)
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, config.num_classes)

    # Print results
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Best Checkpoint Accuracy: {checkpoint.get('best_acc', 0):.2f}%")
    print("="*70 + "\n")

    # Per-class accuracy statistics
    per_class_acc = results['per_class_accuracy'] * 100
    print(f"Per-class Accuracy Statistics:")
    print(f"  Mean: {per_class_acc.mean():.2f}%")
    print(f"  Std:  {per_class_acc.std():.2f}%")
    print(f"  Min:  {per_class_acc.min():.2f}%")
    print(f"  Max:  {per_class_acc.max():.2f}%\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save confusion matrix
    conf_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], conf_matrix_path, config.num_classes)
    print(f"Confusion matrix saved to {conf_matrix_path}")

    # Save detailed results
    results_path = os.path.join(args.output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Evaluation Results\n")
        f.write("="*70 + "\n")
        f.write(f"Model: {config.attention_type}\n")
        f.write(f"Dataset: {config.dataset}\n")
        f.write(f"Overall Accuracy: {results['overall_accuracy']:.2f}%\n")
        f.write(f"Checkpoint Best Accuracy: {checkpoint.get('best_acc', 0):.2f}%\n")
        f.write("\n")
        f.write(f"Per-class Accuracy Statistics:\n")
        f.write(f"  Mean: {per_class_acc.mean():.2f}%\n")
        f.write(f"  Std:  {per_class_acc.std():.2f}%\n")
        f.write(f"  Min:  {per_class_acc.min():.2f}%\n")
        f.write(f"  Max:  {per_class_acc.max():.2f}%\n")
        f.write("="*70 + "\n")

    print(f"Detailed results saved to {results_path}\n")

if __name__ == '__main__':
    main()
