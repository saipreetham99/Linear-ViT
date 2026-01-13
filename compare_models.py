#!/usr/bin/env python3
"""
Comprehensive comparison script for all attention mechanisms.

This script trains all three models (baseline, ripple, hydra) for a specified
number of epochs and generates a detailed comparison report.
"""

import torch
import argparse
import time
import os
from datetime import datetime
import json

from config import Config
from models.vit import build_vit
from data.datasets import get_dataloaders
from utils.helpers import set_seed, get_device, count_parameters
from utils.metrics import AverageMeter, accuracy

def quick_train_eval(attention_type, config, device):
    """
    Quickly train and evaluate a model.

    Args:
        attention_type: Type of attention mechanism
        config: Configuration object
        device: Device to use

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"Training {attention_type.upper()} model")
    print(f"{'='*70}")

    # Update config
    config.attention_type = attention_type

    # Build model
    model = build_vit(config)
    model = model.to(device)

    # Count parameters
    total_params, trainable_params = model.count_parameters()

    # Get data
    train_loader, val_loader = get_dataloaders(config)

    # Setup training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Training loop
    best_acc = 0.0
    train_time_total = 0.0

    for epoch in range(config.epochs):
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        epoch_start = time.time()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, = accuracy(outputs, targets, topk=(1,))
            train_loss.update(loss.item(), images.size(0))
            train_acc.update(acc1.item(), images.size(0))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_time = time.time() - epoch_start
        train_time_total += epoch_time

        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, targets)

                acc1, = accuracy(outputs, targets, topk=(1,))
                val_loss.update(loss.item(), images.size(0))
                val_acc.update(acc1.item(), images.size(0))

        best_acc = max(best_acc, val_acc.avg)

        print(f"Epoch [{epoch+1}/{config.epochs}] "
              f"Train Loss: {train_loss.avg:.4f} "
              f"Train Acc: {train_acc.avg:.2f}% "
              f"Val Loss: {val_loss.avg:.4f} "
              f"Val Acc: {val_acc.avg:.2f}% "
              f"Time: {epoch_time:.1f}s")

    # Compute complexity
    num_patches = (config.img_size // config.patch_size) ** 2
    if attention_type == 'baseline':
        complexity = num_patches ** 2 * config.dim
    else:
        complexity = num_patches * config.dim ** 2

    results = {
        'attention_type': attention_type,
        'final_train_acc': train_acc.avg,
        'final_val_acc': val_acc.avg,
        'best_val_acc': best_acc,
        'total_train_time': train_time_total,
        'avg_epoch_time': train_time_total / config.epochs,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'theoretical_complexity': complexity,
    }

    return results

def generate_comparison_report(results_list, output_path):
    """Generate a detailed comparison report"""

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Accuracy comparison
        f.write("-"*80 + "\n")
        f.write("ACCURACY COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'Final Val Acc':<20} {'Best Val Acc':<20}\n")
        f.write("-"*80 + "\n")

        baseline_acc = None
        for result in results_list:
            name = result['attention_type'].capitalize()
            final_acc = result['final_val_acc']
            best_acc = result['best_val_acc']

            if result['attention_type'] == 'baseline':
                baseline_acc = best_acc
                f.write(f"{name:<20} {final_acc:>18.2f}% {best_acc:>18.2f}%\n")
            else:
                improvement = best_acc - baseline_acc if baseline_acc else 0
                f.write(f"{name:<20} {final_acc:>18.2f}% {best_acc:>15.2f}% (+{improvement:.2f}%)\n")

        f.write("\n")

        # Training time comparison
        f.write("-"*80 + "\n")
        f.write("TRAINING TIME COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'Total Time':<20} {'Avg Epoch Time':<20} {'Speedup':<15}\n")
        f.write("-"*80 + "\n")

        baseline_time = None
        for result in results_list:
            name = result['attention_type'].capitalize()
            total_time = result['total_train_time']
            avg_time = result['avg_epoch_time']

            if result['attention_type'] == 'baseline':
                baseline_time = total_time
                speedup_str = "1.00×"
            else:
                speedup = baseline_time / total_time if baseline_time else 1.0
                speedup_str = f"{speedup:.2f}×"

            f.write(f"{name:<20} {total_time:>17.1f}s {avg_time:>18.1f}s {speedup_str:>14}\n")

        f.write("\n")

        # Model size comparison
        f.write("-"*80 + "\n")
        f.write("MODEL SIZE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'Total Params':<20} {'Trainable Params':<20}\n")
        f.write("-"*80 + "\n")

        for result in results_list:
            name = result['attention_type'].capitalize()
            total = result['total_params']
            trainable = result['trainable_params']
            f.write(f"{name:<20} {total:>18,} {trainable:>18,}\n")

        f.write("\n")

        # Complexity comparison
        f.write("-"*80 + "\n")
        f.write("COMPUTATIONAL COMPLEXITY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'Operations':<20} {'Relative Cost':<20}\n")
        f.write("-"*80 + "\n")

        baseline_complexity = None
        for result in results_list:
            name = result['attention_type'].capitalize()
            complexity = result['theoretical_complexity']

            if result['attention_type'] == 'baseline':
                baseline_complexity = complexity
                relative = "1.00×"
            else:
                relative = f"{complexity / baseline_complexity:.2f}×" if baseline_complexity else "N/A"

            f.write(f"{name:<20} {complexity:>18,} {relative:>19}\n")

        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"\nComparison report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare all attention mechanisms')

    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['cifar100', 'tiny-imagenet'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train each model')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Get device
    device = get_device()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup config
    config = Config()
    config.dataset = args.dataset
    config.epochs = args.epochs
    config.batch_size = args.batch_size

    if config.dataset == 'tiny-imagenet':
        config.img_size = 64
        config.num_classes = 200

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Dataset: {config.dataset}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print("="*70)

    # Train and evaluate all models
    attention_types = ['baseline', 'ripple', 'hydra']
    results_list = []

    for attn_type in attention_types:
        try:
            results = quick_train_eval(attn_type, config, device)
            results_list.append(results)

            # Save individual results
            results_file = os.path.join(args.output_dir, f'{attn_type}_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"\nError training {attn_type}: {str(e)}")
            continue

    # Generate comparison report
    if results_list:
        report_path = os.path.join(args.output_dir, 'comparison_report.txt')
        generate_comparison_report(results_list, report_path)

        # Print summary
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        for result in results_list:
            print(f"{result['attention_type'].capitalize():12s}: "
                  f"Acc={result['best_val_acc']:.2f}%, "
                  f"Time={result['total_train_time']:.1f}s")
        print("="*70 + "\n")

if __name__ == '__main__':
    main()
