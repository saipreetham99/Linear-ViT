#!/usr/bin/env python3
"""
Training script for Linear Vision Transformers.

This script trains Vision Transformers with different attention mechanisms:
- Baseline: Standard O(n²) multi-head attention
- RippleAttention: Linear O(n) attention with spatial ripple pooling
- HydraAttention: Linear O(n) attention with multi-branch architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os

from config import get_config
from models.vit import build_vit
from data.datasets import get_dataloaders, mixup_data, mixup_criterion
from utils.metrics import AverageMeter, accuracy, MetricsTracker
from utils.helpers import (
    set_seed, get_lr, save_checkpoint, load_checkpoint,
    CosineAnnealingWarmupRestarts, log_model_info, get_device, format_time
)

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config, epoch):
    """Train for one epoch"""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup augmentation
        if config.use_augmentation and torch.rand(1).item() < config.mixup_alpha:
            images, targets_a, targets_b, lam = mixup_data(images, targets, config.mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Measure accuracy
        acc1, = accuracy(outputs, targets, topk=(1,))

        # Update meters
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress
        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')

    # Update learning rate
    scheduler.step()

    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Measure accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            # Update meters
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    print(f'Val: Loss {losses.avg:.4f} | Acc@1 {top1.avg:.3f} | Acc@5 {top5.avg:.3f}')

    return losses.avg, top1.avg

def main():
    # Get configuration
    config = get_config()
    print(config)

    # Set random seed
    set_seed(42)

    # Get device
    device = get_device()

    # Build model
    model = build_vit(config)
    model = model.to(device)

    # Log model info
    log_model_info(model, config)

    # Get data loaders
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}\n")

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=config.epochs,
        warmup_steps=config.warmup_epochs,
        max_lr=config.lr,
        min_lr=config.min_lr,
        gamma=1.0
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0

    if config.resume:
        if os.path.isfile(config.resume):
            start_epoch, best_acc = load_checkpoint(config.resume, model, optimizer)
        else:
            print(f"No checkpoint found at '{config.resume}'")

    # TensorBoard writer
    writer = SummaryWriter(config.run_log_dir)

    # Metrics tracker
    metrics = MetricsTracker(config.run_log_dir)

    print("Starting training...\n")
    training_start = time.time()

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Get current learning rate
        current_lr = get_lr(optimizer)

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Update metrics
        metrics.update(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if (epoch + 1) % config.save_freq == 0 or is_best:
            checkpoint_path = os.path.join(
                config.run_checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )

            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, checkpoint_path)

            if is_best:
                best_path = os.path.join(config.run_checkpoint_dir, 'best_model.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                }, best_path)

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{config.epochs} completed in {format_time(epoch_time)}')
        print(f'Best Acc@1: {best_acc:.3f}\n')

    # Training complete
    training_time = time.time() - training_start
    print(f"\nTraining completed in {format_time(training_time)}")

    # Plot and save metrics
    metrics.plot_metrics()
    metrics.print_summary()

    # Close writer
    writer.close()

if __name__ == '__main__':
    main()
