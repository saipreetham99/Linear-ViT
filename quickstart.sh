#!/bin/bash

# Quickstart script for Linear-ViT
# This script helps you get started quickly with training and evaluation

set -e  # Exit on error

echo "=========================================="
echo "Linear-ViT Quickstart"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Run model tests
echo "=========================================="
echo "Testing models..."
echo "=========================================="
python test_models.py
echo ""

# Generate visualizations
echo "=========================================="
echo "Generating visualizations..."
echo "=========================================="
python visualize.py --type all
echo ""

# Ask user what they want to do
echo "=========================================="
echo "What would you like to do?"
echo "=========================================="
echo "1) Train RippleViT on CIFAR-100 (recommended)"
echo "2) Train HydraViT on CIFAR-100"
echo "3) Train Baseline ViT on CIFAR-100"
echo "4) Quick training test (10 epochs)"
echo "5) Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Training RippleViT on CIFAR-100..."
        python train.py --attention_type ripple --dataset cifar100 --epochs 200
        ;;
    2)
        echo ""
        echo "Training HydraViT on CIFAR-100..."
        python train.py --attention_type hydra --dataset cifar100 --epochs 200
        ;;
    3)
        echo ""
        echo "Training Baseline ViT on CIFAR-100..."
        python train.py --attention_type baseline --dataset cifar100 --epochs 200
        ;;
    4)
        echo ""
        echo "Running quick training test (10 epochs)..."
        python train.py --attention_type ripple --dataset cifar100 --epochs 10
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "To monitor training with TensorBoard:"
echo "  tensorboard --logdir logs/"
echo ""
echo "To evaluate a trained model:"
echo "  python evaluate.py --checkpoint checkpoints/*/best_model.pth"
echo ""
