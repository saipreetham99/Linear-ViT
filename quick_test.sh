#!/bin/bash

# Quick test script to verify the installation and basic functionality

echo "=========================================="
echo "Linear-ViT Quick Test"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"
echo ""

# Test 1: Model verification
echo "=========================================="
echo "Test 1: Verifying all models..."
echo "=========================================="
python test_models.py

if [ $? -ne 0 ]; then
    echo "❌ Model tests failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test 2: Quick training test (1 epoch)..."
echo "=========================================="
echo "This will take 2-5 minutes depending on your hardware"
echo ""

# Train for 1 epoch with small batch size
python train.py \
    --attention_type ripple \
    --dataset cifar100 \
    --epochs 1 \
    --batch_size 32 \
    --num_workers 0

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All tests passed!"
    echo "=========================================="
    echo ""
    echo "Your installation is working correctly!"
    echo ""
    echo "Next steps:"
    echo "1. Full training (10 epochs): python train.py --attention_type ripple --epochs 10"
    echo "2. Production training (200 epochs): python train.py --attention_type ripple --epochs 200"
    echo "3. Monitor with TensorBoard: tensorboard --logdir logs/"
    echo ""
else
    echo ""
    echo "❌ Training test failed"
    echo "Check the error messages above"
    echo ""
    exit 1
fi
