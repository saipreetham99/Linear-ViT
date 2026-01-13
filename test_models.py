#!/usr/bin/env python3
"""
Quick test script to verify all models work correctly.

This script performs sanity checks on all three attention mechanisms
without requiring full training.
"""

import torch
import time
from config import Config
from models.vit import build_vit

def test_model(attention_type, device='cpu'):
    """Test a single model configuration"""
    print(f"\n{'='*70}")
    print(f"Testing {attention_type.upper()} attention")
    print(f"{'='*70}")

    # Create config
    config = Config()
    config.attention_type = attention_type
    config.img_size = 32
    config.num_classes = 100
    config.dim = 192  # Smaller for quick testing
    config.depth = 6
    config.heads = 4

    # Build model
    model = build_vit(config)
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, config.img_size, config.img_size).to(device)

    print(f"Input shape: {x.shape}")

    # Warm-up
    with torch.no_grad():
        _ = model(x)

    # Time forward pass
    num_runs = 10
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(x)
    elapsed = (time.time() - start) / num_runs

    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {elapsed*1000:.2f} ms")

    # Check output
    assert output.shape == (batch_size, config.num_classes), "Output shape mismatch!"

    # Test backward pass
    model.train()
    output = model(x)
    loss = output.sum()
    loss.backward()

    print("✓ Forward pass successful")
    print("✓ Backward pass successful")

    # Get complexity estimate
    complexity = model.get_attention_complexity(config.img_size)
    print(f"Attention complexity: {complexity}")

    return True

def main():
    """Run tests for all models"""
    print("\n" + "="*70)
    print("Model Testing Suite")
    print("="*70)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test all attention types
    attention_types = ['baseline', 'ripple', 'hydra']
    results = {}

    for attn_type in attention_types:
        try:
            success = test_model(attn_type, device)
            results[attn_type] = "✓ PASSED"
        except Exception as e:
            results[attn_type] = f"✗ FAILED: {str(e)}"
            print(f"\nError testing {attn_type}: {str(e)}")

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    for attn_type, result in results.items():
        print(f"{attn_type:12s}: {result}")
    print("="*70 + "\n")

    # Check if all passed
    all_passed = all("PASSED" in r for r in results.values())
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == '__main__':
    exit(main())
