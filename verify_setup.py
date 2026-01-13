#!/usr/bin/env python3
"""
Quick verification script to ensure everything is set up correctly.
Tests model creation, data loading, and one forward/backward pass.
"""

import torch
import sys
from config import get_config
from models.vit import build_vit
from data.datasets import get_dataloaders

def verify_setup():
    """Run quick verification tests"""

    print("\n" + "="*70)
    print("Linear-ViT Setup Verification")
    print("="*70 + "\n")

    # Test 1: Config
    print("1. Testing configuration...")
    try:
        sys.argv = ['verify_setup.py', '--attention_type', 'ripple', '--epochs', '1']
        config = get_config()
        print("   ✓ Configuration loaded successfully")
    except Exception as e:
        print(f"   ✗ Configuration failed: {e}")
        return False

    # Test 2: Model creation
    print("2. Testing model creation...")
    try:
        model = build_vit(config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"   ✓ Model created successfully (device: {device})")
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False

    # Test 3: Data loading
    print("3. Testing data loading...")
    try:
        config.num_workers = 0  # Avoid multiprocessing issues
        config.batch_size = 4   # Small batch for quick test
        train_loader, val_loader = get_dataloaders(config)
        print(f"   ✓ Data loaders created (train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)})")
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return False

    # Test 4: Forward pass
    print("4. Testing forward pass...")
    try:
        model.eval()
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        print(f"   ✓ Forward pass successful (output shape: {outputs.shape})")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False

    # Test 5: Backward pass
    print("5. Testing backward pass...")
    try:
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"   ✓ Backward pass successful (loss: {loss.item():.4f})")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        return False

    # Test 6: Checkpoint saving
    print("6. Testing checkpoint saving...")
    try:
        import os
        test_checkpoint_dir = './test_checkpoint'
        os.makedirs(test_checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(test_checkpoint_dir, 'test_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, checkpoint_path)

        # Verify file exists
        if os.path.exists(checkpoint_path):
            print(f"   ✓ Checkpoint saved successfully")
            # Clean up
            os.remove(checkpoint_path)
            os.rmdir(test_checkpoint_dir)
        else:
            print(f"   ✗ Checkpoint file not found")
            return False
    except Exception as e:
        print(f"   ✗ Checkpoint saving failed: {e}")
        return False

    return True

if __name__ == '__main__':
    print("\nThis script verifies your Linear-ViT installation is working correctly.")
    print("It will test model creation, data loading, and training operations.\n")

    success = verify_setup()

    if success:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nYour installation is working correctly!")
        print("\nYou can now:")
        print("  • Run quick training test: python train.py --attention_type ripple --epochs 1")
        print("  • Start full training: python train.py --attention_type ripple --epochs 200")
        print("  • Monitor with TensorBoard: tensorboard --logdir logs/")
        print("\n")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("✗ VERIFICATION FAILED")
        print("="*70)
        print("\nPlease check the error messages above and ensure:")
        print("  1. All dependencies are installed: pip install -r requirements.txt")
        print("  2. You're in the correct directory")
        print("  3. Python version is 3.8 or higher")
        print("\n")
        sys.exit(1)
