import argparse
import os
from datetime import datetime

class Config:
    def __init__(self):
        # Model architecture
        self.img_size = 32  # CIFAR-100 default
        self.patch_size = 4
        self.num_classes = 100
        self.dim = 384
        self.depth = 12
        self.heads = 6
        self.mlp_dim = 1536
        self.dropout = 0.1
        self.emb_dropout = 0.1

        # Attention mechanism
        self.attention_type = 'ripple'  # 'baseline', 'ripple', 'hydra'
        self.ripple_stages = 3  # For RippleAttention
        self.hydra_branches = 4  # For HydraAttention

        # Training
        self.batch_size = 128
        self.epochs = 200
        self.lr = 3e-4
        self.weight_decay = 0.05
        self.warmup_epochs = 10
        self.min_lr = 1e-6

        # Dataset
        self.dataset = 'cifar100'  # 'cifar100' or 'tiny-imagenet'
        self.data_dir = './data'
        self.num_workers = 4

        # Augmentation
        self.use_augmentation = True
        self.cutmix_prob = 0.5
        self.mixup_alpha = 0.8

        # Logging
        self.log_dir = './logs'
        self.checkpoint_dir = './checkpoints'
        self.save_freq = 10
        self.print_freq = 50

        # Hardware
        self.use_cuda_kernels = True
        self.device = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'

    def update_from_args(self, args):
        """Update config from command line arguments"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

        # Adjust image size based on dataset
        if self.dataset == 'tiny-imagenet':
            self.img_size = 64
            self.num_classes = 200

        # Create run directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{self.attention_type}_{self.dataset}_{timestamp}"
        self.run_log_dir = os.path.join(self.log_dir, self.run_name)
        self.run_checkpoint_dir = os.path.join(self.checkpoint_dir, self.run_name)

        os.makedirs(self.run_log_dir, exist_ok=True)
        os.makedirs(self.run_checkpoint_dir, exist_ok=True)

    def __str__(self):
        """Pretty print configuration"""
        config_str = "Configuration:\n"
        config_str += "-" * 50 + "\n"
        for key, value in vars(self).items():
            config_str += f"{key:.<30} {value}\n"
        config_str += "-" * 50
        return config_str

def get_config():
    """Parse command line arguments and return config"""
    parser = argparse.ArgumentParser(description='Linear Vision Transformer Training')

    # Model
    parser.add_argument('--attention_type', type=str, default='ripple',
                        choices=['baseline', 'ripple', 'hydra'],
                        help='Type of attention mechanism')
    parser.add_argument('--dim', type=int, default=384,
                        help='Model dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6,
                        help='Number of attention heads')

    # Training
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'tiny-imagenet'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')

    # Hardware
    parser.add_argument('--use_cuda_kernels', action='store_true',
                        help='Use custom CUDA kernels')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Misc
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation only')

    args = parser.parse_args()

    config = Config()
    config.update_from_args(args)

    return config
