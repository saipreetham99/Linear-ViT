import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions.

    Args:
        output: Model predictions (batch_size, num_classes)
        target: Ground truth labels (batch_size,)
        topk: Tuple of k values for top-k accuracy

    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class MetricsTracker:
    """Track and log training metrics"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Update metrics for current epoch"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)

    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(self.history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # Gap between train and val
        gap = np.array(self.history['train_acc']) - np.array(self.history['val_acc'])
        axes[1, 1].plot(gap)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap (%)')
        axes[1, 1].set_title('Train-Val Accuracy Gap')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.log_dir}/metrics.png', dpi=300, bbox_inches='tight')

        plt.close()

    def print_summary(self):
        """Print summary of best results"""
        best_val_acc = max(self.history['val_acc'])
        best_epoch = self.history['val_acc'].index(best_val_acc)

        print("\n" + "="*50)
        print("Training Summary")
        print("="*50)
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Achieved at Epoch: {best_epoch + 1}")
        print(f"Train Accuracy at Best Epoch: {self.history['train_acc'][best_epoch]:.2f}%")
        print(f"Final Validation Accuracy: {self.history['val_acc'][-1]:.2f}%")
        print("="*50 + "\n")

def evaluate_model(model, test_loader, device, num_classes=100):
    """
    Comprehensive model evaluation with confusion matrix and classification report.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        num_classes: Number of classes

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Overall accuracy
    acc = 100.0 * np.mean(all_preds == all_targets)

    # Per-class accuracy
    conf_matrix = confusion_matrix(all_targets, all_preds)
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    results = {
        'overall_accuracy': acc,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'targets': all_targets
    }

    return results

def plot_confusion_matrix(conf_matrix, save_path, num_classes=100):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))

    # For large matrices, show a subset or aggregated version
    if num_classes > 20:
        # Show aggregated confusion by grouping classes
        block_size = num_classes // 10
        aggregated = np.zeros((10, 10))

        for i in range(10):
            for j in range(10):
                aggregated[i, j] = conf_matrix[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ].sum()

        sns.heatmap(aggregated, annot=True, fmt='.0f', cmap='Blues')
        plt.title('Confusion Matrix (Aggregated)')
    else:
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
