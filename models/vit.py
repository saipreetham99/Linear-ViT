import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.baseline_vit import BaselineTransformerBlock
from models.ripple_attention import RippleTransformerBlock
from models.hydra_attention import HydraTransformerBlock

class VisionTransformer(nn.Module):
    """
    Vision Transformer with configurable attention mechanism.

    Supports three types of attention:
    - 'baseline': Standard O(n²) multi-head attention
    - 'ripple': RippleAttention with O(n) complexity
    - 'hydra': HydraAttention with O(n) complexity

    Args:
        img_size: Input image size
        patch_size: Size of image patches
        num_classes: Number of output classes
        dim: Model dimension
        depth: Number of transformer layers
        heads: Number of attention heads
        mlp_dim: Hidden dimension of MLP
        attention_type: Type of attention ('baseline', 'ripple', 'hydra')
        dropout: Dropout rate
        emb_dropout: Embedding dropout rate
        dim_head: Dimension per attention head
        ripple_stages: Number of stages for RippleAttention
        hydra_branches: Number of branches for HydraAttention
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        num_classes=100,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        attention_type='ripple',
        dropout=0.1,
        emb_dropout=0.1,
        dim_head=64,
        ripple_stages=3,
        hydra_branches=4,
    ):
        super().__init__()

        # Calculate number of patches
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size
        self.dim = dim
        self.attention_type = attention_type

        # Patch embedding: split image into patches and project
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            if attention_type == 'baseline':
                block = BaselineTransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            elif attention_type == 'ripple':
                block = RippleTransformerBlock(dim, heads, dim_head, mlp_dim, dropout, ripple_stages)
            elif attention_type == 'hydra':
                block = HydraTransformerBlock(dim, heads, dim_head, mlp_dim, dropout, hydra_branches)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")

            self.transformer.append(block)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        """
        Forward pass.

        Args:
            img: Input images of shape (batch, channels, height, width)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Convert image to patch embeddings
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)

        # Extract class token and classify
        x = x[:, 0]
        return self.mlp_head(x)

    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_attention_complexity(self, img_size):
        """
        Calculate theoretical computational complexity.

        Returns:
            String describing the complexity
        """
        num_patches = (img_size // self.patch_size) ** 2

        if self.attention_type == 'baseline':
            # O(n²d) complexity
            ops = num_patches ** 2 * self.dim
            return f"O(n²d) ≈ {ops:,} operations per layer"
        else:
            # O(nd²) complexity for linear attention
            ops = num_patches * (self.dim ** 2)
            return f"O(nd²) ≈ {ops:,} operations per layer"

def build_vit(config):
    """
    Build Vision Transformer from config.

    Args:
        config: Configuration object

    Returns:
        VisionTransformer model
    """
    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        attention_type=config.attention_type,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        ripple_stages=config.ripple_stages,
        hydra_branches=config.hydra_branches,
    )

    return model
