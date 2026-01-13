import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention with O(n²) complexity.

    This is the baseline attention mechanism used in the original
    Vision Transformer paper.

    Args:
        dim: Model dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        dropout: Dropout rate
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass with standard O(n²) attention.

        Args:
            x: Input tensor of shape (batch, num_patches, dim)

        Returns:
            Output tensor of shape (batch, num_patches, dim)
        """
        b, n, _ = x.shape

        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Compute attention scores: Q K^T (this is O(n²))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply softmax
        attn = F.softmax(dots, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    """Standard feed-forward network"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class BaselineTransformerBlock(nn.Module):
    """Standard Transformer block with multi-head attention"""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
