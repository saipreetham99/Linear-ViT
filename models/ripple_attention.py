import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class RippleAttention(nn.Module):
    """
    RippleAttention: Linear-complexity attention mechanism using spatial ripple pooling.

    The key idea is to reduce the spatial dimension through progressive pooling,
    compute attention in the compressed space, and then expand back with learned
    ripple propagation. This reduces complexity from O(n²) to O(n).

    Args:
        dim: Model dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        dropout: Dropout rate
        num_stages: Number of ripple pooling stages
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_stages=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.num_stages = num_stages

        # Query, Key, Value projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # Ripple pooling layers - progressive spatial compression
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_head, dim_head),
                nn.LayerNorm(dim_head),
                nn.GELU()
            ) for _ in range(num_stages)
        ])

        # Ripple expansion layers - learned propagation
        self.expand_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_head, dim_head),
                nn.LayerNorm(dim_head),
                nn.GELU()
            ) for _ in range(num_stages)
        ])

        # Feature map approximation with ELU kernel
        self.feature_map = lambda x: F.elu(x) + 1

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of RippleAttention.

        Args:
            x: Input tensor of shape (batch, num_patches, dim)

        Returns:
            Output tensor of shape (batch, num_patches, dim)
        """
        batch, n, _ = x.shape

        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape to multi-head
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # Apply feature map for kernel approximation
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Ripple pooling: progressively compress K and V
        k_compressed = k
        v_compressed = v

        pooling_factor = 2
        for pool_layer in self.pool_layers:
            seq_len = k_compressed.shape[2]

            # Only pool if we have enough sequence length
            if seq_len > 1:
                # Handle sequences that aren't divisible by pooling factor
                remainder = seq_len % pooling_factor
                if remainder != 0:
                    # Pad to make divisible
                    pad_size = pooling_factor - remainder
                    k_compressed = F.pad(k_compressed, (0, 0, 0, pad_size))
                    v_compressed = F.pad(v_compressed, (0, 0, 0, pad_size))
                    seq_len = k_compressed.shape[2]

                # Average pooling
                k_compressed = k_compressed.reshape(
                    k_compressed.shape[0], k_compressed.shape[1],
                    seq_len // pooling_factor, pooling_factor, k_compressed.shape[3]
                )
                k_compressed = pool_layer(k_compressed.mean(dim=3))

                v_compressed = v_compressed.reshape(
                    v_compressed.shape[0], v_compressed.shape[1],
                    seq_len // pooling_factor, pooling_factor, v_compressed.shape[3]
                )
                v_compressed = v_compressed.mean(dim=3)

        # Linear attention: Q(K^T V) instead of (QK^T)V
        # This changes complexity from O(n²d) to O(nd²)
        k_compressed = k_compressed / math.sqrt(k_compressed.shape[2])

        # Compute K^T V first (d x d matrix instead of n x n)
        kv = torch.einsum('bhnd,bhne->bhde', k_compressed, v_compressed)

        # Then compute Q(K^T V)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)

        # Normalize by sum of attention weights
        k_sum = k_compressed.sum(dim=2, keepdim=True)
        q_k_sum = torch.einsum('bhnd,bhnd->bhn', q, k_sum)
        out = out / (q_k_sum.unsqueeze(-1) + 1e-6)

        # Ripple expansion: propagate compressed attention back to full resolution
        for expand_layer in reversed(self.expand_layers):
            out = expand_layer(out)

        # Reshape back to batch format
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Final projection
        out = self.to_out(out)

        return out

class RippleFeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
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

class RippleTransformerBlock(nn.Module):
    """Transformer block with RippleAttention"""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, num_stages=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RippleAttention(dim, heads, dim_head, dropout, num_stages)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = RippleFeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        # Pre-normalization residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
