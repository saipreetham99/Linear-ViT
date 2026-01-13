import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class HydraAttention(nn.Module):
    """
    HydraAttention: Multi-branch linear attention with shared key-value projections.

    Inspired by the mythical Hydra with multiple heads, this mechanism splits
    queries into multiple branches that attend to different feature subspaces.
    By sharing K and V projections across branches and using kernel approximation,
    we achieve O(n) complexity.

    Args:
        dim: Model dimension
        heads: Number of attention heads per branch
        dim_head: Dimension per head
        dropout: Dropout rate
        num_branches: Number of Hydra branches
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_branches=4):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.num_branches = num_branches

        # Shared key and value projections (Hydra body)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # Multiple query branches (Hydra heads)
        self.branch_queries = nn.ModuleList([
            nn.Linear(dim, inner_dim, bias=False)
            for _ in range(num_branches)
        ])

        # Branch-specific feature transformations
        self.branch_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_head, dim_head),
                nn.GELU(),
                nn.LayerNorm(dim_head)
            ) for _ in range(num_branches)
        ])

        # Feature map for kernel approximation (using ReLU kernel)
        # phi(x) = [x_+, x_-] where x_+ = max(0, x), x_- = max(0, -x)
        self.feature_map = lambda x: torch.cat([F.relu(x), F.relu(-x)], dim=-1)

        # Branch fusion layer
        self.branch_fusion = nn.Sequential(
            nn.Linear(inner_dim * num_branches, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.GELU()
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def linear_attention(self, q, k, v):
        """
        Compute linear attention using kernel trick.

        Computes attention as phi(Q)(phi(K)^T V) instead of softmax(QK^T)V,
        reducing complexity from O(n²d) to O(nd²).

        Args:
            q: Query tensor (batch, heads, seq_len, dim_head)
            k: Key tensor (batch, heads, seq_len, dim_head)
            v: Value tensor (batch, heads, seq_len, dim_head)

        Returns:
            Attention output (batch, heads, seq_len, dim_head)
        """
        # Apply feature map
        q = self.feature_map(q)  # (b, h, n, 2*d)
        k = self.feature_map(k)  # (b, h, n, 2*d)

        # Normalize
        q = q / math.sqrt(q.shape[-1])

        # Compute K^T V (this is O(n d² ) instead of O(n²d))
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)

        # Compute Q(K^T V)
        out = torch.einsum('bhnd,bhdm->bhnm', q, kv)

        # Normalize by query-key sums to maintain probability distribution
        k_sum = k.sum(dim=2, keepdim=True)  # (b, h, 1, 2*d)
        normalizer = torch.einsum('bhnd,bhnd->bhn', q, k_sum.expand_as(q))
        out = out / (normalizer.unsqueeze(-1) + 1e-6)

        return out

    def forward(self, x):
        """
        Forward pass of HydraAttention.

        Args:
            x: Input tensor of shape (batch, num_patches, dim)

        Returns:
            Output tensor of shape (batch, num_patches, dim)
        """
        batch, n, _ = x.shape

        # Shared K and V projections
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape to multi-head
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # Process each branch independently
        branch_outputs = []
        for i in range(self.num_branches):
            # Branch-specific query projection
            q_branch = self.branch_queries[i](x)
            q_branch = rearrange(q_branch, 'b n (h d) -> b h n d', h=self.heads)

            # Apply branch-specific transformation
            q_branch = self.branch_transforms[i](q_branch)

            # Compute linear attention for this branch
            branch_out = self.linear_attention(q_branch, k, v)

            branch_outputs.append(branch_out)

        # Concatenate all branch outputs
        out = torch.cat(branch_outputs, dim=1)  # (b, h*num_branches, n, d)

        # Reshape for fusion
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Fuse branches
        out = self.branch_fusion(out)

        # Final output projection
        out = self.to_out(out)

        return out

class HydraFeedForward(nn.Module):
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

class HydraTransformerBlock(nn.Module):
    """Transformer block with HydraAttention"""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, num_branches=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HydraAttention(dim, heads, dim_head, dropout, num_branches)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = HydraFeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        # Pre-normalization residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
