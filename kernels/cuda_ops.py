"""
Custom CUDA kernels for optimized linear attention operations.

These kernels optimize memory access patterns and leverage GPU parallelism
for the key operations in RippleAttention and HydraAttention.
"""

import torch
import torch.nn as nn
import os

# Try to load custom CUDA extension if available
try:
    from torch.utils.cpp_extension import load

    # Build path for CUDA source
    kernel_path = os.path.dirname(os.path.abspath(__file__))

    # Load CUDA extension (JIT compilation)
    cuda_linear_attn = load(
        name='cuda_linear_attn',
        sources=[
            os.path.join(kernel_path, 'linear_attention_kernel.cu'),
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False
    )
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Custom CUDA kernels not available: {e}")
    print("Falling back to PyTorch native operations")
    CUDA_AVAILABLE = False


class LinearAttentionCUDA(torch.autograd.Function):
    """
    Custom CUDA implementation of linear attention.

    Optimizes the computation of phi(Q)(phi(K)^T V) using:
    - Shared memory for frequently accessed data
    - Warp-level primitives for parallel reductions
    - Memory coalescing for efficient global memory access
    """

    @staticmethod
    def forward(ctx, Q, K, V):
        """
        Forward pass: compute linear attention.

        Args:
            Q: Query tensor (batch, heads, seq_len, dim)
            K: Key tensor (batch, heads, seq_len, dim)
            V: Value tensor (batch, heads, seq_len, dim)

        Returns:
            Output tensor (batch, heads, seq_len, dim)
        """
        if CUDA_AVAILABLE and Q.is_cuda:
            # Use custom CUDA kernel
            output = cuda_linear_attn.forward(Q, K, V)
            ctx.save_for_backward(Q, K, V)
            return output
        else:
            # Fallback to PyTorch implementation
            return linear_attention_pytorch(Q, K, V)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradients w.r.t. Q, K, V
        """
        Q, K, V = ctx.saved_tensors

        if CUDA_AVAILABLE and grad_output.is_cuda:
            # Use custom CUDA kernel for backward pass
            grad_Q, grad_K, grad_V = cuda_linear_attn.backward(
                grad_output.contiguous(), Q, K, V
            )
            return grad_Q, grad_K, grad_V
        else:
            # Fallback to PyTorch autograd
            Q = Q.requires_grad_(True)
            K = K.requires_grad_(True)
            V = V.requires_grad_(True)

            output = linear_attention_pytorch(Q, K, V)
            output.backward(grad_output)

            return Q.grad, K.grad, V.grad


def linear_attention_pytorch(Q, K, V):
    """
    PyTorch implementation of linear attention (fallback).

    Computes attention as phi(Q)(phi(K)^T V) for O(nd²) complexity.

    Args:
        Q: Query tensor (batch, heads, seq_len, dim)
        K: Key tensor (batch, heads, seq_len, dim)
        V: Value tensor (batch, heads, seq_len, dim)

    Returns:
        Output tensor (batch, heads, seq_len, dim)
    """
    # Apply feature map (ReLU kernel approximation)
    Q_pos = torch.relu(Q)
    Q_neg = torch.relu(-Q)
    Q_feature = torch.cat([Q_pos, Q_neg], dim=-1)

    K_pos = torch.relu(K)
    K_neg = torch.relu(-K)
    K_feature = torch.cat([K_pos, K_neg], dim=-1)

    # Compute K^T V first (d x d)
    KV = torch.einsum('bhnd,bhnm->bhdm', K_feature, V)

    # Then compute Q(K^T V)
    output = torch.einsum('bhnd,bhdm->bhnm', Q_feature, KV)

    # Normalization
    K_sum = K_feature.sum(dim=2, keepdim=True)
    normalizer = torch.einsum('bhnd,bhmd->bhn', Q_feature, K_sum.transpose(-1, -2))
    output = output / (normalizer.unsqueeze(-1) + 1e-6)

    return output


class FastLinearAttention(nn.Module):
    """
    Wrapper module for optimized linear attention.

    Automatically uses CUDA kernels if available, otherwise falls back
    to PyTorch implementation.
    """

    def __init__(self, use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        if self.use_cuda:
            print("Using custom CUDA kernels for linear attention")
        else:
            print("Using PyTorch implementation for linear attention")

    def forward(self, Q, K, V):
        """
        Forward pass.

        Args:
            Q: Query tensor (batch, heads, seq_len, dim)
            K: Key tensor (batch, heads, seq_len, dim)
            V: Value tensor (batch, heads, seq_len, dim)

        Returns:
            Output tensor (batch, heads, seq_len, dim)
        """
        if self.use_cuda:
            return LinearAttentionCUDA.apply(Q, K, V)
        else:
            return linear_attention_pytorch(Q, K, V)


def benchmark_kernels(batch_size=32, heads=8, seq_len=64, dim=64, num_iterations=100):
    """
    Benchmark CUDA kernels vs PyTorch implementation.

    Args:
        batch_size: Batch size
        heads: Number of attention heads
        seq_len: Sequence length
        dim: Feature dimension
        num_iterations: Number of iterations for timing

    Returns:
        Dictionary with timing results
    """
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate random inputs
    Q = torch.randn(batch_size, heads, seq_len, dim, device=device)
    K = torch.randn(batch_size, heads, seq_len, dim, device=device)
    V = torch.randn(batch_size, heads, seq_len, dim, device=device)

    results = {}

    # Benchmark PyTorch implementation
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(num_iterations):
        _ = linear_attention_pytorch(Q, K, V)
    torch.cuda.synchronize() if device == 'cuda' else None
    pytorch_time = (time.time() - start) / num_iterations
    results['pytorch'] = pytorch_time

    # Benchmark CUDA implementation if available
    if CUDA_AVAILABLE and device == 'cuda':
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = LinearAttentionCUDA.apply(Q, K, V)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / num_iterations
        results['cuda'] = cuda_time
        results['speedup'] = pytorch_time / cuda_time

    return results


if __name__ == '__main__':
    # Test and benchmark kernels
    print("Testing custom CUDA kernels...")
    print(f"CUDA available: {CUDA_AVAILABLE}")

    if torch.cuda.is_available():
        print("\nRunning benchmarks...")
        results = benchmark_kernels()
        print(f"PyTorch time: {results['pytorch']*1000:.3f} ms")
        if 'cuda' in results:
            print(f"CUDA time: {results['cuda']*1000:.3f} ms")
            print(f"Speedup: {results['speedup']:.2f}x")
