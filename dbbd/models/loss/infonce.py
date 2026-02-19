"""
InfoNCE Loss

Core contrastive loss function using PyTorch's cross_entropy for
mathematical correctness and numerical stability.

Mathematical formulation:
    L = -(1/N) * Σ_j log( exp(S_jj/τ) / Σ_k exp(S_jk/τ) )

Where:
    - S_jj = positive pair similarity (same region, different branches)
    - S_jk = negative pair similarities (different regions)
    - τ = temperature hyperparameter

This is equivalent to cross-entropy with labels = arange(N).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)


class _GatherWithGrad(torch.autograd.Function):
    """Gather tensors from all GPUs with gradient support."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        ctx.rank = dist.get_rank()
        ctx.world_size = world_size
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grads):
        return grads[ctx.rank]


def gather_with_grad(tensor):
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    gathered = _GatherWithGrad.apply(tensor)
    return torch.cat(gathered, dim=0)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss.

    Uses PyTorch's F.cross_entropy which is mathematically equivalent to
    InfoNCE when labels are the diagonal indices (arange(N)).

    When running with DDP, gathers embeddings from all GPUs so that
    negatives span the full global batch.

    Args:
        temperature: Temperature hyperparameter for softmax scaling.
            Lower values make the distribution sharper (harder negatives).
            Common values: 0.07 (MoCo), 0.1 (SimCLR), 0.5 (PointContrast).

    Input Requirements:
        - query and key must be L2-normalized
        - query.shape == key.shape == (N, D)
        - Diagonal entries (query[i], key[i]) are positive pairs
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.temperature = temperature
        logger.debug(f"InfoNCELoss initialized with temperature={temperature}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            query: Query features (N, D), should be L2-normalized
            key: Key features (N, D), should be L2-normalized
            offsets: Optional scene offsets (not used in current implementation)

        Returns:
            Scalar loss value
        """
        N = query.shape[0]

        if N == 0:
            return torch.tensor(0.0, device=query.device, requires_grad=True)

        if query.shape != key.shape:
            raise ValueError(
                f"Shape mismatch: query {query.shape} vs key {key.shape}"
            )

        all_key = gather_with_grad(key)

        # Similarity: local queries against all keys
        # Single GPU: (N, N), Multi-GPU: (N, N*world_size)
        sim = torch.mm(query, all_key.T) / self.temperature

        # Labels: positive for query[i] is at position (rank * N + i) in the gathered keys
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            labels = torch.arange(N, device=query.device, dtype=torch.long) + rank * N
        else:
            labels = torch.arange(N, device=query.device, dtype=torch.long)

        loss = F.cross_entropy(sim, labels)

        return loss

    def __repr__(self):
        return f"InfoNCELoss(temperature={self.temperature})"
