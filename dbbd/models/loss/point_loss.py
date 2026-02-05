"""
Point Contrastive Loss

Fine-grained contrastive loss at point level with random subsampling
for memory efficiency.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

from .infonce import InfoNCELoss

logger = logging.getLogger(__name__)


class PointContrastiveLoss(nn.Module):
    """
    Point-level contrastive loss with subsampling.
    
    Computing full (N, N) similarity matrix for 100k+ points is infeasible.
    This module randomly subsamples points before computing InfoNCE.
    
    Args:
        projection_head: ContrastiveProjectionHead for projecting features
        temperature: Temperature for InfoNCE
        num_samples: Maximum number of points to sample (default: 4096)
    """
    
    def __init__(
        self,
        projection_head: nn.Module,
        temperature: float = 0.1,
        num_samples: int = 4096
    ):
        super().__init__()
        
        self.projection_head = projection_head
        self.temperature = temperature
        self.num_samples = num_samples
        self.infonce = InfoNCELoss(temperature=temperature)
        
        logger.debug(
            f"PointContrastiveLoss: temp={temperature}, "
            f"num_samples={num_samples}"
        )
    
    def forward(
        self,
        point_feats_g2l: torch.Tensor,
        point_feats_l2g: torch.Tensor,
        point_offset: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute point contrastive loss with subsampling.
        
        Args:
            point_feats_g2l: (total_points, D) G2L point features
            point_feats_l2g: (total_points, D) L2G point features
            point_offset: (B+1,) scene offsets (not used for sampling)
            
        Returns:
            Scalar loss value
        """
        N = point_feats_g2l.shape[0]
        
        if N == 0:
            return torch.tensor(0.0, device=point_feats_g2l.device, requires_grad=True)
        
        # Subsample if needed
        if N > self.num_samples:
            # Random permutation for subsampling
            indices = torch.randperm(N, device=point_feats_g2l.device)[:self.num_samples]
            point_feats_g2l = point_feats_g2l[indices]
            point_feats_l2g = point_feats_l2g[indices]
        
        # Project features
        g2l_proj = self.projection_head(point_feats_g2l)
        l2g_proj = self.projection_head(point_feats_l2g)
        
        # Compute InfoNCE
        loss = self.infonce(g2l_proj, l2g_proj)
        
        return loss
    
    def __repr__(self):
        return (
            f"PointContrastiveLoss("
            f"temperature={self.temperature}, "
            f"num_samples={self.num_samples})"
        )
