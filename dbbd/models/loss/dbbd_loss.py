"""
DBBD Contrastive Loss

Main loss module combining region and point contrastive losses.

From paper Section 3.6:
    L_total = α * L_region + β * L_point
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .projection import ContrastiveProjectionHead
from .region_loss import RegionContrastiveLoss
from .point_loss import PointContrastiveLoss

logger = logging.getLogger(__name__)


class DBBDContrastiveLoss(nn.Module):
    """
    Combined DBBD contrastive loss.
    
    Combines region-level and point-level contrastive losses with
    configurable weights.
    
    Args:
        encoder_dim: Dimension of encoder output features
        projection_dim: Dimension of projected features for contrastive learning
        projection_hidden_dim: Hidden dimension of projection head
        temperature: Temperature for InfoNCE
        alpha: Weight for region loss
        beta: Weight for point loss
        level_weights: Optional per-level weights for region loss
        point_num_samples: Number of points to sample for point loss
    
    Output:
        total_loss: α * L_region + β * L_point
        loss_dict: Dict with 'total', 'region', 'point', and per-level losses
    """
    
    def __init__(
        self,
        encoder_dim: int = 96,
        projection_dim: int = 128,
        projection_hidden_dim: int = 96,
        temperature: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.5,
        level_weights: Optional[Dict[str, float]] = None,
        point_num_samples: int = 4096
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # Separate projection heads for region and point (different semantics)
        self.region_projection = ContrastiveProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
            num_layers=2
        )
        
        self.point_projection = ContrastiveProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
            num_layers=2
        )
        
        # Loss modules
        self.region_loss = RegionContrastiveLoss(
            projection_head=self.region_projection,
            temperature=temperature,
            level_weights=level_weights
        )
        
        self.point_loss = PointContrastiveLoss(
            projection_head=self.point_projection,
            temperature=temperature,
            num_samples=point_num_samples
        )
        
        logger.info(
            f"DBBDContrastiveLoss: encoder_dim={encoder_dim}, "
            f"projection_dim={projection_dim}, temp={temperature}, "
            f"alpha={alpha}, beta={beta}"
        )
    
    def forward(
        self,
        encoder_output: Dict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined contrastive loss.
        
        Args:
            encoder_output: Phase 3 encoder output dict with:
                - 'g2l': Dict of level -> (N, D) features
                - 'l2g': Dict of level -> (N, D) features
                - 'offsets_by_level': Dict of level -> (B+1,) offsets
                - 'point_feats_g2l': (total_points, D) features
                - 'point_feats_l2g': (total_points, D) features
                - 'point_offset': (B+1,) offsets
                
        Returns:
            total_loss: Scalar combined loss
            loss_dict: Dict with all losses for logging
        """
        # Region loss
        region_total, region_per_level = self.region_loss(
            encoder_output['g2l'],
            encoder_output['l2g'],
            encoder_output['offsets_by_level']
        )
        
        # Point loss
        point_total = self.point_loss(
            encoder_output['point_feats_g2l'],
            encoder_output['point_feats_l2g'],
            encoder_output['point_offset']
        )
        
        # Combined loss
        total_loss = self.alpha * region_total + self.beta * point_total
        
        # Build loss dict for logging
        loss_dict = {
            'total': total_loss,
            'region': region_total,
            'point': point_total,
        }
        
        # Add per-level region losses
        for level_name, level_loss in region_per_level.items():
            loss_dict[f'region_{level_name}'] = level_loss
        
        return total_loss, loss_dict
    
    def __repr__(self):
        return (
            f"DBBDContrastiveLoss(\n"
            f"  encoder_dim={self.encoder_dim},\n"
            f"  projection_dim={self.projection_dim},\n"
            f"  temperature={self.temperature},\n"
            f"  alpha={self.alpha},\n"
            f"  beta={self.beta}\n"
            f")"
        )
