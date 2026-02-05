"""
Region Contrastive Loss

Applies InfoNCE loss across all hierarchy levels, comparing G2L and L2G
features at each level.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .infonce import InfoNCELoss

logger = logging.getLogger(__name__)


class RegionContrastiveLoss(nn.Module):
    """
    Region-level contrastive loss across hierarchy levels.
    
    Computes InfoNCE loss between G2L and L2G features at each level,
    then combines them with optional per-level weights.
    
    Args:
        projection_head: ContrastiveProjectionHead for projecting features
        temperature: Temperature for InfoNCE
        level_weights: Optional dict of level_name -> weight. Default: equal weights.
    """
    
    def __init__(
        self,
        projection_head: nn.Module,
        temperature: float = 0.1,
        level_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.projection_head = projection_head
        self.temperature = temperature
        self.level_weights = level_weights
        self.infonce = InfoNCELoss(temperature=temperature)
        
        logger.debug(
            f"RegionContrastiveLoss: temp={temperature}, "
            f"level_weights={level_weights}"
        )
    
    def forward(
        self,
        g2l_features: Dict[str, torch.Tensor],
        l2g_features: Dict[str, torch.Tensor],
        offsets_by_level: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute region contrastive loss across all levels.
        
        Args:
            g2l_features: Dict of level_name -> (N_level, D) tensors
            l2g_features: Dict of level_name -> (N_level, D) tensors
            offsets_by_level: Dict of level_name -> (B+1,) offset tensors
            
        Returns:
            total_loss: Weighted sum of per-level losses
            per_level_losses: Dict of level_name -> loss tensor
        """
        per_level_losses = {}
        
        for level_name in g2l_features.keys():
            g2l = g2l_features[level_name]
            l2g = l2g_features[level_name]
            
            if g2l.shape[0] == 0:
                # Empty level, skip
                per_level_losses[level_name] = torch.tensor(
                    0.0, device=g2l.device, requires_grad=True
                )
                continue
            
            # Project features
            g2l_proj = self.projection_head(g2l)
            l2g_proj = self.projection_head(l2g)
            
            # Compute InfoNCE
            loss = self.infonce(g2l_proj, l2g_proj)
            per_level_losses[level_name] = loss
        
        # Compute weighted sum
        if self.level_weights is not None:
            weights = self.level_weights
        else:
            # Default: equal weights
            weights = {k: 1.0 for k in per_level_losses}
        
        total_loss = sum(
            weights.get(k, 1.0) * v 
            for k, v in per_level_losses.items()
        )
        
        # Normalize by number of levels if using equal weights
        if self.level_weights is None and len(per_level_losses) > 0:
            total_loss = total_loss / len(per_level_losses)
        
        return total_loss, per_level_losses
    
    def __repr__(self):
        return (
            f"RegionContrastiveLoss("
            f"temperature={self.temperature}, "
            f"level_weights={self.level_weights})"
        )
