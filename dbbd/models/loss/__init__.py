"""
DBBD Contrastive Loss Module

Phase 4: Implements contrastive learning losses for bidirectional 
hierarchical encoder training.

Components:
- ContrastiveProjectionHead: Projects features to contrastive space
- InfoNCELoss: Core InfoNCE contrastive loss
- RegionContrastiveLoss: Multi-level region contrastive loss
- PointContrastiveLoss: Point-level contrastive loss with subsampling
- DBBDContrastiveLoss: Main combined loss module
"""

from .projection import ContrastiveProjectionHead
from .infonce import InfoNCELoss
from .region_loss import RegionContrastiveLoss
from .point_loss import PointContrastiveLoss
from .dbbd_loss import DBBDContrastiveLoss

__all__ = [
    'ContrastiveProjectionHead',
    'InfoNCELoss',
    'RegionContrastiveLoss',
    'PointContrastiveLoss',
    'DBBDContrastiveLoss',
]
