"""
PointCloudEncoder

Wraps a point cloud backbone to encode regions into features.
Shared across both G2L and L2G branches.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


class PointNetBackbone(nn.Module):
    """
    Simple PointNet-style backbone for point cloud encoding.
    
    Uses shared MLPs followed by max pooling for permutation invariance.
    
    Args:
        input_dim: Input feature dimension (coords + features)
        hidden_dims: List of hidden layer dimensions
        output_dim: Output feature dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 128],
        output_dim: int = 96
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build shared MLP
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared MLP.
        
        Args:
            x: (M, input_dim) point features
            
        Returns:
            (M, output_dim) transformed point features
        """
        return self.mlp(x)


class PointCloudEncoder(nn.Module):
    """
    Point cloud encoder that produces region-level and point-level features.
    
    Handles coordinate centering internally and supports variable-size inputs.
    This encoder is SHARED between G2L and L2G branches.
    
    Args:
        input_dim: Input feature dimension (after projection)
        output_dim: Output feature dimension
        backbone: Backbone type ('pointnet' or 'dgcnn')
        pooling: Pooling method ('max' or 'mean')
        hidden_dims: List of hidden dimensions for backbone
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 96,
        backbone: Literal['pointnet'] = 'pointnet',
        pooling: Literal['max', 'mean'] = 'max',
        hidden_dims: list = [64, 128]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = pooling
        
        # Coordinate dimension (XYZ)
        self.coord_dim = 3
        
        # Total input to backbone: centered_coords + features
        backbone_input_dim = self.coord_dim + input_dim
        
        # Initialize backbone
        if backbone == 'pointnet':
            self.backbone = PointNetBackbone(
                input_dim=backbone_input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        logger.debug(
            f"PointCloudEncoder initialized: input_dim={input_dim}, "
            f"output_dim={output_dim}, backbone={backbone}"
        )
    
    def forward(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor,
        center: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a region.
        
        Args:
            coords: (M, 3) point coordinates
            feats: (M, input_dim) point features (already projected)
            center: (3,) optional center for normalization
                   If None, use centroid of coords
                   
        Returns:
            region_feat: (output_dim,) pooled region feature
            point_feats: (M, output_dim) per-point features
        """
        if coords.dim() != 2 or coords.shape[-1] != 3:
            raise ValueError(f"Expected coords shape (M, 3), got {coords.shape}")
        if feats.dim() != 2:
            raise ValueError(f"Expected 2D features, got {feats.dim()}D")
        if coords.shape[0] != feats.shape[0]:
            raise ValueError(
                f"Coord/feat count mismatch: {coords.shape[0]} vs {feats.shape[0]}"
            )
        if feats.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feat dim {self.input_dim}, got {feats.shape[-1]}"
            )
        
        # Step 1: Center coordinates
        if center is None:
            center = coords.mean(dim=0)
        centered_coords = coords - center.unsqueeze(0)
        
        # Step 2: Combine centered coords with features
        encoder_input = torch.cat([centered_coords, feats], dim=-1)
        
        # Step 3: Forward through backbone
        point_feats = self.backbone(encoder_input)  # (M, output_dim)
        
        # Step 4: Pool for region feature
        if self.pooling == 'max':
            region_feat = point_feats.max(dim=0)[0]
        else:  # mean
            region_feat = point_feats.mean(dim=0)
        
        return region_feat, point_feats
    
    def __repr__(self):
        return (
            f"PointCloudEncoder(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, pooling={self.pooling})"
        )
