"""
FeaturePropagator: Global-to-Local (G2L) Feature Propagation

Propagates contextual information from parent regions to child regions
in the hierarchical decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeaturePropagator(nn.Module):
    """
    Feature Propagator for Global-to-Local (G2L) processing.
    
    Enriches child region points with parent region's contextual features.
    Takes parent's encoded features and child's raw coordinates, combines
    them through an MLP to produce propagated features.
    
    Architecture:
        Input: parent_feat (D,) + child_coords (N, 3) → concatenate → (N, D+3)
        MLP: (D+3) → hidden → hidden → D
        Output: (N, D) enriched features for child points
    """
    
    def __init__(
        self,
        parent_dim: int = 96,
        coord_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 96,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize FeaturePropagator.
        
        Args:
            parent_dim: Dimension of parent region features
            coord_dim: Dimension of coordinates (default 3 for XYZ)
            hidden_dim: Hidden layer dimension
            out_dim: Output feature dimension
            num_layers: Number of MLP layers (minimum 2)
            use_layer_norm: Whether to use LayerNorm after each layer
            dropout: Dropout probability (0 = no dropout)
        """
        super().__init__()
        
        self.parent_dim = parent_dim
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = max(2, num_layers)
        self.use_layer_norm = use_layer_norm
        
        # Build MLP layers
        layers = []
        
        # Input layer: (parent_dim + coord_dim) → hidden_dim
        layers.append(nn.Linear(parent_dim + coord_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers: hidden_dim → hidden_dim
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer: hidden_dim → out_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.debug(f"Initialized FeaturePropagator: {parent_dim}+{coord_dim} → {out_dim}")
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        parent_feat: torch.Tensor,
        child_coords: torch.Tensor,
        normalize_coords: bool = False
    ) -> torch.Tensor:
        """
        Propagate parent features to child coordinates.
        
        Args:
            parent_feat: (D,) or (1, D) parent region features
            child_coords: (N, 3) child region point coordinates
            normalize_coords: Whether to normalize coordinates to [-1, 1]
        
        Returns:
            (N, out_dim) propagated features for child points
        """
        # Validate inputs
        if parent_feat.dim() == 1:
            parent_feat = parent_feat.unsqueeze(0)  # (D,) → (1, D)
        
        if parent_feat.shape[1] != self.parent_dim:
            raise ValueError(
                f"Expected parent_feat dim {self.parent_dim}, got {parent_feat.shape[1]}"
            )
        
        if child_coords.dim() != 2 or child_coords.shape[1] != self.coord_dim:
            raise ValueError(
                f"Expected child_coords shape (N, {self.coord_dim}), got {child_coords.shape}"
            )
        
        num_child_points = child_coords.shape[0]
        
        # Optionally normalize coordinates
        if normalize_coords:
            coords_min = child_coords.min(dim=0, keepdim=True)[0]
            coords_max = child_coords.max(dim=0, keepdim=True)[0]
            coords_range = coords_max - coords_min
            coords_range = torch.where(coords_range > 0, coords_range, torch.ones_like(coords_range))
            child_coords = (child_coords - coords_min) / coords_range * 2 - 1  # [-1, 1]
        
        # Expand parent features to match number of child points: (1, D) → (N, D)
        parent_feat_expanded = parent_feat.expand(num_child_points, -1)
        
        # Concatenate: (N, D) + (N, 3) → (N, D+3)
        combined = torch.cat([parent_feat_expanded, child_coords], dim=1)
        
        # Pass through MLP: (N, D+3) → (N, out_dim)
        propagated_feat = self.mlp(combined)
        
        return propagated_feat
    
    def __repr__(self) -> str:
        return (
            f"FeaturePropagator(parent_dim={self.parent_dim}, "
            f"coord_dim={self.coord_dim}, hidden_dim={self.hidden_dim}, "
            f"out_dim={self.out_dim}, layers={self.num_layers})"
        )


class MultiScalePropagator(nn.Module):
    """
    Multi-scale feature propagator that can handle different hierarchical levels.
    
    Uses separate propagators for each level or shares parameters.
    """
    
    def __init__(
        self,
        num_levels: int,
        parent_dim: int = 96,
        coord_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 96,
        shared_weights: bool = True
    ):
        """
        Initialize multi-scale propagator.
        
        Args:
            num_levels: Number of hierarchical levels
            parent_dim: Parent feature dimension
            coord_dim: Coordinate dimension
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            shared_weights: If True, share propagator across levels
        """
        super().__init__()
        
        self.num_levels = num_levels
        self.shared_weights = shared_weights
        
        if shared_weights:
            # Single propagator for all levels
            self.propagator = FeaturePropagator(
                parent_dim=parent_dim,
                coord_dim=coord_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim
            )
        else:
            # Separate propagator per level
            self.propagators = nn.ModuleList([
                FeaturePropagator(
                    parent_dim=parent_dim,
                    coord_dim=coord_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim
                )
                for _ in range(num_levels)
            ])
    
    def forward(
        self,
        parent_feat: torch.Tensor,
        child_coords: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """
        Propagate features at specific hierarchical level.
        
        Args:
            parent_feat: (D,) parent features
            child_coords: (N, 3) child coordinates
            level: Hierarchical level (0-indexed)
        
        Returns:
            (N, out_dim) propagated features
        """
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level {level} out of range [0, {self.num_levels})")
        
        if self.shared_weights:
            return self.propagator(parent_feat, child_coords)
        else:
            return self.propagators[level](parent_feat, child_coords)
    
    def __repr__(self) -> str:
        return (
            f"MultiScalePropagator(num_levels={self.num_levels}, "
            f"shared_weights={self.shared_weights})"
        )
