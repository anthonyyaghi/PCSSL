"""
FeatureAggregator: Local-to-Global (L2G) Feature Aggregation

Aggregates features from multiple child regions into a unified parent
representation in the hierarchical decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


class FeatureAggregator(nn.Module):
    """
    Feature Aggregator for Local-to-Global (L2G) processing.
    
    Combines features from multiple child regions into a single parent
    feature vector. Supports multiple aggregation strategies:
    - Max pooling: Takes element-wise maximum across children
    - Mean pooling: Takes element-wise mean across children
    - Attention: Learns weighted combination of children
    
    Must be permutation-invariant (order of children doesn't matter).
    """
    
    def __init__(
        self,
        feat_dim: int = 96,
        mode: Literal['max', 'mean', 'attention'] = 'max',
        use_pre_mlp: bool = True,
        use_spatial: bool = False,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize FeatureAggregator.
        
        Args:
            feat_dim: Feature dimension
            mode: Aggregation mode ('max', 'mean', or 'attention')
            use_pre_mlp: Whether to process features with MLP before pooling
            use_spatial: Whether to incorporate spatial context
            hidden_dim: Hidden dimension for MLP (default: same as feat_dim)
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.mode = mode
        self.use_pre_mlp = use_pre_mlp
        self.use_spatial = use_spatial
        self.hidden_dim = hidden_dim or feat_dim
        
        # Optional pre-processing MLP
        if use_pre_mlp:
            self.pre_mlp = nn.Sequential(
                nn.Linear(feat_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, feat_dim)
            )
        
        # Attention mechanism (if mode='attention')
        if mode == 'attention':
            self.attention = AttentionAggregation(
                feat_dim=feat_dim,
                use_spatial=use_spatial
            )
        
        # Initialize weights
        self._init_weights()
        
        logger.debug(f"Initialized FeatureAggregator: mode={mode}, dim={feat_dim}")
    
    def _init_weights(self):
        """Initialize network weights."""
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
        child_features: torch.Tensor,
        child_coords: Optional[torch.Tensor] = None,
        parent_coord: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate child region features into parent feature.
        
        Args:
            child_features: (M, D) features from M child regions
            child_coords: (M, 3) optional center coordinates of children
            parent_coord: (3,) optional center coordinate of parent
        
        Returns:
            (D,) aggregated feature for parent region
        """
        # Validate inputs
        if child_features.dim() != 2:
            raise ValueError(
                f"Expected child_features shape (M, D), got {child_features.shape}"
            )
        
        if child_features.shape[1] != self.feat_dim:
            raise ValueError(
                f"Expected feat_dim {self.feat_dim}, got {child_features.shape[1]}"
            )
        
        num_children = child_features.shape[0]
        
        if num_children == 0:
            raise ValueError("Cannot aggregate zero children")
        
        # Optional spatial context validation
        if self.use_spatial and (child_coords is None or parent_coord is None):
            logger.warning(
                "Spatial context requested but coords not provided, "
                "using features only"
            )
        
        # Optional pre-processing
        features = child_features
        if self.use_pre_mlp:
            features = self.pre_mlp(features)  # (M, D)
        
        # Aggregation based on mode
        if self.mode == 'max':
            # Element-wise maximum: (M, D) → (D,)
            aggregated, _ = torch.max(features, dim=0)
        
        elif self.mode == 'mean':
            # Element-wise mean: (M, D) → (D,)
            aggregated = torch.mean(features, dim=0)
        
        elif self.mode == 'attention':
            # Attention-based weighted aggregation
            aggregated = self.attention(
                features,
                child_coords=child_coords,
                parent_coord=parent_coord
            )
        
        else:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")
        
        return aggregated
    
    def __repr__(self) -> str:
        return (
            f"FeatureAggregator(mode={self.mode}, feat_dim={self.feat_dim}, "
            f"pre_mlp={self.use_pre_mlp}, spatial={self.use_spatial})"
        )


class AttentionAggregation(nn.Module):
    """
    Attention-based aggregation mechanism.
    
    Learns to weight child features based on their importance,
    optionally incorporating spatial relationships.
    """
    
    def __init__(
        self,
        feat_dim: int,
        use_spatial: bool = False,
        num_heads: int = 4
    ):
        """
        Initialize attention aggregation.
        
        Args:
            feat_dim: Feature dimension
            use_spatial: Whether to use spatial context
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.use_spatial = use_spatial
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        
        if feat_dim % num_heads != 0:
            raise ValueError(f"feat_dim {feat_dim} must be divisible by num_heads {num_heads}")
        
        # Query: learnable global query vector
        self.query = nn.Parameter(torch.randn(1, feat_dim))
        
        # Key and Value projections
        self.key_proj = nn.Linear(feat_dim, feat_dim)
        self.value_proj = nn.Linear(feat_dim, feat_dim)
        
        # Optional spatial encoding
        if use_spatial:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(3, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, feat_dim)
            )
        
        # Output projection
        self.out_proj = nn.Linear(feat_dim, feat_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        child_coords: Optional[torch.Tensor] = None,
        parent_coord: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention-weighted aggregation.
        
        Args:
            features: (M, D) child features
            child_coords: (M, 3) child coordinates
            parent_coord: (3,) parent coordinate
        
        Returns:
            (D,) aggregated feature
        """
        M, D = features.shape
        
        # Incorporate spatial information if available
        if self.use_spatial and child_coords is not None and parent_coord is not None:
            # Compute relative positions
            relative_pos = child_coords - parent_coord.unsqueeze(0)  # (M, 3)
            spatial_feat = self.spatial_encoder(relative_pos)  # (M, D)
            features = features + spatial_feat  # Residual connection
        
        # Project to keys and values
        keys = self.key_proj(features)  # (M, D)
        values = self.value_proj(features)  # (M, D)
        
        # Reshape for multi-head attention
        keys = keys.view(M, self.num_heads, self.head_dim)  # (M, H, D/H)
        values = values.view(M, self.num_heads, self.head_dim)  # (M, H, D/H)
        query = self.query.view(1, self.num_heads, self.head_dim)  # (1, H, D/H)
        
        # Compute attention scores: (1, H, D/H) × (M, H, D/H)^T → (1, H, M)
        scores = torch.einsum('qhd,mhd->hm', query, keys) / (self.head_dim ** 0.5)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=1)  # (H, M)
        
        # Weighted sum of values: (H, M) × (M, H, D/H) → (H, D/H)
        aggregated = torch.einsum('hm,mhd->hd', attn_weights, values)
        
        # Reshape and project: (H, D/H) → (H*D/H) → (D,)
        aggregated = aggregated.reshape(-1)  # (D,)
        aggregated = self.out_proj(aggregated)
        
        return aggregated


class MultiScaleAggregator(nn.Module):
    """
    Multi-scale feature aggregator for different hierarchical levels.
    
    Can use separate aggregators per level or share parameters.
    """
    
    def __init__(
        self,
        num_levels: int,
        feat_dim: int = 96,
        mode: Literal['max', 'mean', 'attention'] = 'max',
        shared_weights: bool = True
    ):
        """
        Initialize multi-scale aggregator.
        
        Args:
            num_levels: Number of hierarchical levels
            feat_dim: Feature dimension
            mode: Aggregation mode
            shared_weights: If True, share aggregator across levels
        """
        super().__init__()
        
        self.num_levels = num_levels
        self.shared_weights = shared_weights
        
        if shared_weights:
            # Single aggregator for all levels
            self.aggregator = FeatureAggregator(feat_dim=feat_dim, mode=mode)
        else:
            # Separate aggregator per level
            self.aggregators = nn.ModuleList([
                FeatureAggregator(feat_dim=feat_dim, mode=mode)
                for _ in range(num_levels)
            ])
    
    def forward(
        self,
        child_features: torch.Tensor,
        level: int,
        child_coords: Optional[torch.Tensor] = None,
        parent_coord: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate features at specific hierarchical level.
        
        Args:
            child_features: (M, D) child features
            level: Hierarchical level
            child_coords: (M, 3) optional child coordinates
            parent_coord: (3,) optional parent coordinate
        
        Returns:
            (D,) aggregated feature
        """
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level {level} out of range [0, {self.num_levels})")
        
        if self.shared_weights:
            return self.aggregator(child_features, child_coords, parent_coord)
        else:
            return self.aggregators[level](child_features, child_coords, parent_coord)
    
    def __repr__(self) -> str:
        return (
            f"MultiScaleAggregator(num_levels={self.num_levels}, "
            f"shared_weights={self.shared_weights})"
        )
