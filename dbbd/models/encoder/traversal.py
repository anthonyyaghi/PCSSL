"""
Hierarchical Traversal Modules

G2L (Global-to-Local) and L2G (Local-to-Global) traversal implementations.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, List, Optional
from collections import defaultdict
import logging

from ..utils.hierarchy import Region

logger = logging.getLogger(__name__)


def _pool(point_feats: torch.Tensor, mode: str = 'max') -> torch.Tensor:
    if mode == 'max':
        return point_feats.max(dim=0)[0]
    return point_feats.mean(dim=0)


class L2GTraversal(nn.Module):
    """
    Local-to-Global (L2G) traversal.
    
    Bottom-up processing where:
    - Leaves are ENCODED by the backbone
    - Non-leaves are PURELY AGGREGATED from children (no encoding)
    - Traversal order: post-order (children before parent)
    
    Args:
        projection: ProjectionMLP for raw features
        encoder: PointCloudEncoder for leaf regions
        aggregator: FeatureAggregator for non-leaf regions
        use_spatial_context: Whether aggregator uses spatial context
    """
    
    def __init__(
        self,
        projection: nn.Module,
        encoder: nn.Module,
        aggregator: nn.Module,
        use_spatial_context: bool = True,
        use_checkpoint: bool = False,
        pooling: str = 'max'
    ):
        super().__init__()

        self.projection = projection
        self.encoder = encoder
        self.aggregator = aggregator
        self.use_spatial_context = use_spatial_context
        self.use_checkpoint = use_checkpoint
        self.pooling = pooling

    def forward(
        self,
        root: Region,
        coords: torch.Tensor,
        feats: torch.Tensor,
        precomputed_feats: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Traverse hierarchy bottom-up.

        Args:
            root: Root region of hierarchy
            coords: (N, 3) scene coordinates
            feats: (N, C) scene features (raw)
            precomputed_feats: (N, D) optional pre-encoded features (SpUNet path).
                When provided, leaf features are looked up instead of encoded.

        Returns:
            Dict mapping level names to feature tensors
            e.g., {'level_0': (1, D), 'level_1': (4, D), ...}
        """
        features_by_level = defaultdict(list)
        point_features = {}
        point_indices = {}

        def traverse(region: Region) -> torch.Tensor:
            """Recursive post-order traversal."""
            if region.is_leaf:
                if precomputed_feats is not None:
                    point_feats = precomputed_feats[region.indices]
                    region_feat = _pool(point_feats, self.pooling)
                else:
                    region_coords = coords[region.indices]
                    region_feats = feats[region.indices]
                    center = coords[region.center_idx]

                    projected = self.projection(region_feats)

                    if self.use_checkpoint and torch.is_grad_enabled():
                        region_feat, point_feats = grad_checkpoint(
                            self.encoder, region_coords, projected, center,
                            use_reentrant=False
                        )
                    else:
                        region_feat, point_feats = self.encoder(
                            region_coords, projected, center=center
                        )

                features_by_level[f'level_{region.level}'].append(region_feat)
                point_features[id(region)] = point_feats
                point_indices[id(region)] = region.indices

                return region_feat
            
            # Non-leaves: recurse first, then aggregate
            child_feats = []
            child_positions = []
            
            for child in region.children:
                child_feat = traverse(child)
                child_feats.append(child_feat)
                child_positions.append(coords[child.center_idx])
            
            child_feats = torch.stack(child_feats)  # (num_children, D)
            child_positions = torch.stack(child_positions)  # (num_children, 3)
            parent_position = coords[region.center_idx]
            
            # Aggregate children (NO encoding for non-leaves)
            if self.use_spatial_context:
                region_feat = self.aggregator(
                    child_feats, child_positions, parent_position
                )
            else:
                region_feat = self.aggregator(child_feats)
            
            # Store
            features_by_level[f'level_{region.level}'].append(region_feat)
            
            return region_feat
        
        # Start traversal from root
        traverse(root)
        
        # Stack features per level
        result = {}
        for level_name, feat_list in features_by_level.items():
            result[level_name] = torch.stack(feat_list)
        
        result['_point_features'] = point_features
        result['_point_indices'] = point_indices

        return result


class G2LTraversal(nn.Module):
    """
    Global-to-Local (G2L) traversal.
    
    Top-down processing where:
    - Root is encoded without propagation
    - Non-root nodes receive propagated parent context
    - All nodes are ENCODED by the backbone
    - Traversal order: pre-order (parent before children)
    
    Args:
        raw_projection: ProjectionMLP for raw features (root)
        combine_projection: CombineProjection for raw + context (non-root)
        encoder: PointCloudEncoder for all regions
        propagator: FeaturePropagator for context propagation
    """
    
    def __init__(
        self,
        raw_projection: nn.Module,
        combine_projection: nn.Module,
        encoder: nn.Module,
        propagator: nn.Module,
        use_checkpoint: bool = False,
        pooling: str = 'max'
    ):
        super().__init__()

        self.raw_projection = raw_projection
        self.combine_projection = combine_projection
        self.encoder = encoder
        self.propagator = propagator
        self.use_checkpoint = use_checkpoint
        self.pooling = pooling

    def forward(
        self,
        root: Region,
        coords: torch.Tensor,
        feats: torch.Tensor,
        precomputed_feats: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Traverse hierarchy top-down.

        Args:
            root: Root region of hierarchy
            coords: (N, 3) scene coordinates
            feats: (N, C) scene features (raw)
            precomputed_feats: (N, D) optional pre-encoded features (SpUNet path).
                When provided, features are looked up and refined with propagation
                instead of being encoded per-region.

        Returns:
            Dict mapping level names to feature tensors
        """
        features_by_level = defaultdict(list)
        point_features = {}
        point_indices = {}

        def traverse(region: Region, parent_feat: Optional[torch.Tensor] = None):
            """Recursive pre-order traversal."""
            region_coords = coords[region.indices]

            if precomputed_feats is not None:
                if parent_feat is None:
                    point_feats = precomputed_feats[region.indices]
                    region_feat = _pool(point_feats, self.pooling)
                else:
                    spunet_feats = precomputed_feats[region.indices]
                    if self.use_checkpoint and torch.is_grad_enabled():
                        propagated = grad_checkpoint(
                            self.propagator, parent_feat, region_coords,
                            use_reentrant=False
                        )
                    else:
                        propagated = self.propagator(parent_feat, region_coords)
                    point_feats = self.combine_projection(spunet_feats, propagated)
                    region_feat = _pool(point_feats, self.pooling)
            else:
                region_feats = feats[region.indices]
                center = coords[region.center_idx]

                if parent_feat is None:
                    encoder_input = self.raw_projection(region_feats)
                else:
                    if self.use_checkpoint and torch.is_grad_enabled():
                        propagated = grad_checkpoint(
                            self.propagator, parent_feat, region_coords,
                            use_reentrant=False
                        )
                    else:
                        propagated = self.propagator(parent_feat, region_coords)
                    encoder_input = self.combine_projection(region_feats, propagated)

                if self.use_checkpoint and torch.is_grad_enabled():
                    region_feat, point_feats = grad_checkpoint(
                        self.encoder, region_coords, encoder_input, center,
                        use_reentrant=False
                    )
                else:
                    region_feat, point_feats = self.encoder(
                        region_coords, encoder_input, center=center
                    )

            features_by_level[f'level_{region.level}'].append(region_feat)

            if region.is_leaf:
                point_features[id(region)] = point_feats
                point_indices[id(region)] = region.indices

            for child in region.children:
                traverse(child, parent_feat=region_feat)
        
        # Start traversal from root
        traverse(root, parent_feat=None)
        
        # Stack features per level
        result = {}
        for level_name, feat_list in features_by_level.items():
            result[level_name] = torch.stack(feat_list)
        
        result['_point_features'] = point_features
        result['_point_indices'] = point_indices

        return result
