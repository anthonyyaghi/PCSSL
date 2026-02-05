"""
Hierarchical Traversal Modules

G2L (Global-to-Local) and L2G (Local-to-Global) traversal implementations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import defaultdict
import logging

from ..utils.hierarchy import Region

logger = logging.getLogger(__name__)


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
        use_spatial_context: bool = True
    ):
        super().__init__()
        
        self.projection = projection
        self.encoder = encoder
        self.aggregator = aggregator
        self.use_spatial_context = use_spatial_context
    
    def forward(
        self,
        root: Region,
        coords: torch.Tensor,
        feats: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Traverse hierarchy bottom-up.
        
        Args:
            root: Root region of hierarchy
            coords: (N, 3) scene coordinates
            feats: (N, C) scene features (raw)
            
        Returns:
            Dict mapping level names to feature tensors
            e.g., {'level_0': (1, D), 'level_1': (4, D), ...}
        """
        features_by_level = defaultdict(list)
        point_features = {}  # region_id -> point_feats for leaves
        
        def traverse(region: Region) -> torch.Tensor:
            """Recursive post-order traversal."""
            if region.is_leaf:
                # Leaves: encode with backbone
                region_coords = coords[region.indices]
                region_feats = feats[region.indices]
                center = coords[region.center_idx]
                
                # Project raw features
                projected = self.projection(region_feats)
                
                # Encode
                region_feat, point_feats = self.encoder(
                    region_coords, projected, center=center
                )
                
                # Store
                features_by_level[f'level_{region.level}'].append(region_feat)
                point_features[id(region)] = point_feats
                
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
        
        # Add point features for leaves
        result['_point_features'] = point_features
        
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
        propagator: nn.Module
    ):
        super().__init__()
        
        self.raw_projection = raw_projection
        self.combine_projection = combine_projection
        self.encoder = encoder
        self.propagator = propagator
    
    def forward(
        self,
        root: Region,
        coords: torch.Tensor,
        feats: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Traverse hierarchy top-down.
        
        Args:
            root: Root region of hierarchy
            coords: (N, 3) scene coordinates
            feats: (N, C) scene features (raw)
            
        Returns:
            Dict mapping level names to feature tensors
        """
        features_by_level = defaultdict(list)
        point_features = {}  # region_id -> point_feats for leaves
        
        def traverse(region: Region, parent_feat: Optional[torch.Tensor] = None):
            """Recursive pre-order traversal."""
            region_coords = coords[region.indices]
            region_feats = feats[region.indices]
            center = coords[region.center_idx]
            
            # Prepare input features
            if parent_feat is None:
                # Root: just project raw features
                encoder_input = self.raw_projection(region_feats)
            else:
                # Non-root: propagate parent context and combine
                propagated = self.propagator(parent_feat, region_coords)
                encoder_input = self.combine_projection(region_feats, propagated)
            
            # Encode this region
            region_feat, point_feats = self.encoder(
                region_coords, encoder_input, center=center
            )
            
            # Store
            features_by_level[f'level_{region.level}'].append(region_feat)
            
            if region.is_leaf:
                point_features[id(region)] = point_feats
            
            # Recurse to children
            for child in region.children:
                traverse(child, parent_feat=region_feat)
        
        # Start traversal from root
        traverse(root, parent_feat=None)
        
        # Stack features per level
        result = {}
        for level_name, feat_list in features_by_level.items():
            result[level_name] = torch.stack(feat_list)
        
        # Add point features for leaves
        result['_point_features'] = point_features
        
        return result
