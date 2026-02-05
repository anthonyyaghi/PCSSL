"""
Hierarchical Encoder

Main orchestrator module that combines all components into a single
forward pass for DBBD's bidirectional hierarchical encoding.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from .projection import ProjectionMLP, CombineProjection
from .point_encoder import PointCloudEncoder
from .traversal import G2LTraversal, L2GTraversal
from .collector import FeatureCollector
from ..features.propagator import FeaturePropagator
from ..features.aggregator import FeatureAggregator

logger = logging.getLogger(__name__)


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical Encoder for DBBD's bidirectional processing.
    
    Processes point cloud hierarchies through two branches:
    - G2L (Global-to-Local): Top-down with context propagation
    - L2G (Local-to-Global): Bottom-up with aggregation
    
    Supports dual-view mode for contrastive learning:
    - View 1 → G2L branch
    - View 2 → L2G branch
    
    Args:
        input_feat_dim: Raw feature dimension (e.g., 3 for normals)
        hidden_dim: Internal feature dimension
        output_dim: Output feature dimension
        propagator_config: Config dict for FeaturePropagator
        aggregator_config: Config dict for FeatureAggregator
        encoder_config: Config dict for PointCloudEncoder
    """
    
    def __init__(
        self,
        input_feat_dim: int,
        hidden_dim: int = 96,
        output_dim: int = 96,
        propagator_config: Optional[Dict] = None,
        aggregator_config: Optional[Dict] = None,
        encoder_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.input_feat_dim = input_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Default configs
        propagator_config = propagator_config or {}
        aggregator_config = aggregator_config or {}
        encoder_config = encoder_config or {}
        
        # Projection MLPs
        self.raw_projection = ProjectionMLP(
            input_dim=input_feat_dim,
            output_dim=hidden_dim 
        )
        self.combine_projection = CombineProjection(
            raw_dim=input_feat_dim,
            context_dim=output_dim,
            output_dim=hidden_dim
        )
        
        # Shared encoder (used by both branches)
        self.encoder = PointCloudEncoder(
            input_dim=hidden_dim,
            output_dim=output_dim,
            backbone=encoder_config.get('type', 'pointnet'),
            pooling=encoder_config.get('pooling', 'max'),
            hidden_dims=encoder_config.get('hidden_dims', [64, 128])
        )
        
        # G2L-specific: propagator
        self.propagator = FeaturePropagator(
            parent_dim=output_dim,
            coord_dim=3,
            hidden_dim=propagator_config.get('hidden_dim', 128),
            out_dim=output_dim,
            num_layers=propagator_config.get('num_layers', 2)
        )
        
        # L2G-specific: aggregator
        self.aggregator = FeatureAggregator(
            feat_dim=output_dim,
            mode=aggregator_config.get('mode', 'max'),
            use_spatial=aggregator_config.get('use_spatial_context', True),
            use_pre_mlp=aggregator_config.get('use_pre_mlp', True)
        )
        
        # Build traversal modules
        self.g2l_traversal = G2LTraversal(
            raw_projection=self.raw_projection,
            combine_projection=self.combine_projection,
            encoder=self.encoder,
            propagator=self.propagator
        )
        
        self.l2g_traversal = L2GTraversal(
            projection=self.raw_projection,
            encoder=self.encoder,
            aggregator=self.aggregator,
            use_spatial_context=aggregator_config.get('use_spatial_context', True)
        )
        
        logger.info(
            f"HierarchicalEncoder initialized: input_dim={input_feat_dim}, "
            f"hidden_dim={hidden_dim}, output_dim={output_dim}"
        )
    
    def _get_point_features(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Get point-level features by encoding full scene.
        
        Args:
            coords: (N, 3) scene coordinates
            feats: (N, C) scene features
            
        Returns:
            (N, D) point features
        """
        encoder_input = self.raw_projection(feats)
        _, point_feats = self.encoder(coords, encoder_input)
        return point_feats
    
    def _process_scene(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor,
        hierarchy,
        branch: str
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single scene through one branch.
        
        Args:
            coords: (N, 3) scene coordinates
            feats: (N, C) scene features
            hierarchy: Region tree root
            branch: 'g2l' or 'l2g'
            
        Returns:
            Dict with level features
        """
        if branch == 'g2l':
            return self.g2l_traversal(hierarchy, coords, feats)
        else:  # l2g
            return self.l2g_traversal(hierarchy, coords, feats)
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Process batch through both branches.
        
        Args:
            batch: Dict with either:
                - Single view: 'coord', 'feat', 'offset', 'hierarchies'
                - Dual view: 'view1', 'view2', 'hierarchies'
                  where each view has 'coord', 'feat', 'offset'
                  
        Returns:
            Dict with:
            - 'g2l': Dict of level -> (N, D) tensors
            - 'l2g': Dict of level -> (N, D) tensors
            - 'offsets_by_level': Dict of level -> (B+1,) offsets
            - 'point_feats_g2l': (total_points, D) tensor
            - 'point_feats_l2g': (total_points, D) tensor
            - 'point_offset': (B+1,) tensor
        """
        # Determine mode
        is_dual_view = 'view1' in batch and 'view2' in batch
        
        if is_dual_view:
            g2l_data = batch['view1']
            l2g_data = batch['view2']
        else:
            g2l_data = batch
            l2g_data = batch
        
        hierarchies = batch['hierarchies']
        batch_size = len(hierarchies)
        
        collector = FeatureCollector()
        
        # Process each scene
        for i in range(batch_size):
            if is_dual_view:
                g2l_offset = g2l_data['offset']
                l2g_offset = l2g_data['offset']
            else:
                g2l_offset = batch['offset']
                l2g_offset = batch['offset']
            
            # Extract scene data for G2L
            g2l_start = g2l_offset[i].item()
            g2l_end = g2l_offset[i + 1].item()
            g2l_coords = g2l_data['coord'][g2l_start:g2l_end]
            g2l_feats = g2l_data['feat'][g2l_start:g2l_end]
            
            # Extract scene data for L2G
            l2g_start = l2g_offset[i].item()
            l2g_end = l2g_offset[i + 1].item()
            l2g_coords = l2g_data['coord'][l2g_start:l2g_end]
            l2g_feats = l2g_data['feat'][l2g_start:l2g_end]
            
            # Process through branches
            g2l_result = self._process_scene(
                g2l_coords, g2l_feats, hierarchies[i], 'g2l'
            )
            l2g_result = self._process_scene(
                l2g_coords, l2g_feats, hierarchies[i], 'l2g'
            )
            
            # Get point features
            point_feats_g2l = self._get_point_features(g2l_coords, g2l_feats)
            point_feats_l2g = self._get_point_features(l2g_coords, l2g_feats)
            
            # Add to collector
            collector.add_scene(
                g2l=g2l_result,
                l2g=l2g_result,
                point_feats_g2l=point_feats_g2l,
                point_feats_l2g=point_feats_l2g
            )
        
        return collector.get_output()
    
    def __repr__(self):
        return (
            f"HierarchicalEncoder(\n"
            f"  input_feat_dim={self.input_feat_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  output_dim={self.output_dim}\n"
            f")"
        )
