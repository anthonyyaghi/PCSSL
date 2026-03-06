"""
Hierarchical Encoder

Main orchestrator module that combines all components into a single
forward pass for DBBD's bidirectional hierarchical encoding.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Literal
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
        backbone: 'pointnet' or 'spunet'
        spunet_config: Config dict for SpUNetSceneEncoder (when backbone='spunet')
    """

    def __init__(
        self,
        input_feat_dim: int,
        hidden_dim: int = 96,
        output_dim: int = 96,
        propagator_config: Optional[Dict] = None,
        aggregator_config: Optional[Dict] = None,
        encoder_config: Optional[Dict] = None,
        use_gradient_checkpoint: bool = False,
        backbone: Literal['pointnet', 'spunet'] = 'pointnet',
        spunet_config: Optional[Dict] = None
    ):
        super().__init__()

        self.input_feat_dim = input_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.backbone = backbone

        propagator_config = propagator_config or {}
        aggregator_config = aggregator_config or {}
        encoder_config = encoder_config or {}
        spunet_config = spunet_config or {}

        pooling = encoder_config.get('pooling', 'max')

        if backbone == 'spunet':
            from .spunet import SpUNetSceneEncoder

            self.scene_encoder = SpUNetSceneEncoder(
                in_channels=input_feat_dim,
                output_dim=output_dim,
                voxel_size=spunet_config.get('voxel_size', 0.02),
                base_channels=spunet_config.get('base_channels', 32),
                channels=tuple(spunet_config.get('channels', (32, 64, 128, 256, 256, 128, 96, 96))),
                layers=tuple(spunet_config.get('layers', (2, 3, 4, 6, 2, 2, 2, 2))),
            )

            self.combine_projection = CombineProjection(
                raw_dim=output_dim,
                context_dim=output_dim,
                output_dim=hidden_dim
            )

            # Placeholders so traversal modules have valid references
            # (they won't be called in the precomputed path)
            self.raw_projection = nn.Identity()
            self.encoder = nn.Identity()
        else:
            self.raw_projection = ProjectionMLP(
                input_dim=input_feat_dim,
                output_dim=hidden_dim
            )
            self.combine_projection = CombineProjection(
                raw_dim=input_feat_dim,
                context_dim=output_dim,
                output_dim=hidden_dim
            )
            self.encoder = PointCloudEncoder(
                input_dim=hidden_dim,
                output_dim=output_dim,
                backbone=encoder_config.get('type', 'pointnet'),
                pooling=pooling,
                hidden_dims=encoder_config.get('hidden_dims', [64, 128])
            )

        self.propagator = FeaturePropagator(
            parent_dim=output_dim,
            coord_dim=3,
            hidden_dim=propagator_config.get('hidden_dim', 128),
            out_dim=output_dim,
            num_layers=propagator_config.get('num_layers', 2)
        )

        self.aggregator = FeatureAggregator(
            feat_dim=output_dim,
            mode=aggregator_config.get('mode', 'max'),
            use_spatial=aggregator_config.get('use_spatial_context', True),
            use_pre_mlp=aggregator_config.get('use_pre_mlp', True)
        )

        self.g2l_traversal = G2LTraversal(
            raw_projection=self.raw_projection,
            combine_projection=self.combine_projection,
            encoder=self.encoder,
            propagator=self.propagator,
            use_checkpoint=use_gradient_checkpoint,
            pooling=pooling
        )

        self.l2g_traversal = L2GTraversal(
            projection=self.raw_projection,
            encoder=self.encoder,
            aggregator=self.aggregator,
            use_spatial_context=aggregator_config.get('use_spatial_context', True),
            use_checkpoint=use_gradient_checkpoint,
            pooling=pooling
        )

        logger.info(
            f"HierarchicalEncoder initialized: input_dim={input_feat_dim}, "
            f"hidden_dim={hidden_dim}, output_dim={output_dim}, backbone={backbone}"
        )

    def _collect_leaf_point_features(
        self,
        traversal_result: Dict,
        num_points: int
    ) -> torch.Tensor:
        """
        Reassemble full-scene point features from leaf region features.

        Args:
            traversal_result: Dict from traversal containing '_point_features' and '_point_indices'
            num_points: Total number of points in the scene

        Returns:
            (N, D) point features for the full scene
        """
        leaf_point_features = traversal_result['_point_features']
        leaf_indices = traversal_result['_point_indices']

        sample_feats = next(iter(leaf_point_features.values()))
        device = sample_feats.device
        feat_dim = sample_feats.shape[1]

        full_point_feats = torch.zeros(num_points, feat_dim, device=device, dtype=sample_feats.dtype)

        for region_id, point_feats in leaf_point_features.items():
            indices = leaf_indices[region_id]
            full_point_feats[indices] = point_feats

        return full_point_feats

    def _process_scene(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor,
        hierarchy,
        branch: str,
        precomputed_feats: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single scene through one branch.

        Args:
            coords: (N, 3) scene coordinates
            feats: (N, C) scene features
            hierarchy: Region tree root
            branch: 'g2l' or 'l2g'
            precomputed_feats: (N, D) optional pre-encoded features (SpUNet path)

        Returns:
            Dict with level features
        """
        if branch == 'g2l':
            return self.g2l_traversal(hierarchy, coords, feats, precomputed_feats=precomputed_feats)
        else:
            return self.l2g_traversal(hierarchy, coords, feats, precomputed_feats=precomputed_feats)

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
        is_dual_view = 'view1' in batch and 'view2' in batch

        if is_dual_view:
            g2l_data = batch['view1']
            l2g_data = batch['view2']
        else:
            g2l_data = batch
            l2g_data = batch

        hierarchies = batch['hierarchies']
        batch_size = len(hierarchies)

        # Pre-encode through SpUNet if applicable (once per view, entire batch)
        g2l_precomputed = None
        l2g_precomputed = None
        if self.backbone == 'spunet':
            g2l_precomputed = self.scene_encoder(
                g2l_data['coord'], g2l_data['feat'], g2l_data['offset']
            )
            l2g_precomputed = self.scene_encoder(
                l2g_data['coord'], l2g_data['feat'], l2g_data['offset']
            )

        collector = FeatureCollector()

        for i in range(batch_size):
            if is_dual_view:
                g2l_offset = g2l_data['offset']
                l2g_offset = l2g_data['offset']
            else:
                g2l_offset = batch['offset']
                l2g_offset = batch['offset']

            g2l_start = g2l_offset[i].item()
            g2l_end = g2l_offset[i + 1].item()
            g2l_coords = g2l_data['coord'][g2l_start:g2l_end]
            g2l_feats = g2l_data['feat'][g2l_start:g2l_end]

            l2g_start = l2g_offset[i].item()
            l2g_end = l2g_offset[i + 1].item()
            l2g_coords = l2g_data['coord'][l2g_start:l2g_end]
            l2g_feats = l2g_data['feat'][l2g_start:l2g_end]

            scene_g2l_pre = g2l_precomputed[g2l_start:g2l_end] if g2l_precomputed is not None else None
            scene_l2g_pre = l2g_precomputed[l2g_start:l2g_end] if l2g_precomputed is not None else None

            g2l_result = self._process_scene(
                g2l_coords, g2l_feats, hierarchies[i], 'g2l',
                precomputed_feats=scene_g2l_pre
            )
            l2g_result = self._process_scene(
                l2g_coords, l2g_feats, hierarchies[i], 'l2g',
                precomputed_feats=scene_l2g_pre
            )

            point_feats_g2l = self._collect_leaf_point_features(g2l_result, g2l_coords.shape[0])
            point_feats_l2g = self._collect_leaf_point_features(l2g_result, l2g_coords.shape[0])

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
            f"  output_dim={self.output_dim},\n"
            f"  backbone={self.backbone}\n"
            f")"
        )
