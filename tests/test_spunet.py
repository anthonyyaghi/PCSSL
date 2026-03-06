"""
Tests for SpUNet backbone integration.

Tests are split into two categories:
- Voxelization/config tests: Run without spconv
- SpUNet forward/integration tests: Require spconv (skipped if not installed)
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dbbd.models.utils.hierarchy import Region
from dbbd.training.config import TrainingConfig


class TestTrainingConfigSpUNet:
    """Test that TrainingConfig handles spunet fields correctly."""

    def test_default_backbone_is_pointnet(self):
        config = TrainingConfig()
        assert config.backbone == "pointnet"
        assert config.spunet is None

    def test_spunet_backbone_config(self):
        config = TrainingConfig(
            backbone="spunet",
            spunet={"voxel_size": 0.02, "base_channels": 32}
        )
        assert config.backbone == "spunet"
        assert config.spunet["voxel_size"] == 0.02

    def test_yaml_roundtrip_with_spunet(self, tmp_path):
        config = TrainingConfig(
            backbone="spunet",
            spunet={
                "voxel_size": 0.02,
                "base_channels": 32,
                "channels": [32, 64, 128, 256, 256, 128, 96, 96],
                "layers": [2, 3, 4, 6, 2, 2, 2, 2],
            }
        )
        yaml_path = str(tmp_path / "test_config.yaml")
        config.to_yaml(yaml_path)
        loaded = TrainingConfig.from_yaml(yaml_path)
        assert loaded.backbone == "spunet"
        assert loaded.spunet["voxel_size"] == 0.02
        assert loaded.spunet["channels"] == [32, 64, 128, 256, 256, 128, 96, 96]

    def test_pointnet_config_unchanged(self):
        config = TrainingConfig(small_mode=True)
        assert config.backbone == "pointnet"
        assert config.hidden_dim == 64


class TestHierarchicalEncoderSpUNetInit:
    """Test HierarchicalEncoder initialization with spunet backbone (no spconv needed)."""

    def test_pointnet_init_unchanged(self):
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        encoder = HierarchicalEncoder(input_feat_dim=3)
        assert encoder.backbone == 'pointnet'
        assert hasattr(encoder, 'encoder')
        assert hasattr(encoder, 'raw_projection')

    def test_backbone_param_defaults_to_pointnet(self):
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        encoder = HierarchicalEncoder(input_feat_dim=3)
        assert encoder.backbone == 'pointnet'


class TestVoxelization:
    """Test voxelization logic without requiring spconv."""

    def test_voxelize_basic(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNetSceneEncoder

        enc = SpUNetSceneEncoder(in_channels=3, output_dim=96, voxel_size=0.1)

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.05],  # same voxel as above at voxel_size=0.1
            [0.15, 0.15, 0.15],  # different voxel
        ])
        feats = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        grid_coord, voxel_feats, inverse_map = enc.voxelize(coords, feats)

        assert grid_coord.shape[0] == 2  # 2 unique voxels
        assert voxel_feats.shape == (2, 3)
        assert inverse_map.shape == (3,)
        # Points 0 and 1 should map to same voxel
        assert inverse_map[0] == inverse_map[1]
        assert inverse_map[0] != inverse_map[2]

    def test_voxelize_averaged_features(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNetSceneEncoder

        enc = SpUNetSceneEncoder(in_channels=3, output_dim=96, voxel_size=0.1)

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.05],
        ])
        feats = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])

        grid_coord, voxel_feats, inverse_map = enc.voxelize(coords, feats)

        assert grid_coord.shape[0] == 1
        expected = torch.tensor([[1.0, 1.0, 0.0]])
        assert torch.allclose(voxel_feats, expected)

    def test_desparsify_roundtrip(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNetSceneEncoder

        enc = SpUNetSceneEncoder(in_channels=3, output_dim=96, voxel_size=0.1)

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.05],
            [0.15, 0.15, 0.15],
        ])
        feats = torch.randn(3, 3)

        _, voxel_feats, inverse_map = enc.voxelize(coords, feats)

        # Fake some voxel output features
        voxel_out = torch.randn(voxel_feats.shape[0], 96)
        point_out = enc.desparsify(voxel_out, inverse_map)

        assert point_out.shape == (3, 96)
        # Points in the same voxel should get the same features
        assert torch.equal(point_out[0], point_out[1])
        assert not torch.equal(point_out[0], point_out[2])

    def test_voxelize_all_unique(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNetSceneEncoder

        enc = SpUNetSceneEncoder(in_channels=3, output_dim=96, voxel_size=0.01)

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
        ])
        feats = torch.randn(3, 3)

        grid_coord, voxel_feats, inverse_map = enc.voxelize(coords, feats)

        assert grid_coord.shape[0] == 3
        assert torch.allclose(voxel_feats, feats)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SpUNet requires CUDA"
)
class TestSpUNetForward:
    """Test SpUNet forward pass (requires spconv + CUDA)."""

    def test_spunet_output_shape(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNet

        model = SpUNet(
            in_channels=3,
            output_dim=96,
            base_channels=16,
            channels=(16, 32, 64, 128, 128, 64, 48, 48),
            layers=(1, 1, 1, 1, 1, 1, 1, 1),
        ).cuda()

        num_voxels = 500
        grid_coord = torch.randint(0, 50, (num_voxels, 3)).cuda()
        feat = torch.randn(num_voxels, 3).cuda()
        offset = torch.tensor([num_voxels], dtype=torch.long).cuda()

        out = model(grid_coord, feat, offset)
        assert out.shape == (num_voxels, 96)

    def test_spunet_batched(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNet

        model = SpUNet(
            in_channels=3,
            output_dim=96,
            base_channels=16,
            channels=(16, 32, 64, 128, 128, 64, 48, 48),
            layers=(1, 1, 1, 1, 1, 1, 1, 1),
        ).cuda()

        n1, n2 = 300, 200
        grid_coord = torch.cat([
            torch.randint(0, 50, (n1, 3)),
            torch.randint(0, 50, (n2, 3)),
        ]).cuda()
        feat = torch.randn(n1 + n2, 3).cuda()
        offset = torch.tensor([n1, n1 + n2], dtype=torch.long).cuda()

        out = model(grid_coord, feat, offset)
        assert out.shape == (n1 + n2, 96)

    def test_spunet_gradient_flow(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNet

        model = SpUNet(
            in_channels=3,
            output_dim=96,
            base_channels=16,
            channels=(16, 32, 64, 128, 128, 64, 48, 48),
            layers=(1, 1, 1, 1, 1, 1, 1, 1),
        ).cuda()

        grid_coord = torch.randint(0, 50, (200, 3), device='cuda')
        feat = torch.randn(200, 3, device='cuda', requires_grad=True)
        offset = torch.tensor([200], dtype=torch.long, device='cuda')

        out = model(grid_coord, feat, offset)
        loss = out.sum()
        loss.backward()

        assert feat.grad is not None
        assert feat.grad.shape == (200, 3)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SpUNet requires CUDA"
)
class TestSpUNetSceneEncoderForward:
    """Test full SpUNetSceneEncoder pipeline (requires spconv + CUDA)."""

    def test_scene_encoder_single_scene(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNetSceneEncoder

        enc = SpUNetSceneEncoder(
            in_channels=3,
            output_dim=96,
            voxel_size=0.05,
            base_channels=16,
            channels=(16, 32, 64, 128, 128, 64, 48, 48),
            layers=(1, 1, 1, 1, 1, 1, 1, 1),
        ).cuda()

        N = 1000
        coords = torch.randn(N, 3).cuda()
        feats = torch.randn(N, 3).cuda()
        offset = torch.tensor([0, N], dtype=torch.long).cuda()

        out = enc(coords, feats, offset)
        assert out.shape == (N, 96)

    def test_scene_encoder_batched(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNetSceneEncoder

        enc = SpUNetSceneEncoder(
            in_channels=3,
            output_dim=96,
            voxel_size=0.05,
            base_channels=16,
            channels=(16, 32, 64, 128, 128, 64, 48, 48),
            layers=(1, 1, 1, 1, 1, 1, 1, 1),
        ).cuda()

        n1, n2 = 500, 700
        coords = torch.randn(n1 + n2, 3).cuda()
        feats = torch.randn(n1 + n2, 3).cuda()
        offset = torch.tensor([0, n1, n1 + n2], dtype=torch.long).cuda()

        out = enc(coords, feats, offset)
        assert out.shape == (n1 + n2, 96)

    def test_scene_encoder_gradient_flow(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.spunet import SpUNetSceneEncoder

        enc = SpUNetSceneEncoder(
            in_channels=3,
            output_dim=96,
            voxel_size=0.05,
            base_channels=16,
            channels=(16, 32, 64, 128, 128, 64, 48, 48),
            layers=(1, 1, 1, 1, 1, 1, 1, 1),
        ).cuda()

        N = 500
        coords = torch.randn(N, 3, device='cuda')
        feats = torch.randn(N, 3, device='cuda', requires_grad=True)
        offset = torch.tensor([0, N], dtype=torch.long, device='cuda')

        out = enc(coords, feats, offset)
        loss = out.sum()
        loss.backward()

        assert feats.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SpUNet requires CUDA"
)
class TestHierarchicalEncoderSpUNet:
    """Test full HierarchicalEncoder with SpUNet backbone (requires spconv + CUDA)."""

    def _make_hierarchy(self, num_points=200):
        half = num_points // 2
        child1 = Region(
            indices=np.arange(0, half),
            center_idx=0,
            level=1,
            children=[],
        )
        child2 = Region(
            indices=np.arange(half, num_points),
            center_idx=half,
            level=1,
            children=[],
        )
        root = Region(
            indices=np.arange(num_points),
            center_idx=0,
            level=0,
            children=[child1, child2],
        )
        child1.parent = root
        child2.parent = root
        return root

    def _make_batch(self, num_points=200, dual_view=False):
        coords = torch.randn(num_points, 3).cuda()
        feats = torch.randn(num_points, 3).cuda()
        offset = torch.tensor([0, num_points], dtype=torch.long).cuda()
        hierarchy = self._make_hierarchy(num_points)

        if dual_view:
            return {
                'view1': {'coord': coords, 'feat': feats, 'offset': offset},
                'view2': {'coord': coords.clone(), 'feat': feats.clone(), 'offset': offset.clone()},
                'hierarchies': [hierarchy],
            }
        return {
            'coord': coords,
            'feat': feats,
            'offset': offset,
            'hierarchies': [hierarchy],
        }

    def test_spunet_encoder_init(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder

        encoder = HierarchicalEncoder(
            input_feat_dim=3,
            hidden_dim=96,
            output_dim=96,
            backbone='spunet',
            spunet_config={
                'voxel_size': 0.05,
                'base_channels': 16,
                'channels': (16, 32, 64, 128, 128, 64, 48, 48),
                'layers': (1, 1, 1, 1, 1, 1, 1, 1),
            }
        )
        assert encoder.backbone == 'spunet'
        assert hasattr(encoder, 'scene_encoder')

    def test_spunet_single_view_forward(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder

        encoder = HierarchicalEncoder(
            input_feat_dim=3,
            hidden_dim=96,
            output_dim=96,
            backbone='spunet',
            spunet_config={
                'voxel_size': 0.05,
                'base_channels': 16,
                'channels': (16, 32, 64, 128, 128, 64, 48, 48),
                'layers': (1, 1, 1, 1, 1, 1, 1, 1),
            }
        ).cuda()

        batch = self._make_batch(dual_view=False)
        output = encoder(batch)

        assert 'g2l' in output
        assert 'l2g' in output
        assert 'point_feats_g2l' in output
        assert 'point_feats_l2g' in output

    def test_spunet_dual_view_forward(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder

        encoder = HierarchicalEncoder(
            input_feat_dim=3,
            hidden_dim=96,
            output_dim=96,
            backbone='spunet',
            spunet_config={
                'voxel_size': 0.05,
                'base_channels': 16,
                'channels': (16, 32, 64, 128, 128, 64, 48, 48),
                'layers': (1, 1, 1, 1, 1, 1, 1, 1),
            }
        ).cuda()

        batch = self._make_batch(dual_view=True)
        output = encoder(batch)

        assert 'g2l' in output
        assert 'l2g' in output
        assert 'point_feats_g2l' in output
        assert 'point_feats_l2g' in output

    def test_spunet_gradient_flow_end_to_end(self):
        spconv = pytest.importorskip("spconv")
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder

        encoder = HierarchicalEncoder(
            input_feat_dim=3,
            hidden_dim=96,
            output_dim=96,
            backbone='spunet',
            spunet_config={
                'voxel_size': 0.05,
                'base_channels': 16,
                'channels': (16, 32, 64, 128, 128, 64, 48, 48),
                'layers': (1, 1, 1, 1, 1, 1, 1, 1),
            }
        ).cuda()

        batch = self._make_batch(dual_view=False)
        output = encoder(batch)

        total = sum(v.sum() for v in output['g2l'].values())
        total.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        assert has_grad


class TestTraversalPrecomputedFeats:
    """Test traversal modules with precomputed features (no spconv needed)."""

    def _make_hierarchy(self, num_points=100):
        half = num_points // 2
        child1 = Region(
            indices=np.arange(0, half),
            center_idx=0,
            level=1,
            children=[],
        )
        child2 = Region(
            indices=np.arange(half, num_points),
            center_idx=half,
            level=1,
            children=[],
        )
        root = Region(
            indices=np.arange(num_points),
            center_idx=0,
            level=0,
            children=[child1, child2],
        )
        child1.parent = root
        child2.parent = root
        return root

    def test_l2g_with_precomputed(self):
        from dbbd.models.encoder.projection import ProjectionMLP
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.encoder.traversal import L2GTraversal
        from dbbd.models.features.aggregator import FeatureAggregator

        output_dim = 32
        proj = ProjectionMLP(input_dim=3, output_dim=output_dim)
        encoder = PointCloudEncoder(input_dim=output_dim, output_dim=output_dim)
        agg = FeatureAggregator(feat_dim=output_dim, mode='max')
        traversal = L2GTraversal(
            projection=proj, encoder=encoder, aggregator=agg,
            use_spatial_context=True, pooling='max'
        )

        num_points = 100
        coords = torch.randn(num_points, 3)
        feats = torch.randn(num_points, 3)
        precomputed = torch.randn(num_points, output_dim)
        root = self._make_hierarchy(num_points)

        result = traversal(root, coords, feats, precomputed_feats=precomputed)

        assert 'level_0' in result
        assert 'level_1' in result
        assert result['level_0'].shape == (1, output_dim)
        assert result['level_1'].shape == (2, output_dim)

    def test_g2l_with_precomputed(self):
        from dbbd.models.encoder.projection import ProjectionMLP, CombineProjection
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.encoder.traversal import G2LTraversal
        from dbbd.models.features.propagator import FeaturePropagator

        output_dim = 32
        raw_proj = ProjectionMLP(input_dim=3, output_dim=output_dim)
        # For SpUNet path, combine takes (output_dim, output_dim) -> output_dim
        combine = CombineProjection(raw_dim=output_dim, context_dim=output_dim, output_dim=output_dim)
        encoder = PointCloudEncoder(input_dim=output_dim, output_dim=output_dim)
        prop = FeaturePropagator(parent_dim=output_dim, coord_dim=3, out_dim=output_dim)
        traversal = G2LTraversal(
            raw_projection=raw_proj, combine_projection=combine,
            encoder=encoder, propagator=prop, pooling='max'
        )

        num_points = 100
        coords = torch.randn(num_points, 3)
        feats = torch.randn(num_points, 3)
        precomputed = torch.randn(num_points, output_dim)
        root = self._make_hierarchy(num_points)

        result = traversal(root, coords, feats, precomputed_feats=precomputed)

        assert 'level_0' in result
        assert 'level_1' in result
        assert result['level_0'].shape == (1, output_dim)
        assert result['level_1'].shape == (2, output_dim)

    def test_pointnet_path_unchanged_with_none(self):
        from dbbd.models.encoder.projection import ProjectionMLP, CombineProjection
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.encoder.traversal import G2LTraversal
        from dbbd.models.features.propagator import FeaturePropagator

        output_dim = 32
        raw_proj = ProjectionMLP(input_dim=3, output_dim=output_dim)
        combine = CombineProjection(raw_dim=3, context_dim=output_dim, output_dim=output_dim)
        encoder = PointCloudEncoder(input_dim=output_dim, output_dim=output_dim)
        prop = FeaturePropagator(parent_dim=output_dim, coord_dim=3, out_dim=output_dim)
        traversal = G2LTraversal(
            raw_projection=raw_proj, combine_projection=combine,
            encoder=encoder, propagator=prop
        )

        num_points = 100
        coords = torch.randn(num_points, 3)
        feats = torch.randn(num_points, 3)
        root = self._make_hierarchy(num_points)

        # precomputed_feats=None should use the PointNet path
        result = traversal(root, coords, feats, precomputed_feats=None)

        assert 'level_0' in result
        assert 'level_1' in result

    def test_g2l_precomputed_gradient_flow(self):
        from dbbd.models.encoder.projection import ProjectionMLP, CombineProjection
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.encoder.traversal import G2LTraversal
        from dbbd.models.features.propagator import FeaturePropagator

        output_dim = 32
        raw_proj = ProjectionMLP(input_dim=3, output_dim=output_dim)
        combine = CombineProjection(raw_dim=output_dim, context_dim=output_dim, output_dim=output_dim)
        encoder = PointCloudEncoder(input_dim=output_dim, output_dim=output_dim)
        prop = FeaturePropagator(parent_dim=output_dim, coord_dim=3, out_dim=output_dim)
        traversal = G2LTraversal(
            raw_projection=raw_proj, combine_projection=combine,
            encoder=encoder, propagator=prop, pooling='max'
        )

        num_points = 100
        coords = torch.randn(num_points, 3)
        feats = torch.randn(num_points, 3)
        precomputed = torch.randn(num_points, output_dim, requires_grad=True)
        root = self._make_hierarchy(num_points)

        result = traversal(root, coords, feats, precomputed_feats=precomputed)
        total = sum(v.sum() for k, v in result.items() if k.startswith('level_'))
        total.backward()

        assert precomputed.grad is not None
        assert precomputed.grad.abs().sum() > 0
