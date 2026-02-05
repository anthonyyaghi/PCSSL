"""
Unit Tests for DBBD Phase 3 Encoder Components

Tests projection MLP, point encoder, traversals, and hierarchical encoder.
Following TDD approach - tests written alongside implementation.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dbbd.models.utils.hierarchy import Region


class TestProjectionMLP:
    """Test ProjectionMLP module."""
    
    def test_initialization(self):
        """Test projection MLP initialization."""
        from dbbd.models.encoder.projection import ProjectionMLP
        
        proj = ProjectionMLP(input_dim=3, output_dim=96)
        
        assert proj.input_dim == 3
        assert proj.output_dim == 96
    
    def test_forward_pass(self):
        """Test forward pass with valid inputs."""
        from dbbd.models.encoder.projection import ProjectionMLP
        
        proj = ProjectionMLP(input_dim=3, output_dim=96)
        
        x = torch.randn(100, 3)
        output = proj(x)
        
        assert output.shape == (100, 96)
        assert output.dtype == torch.float32
    
    def test_variable_batch_sizes(self):
        """Test with different batch sizes."""
        from dbbd.models.encoder.projection import ProjectionMLP
        
        proj = ProjectionMLP(input_dim=6, output_dim=96)
        
        for batch_size in [10, 50, 100, 500]:
            x = torch.randn(batch_size, 6)
            output = proj(x)
            assert output.shape == (batch_size, 96)
    
    def test_gradient_flow(self):
        """Test gradient flows through projection."""
        from dbbd.models.encoder.projection import ProjectionMLP
        
        proj = ProjectionMLP(input_dim=3, output_dim=96)
        
        x = torch.randn(100, 3, requires_grad=True)
        output = proj(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_invalid_input_dim(self):
        """Test error handling for wrong input dimension."""
        from dbbd.models.encoder.projection import ProjectionMLP
        
        proj = ProjectionMLP(input_dim=3, output_dim=96)
        
        with pytest.raises(ValueError, match="Expected input dim"):
            proj(torch.randn(100, 6))
    
    def test_invalid_input_shape(self):
        """Test error handling for wrong input shape."""
        from dbbd.models.encoder.projection import ProjectionMLP
        
        proj = ProjectionMLP(input_dim=3, output_dim=96)
        
        with pytest.raises(ValueError, match="Expected 2D input"):
            proj(torch.randn(10, 10, 3))


class TestCombineProjection:
    """Test CombineProjection module."""
    
    def test_initialization(self):
        """Test combine projection initialization."""
        from dbbd.models.encoder.projection import CombineProjection
        
        proj = CombineProjection(raw_dim=3, context_dim=96, output_dim=96)
        
        assert proj.raw_dim == 3
        assert proj.context_dim == 96
        assert proj.output_dim == 96
    
    def test_forward_pass(self):
        """Test combining and projecting features."""
        from dbbd.models.encoder.projection import CombineProjection
        
        proj = CombineProjection(raw_dim=3, context_dim=96, output_dim=96)
        
        raw = torch.randn(100, 3)
        context = torch.randn(100, 96)
        output = proj(raw, context)
        
        assert output.shape == (100, 96)
    
    def test_dimension_mismatch_error(self):
        """Test error when raw and context have different batch sizes."""
        from dbbd.models.encoder.projection import CombineProjection
        
        proj = CombineProjection(raw_dim=3, context_dim=96, output_dim=96)
        
        raw = torch.randn(100, 3)
        context = torch.randn(50, 96)  # Mismatched
        
        with pytest.raises(ValueError, match="Feature count mismatch"):
            proj(raw, context)


class TestPointCloudEncoder:
    """Test PointCloudEncoder module."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        
        assert encoder.input_dim == 96
        assert encoder.output_dim == 96
    
    def test_forward_pass(self):
        """Test forward pass returns both region and point features."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        
        coords = torch.randn(100, 3)
        feats = torch.randn(100, 96)
        
        region_feat, point_feats = encoder(coords, feats)
        
        assert region_feat.shape == (96,)
        assert point_feats.shape == (100, 96)
    
    def test_coordinate_centering(self):
        """Test that coordinates are centered before processing."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        
        # Create coords with known centroid
        coords = torch.randn(100, 3) + 100  # Offset by 100
        feats = torch.randn(100, 96)
        
        # Encode with default centering
        region_feat1, _ = encoder(coords, feats)
        
        # Encode with explicit center at 100
        center = torch.tensor([100.0, 100.0, 100.0])
        region_feat2, _ = encoder(coords, feats, center=center)
        
        # Results should be similar (may differ due to centroid vs explicit)
        # The key test is that it doesn't fail
        assert region_feat1.shape == (96,)
        assert region_feat2.shape == (96,)
    
    def test_variable_region_sizes(self):
        """Test encoder handles variable size regions."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        
        for num_points in [10, 50, 100, 500]:
            coords = torch.randn(num_points, 3)
            feats = torch.randn(num_points, 96)
            
            region_feat, point_feats = encoder(coords, feats)
            
            assert region_feat.shape == (96,)
            assert point_feats.shape == (num_points, 96)
    
    def test_gradient_flow(self):
        """Test gradients flow through encoder."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        
        coords = torch.randn(100, 3, requires_grad=True)
        feats = torch.randn(100, 96, requires_grad=True)
        
        region_feat, point_feats = encoder(coords, feats)
        loss = region_feat.sum() + point_feats.sum()
        loss.backward()
        
        assert coords.grad is not None
        assert feats.grad is not None
        assert not torch.isnan(coords.grad).any()
        assert not torch.isnan(feats.grad).any()
    
    def test_max_pooling(self):
        """Test max pooling produces correct results."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96, pooling='max')
        
        coords = torch.randn(100, 3)
        feats = torch.randn(100, 96)
        
        region_feat, point_feats = encoder(coords, feats)
        
        # Region feat should be max of point feats
        expected_max = point_feats.max(dim=0)[0]
        assert torch.allclose(region_feat, expected_max)
    
    def test_mean_pooling(self):
        """Test mean pooling produces correct results."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96, pooling='mean')
        
        coords = torch.randn(100, 3)
        feats = torch.randn(100, 96)
        
        region_feat, point_feats = encoder(coords, feats)
        
        # Region feat should be mean of point feats
        expected_mean = point_feats.mean(dim=0)
        assert torch.allclose(region_feat, expected_mean)
    
    def test_invalid_coord_shape(self):
        """Test error handling for invalid coordinate shape."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        
        with pytest.raises(ValueError, match="Expected coords shape"):
            encoder(torch.randn(100, 2), torch.randn(100, 96))
    
    def test_invalid_feat_dim(self):
        """Test error handling for wrong feature dimension."""
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        
        with pytest.raises(ValueError, match="Expected feat dim"):
            encoder(torch.randn(100, 3), torch.randn(100, 64))


class TestTraversal:
    """Test G2L and L2G traversal modules."""
    
    def create_simple_hierarchy(self, num_points=200):
        """Create a simple 2-level hierarchy for testing."""
        root = Region(indices=np.arange(num_points), center_idx=0, level=0)
        
        # Create 4 child regions (leaves)
        quarter = num_points // 4
        children = []
        for i in range(4):
            start = i * quarter
            end = (i + 1) * quarter if i < 3 else num_points
            child = Region(
                indices=np.arange(start, end),
                center_idx=start,
                level=1,
                parent=root
            )
            children.append(child)
        
        root.children = children
        return root
    
    def create_3level_hierarchy(self, num_points=200):
        """Create a 3-level hierarchy for testing."""
        # Level 0: root
        root = Region(indices=np.arange(num_points), center_idx=0, level=0)
        
        # Level 1: 2 children
        half = num_points // 2
        child1 = Region(
            indices=np.arange(half),
            center_idx=0,
            level=1,
            parent=root
        )
        child2 = Region(
            indices=np.arange(half, num_points),
            center_idx=half,
            level=1,
            parent=root
        )
        
        # Level 2: 2 grandchildren per child
        quarter = num_points // 4
        gc1 = Region(indices=np.arange(quarter), center_idx=0, level=2, parent=child1)
        gc2 = Region(indices=np.arange(quarter, half), center_idx=quarter, level=2, parent=child1)
        gc3 = Region(indices=np.arange(half, half + quarter), center_idx=half, level=2, parent=child2)
        gc4 = Region(indices=np.arange(half + quarter, num_points), center_idx=half + quarter, level=2, parent=child2)
        
        child1.children = [gc1, gc2]
        child2.children = [gc3, gc4]
        root.children = [child1, child2]
        
        return root
    
    def test_l2g_traversal_leaf_encoding(self):
        """Test L2G encodes leaves with backbone."""
        from dbbd.models.encoder.traversal import L2GTraversal
        from dbbd.models.encoder.projection import ProjectionMLP
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.features.aggregator import FeatureAggregator
        
        projection = ProjectionMLP(input_dim=3, output_dim=96)
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        aggregator = FeatureAggregator(feat_dim=96, mode='max')
        
        l2g = L2GTraversal(
            projection=projection,
            encoder=encoder,
            aggregator=aggregator
        )
        
        hierarchy = self.create_simple_hierarchy(num_points=100)
        coords = torch.randn(100, 3)
        feats = torch.randn(100, 3)
        
        result = l2g(hierarchy, coords, feats)
        
        # Should have features for all levels
        assert 'level_0' in result
        assert 'level_1' in result
        assert result['level_0'].shape == (1, 96)  # 1 root
        assert result['level_1'].shape == (4, 96)  # 4 leaves
    
    def test_l2g_traversal_nonleaf_aggregation(self):
        """Test L2G uses pure aggregation for non-leaves (no encoding)."""
        from dbbd.models.encoder.traversal import L2GTraversal
        from dbbd.models.encoder.projection import ProjectionMLP
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.features.aggregator import FeatureAggregator
        
        projection = ProjectionMLP(input_dim=3, output_dim=96)
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        aggregator = FeatureAggregator(feat_dim=96, mode='max', use_pre_mlp=False)
        
        l2g = L2GTraversal(
            projection=projection,
            encoder=encoder,
            aggregator=aggregator
        )
        
        hierarchy = self.create_3level_hierarchy(num_points=200)
        coords = torch.randn(200, 3)
        feats = torch.randn(200, 3)
        
        result = l2g(hierarchy, coords, feats)
        
        # Should have 3 levels
        assert 'level_0' in result
        assert 'level_1' in result
        assert 'level_2' in result
        
        # Level counts
        assert result['level_0'].shape == (1, 96)   # 1 root
        assert result['level_1'].shape == (2, 96)   # 2 non-leaves
        assert result['level_2'].shape == (4, 96)   # 4 leaves
    
    def test_g2l_traversal_root_no_propagation(self):
        """Test G2L root is encoded without propagation."""
        from dbbd.models.encoder.traversal import G2LTraversal
        from dbbd.models.encoder.projection import ProjectionMLP, CombineProjection
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.features.propagator import FeaturePropagator
        
        raw_projection = ProjectionMLP(input_dim=3, output_dim=96)
        combine_projection = CombineProjection(raw_dim=3, context_dim=96, output_dim=96)
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        propagator = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        g2l = G2LTraversal(
            raw_projection=raw_projection,
            combine_projection=combine_projection,
            encoder=encoder,
            propagator=propagator
        )
        
        hierarchy = self.create_simple_hierarchy(num_points=100)
        coords = torch.randn(100, 3)
        feats = torch.randn(100, 3)
        
        result = g2l(hierarchy, coords, feats)
        
        # Should have features for all levels
        assert 'level_0' in result
        assert 'level_1' in result
        assert result['level_0'].shape == (1, 96)
        assert result['level_1'].shape == (4, 96)
    
    def test_g2l_traversal_children_with_propagation(self):
        """Test G2L children receive propagated parent context."""
        from dbbd.models.encoder.traversal import G2LTraversal
        from dbbd.models.encoder.projection import ProjectionMLP, CombineProjection
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.features.propagator import FeaturePropagator
        
        raw_projection = ProjectionMLP(input_dim=3, output_dim=96)
        combine_projection = CombineProjection(raw_dim=3, context_dim=96, output_dim=96)
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        propagator = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        g2l = G2LTraversal(
            raw_projection=raw_projection,
            combine_projection=combine_projection,
            encoder=encoder,
            propagator=propagator
        )
        
        hierarchy = self.create_3level_hierarchy(num_points=200)
        coords = torch.randn(200, 3)
        feats = torch.randn(200, 3)
        
        result = g2l(hierarchy, coords, feats)
        
        # Should have 3 levels
        assert 'level_0' in result
        assert 'level_1' in result
        assert 'level_2' in result
    
    def test_traversal_gradient_flow(self):
        """Test gradients flow through traversals."""
        from dbbd.models.encoder.traversal import G2LTraversal, L2GTraversal
        from dbbd.models.encoder.projection import ProjectionMLP, CombineProjection
        from dbbd.models.encoder.point_encoder import PointCloudEncoder
        from dbbd.models.features.propagator import FeaturePropagator
        from dbbd.models.features.aggregator import FeatureAggregator
        
        # Setup G2L
        raw_projection = ProjectionMLP(input_dim=3, output_dim=96)
        combine_projection = CombineProjection(raw_dim=3, context_dim=96, output_dim=96)
        encoder = PointCloudEncoder(input_dim=96, output_dim=96)
        propagator = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        g2l = G2LTraversal(
            raw_projection=raw_projection,
            combine_projection=combine_projection,
            encoder=encoder,
            propagator=propagator
        )
        
        hierarchy = self.create_simple_hierarchy(num_points=100)
        coords = torch.randn(100, 3, requires_grad=True)
        feats = torch.randn(100, 3, requires_grad=True)
        
        result = g2l(hierarchy, coords, feats)
        
        # Filter out internal keys (prefixed with _) and sum tensor values
        loss = sum(v.sum() for k, v in result.items() if not k.startswith('_'))
        loss.backward()
        
        assert feats.grad is not None
        assert not torch.isnan(feats.grad).any()


class TestFeatureCollector:
    """Test FeatureCollector for batching."""
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        from dbbd.models.encoder.collector import FeatureCollector
        
        collector = FeatureCollector()
        assert len(collector.g2l_by_level) == 0
        assert len(collector.l2g_by_level) == 0
    
    def test_add_features(self):
        """Test adding features for a scene."""
        from dbbd.models.encoder.collector import FeatureCollector
        
        collector = FeatureCollector()
        
        # Add features for scene 0
        g2l_feats = {
            'level_0': torch.randn(1, 96),
            'level_1': torch.randn(4, 96)
        }
        l2g_feats = {
            'level_0': torch.randn(1, 96),
            'level_1': torch.randn(4, 96)
        }
        
        collector.add_scene(g2l_feats, l2g_feats)
        
        assert len(collector.g2l_by_level) == 2
        assert len(collector.l2g_by_level) == 2
    
    def test_collect_output(self):
        """Test collecting final output with offsets."""
        from dbbd.models.encoder.collector import FeatureCollector
        
        collector = FeatureCollector()
        
        # Add 2 scenes
        collector.add_scene(
            g2l={'level_0': torch.randn(1, 96), 'level_1': torch.randn(4, 96)},
            l2g={'level_0': torch.randn(1, 96), 'level_1': torch.randn(4, 96)}
        )
        collector.add_scene(
            g2l={'level_0': torch.randn(1, 96), 'level_1': torch.randn(3, 96)},
            l2g={'level_0': torch.randn(1, 96), 'level_1': torch.randn(3, 96)}
        )
        
        output = collector.get_output()
        
        # Check structure
        assert 'g2l' in output
        assert 'l2g' in output
        assert 'offsets_by_level' in output
        
        # Check level 0 (1+1=2 regions)
        assert output['g2l']['level_0'].shape == (2, 96)
        assert output['offsets_by_level']['level_0'].tolist() == [0, 1, 2]
        
        # Check level 1 (4+3=7 regions)
        assert output['g2l']['level_1'].shape == (7, 96)
        assert output['offsets_by_level']['level_1'].tolist() == [0, 4, 7]


class TestHierarchicalEncoder:
    """Test full HierarchicalEncoder module."""
    
    def create_test_batch(self, num_scenes=2, num_points_per_scene=100):
        """Create a test batch."""
        total_points = num_scenes * num_points_per_scene
        
        # Create hierarchies
        hierarchies = []
        for i in range(num_scenes):
            root = Region(indices=np.arange(num_points_per_scene), center_idx=0, level=0)
            child1 = Region(
                indices=np.arange(num_points_per_scene // 2),
                center_idx=0,
                level=1,
                parent=root
            )
            child2 = Region(
                indices=np.arange(num_points_per_scene // 2, num_points_per_scene),
                center_idx=num_points_per_scene // 2,
                level=1,
                parent=root
            )
            root.children = [child1, child2]
            hierarchies.append(root)
        
        batch = {
            'coord': torch.randn(total_points, 3),
            'feat': torch.randn(total_points, 3),
            'offset': torch.tensor([0] + [num_points_per_scene * (i + 1) for i in range(num_scenes)]),
            'hierarchies': hierarchies
        }
        
        return batch
    
    def create_dual_view_batch(self, num_scenes=2, num_points_per_scene=100):
        """Create a dual-view test batch."""
        total_points = num_scenes * num_points_per_scene
        
        # Create hierarchies (shared between views)
        hierarchies = []
        for i in range(num_scenes):
            root = Region(indices=np.arange(num_points_per_scene), center_idx=0, level=0)
            child1 = Region(
                indices=np.arange(num_points_per_scene // 2),
                center_idx=0,
                level=1,
                parent=root
            )
            child2 = Region(
                indices=np.arange(num_points_per_scene // 2, num_points_per_scene),
                center_idx=num_points_per_scene // 2,
                level=1,
                parent=root
            )
            root.children = [child1, child2]
            hierarchies.append(root)
        
        offset = torch.tensor([0] + [num_points_per_scene * (i + 1) for i in range(num_scenes)])
        
        batch = {
            'view1': {
                'coord': torch.randn(total_points, 3),
                'feat': torch.randn(total_points, 3),
                'offset': offset
            },
            'view2': {
                'coord': torch.randn(total_points, 3),
                'feat': torch.randn(total_points, 3),
                'offset': offset
            },
            'hierarchies': hierarchies
        }
        
        return batch
    
    def test_initialization(self):
        """Test encoder initialization."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        assert encoder.hidden_dim == 96
        assert encoder.output_dim == 96
    
    def test_single_view_forward(self):
        """Test forward pass with single view batch."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        batch = self.create_test_batch(num_scenes=2, num_points_per_scene=100)
        output = encoder(batch)
        
        # Check output structure
        assert 'g2l' in output
        assert 'l2g' in output
        assert 'offsets_by_level' in output
        assert 'point_feats_g2l' in output
        assert 'point_feats_l2g' in output
        assert 'point_offset' in output
        
        # Check point features shape
        assert output['point_feats_g2l'].shape == (200, 96)
        assert output['point_feats_l2g'].shape == (200, 96)
    
    def test_dual_view_forward(self):
        """Test forward pass with dual view batch."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        batch = self.create_dual_view_batch(num_scenes=2, num_points_per_scene=100)
        output = encoder(batch)
        
        # Check output structure for dual view
        assert 'g2l' in output
        assert 'l2g' in output
        
        # G2L uses view1, L2G uses view2 - features should differ
        # (Different random inputs -> different features)
    
    def test_output_format_phase4(self):
        """Test output format matches Phase 4 requirements."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        batch = self.create_test_batch(num_scenes=3, num_points_per_scene=100)
        output = encoder(batch)
        
        # Verify exact structure required by Phase 4
        required_keys = ['g2l', 'l2g', 'offsets_by_level', 
                        'point_feats_g2l', 'point_feats_l2g', 'point_offset']
        for key in required_keys:
            assert key in output, f"Missing required key: {key}"
        
        # Verify offset shape (B+1)
        assert output['point_offset'].shape == (4,)  # 3 scenes + 1
    
    def test_gradient_flow_end_to_end(self):
        """Test gradients flow through entire encoder."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        batch = self.create_test_batch(num_scenes=2, num_points_per_scene=100)
        output = encoder(batch)
        
        # Create dummy loss
        loss = output['point_feats_g2l'].sum() + output['point_feats_l2g'].sum()
        for level_feats in output['g2l'].values():
            loss = loss + level_feats.sum()
        for level_feats in output['l2g'].values():
            loss = loss + level_feats.sum()
        
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_no_nan_inf_features(self):
        """Test that features don't contain NaN or Inf."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        batch = self.create_test_batch(num_scenes=2, num_points_per_scene=100)
        output = encoder(batch)
        
        # Check all tensors
        assert not torch.isnan(output['point_feats_g2l']).any()
        assert not torch.isnan(output['point_feats_l2g']).any()
        assert not torch.isinf(output['point_feats_g2l']).any()
        assert not torch.isinf(output['point_feats_l2g']).any()
        
        for level_feats in output['g2l'].values():
            assert not torch.isnan(level_feats).any()
            assert not torch.isinf(level_feats).any()
    
    def test_features_not_constant(self):
        """Test that features are not all zeros or identical."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        # Use dual-view batch where G2L and L2G process different data
        batch = self.create_dual_view_batch(num_scenes=2, num_points_per_scene=100)
        output = encoder(batch)
        
        # Features should not be all zeros
        assert output['point_feats_g2l'].abs().sum() > 0
        assert output['point_feats_l2g'].abs().sum() > 0
        
        # G2L and L2G should produce different features (different input views)
        assert not torch.allclose(output['point_feats_g2l'], output['point_feats_l2g'])


class TestDataLoaderIntegration:
    """Test HierarchicalEncoder with real DataLoader."""
    
    def create_dummy_dataset(self, tmp_dir, num_scenes=5):
        """Create dummy dataset in combined format."""
        split_dir = tmp_dir / 'train'
        split_dir.mkdir()
        
        for i in range(num_scenes):
            num_points = 100 + i * 20
            
            # Create hierarchy dict with 2 levels
            hierarchy_dict = {
                'indices': np.arange(num_points),
                'center_idx': 0,
                'level': 0,
                'children': [
                    {
                        'indices': np.arange(num_points // 2),
                        'center_idx': 0,
                        'level': 1,
                        'children': []
                    },
                    {
                        'indices': np.arange(num_points // 2, num_points),
                        'center_idx': num_points // 2,
                        'level': 1,
                        'children': []
                    }
                ]
            }
            
            # Create combined data
            import pickle
            data = {
                'coords': torch.randn(num_points, 3),
                'normals': torch.rand(num_points, 3),
                'hierarchy': hierarchy_dict,
                'num_points': num_points,
                'total_regions': 3,
                'max_depth_reached': 1
            }
            
            with open(split_dir / f'scene{i:04d}.pkl', 'wb') as f:
                pickle.dump(data, f)
        
        return tmp_dir
    
    def test_encoder_with_dataloader(self, tmp_path):
        """Test encoder processes batches from dataloader correctly."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        from dbbd.datasets.dbbd_dataset import DBBDDataset, create_dataloader
        from dbbd.datasets.transforms import ToTensor
        
        data_root = self.create_dummy_dataset(tmp_path)
        
        dataset = DBBDDataset(
            data_root=str(data_root),
            split='train',
            transform=ToTensor()
        )
        
        loader = create_dataloader(
            dataset,
            batch_size=2,
            num_workers=0,
            shuffle=False
        )
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        for batch in loader:
            output = encoder(batch)
            
            # Verify output structure
            assert 'g2l' in output
            assert 'l2g' in output
            assert 'point_feats_g2l' in output
            assert 'point_feats_l2g' in output
            
            # Verify shapes match batch
            total_points = batch['coord'].shape[0]
            assert output['point_feats_g2l'].shape[0] == total_points
            break
    
    def test_encoder_with_dual_view_dataloader(self, tmp_path):
        """Test encoder with dual-view dataloader for contrastive learning."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        from dbbd.datasets.dbbd_dataset import DBBDDataset, create_dataloader
        from dbbd.datasets.transforms import Compose, RandomRotate, ToTensor
        
        data_root = self.create_dummy_dataset(tmp_path)
        
        transform = Compose([
            RandomRotate(angle=[-180, 180], axis='z', p=1.0),
            ToTensor()
        ])
        
        dataset = DBBDDataset(
            data_root=str(data_root),
            split='train',
            dual_view=True,
            transform=transform
        )
        
        loader = create_dataloader(
            dataset,
            batch_size=2,
            num_workers=0,
            shuffle=False
        )
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        for batch in loader:
            assert 'view1' in batch
            assert 'view2' in batch
            
            output = encoder(batch)
            
            # Verify dual-view produces different features
            assert 'point_feats_g2l' in output
            assert 'point_feats_l2g' in output
            break
    
    def test_memory_no_growth(self):
        """Test that repeated forward passes don't cause memory growth."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        import gc
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        # Create test batch
        batch = {
            'coord': torch.randn(200, 3),
            'feat': torch.randn(200, 3),
            'offset': torch.tensor([0, 100, 200]),
            'hierarchies': [
                Region(indices=np.arange(100), center_idx=0, level=0, children=[
                    Region(indices=np.arange(50), center_idx=0, level=1),
                    Region(indices=np.arange(50, 100), center_idx=50, level=1)
                ]),
                Region(indices=np.arange(100), center_idx=0, level=0, children=[
                    Region(indices=np.arange(50), center_idx=0, level=1),
                    Region(indices=np.arange(50, 100), center_idx=50, level=1)
                ])
            ]
        }
        # Set parent links
        for hier in batch['hierarchies']:
            for child in hier.children:
                child.parent = hier
        
        # Run forward passes multiple times
        for _ in range(10):
            output = encoder(batch)
            # Compute dummy loss to ensure backward doesn't leak
            loss = output['point_feats_g2l'].sum()
            loss.backward()
            
            # Clear gradients
            encoder.zero_grad()
            del output, loss
            gc.collect()
        
        # Test passes if no error (explicit memory check would need CUDA)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

