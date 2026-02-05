"""
Unit Tests for DBBD Feature Processing Networks

Tests FeaturePropagator and FeatureAggregator modules.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dbbd.models.features.propagator import FeaturePropagator, MultiScalePropagator
from dbbd.models.features.aggregator import FeatureAggregator, MultiScaleAggregator


class TestFeaturePropagator:
    """Test FeaturePropagator module."""
    
    def test_initialization(self):
        """Test propagator initialization."""
        prop = FeaturePropagator(
            parent_dim=96,
            coord_dim=3,
            hidden_dim=128,
            out_dim=96
        )
        
        assert prop.parent_dim == 96
        assert prop.coord_dim == 3
        assert prop.hidden_dim == 128
        assert prop.out_dim == 96
    
    def test_forward_pass(self):
        """Test forward pass with valid inputs."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        parent_feat = torch.randn(96)
        child_coords = torch.randn(50, 3)
        
        output = prop(parent_feat, child_coords)
        
        assert output.shape == (50, 96)
        assert output.dtype == torch.float32
    
    def test_forward_with_batch_parent(self):
        """Test forward with batched parent features."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        parent_feat = torch.randn(1, 96)  # Batched
        child_coords = torch.randn(50, 3)
        
        output = prop(parent_feat, child_coords)
        
        assert output.shape == (50, 96)
    
    def test_variable_child_sizes(self):
        """Test with different numbers of child points."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        parent_feat = torch.randn(96)
        
        for num_children in [10, 50, 100, 200]:
            child_coords = torch.randn(num_children, 3)
            output = prop(parent_feat, child_coords)
            assert output.shape == (num_children, 96)
    
    def test_gradient_flow(self):
        """Test gradient flows through propagator."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        parent_feat = torch.randn(96, requires_grad=True)
        child_coords = torch.randn(50, 3, requires_grad=True)
        
        output = prop(parent_feat, child_coords)
        loss = output.sum()
        loss.backward()
        
        assert parent_feat.grad is not None
        assert child_coords.grad is not None
        assert not torch.isnan(parent_feat.grad).any()
        assert not torch.isnan(child_coords.grad).any()
    
    def test_normalize_coords(self):
        """Test coordinate normalization option."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        parent_feat = torch.randn(96)
        child_coords = torch.randn(50, 3) * 10  # Large scale
        
        output_unnorm = prop(parent_feat, child_coords, normalize_coords=False)
        output_norm = prop(parent_feat, child_coords, normalize_coords=True)
        
        # Outputs should be different
        assert not torch.allclose(output_unnorm, output_norm)
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        
        # Wrong parent dimension
        with pytest.raises(ValueError):
            prop(torch.randn(64), torch.randn(50, 3))
        
        # Wrong coord dimension
        with pytest.raises(ValueError):
            prop(torch.randn(96), torch.randn(50, 2))


class TestFeatureAggregator:
    """Test FeatureAggregator module."""
    
    def test_initialization(self):
        """Test aggregator initialization."""
        agg = FeatureAggregator(feat_dim=96, mode='max')
        assert agg.feat_dim == 96
        assert agg.mode == 'max'
    
    def test_max_pooling(self):
        """Test max pooling aggregation."""
        agg = FeatureAggregator(feat_dim=96, mode='max', use_pre_mlp=False)
        
        child_features = torch.randn(8, 96)
        output = agg(child_features)
        
        assert output.shape == (96,)
        
        # Verify it's actually max pooling
        expected_max, _ = torch.max(child_features, dim=0)
        assert torch.allclose(output, expected_max)
    
    def test_mean_pooling(self):
        """Test mean pooling aggregation."""
        agg = FeatureAggregator(feat_dim=96, mode='mean', use_pre_mlp=False)
        
        child_features = torch.randn(8, 96)
        output = agg(child_features)
        
        assert output.shape == (96,)
        
        # Verify it's actually mean pooling
        expected_mean = torch.mean(child_features, dim=0)
        assert torch.allclose(output, expected_mean)
    
    def test_attention_aggregation(self):
        """Test attention-based aggregation."""
        agg = FeatureAggregator(feat_dim=96, mode='attention')
        
        child_features = torch.randn(8, 96)
        output = agg(child_features)
        
        assert output.shape == (96,)
    
    def test_permutation_invariance(self):
        """Test that aggregation is permutation invariant."""
        agg = FeatureAggregator(feat_dim=96, mode='max', use_pre_mlp=True)
        agg.eval()  # Disable dropout if any
        
        child_features = torch.randn(8, 96)
        
        # Forward with original order
        output1 = agg(child_features)
        
        # Shuffle children
        perm = torch.randperm(8)
        child_features_shuffled = child_features[perm]
        
        # Forward with shuffled order
        output2 = agg(child_features_shuffled)
        
        # Outputs should be identical (or very close due to floating point)
        assert torch.allclose(output1, output2, atol=1e-5)
    
    def test_variable_children(self):
        """Test with different numbers of children."""
        agg = FeatureAggregator(feat_dim=96, mode='max')
        
        for num_children in [2, 5, 10, 20]:
            child_features = torch.randn(num_children, 96)
            output = agg(child_features)
            assert output.shape == (96,)
    
    def test_gradient_flow(self):
        """Test gradient flows through aggregator."""
        agg = FeatureAggregator(feat_dim=96, mode='max')
        
        child_features = torch.randn(8, 96, requires_grad=True)
        
        output = agg(child_features)
        loss = output.sum()
        loss.backward()
        
        assert child_features.grad is not None
        assert not torch.isnan(child_features.grad).any()
    
    def test_with_spatial_context(self):
        """Test aggregation with spatial context."""
        agg = FeatureAggregator(feat_dim=96, mode='attention', use_spatial=True)
        
        child_features = torch.randn(8, 96)
        child_coords = torch.randn(8, 3)
        parent_coord = torch.randn(3)
        
        output = agg(child_features, child_coords, parent_coord)
        
        assert output.shape == (96,)
    
    def test_zero_children_error(self):
        """Test error handling for zero children."""
        agg = FeatureAggregator(feat_dim=96, mode='max')
        
        with pytest.raises(ValueError, match="Cannot aggregate zero children"):
            agg(torch.empty(0, 96))


class TestMultiScaleModules:
    """Test multi-scale variants of propagator and aggregator."""
    
    def test_multiscale_propagator_shared(self):
        """Test multi-scale propagator with shared weights."""
        prop = MultiScalePropagator(
            num_levels=3,
            parent_dim=96,
            hidden_dim=128,
            out_dim=96,
            shared_weights=True
        )
        
        parent_feat = torch.randn(96)
        child_coords = torch.randn(50, 3)
        
        # Forward at different levels should use same weights
        output0 = prop(parent_feat, child_coords, level=0)
        output1 = prop(parent_feat, child_coords, level=1)
        output2 = prop(parent_feat, child_coords, level=2)
        
        # With same inputs, should get same outputs (shared weights)
        assert torch.allclose(output0, output1)
        assert torch.allclose(output1, output2)
    
    def test_multiscale_propagator_separate(self):
        """Test multi-scale propagator with separate weights."""
        prop = MultiScalePropagator(
            num_levels=3,
            parent_dim=96,
            hidden_dim=128,
            out_dim=96,
            shared_weights=False
        )
        
        parent_feat = torch.randn(96)
        child_coords = torch.randn(50, 3)
        
        # Forward at different levels
        output0 = prop(parent_feat, child_coords, level=0)
        output1 = prop(parent_feat, child_coords, level=1)
        
        # With different weights, should get different outputs
        assert not torch.allclose(output0, output1)
    
    def test_multiscale_aggregator(self):
        """Test multi-scale aggregator."""
        agg = MultiScaleAggregator(
            num_levels=3,
            feat_dim=96,
            mode='max',
            shared_weights=True
        )
        
        child_features = torch.randn(8, 96)
        
        output = agg(child_features, level=0)
        assert output.shape == (96,)


class TestIntegration:
    """Integration tests for propagator + aggregator."""
    
    def test_propagate_then_aggregate(self):
        """Test propagating down then aggregating up."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        agg = FeatureAggregator(feat_dim=96, mode='max')
        
        # Start with parent feature
        parent_feat = torch.randn(96)
        
        # Propagate to 3 child regions
        child_coords_list = [
            torch.randn(30, 3),
            torch.randn(40, 3),
            torch.randn(50, 3)
        ]
        
        child_features = []
        for child_coords in child_coords_list:
            child_feat = prop(parent_feat, child_coords)
            # Simulate encoding: just take mean
            encoded = child_feat.mean(dim=0)
            child_features.append(encoded)
        
        # Aggregate back up
        child_features_stacked = torch.stack(child_features)
        aggregated = agg(child_features_stacked)
        
        assert aggregated.shape == (96,)
    
    def test_gradient_flow_bidirectional(self):
        """Test gradients flow through both propagator and aggregator."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        agg = FeatureAggregator(feat_dim=96, mode='max')
        
        parent_feat = torch.randn(96, requires_grad=True)
        
        # Propagate
        child_coords = torch.randn(50, 3)
        child_feat = prop(parent_feat, child_coords)
        
        # Encode (simulate with simple operation)
        encoded = child_feat.view(5, 10, 96).mean(dim=1)  # (5, 96)
        
        # Aggregate
        aggregated = agg(encoded)
        
        # Backward
        loss = aggregated.sum()
        loss.backward()
        
        assert parent_feat.grad is not None
        assert not torch.isnan(parent_feat.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
