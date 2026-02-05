"""
Unit Tests for DBBD Phase 4 Contrastive Loss Components

Tests projection head, InfoNCE loss, region loss, point loss, and combined loss.
Following TDD approach - tests written alongside implementation.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


# =============================================================================
# TestContrastiveProjectionHead
# =============================================================================

class TestContrastiveProjectionHead:
    """Test ContrastiveProjectionHead module."""
    
    def test_initialization(self):
        """Test projection head initialization."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        
        head = ContrastiveProjectionHead(
            input_dim=96,
            hidden_dim=96,
            output_dim=128,
            num_layers=2
        )
        
        assert head.input_dim == 96
        assert head.output_dim == 128
    
    def test_forward_shape(self):
        """Test output shape is correct."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        
        head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        
        x = torch.randn(100, 96)
        out = head(x)
        
        assert out.shape == (100, 128)
    
    def test_output_l2_normalized(self):
        """Test output is L2-normalized (norm ≈ 1.0)."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        
        head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        
        x = torch.randn(100, 96)
        out = head(x)
        
        # Check L2 norm of each row is ~1.0
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_variable_batch_sizes(self):
        """Test with different batch sizes."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        
        head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        
        for batch_size in [1, 10, 100, 500]:
            x = torch.randn(batch_size, 96)
            out = head(x)
            assert out.shape == (batch_size, 128)
    
    def test_gradient_flow(self):
        """Test gradients flow through projection."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        
        head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        
        x = torch.randn(50, 96, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check model params have gradients
        for param in head.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_single_layer(self):
        """Test with num_layers=1 (just linear + normalize)."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        
        head = ContrastiveProjectionHead(
            input_dim=96,
            output_dim=128,
            num_layers=1
        )
        
        x = torch.randn(50, 96)
        out = head(x)
        
        assert out.shape == (50, 128)
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# =============================================================================
# TestInfoNCELoss
# =============================================================================

class TestInfoNCELoss:
    """Test InfoNCELoss module."""
    
    def test_initialization(self):
        """Test InfoNCE loss initialization."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.1)
        assert loss_fn.temperature == 0.1
    
    def test_output_is_scalar(self):
        """Test loss returns a scalar."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.1)
        
        query = F.normalize(torch.randn(50, 128), dim=-1).requires_grad_(True)
        key = F.normalize(torch.randn(50, 128), dim=-1).requires_grad_(True)
        
        loss = loss_fn(query, key)
        
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad
    
    def test_known_answer_identity(self):
        """Test with identity similarity matrix - hand-calculated answer."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        # With identity similarity (perfect alignment), we know the answer.
        # sim[i,i] = 1, sim[i,j!=i] = 0 for normalized vectors
        # Loss = -log(exp(1/τ) / (exp(1/τ) + (N-1)*exp(0/τ)))
        # Loss = -log(exp(1/τ) / (exp(1/τ) + N-1))
        
        temperature = 1.0
        loss_fn = InfoNCELoss(temperature=temperature)
        
        N = 2
        # Create orthonormal vectors (identity similarity)
        query = torch.eye(N)  # Each row is a unit vector
        key = torch.eye(N)    # Same
        
        loss = loss_fn(query, key)
        
        # Expected: -log(e^1 / (e^1 + e^0)) = -log(e / (e + 1)) ≈ 0.3133
        expected = -math.log(math.exp(1) / (math.exp(1) + 1))
        
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)
    
    def test_known_answer_n3(self):
        """Test with N=3 identity - hand-calculated answer."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        temperature = 1.0
        loss_fn = InfoNCELoss(temperature=temperature)
        
        N = 3
        query = torch.eye(N)
        key = torch.eye(N)
        
        loss = loss_fn(query, key)
        
        # Expected: -log(e^1 / (e^1 + 2*e^0)) = -log(e / (e + 2))
        expected = -math.log(math.exp(1) / (math.exp(1) + 2))
        
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)
    
    def test_perfect_alignment_low_loss(self):
        """Test that identical query/key gives low loss."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.1)
        
        # Same features = diagonal similarity = 1.0
        feats = F.normalize(torch.randn(50, 128), dim=-1)
        
        loss = loss_fn(feats, feats)
        
        # Should be low (but not zero due to other pairs)
        assert loss.item() < 1.0
    
    def test_random_higher_loss(self):
        """Test that random features give higher loss than aligned."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.1)
        
        feats = F.normalize(torch.randn(50, 128), dim=-1)
        loss_aligned = loss_fn(feats, feats)
        
        # Random unrelated features
        query = F.normalize(torch.randn(50, 128), dim=-1)
        key = F.normalize(torch.randn(50, 128), dim=-1)
        loss_random = loss_fn(query, key)
        
        assert loss_random.item() > loss_aligned.item()
    
    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        query = F.normalize(torch.randn(50, 128), dim=-1)
        key = F.normalize(torch.randn(50, 128), dim=-1)
        
        loss_low_temp = InfoNCELoss(temperature=0.05)(query, key)
        loss_high_temp = InfoNCELoss(temperature=0.5)(query, key)
        
        # Lower temperature amplifies differences -> higher loss for random pairs
        assert loss_low_temp.item() > loss_high_temp.item()
    
    def test_batch_size_1(self):
        """Test edge case with batch_size=1."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.1)
        
        query = F.normalize(torch.randn(1, 128), dim=-1)
        key = F.normalize(torch.randn(1, 128), dim=-1)
        
        loss = loss_fn(query, key)
        
        # Should not error, just return some scalar
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_gradient_flow(self):
        """Test gradients flow to both query and key."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.1)
        
        query = F.normalize(torch.randn(50, 128), dim=-1)
        query.requires_grad = True
        key = F.normalize(torch.randn(50, 128), dim=-1)
        key.requires_grad = True
        
        # Need to re-normalize after setting requires_grad
        query_norm = F.normalize(query, dim=-1)
        key_norm = F.normalize(key, dim=-1)
        
        loss = loss_fn(query_norm, key_norm)
        loss.backward()
        
        assert query.grad is not None
        assert key.grad is not None
        assert not torch.isnan(query.grad).any()
        assert not torch.isnan(key.grad).any()


# =============================================================================
# TestRegionContrastiveLoss
# =============================================================================

class TestRegionContrastiveLoss:
    """Test RegionContrastiveLoss module."""
    
    def test_initialization(self):
        """Test region loss initialization."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.region_loss import RegionContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        loss_fn = RegionContrastiveLoss(projection_head=proj_head, temperature=0.1)
        
        assert loss_fn.temperature == 0.1
    
    def test_processes_all_levels(self):
        """Test that all hierarchy levels are processed."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.region_loss import RegionContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        loss_fn = RegionContrastiveLoss(projection_head=proj_head, temperature=0.1)
        
        g2l_features = {
            'level_0': torch.randn(2, 96),
            'level_1': torch.randn(6, 96),
            'level_2': torch.randn(12, 96),
        }
        l2g_features = {
            'level_0': torch.randn(2, 96),
            'level_1': torch.randn(6, 96),
            'level_2': torch.randn(12, 96),
        }
        offsets = {
            'level_0': torch.tensor([0, 1, 2]),
            'level_1': torch.tensor([0, 3, 6]),
            'level_2': torch.tensor([0, 6, 12]),
        }
        
        total_loss, per_level = loss_fn(g2l_features, l2g_features, offsets)
        
        assert 'level_0' in per_level
        assert 'level_1' in per_level
        assert 'level_2' in per_level
        assert total_loss.dim() == 0
    
    def test_level_weights(self):
        """Test that level weights are applied correctly."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.region_loss import RegionContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        
        g2l_features = {
            'level_0': torch.randn(2, 96),
            'level_1': torch.randn(6, 96),
        }
        l2g_features = {
            'level_0': torch.randn(2, 96),
            'level_1': torch.randn(6, 96),
        }
        offsets = {
            'level_0': torch.tensor([0, 1, 2]),
            'level_1': torch.tensor([0, 3, 6]),
        }
        
        # Equal weights
        loss_equal = RegionContrastiveLoss(
            projection_head=proj_head,
            level_weights={'level_0': 1.0, 'level_1': 1.0}
        )
        
        # Zero weight on level_1
        loss_only_l0 = RegionContrastiveLoss(
            projection_head=proj_head,
            level_weights={'level_0': 1.0, 'level_1': 0.0}
        )
        
        total_eq, per_eq = loss_equal(g2l_features, l2g_features, offsets)
        total_l0, per_l0 = loss_only_l0(g2l_features, l2g_features, offsets)
        
        # With level_1 weight = 0, total should equal level_0 loss
        assert torch.isclose(total_l0, per_l0['level_0'], atol=1e-5)
    
    def test_gradient_flow(self):
        """Test gradients flow through region loss."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.region_loss import RegionContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        loss_fn = RegionContrastiveLoss(projection_head=proj_head)
        
        g2l = {'level_0': torch.randn(5, 96, requires_grad=True)}
        l2g = {'level_0': torch.randn(5, 96, requires_grad=True)}
        offsets = {'level_0': torch.tensor([0, 2, 5])}
        
        total, _ = loss_fn(g2l, l2g, offsets)
        total.backward()
        
        assert g2l['level_0'].grad is not None
        assert l2g['level_0'].grad is not None


# =============================================================================
# TestPointContrastiveLoss
# =============================================================================

class TestPointContrastiveLoss:
    """Test PointContrastiveLoss module."""
    
    def test_initialization(self):
        """Test point loss initialization."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.point_loss import PointContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        loss_fn = PointContrastiveLoss(
            projection_head=proj_head,
            temperature=0.1,
            num_samples=4096
        )
        
        assert loss_fn.num_samples == 4096
    
    def test_full_batch_when_small(self):
        """Test full batch is used when N <= num_samples."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.point_loss import PointContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        loss_fn = PointContrastiveLoss(
            projection_head=proj_head,
            num_samples=4096
        )
        
        # 100 points < 4096, should use all
        point_feats_g2l = torch.randn(100, 96)
        point_feats_l2g = torch.randn(100, 96)
        point_offset = torch.tensor([0, 50, 100])
        
        loss = loss_fn(point_feats_g2l, point_feats_l2g, point_offset)
        
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_subsampling_when_large(self):
        """Test subsampling is applied when N > num_samples."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.point_loss import PointContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        loss_fn = PointContrastiveLoss(
            projection_head=proj_head,
            num_samples=100  # Small for testing
        )
        
        # 1000 points > 100, should subsample
        point_feats_g2l = torch.randn(1000, 96)
        point_feats_l2g = torch.randn(1000, 96)
        point_offset = torch.tensor([0, 500, 1000])
        
        loss = loss_fn(point_feats_g2l, point_feats_l2g, point_offset)
        
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_subsampling_randomness(self):
        """Test that subsampling is random (different results each call)."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.point_loss import PointContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        proj_head.eval()  # Fix batch norm behavior
        
        loss_fn = PointContrastiveLoss(
            projection_head=proj_head,
            num_samples=50
        )
        
        point_feats_g2l = torch.randn(500, 96)
        point_feats_l2g = torch.randn(500, 96)
        point_offset = torch.tensor([0, 250, 500])
        
        # Due to random subsampling, losses should differ
        torch.manual_seed(42)
        loss1 = loss_fn(point_feats_g2l, point_feats_l2g, point_offset)
        
        torch.manual_seed(123)
        loss2 = loss_fn(point_feats_g2l, point_feats_l2g, point_offset)
        
        # With different seeds, subsampling should differ
        assert not torch.isclose(loss1, loss2)
    
    def test_gradient_flow(self):
        """Test gradients flow through point loss."""
        from dbbd.models.loss.projection import ContrastiveProjectionHead
        from dbbd.models.loss.point_loss import PointContrastiveLoss
        
        proj_head = ContrastiveProjectionHead(input_dim=96, output_dim=128)
        loss_fn = PointContrastiveLoss(projection_head=proj_head, num_samples=50)
        
        g2l = torch.randn(100, 96, requires_grad=True)
        l2g = torch.randn(100, 96, requires_grad=True)
        offset = torch.tensor([0, 50, 100])
        
        loss = loss_fn(g2l, l2g, offset)
        loss.backward()
        
        # Gradients should flow (at least to sampled points)
        assert g2l.grad is not None
        assert l2g.grad is not None


# =============================================================================
# TestDBBDContrastiveLoss
# =============================================================================

class TestDBBDContrastiveLoss:
    """Test DBBDContrastiveLoss main module."""
    
    def create_mock_encoder_output(self, batch_size=2, points_per_scene=100):
        """Create mock encoder output matching Phase 3 format."""
        total_points = batch_size * points_per_scene
        
        return {
            'g2l': {
                'level_0': torch.randn(batch_size, 96),
                'level_1': torch.randn(batch_size * 2, 96),
            },
            'l2g': {
                'level_0': torch.randn(batch_size, 96),
                'level_1': torch.randn(batch_size * 2, 96),
            },
            'offsets_by_level': {
                'level_0': torch.tensor([0] + [i + 1 for i in range(batch_size)]),
                'level_1': torch.tensor([0] + [(i + 1) * 2 for i in range(batch_size)]),
            },
            'point_feats_g2l': torch.randn(total_points, 96),
            'point_feats_l2g': torch.randn(total_points, 96),
            'point_offset': torch.tensor([0] + [points_per_scene * (i + 1) for i in range(batch_size)]),
        }
    
    def test_initialization(self):
        """Test DBBD loss initialization."""
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        
        loss_fn = DBBDContrastiveLoss(
            encoder_dim=96,
            projection_dim=128,
            temperature=0.1,
            alpha=1.0,
            beta=0.5
        )
        
        assert loss_fn.alpha == 1.0
        assert loss_fn.beta == 0.5
    
    def test_accepts_encoder_output_format(self):
        """Test that loss accepts Phase 3 encoder output format."""
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        
        loss_fn = DBBDContrastiveLoss(encoder_dim=96)
        
        encoder_output = self.create_mock_encoder_output()
        total_loss, loss_dict = loss_fn(encoder_output)
        
        assert total_loss.dim() == 0
        assert 'total' in loss_dict
        assert 'region' in loss_dict
        assert 'point' in loss_dict
    
    def test_alpha_beta_weights(self):
        """Test that alpha/beta weights are applied correctly."""
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        
        encoder_output = self.create_mock_encoder_output()
        
        # Alpha=1, Beta=0 -> only region loss
        loss_region_only = DBBDContrastiveLoss(encoder_dim=96, alpha=1.0, beta=0.0)
        total_r, dict_r = loss_region_only(encoder_output)
        
        # Total should equal region loss (approximately, accounting for weighting)
        assert torch.isclose(total_r, dict_r['region'], atol=1e-5)
        
        # Alpha=0, Beta=1 -> only point loss
        loss_point_only = DBBDContrastiveLoss(encoder_dim=96, alpha=0.0, beta=1.0)
        total_p, dict_p = loss_point_only(encoder_output)
        
        assert torch.isclose(total_p, dict_p['point'], atol=1e-5)
    
    def test_loss_dict_keys(self):
        """Test that loss_dict contains expected keys."""
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        
        loss_fn = DBBDContrastiveLoss(encoder_dim=96)
        encoder_output = self.create_mock_encoder_output()
        
        _, loss_dict = loss_fn(encoder_output)
        
        # Check required keys
        assert 'total' in loss_dict
        assert 'region' in loss_dict
        assert 'point' in loss_dict
        
        # Check per-level keys
        assert 'region_level_0' in loss_dict
        assert 'region_level_1' in loss_dict
    
    def test_gradient_flow_to_encoder_output(self):
        """Test gradients flow to encoder output tensors."""
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        
        loss_fn = DBBDContrastiveLoss(encoder_dim=96)
        
        # Create encoder output with requires_grad
        encoder_output = {
            'g2l': {'level_0': torch.randn(2, 96, requires_grad=True)},
            'l2g': {'level_0': torch.randn(2, 96, requires_grad=True)},
            'offsets_by_level': {'level_0': torch.tensor([0, 1, 2])},
            'point_feats_g2l': torch.randn(200, 96, requires_grad=True),
            'point_feats_l2g': torch.randn(200, 96, requires_grad=True),
            'point_offset': torch.tensor([0, 100, 200]),
        }
        
        total_loss, _ = loss_fn(encoder_output)
        total_loss.backward()
        
        assert encoder_output['g2l']['level_0'].grad is not None
        assert encoder_output['l2g']['level_0'].grad is not None
        assert encoder_output['point_feats_g2l'].grad is not None
        assert encoder_output['point_feats_l2g'].grad is not None


# =============================================================================
# TestNumericalStability
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of loss functions."""
    
    def test_no_nan_random_inputs(self):
        """Test no NaN with random inputs."""
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        
        loss_fn = DBBDContrastiveLoss(encoder_dim=96)
        
        for _ in range(10):
            encoder_output = {
                'g2l': {'level_0': torch.randn(5, 96)},
                'l2g': {'level_0': torch.randn(5, 96)},
                'offsets_by_level': {'level_0': torch.tensor([0, 2, 5])},
                'point_feats_g2l': torch.randn(100, 96),
                'point_feats_l2g': torch.randn(100, 96),
                'point_offset': torch.tensor([0, 50, 100]),
            }
            
            total_loss, loss_dict = loss_fn(encoder_output)
            
            assert not torch.isnan(total_loss)
            assert not torch.isinf(total_loss)
            for v in loss_dict.values():
                assert not torch.isnan(v)
                assert not torch.isinf(v)
    
    def test_no_nan_near_identical_features(self):
        """Test no NaN when features are nearly identical (near collapse)."""
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        
        loss_fn = DBBDContrastiveLoss(encoder_dim=96)
        
        # All features very similar (potential collapse scenario)
        base = torch.randn(1, 96)
        noise = torch.randn(5, 96) * 0.001  # Small noise
        similar_feats = base.expand(5, -1) + noise
        
        encoder_output = {
            'g2l': {'level_0': similar_feats.clone()},
            'l2g': {'level_0': similar_feats.clone()},
            'offsets_by_level': {'level_0': torch.tensor([0, 2, 5])},
            'point_feats_g2l': similar_feats[:3].clone(),
            'point_feats_l2g': similar_feats[:3].clone(),
            'point_offset': torch.tensor([0, 1, 3]),
        }
        
        total_loss, _ = loss_fn(encoder_output)
        
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)
    
    def test_no_nan_large_values(self):
        """Test no NaN with large feature values."""
        from dbbd.models.loss.infonce import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.1)
        
        # Large values (but normalized, so should be fine)
        query = torch.randn(50, 128) * 100
        query = F.normalize(query, dim=-1)
        key = torch.randn(50, 128) * 100
        key = F.normalize(key, dim=-1)
        
        loss = loss_fn(query, key)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# =============================================================================
# TestEndToEnd
# =============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline_with_encoder(self):
        """Test full pipeline: encoder -> loss."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        from dbbd.models.utils.hierarchy import Region
        
        # Create encoder
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        
        # Create loss
        loss_fn = DBBDContrastiveLoss(encoder_dim=96)
        
        # Create batch
        num_points = 100
        batch = {
            'coord': torch.randn(num_points, 3),
            'feat': torch.randn(num_points, 3),
            'offset': torch.tensor([0, num_points]),
            'hierarchies': [
                Region(indices=np.arange(num_points), center_idx=0, level=0, children=[
                    Region(indices=np.arange(50), center_idx=0, level=1),
                    Region(indices=np.arange(50, 100), center_idx=50, level=1)
                ])
            ]
        }
        # Set parent links
        for child in batch['hierarchies'][0].children:
            child.parent = batch['hierarchies'][0]
        
        # Forward pass
        encoder_output = encoder(batch)
        total_loss, loss_dict = loss_fn(encoder_output)
        
        assert total_loss.dim() == 0
        assert not torch.isnan(total_loss)
    
    def test_gradients_flow_to_encoder_params(self):
        """Test gradients flow from loss to encoder parameters."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        from dbbd.models.utils.hierarchy import Region
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        loss_fn = DBBDContrastiveLoss(encoder_dim=96)
        
        batch = {
            'coord': torch.randn(100, 3),
            'feat': torch.randn(100, 3),
            'offset': torch.tensor([0, 100]),
            'hierarchies': [
                Region(indices=np.arange(100), center_idx=0, level=0, children=[
                    Region(indices=np.arange(50), center_idx=0, level=1),
                    Region(indices=np.arange(50, 100), center_idx=50, level=1)
                ])
            ]
        }
        for child in batch['hierarchies'][0].children:
            child.parent = batch['hierarchies'][0]
        
        encoder_output = encoder(batch)
        total_loss, _ = loss_fn(encoder_output)
        total_loss.backward()
        
        # Check encoder params have gradients
        grad_count = 0
        for name, param in encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
        
        assert grad_count > 0, "No encoder params received gradients"
    
    def test_training_loop_runs(self):
        """Test that a simple training loop runs without error."""
        from dbbd.models.encoder.hierarchical_encoder import HierarchicalEncoder
        from dbbd.models.loss.dbbd_loss import DBBDContrastiveLoss
        from dbbd.models.utils.hierarchy import Region
        
        encoder = HierarchicalEncoder(input_feat_dim=3, hidden_dim=96, output_dim=96)
        loss_fn = DBBDContrastiveLoss(encoder_dim=96)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(loss_fn.parameters()), lr=1e-3)
        
        batch = {
            'coord': torch.randn(100, 3),
            'feat': torch.randn(100, 3),
            'offset': torch.tensor([0, 100]),
            'hierarchies': [
                Region(indices=np.arange(100), center_idx=0, level=0, children=[
                    Region(indices=np.arange(50), center_idx=0, level=1),
                    Region(indices=np.arange(50, 100), center_idx=50, level=1)
                ])
            ]
        }
        for child in batch['hierarchies'][0].children:
            child.parent = batch['hierarchies'][0]
        
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            encoder_output = encoder(batch)
            total_loss, _ = loss_fn(encoder_output)
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())
        
        # Just check it ran without error and loss is finite
        assert all(np.isfinite(l) for l in losses)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
