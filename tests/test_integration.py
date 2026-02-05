"""
Integration Tests for DBBD Phase 2

Tests complete data pipeline with batching and feature processing.
"""

import pytest
import torch
import numpy as np
import pickle
import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from dbbd.models.utils.hierarchy import Region, dict_to_region
from dbbd.models.utils.batch import collate_fn, extract_scene_from_batch, compute_batch_stats
from dbbd.datasets.dbbd_dataset import DBBDDataset, create_dataloader
from dbbd.datasets.transforms import Compose, RandomRotate, RandomScale, ToTensor
from dbbd.models.features.propagator import FeaturePropagator
from dbbd.models.features.aggregator import FeatureAggregator


class TestBatching:
    """Test batching and collation."""
    
    def create_sample(self, num_points):
        """Create a sample dictionary."""
        hierarchy = Region(indices=np.arange(num_points), center_idx=0, level=0)
        child1 = Region(indices=np.arange(num_points // 2), center_idx=0, level=1, parent=hierarchy)
        child2 = Region(indices=np.arange(num_points // 2, num_points), center_idx=num_points // 2, level=1, parent=hierarchy)
        hierarchy.children = [child1, child2]
        
        return {
            'coord': torch.randn(num_points, 3),
            'feat': torch.rand(num_points, 6),
            'hierarchy': hierarchy,
            'scene_id': f'scene_{num_points}'
        }
    
    def test_collate_single_view(self):
        """Test collation of single-view batch."""
        batch = [
            self.create_sample(100),
            self.create_sample(150),
            self.create_sample(120)
        ]
        
        collated = collate_fn(batch)
        
        assert collated['coord'].shape == (370, 3)  # 100 + 150 + 120
        assert collated['feat'].shape == (370, 6)
        assert len(collated['offset']) == 4  # batch_size + 1
        assert collated['offset'].tolist() == [0, 100, 250, 370]
        assert len(collated['hierarchies']) == 3
        assert len(collated['scene_ids']) == 3
    
    def test_collate_dual_view(self):
        """Test collation of dual-view batch."""
        batch = [
            {
                'view1': self.create_sample(100),
                'view2': self.create_sample(100),  # Same hierarchy, different transform
                'scene_id': 'scene_A'
            },
            {
                'view1': self.create_sample(150),
                'view2': self.create_sample(150),
                'scene_id': 'scene_B'
            }
        ]
        
        collated = collate_fn(batch)
        
        assert 'view1' in collated
        assert 'view2' in collated
        assert collated['view1']['coord'].shape[0] == 250  # 100 + 150
        assert collated['view2']['coord'].shape[0] == 250
        assert len(collated['hierarchies']) == 2
    
    def test_extract_scene_from_batch(self):
        """Test extracting individual scene from batch."""
        batch = [
            self.create_sample(100),
            self.create_sample(150),
            self.create_sample(120)
        ]
        
        collated = collate_fn(batch)
        
        # Extract second scene
        scene = extract_scene_from_batch(collated, scene_idx=1)
        
        assert scene['coord'].shape == (150, 3)
        assert scene['feat'].shape == (150, 6)
        assert isinstance(scene['hierarchy'], Region)
        assert scene['scene_id'] == collated['scene_ids'][1]
    
    def test_batch_stats(self):
        """Test batch statistics computation."""
        batch = [
            self.create_sample(100),
            self.create_sample(150),
            self.create_sample(120)
        ]
        
        collated = collate_fn(batch)
        stats = compute_batch_stats(collated)
        
        assert stats['num_scenes'] == 3
        assert stats['total_points'] == 370
        assert stats['mean_points'] == pytest.approx(123.33, rel=0.01)
        assert stats['min_points'] == 100
        assert stats['max_points'] == 150


class TestDataLoader:
    """Test DataLoader integration."""
    
    def create_dummy_dataset(self, tmp_dir, num_scenes=5):
        """Create dummy dataset in combined format."""
        split_dir = tmp_dir / 'train'
        split_dir.mkdir()
        
        for i in range(num_scenes):
            num_points = 100 + i * 20
            
            # Create hierarchy dict
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
    
    def test_dataloader_iteration(self, tmp_path):
        """Test iterating through DataLoader."""
        data_root = self.create_dummy_dataset(tmp_path)
        
        dataset = DBBDDataset(
            data_root=str(data_root),
            split='train',
            transform=ToTensor()
        )
        
        loader = create_dataloader(
            dataset,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            shuffle=False
        )
        
        batch_count = 0
        total_points = 0
        
        for batch in loader:
            batch_count += 1
            total_points += batch['coord'].shape[0]
            
            assert 'coord' in batch
            assert 'feat' in batch
            assert 'offset' in batch
            assert 'hierarchies' in batch
            assert len(batch['hierarchies']) <= 2  # batch_size
        
        assert batch_count > 0
        assert total_points > 0
    
    def test_dataloader_dual_view(self, tmp_path):
        """Test DataLoader with dual-view dataset."""
        data_root = self.create_dummy_dataset(tmp_path, num_scenes=4)
        
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
        
        for batch in loader:
            assert 'view1' in batch
            assert 'view2' in batch
            assert 'hierarchies' in batch
            
            # Both views should have same batch structure
            assert batch['view1']['coord'].shape[0] == batch['view2']['coord'].shape[0]
            break  # Just test first batch


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def create_test_hierarchy(self, num_points=200):
        """Create a simple 3-level hierarchy."""
        # Level 0: root
        root = Region(indices=np.arange(num_points), center_idx=0, level=0)
        
        # Level 1: 2 children
        child1 = Region(indices=np.arange(num_points // 2), center_idx=0, level=1, parent=root)
        child2 = Region(indices=np.arange(num_points // 2, num_points), center_idx=num_points // 2, level=1, parent=root)
        
        # Level 2: 4 grandchildren
        gc1 = Region(indices=np.arange(num_points // 4), center_idx=0, level=2, parent=child1)
        gc2 = Region(indices=np.arange(num_points // 4, num_points // 2), center_idx=num_points // 4, level=2, parent=child1)
        gc3 = Region(indices=np.arange(num_points // 2, 3 * num_points // 4), center_idx=num_points // 2, level=2, parent=child2)
        gc4 = Region(indices=np.arange(3 * num_points // 4, num_points), center_idx=3 * num_points // 4, level=2, parent=child2)
        
        child1.children = [gc1, gc2]
        child2.children = [gc3, gc4]
        root.children = [child1, child2]
        
        return root
    
    def test_hierarchical_feature_flow(self):
        """Test features flow through hierarchy with propagator and aggregator."""
        prop = FeaturePropagator(parent_dim=96, coord_dim=3, out_dim=96)
        agg = FeatureAggregator(feat_dim=96, mode='max')
        
        # Create data
        coords = torch.randn(200, 3)
        hierarchy = self.create_test_hierarchy(num_points=200)
        
        # Simulate encoding: start with random root feature
        root_feat = torch.randn(96)
        
        # Global-to-local: propagate from root to children
        child_features = []
        for child in hierarchy.children:
            child_coords = coords[child.indices]
            child_feat = prop(root_feat, child_coords)
            
            # Simulate encoding (just take mean for testing)
            encoded = child_feat.mean(dim=0)
            child_features.append(encoded)
        
        # Local-to-global: aggregate children back to parent
        child_features_stacked = torch.stack(child_features)
        aggregated = agg(child_features_stacked)
        
        assert aggregated.shape == (96,)
        assert not torch.isnan(aggregated).any()
    
    def create_dummy_dataset(self, tmp_dir, num_scenes=5):
        """Helper to create dummy dataset in combined format."""
        split_dir = tmp_dir / 'train'
        split_dir.mkdir()
        
        for i in range(num_scenes):
            num_points = 100 + i * 20
            
            # Create hierarchy dict with 3 levels
            hierarchy_dict = {
                'indices': np.arange(num_points),
                'center_idx': 0,
                'level': 0,
                'children': [
                    {
                        'indices': np.arange(num_points // 2),
                        'center_idx': 0,
                        'level': 1,
                        'children': [
                            {
                                'indices': np.arange(num_points // 4),
                                'center_idx': 0,
                                'level': 2,
                                'children': []
                            },
                            {
                                'indices': np.arange(num_points // 4, num_points // 2),
                                'center_idx': num_points // 4,
                                'level': 2,
                                'children': []
                            }
                        ]
                    },
                    {
                        'indices': np.arange(num_points // 2, num_points),
                        'center_idx': num_points // 2,
                        'level': 1,
                        'children': [
                            {
                                'indices': np.arange(num_points // 2, 3 * num_points // 4),
                                'center_idx': num_points // 2,
                                'level': 2,
                                'children': []
                            },
                            {
                                'indices': np.arange(3 * num_points // 4, num_points),
                                'center_idx': 3 * num_points // 4,
                                'level': 2,
                                'children': []
                            }
                        ]
                    }
                ]
            }
            
            # Create combined data
            data = {
                'coords': torch.randn(num_points, 3),
                'normals': torch.rand(num_points, 3),
                'hierarchy': hierarchy_dict,
                'num_points': num_points,
                'total_regions': 7,
                'max_depth_reached': 2
            }
            
            with open(split_dir / f'scene{i:04d}.pkl', 'wb') as f:
                pickle.dump(data, f)
        
        return tmp_dir
    
    def test_memory_efficiency(self, tmp_path):
        """Test that hierarchies don't cause memory leaks."""
        data_root = self.create_dummy_dataset(tmp_path, num_scenes=10)
        
        dataset = DBBDDataset(
            data_root=str(data_root),
            split='train',
            cache_hierarchies=True,
            cache_size=5  # Small cache
        )
        
        # Access all samples multiple times
        for epoch in range(3):
            for i in range(len(dataset)):
                sample = dataset[i]
                assert 'hierarchy' in sample
        
        # Cache should not grow beyond limit
        assert len(dataset.hierarchy_cache) <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
