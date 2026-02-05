"""
Unit Tests for DBBD Dataset Components

Tests hierarchy loading, dataset functionality, and transforms.
"""

import pytest
import numpy as np
import torch
import pickle
import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from dbbd.models.utils.hierarchy import Region, dict_to_region, load_hierarchy, validate_hierarchy, get_hierarchy_stats
from dbbd.datasets.dbbd_dataset import DBBDDataset
from dbbd.datasets.transforms import (
    RandomRotate, RandomScale, RandomTranslation, ColorJitter,
    ChromaticAutoContrast, ToTensor, Compose
)


class TestRegion:
    """Test Region dataclass and methods."""
    
    def test_region_creation(self):
        """Test basic region creation."""
        indices = np.array([0, 1, 2, 3, 4])
        region = Region(indices=indices, center_idx=2, level=0)
        
        assert region.num_points == 5
        assert region.center_idx == 2
        assert region.level == 0
        assert region.is_leaf
        assert region.num_children == 0
    
    def test_region_with_children(self):
        """Test region with child regions."""
        # Create parent
        parent_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        parent = Region(indices=parent_indices, center_idx=0, level=0)
        
        # Create children
        child1 = Region(indices=np.array([0, 1, 2, 3]), center_idx=0, level=1, parent=parent)
        child2 = Region(indices=np.array([4, 5, 6, 7]), center_idx=4, level=1, parent=parent)
        
        parent.children = [child1, child2]
        
        assert not parent.is_leaf
        assert parent.num_children == 2
        assert child1.is_leaf
        assert child2.is_leaf
    
    def test_get_depth(self):
        """Test depth calculation."""
        # Create 3-level hierarchy
        root = Region(indices=np.arange(16), center_idx=0, level=0)
        
        child1 = Region(indices=np.arange(8), center_idx=0, level=1, parent=root)
        child2 = Region(indices=np.arange(8, 16), center_idx=8, level=1, parent=root)
        
        grandchild1 = Region(indices=np.arange(4), center_idx=0, level=2, parent=child1)
        grandchild2 = Region(indices=np.arange(4, 8), center_idx=4, level=2, parent=child1)
        
        child1.children = [grandchild1, grandchild2]
        root.children = [child1, child2]
        
        assert root.get_depth() == 2
        assert child1.get_depth() == 1
        assert child2.get_depth() == 0
        assert grandchild1.get_depth() == 0
    
    def test_get_all_descendants(self):
        """Test getting all descendant."""
        root = Region(indices=np.arange(8), center_idx=0, level=0)
        child1 = Region(indices=np.arange(4), center_idx=0, level=1, parent=root)
        child2 = Region(indices=np.arange(4, 8), center_idx=4, level=1, parent=root)
        root.children = [child1, child2]
        
        descendants = root.get_all_descendants()
        assert len(descendants) == 3  # root + 2 children
        assert root in descendants
        assert child1 in descendants
        assert child2 in descendants


class TestDictToRegion:
    """Test dict_to_region conversion."""
    
    def test_simple_conversion(self):
        """Test converting a simple hierarchy dict."""
        hierarchy_dict = {
            'indices': np.arange(100),
            'center_idx': 0,
            'level': 0,
            'children': []
        }
        
        region = dict_to_region(hierarchy_dict)
        
        assert region.num_points == 100
        assert region.center_idx == 0
        assert region.level == 0
        assert region.is_leaf
    
    def test_nested_conversion(self):
        """Test converting nested hierarchy dict."""
        hierarchy_dict = {
            'indices': np.arange(100),
            'center_idx': 0,
            'level': 0,
            'children': [
                {
                    'indices': np.arange(50),
                    'center_idx': 0,
                    'level': 1,
                    'children': []
                },
                {
                    'indices': np.arange(50, 100),
                    'center_idx': 50,
                    'level': 1,
                    'children': []
                }
            ]
        }
        
        region = dict_to_region(hierarchy_dict)
        
        assert region.num_points == 100
        assert region.num_children == 2
        assert region.children[0].level == 1
        assert region.children[0].parent is region
        assert region.children[1].parent is region
    
    def test_tensor_indices(self):
        """Test conversion with torch tensor indices."""
        hierarchy_dict = {
            'indices': torch.arange(50),
            'center_idx': 0,
            'level': 0,
            'children': []
        }
        
        region = dict_to_region(hierarchy_dict)
        
        assert region.num_points == 50
        assert isinstance(region.indices, np.ndarray)


class TestHierarchyValidation:
    """Test hierarchy validation logic."""
    
    def test_valid_hierarchy(self):
        """Test validation passes for valid hierarchy."""
        root = Region(indices=np.arange(100), center_idx=0, level=0)
        child1 = Region(indices=np.arange(50), center_idx=0, level=1, parent=root)
        child2 = Region(indices=np.arange(50, 100), center_idx=50, level=1, parent=root)
        root.children = [child1, child2]
        
        # Should not raise
        validate_hierarchy(root, num_points=100)
    
    def test_invalid_level(self):
        """Test validation fails for incorrect level."""
        root = Region(indices=np.arange(100), center_idx=0, level=0)
        child = Region(indices=np.arange(50), center_idx=0, level=5, parent=root)  # Wrong level
        root.children = [child]
        
        with pytest.raises(ValueError, match="Level mismatch"):
            validate_hierarchy(root, num_points=100)
    
    def test_invalid_indices(self):
        """Test validation fails for out-of-range indices."""
        root = Region(indices=np.array([0, 1, 2, 150]), center_idx=0, level=0)  # 150 out of range
        
        with pytest.raises(ValueError, match="Invalid indices"):
            validate_hierarchy(root, num_points=100)
    
    def test_overlapping_children(self):
        """Test validation fails for overlapping child indices."""
        root = Region(indices=np.arange(100), center_idx=0, level=0)
        child1 = Region(indices=np.arange(60), center_idx=0, level=1, parent=root)  # 0-59
        child2 = Region(indices=np.arange(40, 100), center_idx=40, level=1, parent=root)  # 40-99 (overlap!)
        root.children = [child1, child2]
        
        with pytest.raises(ValueError, match="Overlapping indices"):
            validate_hierarchy(root, num_points=100)


class TestHierarchyLoadSave:
    """Test hierarchy loading and saving."""
    
    def test_load_save_hierarchy(self):
        """Test saving and loading hierarchy."""
        # Create hierarchy
        root = Region(indices=np.arange(100), center_idx=0, level=0)
        child1 = Region(indices=np.arange(50), center_idx=0, level=1, parent=root)
        child2 = Region(indices=np.arange(50, 100), center_idx=50, level=1, parent=root)
        root.children = [child1, child2]
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
            pickle.dump(root, f)
        
        try:
            # Load back
            loaded_root = load_hierarchy(temp_path, validate=True)
            
            assert loaded_root.num_points == 100
            assert loaded_root.num_children == 2
            assert loaded_root.level == 0
        finally:
            Path(temp_path).unlink()


class TestTransforms:
    """Test augmentation transforms."""
    
    def create_sample_data(self):
        """Create sample point cloud data."""
        return {
            'coord': np.random.randn(100, 3).astype(np.float32),
            'feat': np.random.rand(100, 6).astype(np.float32)
        }
    
    def test_random_rotate(self):
        """Test random rotation."""
        data = self.create_sample_data()
        orig_coord = data['coord'].copy()
        
        transform = RandomRotate(angle=[-180, 180], axis='z', p=1.0)
        transformed = transform(data)
        
        # Check shape preserved
        assert transformed['coord'].shape == orig_coord.shape
        
        # Check coordinates actually changed
        assert not np.allclose(transformed['coord'], orig_coord)
    
    def test_random_scale(self):
        """Test random scaling."""
        data = self.create_sample_data()
        orig_coord = data['coord'].copy()
        
        transform = RandomScale(scale=(0.8, 1.2), p=1.0)
        transformed = transform(data)
        
        assert transformed['coord'].shape == orig_coord.shape
        assert not np.allclose(transformed['coord'], orig_coord)
    
    def test_color_jitter(self):
        """Test color jitter."""
        data = self.create_sample_data()
        orig_feat = data['feat'].copy()
        
        transform = ColorJitter(std=0.1, p=1.0)
        transformed = transform(data)
        
        assert transformed['feat'].shape == orig_feat.shape
        # RGB channels (first 3) should be modified
        assert not np.allclose(transformed['feat'][:, :3], orig_feat[:, :3])
    
    def test_compose(self):
        """Test composing multiple transforms."""
        data = self.create_sample_data()
        
        transform = Compose([
            RandomRotate(angle=[-180, 180], axis='z', p=1.0),
            RandomScale(scale=(0.9, 1.1), p=1.0),
            ColorJitter(std=0.05, p=1.0),
            ToTensor()
        ])
        
        transformed = transform(data)
        
        # Should be tensors now
        assert isinstance(transformed['coord'], torch.Tensor)
        assert isinstance(transformed['feat'], torch.Tensor)
    
    def test_to_tensor(self):
        """Test conversion to tensors."""
        data = self.create_sample_data()
        
        transform = ToTensor()
        transformed = transform(data)
        
        assert isinstance(transformed['coord'], torch.Tensor)
        assert isinstance(transformed['feat'], torch.Tensor)
        assert transformed['coord'].dtype == torch.float32
        assert transformed['feat'].dtype == torch.float32


class TestDataset:
    """Test DBBD dataset with combined format."""
    
    def create_dummy_data(self, tmp_dir, num_scenes=3):
        """Create dummy data files in combined format."""
        split_dir = tmp_dir / 'train'
        split_dir.mkdir()
        
        for i in range(num_scenes):
            num_points = 100 + i * 50
            
            # Create hierarchy dict
            hierarchy_dict = {
                'indices': np.arange(num_points),
                'center_idx': 0,
                'level': 0,
                'center_coords': np.zeros(3),
                'density': 1.0,
                'homogeneity': 1.0,
                'branching_factor': 2,
                'children': [
                    {
                        'indices': np.arange(num_points // 2),
                        'center_idx': 0,
                        'level': 1,
                        'center_coords': np.zeros(3),
                        'density': 1.0,
                        'homogeneity': 1.0,
                        'branching_factor': 0,
                        'children': []
                    },
                    {
                        'indices': np.arange(num_points // 2, num_points),
                        'center_idx': num_points // 2,
                        'level': 1,
                        'center_coords': np.zeros(3),
                        'density': 1.0,
                        'homogeneity': 1.0,
                        'branching_factor': 0,
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
    
    def test_dataset_creation(self, tmp_path):
        """Test dataset initialization."""
        data_root = self.create_dummy_data(tmp_path)
        
        dataset = DBBDDataset(
            data_root=str(data_root),
            split='train',
            dual_view=False
        )
        
        assert len(dataset) == 3
    
    def test_dataset_getitem(self, tmp_path):
        """Test loading a single sample."""
        data_root = self.create_dummy_data(tmp_path)
        
        dataset = DBBDDataset(
            data_root=str(data_root),
            split='train',
            dual_view=False,
            transform=ToTensor()
        )
        
        sample = dataset[0]
        
        assert 'coord' in sample
        assert 'feat' in sample
        assert 'hierarchy' in sample
        assert 'scene_id' in sample
        assert isinstance(sample['coord'], torch.Tensor)
        assert isinstance(sample['hierarchy'], Region)
    
    def test_dataset_dual_view(self, tmp_path):
        """Test dual-view mode."""
        data_root = self.create_dummy_data(tmp_path)
        
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
        
        sample = dataset[0]
        
        assert 'view1' in sample
        assert 'view2' in sample
        assert 'scene_id' in sample
        
        # Both views should have same number of points
        assert sample['view1']['coord'].shape[0] == sample['view2']['coord'].shape[0]
        
        # But coordinates should be different due to augmentation
        assert not torch.allclose(sample['view1']['coord'], sample['view2']['coord'])
        
        # Hierarchies should be identical (same object)
        assert sample['view1']['hierarchy'] is sample['view2']['hierarchy']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
