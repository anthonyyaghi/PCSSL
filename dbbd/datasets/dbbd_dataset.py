"""
DBBD Dataset

Loads point clouds with precomputed hierarchical decompositions.
Supports dual-view augmentation for contrastive learning.
"""

import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging

from ..models.utils.hierarchy import Region, dict_to_region, HierarchyCache, validate_hierarchy

logger = logging.getLogger(__name__)


class DBBDDataset(Dataset):
    """
    Dataset for loading point clouds with precomputed hierarchies.
    
    Supports:
    - Loading combined point cloud + hierarchy data from single .pkl files
    - Data organized in train/val/test split directories
    - Applying augmentation transforms
    - Dual-view mode for contrastive learning
    
    Expected directory structure:
        data_root/
            train/
                scene0000_00.pkl
                scene0000_01.pkl
                ...
            val/
                scene0011_00.pkl
                ...
            test/
                scene0707_00.pkl
                ...
    
    Each .pkl file contains:
        - coords: (N, 3) tensor - point coordinates
        - normals: (N, 3) tensor - point normals (used as features)
        - hierarchy: dict - hierarchical decomposition
        - num_points: int
        - total_regions: int
        - max_depth_reached: int
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        dual_view: bool = False,
        cache_hierarchies: bool = True,
        cache_size: int = 100,
        validate_hierarchies: bool = False,
        data_suffix: str = '.pkl',
        max_scenes: Optional[int] = None
    ):
        """
        Initialize DBBD dataset.
        
        Args:
            data_root: Root directory containing split subdirectories (train/val/test)
            split: Dataset split ('train', 'val', 'test')
            transform: Transform or Compose to apply to each sample
            dual_view: If True, return two augmented views per sample
            cache_hierarchies: Whether to cache recently used hierarchies
            cache_size: Maximum number of hierarchies to cache
            validate_hierarchies: Whether to validate hierarchy structure on load
            data_suffix: File extension for data files
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.dual_view = dual_view
        self.validate_hierarchies = validate_hierarchies
        self.data_suffix = data_suffix
        
        # Construct split directory path
        self.split_dir = self.data_root / split
        
        # Validate directory exists
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        # Initialize hierarchy cache
        self.cache_hierarchies = cache_hierarchies
        if cache_hierarchies:
            self.hierarchy_cache = HierarchyCache(max_size=cache_size)
        else:
            self.hierarchy_cache = None
        
        # Find all data files
        self.data_files = sorted(list(self.split_dir.glob(f'*{data_suffix}')))

        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in {self.split_dir} with suffix {data_suffix}")

        if max_scenes is not None:
            self.data_files = self.data_files[:max_scenes]
        
        logger.info(
            f"Initialized DBBDDataset: {len(self.data_files)} scenes from {self.split_dir}"
        )
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def _load_data(self, idx: int) -> Dict[str, Any]:
        """
        Load combined point cloud and hierarchy data from file.
        
        Args:
            idx: Dataset index
        
        Returns:
            Dictionary with 'coord', 'feat', 'hierarchy', and metadata
        """
        data_path = self.data_files[idx]
        
        # Load pickle file
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Extract point cloud data
        coords = raw_data['coords']
        normals = raw_data['normals']
        
        # Convert to numpy if tensor
        if hasattr(coords, 'numpy'):
            coords = coords.numpy()
        if hasattr(normals, 'numpy'):
            normals = normals.numpy()
        
        # Ensure float32
        coords = coords.astype(np.float32)
        normals = normals.astype(np.float32)
        
        # Build data dict with standardized keys
        data = {
            'coord': coords,
            'feat': normals,
            'num_points': raw_data.get('num_points', len(coords)),
            'total_regions': raw_data.get('total_regions', 0),
            'max_depth': raw_data.get('max_depth_reached', 0)
        }
        
        # Load/cache hierarchy
        hierarchy = self._load_hierarchy(idx, raw_data)
        data['hierarchy'] = hierarchy
        data['scene_id'] = data_path.stem
        
        return data
    
    def _load_hierarchy(self, idx: int, raw_data: Dict) -> Region:
        """
        Load or retrieve cached hierarchy for a scene.
        
        Args:
            idx: Dataset index
            raw_data: Already loaded raw data dict
        
        Returns:
            Region tree structure
        """
        data_path = self.data_files[idx]
        cache_key = str(data_path)
        
        # Try to get from cache
        if self.hierarchy_cache is not None:
            cached = self.hierarchy_cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Convert dict hierarchy to Region objects
        hierarchy_dict = raw_data['hierarchy']
        hierarchy = dict_to_region(hierarchy_dict)
        
        # Optionally validate
        if self.validate_hierarchies:
            num_points = raw_data.get('num_points', len(raw_data['coords']))
            validate_hierarchy(hierarchy, num_points=num_points)
        
        # Add to cache
        if self.hierarchy_cache is not None:
            self.hierarchy_cache.put(cache_key, hierarchy)
        
        return hierarchy
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample or dual-view sample.
        
        Args:
            idx: Dataset index
        
        Returns:
            If dual_view=False:
                Dictionary with 'coord', 'feat', 'hierarchy', 'scene_id'
            If dual_view=True:
                Dictionary with 'view1', 'view2', 'scene_id'
        """
        # Load combined data
        data = self._load_data(idx)
        hierarchy = data['hierarchy']
        
        # Apply transforms
        if self.dual_view:
            # Dual-view mode: return two differently augmented views
            import copy
            
            # Create two independent copies
            data1 = copy.deepcopy(data)
            data2 = copy.deepcopy(data)
            
            # Apply transforms (each will have different random state)
            if self.transform is not None:
                data1 = self.transform(data1)
                data2 = self.transform(data2)
            
            # Return dual-view format
            # Hierarchy is shared (index-based, invariant to transforms)
            output = {
                'view1': {
                    'coord': data1['coord'],
                    'feat': data1['feat'],
                    'hierarchy': hierarchy,  # Same for both views
                    'scene_id': data['scene_id']
                },
                'view2': {
                    'coord': data2['coord'],
                    'feat': data2['feat'],
                    'hierarchy': hierarchy,  # Same for both views
                    'scene_id': data['scene_id']
                },
                'scene_id': data['scene_id']
            }
            
            return output
        
        else:
            # Single view mode
            if self.transform is not None:
                data = self.transform(data)
            
            return data
    
    def get_scene_id(self, idx: int) -> str:
        """Get scene ID for a given index."""
        return self.data_files[idx].stem
    
    def get_data_path(self, idx: int) -> Path:
        """Get data file path for a given index."""
        return self.data_files[idx]
    
    def __repr__(self) -> str:
        return (
            f"DBBDDataset(split='{self.split}', num_scenes={len(self)}, "
            f"dual_view={self.dual_view}, "
            f"data_root='{self.data_root}')"
        )


def create_dataloader(
    dataset: DBBDDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for DBBD dataset with custom collation.
    
    Args:
        dataset: DBBDDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader instance
    """
    from ..models.utils.batch import collate_fn
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return loader
