"""
Batching Utilities

Custom collation functions for batching variable-size point clouds
with hierarchical structures.
"""

import torch
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching point clouds with hierarchies.
    
    Handles variable-size point clouds by concatenating coordinates and features,
    computing offset boundaries, and preserving hierarchies as a list.
    
    Args:
        batch: List of sample dictionaries from dataset.__getitem__()
               Each sample contains:
               - 'coord': (N_i, 3) tensor of coordinates
               - 'feat': (N_i, C) tensor of features
               - 'hierarchy': Region tree structure
               - 'scene_id': str identifier
               - Optional: 'segment', 'semantic_gt', etc.
    
    Returns:
        Batched dictionary with:
        - 'coord': (sum(N_i), 3) concatenated coordinates
        - 'feat': (sum(N_i), C) concatenated features
        - 'offset': (B+1,) boundary indices [0, N_0, N_0+N_1, ...]
        - 'hierarchies': List[Region] of length B
        - 'scene_ids': List[str] of length B
        - Optional fields from input samples
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")
    
    # Handle dual-view case
    if 'view1' in batch[0] and 'view2' in batch[0]:
        return collate_dual_view(batch)
    
    # Single view collation
    batch_size = len(batch)
    
    # Extract and stack coordinates
    coords = []
    feats = []
    hierarchies = []
    scene_ids = []
    offsets = [0]
    
    # Optional fields (may not be present in all samples)
    optional_fields = {}
    
    for sample in batch:
        coord = sample['coord']
        feat = sample['feat']
        
        # Convert to tensors if needed
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord).float()
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).float()
        
        # Validate shapes
        if coord.dim() != 2 or coord.shape[1] != 3:
            raise ValueError(f"Expected coord shape (N, 3), got {coord.shape}")
        if feat.dim() != 2:
            raise ValueError(f"Expected feat shape (N, C), got {feat.shape}")
        if coord.shape[0] != feat.shape[0]:
            raise ValueError(
                f"Coord and feat must have same N: {coord.shape[0]} != {feat.shape[0]}"
            )
        
        num_points = coord.shape[0]
        
        coords.append(coord)
        feats.append(feat)
        hierarchies.append(sample['hierarchy'])
        scene_ids.append(sample.get('scene_id', f'scene_{len(scene_ids)}'))
        offsets.append(offsets[-1] + num_points)
        
        # Collect optional fields
        for key in sample:
            if key not in ['coord', 'feat', 'hierarchy', 'scene_id']:
                if key not in optional_fields:
                    optional_fields[key] = []
                
                value = sample[key]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                optional_fields[key].append(value)
    
    # Concatenate coordinates and features
    coord = torch.cat(coords, dim=0)
    feat = torch.cat(feats, dim=0)
    offset = torch.tensor(offsets, dtype=torch.long)
    
    # Build output dictionary
    output = {
        'coord': coord,
        'feat': feat,
        'offset': offset,
        'hierarchies': hierarchies,
        'scene_ids': scene_ids,
        'batch_size': batch_size
    }
    
    # Add optional fields (concatenated)
    for key, values in optional_fields.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            # Check if all have same shape (except first dimension)
            shapes = [v.shape[1:] for v in values]
            if all(s == shapes[0] for s in shapes):
                output[key] = torch.cat(values, dim=0)
            else:
                # Keep as list if shapes don't match
                output[key] = values
        else:
            # Keep as list for non-tensor values
            output[key] = values
    
    logger.debug(
        f"Collated batch: {batch_size} scenes, "
        f"{coord.shape[0]} total points, "
        f"mean {coord.shape[0] // batch_size} points/scene"
    )
    
    return output


def collate_dual_view(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for dual-view batches (contrastive learning).
    
    Each sample contains 'view1' and 'view2' with different augmentations
    of the same scene. Both views share the same hierarchy.
    
    Args:
        batch: List of dual-view samples, each with:
               - 'view1': Dict with coord, feat, hierarchy
               - 'view2': Dict with coord, feat, hierarchy
               - 'scene_id': str identifier
    
    Returns:
        Batched dictionary with:
        - 'view1': Collated view1 data (same format as collate_fn output)
        - 'view2': Collated view2 data
        - 'scene_ids': List[str] of length B
        
        Note: Both views share the same hierarchies (index-based, augmentation-invariant)
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")
    
    # Extract views
    view1_batch = [sample['view1'] for sample in batch]
    view2_batch = [sample['view2'] for sample in batch]
    scene_ids = [sample.get('scene_id', f'scene_{i}') for i, sample in enumerate(batch)]
    
    # Collate each view separately
    view1_collated = collate_fn(view1_batch)
    view2_collated = collate_fn(view2_batch)
    
    # Hierarchies should be identical (same structure, index-based)
    # We keep them in view1 and remove from view2 to avoid duplication
    hierarchies = view1_collated['hierarchies']
    
    output = {
        'view1': view1_collated,
        'view2': view2_collated,
        'hierarchies': hierarchies,  # Shared between views
        'scene_ids': scene_ids,
        'batch_size': len(batch)
    }
    
    logger.debug(
        f"Collated dual-view batch: {len(batch)} scenes, "
        f"view1: {view1_collated['coord'].shape[0]} points, "
        f"view2: {view2_collated['coord'].shape[0]} points"
    )
    
    return output


def extract_scene_from_batch(
    batch: Dict[str, Any],
    scene_idx: int
) -> Dict[str, Any]:
    """
    Extract a single scene from a batched dictionary.
    
    Useful for processing individual scenes from a batch.
    
    Args:
        batch: Batched dictionary from collate_fn
        scene_idx: Index of scene to extract (0 to batch_size-1)
    
    Returns:
        Dictionary with unbatched data for one scene:
        - 'coord': (N, 3) coordinates for this scene
        - 'feat': (N, C) features for this scene
        - 'hierarchy': Region tree for this scene
        - 'scene_id': str identifier
    """
    if scene_idx < 0 or scene_idx >= batch['batch_size']:
        raise ValueError(
            f"Invalid scene_idx {scene_idx} for batch of size {batch['batch_size']}"
        )
    
    # Get boundaries for this scene
    start_idx = batch['offset'][scene_idx].item()
    end_idx = batch['offset'][scene_idx + 1].item()
    
    # Extract data for this scene
    output = {
        'coord': batch['coord'][start_idx:end_idx],
        'feat': batch['feat'][start_idx:end_idx],
        'hierarchy': batch['hierarchies'][scene_idx],
        'scene_id': batch['scene_ids'][scene_idx]
    }
    
    # Extract optional fields if present
    for key in batch:
        if key not in ['coord', 'feat', 'offset', 'hierarchies', 'scene_ids', 'batch_size']:
            if isinstance(batch[key], torch.Tensor):
                # Assume same indexing as coord/feat
                output[key] = batch[key][start_idx:end_idx]
            elif isinstance(batch[key], list) and len(batch[key]) == batch['batch_size']:
                # Per-scene list
                output[key] = batch[key][scene_idx]
    
    return output


def split_batch_by_offset(
    data: torch.Tensor,
    offset: torch.Tensor
) -> List[torch.Tensor]:
    """
    Split batched tensor back into individual scenes using offset.
    
    Args:
        data: (sum(N_i), ...) batched tensor
        offset: (B+1,) offset boundaries
    
    Returns:
        List of B tensors, each (N_i, ...)
    """
    scenes = []
    batch_size = len(offset) - 1
    
    for i in range(batch_size):
        start = offset[i].item()
        end = offset[i + 1].item()
        scenes.append(data[start:end])
    
    return scenes


def compute_batch_stats(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute statistics about a batch.
    
    Args:
        batch: Batched dictionary from collate_fn
    
    Returns:
        Statistics dictionary with:
        - num_scenes: Number of scenes in batch
        - total_points: Total number of points
        - points_per_scene: List of point counts per scene
        - mean_points: Average points per scene
        - min_points: Minimum points in any scene
        - max_points: Maximum points in any scene
    """
    offset = batch['offset']
    batch_size = len(offset) - 1
    
    points_per_scene = [
        (offset[i + 1] - offset[i]).item()
        for i in range(batch_size)
    ]
    
    stats = {
        'num_scenes': batch_size,
        'total_points': batch['coord'].shape[0],
        'points_per_scene': points_per_scene,
        'mean_points': np.mean(points_per_scene),
        'min_points': min(points_per_scene),
        'max_points': max(points_per_scene),
        'coord_shape': tuple(batch['coord'].shape),
        'feat_shape': tuple(batch['feat'].shape)
    }
    
    return stats
