"""
Feature Collector

Collects features from G2L and L2G branches across scenes and 
organizes them for loss computation in Phase 4.
"""

import torch
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FeatureCollector:
    """
    Collects and batches features from hierarchical encoding.
    
    Tracks features by level for both G2L and L2G branches,
    and computes offset boundaries for each level.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset collector for new batch."""
        # Per-level features
        self.g2l_by_level: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.l2g_by_level: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
        # Scene counts per level
        self.counts_by_level: Dict[str, List[int]] = defaultdict(list)
        
        # Point-level features
        self.point_feats_g2l: List[torch.Tensor] = []
        self.point_feats_l2g: List[torch.Tensor] = []
        self.point_counts: List[int] = []
    
    def add_scene(
        self,
        g2l: Dict[str, torch.Tensor],
        l2g: Dict[str, torch.Tensor],
        point_feats_g2l: Optional[torch.Tensor] = None,
        point_feats_l2g: Optional[torch.Tensor] = None
    ):
        """
        Add features from one scene.
        
        Args:
            g2l: Dict mapping level names to G2L features
            l2g: Dict mapping level names to L2G features
            point_feats_g2l: Optional (N, D) point features from G2L
            point_feats_l2g: Optional (N, D) point features from L2G
        """
        # Collect all level names
        all_levels = set(g2l.keys()) | set(l2g.keys())
        
        # Remove internal keys like '_point_features'
        all_levels = {k for k in all_levels if not k.startswith('_')}
        
        for level in all_levels:
            if level in g2l:
                self.g2l_by_level[level].append(g2l[level])
                self.counts_by_level[level].append(g2l[level].shape[0])
            if level in l2g:
                self.l2g_by_level[level].append(l2g[level])
        
        # Add point features
        if point_feats_g2l is not None:
            self.point_feats_g2l.append(point_feats_g2l)
            self.point_counts.append(point_feats_g2l.shape[0])
        if point_feats_l2g is not None:
            self.point_feats_l2g.append(point_feats_l2g)
    
    def get_output(self) -> Dict[str, torch.Tensor]:
        """
        Get collected features in Phase 4 format.
        
        Returns:
            Dict with:
            - 'g2l': Dict of level -> (N, D) tensors
            - 'l2g': Dict of level -> (N, D) tensors
            - 'offsets_by_level': Dict of level -> (B+1,) offset tensors
            - 'point_feats_g2l': (total_points, D) tensor
            - 'point_feats_l2g': (total_points, D) tensor
            - 'point_offset': (B+1,) tensor
        """
        output = {
            'g2l': {},
            'l2g': {},
            'offsets_by_level': {}
        }
        
        # Concatenate per-level features and compute offsets
        for level in self.g2l_by_level.keys():
            # G2L features
            if self.g2l_by_level[level]:
                output['g2l'][level] = torch.cat(self.g2l_by_level[level], dim=0)
            
            # L2G features
            if self.l2g_by_level[level]:
                output['l2g'][level] = torch.cat(self.l2g_by_level[level], dim=0)
            
            # Offsets for this level
            counts = self.counts_by_level[level]
            offsets = [0]
            for c in counts:
                offsets.append(offsets[-1] + c)
            output['offsets_by_level'][level] = torch.tensor(offsets, dtype=torch.long)
        
        # Point features
        if self.point_feats_g2l:
            output['point_feats_g2l'] = torch.cat(self.point_feats_g2l, dim=0)
        if self.point_feats_l2g:
            output['point_feats_l2g'] = torch.cat(self.point_feats_l2g, dim=0)
        
        # Point offsets
        if self.point_counts:
            offsets = [0]
            for c in self.point_counts:
                offsets.append(offsets[-1] + c)
            output['point_offset'] = torch.tensor(offsets, dtype=torch.long)
        
        return output
    
    def __len__(self):
        """Number of scenes collected."""
        return len(self.point_counts)
    
    def __repr__(self):
        levels = sorted(self.g2l_by_level.keys())
        return (
            f"FeatureCollector(scenes={len(self)}, "
            f"levels={levels})"
        )
