"""
Hierarchy Data Structures and Utilities

Handles loading, validation, and manipulation of hierarchical tree structures
for point cloud regions.
"""

import pickle
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """
    Hierarchical region node in the point cloud decomposition tree.
    
    Attributes:
        indices: Global indices of points in this region (np.ndarray)
        center_idx: Index of the representative center point (from FPS)
        level: Depth in the hierarchy (0=root, higher=finer)
        children: List of child Region objects (empty for leaves)
        parent: Reference to parent Region (None for root)
    """
    indices: np.ndarray
    center_idx: int
    level: int
    children: List['Region'] = field(default_factory=list)
    parent: Optional['Region'] = None
    
    def __post_init__(self):
        """Validate region after initialization."""
        if not isinstance(self.indices, np.ndarray):
            self.indices = np.array(self.indices)
        if self.level < 0:
            raise ValueError(f"Level must be non-negative, got {self.level}")
        if self.center_idx < 0:
            raise ValueError(f"Center index must be non-negative, got {self.center_idx}")
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf region (no children)."""
        return len(self.children) == 0
    
    @property
    def num_points(self) -> int:
        """Number of points in this region."""
        return len(self.indices)
    
    @property
    def num_children(self) -> int:
        """Number of child regions."""
        return len(self.children)
    
    def get_all_descendants(self) -> List['Region']:
        """
        Get all descendant regions (recursive depth-first).
        
        Returns:
            List of all descendant regions including self
        """
        descendants = [self]
        for child in self.children:
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def get_depth(self) -> int:
        """
        Calculate max depth of subtree rooted at this region.
        
        Returns:
            Maximum depth (0 for leaf, >0 for internal nodes)
        """
        if self.is_leaf:
            return 0
        return 1 + max(child.get_depth() for child in self.children)
    
    def __repr__(self) -> str:
        return (f"Region(level={self.level}, points={self.num_points}, "
                f"children={self.num_children}, center={self.center_idx})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Region):
            return NotImplemented
        
        if self is other:
            return True
            
        if (self.center_idx != other.center_idx or 
            self.level != other.level):
            return False
            
        if not np.array_equal(self.indices, other.indices):
            return False
            
        if len(self.children) != len(other.children):
            return False
            
        for c1, c2 in zip(self.children, other.children):
            if c1 != c2:
                return False
                
        return True


def dict_to_region(
    hierarchy_dict: dict,
    parent: Optional[Region] = None
) -> Region:
    """
    Convert a dictionary-based hierarchy to Region objects.
    
    Args:
        hierarchy_dict: Dictionary with keys:
            - 'indices': np.ndarray of point indices
            - 'center_idx': int, center point index
            - 'level': int, hierarchy level
            - 'children': list of child dicts (optional)
        parent: Parent Region (for linking)
    
    Returns:
        Region object with children recursively converted
    """
    # Extract required fields
    indices = hierarchy_dict['indices']
    if hasattr(indices, 'numpy'):
        indices = indices.numpy()
    indices = np.asarray(indices)
    
    center_idx = int(hierarchy_dict['center_idx'])
    level = int(hierarchy_dict['level'])
    
    # Create the region
    region = Region(
        indices=indices,
        center_idx=center_idx,
        level=level,
        parent=parent
    )
    
    # Recursively convert children
    if 'children' in hierarchy_dict and hierarchy_dict['children']:
        region.children = [
            dict_to_region(child_dict, parent=region)
            for child_dict in hierarchy_dict['children']
        ]
    
    return region


def load_hierarchy(hierarchy_path: str, validate: bool = True) -> Region:
    """
    Load hierarchical region tree from pickle file.
    
    Args:
        hierarchy_path: Path to .pkl file containing Region tree
        validate: Whether to validate tree structure after loading
    
    Returns:
        Root Region node of the hierarchy tree
    
    Raises:
        FileNotFoundError: If hierarchy file doesn't exist
        ValueError: If loaded object is not a Region or validation fails
        pickle.UnpicklingError: If file is corrupted
    """
    hierarchy_path = Path(hierarchy_path)
    
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"Hierarchy file not found: {hierarchy_path}")
    
    try:
        with open(hierarchy_path, 'rb') as f:
            root = pickle.load(f)
    except Exception as e:
        raise pickle.UnpicklingError(f"Failed to load hierarchy from {hierarchy_path}: {e}")
    
    if not isinstance(root, Region):
        raise ValueError(f"Expected Region object, got {type(root)}")
    
    if validate:
        validate_hierarchy(root)
    
    logger.debug(f"Loaded hierarchy from {hierarchy_path}: {root}")
    return root


def validate_hierarchy(root: Region, num_points: Optional[int] = None) -> None:
    """
    Validate hierarchical tree structure.
    
    Checks:
    - No cycles in parent-child relationships
    - All indices are valid (within [0, num_points) if specified)
    - No duplicate indices across siblings
    - Level numbering is consistent
    - Parent-child relationships are bidirectional
    
    Args:
        root: Root region of the hierarchy
        num_points: Total number of points (for index validation)
    
    Raises:
        ValueError: If validation fails
    """
    visited = set()
    all_indices = set()
    
    def _validate_node(node: Region, expected_level: int):
        """Recursively validate a node and its subtree."""
        # Check for cycles
        node_id = id(node)
        if node_id in visited:
            raise ValueError(f"Cycle detected in hierarchy at {node}")
        visited.add(node_id)
        
        # Check level consistency
        if node.level != expected_level:
            raise ValueError(
                f"Level mismatch: expected {expected_level}, got {node.level} for {node}"
            )
        
        # Check indices validity
        if num_points is not None:
            if np.any(node.indices < 0) or np.any(node.indices >= num_points):
                raise ValueError(
                    f"Invalid indices in {node}: must be in [0, {num_points})"
                )
        
        # Check center index is in region
        if node.center_idx not in node.indices:
            raise ValueError(
                f"Center index {node.center_idx} not in region indices for {node}"
            )
        
        # Check for duplicate indices within this region
        if len(node.indices) != len(np.unique(node.indices)):
            raise ValueError(f"Duplicate indices within region {node}")
        
        # Check children don't overlap (at same level)
        child_indices = []
        for i, child in enumerate(node.children):
            # Validate parent reference
            if child.parent is not None and child.parent is not node:
                raise ValueError(
                    f"Child {i} of {node} has incorrect parent reference"
                )
            
            # Collect child indices
            child_indices.extend(child.indices.tolist())
            
            # Recursively validate child
            _validate_node(child, expected_level + 1)
        
        # Check children don't have duplicate indices
        if len(child_indices) != len(set(child_indices)):
            raise ValueError(f"Overlapping indices among children of {node}")
        
        # Check children's indices are subset of parent's indices
        if child_indices:
            child_set = set(child_indices)
            parent_set = set(node.indices.tolist())
            if not child_set.issubset(parent_set):
                extra = child_set - parent_set
                raise ValueError(
                    f"Children of {node} contain indices not in parent: {extra}"
                )
    
    # Start validation from root
    _validate_node(root, expected_level=0)
    
    logger.debug(f"Hierarchy validation passed: {len(visited)} nodes validated")


def get_hierarchy_stats(root: Region) -> dict:
    """
    Compute statistics about the hierarchical tree.
    
    Args:
        root: Root region of the hierarchy
    
    Returns:
        Dictionary with statistics:
        - max_depth: Maximum depth of the tree
        - num_nodes: Total number of regions
        - num_leaves: Number of leaf regions
        - avg_branching: Average branching factor
        - points_per_level: List of point counts per level
    """
    all_regions = root.get_all_descendants()
    
    # Organize by level
    levels = {}
    for region in all_regions:
        if region.level not in levels:
            levels[region.level] = []
        levels[region.level].append(region)
    
    # Compute branching factors
    branching_factors = []
    for region in all_regions:
        if not region.is_leaf:
            branching_factors.append(region.num_children)
    
    stats = {
        'max_depth': root.get_depth(),
        'num_nodes': len(all_regions),
        'num_leaves': sum(1 for r in all_regions if r.is_leaf),
        'avg_branching': np.mean(branching_factors) if branching_factors else 0,
        'points_per_level': {
            level: sum(r.num_points for r in regions) 
            for level, regions in levels.items()
        },
        'regions_per_level': {
            level: len(regions) 
            for level, regions in levels.items()
        }
    }
    
    return stats


class HierarchyCache:
    """
    LRU cache for frequently accessed hierarchies.
    
    Useful when the same hierarchies are accessed repeatedly across epochs.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of hierarchies to cache
        """
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
    
    def get(self, path: str) -> Optional[Region]:
        """
        Get hierarchy from cache.
        
        Args:
            path: Path to hierarchy file
        
        Returns:
            Cached Region or None if not in cache
        """
        path = str(Path(path))
        if path in self._cache:
            # Update access order (move to end)
            self._access_order.remove(path)
            self._access_order.append(path)
            return self._cache[path]
        return None
    
    def put(self, path: str, hierarchy: Region) -> None:
        """
        Add hierarchy to cache.
        
        Args:
            path: Path to hierarchy file
            hierarchy: Region tree to cache
        """
        path = str(Path(path))
        
        # If already in cache, update access order
        if path in self._cache:
            self._access_order.remove(path)
        else:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
        
        self._cache[path] = hierarchy
        self._access_order.append(path)
    
    def clear(self) -> None:
        """Clear all cached hierarchies."""
        self._cache.clear()
        self._access_order.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __repr__(self) -> str:
        return f"HierarchyCache(size={len(self)}/{self.max_size})"
