# Phase 1: Offline Preprocessing - Summary

## Overview
Implemented the offline preprocessing pipeline to decompose 3D point clouds into spatial hierarchies. This phase prepares the data used for hierarchical contrastive learning in subsequent phases.

---

## Core Concepts

### 1. Spatial Decomposition
The point cloud is recursively partitioned into regions using Farthest Point Sampling (FPS) for center selection and K-Nearest Neighbors (KNN) or radius-based grouping for partitioning.

| Parameter | Value | Purpose |
|-------|---------|---|
| **Branching Factor** | 8 | Typical child nodes per region (octree-like) |
| **Max Depth** | 4 | Number of hierarchical levels |
| **Center Selection** | FPS | Ensures well-spread representative points |

### 2. Region Structure
Each node in the decomposition tree is a "Region" characterized by its point indices and a representative center.

| Attribute | Type | Purpose |
|-------|---------|---|
| `indices` | `np.ndarray` | Global indices of points belonging to the region |
| `center_idx` | `int` | Global index of the FPS-selected center point |
| `level` | `int` | Depth in hierarchy (0 = Root) |
| `children` | `List[Region]` | Nested child regions |

---

## Data Format

Phase 1 produces combined `.pkl` files for each scene (e.g., ScanNet scenes).

**File Contents:**
```python
{
    'coords': (N, 3),      # Full point cloud coordinates (torch.Tensor)
    'normals': (N, 3),     # Full point cloud normals (torch.Tensor)
    'hierarchy': dict,      # Nested dictionary representation of the Region tree
    'num_points': int,      # N
    'total_regions': int,   # Number of regions in hierarchy (~2,500)
    'max_depth_reached': int # Actual depth reached
}
```

---

## Key Files & Logic

| Component | Path | Purpose |
|-------|---------|---|
| **Preprocessing Script** | `tools/preprocess.py`* | Entry point for hierarchy generation |
| **Hierarchy Engine** | `tools/utils/hierarchy_builder.py`* | Logic for FPS, partitioning, and tree construction |
| **Data Schema** | `dbbd/models/utils/hierarchy.py` | Implementation of `Region` dataclass and validation |

*\*Note: Preprocessing scripts are often run offline and may reside in a separate tools/ directory or be provided as a dependency.*

---

## Results
- **Dataset Support**: Verified on ScanNet (1,201 train / 312 val scenes)
- **Efficiency**: Hierarchies are pre-computed once to eliminate runtime sampling overhead
- **Consistency**: Indices are preserved across levels to allow shared feature maps
