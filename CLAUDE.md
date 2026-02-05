# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DBBD (Dual Branch Bi-Directional) is a self-supervised learning framework for 3D point clouds using hierarchical contrastive learning. The system processes point clouds through bidirectional encoding branches (Global-to-Local and Local-to-Global) to learn multi-scale representations.

**Current Status:** Phase 4 Complete (v0.3.0) - All core components implemented including contrastive losses.

## Common Commands

```bash
# Install in development mode
pip install -e .
pip install -e ".[dev]"    # With dev tools

# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_loss.py -v
pytest tests/test_encoder.py -v
pytest tests/test_integration.py -v

# Run single test class
pytest tests/test_loss.py::TestInfoNCELoss -v

# With coverage
pytest tests/ --cov=dbbd --cov-report=html

# Run demo
python examples/phase2_demo.py
```

## Architecture

### Data Flow
```
Point Cloud (.pkl) → DBBDDataset → Dual-View Augmentation → HierarchicalEncoder
                                                                    ↓
                                                    ┌───────────────┴───────────────┐
                                                    ↓                               ↓
                                              G2L Branch                      L2G Branch
                                          (Top-down traversal)           (Bottom-up traversal)
                                                    ↓                               ↓
                                                    └───────────────┬───────────────┘
                                                                    ↓
                                                        DBBDContrastiveLoss
                                                     (Region + Point InfoNCE)
```

### Key Components

**Data Layer** (`dbbd/datasets/`)
- `DBBDDataset`: Loads combined `.pkl` files containing coords, normals, and hierarchy
- Hierarchy stored as nested dicts, converted to `Region` dataclass at load time
- Dual-view mode returns two augmented views of same scene for contrastive learning

**Feature Processing** (`dbbd/models/features/`)
- `FeaturePropagator`: G2L - propagates parent context to child points via MLP
- `FeatureAggregator`: L2G - aggregates child features to parent (max/mean/attention)

**Hierarchical Encoder** (`dbbd/models/encoder/`)
- `HierarchicalEncoder`: Main orchestrator running both G2L and L2G branches
- `PointCloudEncoder`: Encodes point clouds to region/point features (PointNet backbone)
- `G2LTraversal` / `L2GTraversal`: Recursive tree traversal with feature propagation/aggregation
- `ProjectionMLP`: Projects raw features to hidden dimension (96D default)

**Contrastive Loss** (`dbbd/models/loss/`)
- `DBBDContrastiveLoss`: Combined loss = α×L_region + β×L_point (default α=1.0, β=0.5)
- `InfoNCELoss`: Core contrastive loss using F.cross_entropy for numerical stability
- `ContrastiveProjectionHead`: Uses LayerNorm (not BatchNorm) to support batch_size=1

### Hierarchy Structure

The `Region` dataclass (`dbbd/models/utils/hierarchy.py`) represents tree nodes:
- `indices`: Point indices belonging to this region
- `center_idx`: Representative center point (from FPS)
- `level`: Depth in tree (0=root)
- `children`: List of child regions (up to 8, octree-like)
- `parent`: Reference to parent region

Key utilities: `dict_to_region()`, `validate_hierarchy()`, `get_hierarchy_stats()`

### Data Format

Each scene stored as `.pkl` with structure:
```python
{
    'coords': (N, 3) tensor,      # XYZ coordinates
    'normals': (N, 3) tensor,     # Point normals (used as features)
    'hierarchy': dict,             # Nested dict converted to Region tree
    'num_points': int,
    'total_regions': int,
    'max_depth_reached': int
}
```

Data directory: `datasets/scannet/{train,val,test}/*.pkl`

## Design Decisions

- **96-dim features**: Balances expressiveness vs memory, matches common encoder widths
- **8-way branching**: Octree-like spatial decomposition, ~2,500 regions per scene
- **LayerNorm in projection head**: Supports small batch sizes (batch_size=1)
- **Combined .pkl format**: Single file per scene ensures data-hierarchy consistency
- **Dict-to-Region conversion**: Portable pickle storage with clean API at runtime
