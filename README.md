# DBBD: Dual Branch Bi-Directional Self-Supervised Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)]()

## Overview

DBBD (Dual Branch Bi-Directional) is a self-supervised learning framework for 3D point clouds that learns rich multi-scale representations through hierarchical contrastive learning.

### Current Status: Phase 2 Complete ✓

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Offline preprocessing - hierarchical decompositions | ✅ Complete |
| Phase 2 | Data integration & feature processing | ✅ Complete |
| Phase 3 | Hierarchical encoding - bidirectional traversal | 🔜 Next |
| Phase 4 | Loss functions & training loop | ⏳ Planned |

## Phase 2 Components

### Data Integration
- `DBBDDataset`: Loads point clouds with precomputed hierarchies
- Dual-view augmentation for contrastive learning
- Custom collation for variable-size batching
- Hierarchy caching for efficiency

### Feature Processing Networks
- `FeaturePropagator`: Global-to-local (G2L) feature propagation
- `FeatureAggregator`: Local-to-global (L2G) feature aggregation
- Multi-scale variants for hierarchical levels

### Augmentation Transforms
- Geometric: rotation, scaling, translation
- Appearance: color jitter, chromatic auto-contrast
- Hierarchy-invariant (indices remain valid)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DBBD.git
cd DBBD

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Requirements
- Python >= 3.8
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0

For development/testing:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Data

Organize your data in combined format with train/val/test splits:
```
datasets/
└── scannet/
    ├── train/
    │   ├── scene0000_00.pkl
    │   ├── scene0000_01.pkl
    │   └── ...
    ├── val/
    │   ├── scene0011_00.pkl
    │   └── ...
    └── test/
        ├── scene0707_00.pkl
        └── ...
```

Each `.pkl` file contains combined data and hierarchy:
```python
{
    'coords': Tensor of shape (N, 3),     # XYZ coordinates
    'normals': Tensor of shape (N, 3),    # Point normals (used as features)
    'hierarchy': dict,                     # Hierarchical decomposition
    'num_points': int,                     # Total points
    'total_regions': int,                  # Total regions in hierarchy
    'max_depth_reached': int               # Hierarchy depth
}
```

The hierarchy dict is recursively structured with keys:
- `indices`: Point indices for this region
- `center_idx`: Center point index
- `level`: Hierarchy level (0=root)
- `children`: List of child region dicts

### 2. Create Dataset and DataLoader

```python
from dbbd.datasets import DBBDDataset, create_dataloader
from dbbd.datasets.transforms import Compose, RandomRotate, RandomScale, ToTensor

# Define transforms
transform = Compose([
    RandomRotate(angle=[-180, 180], axis='z'),
    RandomScale(scale=[0.8, 1.2]),
    ToTensor()
])

# Create dataset (combined format with splits)
dataset = DBBDDataset(
    data_root='datasets/scannet',  # Root with train/val/test subdirs
    split='train',                  # 'train', 'val', or 'test'
    dual_view=True,                 # For contrastive learning
    transform=transform
)

# Create dataloader
loader = create_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    shuffle=True
)

# Iterate
for batch in loader:
    # batch contains:
    # - 'view1': {'coord', 'feat', 'hierarchy', ...}
    # - 'view2': {'coord', 'feat', 'hierarchy', ...}
    # - 'hierarchies': List of Region trees
    # - 'scene_ids': List of scene identifiers
    pass
```

### 3. Use Feature Networks

```python
from dbbd.models.features import FeaturePropagator, FeatureAggregator

# Initialize networks
propagator = FeaturePropagator(
    parent_dim=96,
    coord_dim=3,
    hidden_dim=128,
    out_dim=96
)

aggregator = FeatureAggregator(
    feat_dim=96,
    mode='max'  # or 'mean', 'attention'
)

# Global-to-local: propagate parent features to children
parent_feat = torch.randn(96)  # From parent region encoding
child_coords = torch.randn(50, 3)  # Child region coordinates
propagated = propagator(parent_feat, child_coords)  # (50, 96)

# Local-to-global: aggregate children features to parent
child_features = torch.randn(8, 96)  # From 8 child region encodings
aggregated = aggregator(child_features)  # (96,)
```

### 4. Run Demo

```bash
python examples/phase2_demo.py
```

This demonstrates:
- Loading datasets with hierarchies
- Dual-view augmentation
- Batching and collation
- Feature propagator and aggregator usage

## Configuration

Example configuration for ScanNet pretraining:

```python
# dbbd/configs/dbbd_scannet.py
data = dict(
    type='DBBDDataset',
    data_root='datasets/scannet',  # Root with train/val/test subdirs
    split='train',
    dual_view=True,
    cache_hierarchies=True,
    cache_size=100
)

transform = [
    dict(type='RandomRotate', angle=[-180, 180], axis='z', p=0.95),
    dict(type='RandomScale', scale=[0.8, 1.2], p=0.95),
    dict(type='ColorJitter', std=0.05, p=0.8),
    dict(type='ToTensor')
]

model = dict(
    propagator=dict(
        type='FeaturePropagator',
        parent_dim=96,
        hidden_dim=128,
        out_dim=96
    ),
    aggregator=dict(
        type='FeatureAggregator',
        feat_dim=96,
        mode='max'
    )
)
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_dataset.py -v
pytest tests/test_features.py -v
pytest tests/test_integration.py -v
```

With coverage:
```bash
pytest tests/ --cov=dbbd --cov-report=html
```

## API Documentation

### Core Classes

#### `DBBDDataset`
Dataset class for loading combined point clouds with hierarchies.

**Parameters:**
- `data_root` (str): Root directory with train/val/test subdirectories
- `split` (str): Dataset split ('train', 'val', 'test')
- `dual_view` (bool): If True, return two augmented views
- `transform`: Transform or Compose to apply
- `cache_hierarchies` (bool): Whether to cache hierarchies
- `cache_size` (int): Maximum cache size

**Returns:**
Single view mode:
```python
{
    'coord': Tensor (N, 3),
    'feat': Tensor (N, C),
    'hierarchy': Region tree,
    'scene_id': str
}
```

Dual view mode:
```python
{
    'view1': {...},
    'view2': {...},
    'scene_id': str
}
```

#### `FeaturePropagator`
Propagates parent features to child region points.

**Parameters:**
- `parent_dim` (int): Parent feature dimension
- `coord_dim` (int): Coordinate dimension (default: 3)
- `hidden_dim` (int): Hidden layer dimension
- `out_dim` (int): Output feature dimension

**Forward:**
```python
propagator(parent_feat, child_coords) -> propagated_features
```

#### `FeatureAggregator`
Aggregates child region features to parent.

**Parameters:**
- `feat_dim` (int): Feature dimension
- `mode` (str): Aggregation mode ('max', 'mean', 'attention')
- `use_pre_mlp` (bool): Process features before pooling
- `use_spatial` (bool): Incorporate spatial context

**Forward:**
```python
aggregator(child_features) -> aggregated_feature
```

### Transforms

All transforms follow the pattern:
```python
transform(data_dict) -> transformed_data_dict
```

Available transforms:
- `RandomRotate`: Random rotation around axis
- `RandomScale`: Random uniform scaling
- `RandomTranslation`: Random translation (jitter)
- `ColorJitter`: RGB color perturbation
- `ChromaticAutoContrast`: Auto-contrast normalization
- `ToTensor`: Convert numpy arrays to tensors
- `Compose`: Chain multiple transforms

## Project Structure

```
DBBD/
├── dbbd/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── dbbd_dataset.py      # Main dataset class
│   │   └── transforms.py         # Augmentation transforms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── features/
│   │   │   ├── propagator.py    # FeaturePropagator
│   │   │   └── aggregator.py    # FeatureAggregator
│   │   └── utils/
│   │       ├── hierarchy.py     # Region class, loading
│   │       └── batch.py         # Collation functions
│   └── configs/
│       └── dbbd_scannet.py      # Example configuration
├── tests/
│   ├── test_dataset.py          # Dataset tests
│   ├── test_features.py         # Feature network tests
│   └── test_integration.py      # Integration tests
├── examples/
│   └── phase2_demo.py           # Demonstration script
├── requirements.txt
├── setup.py
└── README.md
```

## Troubleshooting

### Common Issues

**1. Split directory not found**
- Ensure `data_root` contains train/val/test subdirectories
- Check `split` parameter matches available directories
- Verify `.pkl` files exist in the split directory

**2. Shape mismatch errors**
- Verify data format: `{'coords': (N, 3), 'normals': (N, 3), 'hierarchy': dict}`
- Check hierarchy indices are within `[0, N)`
- Ensure augmentations don't change number of points

**3. Memory issues**
- Reduce `cache_size` in dataset
- Decrease `batch_size` in dataloader
- Use `num_workers=0` for debugging

**4. Slow data loading**
- Increase `num_workers` (e.g., 4-8)
- Enable `cache_hierarchies=True`
- Use SSD for data storage

## Phase 2 Key Decisions

Important architectural and design decisions made during Phase 2:

### 1. Combined Data Format
**Decision**: Store point clouds and hierarchies together in single `.pkl` files.

**Rationale**: The original design had separate data (`.pth`) and hierarchy (`.pkl`) files. We consolidated to:
- Simplify data management (one file per scene)
- Ensure data-hierarchy consistency
- Reduce I/O overhead during loading

### 2. Dict-to-Region Conversion
**Decision**: Hierarchies stored as nested dicts, converted to `Region` objects at load time.

**Rationale**: 
- Pickle files with raw dicts are more portable
- `Region` dataclass provides clean API with methods like `get_depth()`, `get_all_descendants()`
- Conversion happens once per load, cached for efficiency

### 3. Feature Dimensions
**Decision**: Normals (3D) used as point features; 96-dim feature vectors throughout.

**Rationale**:
- Normals available from ScanNet preprocessing
- 96-dim balances expressiveness vs. memory (matches common encoder widths)
- Can extend to RGB+normals (6D) if needed

### 4. 8-way Branching Factor
**Decision**: Each hierarchy node has up to 8 children.

**Rationale**:
- Matches octree-like spatial decomposition
- Provides ~2,500 regions for typical scenes
- 4 levels of depth sufficient for multi-scale learning

---

## Test Coverage

**49 tests passing** across all modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_dataset.py` | 20 | Dataset, transforms, dict-to-region |
| `test_features.py` | 21 | Propagator, aggregator, multi-scale |
| `test_integration.py` | 8 | Batching, dataloader, end-to-end |

---

## Citation

If you use this code, please cite:

```bibtex
@article{dbbd2024,
  title={DBBD: Dual Branch Bi-Directional Self-Supervised Learning for Point Clouds},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Built on top of [Pointcept](https://github.com/Pointcept/Pointcept) framework
- Inspired by hierarchical and contrastive learning methods for point clouds

## Next Steps: Phase 3

Phase 2 is complete. Phase 3 will implement:

1. **Encoder Backbone Integration**
   - Integrate SpUNet or similar point cloud encoder
   - Region-wise encoding with feature pooling

2. **Bidirectional Traversal**
   - Top-down (G2L): Propagate global context to local regions
   - Bottom-up (L2G): Aggregate local features to global

3. **Hierarchical Feature Pipeline**
   - Combine encoder, propagator, and aggregator
   - Multi-scale feature extraction

See `phase3_final_guide.md` for detailed Phase 3 specifications.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
