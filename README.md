# DBBD: Dual Branch Bi-Directional Self-Supervised Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-120%2B%20passed-brightgreen.svg)]()

## Overview

DBBD (Dual Branch Bi-Directional) is a self-supervised learning framework for 3D point clouds that learns rich multi-scale representations through hierarchical contrastive learning.

### Current Status: Phase 5 Complete ✓

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Offline preprocessing - hierarchical decompositions | ✅ Complete |
| Phase 2 | Data integration & feature processing | ✅ Complete |
| Phase 3 | Hierarchical encoding - bidirectional traversal | ✅ Complete |
| Phase 4 | Contrastive losses | ✅ Complete |
| Phase 5 | Training loop & checkpointing | ✅ Complete |

## Components

### Data Layer (`dbbd/datasets/`)
- `DBBDDataset`: Loads point clouds with precomputed hierarchies
- Dual-view augmentation for contrastive learning
- Custom collation for variable-size batching
- `max_scenes` parameter for limiting dataset size

### Feature Processing (`dbbd/models/features/`)
- `FeaturePropagator`: Global-to-local (G2L) feature propagation
- `FeatureAggregator`: Local-to-global (L2G) feature aggregation (max/mean/attention)

### Hierarchical Encoder (`dbbd/models/encoder/`)
- `HierarchicalEncoder`: Main orchestrator for G2L/L2G branches
- `PointCloudEncoder`: PointNet-style backbone with coordinate centering
- `G2LTraversal`: Top-down (pre-order) traversal with context propagation
- `L2GTraversal`: Bottom-up (post-order) traversal with feature aggregation
- `FeatureCollector`: Collects features by level with offset tracking

### Contrastive Loss (`dbbd/models/loss/`)
- `DBBDContrastiveLoss`: Combined loss = α×L_region + β×L_point
- `InfoNCELoss`: Core contrastive loss using F.cross_entropy
- `RegionContrastiveLoss`: Hierarchy-level contrastive learning
- `PointContrastiveLoss`: Fine-grained point-level loss with subsampling
- `ContrastiveProjectionHead`: LayerNorm-based projection (supports batch_size=1)

### Training (`dbbd/training/`)
- `Trainer`: Full training loop with validation, checkpointing, early stopping
- `TrainingConfig`: YAML-serializable config with `small_mode` for local testing
- TensorBoard integration for loss and learning rate logging

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

### 4. Train the Model

```bash
# Full training
python train.py --data_root datasets/scannet --epochs 100

# Small mode for local testing (reduced backbone, fewer scenes)
python train.py --small_mode --epochs 10

# With config file
python train.py --config configs/debug_local.yaml
```

TensorBoard logs are saved to `runs/` directory:
```bash
tensorboard --logdir runs/
```

### 5. Run Demo

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

#### `HierarchicalEncoder`
Main encoder orchestrating bidirectional traversal.

**Parameters:**
- `input_dim` (int): Input feature dimension (default: 3 for normals)
- `hidden_dim` (int): Hidden/output dimension (default: 96)
- `hidden_dims` (list): PointNet backbone dimensions (default: [64, 128])
- `aggregation` (str): Aggregation mode ('max', 'mean', 'attention')

**Forward:**
```python
encoder(batch) -> {
    'g2l': {'level_0': (N0, D), 'level_1': (N1, D), ...},
    'l2g': {'level_0': (N0, D), 'level_1': (N1, D), ...},
    'point_feats_g2l': (total_points, D),
    'point_feats_l2g': (total_points, D),
    'offsets_by_level': {'level_0': (B+1,), ...},
    'point_offset': (B+1,)
}
```

#### `DBBDContrastiveLoss`
Combined contrastive loss for region and point features.

**Parameters:**
- `encoder_dim` (int): Encoder output dimension (default: 96)
- `proj_dim` (int): Projection dimension (default: 128)
- `temperature` (float): InfoNCE temperature (default: 0.1)
- `alpha` (float): Region loss weight (default: 1.0)
- `beta` (float): Point loss weight (default: 0.5)

**Forward:**
```python
criterion(encoder_output) -> (loss, {'total': ..., 'region': ..., 'point': ...})
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
│   │   ├── dbbd_dataset.py       # Main dataset class
│   │   └── transforms.py          # Augmentation transforms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── features/
│   │   │   ├── propagator.py     # FeaturePropagator (G2L)
│   │   │   └── aggregator.py     # FeatureAggregator (L2G)
│   │   ├── encoder/
│   │   │   ├── hierarchical_encoder.py  # Main encoder
│   │   │   ├── point_encoder.py         # PointNet backbone
│   │   │   ├── traversal.py             # G2L/L2G traversal
│   │   │   ├── projection.py            # Feature projection MLPs
│   │   │   └── collector.py             # Feature collection
│   │   ├── loss/
│   │   │   ├── dbbd_loss.py      # Combined contrastive loss
│   │   │   ├── infonce.py        # InfoNCE implementation
│   │   │   ├── region_loss.py    # Region-level loss
│   │   │   ├── point_loss.py     # Point-level loss
│   │   │   └── projection.py     # Projection head
│   │   └── utils/
│   │       ├── hierarchy.py      # Region class, loading
│   │       └── batch.py          # Collation functions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py             # TrainingConfig
│   │   └── trainer.py            # Trainer class
│   └── configs/
│       └── dbbd_scannet.py       # Example configuration
├── tests/
│   ├── test_dataset.py           # Dataset tests
│   ├── test_features.py          # Feature network tests
│   ├── test_encoder.py           # Encoder tests
│   ├── test_loss.py              # Loss function tests
│   ├── test_training_loop.py     # Training tests
│   └── test_integration.py       # Integration tests
├── docs/
│   ├── phase1_summary.md
│   ├── phase2_summary.md
│   ├── phase3_summary.md
│   ├── phase4_summary.md
│   └── phase5_summary.md
├── examples/
│   └── phase2_demo.py            # Demonstration script
├── train.py                       # Training entry point
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

## Key Design Decisions

### Data Format
- **Combined `.pkl` files**: Point clouds and hierarchies stored together for consistency
- **Dict-to-Region conversion**: Portable pickle storage with clean API at runtime
- **8-way branching**: Octree-like spatial decomposition (~2,500 regions per scene)

### Encoder Architecture
- **Shared PointNet backbone**: Single encoder used by both G2L and L2G branches
- **Coordinate centering**: All region coordinates centered before encoding
- **96-dim features**: Balances expressiveness vs. memory

### Contrastive Learning
- **LayerNorm in projection head**: Supports small batch sizes (batch_size=1)
- **F.cross_entropy for InfoNCE**: Better numerical stability than manual log-exp
- **Point subsampling**: Random 4096-point sampling for efficient point-level loss

### Training
- **Small mode**: Auto-adjusts all settings for local testing
- **TensorBoard always enabled**: Logs to `runs/` when tensorboard is installed

---

## Test Coverage

**120+ tests passing** across all modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_dataset.py` | 20 | Dataset, transforms, dict-to-region |
| `test_features.py` | 21 | Propagator, aggregator, multi-scale |
| `test_encoder.py` | 36 | PointCloudEncoder, traversal, HierarchicalEncoder |
| `test_loss.py` | 40+ | InfoNCE, region loss, point loss, combined loss |
| `test_training_loop.py` | 23 | Config, trainer, checkpointing, validation |
| `test_integration.py` | 8 | Batching, dataloader, end-to-end |

---

## Citation

If you use this code, please cite:

```bibtex
@article{dbbd2026,
  title={DBBD: Dual Branch Bi-Directional Self-Supervised Learning for Point Clouds},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Built on top of [Pointcept](https://github.com/Pointcept/Pointcept) framework
- Inspired by hierarchical and contrastive learning methods for point clouds

## Future Work

- Downstream task evaluation (semantic segmentation, object detection)
- Larger-scale training on full ScanNet
- Integration with SpUNet or other sparse convolution backbones
- Multi-GPU distributed training

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
