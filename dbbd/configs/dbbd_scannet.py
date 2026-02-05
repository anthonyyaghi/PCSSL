"""
DBBD Configuration for ScanNet Pretraining

Configuration for Phase 2 data integration and feature processing.
"""

import os
from pathlib import Path

# Dataset configuration
data = dict(
    type='DBBDDataset',
    data_root='data/scannet/train',  # Path to point cloud files
    hierarchy_root='data/hierarchies/scannet_d3_b8/train',  # Path to precomputed hierarchies
    split='train',
    dual_view=True,  # Use dual-view for contrastive learning
    cache_hierarchies=True,
    cache_size=100,
    validate_hierarchies=False,  # Set to True for debugging
    data_suffix='.pth',
    hierarchy_suffix='.pkl'
)

# Augmentation transforms
transform = [
    # Geometric transforms
    dict(type='RandomRotate', angle=[-180, 180], axis='z', p=0.95),
    dict(type='RandomScale', scale=[0.8, 1.2], anisotropic=False, p=0.95),
    dict(type='RandomTranslation', shift=0.2, p=0.5),
    
    # Appearance transforms (for RGB features)
    dict(type='ColorJitter', std=0.05, clip=True, p=0.8),
    dict(type='ChromaticAutoContrast', randomize_blend_factor=True, p=0.5),
    
    # Convert to tensors
    dict(type='ToTensor')
]

# DataLoader configuration
dataloader = dict(
    batch_size=8,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

# Validation data configuration
val_data = dict(
    type='DBBDDataset',
    data_root='data/scannet/val',
    hierarchy_root='data/hierarchies/scannet_d3_b8/val',
    split='val',
    dual_view=False,  # Single view for validation
    cache_hierarchies=True,
    cache_size=50,
    data_suffix='.pth',
    hierarchy_suffix='.pkl'
)

val_transform = [
    dict(type='ToTensor')  # No augmentation for validation
]

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# Feature processing network configuration
model = dict(
    # Feature Propagator (G2L)
    propagator=dict(
        type='FeaturePropagator',
        parent_dim=96,
        coord_dim=3,
        hidden_dim=128,
        out_dim=96,
        num_layers=2,
        use_layer_norm=True,
        dropout=0.0
    ),
    
    # Feature Aggregator (L2G)
    aggregator=dict(
        type='FeatureAggregator',
        feat_dim=96,
        mode='max',  # Options: 'max', 'mean', 'attention'
        use_pre_mlp=True,
        use_spatial=False
    )
)

# Hierarchy configuration (should match preprocessing parameters from Phase 1)
hierarchy = dict(
    max_depth=3,  # Maximum hierarchy depth
    branching_factor=8,  # Target number of children per region
    min_points_per_region=32  # Minimum points in leaf regions
)

# Output and logging
output_dir = 'outputs/dbbd_phase2'
log_level = 'INFO'

# Device configuration
device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))

# Random seed for reproducibility
seed = 42

# For Phase 3+ (encoder, loss, training)
# These will be used in future phases
encoder = dict(
    type='SpUNet',  # Sparse UNet encoder (from Pointcept)
    in_channels=6,  # XYZ + RGB
    out_channels=96,
    base_channels=32,
    num_stages=4
)

loss = dict(
    type='HierarchicalContrastiveLoss',
    temperature=0.07,
    use_all_levels=True,
    level_weights=[1.0, 1.0, 1.0]  # Weights for each hierarchy level
)

optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.01
)

scheduler = dict(
    type='CosineAnnealingLR',
    T_max=100,
    eta_min=1e-5
)

# Training configuration (for Phase 4)
training = dict(
    num_epochs=100,
    save_interval=10,
    eval_interval=5,
    log_interval=100
)
