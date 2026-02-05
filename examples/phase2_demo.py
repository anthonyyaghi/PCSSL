"""
DBBD Phase 2 Demonstration Script

Demonstrates the usage of Phase 2 components:
- Loading dataset with hierarchies
- Applying augmentations
- Batching with collation
- Feature propagator and aggregator

Run with: python examples/phase2_demo.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pickle
from dbbd.datasets.dbbd_dataset import DBBDDataset, create_dataloader
from dbbd.datasets.transforms import Compose, RandomRotate, RandomScale, ColorJitter, ToTensor
from dbbd.models.features.propagator import FeaturePropagator
from dbbd.models.features.aggregator import FeatureAggregator
from dbbd.models.utils.hierarchy import get_hierarchy_stats
from dbbd.models.utils.batch import extract_scene_from_batch, compute_batch_stats


def create_synthetic_data(output_dir, num_scenes=5):
    """
    Create synthetic data for demonstration.
    
    Args:
        output_dir: Directory to save synthetic data
        num_scenes: Number of scenes to create
    """
    from dbbd.models.utils.hierarchy import Region
    
    data_dir = Path(output_dir) / 'data'
    hierarchy_dir = Path(output_dir) / 'hierarchies'
    data_dir.mkdir(parents=True, exist_ok=True)
    hierarchy_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_scenes} synthetic scenes...")
    
    for i in range(num_scenes):
        # Create synthetic point cloud
        num_points = 500 + i * 100
        coords = np.random.randn(num_points, 3).astype(np.float32)
        colors = np.random.rand(num_points, 3).astype(np.float32)
        
        data = {
            'coord': torch.from_numpy(coords),
            'feat': torch.from_numpy(colors)
        }
        
        torch.save(data, data_dir / f'scene{i:04d}.pth')
        
        # Create simple 3-level hierarchy
        root = Region(indices=np.arange(num_points), center_idx=0, level=0)
        
        # Level 1: 4 children
        children = []
        step = num_points // 4
        for j in range(4):
            start = j * step
            end = (j + 1) * step if j < 3 else num_points
            child = Region(
                indices=np.arange(start, end),
                center_idx=start,
                level=1,
                parent=root
            )
            children.append(child)
        
        root.children = children
        
        # Level 2: grandchildren (2 per child)
        for child in children:
            grandchildren = []
            child_size = len(child.indices)
            half = child_size // 2
            
            gc1 = Region(
                indices=child.indices[:half],
                center_idx=child.indices[0],
                level=2,
                parent=child
            )
            gc2 = Region(
                indices=child.indices[half:],
                center_idx=child.indices[half],
                level=2,
                parent=child
            )
            
            child.children = [gc1, gc2]
        
        # Save hierarchy
        with open(hierarchy_dir / f'scene{i:04d}.pkl', 'wb') as f:
            pickle.dump(root, f)
    
    print(f"✓ Created synthetic data in {output_dir}")
    return data_dir, hierarchy_dir


def demo_dataset_loading(data_dir, hierarchy_dir):
    """Demonstrate dataset loading."""
    print("\n" + "="*60)
    print("1. DATASET LOADING DEMO")
    print("="*60)
    
    # Create transforms
    transform = Compose([
        RandomRotate(angle=[-180, 180], axis='z', p=1.0),
        RandomScale(scale=[0.9, 1.1], p=1.0),
        ColorJitter(std=0.05, p=1.0),
        ToTensor()
    ])
    
    # Create dataset (single view)
    print("\nCreating single-view dataset...")
    dataset = DBBDDataset(
        data_root=str(data_dir),
        hierarchy_root=str(hierarchy_dir),
        dual_view=False,
        transform=transform
    )
    
    print(f"✓ Dataset created: {len(dataset)} scenes")
    
    # Load a sample
    print("\nLoading sample 0...")
    sample = dataset[0]
    
    print(f"  - Coordinates shape: {sample['coord'].shape}")
    print(f"  - Features shape: {sample['feat'].shape}")
    print(f"  - Scene ID: {sample['scene_id']}")
    print(f"  - Hierarchy: {sample['hierarchy']}")
    
    # Get hierarchy stats
    stats = get_hierarchy_stats(sample['hierarchy'])
    print(f"\n  Hierarchy Statistics:")
    print(f"    - Max depth: {stats['max_depth']}")
    print(f"    - Total nodes: {stats['num_nodes']}")
    print(f"    - Leaf nodes: {stats['num_leaves']}")
    print(f"    - Avg branching: {stats['avg_branching']:.2f}")
    print(f"    - Regions per level: {stats['regions_per_level']}")
    
    return dataset


def demo_dual_view(data_dir, hierarchy_dir):
    """Demonstrate dual-view loading for contrastive learning."""
    print("\n" + "="*60)
    print("2. DUAL-VIEW DEMO")
    print("="*60)
    
    transform = Compose([
        RandomRotate(angle=[-180, 180], axis='z', p=1.0),
        RandomScale(scale=[0.8, 1.2], p=1.0),
        ToTensor()
    ])
    
    print("\nCreating dual-view dataset...")
    dataset = DBBDDataset(
        data_root=str(data_dir),
        hierarchy_root=str(hierarchy_dir),
        dual_view=True,
        transform=transform
    )
    
    # Load a sample
    sample = dataset[0]
    
    print(f"✓ Loaded dual-view sample")
    print(f"  - View 1 coords shape: {sample['view1']['coord'].shape}")
    print(f"  - View 2 coords shape: {sample['view2']['coord'].shape}")
    print(f"  - Hierarchies identical: {sample['view1']['hierarchy'] is sample['view2']['hierarchy']}")
    
    # Check that transforms are different
    coord_diff = torch.abs(sample['view1']['coord'] - sample['view2']['coord']).mean()
    print(f"  - Mean coordinate difference: {coord_diff:.4f} (should be > 0)")


def demo_batching(dataset):
    """Demonstrate batching and collation."""
    print("\n" + "="*60)
    print("3. BATCHING DEMO")
    print("="*60)
    
    print("\nCreating DataLoader...")
    loader = create_dataloader(
        dataset,
        batch_size=3,
        num_workers=0,  # Use 0 for demo
        shuffle=False
    )
    
    print(f"✓ DataLoader created with batch_size=3")
    
    # Get first batch
    batch = next(iter(loader))
    
    print(f"\nBatch contents:")
    print(f"  - Total coordinates: {batch['coord'].shape}")
    print(f"  - Total features: {batch['feat'].shape}")
    print(f"  - Offset: {batch['offset'].tolist()}")
    print(f"  - Number of hierarchies: {len(batch['hierarchies'])}")
    print(f"  - Scene IDs: {batch['scene_ids']}")
    
    # Compute batch stats
    stats = compute_batch_stats(batch)
    print(f"\nBatch statistics:")
    print(f"  - Num scenes: {stats['num_scenes']}")
    print(f"  - Total points: {stats['total_points']}")
    print(f"  - Mean points/scene: {stats['mean_points']:.1f}")
    print(f"  - Min points: {stats['min_points']}")
    print(f"  - Max points: {stats['max_points']}")
    
    # Extract individual scene
    print(f"\nExtracting scene 1 from batch...")
    scene = extract_scene_from_batch(batch, scene_idx=1)
    print(f"  - Scene coords shape: {scene['coord'].shape}")
    print(f"  - Scene ID: {scene['scene_id']}")
    
    return batch


def demo_feature_networks(batch):
    """Demonstrate feature propagator and aggregator."""
    print("\n" + "="*60)
    print("4. FEATURE NETWORKS DEMO")
    print("="*60)
    
    # Initialize networks
    print("\nInitializing FeaturePropagator and FeatureAggregator...")
    propagator = FeaturePropagator(
        parent_dim=96,
        coord_dim=3,
        hidden_dim=128,
        out_dim=96
    )
    
    aggregator = FeatureAggregator(
        feat_dim=96,
        mode='max'
    )
    
    print(f"✓ Networks initialized")
    print(f"  - Propagator: {propagator}")
    print(f"  - Aggregator: {aggregator}")
    
    # Extract first scene
    scene = extract_scene_from_batch(batch, scene_idx=0)
    coords = scene['coord']
    hierarchy = scene['hierarchy']
    
    print(f"\nProcessing scene with {coords.shape[0]} points...")
    
    # Simulate hierarchical feature flow
    print("\n  Global-to-Local (G2L) Propagation:")
    
    # Start with random root feature
    root_feat = torch.randn(96)
    print(f"    - Root feature: {root_feat.shape}")
    
    # Propagate to children (level 1)
    level1_features = []
    for i, child in enumerate(hierarchy.children):
        child_coords = coords[child.indices]
        propagated = propagator(root_feat, child_coords)
        
        # Simulate encoding by taking mean
        encoded = propagated.mean(dim=0)
        level1_features.append(encoded)
        
        print(f"    - Child {i}: {len(child.indices)} points → propagated {propagated.shape} → encoded {encoded.shape}")
    
    # Aggregate back up
    print(f"\n  Local-to-Global (L2G) Aggregation:")
    level1_stacked = torch.stack(level1_features)
    aggregated = aggregator(level1_stacked)
    
    print(f"    - Aggregating {level1_stacked.shape[0]} children")
    print(f"    - Aggregated feature: {aggregated.shape}")
    
    # Test gradient flow
    print(f"\n  Testing gradient flow...")
    root_feat_grad = torch.randn(96, requires_grad=True)
    
    child_coords = coords[hierarchy.children[0].indices]
    propagated = propagator(root_feat_grad, child_coords)
    loss = propagated.sum()
    loss.backward()
    
    print(f"    - ✓ Gradients flow through propagator")
    print(f"    - Root grad norm: {root_feat_grad.grad.norm():.4f}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("DBBD PHASE 2 DEMONSTRATION")
    print("="*60)
    
    # Create synthetic data
    output_dir = Path(__file__).parent.parent / 'temp_demo_data'
    data_dir, hierarchy_dir = create_synthetic_data(output_dir, num_scenes=5)
    
    try:
        # Run demos
        dataset = demo_dataset_loading(data_dir, hierarchy_dir)
        demo_dual_view(data_dir, hierarchy_dir)
        batch = demo_batching(dataset)
        demo_feature_networks(batch)
        
        print("\n" + "="*60)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nPhase 2 components are working correctly:")
        print("  ✓ Dataset loads point clouds with hierarchies")
        print("  ✓ Dual-view augmentation works")
        print("  ✓ Batching preserves hierarchies")
        print("  ✓ Feature networks are differentiable")
        print("\nReady for Phase 3: Hierarchical Encoding Pipeline")
        
    finally:
        # Clean up synthetic data
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"\n✓ Cleaned up temporary data")


if __name__ == '__main__':
    main()
