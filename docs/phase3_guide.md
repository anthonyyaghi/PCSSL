# Phase 3: Hierarchical Encoding Pipeline — Implementation Guide

## Overview

This document specifies the implementation of DBBD's hierarchical encoding pipeline. Phase 2 provided the data infrastructure and feature processing modules. Phase 3 builds the encoder that processes hierarchies bidirectionally and outputs features ready for Phase 4's contrastive losses.

**Read this entire document before writing any code.**

---

## Theoretical Foundation (Required Reading)

### The Core Idea

DBBD processes the same hierarchy in two complementary directions:

**Global-to-Local (G2L):** Top-down processing where parent context flows to children
- Start at root (full scene), encode it
- Propagate root's feature to children's point coordinates
- Encode each child with this propagated context
- Recurse to leaves
- **Result:** Every region's feature is informed by its ancestors

**Local-to-Global (L2G):** Bottom-up processing where children compose parents
- Start at leaves, encode each independently
- Aggregate sibling features to form parent's representation (NO encoding of parent points)
- Recurse to root
- **Result:** Every region's feature is composed from its descendants

**Why this works:** The same region gets two different feature vectors — one from global context, one from local composition. Contrastive learning enforces consistency between these, preventing geometric shortcuts and learning semantically meaningful representations.

### Critical Design Decisions (Already Made)

These decisions are final. Do not deviate without explicit approval.

| Decision | Resolution |
|----------|------------|
| Contrastive alignment | Same-level: G2L level-i ↔ L2G level-i |
| L2G non-leaf encoding | Pure aggregation only. Non-leaf nodes are NOT encoded by backbone. |
| Feature combination in G2L | Concatenation followed by MLP projection |
| Coordinate handling | Center each region before encoding (translate centroid to origin) |
| Point-level features | Encode full scene at root level in both branches |
| Positive pairs | Same region across two augmented views (shared hierarchy structure) |
| Aggregator spatial context | Enabled by default, configurable |

---

## Phase 2 Dependencies

Before implementing, verify these exist and understand their interfaces:

### 1. Region Dataclass
```python
@dataclass
class Region:
    indices: List[int]      # Point indices belonging to this region
    center_idx: int         # Index of the center point
    level: int              # Hierarchy level (0 = root, D-1 = leaves)
    children: List[Region]  # Child regions (empty for leaves)
    parent: Optional[Region] # Parent region (None for root)
```

### 2. FeaturePropagator (G2L direction)
```python
class FeaturePropagator(nn.Module):
    def forward(self, parent_feat, child_coords):
        """
        Propagate parent feature to child point positions.
        
        Args:
            parent_feat: (D,) - Parent region's feature vector
            child_coords: (M, 3) - Coordinates of points in child region
            
        Returns:
            propagated: (M, D) - Feature vector for each child point
        """
```

### 3. FeatureAggregator (L2G direction)
```python
class FeatureAggregator(nn.Module):
    def forward(self, child_feats, child_positions=None, parent_position=None):
        """
        Aggregate child features into parent representation.
        
        Args:
            child_feats: (num_children, D) - Feature vectors of children
            child_positions: (num_children, 3) - Optional child center coordinates
            parent_position: (3,) - Optional parent center coordinate
            
        Returns:
            parent_feat: (D,) - Aggregated feature for parent
        """
```

### 4. Batch Format from DataLoader
```python
batch = {
    'coord': Tensor,        # (total_points, 3) - All points concatenated
    'feat': Tensor,         # (total_points, C) - Point features (normals, RGB, etc.)
    'offset': Tensor,       # (B+1,) - Scene boundaries [0, n1, n1+n2, ...]
    'hierarchies': List[Region],  # Length B - Root region for each scene
}
```

### 5. Dual-View Augmentation
The dataloader provides two augmented views of each scene. Verify whether:
- Option A: `batch['view1']` and `batch['view2']` are separate dicts
- Option B: Single batch with augmentation applied twice during forward pass

Check the Phase 2 dataset implementation to confirm the format.

---

## Architecture Components

### Component 1: PointCloudEncoder

**Purpose:** Wrap a point cloud backbone to encode a region into features.

**Requirements:**
- Accept variable-size point subsets
- Return both region-level feature (pooled) and point-level features
- Handle coordinate centering internally

**Interface:**
```python
class PointCloudEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int,           # Input feature dimension (after projection)
                 output_dim: int = 96,     # Output feature dimension
                 backbone: str = 'pointnet'):  # 'pointnet', 'dgcnn', etc.
        ...
    
    def forward(self, coords, feats, center=None):
        """
        Encode a region.
        
        Args:
            coords: (M, 3) - Point coordinates
            feats: (M, C) - Point features
            center: (3,) - Optional center for normalization. If None, use centroid.
            
        Returns:
            region_feat: (output_dim,) - Pooled region feature
            point_feats: (M, output_dim) - Per-point features
        """
        # 1. Center coordinates
        if center is None:
            center = coords.mean(dim=0)
        centered_coords = coords - center
        
        # 2. Combine coords with features as input
        # Common practice: concatenate [centered_coords, feats]
        encoder_input = torch.cat([centered_coords, feats], dim=-1)
        
        # 3. Forward through backbone
        point_feats = self.backbone(encoder_input)  # (M, output_dim)
        
        # 4. Pool for region feature
        region_feat = point_feats.max(dim=0)[0]  # or mean, or attention
        
        return region_feat, point_feats
```

**Backbone options:**
- Start with a simple shared MLP (PointNet-style) for debugging
- Can upgrade to DGCNN, PointTransformer later
- Check what Phase 2 already uses

**Important:** The encoder is shared across both G2L and L2G branches.

---

### Component 2: G2L Traversal

**Purpose:** Top-down encoding with context propagation.

**Algorithm:**
```
function g2l_encode(region, scene_coords, scene_feats, parent_feat=None):
    # Extract this region's points
    region_coords = scene_coords[region.indices]
    region_feats = scene_feats[region.indices]
    center = scene_coords[region.center_idx]
    
    # Prepare input features
    if parent_feat is not None:
        # Propagate parent context to this region's points
        propagated = propagator(parent_feat, region_coords)  # (M, D)
        # Concatenate with raw features
        combined = concat([region_feats, propagated], dim=-1)  # (M, C+D)
        # Project to encoder input dimension
        encoder_input = projection_mlp(combined)  # (M, D_input)
    else:
        # Root has no parent - just project raw features
        encoder_input = projection_mlp(region_feats)  # (M, D_input)
    
    # Encode this region
    region_feat, point_feats = encoder(region_coords, encoder_input, center)
    
    # Store this region's feature for its level
    store_feature(level=region.level, feat=region_feat, branch='g2l')
    
    # If this is a leaf, store point features
    if region.is_leaf:
        store_point_features(region.indices, point_feats, branch='g2l')
    
    # Recurse to children
    for child in region.children:
        g2l_encode(child, scene_coords, scene_feats, parent_feat=region_feat)
```

**Key points:**
- Root node: No propagation (parent_feat=None), just encode
- Non-root nodes: Propagate + concatenate + project, then encode
- Traversal order: Pre-order (parent before children)
- Store features by level for later batching

---

### Component 3: L2G Traversal

**Purpose:** Bottom-up encoding with aggregation. Non-leaf nodes are NOT encoded.

**Algorithm:**
```
function l2g_encode(region, scene_coords, scene_feats):
    if region.is_leaf:
        # Leaves ARE encoded
        region_coords = scene_coords[region.indices]
        region_feats = scene_feats[region.indices]
        center = scene_coords[region.center_idx]
        
        # Project and encode (no propagation in L2G)
        encoder_input = projection_mlp(region_feats)
        region_feat, point_feats = encoder(region_coords, encoder_input, center)
        
        # Store
        store_feature(level=region.level, feat=region_feat, branch='l2g')
        store_point_features(region.indices, point_feats, branch='l2g')
        
        return region_feat
    
    # Non-leaf: Recurse first, then aggregate (post-order traversal)
    child_feats = []
    child_positions = []
    for child in region.children:
        child_feat = l2g_encode(child, scene_coords, scene_feats)
        child_feats.append(child_feat)
        child_positions.append(scene_coords[child.center_idx])
    
    child_feats = torch.stack(child_feats)          # (num_children, D)
    child_positions = torch.stack(child_positions)  # (num_children, 3)
    parent_position = scene_coords[region.center_idx]
    
    # Aggregate children (NO encoding for non-leaves)
    region_feat = aggregator(child_feats, child_positions, parent_position)
    
    # Store
    store_feature(level=region.level, feat=region_feat, branch='l2g')
    
    return region_feat
```

**Key points:**
- Leaves: Encoded by backbone
- Non-leaves: Pure aggregation of children, no backbone encoding
- Traversal order: Post-order (children before parent)
- Aggregator uses spatial context by default

---

### Component 4: Point-Level Features (Full Scene Encoding)

**Purpose:** Get per-point features for the point-level contrastive loss.

**Approach:** Encode the full scene (as if at root level) in both branches.

```python
def get_point_features_g2l(scene_coords, scene_feats):
    """G2L point features: encode full scene without parent context."""
    encoder_input = projection_mlp(scene_feats)
    _, point_feats = encoder(scene_coords, encoder_input)
    return point_feats

def get_point_features_l2g(scene_coords, scene_feats):
    """L2G point features: encode full scene (same as G2L for simplicity)."""
    # Note: These will differ between views due to different augmentations
    encoder_input = projection_mlp(scene_feats)
    _, point_feats = encoder(scene_coords, encoder_input)
    return point_feats
```

**Why this works:** View 1 and View 2 have different augmentations (rotation, translation, etc.), so `scene_coords` and `scene_feats` differ between branches. The features will be different even though the encoding process is the same.

**Alternative (optional enhancement):** Condition L2G point encoding on the aggregated root feature:
```python
def get_point_features_l2g_conditioned(scene_coords, scene_feats, root_l2g_feat):
    """L2G point features conditioned on aggregated root."""
    root_broadcast = root_l2g_feat.unsqueeze(0).expand(len(scene_coords), -1)
    encoder_input = projection_mlp(torch.cat([scene_feats, root_broadcast], dim=-1))
    _, point_feats = encoder(scene_coords, encoder_input)
    return point_feats
```

Start with the simple version. Add conditioning if ablations show benefit.

---

### Component 5: Feature Collection and Batching

**Purpose:** Collect features from both branches across all scenes and organize for loss computation.

**Data structure during processing:**
```python
class FeatureCollector:
    def __init__(self):
        # Per-level features
        self.g2l_by_level = defaultdict(list)  # level -> list of (region_feat,)
        self.l2g_by_level = defaultdict(list)
        
        # Scene boundaries per level
        self.counts_by_level = defaultdict(list)  # level -> [n_regions_scene1, n_regions_scene2, ...]
        
        # Point-level features
        self.point_feats_g2l = []  # list of (N_i, D) tensors per scene
        self.point_feats_l2g = []
```

**Final output format (required by Phase 4):**
```python
output = {
    'g2l': {
        'level_0': Tensor,  # (total_regions_at_level_0, D)
        'level_1': Tensor,  # (total_regions_at_level_1, D)
        ...
    },
    'l2g': {
        'level_0': Tensor,
        'level_1': Tensor,
        ...
    },
    'offsets_by_level': {
        'level_0': Tensor,  # (B+1,) scene boundaries for level 0
        'level_1': Tensor,
        ...
    },
    'point_feats_g2l': Tensor,  # (total_points, D)
    'point_feats_l2g': Tensor,  # (total_points, D)
    'point_offset': Tensor,     # (B+1,) scene boundaries for points
}
```

**How offsets work:**
```python
# If we have 3 scenes with [5, 3, 7] regions at level 1:
offsets_level_1 = [0, 5, 8, 15]  # cumsum with leading 0

# To get scene i's features:
start, end = offsets_level_1[i], offsets_level_1[i+1]
scene_i_feats = level_1_feats[start:end]
```

---

### Component 6: Main HierarchicalEncoder Module

**Purpose:** Orchestrate all components into a single forward pass.

**Interface:**
```python
class HierarchicalEncoder(nn.Module):
    def __init__(self,
                 input_feat_dim: int,        # Raw feature dim (e.g., 3 for normals)
                 hidden_dim: int = 96,       # Internal feature dimension
                 output_dim: int = 96,       # Output feature dimension
                 propagator_config: dict = None,
                 aggregator_config: dict = None,
                 encoder_config: dict = None):
        ...
        
    def forward(self, batch) -> dict:
        """
        Process batch through both branches.
        
        Args:
            batch: Dict with 'coord', 'feat', 'offset', 'hierarchies'
                   OR dict with 'view1' and 'view2' sub-dicts for dual-view
                   
        Returns:
            Dict with g2l features, l2g features, point features, offsets
        """
```

**Dual-view handling:**

Check how Phase 2 provides the two views. The encoder should process View 1 through G2L and View 2 through L2G (or vice versa). The key is that both views share the same hierarchy structure.

```python
def forward(self, batch):
    if 'view1' in batch and 'view2' in batch:
        # Dual-view mode
        # View 1 -> G2L branch
        g2l_features = self._process_g2l(batch['view1'], batch['hierarchies'])
        # View 2 -> L2G branch  
        l2g_features = self._process_l2g(batch['view2'], batch['hierarchies'])
    else:
        # Single-view mode (for inference/debugging)
        g2l_features = self._process_g2l(batch, batch['hierarchies'])
        l2g_features = self._process_l2g(batch, batch['hierarchies'])
    
    return self._collate_outputs(g2l_features, l2g_features)
```

---

## Implementation Order

Follow this order strictly:

### Step 1: Projection MLP
Create a simple MLP that projects raw features to encoder input dimension.
```python
# Test: random input (100, 3) -> output (100, 96)
```

### Step 2: PointCloudEncoder
Implement with a simple PointNet-style backbone first.
```python
# Test: coords (100, 3), feats (100, 96) -> region_feat (96,), point_feats (100, 96)
# Verify: coordinates are centered before processing
```

### Step 3: L2G Traversal (simpler — no propagation)
Implement leaf encoding and aggregation.
```python
# Test with small hierarchy: 1 root, 4 children (leaves)
# Verify: leaves are encoded, root is aggregated
# Verify: aggregator receives correct child positions
```

### Step 4: G2L Traversal
Implement with propagation and concatenation.
```python
# Test with same hierarchy
# Verify: root encoded without propagation
# Verify: children receive propagated parent features
# Verify: concatenation + projection dimensions are correct
```

### Step 5: Point-Level Features
Implement full-scene encoding for both branches.
```python
# Test: scene with 1000 points -> point_feats (1000, 96)
```

### Step 6: Feature Collection
Implement collector and verify batching logic.
```python
# Test with 2 scenes, different hierarchy depths
# Verify: offsets correctly partition features by scene
```

### Step 7: Full Integration
Wire everything into HierarchicalEncoder.
```python
# Test with real batch from dataloader
# Verify: output format matches Phase 4 requirements
```

### Step 8: Gradient Check
Verify gradients flow through entire pipeline.
```python
# Forward pass -> dummy loss -> backward pass
# Check: all parameters have gradients, no NaN/Inf
```

---

## Testing Checklist

Run these tests before declaring Phase 3 complete:

### Unit Tests
- [ ] Projection MLP: correct output shapes
- [ ] PointCloudEncoder: handles variable region sizes
- [ ] PointCloudEncoder: coordinates are centered
- [ ] FeaturePropagator: output shape matches input points
- [ ] FeatureAggregator: handles variable number of children
- [ ] FeatureAggregator: spatial context is used when enabled

### Traversal Tests
- [ ] G2L visits nodes in pre-order (parent before children)
- [ ] L2G visits nodes in post-order (children before parent)
- [ ] G2L root has no propagation (parent_feat=None)
- [ ] L2G non-leaves are NOT encoded (pure aggregation)
- [ ] Features collected at all levels for both branches

### Integration Tests
- [ ] Single scene processes correctly
- [ ] Batch of multiple scenes processes correctly
- [ ] Dual-view mode works (View 1 → G2L, View 2 → L2G)
- [ ] Output format matches Phase 4 requirements exactly
- [ ] Offsets correctly partition features by scene and level

### Gradient Tests
- [ ] Gradients flow to encoder parameters
- [ ] Gradients flow to propagator parameters
- [ ] Gradients flow to aggregator parameters
- [ ] No NaN or Inf in gradients
- [ ] No NaN or Inf in features

### Sanity Checks
- [ ] Features are not all zeros
- [ ] Features are not all identical
- [ ] Different regions produce different features
- [ ] G2L and L2G produce different features for same region
- [ ] Different augmented views produce different features

---

## Common Pitfalls

### 1. Dimension Mismatches
The most common bug. Triple-check dimensions at every step:
- Raw features: (M, C) where C might be 3 (normals) or 6 (normals + RGB)
- After projection: (M, D_input)
- Propagated features: (M, D)
- Concatenated: (M, C + D) or (M, D_input + D) — be consistent!
- Encoder output: (M, D_output)

### 2. Forgetting to Center Coordinates
The encoder must center coordinates before processing. If you see features that depend on absolute position, this is likely the bug.

### 3. Encoding Non-Leaves in L2G
L2G non-leaf nodes must be PURE AGGREGATION. If you're calling the encoder for non-leaves, this is wrong.

### 4. Wrong Traversal Order
- G2L: Pre-order (parent first)
- L2G: Post-order (children first)

If features don't make sense, check traversal order.

### 5. Index Confusion
Region.indices are indices into the SCENE's point arrays, not global batch arrays. When processing scene i:
```python
start, end = batch['offset'][i], batch['offset'][i+1]
scene_coords = batch['coord'][start:end]
scene_feats = batch['feat'][start:end]

# Now region.indices index into scene_coords, scene_feats
region_coords = scene_coords[region.indices]  # Correct
# NOT: batch['coord'][region.indices]  # Wrong!
```

### 6. Shared vs Separate Parameters
The encoder is SHARED between G2L and L2G. The propagator and aggregator are separate (one is G2L-specific, one is L2G-specific).

### 7. Memory Leaks
Don't store intermediate tensors unnecessarily. Especially in recursive traversals, be careful about what you keep in memory.

---

## Output Specification for Phase 4

Phase 4 will receive this exact structure. Do not deviate.

```python
encoder_output = {
    # Region-level features by branch and level
    'g2l': {
        'level_0': Tensor,  # (N_0, D) where N_0 = total regions at level 0 in batch
        'level_1': Tensor,  # (N_1, D)
        'level_2': Tensor,  # (N_2, D)
        # ... up to max depth
    },
    'l2g': {
        'level_0': Tensor,  # Same shapes as g2l
        'level_1': Tensor,
        'level_2': Tensor,
    },
    
    # Scene boundaries for each level
    'offsets_by_level': {
        'level_0': Tensor,  # (B+1,) e.g., [0, 1, 2, 3] for 3 scenes with 1 root each
        'level_1': Tensor,  # (B+1,) e.g., [0, 8, 14, 25] 
        'level_2': Tensor,
    },
    
    # Point-level features
    'point_feats_g2l': Tensor,  # (total_points, D)
    'point_feats_l2g': Tensor,  # (total_points, D)
    
    # Point scene boundaries (same as input batch['offset'])
    'point_offset': Tensor,  # (B+1,)
}
```

**Phase 4 will use this as:**
```python
# Region-level contrastive loss at level i
g2l_level_i = output['g2l'][f'level_{i}']
l2g_level_i = output['l2g'][f'level_{i}']
offsets_i = output['offsets_by_level'][f'level_{i}']

# For each scene, contrast corresponding regions
for scene_idx in range(batch_size):
    start, end = offsets_i[scene_idx], offsets_i[scene_idx + 1]
    g2l_scene = g2l_level_i[start:end]
    l2g_scene = l2g_level_i[start:end]
    loss += contrastive_loss(g2l_scene, l2g_scene)

# Point-level contrastive loss
point_loss = contrastive_loss(output['point_feats_g2l'], output['point_feats_l2g'])
```

---

## Configuration Defaults

Use these as starting values. They can be tuned in Phase 4.

```python
config = {
    'hidden_dim': 96,
    'output_dim': 96,
    'encoder': {
        'type': 'pointnet',  # Start simple
        'pooling': 'max',
    },
    'propagator': {
        'hidden_dims': [96, 96],
        'use_coords': True,
    },
    'aggregator': {
        'hidden_dims': [96, 96],
        'use_spatial_context': True,  # Default ON
        'pooling': 'max',  # or 'attention'
    },
}
```

---

## Questions? Check Phase 2 First

If something is unclear about:
- Data format → Check DBBDDataset implementation
- Region structure → Check Region dataclass
- Propagator interface → Check FeaturePropagator implementation
- Aggregator interface → Check FeatureAggregator implementation

The Phase 2 code is the source of truth for interfaces.

---

## Success Criteria

Phase 3 is complete when:

1. **All tests pass** (see checklist above)
2. **Output format exactly matches specification**
3. **Gradients flow end-to-end without NaN/Inf**
4. **Code is clean, documented, and follows existing project style**
5. **A simple training loop can call forward() repeatedly without memory growth**

Do not proceed to Phase 4 until all criteria are met.
