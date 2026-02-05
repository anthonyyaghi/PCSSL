# Phase 4: Contrastive Losses - Implementation Guide

## Overview

Phase 4 implements the contrastive learning objectives that train the DBBD encoder to learn meaningful representations. The losses enforce that:
1. Same regions processed through G2L and L2G branches produce similar features
2. Point-level features are consistent across branches
3. Representations are transformation-invariant (via dual-view augmentation)

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Projection head location | Part of loss module (Option A) | Encoder outputs raw features usable for downstream; projection is SSL-specific |
| Loss computation scope | Across batch (Option B) | More negatives = better contrastive signal; standard practice |
| Per-level weighting | Configurable, default equal (Option B) | Flexibility for ablations; equal is paper's default |

---

## Phase 3 Output (Input to Phase 4)

```python
encoder_output = {
    'g2l': {'level_0': (N0, 96), 'level_1': (N1, 96), ...},
    'l2g': {'level_0': (N0, 96), 'level_1': (N1, 96), ...},
    'offsets_by_level': {'level_0': (B+1,), 'level_1': (B+1,), ...},
    'point_feats_g2l': (total_points, 96),
    'point_feats_l2g': (total_points, 96),
    'point_offset': (B+1,)
}
```

**Key insight:** Features at the same level have the same count (N_i) because both branches process the same hierarchy. The `offsets_by_level` tensor tells us which features belong to which scene in the batch.

---

## Components to Implement

### 1. ContrastiveProjectionHead

**Purpose:** Project encoder features to a space optimized for contrastive learning.

**Why needed:**
- Standard practice in SimCLR, MoCo, PointContrast
- Allows encoder to learn general features while projection learns contrastive-specific mapping
- Projection head is discarded after pretraining; encoder features are used downstream

**Architecture:**
```
Input (D_encoder) -> Linear -> BN -> ReLU -> Linear -> Output (D_projection)
```

**Interface:**
```python
class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 96,
        hidden_dim: int = 96,
        output_dim: int = 128,
        num_layers: int = 2
    ):
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Features (N, input_dim)
        Returns:
            Projected features (N, output_dim), L2-normalized
        """
```

**Critical:** Output MUST be L2-normalized for cosine similarity computation.

---

### 2. InfoNCELoss

**Purpose:** Core contrastive loss function.

**Mathematical formulation (from paper Section 3.6):**

For features `F_g2l` and `F_l2g` at the same level:

```
S_jk = (F_j_g2l)^T · F_k_l2g / τ

L = -(1/N) * Σ_j log( exp(S_jj) / Σ_k exp(S_jk) )
```

Where:
- `S_jj` = positive pair (same region, different branches)
- `S_jk` (j≠k) = negative pairs (different regions)
- `τ` = temperature hyperparameter

**Interface:**
```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        ...
    
    def forward(
        self,
        query: Tensor,      # (N, D) - e.g., G2L features
        key: Tensor,        # (N, D) - e.g., L2G features  
        offsets: Tensor = None  # (B+1,) - optional scene boundaries
    ) -> Tensor:
        """
        Args:
            query: Query features (N, D), L2-normalized
            key: Key features (N, D), L2-normalized
            offsets: Scene boundaries for within-scene negatives only (optional)
        
        Returns:
            Scalar loss
        """
```

**Implementation notes:**

1. **Similarity matrix:** `sim = query @ key.T / temperature`  → (N, N)

2. **Labels:** Diagonal entries are positives. For standard batch-wide negatives:
   ```python
   labels = torch.arange(N, device=query.device)
   loss = F.cross_entropy(sim, labels)
   ```

3. **Numerical stability:** Subtract max before softmax:
   ```python
   sim = sim - sim.max(dim=1, keepdim=True).values
   ```

4. **Symmetry (optional):** Can compute loss in both directions and average:
   ```python
   loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
   ```

**Temperature guidance:**
- Common values: 0.07 (MoCo), 0.1 (SimCLR), 0.5 (PointContrast)
- Lower τ → sharper distribution, harder negatives emphasized
- Higher τ → softer distribution, more uniform gradients
- Start with τ=0.1, tune based on training dynamics

---

### 3. RegionContrastiveLoss

**Purpose:** Compute InfoNCE loss across all hierarchy levels.

**Interface:**
```python
class RegionContrastiveLoss(nn.Module):
    def __init__(
        self,
        projection_head: ContrastiveProjectionHead,
        temperature: float = 0.1,
        level_weights: Dict[str, float] = None  # Optional per-level weights
    ):
        ...
    
    def forward(
        self,
        g2l_features: Dict[str, Tensor],  # {'level_0': (N0, D), ...}
        l2g_features: Dict[str, Tensor],  # {'level_0': (N0, D), ...}
        offsets_by_level: Dict[str, Tensor]  # {'level_0': (B+1,), ...}
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Returns:
            total_loss: Scalar, weighted sum across levels
            per_level_losses: Dict for logging/debugging
        """
```

**Algorithm:**
```python
def forward(self, g2l_features, l2g_features, offsets_by_level):
    losses = {}
    
    for level_name in g2l_features.keys():
        g2l = g2l_features[level_name]  # (N_level, D)
        l2g = l2g_features[level_name]  # (N_level, D)
        offsets = offsets_by_level[level_name]
        
        # Project
        g2l_proj = self.projection_head(g2l)  # (N_level, D_proj)
        l2g_proj = self.projection_head(l2g)  # (N_level, D_proj)
        
        # Compute InfoNCE
        loss = self.infonce(g2l_proj, l2g_proj, offsets)
        losses[level_name] = loss
    
    # Weighted sum
    weights = self.level_weights or {k: 1.0 for k in losses}
    total = sum(weights[k] * losses[k] for k in losses)
    
    return total, losses
```

**Design choice:** Use same projection head for all levels vs. separate heads per level.
- **Recommendation:** Same head (simpler, fewer parameters, encourages scale-invariant representations)
- **Alternative:** Per-level heads if ablations show benefit

---

### 4. PointContrastiveLoss

**Purpose:** Fine-grained contrastive loss at point level.

**Interface:**
```python
class PointContrastiveLoss(nn.Module):
    def __init__(
        self,
        projection_head: ContrastiveProjectionHead,
        temperature: float = 0.1,
        num_samples: int = 4096  # Subsample for memory efficiency
    ):
        ...
    
    def forward(
        self,
        point_feats_g2l: Tensor,  # (total_points, D)
        point_feats_l2g: Tensor,  # (total_points, D)
        point_offset: Tensor      # (B+1,)
    ) -> Tensor:
        """
        Returns:
            Scalar loss
        """
```

**Memory consideration:**
Point clouds can have 100k+ points. Computing full (100k, 100k) similarity matrix is infeasible.

**Solution: Random subsampling (following PointContrast)**
```python
def forward(self, point_feats_g2l, point_feats_l2g, point_offset):
    N = point_feats_g2l.shape[0]
    
    if N > self.num_samples:
        # Random subsample
        indices = torch.randperm(N, device=point_feats_g2l.device)[:self.num_samples]
        point_feats_g2l = point_feats_g2l[indices]
        point_feats_l2g = point_feats_l2g[indices]
    
    # Project
    g2l_proj = self.projection_head(point_feats_g2l)
    l2g_proj = self.projection_head(point_feats_l2g)
    
    # InfoNCE
    return self.infonce(g2l_proj, l2g_proj)
```

**Alternative: Hard negative mining**
Instead of random sampling, select points with high similarity to other points (harder negatives). More complex but potentially better gradients.

---

### 5. DBBDContrastiveLoss (Main Loss Module)

**Purpose:** Combines region and point losses with configurable weights.

**From paper Section 3.6:**
```
L_total = α * L_region + β * L_global
```

**Interface:**
```python
class DBBDContrastiveLoss(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 96,
        projection_dim: int = 128,
        temperature: float = 0.1,
        alpha: float = 1.0,      # Region loss weight
        beta: float = 0.5,       # Point loss weight
        level_weights: Dict[str, float] = None,
        point_num_samples: int = 4096
    ):
        ...
    
    def forward(self, encoder_output: Dict) -> Tuple[Tensor, Dict]:
        """
        Args:
            encoder_output: Phase 3 output dict
        
        Returns:
            total_loss: Scalar
            loss_dict: {
                'total': ...,
                'region': ...,
                'point': ...,
                'region_level_0': ...,
                'region_level_1': ...,
                ...
            }
        """
```

**Implementation:**
```python
def forward(self, encoder_output):
    # Region loss
    region_loss, region_losses = self.region_loss(
        encoder_output['g2l'],
        encoder_output['l2g'],
        encoder_output['offsets_by_level']
    )
    
    # Point loss
    point_loss = self.point_loss(
        encoder_output['point_feats_g2l'],
        encoder_output['point_feats_l2g'],
        encoder_output['point_offset']
    )
    
    # Combined
    total_loss = self.alpha * region_loss + self.beta * point_loss
    
    # Logging dict
    loss_dict = {
        'total': total_loss,
        'region': region_loss,
        'point': point_loss,
        **{f'region_{k}': v for k, v in region_losses.items()}
    }
    
    return total_loss, loss_dict
```

---

## Implementation Order

1. **ContrastiveProjectionHead** — Simple MLP, easy to test
2. **InfoNCELoss** — Core building block, test with synthetic data
3. **RegionContrastiveLoss** — Wire up projection + InfoNCE
4. **PointContrastiveLoss** — Add subsampling logic
5. **DBBDContrastiveLoss** — Combine everything
6. **Integration test** — Full forward pass with real encoder output

---

## Testing Strategy

### Unit Tests

**TestContrastiveProjectionHead:**
- [ ] Output shape correct
- [ ] Output is L2-normalized (norm ≈ 1.0)
- [ ] Gradients flow
- [ ] Handles variable batch sizes

**TestInfoNCELoss:**
- [ ] Loss is scalar
- [ ] Loss decreases when positives are more similar
- [ ] Loss increases when positives are less similar
- [ ] Temperature affects loss magnitude
- [ ] Symmetric version matches expected behavior
- [ ] Handles batch size = 1 edge case
- [ ] Gradients flow to both query and key

**TestRegionContrastiveLoss:**
- [ ] Processes all levels
- [ ] Per-level losses returned correctly
- [ ] Level weights applied correctly
- [ ] Handles missing levels gracefully (or raises clear error)

**TestPointContrastiveLoss:**
- [ ] Subsampling works when N > num_samples
- [ ] Full batch used when N <= num_samples
- [ ] Subsampling is random (different each call)

**TestDBBDContrastiveLoss:**
- [ ] Accepts Phase 3 output format
- [ ] Alpha/beta weights applied correctly
- [ ] Loss dict contains all expected keys
- [ ] Gradients flow to encoder (via encoder_output tensors)

### Integration Tests

**TestEndToEnd:**
- [ ] Full pipeline: batch → encoder → loss
- [ ] Gradients flow from loss to encoder parameters
- [ ] Training loop runs without error
- [ ] Loss decreases over iterations (sanity check)
- [ ] No memory leaks

### Numerical Tests

**TestNumericalStability:**
- [ ] No NaN/Inf with random inputs
- [ ] No NaN/Inf with very similar features (near-collapse)
- [ ] No NaN/Inf with very dissimilar features
- [ ] Handles zero vectors gracefully (or raises clear error)

---

## Common Pitfalls

### 1. Forgetting L2 normalization
InfoNCE assumes unit vectors. Without normalization, dot products can be arbitrarily large, causing numerical instability.

**Fix:** Always normalize before computing similarity:
```python
query = F.normalize(query, dim=-1)
key = F.normalize(key, dim=-1)
```

### 2. Wrong similarity matrix dimensions
If query is (N, D) and key is (M, D) with N ≠ M, the loss will fail or give wrong results.

**Fix:** Assert shapes match:
```python
assert query.shape[0] == key.shape[0], f"Mismatched batch: {query.shape[0]} vs {key.shape[0]}"
```

### 3. Temperature too low
Very low temperature (e.g., 0.01) can cause numerical overflow in exp().

**Fix:** Use reasonable range (0.05 - 0.5), or clamp similarities:
```python
sim = torch.clamp(sim / temperature, max=80)  # Prevent exp overflow
```

### 4. Gradient explosion with large batches
Large similarity matrices can cause gradient issues.

**Fix:** Monitor gradient norms, use gradient clipping if needed.

### 5. Point loss dominating
If point_num_samples is large, point loss may have many more terms than region loss, dominating gradients.

**Fix:** Scale beta appropriately, or normalize losses by number of samples.

### 6. Collapse to constant features
If all features become identical, loss goes to log(N) and gradients vanish.

**Fix:** 
- Monitor feature variance during training
- Use batch normalization in projection head
- Ensure augmentations are diverse enough

---

## Hyperparameter Defaults

Based on literature review (SimCLR, MoCo, PointContrast):

```python
defaults = {
    'encoder_dim': 96,
    'projection_dim': 128,
    'projection_hidden_dim': 96,
    'projection_num_layers': 2,
    'temperature': 0.1,
    'alpha': 1.0,
    'beta': 0.5,
    'level_weights': None,  # Equal weights
    'point_num_samples': 4096,
}
```

**Temperature sensitivity:**
- 0.07: Used by MoCo, aggressive hard negative focus
- 0.1: Used by SimCLR, balanced
- 0.5: Used by some PointContrast variants, softer

Start with 0.1, tune based on:
- Training loss curves (should decrease smoothly)
- Feature uniformity (should not collapse)
- Downstream task performance

---

## Output Specification

**DBBDContrastiveLoss.forward() returns:**

```python
(
    total_loss,  # Scalar tensor, requires_grad=True
    {
        'total': Tensor,           # Same as total_loss
        'region': Tensor,          # α * L_region
        'point': Tensor,           # β * L_point  
        'region_level_0': Tensor,  # L_region at level 0
        'region_level_1': Tensor,  # L_region at level 1
        ...
    }
)
```

**Usage in training loop:**
```python
loss, loss_dict = criterion(encoder_output)
loss.backward()
optimizer.step()

# Logging
wandb.log({f'loss/{k}': v.item() for k, v in loss_dict.items()})
```

---

## File Structure

```
dbbd/models/loss/
├── __init__.py
├── projection.py        # ContrastiveProjectionHead
├── infonce.py           # InfoNCELoss
├── region_loss.py       # RegionContrastiveLoss
├── point_loss.py        # PointContrastiveLoss
└── dbbd_loss.py         # DBBDContrastiveLoss (main)

tests/
└── test_loss.py         # All loss tests
```

---

## Success Criteria

- [ ] All unit tests pass
- [ ] Integration test with real encoder output passes
- [ ] Loss decreases during training (sanity check)
- [ ] No NaN/Inf in any scenario
- [ ] Gradients flow to encoder parameters
- [ ] Memory usage stable during training
- [ ] Code follows project conventions

---

## References

- **SimCLR:** Chen et al., 2020 — Temperature analysis, projection head design
- **MoCo:** He et al., 2020 — Temperature = 0.07
- **PointContrast:** Xie et al., 2020 — Point-level contrastive, subsampling
- **InfoNCE:** van den Oord et al., 2018 — Original loss formulation
- **Temperature schedules:** Kukleva et al., ICLR 2023 — Temperature effects on learning

---

## Questions to Resolve

Before implementing, verify:

1. **Shared vs separate projection heads for region vs point loss?**
   - Recommendation: Separate (they operate on different semantic granularities)
   
2. **Same projection head across all hierarchy levels?**
   - Recommendation: Same (simpler, scale-invariant)

3. **Gradient detach on one branch?**
   - Some methods (BYOL) detach gradients from one branch to prevent collapse
   - Recommendation: Don't detach initially; add if collapse occurs

These can be decided during implementation based on initial experiments.
