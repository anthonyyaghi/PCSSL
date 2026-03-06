# Spine / Backbone Migration Plan: Integrating SparseUNet for DBBD and Baselines

This document outlines the explicit architectural requirements and implementation steps for migrating the `DBBD` framework from a simple `PointNet` backbone to a modern `SpUNet` (Sparse UNet) sourced from Pointcept.

This restructuring serves a dual purpose:
1.  **Enhance our custom pipeline** (Setup 1) with state-of-the-art scene-level feature extraction.
2.  **Establish a direct baseline comparison** (Setup 2) using standard Pointcept SSL frameworks (like MSC or PointContrast) leveraging the exact same backbone architecture.

---

## Part 1: Dependency & Environment Updates
The `SpUNet` requires sparse tensor operations. While some frameworks use `MinkowskiEngine`, the Pointcept standard and our selected path is `spconv`.

1.  **Update `requirements.txt` / Environment Configuration:**
    *   Add `spconv-cu118` (ensure the CUDA version matches the project's PyTorch build, e.g., 11.8).
    *   Ensure any Pointcept utility dependencies required for the `SpUNet` extraction are satisfied (e.g., `tensorboard`, `einops`).

---

## Part 2: Sourcing and Adapting SpUNet
We will port the core `SpUNet` implementation from Pointcept to maintain architectural purity for the baseline comparison, avoiding the overhead of including the entire Pointcept repository as a submodule if possible.

1.  **Create Backbone Module:**
    *   Create a new file: `dbbd/models/encoder/spunet.py`.
    *   Extract the `SpUNet-v1m1` or standard `SpUNet` architecture from Pointcept's model repository.
    *   Wrap it in a standard PyTorch `nn.Module` that accepts quantified sparse coordinates and features.

2.  **Define the Interface:**
    The new backbone must support transforming continuous point clouds into the sparse voxel format required by `spconv`.
    *   **Input:** `coords` (N, 3) and `features` (N, C).
    *   **Pre-processing (Voxelization):** A quantization layer that maps `coords` to sparse tensor indices based on a defined `voxel_size` (e.g., 0.05m or 0.02m).
    *   **Network Pass:** The `spconv` layers process the `SparseTensor`.
    *   **Post-processing (Desparsification):** The resultant sparse features are mapped back (broadcasted) to the original `(N, 3)` query coordinates to produce the final `(N, output_dim)` dense point features.

3.  **Update `PointCloudEncoder` (`dbbd/models/encoder/point_encoder.py`):**
    *   Modify the `__init__` to accept `backbone='spunet'` alongside the existing `'pointnet'`.
    *   Add configuration arguments for `voxel_size` and `spconv` parameters.
    *   In the `forward` pass, branch logic: if `spunet`, route the `coords` and `feats` through the voxelization -> SpUNet -> desparsification pipeline.
    *   Retain the pooling logic (`max` or `mean`) to derive the region-level feature from the output point-level features.

---

## Part 3: Establishing the Two Experimental Setups

To prove the efficacy of the DBBD approach against established norms, we will implement two distinct train loops/configurations that utilize the exact same `SpUNet` module.

### Setup 1: Our Custom Approach (DBBD)
*This setup integrates SpUNet into the existing DBBD hierarchical contrastive learning framework.*

1.  **Configuration (`configs/train_dbbd_spunet.yaml`):**
    *   Set `backbone: spunet`.
    *   Define `voxel_size` to balance memory and resolution.
    *   Keep existing DBBD loss weights (`temperature`, `alpha`, `beta`).
2.  **Data Flow:**
    *   Dataloader provides `(M, 3)` augmented local regions and global scenes.
    *   `PointCloudEncoder` processes these using the `SpUNet` logic defined in Part 2.
    *   Outputs feed into our custom `DBBDContrastiveLoss` (Region and Point contrastive heads).

### Setup 2: Baseline Pointcept Framework (MSC / PointContrast)
*This setup uses the SpUNet in a standard, point-to-point contrastive learning paradigm across two augmented views of the same scene, mirroring a Pointcept baseline.*

1.  **Baseline Module Creation (`dbbd/models/baselines/msc_contrastive.py`):**
    *   Implement a standard InfoNCE or MSC-style loss module. This will be an alternative to `DBBDContrastiveLoss`.
    *   The loss should operate on dense point correspondences between two overlapping, differently augmented views of a whole scene.
2.  **Configuration (`configs/train_baseline_msc.yaml`):**
    *   Set `backbone: spunet` (identical to Setup 1).
    *   Configure dataloaders to output two independent views of the same scan (e.g., `view1`, `view2` with distinct random rotations and jitter).
    *   Switch the loss function to the MSC/PointContrast baseline loss.
3.  **Training Loop Adjustment:**
    *   If using the same training loop (`tests/test_training_loop.py` or main script), ensure it can cleanly branch between a "DBBD Mode" (passing hierarchical regions to loss) and a "Baseline Mode" (passing two views of a scene to the baseline loss).

---

## Part 4: Downstream Evaluation Pipeline
To compare the pre-trained weights from Setup 1 and Setup 2 directly.

1.  **Finetuning Script:**
    *   Create a finetuning script (e.g., Semantic Segmentation on ScanNet / S3DIS) that takes pre-trained `SpUNet` weights.
2.  **Standardized Layers:**
    *   Attach a simple linear classifier or lightweight decoder to the `SpUNet` outputs.
3.  **Metrics:**
    *   Compare `mIoU` across the Setup 1 and Setup 2 weights under identical finetuning regimes (epochs, learning rate, frozen vs. unfrozen backbone).
