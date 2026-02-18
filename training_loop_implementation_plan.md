# Implementation Plan - Training Loop

We will implement a robust training loop for the DBBD framework using Test-Driven Development (TDD). The implementation will focus on correctness, modularity, and the ability to run on local machines with limited resources via a "Small Backbone" mode.

## User Review Required

> [!IMPORTANT]
> **Small Backbone Mode**: To address local machine limitations, we will introduce a `small_mode` in the configuration. This will:
> 1. Use a lighter PointNet backbone (fewer filters).
> 2. specificy a `SyntheticDataset` for testing (tiny point clouds, shallow hierarchy).
> 3. Disable heavy logging/profiling by default.

## Proposed Changes

### `dbbd/training/`
We will create a new package `dbbd.training` to house the training logic.

#### [NEW] [config.py](file:///c:/Users/antho/DBBD/dbbd/training/config.py)
Defines the `TrainingConfig` dataclass to manage hyperparameters, model settings, and paths.

#### [NEW] [trainer.py](file:///c:/Users/antho/DBBD/dbbd/training/trainer.py)
Implements the `Trainer` class with the following methods:
- `__init__`: Initializes Model, Optimizer, Loss, Dataloaders based on config.
- `train_step`: Performs a single forward/backward pass.
- `validate`: Runs validation loop.
- `save_checkpoint`: Saves model state.
- `fit`: Main training loop iterating over epochs.

### `tests/`
#### [NEW] [test_training_loop.py](file:///c:/Users/antho/DBBD/tests/test_training_loop.py)
This is the core of our TDD approach. It will contain tests that we will write *before* the implementation:
1. `test_trainer_initialization`: Verifies components are loaded correctly.
2. `test_train_step_gradient`: Runs one step and checks if gradients are generated.
3. `test_overfit_one_batch`: Checks if loss decreases on a single repeated batch.
4. `test_save_load_checkpoint`: Verifies state persistence.
5. `test_small_backbone_mode`: Verifies the model runs with reduced capacity.

## Verification Plan

### Automated Tests
We will run the newly created tests using `pytest`.
```bash
pytest tests/test_training_loop.py -v
```

### Manual Verification
After TDD is complete, we will create a run script `train.py` and run a short training session (2 epochs) in "Small Backbone" mode to verify end-to-end functionality.
```bash
python train.py --config configs/debug_local.yaml
```
(We will create `train.py` and the config as the final step).
