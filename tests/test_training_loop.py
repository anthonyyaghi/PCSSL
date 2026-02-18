"""
Unit Tests for DBBD Training Loop

TDD approach: Tests written BEFORE implementation.
Uses REAL data from datasets/scannet for meaningful validation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

import sys
sys.path.append(str(Path(__file__).parent.parent))

DATA_ROOT = Path(__file__).parent.parent / "datasets" / "scannet"


def has_real_data():
    """Check if real ScanNet data is available."""
    train_dir = DATA_ROOT / "train"
    return train_dir.exists() and any(train_dir.glob("*.pkl"))


requires_data = pytest.mark.skipif(
    not has_real_data(),
    reason="Real ScanNet data not available"
)


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_initialization(self):
        """Test default config values."""
        from dbbd.training.config import TrainingConfig

        config = TrainingConfig()

        assert config.hidden_dim == 96
        assert config.encoder_hidden_dims == [64, 128]
        assert config.small_mode == False
        assert config.batch_size == 8

    def test_small_mode_overrides(self):
        """Test small_mode auto-adjusts parameters."""
        from dbbd.training.config import TrainingConfig

        config = TrainingConfig(small_mode=True)

        assert config.encoder_hidden_dims == [32, 64]
        assert config.hidden_dim == 64
        assert config.output_dim == 64
        assert config.batch_size == 2
        assert config.num_workers == 0
        assert config.max_train_scenes == 4
        assert config.max_val_scenes == 2

    def test_yaml_roundtrip(self, tmp_path):
        """Test config save/load from YAML."""
        from dbbd.training.config import TrainingConfig

        config = TrainingConfig(learning_rate=0.005, num_epochs=50)
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(str(yaml_path))

        loaded = TrainingConfig.from_yaml(str(yaml_path))

        assert loaded.learning_rate == 0.005
        assert loaded.num_epochs == 50

    def test_custom_overrides_not_affected_by_small_mode(self):
        """Test that explicit values override small_mode defaults."""
        from dbbd.training.config import TrainingConfig

        config = TrainingConfig(
            small_mode=True,
            batch_size=4,
            max_train_scenes=10
        )

        assert config.batch_size == 4
        assert config.max_train_scenes == 10


class TestDataLoading:
    """Test data loading with real ScanNet data."""

    @requires_data
    def test_load_real_scene(self):
        """Test DBBDDataset loads a real .pkl file."""
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=False,
            max_scenes=1
        )

        assert len(dataset) >= 1

        sample = dataset[0]
        assert "coord" in sample
        assert "feat" in sample
        assert "hierarchy" in sample
        assert sample["coord"].shape[1] == 3

    @requires_data
    def test_dual_view_format(self):
        """Test dual_view returns view1, view2."""
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=1
        )

        sample = dataset[0]

        assert "view1" in sample
        assert "view2" in sample
        assert "coord" in sample["view1"]
        assert "coord" in sample["view2"]

    @requires_data
    def test_collation(self):
        """Test collate_fn produces correct batch format."""
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=2
        )

        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        collated = collate_fn(batch)

        assert "view1" in collated
        assert "view2" in collated
        assert "hierarchies" in collated
        assert "offset" in collated["view1"]

    @requires_data
    def test_max_scenes_limits_dataset(self):
        """Test max_scenes parameter limits dataset size."""
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor

        full_dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=False
        )

        limited_dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=False,
            max_scenes=3
        )

        assert len(limited_dataset) == 3
        assert len(limited_dataset) < len(full_dataset)


class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_trainer_creates_components(self):
        """Test trainer initializes encoder, loss, optimizer."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer

        config = TrainingConfig(small_mode=True, device="cpu")
        trainer = Trainer(config)

        assert trainer.encoder is not None
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_small_mode_encoder_dims(self):
        """Test small_mode creates smaller encoder."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer

        config = TrainingConfig(small_mode=True, device="cpu")
        trainer = Trainer(config)

        assert trainer.config.hidden_dim == 64
        assert trainer.config.encoder_hidden_dims == [32, 64]

    def test_device_placement(self):
        """Test model is placed on correct device."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer

        config = TrainingConfig(small_mode=True, device="cpu")
        trainer = Trainer(config)

        param = next(trainer.encoder.parameters())
        assert param.device.type == "cpu"


class TestTrainStep:
    """Test single training step."""

    @requires_data
    def test_train_step_returns_loss(self):
        """Test train_step returns scalar loss."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn

        config = TrainingConfig(small_mode=True, device="cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(config)

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=2
        )
        batch = collate_fn([dataset[0], dataset[1]])

        loss, metrics = trainer.train_step(batch)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert "total" in metrics
        assert "region" in metrics
        assert "point" in metrics

    @requires_data
    def test_train_step_updates_gradients(self):
        """Test train_step generates gradients and updates params."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn

        config = TrainingConfig(small_mode=True, device="cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(config)

        initial_params = {
            name: param.clone().detach()
            for name, param in trainer.encoder.named_parameters()
        }

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=2
        )
        batch = collate_fn([dataset[0], dataset[1]])

        trainer.train_step(batch)

        changed = False
        for name, param in trainer.encoder.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                changed = True
                break

        assert changed, "No parameters were updated"


class TestOverfitOneBatch:
    """Test overfitting on a single batch - KEY TEST."""

    @requires_data
    def test_loss_decreases_on_real_data(self):
        """Test loss decreases when overfitting one batch of real data."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn

        config = TrainingConfig(
            small_mode=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            learning_rate=1e-2
        )
        trainer = Trainer(config)

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=2
        )
        batch = collate_fn([dataset[0], dataset[1]])

        losses = []
        for _ in range(20):
            loss, _ = trainer.train_step(batch)
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

        improvement = (losses[0] - losses[-1]) / losses[0]
        assert improvement > 0.05, f"Loss only improved by {improvement*100:.1f}%"


class TestCheckpointing:
    """Test checkpoint save/load."""

    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint is saved correctly."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer

        config = TrainingConfig(small_mode=True, device="cpu")
        trainer = Trainer(config)
        trainer.current_epoch = 5
        trainer.global_step = 100

        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(str(ckpt_path))

        assert ckpt_path.exists()

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["epoch"] == 5
        assert ckpt["global_step"] == 100
        assert "encoder_state_dict" in ckpt
        assert "loss_fn_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt

    def test_load_checkpoint_restores_state(self, tmp_path):
        """Test loading restores training state."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer

        config = TrainingConfig(small_mode=True, device="cpu")
        trainer1 = Trainer(config)
        trainer1.current_epoch = 10
        trainer1.global_step = 500
        trainer1.best_val_loss = 0.5

        ckpt_path = tmp_path / "test.pth"
        trainer1.save_checkpoint(str(ckpt_path))

        trainer2 = Trainer(config)
        trainer2.load_checkpoint(str(ckpt_path))

        assert trainer2.current_epoch == 10
        assert trainer2.global_step == 500
        assert trainer2.best_val_loss == 0.5

    def test_checkpoint_weights_restored(self, tmp_path):
        """Test that model weights are correctly restored."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer

        config = TrainingConfig(small_mode=True, device="cpu")
        trainer1 = Trainer(config)

        original_weights = {
            name: param.clone()
            for name, param in trainer1.encoder.named_parameters()
        }

        ckpt_path = tmp_path / "test.pth"
        trainer1.save_checkpoint(str(ckpt_path))

        trainer2 = Trainer(config)
        trainer2.load_checkpoint(str(ckpt_path))

        for name, param in trainer2.encoder.named_parameters():
            assert torch.allclose(param, original_weights[name])


class TestValidation:
    """Test validation loop."""

    @requires_data
    def test_validate_returns_metrics(self):
        """Test validate returns averaged metrics."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn
        from torch.utils.data import DataLoader

        config = TrainingConfig(small_mode=True, device="cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(config)

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=4
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        metrics = trainer.validate(loader)

        assert "total" in metrics
        assert "region" in metrics
        assert "point" in metrics
        assert all(not np.isnan(v) for v in metrics.values())

    @requires_data
    def test_validate_no_gradient(self):
        """Test validation doesn't compute gradients."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn
        from torch.utils.data import DataLoader

        config = TrainingConfig(small_mode=True, device="cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(config)

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=2
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        for param in trainer.encoder.parameters():
            param.grad = None

        trainer.validate(loader)

        for param in trainer.encoder.parameters():
            assert param.grad is None


class TestFit:
    """Test full training loop."""

    @requires_data
    def test_fit_runs_epochs(self):
        """Test fit runs specified number of epochs."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn
        from torch.utils.data import DataLoader

        config = TrainingConfig(
            small_mode=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_epochs=3,
            log_interval=1,
            eval_interval=1,
            save_interval=10,
            verbose=False
        )
        trainer = Trainer(config)

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=4
        )
        train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        val_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        history = trainer.fit(train_loader, val_loader)

        assert len(history["train_loss"]) == 3
        assert trainer.current_epoch == 2

    @requires_data
    def test_fit_loss_decreases(self):
        """Test loss decreases over epochs."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn
        from torch.utils.data import DataLoader

        config = TrainingConfig(
            small_mode=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_epochs=5,
            learning_rate=1e-2,
            verbose=False
        )
        trainer = Trainer(config)

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=4
        )
        train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        history = trainer.fit(train_loader)

        assert history["train_loss"][-1] < history["train_loss"][0]


class TestSmallBackboneMode:
    """Test small backbone mode specifically."""

    @requires_data
    def test_full_pipeline_small_mode(self):
        """Test complete training pipeline in small mode."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer
        from dbbd.datasets.dbbd_dataset import DBBDDataset
        from dbbd.datasets.transforms import ToTensor
        from dbbd.models.utils.batch import collate_fn
        from torch.utils.data import DataLoader

        config = TrainingConfig(
            small_mode=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_epochs=2,
            verbose=False
        )
        trainer = Trainer(config)

        dataset = DBBDDataset(
            data_root=str(DATA_ROOT),
            split="train",
            transform=ToTensor(),
            dual_view=True,
            max_scenes=4
        )
        train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        history = trainer.fit(train_loader)

        assert len(history["train_loss"]) == 2
        assert all(np.isfinite(l) for l in history["train_loss"])

    @requires_data
    def test_small_mode_memory_efficient(self):
        """Test small mode uses less memory than full mode."""
        from dbbd.training.config import TrainingConfig
        from dbbd.training.trainer import Trainer

        small_config = TrainingConfig(small_mode=True, device="cpu")
        small_trainer = Trainer(small_config)
        small_params = sum(p.numel() for p in small_trainer.encoder.parameters())

        full_config = TrainingConfig(small_mode=False, device="cpu")
        full_trainer = Trainer(full_config)
        full_params = sum(p.numel() for p in full_trainer.encoder.parameters())

        assert small_params < full_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
