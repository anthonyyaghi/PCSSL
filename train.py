"""
DBBD Training Script

Usage:
    python train.py --config configs/debug_local.yaml
    python train.py --small_mode

    # Multi-GPU (single machine):
    torchrun --nproc_per_node=2 train.py --config configs/debug_local.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dbbd.training.config import TrainingConfig
from dbbd.training.trainer import Trainer
from dbbd.datasets.dbbd_dataset import DBBDDataset
from dbbd.datasets.transforms import Compose, RandomRotate, RandomScale, ToTensor
from dbbd.models.utils.batch import collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train DBBD model")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--small_mode", action="store_true", help="Use small backbone mode")
    parser.add_argument("--data_root", type=str, help="Override data root path")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--device", type=str, help="Device (cuda or cpu)")
    args = parser.parse_args()

    is_distributed = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if is_distributed:
        backend = "gloo" if os.name == "nt" else "nccl"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    is_rank_zero = local_rank == 0

    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(small_mode=args.small_mode or False)

    if args.data_root:
        config.data_root = args.data_root
    if args.epochs:
        config.num_epochs = args.epochs
    if args.device:
        config.device = args.device
    if args.small_mode:
        config.small_mode = True
        config.__post_init__()

    if is_rank_zero:
        logger.info(f"Training config: small_mode={config.small_mode}, epochs={config.num_epochs}")
        logger.info(f"Device: {config.device}, Batch size: {config.batch_size}")
        logger.info(f"Data root: {config.data_root}")
        if is_distributed:
            logger.info(f"Distributed training: {world_size} GPUs")

    transform = Compose([
        RandomRotate(angle=[-180, 180], axis="z", p=0.95),
        RandomScale(scale=[0.9, 1.1], p=0.95),
        ToTensor(),
    ])

    train_dataset = DBBDDataset(
        data_root=config.data_root,
        split="train",
        dual_view=True,
        transform=transform,
        max_scenes=config.max_train_scenes,
    )

    val_dataset = DBBDDataset(
        data_root=config.data_root,
        split="val",
        dual_view=True,
        transform=ToTensor(),
        max_scenes=config.max_val_scenes,
    )

    train_sampler = None
    val_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if is_rank_zero:
        logger.info(f"Train: {len(train_dataset)} scenes, Val: {len(val_dataset)} scenes")

    trainer = Trainer(config, local_rank=local_rank, world_size=world_size)
    history = trainer.fit(train_loader, val_loader, train_sampler=train_sampler)

    if is_rank_zero:
        logger.info(f"Training complete. Final train loss: {history['train_loss'][-1]:.4f}")
        trainer.save_checkpoint(f"{config.checkpoint_dir}/final.pth")
        logger.info(f"Saved final checkpoint to {config.checkpoint_dir}/final.pth")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
