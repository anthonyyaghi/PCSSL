"""DBBD Trainer"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
from datetime import datetime
from tqdm import tqdm

from .config import TrainingConfig
from ..models.encoder.hierarchical_encoder import HierarchicalEncoder
from ..models.loss.dbbd_loss import DBBDContrastiveLoss

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Trainer:
    def __init__(self, config: TrainingConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_rank_zero = local_rank == 0

        if self.is_distributed:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(
                config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
            )

        self._setup_seed()
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._setup_tensorboard()

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _setup_seed(self):
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _build_model(self):
        encoder_config = {
            "type": "pointnet",
            "pooling": self.config.encoder_pooling,
            "hidden_dims": self.config.encoder_hidden_dims,
        }
        propagator_config = {
            "hidden_dim": self.config.propagator_hidden_dim,
            "num_layers": self.config.propagator_num_layers,
        }
        aggregator_config = {
            "mode": self.config.aggregator_mode,
            "use_spatial_context": self.config.aggregator_use_spatial,
        }

        self.encoder = HierarchicalEncoder(
            input_feat_dim=self.config.input_feat_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            propagator_config=propagator_config,
            aggregator_config=aggregator_config,
            encoder_config=encoder_config,
        ).to(self.device)

        self.loss_fn = DBBDContrastiveLoss(
            encoder_dim=self.config.output_dim,
            projection_dim=self.config.projection_dim,
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            beta=self.config.beta,
            point_num_samples=self.config.point_num_samples,
        ).to(self.device)

        if self.is_distributed:
            self.encoder = DDP(
                self.encoder, device_ids=[self.local_rank], find_unused_parameters=False
            )
            self.loss_fn = DDP(
                self.loss_fn, device_ids=[self.local_rank], find_unused_parameters=False
            )

    def _unwrap(self, module):
        return module.module if self.is_distributed else module

    def _build_optimizer(self):
        params = list(self.encoder.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self):
        if self.config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.T_max,
                eta_min=self.config.eta_min,
            )
        else:
            self.scheduler = None

    def _setup_tensorboard(self):
        self.writer = None
        if not self.is_rank_zero:
            return
        if TENSORBOARD_AVAILABLE:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(self.config.log_dir) / run_name
            self.writer = SummaryWriter(log_dir=str(log_path))
            logger.info(f"TensorBoard logging to {log_path}")
        else:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")

    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def _log_lr(self, step: int):
        if self.writer is not None:
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/learning_rate", lr, step)

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        def move(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            elif isinstance(obj, dict):
                return {k: move(v) for k, v in obj.items()}
            else:
                return obj

        moved = move(batch)
        if "hierarchies" in batch:
            moved["hierarchies"] = batch["hierarchies"]
        return moved

    @torch.no_grad()
    def _compute_embedding_metrics(self, encoder_output: Dict) -> Dict[str, float]:
        metrics = {}

        g2l_pts = encoder_output['point_feats_g2l']
        l2g_pts = encoder_output['point_feats_l2g']
        g2l_norm = F.normalize(g2l_pts, dim=1)
        l2g_norm = F.normalize(l2g_pts, dim=1)
        alignment = (g2l_norm * l2g_norm).sum(dim=1).mean()
        metrics['alignment'] = alignment.item()

        feat_std = g2l_pts.std().item()
        metrics['feat_std'] = feat_std

        feat_norm = g2l_pts.norm(dim=1).mean().item()
        metrics['feat_norm'] = feat_norm

        return metrics

    def train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.encoder.train()
        self.loss_fn.train()

        batch = self._move_batch_to_device(batch)

        self.optimizer.zero_grad()

        encoder_output = self.encoder(batch)
        total_loss, loss_dict = self.loss_fn(encoder_output)

        embed_metrics = self._compute_embedding_metrics(encoder_output)

        total_loss.backward()
        self.optimizer.step()

        metrics = {k: v.item() for k, v in loss_dict.items()}
        metrics.update(embed_metrics)
        self.global_step += 1

        return total_loss, metrics

    def _reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not self.is_distributed:
            return metrics
        reduced = {}
        for k, v in metrics.items():
            tensor = torch.tensor(v, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            reduced[k] = tensor.item()
        return reduced

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.encoder.eval()
        self.loss_fn.eval()

        total_metrics = {}
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False,
                          disable=not self.is_rank_zero or not self.config.verbose):
            batch = self._move_batch_to_device(batch)
            encoder_output = self.encoder(batch)
            _, loss_dict = self.loss_fn(encoder_output)

            embed_metrics = self._compute_embedding_metrics(encoder_output)

            for k, v in loss_dict.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            for k, v in embed_metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v

            num_batches += 1

        if num_batches > 0:
            for k in total_metrics:
                total_metrics[k] /= num_batches

        return self._reduce_metrics(total_metrics)

    def save_checkpoint(self, path: str, extra: Optional[Dict] = None):
        if not self.is_rank_zero:
            return

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "encoder_state_dict": self._unwrap(self.encoder).state_dict(),
            "loss_fn_state_dict": self._unwrap(self.loss_fn).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if extra:
            checkpoint.update(extra)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

        if self.config.verbose:
            logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._unwrap(self.encoder).load_state_dict(checkpoint["encoder_state_dict"])
        self._unwrap(self.loss_fn).load_state_dict(checkpoint["loss_fn_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.is_rank_zero and self.config.verbose:
            logger.info(f"Loaded checkpoint from {path}, epoch={self.current_epoch}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        train_sampler: Optional[DistributedSampler] = None,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": []}

        show_progress = self.is_rank_zero and self.config.verbose

        epoch_pbar = tqdm(
            range(self.current_epoch, self.config.num_epochs),
            desc="Training",
            unit="epoch",
            disable=not show_progress
        )

        for epoch in epoch_pbar:
            self.current_epoch = epoch

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

            batch_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                leave=False,
                disable=not show_progress
            )

            for batch in batch_pbar:
                loss, metrics = self.train_step(batch)
                epoch_loss += loss.item()
                num_batches += 1

                self._log_metrics(metrics, self.global_step, prefix="train")
                self._log_lr(self.global_step)

                if show_progress:
                    batch_pbar.set_postfix({
                        'loss': f"{loss.item():.3f}",
                        'align': f"{metrics.get('alignment', 0):.3f}",
                        'std': f"{metrics.get('feat_std', 0):.3f}",
                    })

            avg_train_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_train_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            val_loss = None
            if val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics.get("total", float("inf"))
                history["val_loss"].append(val_loss)

                self._log_metrics(val_metrics, epoch, prefix="val")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"{self.config.checkpoint_dir}/best.pth")

            if show_progress:
                epoch_pbar.set_postfix({
                    'loss': f"{avg_train_loss:.4f}",
                    'val': f"{val_loss:.4f}" if val_loss else "N/A"
                })

            if self.writer is not None:
                self.writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"{self.config.checkpoint_dir}/epoch_{epoch+1}.pth")

        if self.writer is not None:
            self.writer.close()

        return history
