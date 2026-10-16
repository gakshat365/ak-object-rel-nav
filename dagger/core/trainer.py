"""
DAgger Recovery Policy Fine-Tuner.

Fine-tunes the ObjectReact / GNM model on an aggregated DAgger recovery dataset.
Uses the native waypoint MSE loss (identical to the original training pipeline).
Saves checkpoints to dagger/checkpoints/policy_iter_{i}.pth and latest.pth.
"""

import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[Trainer]")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "libs" / "control" / "object_react" / "train"))

from vint_train.training.train_eval_loop import load_model
from vint_train.training.train_utils import get_obs_image, get_goal_image


class DAggerTrainer:
    """
    Fine-tunes ObjectReact policy on DAgger aggregated recovery samples.
    """

    def __init__(self, config_path: str = "dagger/config.yaml"):
        import yaml
        cfg_file = REPO_ROOT / config_path
        with open(cfg_file, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.train_cfg = self.cfg["training"]
        self.ckpt_dir = REPO_ROOT / self.cfg["paths"]["checkpoints_dir"]
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        logger.info(f"Trainer using device: {self.device}")

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, checkpoint_path: str):
        """Instantiate GNM and load weights from a checkpoint file."""
        from libs.control.objectreact import ObjRelLearntController
        import yaml

        ctrl_cfg_path = REPO_ROOT / self.cfg["paths"]["controller_config"]
        with open(ctrl_cfg_path, "r") as f:
            ctrl_cfg = yaml.safe_load(f)
        ctrl_cfg["load_run"] = str(checkpoint_path)
        controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")
        model = controller.model
        return model

    def _compute_loss(
        self,
        model: torch.nn.Module,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        action_label: torch.Tensor,
        action_mask: torch.Tensor,
        alpha: float = 0.05,
    ) -> torch.Tensor:
        """Forward pass and MSE waypoint action loss (no distance head for DAgger)."""
        obs_in, _ = get_obs_image(obs_img, "disabled", self.transform, self.device)
        goal_in, _ = get_goal_image(goal_img, "image_mask_enc", self.transform, self.device)

        _, action_pred = model(obs_in, goal_in)  # (B, 10, 4)
        action_label = action_label.to(self.device)
        action_mask = action_mask.to(self.device)

        # MSE loss, masked to valid samples
        per_sample_loss = F.mse_loss(action_pred, action_label, reduction="none")
        # Average across waypoints and action dims, then mask
        while per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.mean(dim=-1)
        loss = (per_sample_loss * action_mask).mean() / (action_mask.mean() + 1e-2)
        return loss

    def train(
        self,
        manifest_path: str,
        base_checkpoint_path: str,
        iteration_idx: int,
        epochs: Optional[int] = None,
        val_fraction: float = 0.15,
        save_every: int = 1,
    ) -> str:
        """
        Fine-tune from base_checkpoint_path on the dataset in manifest_path.
        Saves dagger/checkpoints/policy_iter_{iteration_idx}.pth and updates latest.pth.
        Returns path to the saved checkpoint.
        """
        from dagger.core.data_aggregator import DAggerRecoveryDataset

        epochs = epochs or self.train_cfg.get("epochs_per_iter", 10)
        batch_size = self.train_cfg.get("batch_size", 32)
        lr = float(self.train_cfg.get("learning_rate", 1e-4))
        weight_decay = float(self.train_cfg.get("weight_decay", 1e-4))
        num_workers = self.train_cfg.get("num_workers", 0)

        logger.info(f"\n=== DAgger Fine-Tuning: Iteration {iteration_idx} ===")
        logger.info(f"Manifest: {manifest_path}")
        logger.info(f"Base checkpoint: {base_checkpoint_path}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

        # --- Dataset & DataLoader ---
        full_dataset = DAggerRecoveryDataset(manifest_path)
        n_total = len(full_dataset)
        if n_total == 0:
            raise ValueError(f"Empty dataset at {manifest_path}")

        n_val = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info(f"Dataset split: {n_train} train / {n_val} val (total {n_total})")

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # --- Model ---
        model = self._load_model(base_checkpoint_path)
        model = model.to(self.device)
        model.train()

        # --- Optimizer ---
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_ckpt_path = None

        for epoch in range(1, epochs + 1):
            # --- Training epoch ---
            model.train()
            train_losses = []
            train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] Training", leave=False, dynamic_ncols=True)
            for batch in train_pbar:
                obs_img, goal_img, action_label, _, _, _, action_mask = batch
                optimizer.zero_grad()
                loss = self._compute_loss(model, obs_img, goal_img, action_label, action_mask)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            scheduler.step()
            mean_train = float(np.mean(train_losses)) if train_losses else 0.0

            # --- Validation ---
            model.eval()
            val_losses = []
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch}/{epochs}] Validation", leave=False, dynamic_ncols=True)
            with torch.no_grad():
                for batch in val_pbar:
                    obs_img, goal_img, action_label, _, _, _, action_mask = batch
                    loss = self._compute_loss(model, obs_img, goal_img, action_label, action_mask)
                    val_losses.append(loss.item())
                    val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            mean_val = float(np.mean(val_losses)) if val_losses else 0.0

            logger.info(
                f"Epoch [{epoch}/{epochs}] "
                f"train_loss={mean_train:.4f}  val_loss={mean_val:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            # --- Checkpoint ---
            if epoch % save_every == 0 or epoch == epochs:
                checkpoint = {
                    "epoch": epoch,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "train_loss": mean_train,
                    "val_loss": mean_val,
                    "iteration": iteration_idx,
                    "manifest": str(manifest_path),
                }
                iter_ckpt_path = self.ckpt_dir / f"policy_iter_{iteration_idx}.pth"
                latest_ckpt_path = self.ckpt_dir / "latest.pth"
                torch.save(checkpoint, iter_ckpt_path)
                torch.save(checkpoint, latest_ckpt_path)

                if mean_val < best_val_loss:
                    best_val_loss = mean_val
                    best_ckpt_path = str(iter_ckpt_path)
                    torch.save(checkpoint, self.ckpt_dir / f"policy_iter_{iteration_idx}_best.pth")
                    logger.info(f"  → New best val_loss={best_val_loss:.4f}, saved to {best_ckpt_path}")

        logger.info(
            f"\nTraining complete. Best val_loss={best_val_loss:.4f}. "
            f"Checkpoint: {best_ckpt_path}"
        )
        return best_ckpt_path or str(self.ckpt_dir / f"policy_iter_{iteration_idx}.pth")
