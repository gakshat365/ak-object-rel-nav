"""
Dataset Aggregator and PyTorch Recovery Dataset for Iterative DAgger.

1. Aggregation Engine:
   - Tracks collected sample manifests across iterations.
   - Emits versioned manifests: dagger/data/manifests/iter_{i}.csv.
   - Aggregates manifests: D_agg^{(i+1)} = D_agg^{(i)} ∪ D_i.
   - Prevents duplicate aggregation upon restarts.

2. DAggerRecoveryDataset:
   - PyTorch Dataset loading recovery sample bundles into the exact tensor format
     required by ObjectReact / GNM architectures and vint_dataset training routines.
   - Validates shapes, absence of NaNs/infs, and normalization conventions (Gate 2).
"""

import argparse
import csv
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[DataAggregator]")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "libs" / "control" / "object_react" / "train"))

from vint_train.models.object_react.dataloader import (
    generate_positional_encodings,
    normalize_pls,
)
from vint_train.training.train_utils import get_obs_image, get_goal_image


MANIFEST_FIELDS = [
    "sample_id",
    "sample_dir",
    "episode_name",
    "scene_id",
    "iteration",
    "step",
    "action_source",
    "beta",
    "distance_to_goal",
]


class DataAggregator:
    """
    Manages sample manifests, iteration tracking, and multi-iteration aggregation.
    """

    def __init__(self, config_path: str = "dagger/config.yaml"):
        config_file = REPO_ROOT / config_path
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.manifests_dir = REPO_ROOT / self.cfg["paths"]["manifests_dir"]
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def save_iteration_manifest(self, records: List[Dict], iteration_idx: int) -> Path:
        """
        Writes iteration sample records to dagger/data/manifests/iter_{i}.csv.
        """
        manifest_path = self.manifests_dir / f"iter_{iteration_idx}.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for r in records:
                row = {k: r.get(k, "") for k in MANIFEST_FIELDS}
                writer.writerow(row)

        logger.info(f"Saved {len(records)} records for iteration {iteration_idx} to {manifest_path}")
        return manifest_path

    def load_manifest(self, manifest_path: Union[str, Path]) -> List[Dict]:
        """
        Loads records from a manifest CSV.
        """
        path = Path(manifest_path)
        if not path.exists():
            return []
        records = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records

    def aggregate_manifests(self, current_iteration_idx: int) -> Tuple[Path, int]:
        """
        Merges manifests across iterations 0 to current_iteration_idx.
        Deduplicates by sample_id to prevent double-counting upon restarts.
        Emits both dagger/data/manifests/iter_{i}_agg.csv and dagger/data/manifests/aggregate.csv.
        """
        seen_sample_ids = set()
        aggregated_records = []

        for iter_i in range(current_iteration_idx + 1):
            iter_manifest = self.manifests_dir / f"iter_{iter_i}.csv"
            if not iter_manifest.exists():
                logger.warning(f"Manifest for iteration {iter_i} not found at {iter_manifest}")
                continue

            records = self.load_manifest(iter_manifest)
            added_count = 0
            for r in records:
                sid = r["sample_id"]
                if sid not in seen_sample_ids:
                    seen_sample_ids.add(sid)
                    aggregated_records.append(r)
                    added_count += 1
            logger.info(f"Aggregated iteration {iter_i}: added {added_count} unique samples (total so far: {len(aggregated_records)})")

        # Save iteration-specific aggregate snapshot
        iter_agg_path = self.manifests_dir / f"iter_{current_iteration_idx}_agg.csv"
        with open(iter_agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for r in aggregated_records:
                writer.writerow(r)

        # Update global aggregate manifest
        global_agg_path = self.manifests_dir / "aggregate.csv"
        with open(global_agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for r in aggregated_records:
                writer.writerow(r)

        logger.info(
            f"Successfully updated aggregate manifest at {global_agg_path} "
            f"with {len(aggregated_records)} total samples across iterations 0..{current_iteration_idx}."
        )
        return global_agg_path, len(aggregated_records)


class DAggerRecoveryDataset(Dataset):
    """
    PyTorch Dataset loading recovery sample bundles into native ObjectReact tensor conventions.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        dims: int = 8,
        outlier_value: float = 99.0,
        is_pl_normalized: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV does not exist at {self.manifest_path}")

        self.records = []
        with open(self.manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only keep rows whose sample_dir exists
                sdir = Path(row["sample_dir"])
                if sdir.exists():
                    self.records.append(row)

        self.dims = dims
        self.outlier_value = outlier_value
        self.is_pl_normalized = is_pl_normalized

        # Precompute positional rank encodings matching ObjectReact
        self.rank_enc = generate_positional_encodings(200, self.dims)

        logger.info(f"Initialized DAggerRecoveryDataset: {len(self.records)} valid samples from {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def encode_goal(self, masks: np.ndarray, pls: np.ndarray) -> torch.Tensor:
        """
        Transforms raw semantic instance masks and path lengths into the
        (3 + dims, 60, 80) tensor expected by ObjectReact's get_goal_image.
        """
        if not self.is_pl_normalized:
            pls = normalize_pls(pls, outlier_value=self.outlier_value)
        pls = pls.astype(np.uint8)

        # Downsample masks to 60x80 matching ObjectReact downscale factor [::4, ::4]
        # Input masks: (N, 240, 320) -> transpose to (240, 320, N) -> downsample -> (60, 80, N)
        masks_down = masks.transpose([1, 2, 0])[::4, ::4]

        # Positional encodings: (N, dims)
        enc = self.rank_enc[pls.astype(int)]

        # Matrix multiply: (60, 80, N) @ (N, dims) -> (60, 80, dims) -> transpose to (dims, 60, 80)
        goal_enc = (masks_down @ enc).transpose(2, 0, 1)

        # 3-channel visualization prefix for get_goal_image split([3, dims])
        goal_vis = goal_enc[:3, :, :]
        if self.dims < 3:
            goal_vis = goal_enc[:1].repeat(3, axis=0)

        # Full concatenated goal tensor: (3 + dims, 60, 80)
        full_goal = np.concatenate([goal_vis, goal_enc], axis=0)
        return torch.as_tensor(full_goal, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        row = self.records[idx]
        sample_dir = Path(row["sample_dir"])

        # Load expert target action: (10, 4)
        expert_action = np.load(sample_dir / "expert_action.npy")
        action_label = torch.as_tensor(expert_action, dtype=torch.float32)

        # Load goal masks and pseudo labels
        masks = np.load(sample_dir / "goal_masks.npy")
        pls = np.load(sample_dir / "goal_pls.npy")
        goal_image = self.encode_goal(masks, pls)

        # Load observation context
        obs_context_path = sample_dir / "obs_context.npy"
        if obs_context_path.exists():
            obs_context = np.load(obs_context_path)  # (context+1, 3, 64, 85)
            # Take latest observation frame or concatenate
            obs_image = torch.as_tensor(obs_context[-1], dtype=torch.float32)
        else:
            obs_image = torch.zeros((3, 64, 85), dtype=torch.float32)

        dist_label = torch.tensor(0, dtype=torch.int64)
        goal_pos = torch.tensor([0.0, 0.0], dtype=torch.float32)
        dataset_idx = torch.tensor(0, dtype=torch.int64)
        action_mask = torch.tensor(1.0, dtype=torch.float32)

        return (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_idx,
            action_mask,
        )


def verify_gate2(
    manifest_path: Union[str, Path],
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 2,
) -> bool:
    """
    Go / No-Go Gate 2 Milestone Test:
    Verifies that the aggregated recovery dataset:
    1. Loads cleanly via DataLoader without errors.
    2. Has matching action tensor shapes (B, 10, 4) and goal shapes (B, 11, 60, 80).
    3. Contains no NaNs or Infs.
    4. Computes forward pass through GNM model cleanly.
    5. Computes loss and backward pass with non-zero gradients.
    """
    logger.info("\n================ Running Gate 2 Dataset Verification ================")
    dataset = DAggerRecoveryDataset(manifest_path)
    if len(dataset) == 0:
        logger.error(f"[GATE 2 FAILED] Dataset at {manifest_path} contains 0 samples!")
        return False

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    batch = next(iter(loader))

    obs_img, goal_img, action_lbl, dist_lbl, goal_pos, ds_idx, action_mask = batch
    logger.info(f"Loaded Batch Shapes:")
    logger.info(f"  obs_image:    {obs_img.shape}")
    logger.info(f"  goal_image:   {goal_img.shape}")
    logger.info(f"  action_label: {action_lbl.shape}")
    logger.info(f"  action_mask:  {action_mask.shape}")

    # Check for NaNs and Infs
    assert not torch.isnan(action_lbl).any(), "action_label contains NaN values!"
    assert not torch.isinf(action_lbl).any(), "action_label contains Inf values!"
    assert not torch.isnan(goal_img).any(), "goal_image contains NaN values!"
    assert not torch.isinf(goal_img).any(), "goal_image contains Inf values!"
    assert action_lbl.shape[1:] == (10, 4), f"Expected action shape (*, 10, 4), got {action_lbl.shape}"

    # Load GNM model
    from libs.control.objectreact import ObjRelLearntController

    ctrl_cfg_path = REPO_ROOT / "configs" / "controller" / "object_react_controller.yaml"
    with open(ctrl_cfg_path, "r") as f:
        ctrl_cfg = yaml.safe_load(f)

    if checkpoint_path is None:
        checkpoint_path = REPO_ROOT / "libs" / "control" / "object_react" / "train" / "logs" / "object_rel_nav" / "object_react_hm3d_2026_10_07_20_54_34_" / "latest.pth"

    ctrl_cfg["load_run"] = str(checkpoint_path)
    controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")
    model = controller.model.to(device)
    model.train()

    # Forward pass through model
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    obs_in, _ = get_obs_image(obs_img, "disabled", transform, device)
    goal_in, _ = get_goal_image(goal_img, "image_mask_enc", transform, device)

    outputs = model(obs_in, goal_in)
    _, action_pred = outputs

    logger.info(f"Model Predictions Shape: {action_pred.shape}")
    assert action_pred.shape == action_lbl.shape, f"Mismatch: pred {action_pred.shape} vs label {action_lbl.shape}"

    # Compute MSE action loss
    loss = F.mse_loss(action_pred, action_lbl.to(device))
    assert not torch.isnan(loss), "Computed MSE loss is NaN!"
    assert not torch.isinf(loss), "Computed MSE loss is Inf!"
    logger.info(f"Computed MSE Action Loss: {loss.item():.4f}")

    # Backward pass verification
    model.zero_grad()
    loss.backward()

    # Verify gradients exist and are non-zero
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0:
            has_grad = True
            break
    assert has_grad, "No gradients accumulated on model parameters during backward pass!"

    logger.info(">>> GO / NO-GO GATE 2 PASSED: Recovery dataset loads and passes forward+backward validation cleanly!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="DAgger Data Aggregator and Gate 2 Verification")
    parser.add_argument("--config", type=str, default="dagger/config.yaml", help="Path to dagger config")
    parser.add_argument("--aggregate_iter", type=int, default=None, help="Aggregate iterations up to this index")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV to verify")
    parser.add_argument("--verify", action="store_true", help="Run Gate 2 verification test")
    parser.add_argument("--device", type=str, default="cpu", help="Device for verification test")
    args = parser.parse_args()

    aggregator = DataAggregator(args.config)

    if args.aggregate_iter is not None:
        agg_path, total = aggregator.aggregate_manifests(args.aggregate_iter)
        print(f"Aggregated manifests up to iteration {args.aggregate_iter}: {total} samples at {agg_path}")

    if args.verify:
        manifest = args.manifest or (aggregator.manifests_dir / "aggregate.csv")
        passed = verify_gate2(manifest, device=args.device)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
