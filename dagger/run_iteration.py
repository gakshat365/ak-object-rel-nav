"""
DAgger Single Iteration Runner.

Executes one complete DAgger iteration:
  1. Rollout Collection  (beta-interleaved, expert labelled)
  2. Data Aggregation    (merge into growing D_agg manifest)
  3. Policy Fine-Tuning  (MSE waypoint loss on D_agg)
  4. Recovery Evaluation (R-SR, R-SPL, Soft-SPL on held-out test set)

Usage (from repo root):
  pixi run python dagger/run_iteration.py \
      --iteration 0 \
      --beta 0.7 \
      --checkpoint <path_to_base.pth> \
      [--config dagger/config.yaml]
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[RunIteration]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dagger.core.failure_loader import FailureLoader
from dagger.core.rollout_collector import RolloutCollector
from dagger.core.data_aggregator import DataAggregator
from dagger.core.trainer import DAggerTrainer
from dagger.evaluate_recovery import evaluate_checkpoint_on_cases


def _ensure_dirs(cfg: dict):
    for key in ["checkpoints_dir", "manifests_dir", "logs_dir", "output_dir"]:
        path = REPO_ROOT / cfg["paths"][key]
        path.mkdir(parents=True, exist_ok=True)


def run_iteration(
    iteration_idx: int,
    beta: float,
    checkpoint_path: str,
    cfg: dict,
    config_path: str,
    prior_manifest: str | None = None,
    skip_rollout: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
) -> dict:
    """
    Run one full DAgger iteration and return a results dict with metrics.

    Args:
        iteration_idx:    0-indexed DAgger iteration number.
        beta:             Expert mixing probability for this iteration.
        checkpoint_path:  Policy checkpoint to seed the rollout (and fine-tune from).
        cfg:              Parsed config.yaml dictionary.
        config_path:      Path to config.yaml (str).
        prior_manifest:   Path to the aggregated manifest from previous iterations
                          (None for iteration 0 → only current rollout data is used).
        skip_rollout:     Skip rollout collection (re-use existing iteration manifest).
        skip_train:       Skip training (useful when evaluating pre-trained baselines).
        skip_eval:        Skip evaluation step.

    Returns:
        dict with keys: iteration, beta, manifest_path, checkpoint_path, eval_summary.
    """
    _ensure_dirs(cfg)

    logs_dir = REPO_ROOT / cfg["paths"]["logs_dir"]
    manifests_dir = REPO_ROOT / cfg["paths"]["manifests_dir"]
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints_dir"]

    t0 = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Rollout Collection
    # ─────────────────────────────────────────────────────────────────────────
    iter_manifest = manifests_dir / f"iter_{iteration_idx}_manifest.csv"

    if not skip_rollout:
        logger.info(f"\n{'='*60}")
        logger.info(f"  STEP 1 — Rollout Collection  [iter={iteration_idx}, beta={beta}]")
        logger.info(f"{'='*60}")

        loader = FailureLoader(config_path)
        all_cases = loader.load_from_failed_cases_dir()
        train_cases, _, _ = loader.split_cases(all_cases)
        logger.info(f"Using {len(train_cases)} failure cases for rollout collection.")

        collector = RolloutCollector(config_path=config_path)
        output_samples_dir = REPO_ROOT / cfg["paths"]["data_dir"] / f"iter_{iteration_idx}" / "samples"
        records = collector.collect_iteration(
            cases=train_cases,
            checkpoint_path=checkpoint_path,
            beta=beta,
            iteration_idx=iteration_idx,
            output_dir=output_samples_dir,
        )
        aggregator_helper = DataAggregator(config_path)
        # Write to iter_manifest
        with open(iter_manifest, "w", newline="") as f:
            import csv
            from dagger.core.data_aggregator import MANIFEST_FIELDS
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for r in records:
                row = {k: r.get(k, "") for k in MANIFEST_FIELDS}
                writer.writerow(row)
        logger.info(f"Iteration manifest written with {len(records)} samples: {iter_manifest}")
    else:
        if not iter_manifest.exists():
            raise FileNotFoundError(
                f"skip_rollout=True but manifest not found: {iter_manifest}"
            )
        logger.info(f"[SKIP] Rollout collection. Using existing: {iter_manifest}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Data Aggregation
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  STEP 2 — Data Aggregation  [iter={iteration_idx}]")
    logger.info(f"{'='*60}")

    aggregator = DataAggregator(config_path)
    manifests_to_merge = [str(iter_manifest)]
    if prior_manifest and Path(prior_manifest).exists():
        manifests_to_merge.insert(0, prior_manifest)
        logger.info(f"Merging with prior manifest: {prior_manifest}")

    agg_manifest = manifests_dir / f"aggregated_iter_{iteration_idx}.csv"
    aggregator.merge_manifests(manifests_to_merge, output_manifest=str(agg_manifest))
    n_samples = sum(1 for _ in open(agg_manifest)) - 1  # subtract header
    logger.info(f"Aggregated manifest: {agg_manifest}  ({n_samples} samples)")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Policy Fine-Tuning
    # ─────────────────────────────────────────────────────────────────────────
    fine_tuned_ckpt = str(ckpt_dir / f"policy_iter_{iteration_idx}.pth")

    if not skip_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"  STEP 3 — Fine-Tuning  [iter={iteration_idx}]")
        logger.info(f"{'='*60}")

        trainer = DAggerTrainer(config_path=str(REPO_ROOT / "dagger" / "config.yaml"))
        fine_tuned_ckpt = trainer.train(
            manifest_path=str(agg_manifest),
            base_checkpoint_path=checkpoint_path,
            iteration_idx=iteration_idx,
        )
        logger.info(f"Fine-tuned checkpoint saved: {fine_tuned_ckpt}")
    else:
        logger.info(f"[SKIP] Training. Using checkpoint: {fine_tuned_ckpt}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Recovery Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    eval_summary = {}
    if not skip_eval:
        logger.info(f"\n{'='*60}")
        logger.info(f"  STEP 4 — Recovery Evaluation  [iter={iteration_idx}]")
        logger.info(f"{'='*60}")

        loader = FailureLoader(config_path)
        all_cases = loader.load_from_failed_cases_dir()
        _, _, test_cases = loader.split_cases(all_cases)
        logger.info(f"Evaluating on {len(test_cases)} held-out test cases.")

        eval_ckpt = fine_tuned_ckpt if not skip_train else checkpoint_path
        eval_summary = evaluate_checkpoint_on_cases(
            checkpoint_path=eval_ckpt,
            cases=test_cases,
            cfg=cfg,
            split_name="test",
        )

        # Save per-iteration eval report
        eval_out = logs_dir / f"eval_iter_{iteration_idx}.json"
        with open(eval_out, "w") as f:
            json.dump(eval_summary, f, indent=2)
        logger.info(f"Evaluation report saved: {eval_out}")
    else:
        logger.info("[SKIP] Evaluation step.")

    elapsed = time.time() - t0
    logger.info(f"\nIteration {iteration_idx} complete in {elapsed/60:.1f} minutes.")

    return {
        "iteration": iteration_idx,
        "beta": beta,
        "manifest_path": str(agg_manifest),
        "checkpoint_path": fine_tuned_ckpt,
        "eval_summary": eval_summary,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="DAgger Single Iteration Runner")
    parser.add_argument("--iteration", type=int, required=True, help="0-indexed iteration number")
    parser.add_argument("--beta", type=float, required=True, help="Expert mixing probability (0..1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Seed checkpoint path")
    parser.add_argument(
        "--prior-manifest",
        type=str,
        default=None,
        help="Aggregated manifest from prior iterations (optional)",
    )
    parser.add_argument("--config", type=str, default="dagger/config.yaml")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = args.checkpoint or cfg["paths"]["base_checkpoint"]

    result = run_iteration(
        iteration_idx=args.iteration,
        beta=args.beta,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        config_path=str(config_path),
        prior_manifest=args.prior_manifest,
        skip_rollout=args.skip_rollout,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
    )

    # Print final metrics
    es = result.get("eval_summary", {})
    logger.info("\n=== Iteration Result ===")
    logger.info(f"  Checkpoint : {result['checkpoint_path']}")
    if es:
        logger.info(f"  R-SR       : {es.get('recovery_success_rate', 'N/A'):.2f}%")
        logger.info(f"  R-SPL      : {es.get('recovery_spl', 'N/A'):.2f}%")
        logger.info(f"  Soft-SPL   : {es.get('recovery_soft_spl', 'N/A'):.2f}%")
        logger.info(f"  Dist Closed: {es.get('mean_distance_closed', 'N/A'):.3f} m")


if __name__ == "__main__":
    main()
