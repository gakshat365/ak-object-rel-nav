#!/usr/bin/env python3
"""
Master Multi-Chunk DAgger Orchestrator (Paradigm 2: Active Closed-Loop DAgger)

Controls the complete iterative DAgger recovery lifecycle across 5,000 trajectories:
  Chunk 0 (Episodes 0-1000):   Evaluate pi_0 -> mine traps -> DAgger recovery -> train pi_1 -> test
  Chunk 1 (Episodes 1000-2000): Evaluate pi_1 -> mine traps -> DAgger recovery -> train pi_2 -> test
  Chunk 2 (Episodes 2000-3000): Evaluate pi_2 -> mine traps -> DAgger recovery -> train pi_3 -> test
  Chunk 3 (Episodes 3000-4000): Evaluate pi_3 -> mine traps -> DAgger recovery -> train pi_4 -> test
  Chunk 4 (Episodes 4000-5000): Evaluate pi_4 -> mine traps -> DAgger recovery -> train pi_5 -> test

All experiment configurations, trajectory paths, failure definitions, controller settings,
and training parameters are centralized and configurable from this single script.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("[MultiChunkDAgger]")

REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# DEFAULT EXPERIMENT CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    # 1. Dataset & Paths
    "paths": {
        "trajectories_dir": "dagger/data/initial_data",
        "hm3d_root": "data/hm3d_v0.2",
        "base_checkpoint": "dagger/model/baseline_model/latest.pth",
        "controller_config": "dagger/model/baseline_model/config.yaml",
        "pipeline_workspace": "dagger/multi_chunk_runs",
    },
    # 2. Controller Settings (Preserving Original Baseline)
    "controller": {
        "type": "SafeTurnController",
        "use_adaptive_lookahead": False,  # SafeTurnController Flag OFF
        "waypoint_index": -1,             # Standard 10th waypoint lookahead
        "linear_vel": 0.25,
        "angular_vel": 0.5236,
    },
    # 3. Failure & Backtrack Definitions
    "failure_criteria": {
        "trapped_radius": 0.2,            # Stagnation radius in meters (0.2m)
        "trapped_steps": 20,              # Window of consecutive steps (20 steps)
        "lead_in_steps": 50,              # Backtrack 50 steps before trap state
        "threshold_dist": 1.0,            # Success distance threshold in meters
        "max_steps": 300,                 # Max steps per rollout
        "time_delta": 0.25,               # Physics simulation time step
    },
    # 4. Multi-Chunk Partitioning across 5,000 trajectories
    "chunking": {
        "total_trajectories": 5000,
        "num_chunks": 5,                  # 5 chunks of 1,000 trajectories each
        "chunk_size": 1000,
        "shuffle_trajectories": True,     # Mix/shuffle across scenes to avoid domain-shift
        "seed": 42,                       # Reproducible random seed
        "beta_start": 0.70,               # Initial expert mixing probability
        "beta_end": 0.05,                 # Final expert mixing probability
        "beta_schedule": None,            # Dynamically computed based on num_chunks if None
    },
    # 5. Training Hyperparameters
    "training": {
        "epochs_per_iter": 5,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "device": "cuda",
        "gpu_id": "1",
    },
}


def compute_beta_schedule(num_chunks: int, beta_start: float = 0.70, beta_end: float = 0.05) -> List[float]:
    """Dynamically computes linear beta annealing schedule based on number of chunks."""
    if num_chunks <= 1:
        return [beta_start]
    step = (beta_start - beta_end) / (num_chunks - 1)
    return [round(max(beta_end, beta_start - i * step), 3) for i in range(num_chunks)]


class MultiChunkPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.workspace = REPO_ROOT / cfg["paths"]["pipeline_workspace"]
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.workspace / "checkpoints"
        self.manifests_dir = self.workspace / "manifests"
        self.logs_dir = self.workspace / "logs"
        self.data_dir = self.workspace / "data"

        for d in [self.checkpoints_dir, self.manifests_dir, self.logs_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.gpu_id = str(cfg["training"].get("gpu_id", "1"))
        pixi_python = REPO_ROOT / ".pixi/envs/default/bin/python"
        self.python_bin = str(pixi_python) if pixi_python.exists() else sys.executable

        # Save active run config
        with open(self.workspace / "experiment_config.yaml", "w") as f:
            yaml.dump(self.cfg, f, indent=2)

    def _get_python_env(self) -> dict:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        pixi_lib = str(REPO_ROOT / ".pixi/envs/default/lib")
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{pixi_lib}:{existing_ld}" if existing_ld else pixi_lib
        return env

    def run_benchmark_chunk(
        self,
        chunk_idx: int,
        checkpoint_path: str,
        start_idx: int,
        max_episodes: int,
    ) -> Path:
        """Runs benchmark_safe_controller on chunk k using current checkpoint."""
        logger.info(f"\n{'='*70}")
        logger.info(f"  CHUNK {chunk_idx}: STEP 1 - BENCHMARK & MINE TRAPPED FAILURES")
        logger.info(f"  Range: Episodes [{start_idx} to {start_idx + max_episodes}]")
        logger.info(f"  Policy Checkpoint: {checkpoint_path}")
        logger.info(f"{'='*70}")

        chunk_manifest = self.manifests_dir / f"chunk_{chunk_idx}_trapped_manifest.csv"
        chunk_results_csv = self.logs_dir / f"chunk_{chunk_idx}_benchmark_results.csv"

        # Check if benchmark results and trapped manifest already exist from a previous run
        if chunk_manifest.exists() and chunk_results_csv.exists():
            with open(chunk_manifest, "r") as f:
                trapped_count = max(0, sum(1 for _ in f) - 1)
            if trapped_count > 0:
                logger.info(
                    f"Chunk {chunk_idx}: Found existing benchmark output with {trapped_count} trapped failure episodes "
                    f"({chunk_manifest.name}). Skipping benchmark."
                )
                return chunk_manifest, trapped_count

        cmd = [
            self.python_bin,
            "-u",
            "dagger/benchmark_safe_controller.py",
            "--data_dir", self.cfg["paths"]["trajectories_dir"],
            "--output_csv", str(chunk_results_csv),
            "--trapped_manifest", str(chunk_manifest),
            "--checkpoint", str(checkpoint_path),
            "--start_idx", str(start_idx),
            "--max_episodes", str(max_episodes),
            "--max_steps", str(self.cfg["failure_criteria"]["max_steps"]),
            "--threshold_dist", str(self.cfg["failure_criteria"]["threshold_dist"]),
            "--trapped_radius", str(self.cfg["failure_criteria"]["trapped_radius"]),
            "--trapped_steps", str(self.cfg["failure_criteria"]["trapped_steps"]),
            "--lead_in_steps", str(self.cfg["failure_criteria"]["lead_in_steps"]),
            "--gpu", self.gpu_id,
            "--mode", "flag_off_only",
            "--wp_a", str(self.cfg["controller"]["waypoint_index"]),
        ]
        if self.cfg["chunking"].get("shuffle_trajectories", False):
            cmd.extend([
                "--shuffle",
                "--seed", str(self.cfg["chunking"].get("seed", 42)),
            ])

        logger.info(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=self._get_python_env())
        if res.returncode != 0:
            raise RuntimeError(f"Benchmark failed on chunk {chunk_idx} with return code {res.returncode}")

        # Log trapped summary count
        trapped_count = 0
        if chunk_manifest.exists():
            with open(chunk_manifest, "r") as f:
                trapped_count = max(0, sum(1 for _ in f) - 1)
        logger.info(f"Chunk {chunk_idx} Benchmark Complete: Found {trapped_count} trapped failure episodes.")
        return chunk_manifest, trapped_count

    def run_dagger_iteration(
        self,
        chunk_idx: int,
        checkpoint_path: str,
        trapped_manifest: Path,
        beta: float,
        prior_aggregated_manifest: Optional[Path] = None,
    ) -> Dict:
        """Collects recovery rollouts, aggregates data, fine-tunes policy, and evaluates."""
        logger.info(f"\n{'='*70}")
        logger.info(f"  CHUNK {chunk_idx}: STEP 2 - DAGGER ROLLOUT, AGGREGATION & FINE-TUNING")
        logger.info(f"  Expert mixing beta: {beta:.2f}")
        logger.info(f"  Prior Manifest:     {prior_aggregated_manifest}")
        logger.info(f"{'='*70}")

        # Create chunk-specific config for old_dagger
        chunk_cfg = {
            "paths": {
                "repo_root": ".",
                "hm3d_scenes_path": self.cfg["paths"]["hm3d_root"],
                "trapped_cases_csv": str(trapped_manifest),
                "base_checkpoint": str(checkpoint_path),
                "controller_config": self.cfg["paths"]["controller_config"],
                "output_dir": str(self.workspace / f"chunk_{chunk_idx}_output"),
                "checkpoints_dir": str(self.checkpoints_dir),
                "data_dir": str(self.data_dir),
                "manifests_dir": str(self.manifests_dir),
                "logs_dir": str(self.logs_dir),
                "failures_cases_dir": "dagger/data/trapped_cases",
            },
            "dagger": {
                "failure_source": "trapped_csv",
                "split_ratios": {
                    "train": 1.0 - (self.cfg["training"]["val_fraction"] + self.cfg["training"]["test_fraction"]),
                    "val": self.cfg["training"]["val_fraction"],
                    "test": self.cfg["training"]["test_fraction"],
                },
                "split_by_scene": True,
                "seed": 42 + chunk_idx,
            },
            "simulation": {
                "width": 320,
                "height": 240,
                "hfov": 120,
                "sensor_height": 0.4,
                "sensor_height_map": 1.31,
                "time_delta": self.cfg["failure_criteria"]["time_delta"],
                "threshold_goal_distance": self.cfg["failure_criteria"]["threshold_dist"],
                "max_rollout_steps": self.cfg["failure_criteria"]["max_steps"],
                "num_workers": 1,
            },
            "model": {
                "len_traj_pred": 10,
                "learn_angle": True,
                "normalize": True,
                "metric_waypoint_spacing": 0.5,
                "waypoint_spacing": 1,
                "context_size": 5,
                "image_size": [85, 64],
            },
            "training": {
                "epochs_per_iter": self.cfg["training"]["epochs_per_iter"],
                "batch_size": self.cfg["training"]["batch_size"],
                "learning_rate": self.cfg["training"]["learning_rate"],
                "weight_decay": self.cfg["training"]["weight_decay"],
                "num_workers": 0,
                "device": self.cfg["training"]["device"],
            },
        }

        chunk_config_file = self.workspace / f"chunk_{chunk_idx}_dagger_config.yaml"
        with open(chunk_config_file, "w") as f:
            yaml.dump(chunk_cfg, f, indent=2)

        cmd = [
            self.python_bin,
            "-u",
            "old_dagger/run_iteration.py",
            "--iteration", str(chunk_idx),
            "--beta", str(beta),
            "--checkpoint", str(checkpoint_path),
            "--config", str(chunk_config_file),
        ]
        if prior_aggregated_manifest and prior_aggregated_manifest.exists():
            cmd.extend(["--prior-manifest", str(prior_aggregated_manifest)])

        logger.info(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=self._get_python_env())
        if res.returncode != 0:
            raise RuntimeError(f"DAgger iteration failed on chunk {chunk_idx} with code {res.returncode}")

        # Check outputs
        new_checkpoint = self.checkpoints_dir / f"policy_iter_{chunk_idx}.pth"
        current_aggregated_manifest = self.manifests_dir / f"aggregated_iter_{chunk_idx}.csv"
        eval_report_file = self.logs_dir / f"eval_iter_{chunk_idx}.json"

        eval_summary = {}
        if eval_report_file.exists():
            with open(eval_report_file, "r") as f:
                eval_summary = json.load(f)

        return {
            "chunk_idx": chunk_idx,
            "new_checkpoint": str(new_checkpoint),
            "aggregated_manifest": current_aggregated_manifest,
            "eval_summary": eval_summary,
        }

    def run_pipeline(
        self,
        start_chunk: int = 0,
        end_chunk: Optional[int] = None,
        dry_run: bool = False,
    ):
        """Runs the entire multi-chunk pipeline sequentially across all 5,000 trajectories."""
        num_chunks = self.cfg["chunking"]["num_chunks"]
        chunk_size = self.cfg["chunking"]["chunk_size"]
        total_trajectories = self.cfg["chunking"]["total_trajectories"]
        beta_schedule = self.cfg["chunking"].get("beta_schedule")
        if not beta_schedule:
            beta_schedule = compute_beta_schedule(
                num_chunks=num_chunks,
                beta_start=self.cfg["chunking"].get("beta_start", 0.70),
                beta_end=self.cfg["chunking"].get("beta_end", 0.05),
            )

        end_chunk = end_chunk if end_chunk is not None else num_chunks

        logger.info("\n" + "=" * 75)
        logger.info("  LAUNCHING MULTI-CHUNK DAGGER PIPELINE (5,000 TRAJECTORIES)")
        logger.info("=" * 75)
        logger.info(f"Total Trajectories:    {total_trajectories}")
        logger.info(f"Number of Chunks:      {num_chunks} (size = {chunk_size} episodes/chunk)")
        logger.info(f"Active Chunk Range:    Chunk {start_chunk} -> Chunk {end_chunk - 1}")
        logger.info(f"Beta Schedule:         {beta_schedule}")
        logger.info(f"Shuffle Trajectories:  {self.cfg['chunking'].get('shuffle_trajectories', True)} (seed={self.cfg['chunking'].get('seed', 42)})")
        logger.info(f"Base Model:            {self.cfg['paths']['base_checkpoint']}")
        logger.info(f"SafeTurn Flag:         {self.cfg['controller']['use_adaptive_lookahead']} (FLAG OFF)")
        logger.info(f"Trapped Definition:    <= {self.cfg['failure_criteria']['trapped_radius']}m over {self.cfg['failure_criteria']['trapped_steps']} steps")
        logger.info(f"Lead-in Backtrack:     {self.cfg['failure_criteria']['lead_in_steps']} steps")
        logger.info(f"Output Workspace:      {self.workspace}")
        logger.info("=" * 75 + "\n")

        current_checkpoint = self.cfg["paths"]["base_checkpoint"]
        prior_aggregated_manifest = None

        # Resume state if start_chunk > 0
        if start_chunk > 0:
            prev_ckpt = self.checkpoints_dir / f"policy_iter_{start_chunk - 1}.pth"
            if prev_ckpt.exists():
                current_checkpoint = str(prev_ckpt)
                logger.info(f"Resuming: found previous checkpoint {current_checkpoint}")
            prev_manifest = self.manifests_dir / f"aggregated_iter_{start_chunk - 1}.csv"
            if prev_manifest.exists():
                prior_aggregated_manifest = prev_manifest
                logger.info(f"Resuming: found previous aggregated manifest {prior_aggregated_manifest}")

        pipeline_records = []

        for chunk_idx in range(start_chunk, end_chunk):
            t_chunk_start = time.time()
            start_ep = chunk_idx * chunk_size
            ep_count = min(chunk_size, total_trajectories - start_ep)
            beta = beta_schedule[chunk_idx] if chunk_idx < len(beta_schedule) else beta_schedule[-1]

            logger.info(f"\n{'#'*75}")
            logger.info(f"### STARTING CHUNK {chunk_idx} / {num_chunks - 1} (Episodes {start_ep} to {start_ep + ep_count})")
            logger.info(f"### Seeding with Policy Checkpoint: {current_checkpoint}")
            logger.info(f"{'#'*75}")

            if dry_run:
                logger.info("[DRY RUN] Skipping actual execution.")
                continue

            # 1. Benchmark Chunk k with current policy -> mine trapped cases
            trapped_manifest, trapped_count = self.run_benchmark_chunk(
                chunk_idx=chunk_idx,
                checkpoint_path=current_checkpoint,
                start_idx=start_ep,
                max_episodes=ep_count,
            )

            if trapped_count == 0:
                logger.info(f"Chunk {chunk_idx}: 0 trapped failures mined! Keeping current checkpoint {current_checkpoint}.")
                continue

            # 2. Run DAgger Iteration (Rollout -> Aggregate -> Train -> Eval)
            iter_result = self.run_dagger_iteration(
                chunk_idx=chunk_idx,
                checkpoint_path=current_checkpoint,
                trapped_manifest=trapped_manifest,
                beta=beta,
                prior_aggregated_manifest=prior_aggregated_manifest,
            )

            # Update pointers for next chunk
            current_checkpoint = iter_result["new_checkpoint"]
            prior_aggregated_manifest = iter_result["aggregated_manifest"]
            elapsed_min = (time.time() - t_chunk_start) / 60.0

            iter_record = {
                "chunk": chunk_idx,
                "episodes_range": f"{start_ep}-{start_ep + ep_count}",
                "beta": beta,
                "checkpoint": current_checkpoint,
                "elapsed_minutes": round(elapsed_min, 1),
                "eval_summary": iter_result["eval_summary"],
            }
            pipeline_records.append(iter_record)

            # Save cumulative pipeline progress report
            with open(self.workspace / "pipeline_summary.json", "w") as f:
                json.dump(pipeline_records, f, indent=2)

            self.print_summary_table(pipeline_records)

        logger.info("\n" + "=" * 75)
        logger.info("  MULTI-CHUNK PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info(f"  Final Policy Checkpoint: {current_checkpoint}")
        logger.info(f"  Summary Report Saved:    {self.workspace / 'pipeline_summary.json'}")
        logger.info("=" * 75 + "\n")

    def print_summary_table(self, records: List[Dict]):
        """Prints a clean tabular overview of progress across all completed chunks."""
        logger.info("\n" + "=" * 85)
        logger.info(f"{'Chunk':<7} | {'Episodes':<11} | {'Beta':<6} | {'R-SR (%)':<9} | {'Soft-SPL (%)':<13} | {'Dist Closed':<12} | {'Time':<8}")
        logger.info("-" * 85)
        for r in records:
            es = r.get("eval_summary", {})
            sr = es.get("recovery_success_rate", 0.0)
            sspl = es.get("recovery_soft_spl", 0.0)
            dc = es.get("mean_distance_closed", 0.0)
            time_str = f"{r.get('elapsed_minutes', 0.0)}m"
            logger.info(f"{r['chunk']:<7} | {r['episodes_range']:<11} | {r['beta']:<6.2f} | {sr:<9.2f} | {sspl:<13.2f} | {dc:<12.3f} | {time_str:<8}")
        logger.info("=" * 85 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Master Multi-Chunk DAgger Orchestrator (5,000 Trajectories)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom YAML config file (optional)")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to 5,000 trajectories directory")
    parser.add_argument("--base_checkpoint", type=str, default=None, help="Override path to initial base checkpoint")
    parser.add_argument("--workspace", type=str, default=None, help="Override pipeline output workspace folder")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA GPU ID (e.g., '1')")
    parser.add_argument("--num_chunks", type=int, default=None, help="Number of chunks across the 5,000 trajectories (default 5)")
    parser.add_argument("--chunk_size", type=int, default=None, help="Number of episodes per chunk (default 1000)")
    parser.add_argument("--start_chunk", type=int, default=0, help="Chunk index to start/resume from (0-indexed)")
    parser.add_argument("--end_chunk", type=int, default=None, help="Chunk index to stop before")
    parser.add_argument("--epochs", type=int, default=None, help="Fine-tuning epochs per iteration (default 5)")
    parser.add_argument("--trapped_radius", type=float, default=None, help="Trapped radius threshold in meters (default 0.2)")
    parser.add_argument("--trapped_steps", type=int, default=None, help="Trapped window consecutive steps (default 20)")
    parser.add_argument("--lead_in_steps", type=int, default=None, help="Backtrack lead-in steps before trap (default 50)")
    parser.add_argument("--dry_run", action="store_true", help="Print pipeline execution plan without running")
    parser.add_argument("--shuffle", action="store_true", default=None, help="Enable trajectory mixing across chunks")
    parser.add_argument("--no_shuffle", action="store_true", default=None, help="Disable trajectory mixing across chunks")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling (default 42)")
    parser.add_argument("--beta_start", type=float, default=None, help="Initial expert beta probability (default 0.70)")
    parser.add_argument("--beta_end", type=float, default=None, help="Final expert beta probability (default 0.05)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Base Configuration
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        cfg_path = REPO_ROOT / args.config
        with open(cfg_path, "r") as f:
            user_cfg = yaml.safe_load(f)
            # Deep merge
            for sec, vals in user_cfg.items():
                if isinstance(vals, dict) and sec in cfg:
                    cfg[sec].update(vals)
                else:
                    cfg[sec] = vals

    # 2. CLI Overrides
    if args.data_dir:
        cfg["paths"]["trajectories_dir"] = args.data_dir
    if args.base_checkpoint:
        cfg["paths"]["base_checkpoint"] = args.base_checkpoint
    if args.workspace:
        cfg["paths"]["pipeline_workspace"] = args.workspace
    if args.gpu:
        cfg["training"]["gpu_id"] = str(args.gpu)
    if args.num_chunks:
        cfg["chunking"]["num_chunks"] = args.num_chunks
    if args.chunk_size:
        cfg["chunking"]["chunk_size"] = args.chunk_size
    if args.shuffle:
        cfg["chunking"]["shuffle_trajectories"] = True
    elif args.no_shuffle:
        cfg["chunking"]["shuffle_trajectories"] = False
    if args.seed is not None:
        cfg["chunking"]["seed"] = args.seed
    if args.beta_start is not None:
        cfg["chunking"]["beta_start"] = args.beta_start
    if args.beta_end is not None:
        cfg["chunking"]["beta_end"] = args.beta_end
    if args.epochs:
        cfg["training"]["epochs_per_iter"] = args.epochs
    if args.trapped_radius:
        cfg["failure_criteria"]["trapped_radius"] = args.trapped_radius
    if args.trapped_steps:
        cfg["failure_criteria"]["trapped_steps"] = args.trapped_steps
    if args.lead_in_steps:
        cfg["failure_criteria"]["lead_in_steps"] = args.lead_in_steps

    # 3. Instantiate and Launch Pipeline
    pipeline = MultiChunkPipeline(cfg)
    pipeline.run_pipeline(
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
