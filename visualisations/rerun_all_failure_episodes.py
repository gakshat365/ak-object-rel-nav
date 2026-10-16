#!/usr/bin/env python3
"""
ObjectReact Failure Episodes Benchmark Runner.

Runs the benchmark across all failure episodes with full recordings:
- repeat.mp4 (agent view + top-down BEV map)
- results.csv (step telemetry)
- results_dict.npz (full trajectories & states)
- metadata.txt (final metric summary)

Safe sequential execution prevents EGL/OpenGL driver deadlocks on GPU.
Automatically resumes by skipping episodes that already have valid recordings.

Usage:
  pixi run python visualisations/rerun_all_failure_episodes.py --exp_name nav-15-oct-26
  pixi run python visualisations/rerun_all_failure_episodes.py --force-all
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import yaml
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dagger.core.failure_loader import find_scene_glb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[RerunFailureEpisodes]")


def run_benchmark(
    mapping_path: str = "visualisations/episode_mapping.json",
    exp_name: str = "nav-15-oct-26",
    output_dir: Optional[str] = None,
    force_all: bool = False,
    max_episodes: int = 0,
    max_steps: Optional[int] = None,
):
    mapping_file = REPO_ROOT / mapping_path
    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    logger.info(f"Loaded {len(mapping)} episodes from {mapping_path}")

    train_base = Path("/scratch/toponavgroup/indoor-topo-loc/datasets/hm3d_navigation/hm3d_iin_train")
    val_base = REPO_ROOT / "data/hm3d_iin_val"
    hm3d_root = REPO_ROOT / "data/hm3d_v0.2"

    if output_dir:
        out_base = Path(output_dir)
        if not out_base.is_absolute():
            out_base = (REPO_ROOT / out_base).resolve()
    else:
        out_base = (REPO_ROOT / "out/hm3d_objectreact_eval/episodes").resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    # Load baseline evaluation configuration
    with open(REPO_ROOT / "configs/object_react.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg["exp_name"] = exp_name
    cfg["goal_source"] = "gt_metric"
    cfg["save_vis"] = False  # Avoids slow matplotlib jpg saving; repeat.mp4 generated via log_robot
    cfg["plot"] = False
    cfg["env"] = "sim"
    cfg["log_robot"] = True
    cfg["split"] = "train"
    if max_steps is not None:
        cfg["max_steps"] = max_steps

    class Args:
        pass

    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    setattr(
        args,
        "traversable_class_names",
        [
            "floor", "flooring", "floor mat", "floor vent", "carpet",
            "mat", "rug", "doormat", "shower floor", "pavement", "ground", "tiles"
        ],
    )

    from libs.common import utils
    from libs.experiments import task_setup

    logger.info("Preloading ObjectReact policy model..yes.")
    preload_data = task_setup.preload_models(args)

    # Sort episodes by scene to optimize Habitat asset caching
    sorted_episodes = sorted(mapping, key=lambda x: (x["scene"], x["code"]))
    if max_episodes > 0:
        sorted_episodes = sorted_episodes[:max_episodes]

    total_eps = len(sorted_episodes)
    logger.info(f"Targeting {total_eps} episodes (force_all={force_all}). Output: {out_base}")

    from tqdm import tqdm
    pbar = tqdm(sorted_episodes, desc="Evaluating Episodes", unit="ep")

    completed_count = 0
    videos_created = 0
    skipped_count = 0
    errors_count = 0

    code_to_mapping = {m["code"]: m for m in mapping}

    for item in pbar:
        code = item["code"]
        ep_name = item["episode_name"]
        folder_name = item["folder_name"]
        scene_str = item["scene"]
        scene_id = scene_str.split("-")[-1]

        ep_target_dir = out_base / folder_name
        video_file = ep_target_dir / "repeat.mp4"

        # Check if already completed and valid
        if not force_all and video_file.exists() and video_file.stat().st_size > 10000:
            skipped_count += 1
            if code in code_to_mapping:
                code_to_mapping[code]["video_available"] = True
                code_to_mapping[code]["video_path"] = str(video_file.resolve().relative_to(REPO_ROOT.resolve()))
            pbar.set_postfix({"done": completed_count, "skipped": skipped_count, "errors": errors_count})
            continue

        # Locate episode demonstration directory
        ep_path = None
        for base in [train_base, val_base]:
            cand = base / f"{ep_name}_"
            if cand.exists():
                ep_path = cand
                break
            cand2 = base / ep_name
            if cand2.exists():
                ep_path = cand2
                break

        if ep_path is None:
            logger.warning(f"Episode directory not found for #{code} {ep_name}")
            errors_count += 1
            continue

        # Locate scene mesh across train and val
        scene_glb = find_scene_glb(hm3d_root, scene_id)
        if scene_glb is None or not scene_glb.exists():
            logger.warning(f"Scene mesh not found for #{code} {scene_id}")
            errors_count += 1
            continue

        episode_runner = None
        step = 0
        try:
            episode_runner = task_setup.Episode(
                args, ep_path, str(scene_glb), out_base, preload_data
            )

            for step in range(args.max_steps):
                if episode_runner.is_done():
                    break

                obs = episode_runner.sim.get_sensor_observations()
                display_img, depth, semantic = utils.split_observations(obs)
                episode_runner.get_goal(display_img, depth, semantic)
                episode_runner.get_control_signal(step, display_img, depth)
                episode_runner.execute_action()
                episode_runner.log_results(step)

            episode_runner.close(step)
            completed_count += 1

            has_video = video_file.exists() and video_file.stat().st_size > 1000
            if has_video:
                videos_created += 1
                if code in code_to_mapping:
                    code_to_mapping[code]["video_available"] = True
                    code_to_mapping[code]["video_path"] = str(video_file.resolve().relative_to(REPO_ROOT.resolve()))
                    code_to_mapping[code]["status"] = episode_runner.success_status
                    code_to_mapping[code]["steps"] = str(step + 1)
                    code_to_mapping[code]["final_distance"] = float(episode_runner.distance_to_goal)

        except Exception as e:
            logger.error(f"Error on #{code} ({ep_name}): {e}")
            errors_count += 1
            if episode_runner and hasattr(episode_runner, "sim") and episode_runner.sim:
                try:
                    episode_runner.sim.close()
                except Exception:
                    pass
        finally:
            if episode_runner and hasattr(episode_runner, "sim") and episode_runner.sim:
                try:
                    episode_runner.sim.close()
                except Exception:
                    pass

        pbar.set_postfix({"done": completed_count, "skipped": skipped_count, "errors": errors_count})

        # Periodically persist mapping every 10 completed episodes
        if completed_count > 0 and completed_count % 10 == 0:
            with open(mapping_file, "w") as f:
                json.dump(mapping, f, indent=2)

    pbar.close()

    # Final mapping file update
    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)

    csv_path = Path(mapping_path).with_suffix(".csv")
    fieldnames = [
        "code", "episode_name", "folder_name", "scene", "target",
        "status", "steps", "final_distance", "video_available", "video_path"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapping:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info("=" * 65)
    logger.info(f"Done! Evaluated: {completed_count}, Skipped (already done): {skipped_count}, Errors: {errors_count}")
    logger.info(f"Updated mapping files saved: {mapping_path} and {csv_path}")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliable Benchmark Evaluation for Failure Episodes")
    parser.add_argument("--exp_name", type=str, default="nav-15-oct-26", help="Experiment name")
    parser.add_argument("--mapping", type=str, default="visualisations/episode_mapping.json")
    parser.add_argument("--output_dir", type=str, default="out/hm3d_objectreact_eval/episodes")
    parser.add_argument("--force-all", action="store_true", help="Force rerun all episodes even if video exists")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 to run all episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="Max step budget per episode (default: 300)")
    args = parser.parse_args()

    run_benchmark(
        mapping_path=args.mapping,
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        force_all=args.force_all,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
    )
