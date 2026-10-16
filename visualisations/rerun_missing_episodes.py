#!/usr/bin/env python3
"""
Runs ObjectReact benchmark evaluation on a specified subset of episodes with save_vis enabled.
Generates repeat.mp4 (agent view + topdown view) for the specified episodes.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import yaml
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[RerunMissingEpisodes]")


def rerun_episodes(
    mapping_path: str = "visualisations/episode_mapping.json",
    output_dir: str = "out/hm3d_objectreact_eval/episodes",
    max_episodes: int = 0,
):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    missing = [m for m in mapping if not m["video_available"]]
    logger.info(f"Total episodes lacking video: {len(missing)}")

    if max_episodes > 0:
        missing = missing[:max_episodes]
        logger.info(f"Limiting execution to {max_episodes} episodes")

    train_base = Path("/scratch/toponavgroup/indoor-topo-loc/datasets/hm3d_navigation/hm3d_iin_train")
    scenes_root = REPO_ROOT / "data/hm3d_v0.2"
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Load baseline config
    with open(REPO_ROOT / "configs/object_react.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Match exact evaluation benchmark settings (from out/hm3d_objectreact_eval/summary.json)
    cfg["goal_source"] = "gt_metric"
    cfg["save_vis"] = True
    cfg["env"] = "sim"
    cfg["log_robot"] = True
    cfg["split"] = "train"

    # Convert to namespace
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

    logger.info("Preloading models...")
    preload_data = task_setup.preload_models(args)

    for i, m in enumerate(missing, 1):
        code = m["code"]
        ep_name = m["episode_name"]
        folder_name = m["folder_name"]
        scene_str = m["scene"]
        scene_clean = scene_str.split("-")[-1]

        ep_path = train_base / f"{ep_name}_"
        if not ep_path.exists():
            logger.warning(f"[{i}/{len(missing)}] Episode dir not found: {ep_path}")
            continue

        scene_matches = list(scenes_root.glob(f"*{scene_clean}*/*basis.glb")) + list(scenes_root.glob(f"*/*{scene_clean}*/*basis.glb"))
        if not scene_matches:
            logger.warning(f"[{i}/{len(missing)}] Scene mesh not found for {scene_clean}")
            continue
        scene_glb = str(scene_matches[0])

        ep_results_dir = output_base / folder_name
        ep_results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{i}/{len(missing)}] Running episode #{code}: {ep_name} in {scene_clean}...")

        episode_runner = None
        try:
            episode_runner = task_setup.Episode(
                args, ep_path, scene_glb, ep_results_dir, preload_data
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
            logger.info(f"[{i}/{len(missing)}] Completed #{code} -> {ep_results_dir / 'repeat.mp4'}")

            # Update mapping file
            m["video_available"] = (ep_results_dir / "repeat.mp4").exists()
            if m["video_available"]:
                m["video_path"] = str((ep_results_dir / "repeat.mp4").relative_to(REPO_ROOT))

        except Exception as e:
            logger.error(f"Error evaluating #{code} {ep_name}: {e}")
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

    # Save updated mapping
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="visualisations/episode_mapping.json")
    parser.add_argument("--output_dir", default="out/hm3d_objectreact_eval/episodes")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 to run all missing")
    args = parser.parse_args()

    rerun_episodes(args.mapping, args.output_dir, args.max_episodes)
