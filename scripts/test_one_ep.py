#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import yaml

REPO_ROOT = Path("/scratch2/akshat.gupta/ak-object-rel-nav")
sys.path.insert(0, str(REPO_ROOT))

from dagger.core.failure_loader import find_scene_glb
from libs.common import utils
from libs.experiments import task_setup

def test():
    # Load baseline config
    with open(REPO_ROOT / "configs/object_react.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg["goal_source"] = "gt_metric"
    cfg["save_vis"] = False
    cfg["plot"] = False
    cfg["env"] = "sim"
    cfg["log_robot"] = True
    cfg["split"] = "train"
    cfg["max_steps"] = 10  # test only 10 steps for speed

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

    print("Preloading models...")
    preload_data = task_setup.preload_models(args)

    ep_path = Path("/scratch/toponavgroup/indoor-topo-loc/datasets/hm3d_navigation/hm3d_iin_train/HZ2iMMBsBQ9_0032299_plant_480_")
    scene_glb = find_scene_glb(REPO_ROOT / "data/hm3d_v0.2", "HZ2iMMBsBQ9")
    print(f"Found scene: {scene_glb}")

    test_output_dir = REPO_ROOT / "out/test_single_ep"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Episode...")
    episode_runner = task_setup.Episode(args, ep_path, str(scene_glb), test_output_dir, preload_data)

    print("Running 10 steps...")
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
    print("Episode closed cleanly!")
    
    # Check outputs
    ep_dir = test_output_dir / f"{ep_path.parts[-1]}_{args.method.lower()}_{args.goal_source}"
    print("Files produced:")
    for f in ep_dir.iterdir():
        print(f"  {f.name} ({f.stat().st_size} bytes)")

if __name__ == "__main__":
    test()
