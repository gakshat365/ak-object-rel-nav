"""
Autonomous Recovery Evaluation Script (Beta = 0).

Evaluates any specified policy checkpoint (or baseline) on recovery episodes
starting from the failure states without any expert intervention.
Computes Recovery Success Rate (R-SR), Recovery SPL, Soft-SPL, and Distance Closed.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[EvalRecovery]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from libs.common import utils
from libs.common import utils_sim_traj as ust
from dagger.core.failure_loader import FailureLoader, FailureCase
from dagger.validate_expert import find_scene_glb


def evaluate_checkpoint_on_cases(
    checkpoint_path: str,
    cases: List[FailureCase],
    cfg: Dict,
    split_name: str = "test",
) -> Dict:
    """
    Evaluates policy checkpoint autonomously on given failure cases.
    """
    sim_cfg = cfg["simulation"]
    threshold_dist = sim_cfg["threshold_goal_distance"]
    max_steps = sim_cfg["max_rollout_steps"]
    hm3d_root = REPO_ROOT / cfg["paths"]["hm3d_scenes_path"]

    # Preload Controller
    from libs.control.objectreact import ObjRelLearntController

    ctrl_cfg_path = REPO_ROOT / cfg["paths"]["controller_config"]
    with open(ctrl_cfg_path, "r") as f:
        ctrl_cfg = yaml.safe_load(f)

    # Point controller to checkpoint
    ctrl_cfg["load_run"] = str(checkpoint_path)
    logger.info(f"Loading controller checkpoint from: {checkpoint_path}")
    controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")

    results = []
    num_success = 0
    spl_list = []
    soft_spl_list = []
    dist_closed_list = []

    for case in tqdm(cases, desc=f"Evaluating Recovery ({split_name})"):
        scene_glb = find_scene_glb(hm3d_root, case.scene_id)
        if scene_glb is None or not scene_glb.exists():
            logger.warning(f"Skipping {case.episode_name}: scene mesh not found")
            continue

        sim, agent, _ = utils.get_sim_agent(
            str(scene_glb),
            updateNavMesh=False,
            width=sim_cfg["width"],
            height=sim_cfg["height"],
            hfov=sim_cfg["hfov"],
            sensor_height=sim_cfg["sensor_height"],
        )

        import habitat_sim
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True

        try:
            # Set agent state to failure pose
            start_state = agent.get_state()
            start_state.position = case.failure_position
            # Quaternion format
            q = case.failure_rotation
            import quaternion
            start_state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
            agent.set_state(start_state)

            initial_dist = ust.find_shortest_path(sim, case.failure_position, case.goal_position)[0]
            controller.reset_params()

            curr_pos = case.failure_position
            positions_history = [curr_pos[[0, 2]]]
            status = "exceeded_steps"
            final_dist = initial_dist

            for step in range(max_steps):
                curr_dist = ust.find_shortest_path(sim, agent.get_state().position, case.goal_position)[0]
                final_dist = curr_dist

                if curr_dist <= threshold_dist:
                    status = "success"
                    break

                obs = sim.get_sensor_observations()
                display_img, depth, semantic = utils.split_observations(obs)

                # Generate goal mask for policy
                # Using geometric goal distance
                masks = (semantic[None, ...] == np.unique(semantic)[:, None, None])
                pls = np.array([curr_dist] * len(masks))
                control_input = [masks, pls]

                v, w, _ = controller.predict(display_img, control_input)

                # Step simulation
                agent, sim, _ = utils.apply_velocity(
                    vel_control=vel_control,
                    agent=agent,
                    sim=sim,
                    velocity=v,
                    steer=-w,
                    time_step=sim_cfg["time_delta"],
                )
                positions_history.append(agent.get_state().position[[0, 2]])

            # Metrics
            is_success = (status == "success")
            if is_success:
                num_success += 1

            path_pts = np.array(positions_history)
            actual_path_length = np.linalg.norm(np.diff(path_pts, axis=0), axis=1).sum()

            spl = 0.0
            if is_success:
                spl = initial_dist / max(initial_dist, actual_path_length)
            spl_list.append(spl)

            soft_spl = max(0.0, 1.0 - (final_dist / max(initial_dist, 1e-4)))
            soft_spl_list.append(soft_spl)

            dist_closed = initial_dist - final_dist
            dist_closed_list.append(dist_closed)

            results.append({
                "episode_name": case.episode_name,
                "scene_id": case.scene_id,
                "status": status,
                "steps": step + 1,
                "initial_distance": initial_dist,
                "final_distance": final_dist,
                "distance_closed": dist_closed,
                "spl": spl,
                "soft_spl": soft_spl,
            })

        finally:
            sim.close()

    total = max(1, len(results))
    sr = (num_success / total) * 100.0
    mean_spl = float(np.mean(spl_list)) * 100.0 if spl_list else 0.0
    mean_soft_spl = float(np.mean(soft_spl_list)) * 100.0 if soft_spl_list else 0.0
    mean_dist_closed = float(np.mean(dist_closed_list)) if dist_closed_list else 0.0

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "total_episodes": total,
        "successful_episodes": num_success,
        "recovery_success_rate": sr,
        "recovery_spl": mean_spl,
        "recovery_soft_spl": mean_soft_spl,
        "mean_distance_closed": mean_dist_closed,
        "results": results,
    }

    logger.info(f"\n================ Recovery Evaluation Summary ({split_name}) ================")
    logger.info(f"Checkpoint:            {checkpoint_path}")
    logger.info(f"Total Episodes:        {total}")
    logger.info(f"Recovery Success Rate: {sr:.2f}% ({num_success}/{total})")
    logger.info(f"Recovery SPL:          {mean_spl:.2f}%")
    logger.info(f"Recovery Soft-SPL:     {mean_soft_spl:.2f}%")
    logger.info(f"Mean Distance Closed:  {mean_dist_closed:.3f} m")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Policy Checkpoint on Failure Recovery")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file")
    parser.add_argument("--config", type=str, default="dagger/config.yaml", help="Path to dagger config")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--out", type=str, default=None, help="Path to save summary json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint = args.checkpoint or cfg["paths"]["base_checkpoint"]

    loader = FailureLoader(args.config)
    all_cases = loader.load_from_failed_cases_dir()
    train_c, val_c, test_c = loader.split_cases(all_cases)

    case_splits = {
        "train": train_c,
        "val": val_c,
        "test": test_c,
        "all": all_cases,
    }

    selected_cases = case_splits[args.split]
    summary = evaluate_checkpoint_on_cases(checkpoint, selected_cases, cfg, split_name=args.split)

    out_path = args.out or (Path(cfg["paths"]["logs_dir"]) / f"eval_recovery_{args.split}.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved evaluation metrics to {out_path}")


if __name__ == "__main__":
    main()
