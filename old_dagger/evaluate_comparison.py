"""
Comparative Evaluation: Baseline Controller vs. Adaptive SafeTurn Controller.

Runs side-by-side evaluation on identical failure cases to directly benchmark:
1. Baseline ObjRelLearntController (Fixed lookahead -1, w clamped to 0.1 rad, no in-place turning)
2. Adaptive SafeTurnController (Adaptive lookahead 0.5m-1.8m, hysteresis in-place turning gate, lag flush)

Reports:
- Escape Rate (R_escape): Escaped the initial failure stagnation sphere (>1.0m displacement)
- Task Success Rate (R_task): Reached threshold goal distance (<=1.0m)
- Distance Closed: (Initial Distance - Final Distance)
- Stagnation Events: Episodes where displacement remained < 0.35m over 25 steps
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import quaternion
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[CompareControllers]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from libs.common import utils
from libs.common import utils_sim_traj as ust
from old_dagger.core.failure_loader import FailureLoader, FailureCase, find_scene_glb
from libs.control.objectreact import ObjRelLearntController
from libs.control.safe_controller import SafeTurnController


def run_single_rollout(
    controller_cls,
    controller_name: str,
    case: FailureCase,
    cfg: Dict,
    checkpoint_path: Path,
    max_steps: int = 200,
) -> Dict:
    """Executes a single rollout with the given controller class on a failure case."""
    sim_cfg = cfg["simulation"]
    threshold_dist = sim_cfg["threshold_goal_distance"]
    hm3d_root = REPO_ROOT / cfg["paths"]["hm3d_scenes_path"]

    scene_glb = find_scene_glb(hm3d_root, case.scene_id)
    if scene_glb is None or not scene_glb.exists():
        logger.error(f"Scene GLB not found for {case.scene_id}")
        return None

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
        # Preload controller
        ctrl_cfg_path = REPO_ROOT / cfg["paths"]["controller_config"]
        with open(ctrl_cfg_path, "r") as f:
            ctrl_cfg = yaml.safe_load(f)
        ctrl_cfg["load_run"] = str(checkpoint_path)

        controller = controller_cls(ctrl_cfg, goal_source="gt_topometric")

        # Set exact failure start pose
        start_state = agent.get_state()
        start_state.position = case.failure_position
        q = case.failure_rotation
        start_state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
        agent.set_state(start_state)

        initial_dist = ust.find_shortest_path(sim, case.failure_position, case.goal_position)[0]
        curr_pos = case.failure_position
        positions_history = [curr_pos[[0, 2]]]

        status = "exceeded_steps"
        final_dist = initial_dist
        turn_in_place_steps = 0
        max_displacement_from_start = 0.0

        for step in range(max_steps):
            curr_pos = agent.get_state().position
            disp = float(np.linalg.norm(curr_pos[[0, 2]] - case.failure_position[[0, 2]]))
            if disp > max_displacement_from_start:
                max_displacement_from_start = disp

            euclidean_dist = float(np.linalg.norm(curr_pos - case.goal_position))
            if euclidean_dist <= threshold_dist + 1.5:
                curr_dist = ust.find_shortest_path(sim, curr_pos, case.goal_position)[0]
            else:
                curr_dist = euclidean_dist
            final_dist = curr_dist

            if curr_dist <= threshold_dist:
                status = "success"
                break

            obs = sim.get_sensor_observations()
            display_img, depth, semantic = utils.split_observations(obs)

            # Generate goal mask for policy
            masks = (semantic[None, ...] == np.unique(semantic)[:, None, None])
            pls = np.array([curr_dist] * len(masks))
            control_input = [masks, pls]

            v, w, _ = controller.predict(display_img, control_input)

            if getattr(controller, "is_turning_in_place", False):
                turn_in_place_steps += 1

            agent, sim, _ = utils.apply_velocity(
                vel_control=vel_control,
                agent=agent,
                sim=sim,
                velocity=v,
                steer=-w,
                time_step=sim_cfg["time_delta"],
            )
            positions_history.append(agent.get_state().position[[0, 2]])

        # Stagnation check: did position change by < 0.35m over any 25-step window?
        pts = np.array(positions_history)
        is_stagnant = False
        if len(pts) >= 25:
            displacements_25 = np.linalg.norm(pts[25:] - pts[:-25], axis=1)
            is_stagnant = bool(np.all(displacements_25[-15:] < 0.35))

        # Metrics
        is_task_success = (status == "success")
        is_escape_success = (max_displacement_from_start >= 1.0 and not is_stagnant)
        dist_closed = max(0.0, initial_dist - final_dist)

        return {
            "controller": controller_name,
            "episode_name": case.episode_name,
            "scene_id": case.scene_id,
            "initial_dist": float(initial_dist),
            "final_dist": float(final_dist),
            "dist_closed": float(dist_closed),
            "steps_taken": step + 1,
            "max_displacement": float(max_displacement_from_start),
            "turn_in_place_steps": turn_in_place_steps,
            "is_task_success": is_task_success,
            "is_escape_success": is_escape_success,
            "is_stagnant": is_stagnant,
            "final_status": status,
        }
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description="Side-by-side evaluation of Baseline vs Adaptive Safe Controller")
    parser.add_argument("--config", default="dagger/config.yaml", help="Path to dagger config")
    parser.add_argument("--checkpoint", default="dagger/checkpoints/policy_iter_2.pth", help="Checkpoint")
    parser.add_argument("--case_idx", type=int, default=0, help="Starting test case index")
    parser.add_argument("--num_cases", type=int, default=5, help="Number of test cases to benchmark")
    parser.add_argument("--max_steps", type=int, default=150, help="Max step budget per episode")
    parser.add_argument("--gpu", type=str, default="1", help="CUDA GPU ID")
    parser.add_argument("--output_json", default="dagger/output/controller_comparison.json", help="Output JSON results")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(REPO_ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    loader = FailureLoader(str(REPO_ROOT / args.config))
    all_cases = loader.load_from_csv()
    train_c, val_c, test_cases = loader.split_cases(all_cases)
    selected_cases = test_cases[args.case_idx : args.case_idx + args.num_cases]

    checkpoint_file = REPO_ROOT / args.checkpoint
    if not checkpoint_file.exists():
        checkpoint_file = REPO_ROOT / "dagger/checkpoints/policy_iter_1.pth"

    logger.info(f"Comparing controllers across {len(selected_cases)} cases with {args.max_steps} max steps...")

    baseline_results = []
    safe_results = []

    for i, case in enumerate(selected_cases):
        case_num = args.case_idx + i
        logger.info(f"\n=======================================================")
        logger.info(f"Case {case_num:02d}: {case.episode_name} ({case.scene_id})")
        logger.info(f"=======================================================")

        # 1. Run Baseline Controller
        logger.info(">>> Running Baseline Controller...")
        res_base = run_single_rollout(
            controller_cls=ObjRelLearntController,
            controller_name="Baseline",
            case=case,
            cfg=cfg,
            checkpoint_path=checkpoint_file,
            max_steps=args.max_steps,
        )
        if res_base:
            baseline_results.append(res_base)
            logger.info(
                f"[Baseline Result] Status: {res_base['final_status']} | "
                f"Dist: {res_base['initial_dist']:.2f}m -> {res_base['final_dist']:.2f}m "
                f"(Closed: {res_base['dist_closed']:.2f}m) | "
                f"MaxDisp: {res_base['max_displacement']:.2f}m | Escape: {res_base['is_escape_success']}"
            )

        # 2. Run Adaptive Safe Controller
        logger.info(">>> Running Adaptive SafeTurn Controller...")
        res_safe = run_single_rollout(
            controller_cls=SafeTurnController,
            controller_name="AdaptiveSafeTurn",
            case=case,
            cfg=cfg,
            checkpoint_path=checkpoint_file,
            max_steps=args.max_steps,
        )
        if res_safe:
            safe_results.append(res_safe)
            logger.info(
                f"[SafeTurn Result] Status: {res_safe['final_status']} | "
                f"Dist: {res_safe['initial_dist']:.2f}m -> {res_safe['final_dist']:.2f}m "
                f"(Closed: {res_safe['dist_closed']:.2f}m) | "
                f"MaxDisp: {res_safe['max_displacement']:.2f}m | Escape: {res_safe['is_escape_success']} | "
                f"TurnInPlace Steps: {res_safe['turn_in_place_steps']}/{res_safe['steps_taken']}"
            )

    # Compute aggregate metrics
    def summarize(res_list):
        if not res_list:
            return {}
        n = len(res_list)
        escape_rate = sum(r["is_escape_success"] for r in res_list) / n * 100
        task_rate = sum(r["is_task_success"] for r in res_list) / n * 100
        avg_closed = np.mean([r["dist_closed"] for r in res_list])
        avg_disp = np.mean([r["max_displacement"] for r in res_list])
        stagnant_rate = sum(r["is_stagnant"] for r in res_list) / n * 100
        return {
            "count": n,
            "escape_rate_pct": float(escape_rate),
            "task_success_rate_pct": float(task_rate),
            "avg_distance_closed_m": float(avg_closed),
            "avg_max_displacement_m": float(avg_disp),
            "stagnation_rate_pct": float(stagnant_rate),
        }

    summary_base = summarize(baseline_results)
    summary_safe = summarize(safe_results)

    comparison_data = {
        "summary": {
            "baseline": summary_base,
            "adaptive_safeturn": summary_safe,
        },
        "per_episode": [
            {"baseline": b, "adaptive_safeturn": s}
            for b, s in zip(baseline_results, safe_results)
        ],
    }

    out_json_path = REPO_ROOT / args.output_json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    logger.info(f"\n=======================================================")
    logger.info("FINAL COMPARATIVE BENCHMARK SUMMARY")
    logger.info(f"=======================================================")
    logger.info(f"{'Metric':<28} | {'Baseline':<16} | {'Adaptive SafeTurn':<18}")
    logger.info("-" * 68)
    logger.info(f"{'Escape Rate (R_escape)':<28} | {summary_base.get('escape_rate_pct', 0):>14.1f}% | {summary_safe.get('escape_rate_pct', 0):>16.1f}%")
    logger.info(f"{'Task Success (R_task)':<28} | {summary_base.get('task_success_rate_pct', 0):>14.1f}% | {summary_safe.get('task_success_rate_pct', 0):>16.1f}%")
    logger.info(f"{'Avg Distance Closed':<28} | {summary_base.get('avg_distance_closed_m', 0):>14.2f}m | {summary_safe.get('avg_distance_closed_m', 0):>16.2f}m")
    logger.info(f"{'Avg Max Displacement':<28} | {summary_base.get('avg_max_displacement_m', 0):>14.2f}m | {summary_safe.get('avg_max_displacement_m', 0):>16.2f}m")
    logger.info(f"{'Stagnation Rate':<28} | {summary_base.get('stagnation_rate_pct', 0):>14.1f}% | {summary_safe.get('stagnation_rate_pct', 0):>16.1f}%")
    logger.info("=" * 68)
    logger.info(f"Saved full comparison details to: {out_json_path}")


if __name__ == "__main__":
    main()
