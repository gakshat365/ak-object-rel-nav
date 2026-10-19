#!/usr/bin/env python3
"""
Benchmark SafeTurnController: Adaptive Lookahead Flag ON vs. Flag OFF.

Evaluates navigation trajectories from dagger/data/initial_data across both controller modes:
  - flag_on:  use_adaptive_lookahead = True  (dynamic lookahead scaling 0.5m - 1.8m)
  - flag_off: use_adaptive_lookahead = False (fixed waypoint lookahead)

Categorizes outcomes into three mutually exclusive categories:
  1. success: reached goal (geodesic/euclidean distance <= 1.0m)
  2. trapped: agent hasn't gone out of a 1.0m radius for the previous 20 steps
  3. ran out of steps: step limit reached without success or trapping

Saves results incrementally to a CSV with columns:
  scene,flag_on,flag_off
"""

import argparse
import csv
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import habitat_sim
import numpy as np
import quaternion
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "libs" / "control" / "object_react" / "train"))

from libs.common import utils
from libs.common import utils_sim_traj as ust
from libs.control.safe_controller import SafeTurnController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[BenchmarkSafeController]")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SafeTurnController Flag ON vs OFF")
    parser.add_argument("--config", type=str, default="dagger/config.yaml", help="Path to dagger config")
    parser.add_argument("--data_dir", type=str, default="dagger/data/initial_data", help="Directory of trajectories")
    parser.add_argument("--output_csv", type=str, default="dagger/benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint (.pth)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum number of episodes to evaluate")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting episode index (for batching)")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode rollout")
    parser.add_argument("--threshold_dist", type=float, default=1.0, help="Goal arrival distance threshold (m)")
    parser.add_argument("--trapped_radius", type=float, default=0.2, help="Radius threshold for trapped condition (m, default 0.2m)")
    parser.add_argument("--trapped_steps", type=int, default=20, help="Window of consecutive steps for trapped condition")
    parser.add_argument("--gpu", type=str, default="1", help="CUDA GPU ID")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing CSV if present")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output CSV instead of resuming")
    parser.add_argument("--mode", type=str, default="flag_off_only", choices=["flag_off_only", "flag_on_off", "waypoints"], help="Benchmark comparison mode (default: flag_off_only)")
    parser.add_argument("--lead_in_steps", type=int, default=50, help="Steps to backtrack when trapped (default 50)")
    parser.add_argument("--trapped_manifest", type=str, default="dagger/data/trapped_cases_manifest.csv", help="Output CSV path for trapped failure recovery cases")
    parser.add_argument("--trapped_dir", type=str, default="dagger/data/trapped_cases", help="Directory path to save trapped case.npz files")
    parser.add_argument("--wp_a", type=int, default=-1, help="Waypoint index for mode A (default -1 = 10 waypoints)")
    parser.add_argument("--wp_b", type=int, default=6, help="Waypoint index for mode B (default 6 = 7 waypoints)")
    parser.add_argument("--label_a", type=str, default=None, help="Label for mode A column")
    parser.add_argument("--label_b", type=str, default=None, help="Label for mode B column")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle/mix trajectories with a reproducible seed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory shuffling")
    return parser.parse_args()


def find_scene_glb(hm3d_root: Path, scene_id: str) -> Optional[Path]:
    """Finds the basis.glb scene mesh for a given scene ID across HM3D scenes directories."""
    clean_id = scene_id.split("-")[-1].strip()
    candidates = [hm3d_root]
    for sub in ["train", "val", "minival"]:
        candidates.append(hm3d_root / sub)
    if hm3d_root.parent.exists():
        for sub in ["train", "val", "minival"]:
            candidates.append(hm3d_root.parent / sub)

    search_dirs = []
    seen = set()
    for c in candidates:
        if c.is_dir() and str(c) not in seen:
            seen.add(str(c))
            search_dirs.append(c)

    for sdir in search_dirs:
        matches = list(sdir.glob(f"*{clean_id}*/*basis.glb"))
        if not matches:
            matches = list(sdir.glob(f"*{clean_id}*/*.glb"))
        if matches:
            return matches[0]

    return None


def set_agent_state(agent: habitat_sim.Agent, raw_state: any) -> None:
    """Safely resets agent state from either habitat_sim.AgentState or dict."""
    if hasattr(raw_state, "position") and hasattr(raw_state, "rotation"):
        agent.set_state(raw_state)
    elif isinstance(raw_state, dict):
        st = habitat_sim.AgentState()
        st.position = np.asarray(raw_state["position"], dtype=np.float32)
        q = raw_state["rotation"]
        if isinstance(q, (list, tuple, np.ndarray)):
            st.rotation = quaternion.quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        else:
            st.rotation = q
        agent.set_state(st)
    else:
        agent.set_state(raw_state)


def get_goal_position(teach_states: np.ndarray, ep_dir: Optional[Path] = None) -> np.ndarray:
    """Extracts target goal coordinate [x, y, z] from goal_position.npy or final teach state."""
    if ep_dir is not None:
        goal_file = ep_dir / "goal_position.npy"
        if goal_file.exists():
            return np.load(goal_file)
    last_st = teach_states[-1]
    if hasattr(last_st, "position"):
        return np.asarray(last_st.position, dtype=np.float32)
    elif isinstance(last_st, dict):
        return np.asarray(last_st["position"], dtype=np.float32)
    return np.asarray(last_st, dtype=np.float32)


@dataclass
class RolloutResult:
    status: str
    total_steps: int
    final_distance: float
    trap_step: Optional[int] = None
    recovery_step: Optional[int] = None
    recovery_position: Optional[np.ndarray] = None
    recovery_rotation: Optional[np.ndarray] = None
    trapped_position: Optional[np.ndarray] = None
    trapped_rotation: Optional[np.ndarray] = None

    def __str__(self):
        return self.status

    def __eq__(self, other):
        if isinstance(other, str):
            return self.status == other
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.status)


def run_single_rollout(
    sim: habitat_sim.Simulator,
    agent: habitat_sim.Agent,
    vel_control: habitat_sim.physics.VelocityControl,
    controller: SafeTurnController,
    start_state: any,
    goal_pos: np.ndarray,
    use_adaptive_lookahead: bool,
    waypoint_index: int = -1,
    max_steps: int = 300,
    threshold_dist: float = 1.0,
    trapped_radius: float = 0.2,
    trapped_steps: int = 20,
    lead_in_steps: int = 50,
    time_delta: float = 0.25,
) -> RolloutResult:
    """
    Executes a single evaluation rollout.

    Returns RolloutResult containing:
      - status: 'success', 'trapped', 'ran out of steps'
      - trap_step: Step where trapped condition occurred
      - recovery_step: max(0, trap_step - lead_in_steps)
      - recovery_position: [x, y, z] agent pose at recovery_step
      - recovery_rotation: [w, x, y, z] quaternion at recovery_step
    """
    # 1. Reset agent to start state
    set_agent_state(agent, start_state)

    # 2. Reset controller state and set adaptive lookahead flag & waypoint index
    controller.reset_params()
    controller.is_turning_in_place = False
    controller.turn_in_place_steps = 0
    controller.forward_cooldown_remaining = 0
    controller.locked_turn_sign = 0.0
    controller.use_adaptive_lookahead = use_adaptive_lookahead
    controller.waypoint_index = waypoint_index

    # Sanity guard: prevent mathematically impossible trapped thresholds (speed <= 0.05m/step)
    max_possible_disp = trapped_steps * 0.05
    effective_trapped_radius = trapped_radius
    if effective_trapped_radius >= max_possible_disp:
        logger.warning(
            f"trapped_radius ({trapped_radius}m) >= max possible displacement over {trapped_steps} steps "
            f"({max_possible_disp:.2f}m). Clamping to 0.2m to prevent false stagnation traps."
        )
        effective_trapped_radius = 0.2

    # 3. State & Position history for trapped condition & recovery extraction
    init_state = agent.get_state()
    state_history: List[habitat_sim.AgentState] = [init_state]
    positions: List[np.ndarray] = [np.asarray(init_state.position, dtype=np.float32)]

    status = "ran out of steps"
    trap_step = None
    dist_to_goal = float(np.linalg.norm(positions[0] - goal_pos))

    for step in range(max_steps):
        curr_pos = np.asarray(agent.get_state().position, dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(curr_pos - goal_pos))

        # Check success condition
        if dist_to_goal <= threshold_dist:
            status = "success"
            break

        # Check trapped/stagnation condition: hasn't moved more than effective_trapped_radius over trapped_steps
        if len(positions) >= trapped_steps:
            recent_pts = np.array(positions[-trapped_steps:])
            max_disp = float(np.max(np.linalg.norm(recent_pts[:, [0, 2]] - recent_pts[0, [0, 2]], axis=-1)))
            if max_disp <= effective_trapped_radius:
                status = "trapped"
                trap_step = step
                break

        # Observations and controller prediction
        obs = sim.get_sensor_observations()
        display_img, depth, semantic = utils.split_observations(obs)

        # Compute ground-truth geodesic pathlengths for semantic objects to provide proper goal guidance
        try:
            _, plsDict, _ = ust.get_pathlength_GT(
                sim, agent, depth, semantic, goal_pos, None
            )
            if plsDict is not None and len(plsDict) > 0:
                instaIds, pls = list(zip(*plsDict.items()))
                masks = semantic[None, ...] == np.array(instaIds)[:, None, None]
                control_input = [masks, np.array(pls, dtype=np.float32)]
            else:
                unique_classes = np.unique(semantic)
                masks = semantic[None, ...] == unique_classes[:, None, None]
                pls = np.linspace(dist_to_goal, dist_to_goal + 4.0, len(unique_classes), dtype=np.float32)
                control_input = [masks, pls]
        except Exception:
            unique_classes = np.unique(semantic)
            masks = semantic[None, ...] == unique_classes[:, None, None]
            pls = np.linspace(dist_to_goal, dist_to_goal + 4.0, len(unique_classes), dtype=np.float32)
            control_input = [masks, pls]

        try:
            v, w, _ = controller.predict(display_img, control_input)
        except Exception as e:
            logger.warning(f"Controller predict fallback at step {step}: {e}")
            v, w = 0.05, 0.0

        # Apply velocity command (Habitat steer sign convention: -w)
        agent, sim, _ = utils.apply_velocity(
            vel_control=vel_control,
            agent=agent,
            sim=sim,
            velocity=v,
            steer=-w,
            time_step=time_delta,
        )

        st = agent.get_state()
        state_history.append(st)
        positions.append(np.asarray(st.position, dtype=np.float32))

    # Final check if loop exited at max_steps
    if status == "ran out of steps" and len(positions) >= trapped_steps:
        recent_pts = np.array(positions[-trapped_steps:])
        max_disp = float(np.max(np.linalg.norm(recent_pts[:, [0, 2]] - recent_pts[0, [0, 2]], axis=-1)))
        if max_disp <= effective_trapped_radius:
            status = "trapped"
            trap_step = len(state_history) - 1

    # Extract 50-step-back recovery state if trapped
    recovery_step = None
    recovery_pos = None
    recovery_rot_quat = None
    trapped_pos = None
    trapped_rot_quat = None

    def quat_to_array(q):
        if hasattr(q, "w"):
            return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        elif hasattr(q, "real"):
            return np.array([q.real, q.imag[0], q.imag[1], q.imag[2]], dtype=np.float32)
        return np.asarray(q, dtype=np.float32)

    if status == "trapped":
        if trap_step is None:
            trap_step = len(state_history) - 1
        # Go 50 steps behind, or maximum steps possible (clamped to 0)
        recovery_step = max(0, trap_step - lead_in_steps)
        rec_st = state_history[recovery_step]
        recovery_pos = np.asarray(rec_st.position, dtype=np.float32)
        recovery_rot_quat = quat_to_array(rec_st.rotation)

        trap_st = state_history[trap_step]
        trapped_pos = np.asarray(trap_st.position, dtype=np.float32)
        trapped_rot_quat = quat_to_array(trap_st.rotation)

    return RolloutResult(
        status=status,
        total_steps=len(state_history) - 1,
        final_distance=dist_to_goal,
        trap_step=trap_step,
        recovery_step=recovery_step,
        recovery_position=recovery_pos,
        recovery_rotation=recovery_rot_quat,
        trapped_position=trapped_pos,
        trapped_rotation=trapped_rot_quat,
    )


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg_path = REPO_ROOT / args.config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_file = args.checkpoint or cfg["paths"].get("model_root", "dagger/model") + "/baseline_model/latest.pth"
    if not Path(checkpoint_file).is_absolute():
        checkpoint_file = str(REPO_ROOT / checkpoint_file)

    ctrl_cfg_path = REPO_ROOT / cfg["paths"]["controller_config"]
    with open(ctrl_cfg_path, "r") as f:
        ctrl_cfg = yaml.safe_load(f)
    ctrl_cfg["load_run"] = checkpoint_file

    logger.info("=================================================================")
    logger.info("BENCHMARK SAFETURN CONTROLLER: FLAG ON vs FLAG OFF")
    logger.info("=================================================================")
    logger.info(f"Model Checkpoint:  {ctrl_cfg['load_run']}")
    logger.info(f"Trajectories Dir:  {args.data_dir}")
    logger.info(f"Output CSV:        {args.output_csv}")
    logger.info(f"Max Steps:         {args.max_steps}")
    logger.info(f"Goal Threshold:    {args.threshold_dist} m")
    logger.info(f"Trapped Condition: <= {args.trapped_radius} m for {args.trapped_steps} consecutive steps")
    logger.info(f"GPU ID:            {args.gpu}")
    logger.info("=================================================================")

    # Initialize Controller once in memory
    logger.info("Initializing SafeTurnController...")
    controller = SafeTurnController(ctrl_cfg, goal_source="gt_topometric")
    logger.info("Controller initialized successfully.")

    # Discover trajectory episodes
    data_path = REPO_ROOT / args.data_dir
    if not data_path.exists():
        data_path = REPO_ROOT / "data/initial_data"

    all_episodes = sorted([d for d in data_path.iterdir() if d.is_dir()])
    total_found = len(all_episodes)
    logger.info(f"Discovered {total_found} total trajectory folders in {data_path}.")

    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(all_episodes)
        logger.info(f"Shuffled/mixed all {total_found} trajectories across scenes with seed {args.seed}.")

    if args.start_idx > 0:
        all_episodes = all_episodes[args.start_idx:]
    if args.max_episodes is not None:
        all_episodes = all_episodes[:args.max_episodes]

    logger.info(f"Queued {len(all_episodes)} episodes for this benchmark run.")

    # Prepare CSV output and check for resumption
    output_csv_path = REPO_ROOT / args.output_csv
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    evaluated_episodes = set()
    if not args.overwrite and args.resume and output_csv_path.exists():
        with open(output_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "scene" in row:
                    evaluated_episodes.add(row["scene"])
        logger.info(f"Resuming: found {len(evaluated_episodes)} existing results in {output_csv_path.name}.")

    label_a = args.label_a or ("flag_on" if args.mode == "flag_on_off" else "wp_10")
    label_b = args.label_b or ("flag_off" if args.mode == "flag_on_off" else "wp_7")

    write_header = args.overwrite or not output_csv_path.exists() or os.path.getsize(output_csv_path) == 0
    mode = "w" if args.overwrite else "a"
    csv_file = open(output_csv_path, mode, newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        if args.mode == "flag_off_only":
            csv_writer.writerow(["scene", "status", "trap_step", "recovery_step", "final_dist"])
        else:
            csv_writer.writerow(["scene", label_a, label_b])
        csv_file.flush()

    # Trapped cases export preparation
    trapped_manifest_path = REPO_ROOT / args.trapped_manifest
    trapped_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    trapped_dir_path = REPO_ROOT / args.trapped_dir
    trapped_dir_path.mkdir(parents=True, exist_ok=True)

    trapped_write_header = (
        args.overwrite
        or not trapped_manifest_path.exists()
        or os.path.getsize(trapped_manifest_path) == 0
    )
    trapped_mode = "w" if args.overwrite else "a"
    trapped_csv_file = open(trapped_manifest_path, trapped_mode, newline="")
    trapped_writer = csv.writer(trapped_csv_file)
    if trapped_write_header:
        trapped_writer.writerow([
            "episode_name",
            "scene_id",
            "trap_step",
            "recovery_step",
            "rec_pos_x",
            "rec_pos_y",
            "rec_pos_z",
            "rot_w",
            "rot_x",
            "rot_y",
            "rot_z",
            "goal_x",
            "goal_y",
            "goal_z",
            "initial_dist",
            "final_dist",
        ])
        trapped_csv_file.flush()

    # Group episodes by scene_id to minimize Habitat scene mesh reload overhead
    episodes_by_scene: Dict[str, List[Path]] = {}
    for ep_dir in all_episodes:
        if ep_dir.name in evaluated_episodes:
            continue
        scene_id = ep_dir.name.split("_")[0]
        episodes_by_scene.setdefault(scene_id, []).append(ep_dir)

    remaining_count = sum(len(eps) for eps in episodes_by_scene.values())
    logger.info(f"Episodes remaining to evaluate: {remaining_count}")

    hm3d_root = REPO_ROOT / cfg["paths"].get("hm3d_root", "data/hm3d_v0.2")
    scene_glb_cache: Dict[str, Optional[Path]] = {}

    if args.mode == "flag_off_only":
        stats = {"success": 0, "trapped": 0, "ran out of steps": 0}
    else:
        stats = {
            label_a: {"success": 0, "trapped": 0, "ran out of steps": 0},
            label_b: {"success": 0, "trapped": 0, "ran out of steps": 0},
        }
    completed = 0
    trapped_saved_count = 0

    try:
        for scene_id, scene_eps in episodes_by_scene.items():
            if scene_id not in scene_glb_cache:
                scene_glb_cache[scene_id] = find_scene_glb(hm3d_root, scene_id)

            scene_glb = scene_glb_cache[scene_id]
            if scene_glb is None or not scene_glb.exists():
                logger.warning(f"Scene GLB not found for scene_id='{scene_id}'. Skipping {len(scene_eps)} episodes.")
                continue

            # Load Habitat simulator once for all episodes in this scene
            sim, agent, _ = utils.get_sim_agent(
                str(scene_glb),
                updateNavMesh=False,
                width=320,
                height=240,
                hfov=120,
                sensor_height=0.4,
            )

            vel_control = habitat_sim.physics.VelocityControl()
            vel_control.controlling_lin_vel = True
            vel_control.lin_vel_is_local = True
            vel_control.controlling_ang_vel = True
            vel_control.ang_vel_is_local = True

            try:
                for ep_dir in scene_eps:
                    ep_name = ep_dir.name
                    agent_states_file = ep_dir / "agent_states.npy"
                    if not agent_states_file.exists():
                        logger.warning(f"agent_states.npy missing in {ep_dir}. Skipping.")
                        continue

                    teach_states = np.load(agent_states_file, allow_pickle=True)
                    start_state = teach_states[0]
                    goal_pos = get_goal_position(teach_states, ep_dir)

                    if args.mode == "flag_off_only":
                        # Run Flag OFF (adaptive lookahead disabled, normal waypoint thing)
                        res = run_single_rollout(
                            sim=sim,
                            agent=agent,
                            vel_control=vel_control,
                            controller=controller,
                            start_state=start_state,
                            goal_pos=goal_pos,
                            use_adaptive_lookahead=False,
                            waypoint_index=-1,
                            max_steps=args.max_steps,
                            threshold_dist=args.threshold_dist,
                            trapped_radius=args.trapped_radius,
                            trapped_steps=args.trapped_steps,
                            lead_in_steps=args.lead_in_steps,
                            time_delta=cfg["rollout"].get("time_delta", 0.25),
                        )

                        stats[res.status] += 1
                        completed += 1

                        if res.status == "trapped":
                            trapped_saved_count += 1
                            start_pos_3d = np.asarray(start_state["position"] if isinstance(start_state, dict) else start_state.position, dtype=np.float32)
                            dist_init = float(np.linalg.norm(start_pos_3d - goal_pos))

                            # 1. Append to trapped manifest
                            trapped_writer.writerow([
                                ep_name,
                                scene_id,
                                res.trap_step,
                                res.recovery_step,
                                f"{res.recovery_position[0]:.4f}",
                                f"{res.recovery_position[1]:.4f}",
                                f"{res.recovery_position[2]:.4f}",
                                f"{res.recovery_rotation[0]:.4f}",
                                f"{res.recovery_rotation[1]:.4f}",
                                f"{res.recovery_rotation[2]:.4f}",
                                f"{res.recovery_rotation[3]:.4f}",
                                f"{goal_pos[0]:.4f}",
                                f"{goal_pos[1]:.4f}",
                                f"{goal_pos[2]:.4f}",
                                f"{dist_init:.4f}",
                                f"{res.final_distance:.4f}",
                            ])
                            trapped_csv_file.flush()

                            # 2. Save individual case.npz and metadata.json
                            ep_case_dir = trapped_dir_path / ep_name
                            ep_case_dir.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                ep_case_dir / "case.npz",
                                episode_name=ep_name,
                                scene_id=scene_id,
                                trap_step=res.trap_step,
                                recovery_step=res.recovery_step,
                                failure_position=res.recovery_position,
                                failure_rotation=res.recovery_rotation,
                                trapped_position=res.trapped_position,
                                goal_position=goal_pos,
                                distance_to_goal=np.array([res.final_distance], dtype=np.float32),
                                initial_goal_dist=float(dist_init),
                            )
                            with open(ep_case_dir / "metadata.json", "w") as mf:
                                json.dump({
                                    "episode_name": ep_name,
                                    "scene_id": scene_id,
                                    "trap_step": int(res.trap_step),
                                    "recovery_step": int(res.recovery_step),
                                    "failure_position": res.recovery_position.tolist(),
                                    "failure_rotation": res.recovery_rotation.tolist(),
                                    "trapped_position": res.trapped_position.tolist(),
                                    "goal_position": goal_pos.tolist(),
                                    "initial_goal_dist": float(dist_init),
                                    "final_distance": float(res.final_distance),
                                    "lead_in_steps": int(args.lead_in_steps),
                                }, mf, indent=2)

                            csv_writer.writerow([ep_name, res.status, res.trap_step, res.recovery_step, f"{res.final_distance:.3f}"])
                        else:
                            csv_writer.writerow([ep_name, res.status, "", "", f"{res.final_distance:.3f}"])

                        csv_file.flush()

                        logger.info(
                            f"[{completed}/{remaining_count}] {ep_name} -> status: '{res.status}' "
                            f"(steps={res.total_steps}, final_dist={res.final_distance:.2f}m)"
                            + (f" [TRAPPED at step {res.trap_step} -> 50-step recovery at step {res.recovery_step}]" if res.status == 'trapped' else "")
                        )

                    else:
                        # 1. Run Mode A
                        res_a = run_single_rollout(
                            sim=sim,
                            agent=agent,
                            vel_control=vel_control,
                            controller=controller,
                            start_state=start_state,
                            goal_pos=goal_pos,
                            use_adaptive_lookahead=(args.mode == "flag_on_off"),
                            waypoint_index=args.wp_a if args.mode == "waypoints" else -1,
                            max_steps=args.max_steps,
                            threshold_dist=args.threshold_dist,
                            trapped_radius=args.trapped_radius,
                            trapped_steps=args.trapped_steps,
                            lead_in_steps=args.lead_in_steps,
                            time_delta=cfg["rollout"].get("time_delta", 0.25),
                        )

                        # 2. Run Mode B
                        res_b = run_single_rollout(
                            sim=sim,
                            agent=agent,
                            vel_control=vel_control,
                            controller=controller,
                            start_state=start_state,
                            goal_pos=goal_pos,
                            use_adaptive_lookahead=False,
                            waypoint_index=args.wp_b if args.mode == "waypoints" else -1,
                            max_steps=args.max_steps,
                            threshold_dist=args.threshold_dist,
                            trapped_radius=args.trapped_radius,
                            trapped_steps=args.trapped_steps,
                            lead_in_steps=args.lead_in_steps,
                            time_delta=cfg["rollout"].get("time_delta", 0.25),
                        )

                        # Update counts
                        stats[label_a][res_a.status] += 1
                        stats[label_b][res_b.status] += 1
                        completed += 1

                        # Write row to CSV and flush immediately
                        csv_writer.writerow([ep_name, res_a.status, res_b.status])
                        csv_file.flush()

                        logger.info(
                            f"[{completed}/{remaining_count}] {ep_name} -> "
                            f"{label_a}: '{res_a.status}' | {label_b}: '{res_b.status}'"
                        )

            finally:
                sim.close()

    finally:
        csv_file.close()
        trapped_csv_file.close()

    logger.info("=================================================================")
    logger.info("BENCHMARK COMPLETED")
    logger.info("=================================================================")
    logger.info(f"Results successfully saved to: {output_csv_path}")
    logger.info(f"Total Evaluated in this run: {completed}")
    if args.mode == "flag_off_only":
        logger.info("Mode: Flag OFF (Adaptive Lookahead Disabled - Fixed Waypoint Lookahead):")
        logger.info(f"  Success:                {stats['success']}")
        logger.info(f"  Trapped:                {stats['trapped']}")
        logger.info(f"  Ran out of steps:       {stats['ran out of steps']}")
        logger.info(f"  Trapped cases exported: {trapped_saved_count}")
        logger.info(f"  Trapped manifest saved: {trapped_manifest_path}")
        logger.info(f"  Trapped npz dir saved:  {trapped_dir_path}")
    else:
        logger.info(f"{label_a}:")
        logger.info(f"  Success:           {stats[label_a]['success']}")
        logger.info(f"  Trapped:           {stats[label_a]['trapped']}")
        logger.info(f"  Ran out of steps:  {stats[label_a]['ran out of steps']}")
        logger.info(f"{label_b}:")
        logger.info(f"  Success:           {stats[label_b]['success']}")
        logger.info(f"  Trapped:           {stats[label_b]['trapped']}")
        logger.info(f"  Ran out of steps:  {stats[label_b]['ran out of steps']}")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
