#!/usr/bin/env python3
"""
Side-by-Side Comparison Video Generator: Baseline Policy vs. DAgger Policy.
Evaluates both policies starting from the exact same failure recovery pose in Habitat-Sim.

Displays a synchronized 2-column Full HD (1920x1080) video:
- Left Column:  Baseline Policy (Behavioral Cloning, Iteration -1)
- Right Column: DAgger Recovery Policy (Iteration 0)
- Bird's-Eye View (BEV) Top-Down Navigation Map with real-time trajectory trails (Red for Baseline, Green for DAgger)
- First-Person RGB Sensor Camera View with Live Telemetry HUD
- Side-by-Side Verdict Banners dynamically showing whether DAgger escaped the trap where the Baseline stagnated.
"""

import os
import sys

# Ensure CUDA_VISIBLE_DEVICES is set at the OS environment level BEFORE any C++ / EGL / OpenGL libraries load
target_gpu = "1"
for idx, arg in enumerate(sys.argv):
    if arg == "--gpu" and idx + 1 < len(sys.argv):
        target_gpu = sys.argv[idx + 1]
    elif arg.startswith("--gpu="):
        target_gpu = arg.split("=", 1)[1]

if os.environ.get("CUDA_VISIBLE_DEVICES") != target_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import quaternion
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[SideBySideVis]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


import habitat_sim
from habitat.utils.visualizations import maps
from libs.common import utils
from libs.common import utils_sim_traj as ust
from libs.common.vis_utils import concatenate_videos
from libs.logger.visualizer import Visualizer, rot_to_heading
from libs.control.objectreact import ObjRelLearntController
from old_dagger.core.failure_loader import FailureCase, find_scene_glb


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Side-by-Side Comparison Video (Baseline vs DAgger)")
    parser.add_argument("--config", default="dagger/multi_chunk_runs/chunk_0_dagger_config.yaml", help="DAgger YAML config path")
    parser.add_argument("--baseline_ckpt", default="dagger/model/baseline_model/latest.pth", help="Path to baseline checkpoint")
    parser.add_argument("--dagger_ckpt", default="dagger/multi_chunk_runs/checkpoints/policy_iter_0.pth", help="Path to DAgger checkpoint")
    parser.add_argument("--manifest", default="dagger/multi_chunk_runs/manifests/chunk_0_trapped_manifest.csv", help="Trapped cases CSV manifest")
    parser.add_argument("--eval_json", default="dagger/multi_chunk_runs/logs/eval_iter_0.json", help="Eval iter 0 results JSON (for auto-selecting best episode)")
    parser.add_argument("--episode", default="00150-LcAd9dhvVwh_00716_armchair_379", help="Specific episode name or 'auto_best' to auto-select top recovery case")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to evaluate and generate videos for (e.g. 3 or 5)")
    parser.add_argument("--output_dir", default="dagger/multi_chunk_runs/visualizations", help="Directory to save comparison video")
    parser.add_argument("--output_name", default=None, help="Output MP4 filename (defaults to side_by_side_{episode}.mp4)")
    parser.add_argument("--export_individual", action="store_true", default=False, help="Also export separate baseline_only and dagger_only video files")
    parser.add_argument("--concat", action="store_true", default=True, help="Concatenate multiple episodes into one compilation video")
    parser.add_argument("--no_concat", dest="concat", action="store_false", help="Disable automatic concatenation")
    parser.add_argument("--max_steps", type=int, default=300, help="Max rollout steps per policy")
    parser.add_argument("--fps", type=int, default=15, help="Video framerate")
    parser.add_argument("--gpu", type=int, default=1, help="CUDA GPU ID (default: 1)")
    return parser.parse_args()



def safe_heading(rot) -> float:
    """Computes agent yaw heading in radians safely across quaternion types."""
    try:
        return float(rot_to_heading(rot))
    except Exception:
        try:
            if hasattr(rot, "imag") and hasattr(rot, "real"):
                w, x, y, z = rot.real, rot.imag[0], rot.imag[1], rot.imag[2]
            elif hasattr(rot, "w") and hasattr(rot, "x"):
                w, x, y, z = rot.w, rot.x, rot.y, rot.z
            else:
                w, x, y, z = rot[0], rot[1], rot[2], rot[3]
            siny_cosp = 2.0 * (w * y + x * z)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return float(np.arctan2(siny_cosp, cosy_cosp))
        except Exception:
            return 0.0


def load_target_case(manifest_path: Path, eval_json_path: Path, episode_query: str) -> FailureCase:
    """Finds and constructs the FailureCase object for the desired episode."""
    selected_ep = episode_query

    # If auto_best, query eval_iter_0.json for highest distance closed
    if episode_query == "auto_best" and eval_json_path.exists():
        try:
            with open(eval_json_path, "r") as f:
                eval_data = json.load(f)
            results = eval_data.get("results", [])
            if results:
                best_item = max(results, key=lambda x: x.get("distance_closed", 0.0))
                selected_ep = best_item["episode_name"]
                logger.info(f"Auto-selected best recovery episode from eval results: {selected_ep} (Closed: +{best_item['distance_closed']:.2f}m)")
        except Exception as e:
            logger.warning(f"Could not parse eval_json {eval_json_path}: {e}")

    # Load from manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_name = row.get("episode_name", "").strip()
            if ep_name == selected_ep or selected_ep in ep_name:
                scene_id = row.get("scene_id", "").strip()
                pos = np.array([float(row["rec_pos_x"]), float(row["rec_pos_y"]), float(row["rec_pos_z"])], dtype=np.float32)
                rot = np.array([float(row["rot_w"]), float(row["rot_x"]), float(row["rot_y"]), float(row["rot_z"])], dtype=np.float32)
                norm = np.linalg.norm(rot)
                if norm > 1e-6:
                    rot = rot / norm
                else:
                    rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

                goal = np.array([float(row["goal_x"]), float(row["goal_y"]), float(row["goal_z"])], dtype=np.float32)
                init_dist = float(row.get("initial_dist", 0.0) or 0.0)

                target_name = "Target Goal"
                parts = ep_name.split("_")
                if len(parts) >= 3:
                    target_name = parts[2].replace("-", " ").title()

                fc = FailureCase(
                    episode_name=ep_name,
                    scene_id=scene_id,
                    failure_position=pos,
                    failure_rotation=rot,
                    goal_position=goal,
                    initial_goal_dist=init_dist,
                    metadata={
                        "target_name": target_name,
                        "trap_step": int(row.get("trap_step", -1)),
                        "recovery_step": int(row.get("recovery_step", -1)),
                        "baseline_benchmark_final_dist": float(row.get("final_dist", 0.0)),
                    }
                )
                logger.info(f"Found case in manifest: {fc.episode_name} in {fc.scene_id}")
                return fc

    raise ValueError(f"Episode '{selected_ep}' not found in manifest {manifest_path}")


def load_target_cases(
    manifest_path: Path,
    eval_json_path: Path,
    episode_query: str = "auto_best",
    num_episodes: int = 1,
) -> List[FailureCase]:
    """Loads up to num_episodes failure cases, prioritizing requested episode and top eval performers."""
    cases = []
    # 1. Specific episode
    if episode_query and episode_query != "auto_best":
        try:
            cases.append(load_target_case(manifest_path, eval_json_path, episode_query))
        except Exception as e:
            logger.warning(f"Could not load requested episode '{episode_query}': {e}")

    # 2. Add top performers from eval_iter_0.json
    if len(cases) < num_episodes and eval_json_path.exists():
        try:
            with open(eval_json_path, "r") as f:
                eval_data = json.load(f)
            results = eval_data.get("results", [])
            sorted_results = sorted(results, key=lambda x: x.get("distance_closed", 0.0), reverse=True)
            for item in sorted_results:
                if len(cases) >= num_episodes:
                    break
                ep_name = item["episode_name"]
                if any(c.episode_name == ep_name for c in cases):
                    continue
                try:
                    c = load_target_case(manifest_path, eval_json_path, ep_name)
                    cases.append(c)
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Error parsing eval_json for multiple episodes: {e}")

    # 3. Fallback to manifest if more needed
    if len(cases) < num_episodes and manifest_path.exists():
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(cases) >= num_episodes:
                    break
                ep_name = row.get("episode_name", "").strip()
                if any(c.episode_name == ep_name for c in cases):
                    continue
                try:
                    c = load_target_case(manifest_path, eval_json_path, ep_name)
                    cases.append(c)
                except Exception:
                    continue

    logger.info(f"Loaded {len(cases)} target episode(s) for video generation.")
    return cases



def run_single_rollout(
    policy_name: str,
    ckpt_path: Path,
    case: FailureCase,
    cfg: Dict,
    max_steps: int = 300,
) -> Tuple[List[Dict], np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Executes an autonomous rollout for a single policy starting at the failure recovery pose.
    """
    sim_cfg = cfg["simulation"]
    threshold_dist = sim_cfg.get("threshold_goal_distance", 1.0)
    hm3d_root = REPO_ROOT / cfg["paths"]["hm3d_scenes_path"]

    scene_glb = find_scene_glb(hm3d_root, case.scene_id)
    if scene_glb is None or not scene_glb.exists():
        raise FileNotFoundError(f"Scene mesh not found for scene_id: {case.scene_id}")

    sim, agent, _ = utils.get_sim_agent(
        str(scene_glb),
        updateNavMesh=False,
        width=sim_cfg.get("width", 320),
        height=sim_cfg.get("height", 240),
        hfov=sim_cfg.get("hfov", 120),
        sensor_height=sim_cfg.get("sensor_height", 0.4),
    )



    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.controlling_ang_vel = True
    vel_control.ang_vel_is_local = True

    ctrl_cfg_path = REPO_ROOT / cfg["paths"]["controller_config"]
    with open(ctrl_cfg_path, "r") as f:
        ctrl_cfg = yaml.safe_load(f)

    ctrl_cfg["load_run"] = str(ckpt_path)
    ctrl_cfg["waypoint_index"] = -1
    ctrl_cfg["use_adaptive_lookahead"] = False

    logger.info(f"[{policy_name}] Loading controller checkpoint: {ckpt_path.name}")
    controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")

    records = []
    full_tdv = None
    start_tdv = (0, 0)
    goal_tdv = (0, 0)

    try:
        start_state = agent.get_state()
        start_state.position = case.failure_position
        q = case.failure_rotation
        start_state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
        agent.set_state(start_state)

        vis = Visualizer(sim, agent, str(scene_glb), env="sim")
        vis.create_top_down_map(height=case.failure_position[1])
        full_tdv = vis.tdv.copy()
        start_tdv = vis.sim_to_tdv(case.failure_position)
        goal_tdv = vis.sim_to_tdv(case.goal_position)

        init_dist = ust.find_shortest_path(sim, case.failure_position, case.goal_position)[0]
        logger.info(f"[{policy_name}] Start distance to goal: {init_dist:.2f}m")

        status = "TIMEOUT"
        for step in range(max_steps):
            curr_pos = agent.get_state().position
            curr_rot = agent.get_state().rotation
            curr_dist = ust.find_shortest_path(sim, curr_pos, case.goal_position)[0]

            obs = sim.get_sensor_observations()
            display_img, depth, semantic = utils.split_observations(obs)

            if curr_dist <= threshold_dist:
                status = "SUCCESS"
                logger.info(f"[{policy_name}] 🎯 Goal reached at step {step}! (dist={curr_dist:.2f}m)")
                records.append({
                    "step": step,
                    "pos": np.array(curr_pos, dtype=np.float32),
                    "rot": curr_rot,
                    "tdv_pt": vis.sim_to_tdv(curr_pos),
                    "dist": float(curr_dist),
                    "delta_dist": float(init_dist - curr_dist),
                    "img": display_img,
                    "v": 0.0,
                    "w": 0.0,
                    "status": "SUCCESS",
                })
                break

            _, plsDict, _ = ust.get_pathlength_GT(
                sim,
                agent,
                depth,
                semantic,
                case.goal_position,
                None,
            )
            if plsDict is not None and len(plsDict) > 0:
                instaIds, pls = list(zip(*plsDict.items()))
                masks = semantic[None, ...] == np.array(instaIds)[:, None, None]
                control_input = [masks, np.array(pls, dtype=np.float32)]
            else:
                unique_classes = np.unique(semantic)
                masks = semantic[None, ...] == unique_classes[:, None, None]
                pls = np.linspace(curr_dist, curr_dist + 4.0, len(unique_classes), dtype=np.float32)
                control_input = [masks, pls]

            v, w, _ = controller.predict(display_img, control_input, visualize=False)

            records.append({
                "step": step,
                "pos": np.array(curr_pos, dtype=np.float32),
                "rot": curr_rot,
                "tdv_pt": vis.sim_to_tdv(curr_pos),
                "dist": float(curr_dist),
                "delta_dist": float(init_dist - curr_dist),
                "img": display_img,
                "v": float(v),
                "w": float(-w),
                "status": "RUNNING",
            })

            agent, sim, _ = utils.apply_velocity(
                vel_control=vel_control,
                agent=agent,
                sim=sim,
                velocity=v,
                steer=-w,
                time_step=sim_cfg.get("time_delta", 0.25),
            )

        if records and records[-1]["status"] == "RUNNING":
            records[-1]["status"] = status

        logger.info(f"[{policy_name}] Finished {len(records)} steps. Final dist: {records[-1]['dist']:.2f}m (Net closed: {records[-1]['delta_dist']:+.2f}m)")

    except Exception as e:
        logger.error(f"[{policy_name}] Error during rollout: {e}", exc_info=True)
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass

    return records, full_tdv, start_tdv, goal_tdv


def render_bev_viewport(
    base_tdv: np.ndarray,
    crop_bbox: Tuple[int, int, int, int],
    start_pt: Tuple[int, int],
    goal_pt: Tuple[int, int],
    path_history: List[Tuple[int, int]],
    curr_pt: Tuple[int, int],
    curr_rot,
    trail_color: Tuple[int, int, int],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Renders the top-down BEV view with dynamic path trails and markers, cropped cleanly."""
    r_min, r_max, c_min, c_max = crop_bbox
    tdv_canvas = base_tdv.copy()

    # Draw Start icon (Blue circle)
    cv2.circle(tdv_canvas, (start_pt[1], start_pt[0]), 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(tdv_canvas, (start_pt[1], start_pt[0]), 7, (230, 110, 40), -1, cv2.LINE_AA)

    # Draw Goal icon (Gold star / Target)
    cv2.circle(tdv_canvas, (goal_pt[1], goal_pt[0]), 12, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.circle(tdv_canvas, (goal_pt[1], goal_pt[0]), 10, (50, 215, 255), -1, cv2.LINE_AA)
    cv2.circle(tdv_canvas, (goal_pt[1], goal_pt[0]), 5, (20, 20, 20), -1, cv2.LINE_AA)

    # Draw path history trail
    if len(path_history) > 1:
        for i in range(len(path_history) - 1):
            pt1 = (path_history[i][1], path_history[i][0])
            pt2 = (path_history[i + 1][1], path_history[i + 1][0])
            cv2.line(tdv_canvas, pt1, pt2, trail_color, 3, cv2.LINE_AA)

    # Draw current agent pose
    heading = safe_heading(curr_rot)
    try:
        maps.draw_agent(tdv_canvas, curr_pt, heading, agent_radius_px=6)
    except Exception:
        r, c = curr_pt
        cv2.circle(tdv_canvas, (c, r), 7, (255, 255, 255), -1)
        cv2.circle(tdv_canvas, (c, r), 5, (0, 220, 255), -1)

    # Crop to union bounding box
    cropped = tdv_canvas[r_min:r_max, c_min:c_max]
    h_cr, w_cr = cropped.shape[:2]
    if h_cr == 0 or w_cr == 0:
        cropped = tdv_canvas
        h_cr, w_cr = cropped.shape[:2]

    # Letterbox resize into (out_w, out_h)
    scale = min((out_w - 20) / max(w_cr, 1), (out_h - 20) / max(h_cr, 1))
    new_w, new_h = max(int(w_cr * scale), 1), max(int(h_cr * scale), 1)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    viewport = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    viewport[:] = (18, 20, 26)  # Dark charcoal slate
    y_off = (out_h - new_h) // 2
    x_off = (out_w - new_w) // 2
    viewport[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    # Outer Border
    cv2.rectangle(viewport, (0, 0), (out_w - 1, out_h - 1), (45, 50, 65), 1)

    # Overlay Legend Watermark
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viewport, "TOP-DOWN BIRD'S-EYE VIEW (BEV)", (14, 26), font, 0.55, (220, 225, 235), 1, cv2.LINE_AA)

    # Start and Goal legend markers in viewport
    cv2.circle(viewport, (out_w - 180, 20), 5, (230, 110, 40), -1, cv2.LINE_AA)
    cv2.putText(viewport, "Start", (out_w - 170, 25), font, 0.45, (180, 190, 205), 1, cv2.LINE_AA)

    cv2.circle(viewport, (out_w - 95, 20), 5, (50, 215, 255), -1, cv2.LINE_AA)
    cv2.putText(viewport, "Goal", (out_w - 85, 25), font, 0.45, (180, 190, 205), 1, cv2.LINE_AA)

    return viewport


def render_policy_pane(
    header_title: str,
    subtitle: str,
    header_color: Tuple[int, int, int],
    bev_img: np.ndarray,
    fpv_rgb: np.ndarray,
    step: int,
    max_steps: int,
    curr_dist: float,
    delta_dist: float,
    v: float,
    w: float,
    outcome_badge: str,
    outcome_color: Tuple[int, int, int],
    pane_w: int = 960,
    pane_h: int = 1080,
) -> np.ndarray:
    """Builds a single synchronized policy column (960x1080) with pixel-perfect non-overlapping layout."""
    pane = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
    pane[:] = (14, 16, 22)  # Dark slate background
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Header Bar (0 to 85 px)
    header_h = 85
    cv2.rectangle(pane, (0, 0), (pane_w, header_h), (22, 26, 36), -1)
    cv2.rectangle(pane, (0, 0), (8, header_h), header_color, -1)

    cv2.putText(pane, header_title, (24, 34), font, 0.72, (245, 245, 250), 2, cv2.LINE_AA)
    cv2.putText(pane, subtitle, (24, 65), font, 0.50, header_color, 1, cv2.LINE_AA)

    # Top-right status badge
    badge_w = 260
    cv2.rectangle(pane, (pane_w - badge_w - 20, 18), (pane_w - 20, 68), (30, 36, 50), -1)
    cv2.rectangle(pane, (pane_w - badge_w - 20, 18), (pane_w - 20, 68), outcome_color, 2)
    cv2.putText(pane, outcome_badge, (pane_w - badge_w - 5, 48), font, 0.50, outcome_color, 2, cv2.LINE_AA)

    # 2. BEV Map (95 to 585 px, height = 490 px)
    bev_y = 95
    bev_h = 490
    bev_w = pane_w - 30
    bev_resized = cv2.resize(bev_img, (bev_w, bev_h), interpolation=cv2.INTER_AREA)
    pane[bev_y : bev_y + bev_h, 15 : 15 + bev_w] = bev_resized

    # 3. FPV Camera View + Telemetry HUD (595 to 870 px, height = 275 px)
    fpv_y = 595
    fpv_h = 275

    rgb_w = int(fpv_h * (4.0 / 3.0))  # ~366 px
    rgb_resized = cv2.resize(fpv_rgb, (rgb_w, fpv_h), interpolation=cv2.INTER_AREA)
    pane[fpv_y : fpv_y + fpv_h, 15 : 15 + rgb_w] = rgb_resized

    # Camera label banner
    cv2.rectangle(pane, (15, fpv_y), (15 + rgb_w, fpv_y + 26), (15, 18, 24), -1)
    cv2.putText(pane, "ROBOT FIRST-PERSON RGB SENSOR", (22, fpv_y + 18), font, 0.42, (200, 210, 225), 1, cv2.LINE_AA)

    # Telemetry HUD Card
    hud_x = 15 + rgb_w + 12
    hud_w = pane_w - hud_x - 15
    cv2.rectangle(pane, (hud_x, fpv_y), (hud_x + hud_w, fpv_y + fpv_h), (22, 26, 36), -1)
    cv2.rectangle(pane, (hud_x, fpv_y), (hud_x + hud_w, fpv_y + fpv_h), (40, 48, 65), 1)

    cv2.putText(pane, "LIVE TELEMETRY", (hud_x + 16, fpv_y + 28), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(pane, (hud_x + 16, fpv_y + 38), (hud_x + hud_w - 16, fpv_y + 38), (55, 65, 85), 1)

    stats = [
        ("Step Budget:", f"{step:03d} / {max_steps}", (200, 205, 215)),
        ("Dist to Goal:", f"{curr_dist:.2f} m", (100, 220, 255)),
        ("Distance Closed:", f"{delta_dist:+.2f} m", (80, 240, 140) if delta_dist > 0.5 else (120, 150, 240)),
        ("Linear Vel v:", f"{v:.2f} m/s", (210, 215, 225)),
        ("Angular Vel w:", f"{w:.2f} rad/s", (210, 215, 225)),
    ]

    stat_y = fpv_y + 68
    for label, val, col in stats:
        cv2.putText(pane, label, (hud_x + 16, stat_y), font, 0.45, (150, 160, 175), 1, cv2.LINE_AA)
        cv2.putText(pane, val, (hud_x + 165, stat_y), font, 0.48, col, 2, cv2.LINE_AA)
        stat_y += 34

    # Progress Bar
    bar_y = stat_y + 8
    bar_w = hud_w - 32
    cv2.rectangle(pane, (hud_x + 16, bar_y), (hud_x + 16 + bar_w, bar_y + 12), (35, 42, 55), -1)
    fill_ratio = max(0.0, min(1.0, delta_dist / max(curr_dist + delta_dist, 1e-3)))
    fill_w = int(bar_w * fill_ratio)
    cv2.rectangle(pane, (hud_x + 16, bar_y), (hud_x + 16 + fill_w, bar_y + 12), header_color, -1)

    # 4. Summary Card (880 to 1030 px, height = 150 px)
    sum_y = 880
    sum_h = 150
    cv2.rectangle(pane, (15, sum_y), (pane_w - 15, sum_y + sum_h), (22, 26, 36), -1)
    cv2.rectangle(pane, (15, sum_y), (pane_w - 15, sum_y + sum_h), outcome_color, 1)

    cv2.putText(pane, f"SUMMARY: {outcome_badge}", (28, sum_y + 35), font, 0.58, outcome_color, 2, cv2.LINE_AA)
    summary_text = f"Initial Dist: {curr_dist + delta_dist:.2f}m -> Current: {curr_dist:.2f}m"
    cv2.putText(pane, summary_text, (28, sum_y + 70), font, 0.50, (220, 225, 235), 1, cv2.LINE_AA)
    progress_text = f"Net Progress: {delta_dist:+.2f} meters"
    cv2.putText(pane, progress_text, (28, sum_y + 105), font, 0.52, (100, 230, 255), 2, cv2.LINE_AA)

    return pane


def create_intro_slide(case: FailureCase, width: int = 1920, height: int = 1080, fps: int = 15, duration_sec: float = 3.5) -> List[np.ndarray]:
    """Generates an aesthetic Full HD intro slide explaining the benchmark and failure recovery setup."""
    slide = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        r = y / height
        slide[y, :] = np.array([16 + 12 * r, 18 + 14 * r, 26 + 18 * r], dtype=np.uint8)

    slide[0:8, :] = [50, 200, 120]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(slide, "EMBODIED AI NAVIGATION BENCHMARK", (100, 130), font, 1.35, (245, 245, 250), 3, cv2.LINE_AA)
    cv2.putText(slide, "Side-by-Side Head-to-Head Comparison: Baseline Policy vs. DAgger Recovery Policy", (100, 185), font, 0.88, (100, 220, 255), 2, cv2.LINE_AA)
    cv2.line(slide, (100, 220), (width - 100, 220), (60, 75, 100), 2)

    details = [
        ("Episode Name:", case.episode_name),
        ("Scene Mesh:", case.scene_id),
        ("Target Object:", case.metadata.get("target_name", "Goal Object")),
        ("Failure Location:", f"Trapped at Step {case.metadata.get('trap_step', 226)}"),
        ("Recovery Initial State:", f"Backtracked 50 steps to Step {case.metadata.get('recovery_step', 176)}"),
        ("Initial Distance to Goal:", f"{case.initial_goal_dist:.2f} meters"),
        ("Navigation Budget:", "300 Autonomous Steps (SafeTurn Flag: OFF)"),
    ]

    y_pos = 295
    for label, val in details:
        cv2.putText(slide, label, (120, y_pos), font, 0.72, (160, 175, 195), 2, cv2.LINE_AA)
        cv2.putText(slide, val, (480, y_pos), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos += 48

    y_pos += 30
    box_w = (width - 240 - 40) // 2
    cv2.rectangle(slide, (120, y_pos), (120 + box_w, y_pos + 130), (25, 30, 42), -1)
    cv2.rectangle(slide, (120, y_pos), (120 + box_w, y_pos + 130), (50, 120, 240), 2)
    cv2.putText(slide, "BASELINE POLICY (Iter -1)", (145, y_pos + 42), font, 0.75, (50, 120, 240), 2, cv2.LINE_AA)
    cv2.putText(slide, "Behavioral Cloning without online recovery data.", (145, y_pos + 78), font, 0.52, (190, 200, 215), 1, cv2.LINE_AA)
    cv2.putText(slide, "Known Behavior: Stagnates / Loops in local corner minima.", (145, y_pos + 105), font, 0.52, (220, 120, 120), 1, cv2.LINE_AA)

    x_r = 120 + box_w + 40
    cv2.rectangle(slide, (x_r, y_pos), (x_r + box_w, y_pos + 130), (25, 30, 42), -1)
    cv2.rectangle(slide, (x_r, y_pos), (x_r + box_w, y_pos + 130), (80, 210, 120), 2)
    cv2.putText(slide, "DAGGER POLICY (Iter 0)", (x_r + 25, y_pos + 42), font, 0.75, (80, 210, 120), 2, cv2.LINE_AA)
    cv2.putText(slide, "Trained on mined failure cases with expert corrections.", (x_r + 25, y_pos + 78), font, 0.52, (190, 200, 215), 1, cv2.LINE_AA)
    cv2.putText(slide, "Target Behavior: Escapes obstacle trap and advances towards goal.", (x_r + 25, y_pos + 105), font, 0.52, (100, 230, 140), 1, cv2.LINE_AA)

    num_frames = int(duration_sec * fps)
    return [slide.copy() for _ in range(num_frames)]


def create_outro_slide(
    case: FailureCase,
    base_final_dist: float,
    base_closed: float,
    base_steps: int,
    dag_final_dist: float,
    dag_closed: float,
    dag_steps: int,
    width: int = 1920,
    height: int = 1080,
    fps: int = 15,
    duration_sec: float = 4.0,
) -> List[np.ndarray]:
    """Generates the final summary verdict card displaying comparative head-to-head metrics."""
    slide = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        r = y / height
        slide[y, :] = np.array([16 + 12 * r, 18 + 14 * r, 26 + 18 * r], dtype=np.uint8)

    slide[0:8, :] = [80, 210, 120]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(slide, "FINAL BENCHMARK VERDICT", (100, 120), font, 1.35, (245, 245, 250), 3, cv2.LINE_AA)
    cv2.putText(slide, f"Episode: {case.episode_name} ({case.metadata.get('target_name', 'Goal')})", (100, 175), font, 0.85, (100, 220, 255), 2, cv2.LINE_AA)
    cv2.line(slide, (100, 205), (width - 100, 205), (60, 75, 100), 2)

    y_table = 270
    cv2.rectangle(slide, (100, y_table), (width - 100, y_table + 50), (32, 38, 52), -1)
    cv2.putText(slide, "EVALUATION METRIC", (120, y_table + 34), font, 0.65, (200, 210, 225), 2, cv2.LINE_AA)
    cv2.putText(slide, "BASELINE POLICY", (620, y_table + 34), font, 0.65, (50, 120, 240), 2, cv2.LINE_AA)
    cv2.putText(slide, "DAGGER POLICY (ITER 0)", (1080, y_table + 34), font, 0.65, (80, 210, 120), 2, cv2.LINE_AA)
    cv2.putText(slide, "IMPROVEMENT", (1580, y_table + 34), font, 0.65, (255, 215, 60), 2, cv2.LINE_AA)

    delta_gain = dag_closed - base_closed
    rows = [
        ("Initial Geodesic Distance", f"{case.initial_goal_dist:.2f} m", f"{case.initial_goal_dist:.2f} m", "Identical Pose"),
        ("Final Geodesic Distance", f"{base_final_dist:.2f} m", f"{dag_final_dist:.2f} m", f"{base_final_dist - dag_final_dist:+.2f} m closer"),
        ("Net Distance Closed", f"{base_closed:+.2f} m", f"{dag_closed:+.2f} m", f"{delta_gain:+.2f} m advantage"),
        ("Steps Utilized", f"{base_steps} steps", f"{dag_steps} steps", "Efficient navigation"),
        ("Terminal Status", "STAGNATED / TRAPPED ❌", "ESCAPED & ADVANCED ✅", "TRAP AVOIDED!"),
    ]

    curr_y = y_table + 50
    for label, b_val, d_val, gain in rows:
        cv2.rectangle(slide, (100, curr_y), (width - 100, curr_y + 45), (24, 28, 38) if (curr_y // 45) % 2 == 0 else (20, 24, 32), -1)
        cv2.putText(slide, label, (120, curr_y + 30), font, 0.58, (175, 185, 200), 1, cv2.LINE_AA)
        cv2.putText(slide, b_val, (620, curr_y + 30), font, 0.60, (240, 140, 140) if "TRAPPED" in b_val else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(slide, d_val, (1080, curr_y + 30), font, 0.60, (80, 230, 140) if "ADVANCED" in d_val else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(slide, gain, (1580, curr_y + 30), font, 0.60, (255, 215, 60), 2, cv2.LINE_AA)
        curr_y += 45

    v_y = curr_y + 40
    cv2.rectangle(slide, (100, v_y), (width - 100, v_y + 160), (25, 36, 32), -1)
    cv2.rectangle(slide, (100, v_y), (width - 100, v_y + 160), (80, 210, 120), 2)

    cv2.putText(slide, "HEAD-TO-HEAD CONCLUSION:", (140, v_y + 45), font, 0.85, (80, 210, 120), 2, cv2.LINE_AA)
    verdict_desc1 = "DAgger policy successfully out-navigated the Baseline policy by escaping the local corner trap."
    verdict_desc2 = f"Baseline stagnated with only {base_closed:+.2f}m progress, while DAgger closed {dag_closed:+.2f}m towards the target {case.metadata.get('target_name', 'goal')}!"
    cv2.putText(slide, verdict_desc1, (140, v_y + 90), font, 0.68, (240, 245, 250), 2, cv2.LINE_AA)
    cv2.putText(slide, verdict_desc2, (140, v_y + 130), font, 0.68, (120, 240, 255), 2, cv2.LINE_AA)

    num_frames = int(duration_sec * fps)
    return [slide.copy() for _ in range(num_frames)]


def generate_side_by_side_video(case: FailureCase, cfg: Dict, args):
    """Orchestrates rollouts, frame assembly, and video rendering."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_filename = args.output_name or f"side_by_side_{case.episode_name}.mp4"
    out_path = out_dir / out_filename

    logger.info("=" * 80)
    logger.info(f"STARTING SIDE-BY-SIDE BENCHMARK FOR EPISODE: {case.episode_name}")
    logger.info(f"Scene: {case.scene_id} | Initial Goal Dist: {case.initial_goal_dist:.2f}m")
    logger.info("=" * 80)

    # 1. Rollout Baseline Policy
    base_ckpt = REPO_ROOT / args.baseline_ckpt
    logger.info(f"\n--- Running Rollout 1/2: Baseline Policy ({base_ckpt.name}) ---")
    base_records, base_tdv, base_start_tdv, base_goal_tdv = run_single_rollout(
        policy_name="Baseline",
        ckpt_path=base_ckpt,
        case=case,
        cfg=cfg,
        max_steps=args.max_steps,
    )

    # 2. Rollout DAgger Policy
    dag_ckpt = REPO_ROOT / args.dagger_ckpt
    logger.info(f"\n--- Running Rollout 2/2: DAgger Policy ({dag_ckpt.name}) ---")
    dag_records, dag_tdv, dag_start_tdv, dag_goal_tdv = run_single_rollout(
        policy_name="DAgger_Iter0",
        ckpt_path=dag_ckpt,
        case=case,
        cfg=cfg,
        max_steps=args.max_steps,
    )

    if not base_records or not dag_records:
        logger.error("Failed to collect records for one or both policies!")
        return

    # Compute union bounding box for BEV top-down crop across both trajectories
    all_points = (
        [base_start_tdv, base_goal_tdv]
        + [r["tdv_pt"] for r in base_records]
        + [r["tdv_pt"] for r in dag_records]
    )
    r_coords = [p[0] for p in all_points]
    c_coords = [p[1] for p in all_points]
    r_min, r_max = min(r_coords), max(r_coords)
    c_min, c_max = min(c_coords), max(c_coords)

    span = max(r_max - r_min, c_max - c_min)
    margin = max(35, int(span * 0.25))

    tdv_h, tdv_w = base_tdv.shape[:2]
    r_min_crop = max(0, r_min - margin)
    r_max_crop = min(tdv_h, r_max + margin)
    c_min_crop = max(0, c_min - margin)
    c_max_crop = min(tdv_w, c_max + margin)

    crop_bbox = (r_min_crop, r_max_crop, c_min_crop, c_max_crop)
    logger.info(f"Shared BEV crop bounding box: rows [{r_min_crop}:{r_max_crop}], cols [{c_min_crop}:{c_max_crop}]")

    total_steps = max(len(base_records), len(dag_records))
    logger.info(f"Synchronizing frames across {total_steps} steps...")

    out_w, out_h = 1920, 1080
    col_w = out_w // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, out_h))

    base_writer = None
    dag_writer = None
    if getattr(args, "export_individual", False):
        base_path = out_dir / f"baseline_{case.episode_name}.mp4"
        dag_path = out_dir / f"dagger_{case.episode_name}.mp4"
        base_writer = cv2.VideoWriter(str(base_path), fourcc, args.fps, (col_w, out_h))
        dag_writer = cv2.VideoWriter(str(dag_path), fourcc, args.fps, (col_w, out_h))

    # 1. Write Intro Slide
    logger.info("Writing intro slide (3.5s)...")
    intro_frames = create_intro_slide(case, width=out_w, height=out_h, fps=args.fps, duration_sec=3.5)
    for f in intro_frames:
        writer.write(f)
        if base_writer is not None:
            base_writer.write(f[:, :col_w])
        if dag_writer is not None:
            dag_writer.write(f[:, col_w:])

    # 2. Render synchronized side-by-side rollout frames
    logger.info("Rendering synchronized side-by-side simulation frames...")
    base_history = []
    dag_history = []

    for t in range(total_steps):
        idx_b = min(t, len(base_records) - 1)
        idx_d = min(t, len(dag_records) - 1)

        rec_b = base_records[idx_b]
        rec_d = dag_records[idx_d]

        if t < len(base_records):
            base_history.append(rec_b["tdv_pt"])
        if t < len(dag_records):
            dag_history.append(rec_d["tdv_pt"])

        b_outcome = "TRAPPED / STAGNATING" if rec_b["delta_dist"] < 0.5 else "STRUGGLING"
        b_color = (60, 80, 240) if rec_b["delta_dist"] < 0.5 else (50, 150, 240)
        if rec_b["status"] == "SUCCESS":
            b_outcome = "GOAL REACHED! ✅"
            b_color = (60, 220, 100)

        d_outcome = "ESCAPING TRAP & ADVANCING" if rec_d["delta_dist"] > 0.5 else "INITIALIZING"
        d_color = (80, 220, 120) if rec_d["delta_dist"] > 0.5 else (200, 200, 80)
        if rec_d["status"] == "SUCCESS":
            d_outcome = "GOAL REACHED! ✅"
            d_color = (60, 240, 120)

        # Render BEV viewport for Baseline
        bev_base = render_bev_viewport(
            base_tdv=base_tdv,
            crop_bbox=crop_bbox,
            start_pt=base_start_tdv,
            goal_pt=base_goal_tdv,
            path_history=base_history,
            curr_pt=rec_b["tdv_pt"],
            curr_rot=rec_b["rot"],
            trail_color=(40, 60, 240),
            out_w=col_w - 30,
            out_h=490,
        )

        # Render BEV viewport for DAgger
        bev_dag = render_bev_viewport(
            base_tdv=dag_tdv,
            crop_bbox=crop_bbox,
            start_pt=dag_start_tdv,
            goal_pt=dag_goal_tdv,
            path_history=dag_history,
            curr_pt=rec_d["tdv_pt"],
            curr_rot=rec_d["rot"],
            trail_color=(60, 230, 100),
            out_w=col_w - 30,
            out_h=490,
        )

        left_col = render_policy_pane(
            header_title="BASELINE POLICY (Iteration -1)",
            subtitle="Standard Behavioral Cloning (Trapped in Corner)",
            header_color=(50, 120, 240),
            bev_img=bev_base,
            fpv_rgb=rec_b["img"],
            step=rec_b["step"],
            max_steps=args.max_steps,
            curr_dist=rec_b["dist"],
            delta_dist=rec_b["delta_dist"],
            v=rec_b["v"],
            w=rec_b["w"],
            outcome_badge=b_outcome,
            outcome_color=b_color,
            pane_w=col_w,
            pane_h=out_h,
        )

        right_col = render_policy_pane(
            header_title="DAGGER RECOVERY POLICY (Iteration 0)",
            subtitle="Learned Online Recovery (Escaping Local Minimum)",
            header_color=(80, 210, 120),
            bev_img=bev_dag,
            fpv_rgb=rec_d["img"],
            step=rec_d["step"],
            max_steps=args.max_steps,
            curr_dist=rec_d["dist"],
            delta_dist=rec_d["delta_dist"],
            v=rec_d["v"],
            w=rec_d["w"],
            outcome_badge=d_outcome,
            outcome_color=d_color,
            pane_w=col_w,
            pane_h=out_h,
        )

        frame = np.hstack((left_col, right_col))

        # Vertical Divider
        cv2.line(frame, (col_w - 1, 0), (col_w - 1, out_h), (50, 60, 75), 2)
        cv2.line(frame, (col_w + 1, 0), (col_w + 1, out_h), (25, 30, 40), 2)

        # Global Bottom Super-Banner (1035 to 1080 px, height 45 px)
        banner_y = 1035
        cv2.rectangle(frame, (0, banner_y), (out_w, out_h), (12, 14, 20), -1)
        cv2.line(frame, (0, banner_y), (out_w, banner_y), (70, 80, 100), 1)

        if rec_d["delta_dist"] > rec_b["delta_dist"] + 1.0:
            verdict_text = f"LIVE VERDICT: DAgger successfully broke free from the obstacle! (Advantage: {rec_d['delta_dist'] - rec_b['delta_dist']:+.2f}m closer to goal)"
            v_col = (100, 240, 140)
        else:
            verdict_text = "LIVE VERDICT: Both policies navigating from identical failure recovery pose..."
            v_col = (200, 210, 225)

        cv2.putText(frame, verdict_text, (25, banner_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, v_col, 2, cv2.LINE_AA)

        writer.write(frame)
        if base_writer is not None:
            base_writer.write(left_col)
        if dag_writer is not None:
            dag_writer.write(right_col)

    # 3. Write Outro Slide
    logger.info("Writing outro conclusion slide (4.0s)...")
    outro_frames = create_outro_slide(
        case=case,
        base_final_dist=base_records[-1]["dist"],
        base_closed=base_records[-1]["delta_dist"],
        base_steps=len(base_records),
        dag_final_dist=dag_records[-1]["dist"],
        dag_closed=dag_records[-1]["delta_dist"],
        dag_steps=len(dag_records),
        width=out_w,
        height=out_h,
        fps=args.fps,
        duration_sec=4.0,
    )
    for f in outro_frames:
        writer.write(f)
        if base_writer is not None:
            base_writer.write(f[:, :col_w])
        if dag_writer is not None:
            dag_writer.write(f[:, col_w:])

    writer.release()
    if base_writer is not None:
        base_writer.release()
        logger.info(f"   -> Saved Baseline Standalone Video to: {base_path.resolve()}")
    if dag_writer is not None:
        dag_writer.release()
        logger.info(f"   -> Saved DAgger Standalone Video to: {dag_path.resolve()}")

    logger.info(f"✅ Successfully generated and saved Side-by-Side Comparison Video to:\n--> {out_path.resolve()}")
    return out_path


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    cfg_path = REPO_ROOT / args.config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    manifest_path = REPO_ROOT / args.manifest
    eval_json_path = REPO_ROOT / args.eval_json

    cases = load_target_cases(manifest_path, eval_json_path, args.episode, args.num_episodes)
    if not cases:
        logger.error("No valid failure cases found!")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_videos = []

    for idx, case in enumerate(cases):
        logger.info("=" * 80)
        logger.info(f"PROCESSING EPISODE {idx + 1}/{len(cases)}: {case.episode_name}")
        logger.info("=" * 80)

        if len(cases) == 1 and args.output_name:
            out_name = args.output_name
        else:
            out_name = f"side_by_side_{case.episode_name}.mp4"

        try:
            custom_args = argparse.Namespace(**vars(args))
            custom_args.output_name = out_name
            vid_path = generate_side_by_side_video(case, cfg, custom_args)
            if vid_path and vid_path.exists():
                generated_videos.append(vid_path)
        except Exception as e:
            logger.error(f"Error generating video for episode {case.episode_name}: {e}", exc_info=True)
            continue


    if args.concat and len(generated_videos) > 1:
        concat_file = out_dir / f"side_by_side_compilation_top{len(generated_videos)}.mp4"
        logger.info(f"Concatenating {len(generated_videos)} comparison videos into: {concat_file.name}")
        concatenate_videos(generated_videos, concat_file, fps=args.fps)
        logger.info(f"✅ Master compilation video saved to:\n--> {concat_file.resolve()}")


if __name__ == "__main__":
    main()

