#!/usr/bin/env python3
"""
Single-Policy Deep Inspection Video Generator for DAgger Iteration 0:
Visualizes the autonomous recovery behavior of policy_iter_0 on failure episodes.

Creates a unified 1080p (1920x1080) synchronized inspection dashboard containing:
  1. Top-Down Floorplan Map (BEV): Full environment floorplan with start, goal, and agent visited path trail.
  2. Robot First-Person Camera (RGB Sensor): Exact visual observations with directional steering HUD.
  3. Predicted Local Waypoints (Metric Coordinate Grid): 2D plot of the 10 local waypoints output by the model.
  4. Viridis Goal Distance Cost Map: Semantic geodesic distance distribution across visible scene objects.
  5. Live Telemetry HUD: Step budget, distance to goal, progress closed, and linear/angular velocities.

Supports multi-episode batch evaluation and automatic concatenation into a single master compilation video.
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
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[PolicyInspection]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import habitat_sim
from habitat.utils.visualizations import maps
from libs.common import utils
from libs.common import utils_sim_traj as ust
from libs.common.vis_utils import letterbox_to_dims, concatenate_videos, create_intro_slide
from libs.logger.visualizer import Visualizer, rot_to_heading
from libs.control.objectreact import ObjRelLearntController
from old_dagger.core.failure_loader import FailureCase, find_scene_glb


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Policy Deep Inspection Video for DAgger Iter 0")
    parser.add_argument("--config", default="dagger/multi_chunk_runs/chunk_0_dagger_config.yaml", help="DAgger YAML config path")
    parser.add_argument("--checkpoint", default="dagger/multi_chunk_runs/checkpoints/policy_iter_0.pth", help="Policy checkpoint (.pth)")
    parser.add_argument("--manifest", default="dagger/multi_chunk_runs/manifests/chunk_0_trapped_manifest.csv", help="Trapped cases CSV manifest")
    parser.add_argument("--eval_json", default="dagger/multi_chunk_runs/logs/eval_iter_0.json", help="Eval results JSON")
    parser.add_argument("--episode", default="auto_best", help="Specific episode name or 'auto_best' to pick top recovery cases")
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to evaluate (defaults to 1 if specific episode is given, 5 if auto_best)")
    parser.add_argument("--output_dir", default="dagger/multi_chunk_runs/visualizations/policy_inspection", help="Output directory")
    parser.add_argument("--output_name", default=None, help="Custom output video filename")
    parser.add_argument("--concat", action="store_true", default=True, help="Concatenate all generated videos into a single master video (default: True)")
    parser.add_argument("--no_concat", dest="concat", action="store_false", help="Disable automatic concatenation")
    parser.add_argument("--max_steps", type=int, default=300, help="Max rollout steps per episode")
    parser.add_argument("--fps", type=int, default=15, help="Video framerate")
    parser.add_argument("--gpu", type=int, default=1, help="CUDA GPU ID (default: 1)")
    parsed = parser.parse_args()
    if parsed.num_episodes is None:
        parsed.num_episodes = 1 if (parsed.episode and parsed.episode != "auto_best") else 5
    return parsed


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
    """Loads a single FailureCase by episode name."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_name = row.get("episode_name", "").strip()
            if ep_name == episode_query or episode_query in ep_name:
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
                return fc

    raise ValueError(f"Episode '{episode_query}' not found in manifest {manifest_path}")


def load_target_cases(
    manifest_path: Path,
    eval_json_path: Path,
    episode_query: str = "auto_best",
    num_episodes: int = 5,
) -> List[FailureCase]:
    """Loads a list of candidate failure recovery cases."""
    cases = []
    if episode_query and episode_query != "auto_best":
        try:
            cases.append(load_target_case(manifest_path, eval_json_path, episode_query))
        except Exception as e:
            logger.warning(f"Could not load requested episode '{episode_query}': {e}")

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
            logger.warning(f"Error reading eval_json: {e}")

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

    logger.info(f"Loaded {len(cases)} target episode(s) for inspection video generation.")
    return cases


def render_waypoint_curve(
    pred_waypoints: np.ndarray,
    width: int = 520,
    height: int = 440,
    title: str = "Local Predicted Waypoints (GNM)",
) -> np.ndarray:
    """Renders local predicted waypoints on a high-contrast metric coordinate grid."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#161822")
    ax.set_facecolor("#161822")

    pts = pred_waypoints.copy()
    if pts is not None and len(pts) > 0:
        # Check if coordinates are in cm (e.g. max value > 15) and convert to meters
        if np.max(np.abs(pts[:, :2])) > 15.0:
            pts = pts[:, :2] / 100.0
        else:
            pts = pts[:, :2]

        # In robot frame: pts[:, 0] is forward (X), pts[:, 1] is lateral (Y)
        # Plot with lateral on X-axis (inverted to match camera: left is +Y, right is -Y) and forward on Y-axis
        ax.plot(
            pts[:, 1],
            pts[:, 0],
            color="#00e5ff",
            linewidth=2.8,
            marker="o",
            markersize=7,
            markerfacecolor="#ffe600",
            markeredgecolor="#ffffff",
            markeredgewidth=1.4,
            label="Predicted Trajectory",
            zorder=4,
        )

        # Plot individual waypoint indices
        for i, pt in enumerate(pts):
            ax.text(
                pt[1] + 0.15,
                pt[0],
                f"w{i+1}",
                color="#e0e8f5",
                fontsize=7.5,
                weight="bold",
                zorder=5,
            )

    # Robot origin marker at (0, 0)
    ax.scatter([0], [0], color="#ff3366", s=110, marker="^", label="Robot Position", zorder=6)

    ax.set_xlim(3.5, -3.5)  # Inverted so left in plot matches left in camera view
    ax.set_ylim(-0.5, 6.5)
    ax.set_xlabel("Lateral Offset Y (meters)", color="#a0abbd", fontsize=9.5)
    ax.set_ylabel("Forward Distance X (meters)", color="#a0abbd", fontsize=9.5)
    ax.set_title(title, color="#ffffff", fontsize=11, fontweight="bold", pad=8)

    ax.tick_params(colors="#808b96", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#3b4252")
    ax.grid(True, color="#2e3440", linestyle="--", alpha=0.75)

    plt.tight_layout()
    fig.canvas.draw()
    try:
        plot_img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    except AttributeError:
        plot_img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    return cv2.resize(plot_img, (width, height), interpolation=cv2.INTER_AREA)


def render_viridis_costmap(
    goal_mask_vis: Optional[np.ndarray],
    width: int = 520,
    height: int = 440,
) -> np.ndarray:
    """Renders the viridis goal distance cost map."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#161822")
    ax.set_facecolor("#161822")

    if goal_mask_vis is not None and np.any(goal_mask_vis > 0):
        im = ax.imshow(goal_mask_vis, cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#a0abbd", labelsize=8)
        for spine in cbar.ax.spines.values():
            spine.set_color("#3b4252")
        cbar.set_label("Geodesic Distance / Cost to Goal", color="#a0abbd", fontsize=8.5)
    else:
        ax.text(0.5, 0.5, "No Object Distance Map Available", color="#808b96", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Viridis Cost Map (Yellow = Target Object)", color="#ffffff", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    try:
        plot_img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    except AttributeError:
        plot_img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    return cv2.resize(plot_img, (width, height), interpolation=cv2.INTER_AREA)


def build_inspection_frame(
    bev_map: np.ndarray,
    rgb_img: np.ndarray,
    wp_plot_img: np.ndarray,
    costmap_img: np.ndarray,
    step: int,
    max_steps: int,
    curr_dist: float,
    init_dist: float,
    v: float,
    w: float,
    target_name: str,
    status_text: str = "EVALUATING",
) -> np.ndarray:
    """
    Assembles a unified Full HD (1920x1080) inspection dashboard:
      - Left Column (860x1080): Header telemetry + High-Resolution Top-Down BEV Floorplan
      - Right Column (1060x1080): 2x2 High-Resolution Inspection Grid:
          [RGB Camera with HUD]             [Viridis Goal Distance Mask]
          [Local Predicted Waypoint Curve]  [Live Telemetry & Velocity Gauge]
    """
    total_w, total_h = 1920, 1080
    frame = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    frame[:] = (16, 18, 24)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # =========================================================================
    # 1. LEFT COLUMN: Top-Down BEV Floorplan & Header (Width = 860 px)
    # =========================================================================
    left_w = 860
    header_h = 95
    cv2.rectangle(frame, (12, 10), (left_w - 12, header_h), (25, 29, 39), -1)
    cv2.rectangle(frame, (12, 10), (left_w - 12, header_h), (50, 60, 80), 1)
    cv2.rectangle(frame, (12, 10), (20, header_h), (80, 220, 120), -1)  # Emerald accent bar

    cv2.putText(frame, f"TARGET: {target_name.upper()}", (32, 42), font, 0.75, (245, 245, 250), 2, cv2.LINE_AA)

    # Status badge
    b_color = (60, 220, 100) if status_text == "SUCCESS" else ((220, 80, 80) if "TRAP" in status_text else (80, 210, 255))
    cv2.putText(frame, f"[{status_text}]", (left_w - 220, 42), font, 0.65, b_color, 2, cv2.LINE_AA)

    delta_dist = init_dist - curr_dist
    telemetry_str = f"Step: {step:03d}/{max_steps}  |  Goal Dist: {curr_dist:.2f}m  |  Net Progress: {delta_dist:+.2f}m"
    cv2.putText(frame, telemetry_str, (32, 78), font, 0.52, (200, 210, 225), 1, cv2.LINE_AA)

    # Top-Down BEV placement
    bev_area_h = total_h - header_h - 25
    bev_letterboxed = letterbox_to_dims(bev_map, left_w - 24, bev_area_h)
    frame[header_h + 15 : header_h + 15 + bev_area_h, 12 : left_w - 12] = bev_letterboxed

    # Border & Watermark on BEV
    cv2.rectangle(frame, (12, header_h + 15), (left_w - 12, header_h + 15 + bev_area_h), (50, 60, 80), 1)
    cv2.putText(frame, "TOP-DOWN BIRD'S-EYE VIEW (BEV)", (26, header_h + 40), font, 0.52, (220, 225, 235), 1, cv2.LINE_AA)

    # Divider between Left and Right columns
    cv2.line(frame, (left_w, 0), (left_w, total_h), (45, 55, 75), 2)

    # =========================================================================
    # 2. RIGHT COLUMN: 2x2 Grid Inspection Panels (Width = 1060 px)
    # =========================================================================
    right_x = left_w + 12
    right_w = total_w - right_x - 12  # ~1036 px
    quad_w = (right_w - 12) // 2       # ~512 px
    quad_h = (total_h - 24 - 12) // 2  # ~522 px

    # --- Quadrant 1 (Top-Left): Robot First-Person RGB Sensor ---
    q1 = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    q1[:] = (22, 26, 36)
    cv2.rectangle(q1, (0, 0), (quad_w - 1, quad_h - 1), (45, 55, 75), 1)
    cv2.putText(q1, "1. ROBOT FIRST-PERSON RGB SENSOR", (16, 26), font, 0.52, (80, 220, 140), 1, cv2.LINE_AA)

    rgb_vis = rgb_img.copy()
    # Draw forward steering heading indicator arrow on camera view
    h_c, w_c = rgb_vis.shape[:2]
    cx, cy = w_c // 2, h_c - 15
    arrow_len = 35
    end_x = int(cx + arrow_len * math.sin(-w * 2.5))
    end_y = int(cy - arrow_len * math.cos(-w * 2.5))
    cv2.arrowedLine(rgb_vis, (cx, cy), (end_x, end_y), (0, 235, 255), 3, tipLength=0.35)
    cv2.circle(rgb_vis, (cx, cy), 4, (255, 255, 255), -1)

    rgb_box = letterbox_to_dims(rgb_vis, quad_w - 20, quad_h - 45)
    q1[38 : 38 + rgb_box.shape[0], 10 : 10 + rgb_box.shape[1]] = rgb_box

    # --- Quadrant 2 (Top-Right): Viridis Goal Distance Cost Map ---
    q2 = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    q2[:] = (22, 26, 36)
    cv2.rectangle(q2, (0, 0), (quad_w - 1, quad_h - 1), (45, 55, 75), 1)
    cv2.putText(q2, "2. VIRIDIS GOAL DISTANCE COST MAP", (16, 26), font, 0.52, (255, 220, 80), 1, cv2.LINE_AA)
    cost_box = letterbox_to_dims(costmap_img, quad_w - 20, quad_h - 45)
    q2[38 : 38 + cost_box.shape[0], 10 : 10 + cost_box.shape[1]] = cost_box

    # --- Quadrant 3 (Bottom-Left): Local Predicted Waypoints Plot ---
    q3 = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    q3[:] = (22, 26, 36)
    cv2.rectangle(q3, (0, 0), (quad_w - 1, quad_h - 1), (45, 55, 75), 1)
    cv2.putText(q3, "3. LOCAL PREDICTED WAYPOINTS (METRIC GRID)", (16, 26), font, 0.52, (80, 210, 255), 1, cv2.LINE_AA)
    wp_box = letterbox_to_dims(wp_plot_img, quad_w - 20, quad_h - 45)
    q3[38 : 38 + wp_box.shape[0], 10 : 10 + wp_box.shape[1]] = wp_box

    # --- Quadrant 4 (Bottom-Right): Live Telemetry & Control Gauge Card ---
    q4 = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    q4[:] = (22, 26, 36)
    cv2.rectangle(q4, (0, 0), (quad_w - 1, quad_h - 1), (45, 55, 75), 1)
    cv2.putText(q4, "4. REAL-TIME NAVIGATION TELEMETRY", (16, 26), font, 0.52, (220, 225, 235), 1, cv2.LINE_AA)
    cv2.line(q4, (16, 36), (quad_w - 16, 36), (45, 55, 75), 1)

    card_stats = [
        ("Evaluated Policy:", "DAgger Recovery Policy (Iter 0)", (80, 220, 140)),
        ("Step Count:", f"{step:03d} / {max_steps}", (200, 210, 225)),
        ("Start Distance:", f"{init_dist:.2f} meters", (160, 175, 195)),
        ("Current Distance:", f"{curr_dist:.2f} meters", (100, 220, 255)),
        ("Distance Closed:", f"{delta_dist:+.2f} meters", (80, 240, 140) if delta_dist > 0.5 else (120, 150, 240)),
        ("Linear Velocity v:", f"{v:.3f} m/s", (220, 225, 235)),
        ("Angular Velocity w:", f"{w:.3f} rad/s", (220, 225, 235)),
    ]

    card_y = 70
    for label, val, col in card_stats:
        cv2.putText(q4, label, (22, card_y), font, 0.48, (140, 155, 175), 1, cv2.LINE_AA)
        cv2.putText(q4, val, (225, card_y), font, 0.52, col, 2, cv2.LINE_AA)
        card_y += 38

    # Progress Bar at bottom of card
    card_y += 15
    cv2.putText(q4, "Distance Closed Progress:", (22, card_y), font, 0.48, (140, 155, 175), 1, cv2.LINE_AA)
    bar_y = card_y + 12
    bar_w = quad_w - 44
    cv2.rectangle(q4, (22, bar_y), (22 + bar_w, bar_y + 16), (35, 42, 55), -1)
    ratio = max(0.0, min(1.0, delta_dist / max(init_dist, 1e-3)))
    fill_w = int(bar_w * ratio)
    cv2.rectangle(q4, (22, bar_y), (22 + fill_w, bar_y + 16), (80, 220, 120), -1)
    cv2.putText(q4, f"{ratio*100:.1f}%", (22 + bar_w + 6, bar_y + 13), font, 0.42, (180, 190, 205), 1, cv2.LINE_AA)

    # Place the 4 quadrants on the right column
    y1 = 12
    y2 = y1 + quad_h + 12
    x1 = right_x
    x2 = x1 + quad_w + 12

    frame[y1 : y1 + quad_h, x1 : x1 + quad_w] = q1
    frame[y1 : y1 + quad_h, x2 : x2 + quad_w] = q2
    frame[y2 : y2 + quad_h, x1 : x1 + quad_w] = q3
    frame[y2 : y2 + quad_h, x2 : x2 + quad_w] = q4

    return frame


def run_inspection_rollout(case: FailureCase, cfg: Dict, args, output_path: Path) -> bool:
    """Executes a single autonomous evaluation rollout for policy_iter_0 and records the inspection dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim_cfg = cfg["simulation"]
    threshold_dist = sim_cfg.get("threshold_goal_distance", 1.0)
    hm3d_root = REPO_ROOT / cfg["paths"]["hm3d_scenes_path"]

    scene_glb = find_scene_glb(hm3d_root, case.scene_id)
    if scene_glb is None or not scene_glb.exists():
        logger.error(f"Scene GLB not found for {case.scene_id}")
        return False

    # Initialize Simulator
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

    # Preload Controller
    ctrl_cfg_path = REPO_ROOT / cfg["paths"]["controller_config"]
    with open(ctrl_cfg_path, "r") as f:
        ctrl_cfg = yaml.safe_load(f)

    checkpoint_file = REPO_ROOT / args.checkpoint
    ctrl_cfg["load_run"] = str(checkpoint_file)
    ctrl_cfg["waypoint_index"] = -1
    ctrl_cfg["use_adaptive_lookahead"] = False

    logger.info(f"Loading controller checkpoint: {checkpoint_file.name}")
    controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")

    target_name = case.metadata.get("target_name", "Target Goal")
    logger.info(f"Running Policy Inspection Rollout: {case.episode_name} ({case.scene_id}) -> {target_name}")

    # Set start pose (50-step backtrack recovery pose)
    start_state = agent.get_state()
    start_state.position = case.failure_position
    q = case.failure_rotation
    start_state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
    agent.set_state(start_state)

    # Visualizer for Top-Down BEV Map
    vis = Visualizer(sim, agent, str(scene_glb), env="sim")
    vis.create_top_down_map(height=case.failure_position[1])
    vis.draw_goal(vis.sim_to_tdv(case.goal_position))
    vis.draw_start(vis.sim_to_tdv(case.failure_position))

    init_dist = ust.find_shortest_path(sim, case.failure_position, case.goal_position)[0]
    final_status = "EXCEEDED_STEPS"
    final_dist = init_dist

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (1920, 1080))

    # 1. Write Intro Slide
    intro_frames = create_intro_slide(
        case,
        max_steps=args.max_steps,
        width=1920,
        height=1080,
        duration_sec=3.0,
        fps=args.fps,
        main_title="DAGGER ITERATION 0 EVALUATION INSPECTION",
        subtitle="Autonomous Recovery Rollout (BEV + Waypoints + Camera)",
        accent_color=(80, 220, 120),
    )
    for f in intro_frames:
        out_writer.write(f)

    try:
        last_frame = None
        for step in range(args.max_steps):
            curr_pos = agent.get_state().position
            curr_dist = ust.find_shortest_path(sim, curr_pos, case.goal_position)[0]
            final_dist = curr_dist

            if curr_dist <= threshold_dist:
                final_status = "SUCCESS"
                logger.info(f"🎯 Goal reached at step {step}! (dist={curr_dist:.2f}m)")
                break

            obs = sim.get_sensor_observations()
            display_img, depth, semantic = utils.split_observations(obs)

            # Ground-truth geodesic distance inputs
            _, plsDict, _ = ust.get_pathlength_GT(
                sim, agent, depth, semantic, case.goal_position, None
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

            # Model prediction
            v, w, _ = controller.predict(display_img, control_input, visualize=False)
            pred_waypoints = controller.action_pred
            goal_mask_vis = getattr(controller, "goal_mask_vis", None)

            # Step agent
            agent, sim, _ = utils.apply_velocity(
                vel_control=vel_control,
                agent=agent,
                sim=sim,
                velocity=v,
                steer=-w,
                time_step=sim_cfg.get("time_delta", 0.25),
            )

            # Draw step on BEV map
            vis.draw_infer_step(agent.get_state())

            # Render Waypoint Curve and Viridis Costmap
            wp_plot_img = render_waypoint_curve(pred_waypoints, width=512, height=440)
            costmap_img = render_viridis_costmap(goal_mask_vis, width=512, height=440)

            # Determine dynamic status
            delta_d = init_dist - curr_dist
            status_text = "SUCCESS" if curr_dist <= threshold_dist else ("ADVANCING" if delta_d > 0.5 else "SPINNING / SEARCHING")

            frame = build_inspection_frame(
                bev_map=vis.tdv,
                rgb_img=display_img,
                wp_plot_img=wp_plot_img,
                costmap_img=costmap_img,
                step=step + 1,
                max_steps=args.max_steps,
                curr_dist=curr_dist,
                init_dist=init_dist,
                v=v,
                w=-w,
                target_name=target_name,
                status_text=status_text,
            )
            out_writer.write(frame)
            last_frame = frame

        # Append final badge frames (2.0s)
        if last_frame is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            completion_text = f"EVALUATION COMPLETE: {final_status} (Net Closed: {init_dist - final_dist:+.2f}m)"
            b_color = (60, 220, 100) if final_status == "SUCCESS" else (80, 180, 240)
            cv2.rectangle(last_frame, (20, 1080 - 65), (1920 - 20, 1080 - 15), (20, 24, 32), -1)
            cv2.rectangle(last_frame, (20, 1080 - 65), (1920 - 20, 1080 - 15), b_color, 2)
            cv2.putText(last_frame, completion_text, (40, 1080 - 28), font, 0.82, b_color, 2, cv2.LINE_AA)
            for _ in range(int(2.0 * args.fps)):
                out_writer.write(last_frame)

    finally:
        out_writer.release()
        sim.close()

    logger.info(f"✅ Successfully saved Policy Inspection Video to: {output_path.resolve()}")
    return True


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
        logger.info(f"INSPECTING EPISODE {idx + 1}/{len(cases)}: {case.episode_name}")
        logger.info("=" * 80)

        if len(cases) == 1 and args.output_name:
            out_name = args.output_name
        else:
            out_name = f"inspection_{case.episode_name}.mp4"

        out_path = out_dir / out_name
        try:
            success = run_inspection_rollout(case, cfg, args, out_path)
            if success and out_path.exists():
                generated_videos.append(out_path)
        except Exception as e:
            logger.error(f"Error during rollout for {case.episode_name}: {e}", exc_info=True)
            continue

    # Automatic concatenation across all evaluated episodes into a single compilation video
    if args.concat and len(generated_videos) > 1:
        default_concat_name = f"dagger_iter0_inspection_compilation_top{len(generated_videos)}.mp4"
        concat_file = out_dir / (args.output_name if args.output_name else default_concat_name)
        logger.info(f"Concatenating {len(generated_videos)} inspection videos into: {concat_file.name}")
        concatenate_videos(generated_videos, concat_file, fps=args.fps)
        logger.info(f"✅ Master inspection compilation saved to:\n--> {concat_file.resolve()}")


if __name__ == "__main__":
    main()
