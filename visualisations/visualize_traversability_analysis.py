#!/usr/bin/env python3
"""
Traversability Analysis Visualizer for ObjectReact & DAgger Navigation.

Visualizes what the agent/perception system identifies as traversable vs non-traversable
across failure episodes during autonomous policy recovery rollouts.

For each episode, renders a 1080p (1920x1080) visual dashboard featuring:
  1. Global Top-Down BEV Map (uncropped, complete floorplan with agent trail & goal)
  2. First-Person RGB Camera with Traversability Overlay & Predicted Waypoints
  3. Local Waypoint Trajectory Curve with distance coordinate grid
  4. Viridis Goal Distance Cost Map (geodesic distance gradient to target)
  5. High-Contrast Binary Traversability Mask (White = Floor, Black = Obstacles/Walls)

Supports evaluating N failure cases starting from any index, saving both individual
episode videos and automatically concatenated compilation videos.

Usage:
  pixi run python visualisations/visualize_traversability_analysis.py --case_idx 0 --num_cases 5 --max_steps 300
"""

import argparse
from dataclasses import dataclass
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Set

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import habitat_sim
from habitat_sim.utils.common import quat_to_magnum, quat_from_magnum

from dagger.core.failure_loader import FailureCase, FailureLoader, find_scene_glb
from libs.common import utils, utils_sim_traj as ust
from libs.logger.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[TraversabilityAnalysis]")

TRAVERSABLE_CATEGORIES: Set[str] = {
    "floor", "flooring", "floor mat", "floor vent", "carpet",
    "mat", "rug", "doormat", "shower floor", "pavement", "ground", "tiles"
}


def get_floor_instance_ids(semantic_scene) -> Set[int]:
    """Extracts all instance IDs classified as ground/floor/carpet in HM3D."""
    floor_ids = set()
    if not semantic_scene or not hasattr(semantic_scene, "objects"):
        return floor_ids

    for obj in semantic_scene.objects:
        if obj.category:
            cname = obj.category.name().lower()
            if any(t in cname for t in TRAVERSABLE_CATEGORIES):
                try:
                    floor_ids.add(int(obj.id.split("_")[-1]))
                except Exception:
                    pass
    return floor_ids


def get_target_name(case: FailureCase) -> str:
    """Extracts human-readable target name from case metadata or episode string."""
    target = case.metadata.get("target")
    if target and target != "Target Goal":
        return target
    parts = case.episode_name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[2:-1])
    return "Target Goal"


def create_intro_slide(
    case: FailureCase,
    max_steps: int = 300,
    width: int = 1920,
    height: int = 1080,
    duration_sec: float = 3.0,
    fps: int = 12,
) -> List[np.ndarray]:
    """Generates an HD title card with metadata."""
    slide = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / height
        slide[y, :] = np.array([22 + 18 * ratio, 26 + 22 * ratio, 35 + 28 * ratio], dtype=np.uint8)

    # Accent top border (emerald green for traversability)
    slide[0:8, :] = [80, 220, 120]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(slide, "TRAVERSABILITY & PERCEPTION ANALYSIS", (100, 130), font, 1.35, (245, 245, 245), 3, cv2.LINE_AA)
    cv2.putText(slide, "ObjectReact / DAgger Failure Case Inspection", (100, 185), font, 0.9, (80, 220, 120), 2, cv2.LINE_AA)
    cv2.line(slide, (100, 215), (width - 100, 215), (70, 80, 100), 2)

    target_name = get_target_name(case)
    fields = [
        ("Episode Name:", f"{case.episode_name}"),
        ("Scene ID:", f"{case.scene_id}"),
        ("Target Goal Object:", f"{target_name.upper()}"),
        ("Initial Dist to Goal:", f"{case.initial_goal_dist:.2f} meters"),
        ("Step Budget:", f"{max_steps} steps"),
        ("Analysis Panels:", "Top-Down BEV | RGB + Traversability | Waypoints | Viridis Cost Map | Binary Mask"),
    ]

    y_pos = 295
    for label, val in fields:
        cv2.putText(slide, label, (120, y_pos), font, 0.8, (160, 170, 185), 2, cv2.LINE_AA)
        val_color = (100, 230, 150) if "Target" in label or "Analysis" in label else (255, 255, 255)
        cv2.putText(slide, val, (440, y_pos), font, 0.85, val_color, 2, cv2.LINE_AA)
        y_pos += 60

    num_frames = int(duration_sec * fps)
    return [slide.copy() for _ in range(num_frames)]


def render_waypoint_plot(pred_waypoints: np.ndarray, width: int = 560, height: int = 490) -> np.ndarray:
    """Renders the local waypoint trajectory curve with clean coordinate axes."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#16181f")
    ax.set_facecolor("#16181f")

    # Plot waypoints
    ax.plot(
        pred_waypoints[:, 1],
        pred_waypoints[:, 0],
        color="#00e5ff",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="#ffea00",
        markeredgecolor="#ffffff",
        markeredgewidth=1.2,
        label="Waypoints",
    )

    # Robot origin marker
    ax.scatter([0], [0], color="#ff3366", s=80, marker="^", label="Robot Pose", zorder=5)

    ax.set_xlim(4, -4)  # Inverted x-axis matches robot forward coordinate system
    ax.set_ylim(-1, 10)
    ax.set_xlabel("Lateral Offset (m)", color="#a0abbd", fontsize=9)
    ax.set_ylabel("Forward Distance (m)", color="#a0abbd", fontsize=9)
    ax.set_title("Predicted Waypoints (Local Frame)", color="#ffffff", fontsize=11, fontweight="bold", pad=8)

    ax.tick_params(colors="#808b96", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#3b4252")
    ax.grid(True, color="#2e3440", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.canvas.draw()
    try:
        plot_img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    except AttributeError:
        plot_img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    return cv2.resize(plot_img, (width, height))


def render_viridis_costmap(goal_mask_vis: Optional[np.ndarray], width: int = 560, height: int = 490) -> np.ndarray:
    """Renders the Viridis goal distance cost map with metric scale colorbar."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#16181f")
    ax.set_facecolor("#16181f")

    if goal_mask_vis is not None and np.any(goal_mask_vis > 0):
        im = ax.imshow(goal_mask_vis, cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#a0abbd", labelsize=8)
        for spine in cbar.ax.spines.values():
            spine.set_color("#3b4252")
        cbar.set_label("Geodesic Distance / Cost to Goal", color="#a0abbd", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No Object Distance Map Available", color="#808b96", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Viridis Cost Map (Yellow = Target Goal)", color="#ffffff", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    try:
        plot_img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    except AttributeError:
        plot_img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    return cv2.resize(plot_img, (width, height))


def letterbox_to_dims(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scales image to fit inside target box with aspect-ratio preservation."""
    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def build_traversability_dashboard(
    bev_map: np.ndarray,
    rgb_img: np.ndarray,
    traversable_mask: np.ndarray,
    goal_mask_vis: Optional[np.ndarray],
    pred_waypoints: np.ndarray,
    step: int,
    max_steps: int,
    curr_dist: float,
    target_name: str,
    status_text: str = "NAVIGATING",
) -> np.ndarray:
    """
    Assembles a full 1080p (1920x1080) synchronized inspection dashboard:
      - Left side (800x1080): Header telemetry + Uncropped Global BEV Floorplan
      - Right side (1120x1080): 2x2 High-Resolution Inspection Grid:
          [RGB + Traversability Overlay]  [Waypoint Trajectory Curve]
          [Viridis Goal Distance Mask]    [High-Contrast Binary Mask]
    """
    total_w, total_h = 1920, 1080
    dashboard = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    dashboard[:] = (18, 20, 26)  # Dark slate background

    # ==========================================
    # 1. LEFT SIDE: BEV Floorplan & Header
    # ==========================================
    left_w = 800
    left_panel = np.zeros((total_h, left_w, 3), dtype=np.uint8)
    left_panel[:] = (18, 20, 26)

    # Top Header Banner
    header_h = 95
    cv2.rectangle(left_panel, (15, 12), (left_w - 15, header_h), (28, 32, 42), -1)
    cv2.rectangle(left_panel, (15, 12), (left_w - 15, header_h), (55, 65, 85), 1)
    cv2.line(left_panel, (15, 12), (left_w - 15, 12), (80, 220, 120), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(left_panel, f"TARGET: {target_name.upper()}", (30, 42), font, 0.75, (100, 230, 160), 2, cv2.LINE_AA)
    badge_color = (40, 200, 80) if status_text == "SUCCESS" else ((60, 60, 220) if "EXCEED" in status_text else (230, 160, 40))
    cv2.putText(left_panel, f"[{status_text}]", (left_w - 180, 42), font, 0.65, badge_color, 2, cv2.LINE_AA)

    # Telemetry Line: Step, Dist, % Traversable
    floor_pct = float(traversable_mask.mean() * 100) if traversable_mask is not None else 0.0
    telemetry = f"Step: {step:03d}/{max_steps}  |  Goal Dist: {curr_dist:.2f}m  |  Floor in View: {floor_pct:.1f}%"
    cv2.putText(left_panel, telemetry, (30, 78), font, 0.58, (220, 225, 235), 1, cv2.LINE_AA)

    # BEV Map placement
    bev_area_h = total_h - header_h - 30
    bev_letterboxed = letterbox_to_dims(bev_map, left_w - 30, bev_area_h)
    left_panel[header_h + 15 : header_h + 15 + bev_area_h, 15 : left_w - 15] = bev_letterboxed
    dashboard[:, :left_w] = left_panel

    # Vertical Divider Line
    cv2.line(dashboard, (left_w, 0), (left_w, total_h), (50, 58, 75), 2)

    # ==========================================
    # 2. RIGHT SIDE: 2x2 Inspection Grid
    # ==========================================
    right_w = total_w - left_w  # 1120 px
    quad_w = (right_w - 30) // 2  # ~545 px
    quad_h = (total_h - 30) // 2  # ~525 px

    # --- Quadrant 1 (Top-Left): RGB + Traversability Overlay ---
    rgb_vis = rgb_img.copy()
    if traversable_mask is not None:
        # Create emerald green overlay on floor pixels
        t_mask_resized = cv2.resize(traversable_mask.astype(np.uint8), (rgb_vis.shape[1], rgb_vis.shape[0])) > 0
        overlay = rgb_vis.copy()
        overlay[t_mask_resized] = [30, 220, 100]  # Vibrant green
        cv2.addWeighted(overlay, 0.40, rgb_vis, 0.60, 0, rgb_vis)

    # Draw waypoint steer arrow HUD on RGB
    if pred_waypoints is not None and len(pred_waypoints) > 0:
        h_rgb, w_rgb = rgb_vis.shape[:2]
        center_x, center_y = w_rgb // 2, h_rgb - 20
        target_wp = pred_waypoints[0]
        heading = math.atan2(target_wp[1], max(target_wp[0], 0.05))
        arrow_len = 32
        end_x = int(center_x - arrow_len * math.sin(heading))
        end_y = int(center_y - arrow_len * math.cos(heading))
        cv2.arrowedLine(rgb_vis, (center_x, center_y), (end_x, end_y), (0, 235, 255), 3, tipLength=0.35)
        cv2.circle(rgb_vis, (center_x, center_y), 3, (255, 255, 255), -1)

    rgb_letterbox = letterbox_to_dims(rgb_vis, quad_w, quad_h - 40)
    q1 = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    q1[:] = (24, 27, 36)
    # Header badge
    cv2.putText(q1, "1. RGB CAMERA + TRAVERSABLE FLOOR OVERLAY", (15, 26), font, 0.52, (100, 230, 160), 1, cv2.LINE_AA)
    q1[35 : 35 + rgb_letterbox.shape[0], :] = rgb_letterbox

    # --- Quadrant 2 (Top-Right): Waypoint Trajectory Curve ---
    wp_plot = render_waypoint_plot(pred_waypoints, width=quad_w, height=quad_h - 40)
    q2 = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    q2[:] = (24, 27, 36)
    cv2.putText(q2, "2. LOCAL WAYPOINT TRAJECTORY (METRIC GRID)", (15, 26), font, 0.52, (80, 210, 255), 1, cv2.LINE_AA)
    q2[35 : 35 + wp_plot.shape[0], :] = wp_plot

    # --- Quadrant 3 (Bottom-Left): Viridis Goal Distance Cost Map ---
    vir_plot = render_viridis_costmap(goal_mask_vis, width=quad_w, height=quad_h - 40)
    q3 = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    q3[:] = (24, 27, 36)
    cv2.putText(q3, "3. VIRIDIS GOAL DISTANCE COST MAP", (15, 26), font, 0.52, (255, 225, 80), 1, cv2.LINE_AA)
    q3[35 : 35 + vir_plot.shape[0], :] = vir_plot

    # --- Quadrant 4 (Bottom-Right): High-Contrast Binary Mask ---
    if traversable_mask is not None:
        bin_img = (traversable_mask.astype(np.uint8) * 255)
        bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    else:
        bin_bgr = np.zeros((240, 320, 3), dtype=np.uint8)

    # Add corridor clearance gauge
    # Check lower central corridor (pixels directly in front of wheels)
    h_b, w_b = bin_bgr.shape[:2]
    corridor = bin_bgr[int(h_b * 0.70) : int(h_b * 0.95), int(w_b * 0.40) : int(w_b * 0.60)]
    corridor_clear = corridor.mean() > 180

    bin_letterbox = letterbox_to_dims(bin_bgr, quad_w, quad_h - 40)
    q4 = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    q4[:] = (24, 27, 36)
    gauge_txt = "CORRIDOR: CLEAR" if corridor_clear else "CORRIDOR: BLOCKED/WALL"
    g_col = (40, 220, 100) if corridor_clear else (50, 50, 240)
    cv2.putText(q4, f"4. BINARY TRAVERSABILITY MASK ({gauge_txt})", (15, 26), font, 0.52, g_col, 1, cv2.LINE_AA)
    q4[35 : 35 + bin_letterbox.shape[0], :] = bin_letterbox

    # Place the 4 quadrants on the right side
    r_x1 = left_w + 10
    r_x2 = left_w + 10 + quad_w + 10
    dashboard[10 : 10 + quad_h, r_x1 : r_x1 + quad_w] = q1
    dashboard[10 : 10 + quad_h, r_x2 : r_x2 + quad_w] = q2
    dashboard[10 + quad_h + 10 : 10 + quad_h + 10 + quad_h, r_x1 : r_x1 + quad_w] = q3
    dashboard[10 + quad_h + 10 : 10 + quad_h + 10 + quad_h, r_x2 : r_x2 + quad_w] = q4

    # Grid divider borders on right side
    cv2.line(dashboard, (left_w + 10 + quad_w + 5, 10), (left_w + 10 + quad_w + 5, total_h - 10), (50, 58, 75), 2)
    cv2.line(dashboard, (left_w + 10, 10 + quad_h + 5), (total_w - 10, 10 + quad_h + 5), (50, 58, 75), 2)

    return dashboard


def run_traversability_rollout(
    case: FailureCase,
    cfg: Dict,
    checkpoint_path: Path,
    output_path: Path,
    max_steps: int = 300,
    fps: int = 12,
) -> bool:
    """Executes a recovery rollout while recording the 1080p Traversability Dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim_cfg = cfg["simulation"]
    threshold_dist = sim_cfg["threshold_goal_distance"]
    hm3d_root = REPO_ROOT / cfg["paths"]["hm3d_scenes_path"]

    scene_glb = find_scene_glb(hm3d_root, case.scene_id)
    if scene_glb is None or not scene_glb.exists():
        logger.error(f"Scene GLB not found for {case.scene_id}")
        return False

    # Initialize Simulator
    sim, agent, _ = utils.get_sim_agent(
        str(scene_glb),
        updateNavMesh=False,
        width=sim_cfg["width"],
        height=sim_cfg["height"],
        hfov=sim_cfg["hfov"],
        sensor_height=sim_cfg["sensor_height"],
    )

    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.controlling_ang_vel = True
    vel_control.ang_vel_is_local = True

    # Preload Controller
    from libs.control.objectreact import ObjRelLearntController
    ctrl_cfg_path = REPO_ROOT / cfg["paths"]["controller_config"]
    with open(ctrl_cfg_path, "r") as f:
        ctrl_cfg = yaml.safe_load(f)

    ctrl_cfg["load_run"] = str(checkpoint_path)
    controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")

    target_name = get_target_name(case)
    logger.info(f"Analyzing Traversability for: {case.episode_name} ({case.scene_id}) -> {target_name}")

    # Set exact failure start pose
    start_state = agent.get_state()
    start_state.position = case.failure_position
    q = case.failure_rotation
    start_state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
    agent.set_state(start_state)

    # Cache all floor instance IDs in this scene
    floor_instance_ids = get_floor_instance_ids(sim.semantic_scene)
    logger.info(f"Loaded {len(floor_instance_ids)} floor/ground instance IDs from HM3D semantic scene.")

    # Top-Down BEV Visualizer
    vis = Visualizer(sim, agent, str(scene_glb), env="sim")
    vis.create_top_down_map(height=case.failure_position[1])
    vis.draw_goal(vis.sim_to_tdv(case.goal_position))
    vis.draw_start(vis.sim_to_tdv(case.failure_position))

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (1920, 1080))

    # 1. Write title slide
    intro_frames = create_intro_slide(case, max_steps=max_steps, width=1920, height=1080, duration_sec=2.5, fps=fps)
    for f in intro_frames:
        out_writer.write(f)

    final_status = "EXCEEDED_STEPS"
    final_dist = case.initial_goal_dist

    try:
        for step in range(max_steps):
            curr_pos = agent.get_state().position
            curr_dist = ust.find_shortest_path(sim, curr_pos, case.goal_position)[0]
            final_dist = curr_dist

            if curr_dist <= threshold_dist:
                final_status = "SUCCESS"
                logger.info(f"✅ Reached goal at step {step} (dist={curr_dist:.2f}m)")
                break

            obs = sim.get_sensor_observations()
            display_img, depth, semantic = utils.split_observations(obs)

            # Ground-truth geodesic pathlengths for semantic objects
            _, plsDict, goal_mask_2d = ust.get_pathlength_GT(
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
            v, w, vis_img = controller.predict(display_img, control_input, visualize=True)
            pred_waypoints = controller.action_pred

            # Step agent
            agent, sim, _ = utils.apply_velocity(
                vel_control=vel_control,
                agent=agent,
                sim=sim,
                velocity=v,
                steer=-w,
                time_step=sim_cfg["time_delta"],
            )

            # Update BEV trail
            vis.draw_infer_step(agent.get_state())

            # Binary traversability mask (semantic floor IDs)
            traversable_mask = np.isin(semantic, list(floor_instance_ids))

            # Goal distance cost map from controller
            goal_mask_vis = getattr(controller, "goal_mask_vis", None)

            # Assemble full 1080p dashboard
            status_text = "SUCCESS" if final_status == "SUCCESS" else "NAVIGATING"
            frame = build_traversability_dashboard(
                bev_map=vis.tdv,
                rgb_img=display_img,
                traversable_mask=traversable_mask,
                goal_mask_vis=goal_mask_vis,
                pred_waypoints=pred_waypoints,
                step=step + 1,
                max_steps=max_steps,
                curr_dist=curr_dist,
                target_name=target_name,
                status_text=status_text,
            )
            out_writer.write(frame)

        # Append final completion badge frames (1.5 seconds)
        font = cv2.FONT_HERSHEY_SIMPLEX
        completion_text = f"FINISHED: {final_status} (Final Dist: {final_dist:.2f}m)"
        b_color = (40, 200, 80) if final_status == "SUCCESS" else (50, 50, 220)
        final_frame = frame.copy()
        cv2.rectangle(final_frame, (20, 1080 - 60), (1920 - 20, 1080 - 15), (20, 22, 28), -1)
        cv2.putText(final_frame, completion_text, (40, 1080 - 26), font, 0.85, b_color, 2, cv2.LINE_AA)
        for _ in range(int(1.5 * fps)):
            out_writer.write(final_frame)

    finally:
        out_writer.release()
        sim.close()

    logger.info(f"✅ Saved Traversability Video to: {output_path}")
    return True


def concatenate_videos(video_paths: List[Path], output_path: Path, fps: int = 12):
    """Concatenates multiple video files into a single continuous compilation."""
    valid_paths = [p for p in video_paths if p.exists() and p.stat().st_size > 0]
    if not valid_paths:
        logger.warning("No valid videos to concatenate.")
        return

    logger.info(f"Concatenating {len(valid_paths)} videos into {output_path.name}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = None

    for vp in valid_paths:
        cap = cv2.VideoCapture(str(vp))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if out_writer is None:
            out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_writer.write(frame)
        cap.release()

    if out_writer is not None:
        out_writer.release()
        logger.info(f"✅ Saved Concatenated Compilation Video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Traversability & Perception Analysis for Failure Cases")
    parser.add_argument("--config", default="dagger/config.yaml", help="Path to dagger config")
    parser.add_argument("--checkpoint", default="dagger/checkpoints/policy_iter_2.pth", help="Policy checkpoint")
    parser.add_argument("--output_dir", default="visualisations/traversability_analysis", help="Output directory")
    parser.add_argument("--case_idx", type=int, default=0, help="Starting failure case index")
    parser.add_argument("--num_cases", type=int, default=5, help="Number of failure cases to analyze")
    parser.add_argument("--max_steps", type=int, default=300, help="Max step budget per episode")
    parser.add_argument("--fps", type=int, default=12, help="Video framerate")
    parser.add_argument("--concat", action="store_true", default=True, help="Create concatenated video (default: True)")
    parser.add_argument("--no_concat", dest="concat", action="store_false", help="Disable concatenation")
    parser.add_argument("--gpu", type=str, default="1", help="CUDA GPU ID (default: 1 to avoid GPU 0 conflicts)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(REPO_ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    loader = FailureLoader(str(REPO_ROOT / args.config))
    all_cases = loader.load_from_csv()
    if not all_cases:
        logger.error("Failed to load failure cases from CSV!")
        return

    train_c, val_c, test_cases = loader.split_cases(all_cases)
    logger.info(f"Loaded {len(all_cases)} total failure cases ({len(test_cases)} held-out test cases).")

    # Select test cases
    selected_cases = test_cases[args.case_idx : args.case_idx + args.num_cases]
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = REPO_ROOT / args.checkpoint
    if not checkpoint_file.exists():
        logger.warning(f"Checkpoint {checkpoint_file} not found! Falling back to policy_iter_1.pth or latest.")
        cand = REPO_ROOT / "dagger/checkpoints/policy_iter_1.pth"
        if cand.exists():
            checkpoint_file = cand

    generated_videos = []
    for case in selected_cases:
        out_file = out_dir / f"traversability_{case.episode_name}.mp4"
        success = run_traversability_rollout(
            case=case,
            cfg=cfg,
            checkpoint_path=checkpoint_file,
            output_path=out_file,
            max_steps=args.max_steps,
            fps=args.fps,
        )
        if success and out_file.exists():
            generated_videos.append(out_file)

    if args.concat and len(generated_videos) > 1:
        start_idx = args.case_idx
        end_idx = args.case_idx + len(generated_videos) - 1
        concat_file = out_dir / f"concat_traversability_episodes_{start_idx:02d}_to_{end_idx:02d}.mp4"
        concatenate_videos(generated_videos, concat_file, fps=args.fps)


if __name__ == "__main__":
    main()
