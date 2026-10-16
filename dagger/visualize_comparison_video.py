#!/usr/bin/env python3
"""
4-Way Comparison Video Generator:
Compares Behavioral Cloning vs DAgger Iterations (Iter 0, Iter 1, Iter 2)
on failure recovery cases in Habitat-Sim.

Tiles all 4 policies in a synchronized 2x2 grid with:
- Top-down BEV path maps
- 3-panel policy view (Waypoint plot, RGB camera, Viridis goal mask)
- Real-time step counter, distance to goal, and success badges
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import quaternion
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[DAggerComparisonVideo]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import habitat_sim
from libs.common import utils
from libs.common import utils_sim_traj as ust
from libs.logger.visualizer import Visualizer
from libs.control.objectreact import ObjRelLearntController
from dagger.core.failure_loader import FailureLoader, FailureCase, find_scene_glb


POLICIES = [
    {
        "id": "bc",
        "title": "Baseline B: Behavioral Cloning (β=1.0)",
        "ckpt": "dagger/checkpoints/policy_iter_-1.pth",
        "color": (50, 120, 240),  # Amber/Orange
    },
    {
        "id": "iter0",
        "title": "DAgger Iteration 0 (β=0.5)",
        "ckpt": "dagger/checkpoints/policy_iter_0.pth",
        "color": (230, 180, 50),  # Cyan
    },
    {
        "id": "iter1",
        "title": "DAgger Iteration 1 (β=0.0)",
        "ckpt": "dagger/checkpoints/policy_iter_1.pth",
        "color": (80, 200, 120),  # Emerald Green
    },
    {
        "id": "iter2",
        "title": "DAgger Iteration 2 (Refined)",
        "ckpt": "dagger/checkpoints/policy_iter_2.pth",
        "color": (200, 100, 220), # Purple
    },
]


def create_intro_slide(
    case: FailureCase,
    max_steps: int = 300,
    width: int = 1280,
    height: int = 720,
    duration_sec: float = 3.0,
    fps: int = 15,
) -> List[np.ndarray]:
    slide = np.zeros((height, width, 3), dtype=np.uint8)
    # Background subtle gradient
    for y in range(height):
        ratio = y / height
        slide[y, :] = np.array([20 + 20 * ratio, 22 + 20 * ratio, 32 + 30 * ratio], dtype=np.uint8)

    # Accent top border
    slide[0:8, :] = [240, 140, 50]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(slide, "FAILURE RECOVERY BENCHMARK", (80, 110), font, 1.3, (240, 240, 240), 3, cv2.LINE_AA)
    cv2.putText(slide, "4-Way Comparison: Behavioral Cloning vs. DAgger Iterations", (80, 160), font, 0.85, (100, 220, 255), 2, cv2.LINE_AA)
    cv2.line(slide, (80, 190), (width - 80, 190), (70, 80, 100), 2)

    fields = [
        ("Episode Name:", f"{case.episode_name}"),
        ("Scene ID:", f"{case.scene_id}"),
        ("Target Object:", f"{case.metadata.get('target', 'Target Goal')}"),
        ("Initial Dist to Goal:", f"{case.initial_goal_dist:.2f} m"),
        ("Max Step Budget:", f"{max_steps} steps"),
    ]

    y_pos = 260
    for label, val in fields:
        cv2.putText(slide, label, (100, y_pos), font, 0.75, (160, 170, 185), 2, cv2.LINE_AA)
        cv2.putText(slide, val, (380, y_pos), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos += 50

    # Policy legend boxes
    y_pos += 20
    cv2.putText(slide, "Comparing Models:", (100, y_pos), font, 0.75, (160, 170, 185), 2, cv2.LINE_AA)
    y_pos += 45
    box_w = (width - 200 - 3 * 20) // 4
    for idx, p in enumerate(POLICIES):
        x = 100 + idx * (box_w + 20)
        cv2.rectangle(slide, (x, y_pos), (x + box_w, y_pos + 60), (35, 40, 50), -1)
        cv2.rectangle(slide, (x, y_pos), (x + box_w, y_pos + 60), p["color"], 2)
        cv2.putText(slide, p["id"].upper(), (x + 15, y_pos + 26), font, 0.6, p["color"], 2, cv2.LINE_AA)
        cv2.putText(slide, p["title"].split(":")[0][:16], (x + 15, y_pos + 48), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    num_frames = int(duration_sec * fps)
    return [slide.copy() for _ in range(num_frames)]


def run_policy_rollout(
    policy_cfg: Dict,
    case: FailureCase,
    cfg: Dict,
    max_steps: int = 300,
) -> Tuple[List[np.ndarray], Dict]:
    """Runs a single policy autonomous recovery rollout and returns recorded visual frames."""
    sim_cfg = cfg["simulation"]
    threshold_dist = sim_cfg["threshold_goal_distance"]
    hm3d_root = REPO_ROOT / cfg["paths"]["hm3d_scenes_path"]

    scene_glb = find_scene_glb(hm3d_root, case.scene_id)
    if scene_glb is None or not scene_glb.exists():
        logger.error(f"Scene mesh not found for {case.scene_id}")
        return [], {}

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

    # Preload Controller for this policy
    ctrl_cfg_path = REPO_ROOT / cfg["paths"]["controller_config"]
    with open(ctrl_cfg_path, "r") as f:
        ctrl_cfg = yaml.safe_load(f)

    ckpt_path = REPO_ROOT / policy_cfg["ckpt"]
    ctrl_cfg["load_run"] = str(ckpt_path)
    controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")

    frames = []
    final_status = "TIMEOUT"
    final_dist = case.initial_goal_dist
    steps_taken = 0

    try:
        # Spawn agent at failure pose
        start_state = agent.get_state()
        start_state.position = case.failure_position
        q = case.failure_rotation
        start_state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
        agent.set_state(start_state)

        # Setup Visualizer (draws full uncropped topdown map)
        vis = Visualizer(sim, agent, str(scene_glb), env="sim")
        vis.create_top_down_map(height=case.failure_position[1])
        vis.draw_goal(vis.sim_to_tdv(case.goal_position))
        vis.draw_start(vis.sim_to_tdv(case.failure_position))

        for step in range(max_steps):
            steps_taken = step + 1
            curr_pos = agent.get_state().position
            curr_dist = ust.find_shortest_path(sim, curr_pos, case.goal_position)[0]
            final_dist = curr_dist

            if curr_dist <= threshold_dist:
                final_status = "SUCCESS"
                logger.info(f"[{policy_cfg['id']}] Reached goal at step {step} (dist={curr_dist:.2f}m)")
                break

            obs = sim.get_sensor_observations()
            display_img, depth, semantic = utils.split_observations(obs)

            # Compute ground-truth geodesic pathlengths for all visible semantic instances
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

            v, w, vis_img = controller.predict(display_img, control_input, visualize=True)

            # Step agent
            agent, sim, _ = utils.apply_velocity(
                vel_control=vel_control,
                agent=agent,
                sim=sim,
                velocity=v,
                steer=-w,
                time_step=sim_cfg["time_delta"],
            )

            # Draw live step marker on top-down map
            vis.draw_infer_step(agent.get_state())

            if vis_img is None:
                vis_img = display_img

            # Stack topdown map + 3-panel policy view
            ratio = vis_img.shape[1] / max(vis.tdv.shape[1], 1)
            tdv_resized = cv2.resize(vis.tdv, dsize=None, fx=ratio, fy=ratio)
            combined = np.concatenate((tdv_resized, vis_img), axis=0)

            # Add header overlay banner
            h_comb, w_comb = combined.shape[:2]
            header = np.zeros((46, w_comb, 3), dtype=np.uint8)
            header[:] = (20, 22, 28)
            header[0:3, :] = policy_cfg["color"]

            font = cv2.FONT_HERSHEY_SIMPLEX
            # Policy title
            cv2.putText(header, policy_cfg["title"], (10, 20), font, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
            # Distance counter & step
            stat_text = f"Step: {step:03d} | Dist: {curr_dist:.2f}m"
            cv2.putText(header, stat_text, (10, 38), font, 0.42, (100, 220, 255), 1, cv2.LINE_AA)

            frame_with_header = np.concatenate((header, combined), axis=0)
            frames.append(frame_with_header)

    except Exception as e:
        logger.error(f"Error during {policy_cfg['id']} rollout: {e}")
    finally:
        if sim:
            try:
                sim.close()
            except Exception:
                pass

    meta = {
        "status": final_status,
        "final_dist": float(final_dist),
        "steps": steps_taken,
    }
    return frames, meta


def letterbox_to_quadrant(frame: np.ndarray, quad_w: int, quad_h: int) -> np.ndarray:
    """Resizes frame into quadrant maintaining aspect ratio with black borders."""
    h, w = frame.shape[:2]
    scale = min(quad_w / w, quad_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)
    top = (quad_h - nh) // 2
    left = (quad_w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def generate_comparison_video(
    case: FailureCase,
    cfg: Dict,
    output_path: Path,
    max_steps: int = 300,
    fps: int = 12,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating 4-Way Comparison for: {case.episode_name} ({case.scene_id})")

    all_policy_frames = []
    all_policy_meta = []

    # Run rollouts for all 4 policies
    for p in POLICIES:
        logger.info(f"Running Rollout for: {p['title']}...")
        frames, meta = run_policy_rollout(p, case, cfg, max_steps=max_steps)
        all_policy_frames.append(frames)
        all_policy_meta.append(meta)

    # Determine maximum duration across policies
    max_frames = max([len(f) for f in all_policy_frames] + [1])
    if max_frames == 1 and len(all_policy_frames[0]) == 0:
        logger.error(f"No frames collected for {case.episode_name}")
        return

    # Pad shorter runs with their final frame and mark completion badge
    for idx, (p, frames, meta) in enumerate(zip(POLICIES, all_policy_frames, all_policy_meta)):
        if not frames:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Execution Failed", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            all_policy_frames[idx] = [blank] * max_frames
            continue

        last_frame = frames[-1].copy()
        # Overlay completion badge on last frame
        badge_text = f"FINISHED: {meta['status']} (Dist: {meta['final_dist']:.2f}m)"
        b_color = (40, 200, 80) if meta["status"] == "SUCCESS" else (50, 50, 220)
        cv2.rectangle(last_frame, (5, last_frame.shape[0] - 35), (last_frame.shape[1] - 5, last_frame.shape[0] - 5), (20, 20, 25), -1)
        cv2.putText(last_frame, badge_text, (15, last_frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, b_color, 2, cv2.LINE_AA)

        while len(frames) < max_frames:
            frames.append(last_frame)

    # Target output dimensions (1080p: 1920x1080, quadrants are 960x540)
    out_w, out_h = 1920, 1080
    quad_w, quad_h = out_w // 2, out_h // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    # 1. Write intro title card
    intro_frames = create_intro_slide(case, max_steps=max_steps, width=out_w, height=out_h, duration_sec=3.0, fps=fps)
    for f in intro_frames:
        out_writer.write(f)

    # 2. Write synchronized 2x2 grid frames
    for t in range(max_frames):
        tl = letterbox_to_quadrant(all_policy_frames[0][t], quad_w, quad_h)  # BC
        tr = letterbox_to_quadrant(all_policy_frames[1][t], quad_w, quad_h)  # Iter 0
        bl = letterbox_to_quadrant(all_policy_frames[2][t], quad_w, quad_h)  # Iter 1
        br = letterbox_to_quadrant(all_policy_frames[3][t], quad_w, quad_h)  # Iter 2

        # Grid dividers
        top_row = np.hstack((tl, tr))
        bot_row = np.hstack((bl, br))
        grid = np.vstack((top_row, bot_row))

        # Thin divider lines
        cv2.line(grid, (quad_w, 0), (quad_w, out_h), (50, 55, 65), 3)
        cv2.line(grid, (0, quad_h), (out_w, quad_h), (50, 55, 65), 3)

        out_writer.write(grid)

    out_writer.release()
    logger.info(f"✅ Saved 4-Way Comparison Video to: {output_path}")


def concatenate_videos(video_paths: List[Path], output_path: Path, fps: int = 12):
    """Concatenates multiple comparison video files into a single continuous video."""
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
        logger.info(f"✅ Saved Concatenated Comparison Video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 4-Way Comparison Videos for DAgger")
    parser.add_argument("--config", default="dagger/config.yaml")
    parser.add_argument("--output_dir", default="dagger/output/comparison_videos")
    parser.add_argument("--case_idx", type=int, default=0, help="Test case index (from held-out split)")
    parser.add_argument("--num_cases", type=int, default=1, help="Number of comparison videos to generate")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per rollout")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--concat", action="store_true", default=True, help="Generate concatenated video when num_cases > 1 (default: True)")
    parser.add_argument("--no_concat", dest="concat", action="store_false", help="Disable concatenated video generation")
    args = parser.parse_args()

    with open(REPO_ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    loader = FailureLoader(str(REPO_ROOT / args.config))
    all_cases = loader.load_from_csv()
    if not all_cases:
        logger.error("Failed to load failure cases from CSV!")
        return

    train_c, val_c, test_cases = loader.split_cases(all_cases)
    logger.info(f"Loaded failure cases: {len(test_cases)} held-out test cases.")

    selected_cases = test_cases[args.case_idx : args.case_idx + args.num_cases]
    out_dir = REPO_ROOT / args.output_dir
    generated_videos = []

    for case in selected_cases:
        out_file = out_dir / f"compare_{case.episode_name}.mp4"
        generate_comparison_video(
            case=case,
            cfg=cfg,
            output_path=out_file,
            max_steps=args.max_steps,
            fps=args.fps,
        )
        if out_file.exists():
            generated_videos.append(out_file)

    # Automatically create concatenated compilation video when multiple episodes are evaluated
    if args.concat and len(generated_videos) > 1:
        start_idx = args.case_idx
        end_idx = args.case_idx + len(generated_videos) - 1
        concat_file = out_dir / f"concat_episodes_{start_idx:02d}_to_{end_idx:02d}.mp4"
        concatenate_videos(generated_videos, concat_file, fps=args.fps)


if __name__ == "__main__":
    main()
