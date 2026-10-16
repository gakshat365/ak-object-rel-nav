#!/usr/bin/env python3
"""
Creates video compilation with title cards for ObjectReact failure episodes.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[VideoCompiler]")

REPO_ROOT = Path(__file__).resolve().parent.parent


def create_title_slide(
    code: int,
    episode_name: str,
    target: str,
    scene: str,
    steps: str,
    final_distance: float,
    width: int = 1280,
    height: int = 720,
    duration_sec: float = 3.0,
    fps: int = 30,
) -> np.ndarray:
    """Creates a sleek title card slide."""
    slide = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(height):
        ratio = y / height
        color = np.array([20 + 25 * ratio, 20 + 20 * ratio, 28 + 35 * ratio])
        slide[y, :] = color.astype(np.uint8)

    # Accent top border
    slide[0:6, :] = [235, 140, 50]  # vibrant cyan/amber

    # Text rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Episode Code Pill
    code_text = f"FAILURE EPISODE #{code:03d}"
    cv2.putText(slide, code_text, (80, 140), font, 1.4, (240, 240, 240), 3, cv2.LINE_AA)
    
    # Sub-divider line
    cv2.line(slide, (80, 170), (width - 80, 170), (80, 85, 100), 2)

    # Details Grid
    labels = [
        ("Target Object:", f"{target.upper()}"),
        ("Episode Name:", f"{episode_name}"),
        ("Scene ID:", f"{scene}"),
        ("Failure Status:", "EXCEEDED STEPS (Timeout)"),
        ("Trajectory Steps:", f"{steps}"),
        ("Final Dist to Goal:", f"{final_distance:.2f} meters"),
    ]

    start_y = 250
    spacing = 65

    for i, (label, val) in enumerate(labels):
        y_pos = start_y + i * spacing
        # Label
        cv2.putText(slide, label, (80, y_pos), font, 0.9, (160, 165, 180), 2, cv2.LINE_AA)
        # Value
        val_color = (80, 220, 255) if "Target" in label else (230, 230, 230)
        if "Dist" in label:
            val_color = (100, 180, 255)
        cv2.putText(slide, val, (380, y_pos), font, 0.95, val_color, 2, cv2.LINE_AA)

    # Footer note
    footer = "ObjectReact Visual Navigation | Habitat-Sim Benchmark Replay"
    cv2.putText(slide, footer, (80, height - 50), font, 0.65, (110, 115, 130), 1, cv2.LINE_AA)

    num_frames = int(duration_sec * fps)
    return slide, num_frames


def letterbox_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize preserving aspect ratio with black borders (letterboxing)."""
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def compile_videos(
    mapping_json: str,
    output_path: str,
    max_episodes: int = 0,
    slide_duration: float = 3.0,
):
    with open(mapping_json, "r") as f:
        mapping = json.load(f)

    # Filter episodes that have videos
    valid_episodes = [m for m in mapping if m["video_available"]]
    logger.info(f"Loaded {len(mapping)} total episodes. {len(valid_episodes)} have repeat.mp4 videos.")

    if max_episodes > 0:
        valid_episodes = valid_episodes[:max_episodes]
        logger.info(f"Limiting to first {max_episodes} episodes for this compilation.")

    if not valid_episodes:
        logger.error("No valid video files found!")
        return

    # Probe first video to determine resolution and fps
    first_vid_path = REPO_ROOT / valid_episodes[0]["video_path"]
    cap = cv2.VideoCapture(str(first_vid_path))
    target_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if target_width == 0 or target_height == 0:
        target_width, target_height = 1280, 720
        fps = 30.0

    logger.info(f"Target Video Specification: {target_width}x{target_height} @ {fps:.1f} fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    for idx, ep in enumerate(valid_episodes, 1):
        code = ep["code"]
        target = ep["target"]
        ep_name = ep["episode_name"]
        scene = ep["scene"]
        steps = ep["steps"]
        final_dist = ep["final_distance"]
        vid_path = REPO_ROOT / ep["video_path"]

        logger.info(f"[{idx}/{len(valid_episodes)}] Adding Episode #{code} ({target}) from {vid_path.name}...")

        # 1. Write title slide
        slide_img, num_slide_frames = create_title_slide(
            code=code,
            episode_name=ep_name,
            target=target,
            scene=scene,
            steps=steps,
            final_distance=final_dist,
            width=target_width,
            height=target_height,
            duration_sec=slide_duration,
            fps=int(fps),
        )
        for _ in range(num_slide_frames):
            out_writer.write(slide_img)

        # 2. Append episode video frames
        cap = cv2.VideoCapture(str(vid_path))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = letterbox_frame(frame, target_width, target_height)
            
            # Optional overlay in top-right corner during replay
            overlay_text = f"Episode #{code:03d} | Target: {target}"
            cv2.putText(frame, overlay_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, overlay_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            
            out_writer.write(frame)
        cap.release()

    out_writer.release()
    logger.info(f"✅ Compilation finished! Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="visualisations/episode_mapping.json")
    parser.add_argument("--output", default="visualisations/failure_episodes_compilation.mp4")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 for all available")
    parser.add_argument("--slide_duration", type=float, default=2.5, help="Duration of title slide in seconds")
    args = parser.parse_args()

    compile_videos(args.mapping, args.output, args.max_episodes, args.slide_duration)
