"""
Shared Visualization Utilities for Navigation, Perception, and Video Generation.
Consolidates duplicated video concatenation, letterboxing, and title slide logic across:
  - visualisations/visualize_traversability_analysis.py
  - dagger/visualize_comparison_video.py
  - visualisations/compile_failure_videos.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def concatenate_videos(video_paths: List[Path], output_path: Path, fps: int = 12):
    """Concatenates multiple video files into a single continuous compilation."""
    valid_paths = [p for p in video_paths if p.exists() and p.stat().st_size > 0]
    if not valid_paths:
        return

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


def get_target_name(case) -> str:
    """Extracts human-readable target name from case metadata or episode string."""
    target = getattr(case, "metadata", {}).get("target") if hasattr(case, "metadata") else None
    if target and target != "Target Goal":
        return target
    ep_name = getattr(case, "episode_name", str(case))
    parts = ep_name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[2:-1])
    return "Target Goal"


def create_intro_slide(
    case,
    max_steps: int = 300,
    width: int = 1920,
    height: int = 1080,
    duration_sec: float = 2.5,
    fps: int = 12,
    main_title: str = "TRAVERSABILITY & PERCEPTION ANALYSIS",
    subtitle: str = "ObjectReact / DAgger Failure Case Inspection",
    accent_color: Tuple[int, int, int] = (80, 220, 120),
    extra_fields: Optional[List[Tuple[str, str]]] = None,
) -> List[np.ndarray]:
    """Generates an HD title card with metadata."""
    slide = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / height
        slide[y, :] = np.array([22 + 18 * ratio, 26 + 22 * ratio, 35 + 28 * ratio], dtype=np.uint8)

    # Accent top border
    slide[0:8, :] = list(accent_color)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_factor = width / 1920.0
    cv2.putText(slide, main_title, (int(100 * scale_factor), int(130 * scale_factor)), font, 1.35 * scale_factor, (245, 245, 245), int(3 * scale_factor), cv2.LINE_AA)
    cv2.putText(slide, subtitle, (int(100 * scale_factor), int(185 * scale_factor)), font, 0.9 * scale_factor, accent_color, int(2 * scale_factor), cv2.LINE_AA)
    cv2.line(slide, (int(100 * scale_factor), int(215 * scale_factor)), (width - int(100 * scale_factor), int(215 * scale_factor)), (70, 80, 100), int(2 * scale_factor))

    target_name = get_target_name(case)
    ep_name = getattr(case, "episode_name", str(case))
    scene_id = getattr(case, "scene_id", "Unknown")
    init_dist = getattr(case, "initial_goal_dist", 0.0)

    fields = [
        ("Episode Name:", f"{ep_name}"),
        ("Scene ID:", f"{scene_id}"),
        ("Target Goal Object:", f"{target_name.upper()}"),
        ("Initial Dist to Goal:", f"{init_dist:.2f} meters"),
        ("Step Budget:", f"{max_steps} steps"),
    ]
    if extra_fields:
        fields.extend(extra_fields)

    y_pos = int(295 * scale_factor)
    spacing = int(60 * scale_factor)
    for label, val in fields:
        cv2.putText(slide, label, (int(120 * scale_factor), y_pos), font, 0.8 * scale_factor, (160, 170, 185), int(2 * scale_factor), cv2.LINE_AA)
        val_color = (100, 230, 150) if "Target" in label or "Analysis" in label else (255, 255, 255)
        cv2.putText(slide, val, (int(440 * scale_factor), y_pos), font, 0.85 * scale_factor, val_color, int(2 * scale_factor), cv2.LINE_AA)
        y_pos += spacing

    num_frames = int(duration_sec * fps)
    return [slide.copy() for _ in range(num_frames)]


def render_metric_waypoint_plot(
    pred_waypoints: np.ndarray,
    width: int = 560,
    height: int = 490,
    title: str = "Predicted Waypoints (Local Frame)",
) -> np.ndarray:
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

    ax.set_xlim(4, -4)  # Inverted x-axis matches robot forward coordinate system (Left = +Y, Right = -Y)
    ax.set_ylim(-1, 10)
    ax.set_xlabel("Lateral Offset (m)", color="#a0abbd", fontsize=9)
    ax.set_ylabel("Forward Distance (m)", color="#a0abbd", fontsize=9)
    ax.set_title(title, color="#ffffff", fontsize=11, fontweight="bold", pad=8)

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
