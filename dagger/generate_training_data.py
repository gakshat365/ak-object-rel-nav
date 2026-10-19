"""
Synthetic Trajectory & Training Data Generator for HM3D Meshes.

Generates multi-modal visual navigation trajectories directly from HM3D 3D meshes:
1. Loads HM3D scene meshes (*.basis.glb, *.semantic.glb, *.basis.navmesh).
2. Discovers semantic target objects (chair, sofa, bed, tv_monitor, plant, toilet, table, etc.).
3. Identifies unobstructed, walkable interaction viewpoints adjacent to each object.
4. Samples start locations at controllable geodesic distances (easy, medium, hard).
5. Steps an agent along the geodesic shortest path in Habitat-Sim, rendering:
   - First-person RGB camera images (320x240, 120° HFOV)
   - Depth maps (optional)
   - Semantic instance segmentation masks (optional)
   - 10-step future waypoint action tensors [x, y, cos(th), sin(th)]
   - Object distance rank costmaps (pls)
   - Top-down 2D floorplan trajectory maps

Usage:
  # Generate 5 trajectories in a specific scene
  CUDA_VISIBLE_DEVICES=1 pixi run python dagger/generate_training_data.py \
      --scene_id 00800-TEEsavR23oF \
      --num_trajectories 5 \
      --min_dist 3.0 --max_dist 10.0

  # Sample 20 trajectories across validation scenes
  CUDA_VISIBLE_DEVICES=1 pixi run python dagger/generate_training_data.py \
      --split val \
      --num_trajectories 20 \
      --categories chair,bed,sofa,tv_monitor,plant,toilet
"""

import argparse
import collections
import json
import logging
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import habitat_sim
from habitat.utils.visualizations import maps
import magnum as mn
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[DataGenerator]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from libs.common import utils
from libs.common import utils_sim_traj as ust


# Canonical categories and synonyms in HM3D
CATEGORY_ALIASES = {
    "chair": ["chair", "armchair", "stool", "swivel chair"],
    "sofa": ["sofa", "couch", "settee", "loveseat"],
    "bed": ["bed", "bunk bed"],
    "tv_monitor": ["tv", "tv_monitor", "television", "monitor", "screen"],
    "plant": ["plant", "potted plant", "houseplant", "vase"],
    "toilet": ["toilet", "commode", "water closet"],
    "table": ["table", "dining table", "coffee table", "desk"],
    "refrigerator": ["refrigerator", "fridge"],
    "sink": ["sink", "washbasin"],
}


def find_scene_glb(hm3d_root: Path, scene_id: str) -> Optional[Path]:
    """Finds the *.basis.glb mesh for a given scene ID."""
    clean_id = scene_id.split("-")[-1].strip()
    candidates = [hm3d_root]
    for sub in ["train", "val", "minival"]:
        if (hm3d_root / sub).is_dir():
            candidates.append(hm3d_root / sub)

    for c in candidates:
        matches = list(c.glob(f"*{clean_id}*/*basis.glb"))
        if not matches:
            matches = list(c.glob(f"*{clean_id}*/*.glb"))
        if matches:
            return matches[0]
    return None


def get_available_scene_ids(hm3d_root: Path, split: str = "val") -> List[str]:
    """Returns list of scene IDs for a given split (val, train, minival)."""
    split_dir = hm3d_root / split
    if not split_dir.exists():
        return []
    scene_folders = [f.name for f in split_dir.iterdir() if f.is_dir() and "-" in f.name]
    return sorted(scene_folders)


def quat_to_yaw(x, y, z, w) -> float:
    """Computes counterclockwise yaw angle (radians) from quaternion components."""
    fx, fy, fz, fw = float(x), float(y), float(z), float(w)
    t3 = 2.0 * (fw * fz + fx * fy)
    t4 = 1.0 - 2.0 * (fy * fy + fz * fz)
    return float(np.arctan2(t3, t4))


def yaw_rotmat_2d(yaw: float) -> np.ndarray:
    """2D planar rotation matrix for egocentric coordinate transformation."""
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ]
    )


def resample_lookahead_waypoints(path_points: np.ndarray, num_points: int = 10, spacing: float = 0.5) -> np.ndarray:
    """Resamples forward lookahead points at fixed metric intervals along path polyline."""
    if len(path_points) == 0:
        return np.zeros((num_points, 2))

    if len(path_points) == 1:
        return np.tile(path_points[0, [0, 2]], (num_points, 1))

    seg_diffs = np.diff(path_points[:, [0, 2]], axis=0)
    seg_lens = np.linalg.norm(seg_diffs, axis=1)
    cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_len = cum_lens[-1]

    target_distances = np.arange(1, num_points + 1) * spacing
    resampled = []
    for td in target_distances:
        if td >= total_len:
            resampled.append(path_points[-1, [0, 2]])
        else:
            idx = np.searchsorted(cum_lens, td) - 1
            idx = max(0, min(idx, len(seg_lens) - 1))
            seg_ratio = (td - cum_lens[idx]) / max(seg_lens[idx], 1e-6)
            pt2d = path_points[idx, [0, 2]] + seg_ratio * seg_diffs[idx]
            resampled.append(pt2d)

    return np.array(resampled)


def compute_action_label(
    sim: habitat_sim.Simulator,
    curr_pos: np.ndarray,
    curr_yaw: float,
    goal_pos: np.ndarray,
    num_wps: int = 10,
    spacing: float = 0.5,
) -> np.ndarray:
    """
    Computes (num_wps, 4) action tensor in robot's local egocentric coordinates:
    [x_local, y_local, cos(th), sin(th)]
    """
    path = habitat_sim.ShortestPath()
    path.requested_start = curr_pos
    path.requested_end = goal_pos
    sim.pathfinder.find_path(path)

    if len(path.points) < 2:
        return np.zeros((num_wps, 4), dtype=np.float32)

    raw_wps_2d = resample_lookahead_waypoints(np.array(path.points), num_points=num_wps, spacing=spacing)

    # Transform to robot local frame
    rotmat = yaw_rotmat_2d(curr_yaw)
    curr_2d = np.array([curr_pos[0], curr_pos[2]])
    local_wps = (raw_wps_2d - curr_2d) @ rotmat

    # Local orientations
    local_yaws = np.arctan2(local_wps[:, 1], np.maximum(local_wps[:, 0], 0.05))
    cos_th = np.cos(local_yaws)[:, None]
    sin_th = np.sin(local_yaws)[:, None]

    action_tensor = np.concatenate([local_wps, cos_th, sin_th], axis=-1).astype(np.float32)
    return action_tensor


class HM3DTrajectoryGenerator:
    """
    Automated trajectory generation pipeline for Habitat-Sim and HM3D.
    """

    def __init__(
        self,
        hm3d_root: Path,
        width: int = 320,
        height: int = 240,
        hfov: int = 120,
        sensor_height: float = 0.4,
        step_size: float = 0.25,
    ):
        self.hm3d_root = Path(hm3d_root).resolve()
        self.width = width
        self.height = height
        self.hfov = hfov
        self.sensor_height = sensor_height
        self.step_size = step_size

    def init_sim(self, scene_glb: Path) -> Tuple[habitat_sim.Simulator, habitat_sim.Agent]:
        """Initializes Habitat-Sim for a given scene mesh."""
        sim, agent, _ = utils.get_sim_agent(
            str(scene_glb),
            updateNavMesh=False,
            width=self.width,
            height=self.height,
            hfov=self.hfov,
            sensor_height=self.sensor_height,
        )
        return sim, agent

    def find_target_objects(
        self,
        sim: habitat_sim.Simulator,
        desired_categories: List[str],
    ) -> List[Dict]:
        """
        Discovers instances of target categories in the scene's semantic annotations.
        """
        targets = []
        # Expand categories with aliases
        allowed_names = set()
        for cat in desired_categories:
            cat_lower = cat.lower().strip()
            aliases = CATEGORY_ALIASES.get(cat_lower, [cat_lower])
            allowed_names.update(aliases)

        semantic_scene = sim.semantic_scene
        if not semantic_scene or not semantic_scene.objects:
            logger.warning("No semantic scene objects found in mesh!")
            return targets

        for obj in semantic_scene.objects:
            if not obj or not obj.category:
                continue
            cat_name = obj.category.name().lower().strip()
            if cat_name in allowed_names:
                obj_center = np.array(obj.aabb.center, dtype=np.float32)
                # Ensure object is near valid walkable space
                snapped = sim.pathfinder.snap_point(obj_center)
                if not np.isnan(snapped).any() and sim.pathfinder.is_navigable(snapped):
                    targets.append(
                        {
                            "category": cat_name,
                            "instance_id": int(obj.id.split("_")[-1]) if "_" in str(obj.id) else int(obj.id),
                            "center": obj_center,
                            "snapped_nav": snapped,
                            "aabb_sizes": np.array(obj.aabb.sizes, dtype=np.float32),
                        }
                    )
        return targets

    def find_viewpoint_for_target(
        self,
        sim: habitat_sim.Simulator,
        target_info: Dict,
        num_candidates: int = 16,
        radius: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Finds an adjacent walkable NavMesh viewpoint facing the target object.
        """
        center = target_info["center"]
        angles = np.linspace(0, 2 * np.pi, num_candidates, endpoint=False)
        best_pt = None
        min_dist = float("inf")

        for angle in angles:
            offset = np.array([radius * np.cos(angle), 0.0, radius * np.sin(angle)])
            cand_pt = center + offset
            cand_pt[1] = target_info["snapped_nav"][1]  # Match floor height

            if not sim.pathfinder.is_navigable(cand_pt):
                cand_pt = sim.pathfinder.snap_point(cand_pt)

            if not np.isnan(cand_pt).any() and sim.pathfinder.is_navigable(cand_pt):
                d = np.linalg.norm(cand_pt - center)
                if 0.5 <= d <= 2.2:
                    # Check line of sight
                    if sim.pathfinder.island_radius(cand_pt) > 0.5:
                        return cand_pt
        return target_info["snapped_nav"]

    def sample_trajectory_points(
        self,
        sim: habitat_sim.Simulator,
        p_goal: np.ndarray,
        min_dist: float = 3.0,
        max_dist: float = 10.0,
        max_attempts: int = 150,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Samples a valid start point at desired geodesic distance on the same floor.
        Returns: (start_position, path_points, geodesic_distance)
        """
        floor_y = p_goal[1]

        for _ in range(max_attempts):
            p_start = sim.pathfinder.get_random_navigable_point()
            if np.isnan(p_start).any():
                continue

            # Same floor constraint (within 0.5m)
            if abs(p_start[1] - floor_y) > 0.5:
                continue

            path = habitat_sim.ShortestPath()
            path.requested_start = p_start
            path.requested_end = p_goal
            found = sim.pathfinder.find_path(path)

            if found and min_dist <= path.geodesic_distance <= max_dist:
                raw_points = np.array(path.points)
                if len(raw_points) >= 2:
                    return p_start, raw_points, path.geodesic_distance

        return None

    def discretize_path(self, path_points: np.ndarray) -> np.ndarray:
        """Discretizes a polyline path with uniform step size."""
        pts = [np.array([p[0], p[1], p[2]], dtype=np.float32) for p in path_points]
        interpolated = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            seg_vec = p1 - p0
            seg_dist = np.linalg.norm(seg_vec)
            if seg_dist < 1e-4:
                continue
            steps = max(1, int(np.ceil(seg_dist / self.step_size)))
            for s in range(1, steps + 1):
                interpolated.append(p0 + (s / steps) * seg_vec)
        return np.asarray(interpolated, dtype=np.float32)

    def generate_single_trajectory(
        self,
        sim: habitat_sim.Simulator,
        agent: habitat_sim.Agent,
        target_info: Dict,
        p_goal: np.ndarray,
        p_start: np.ndarray,
        raw_path: np.ndarray,
        output_dir: Path,
        episode_idx: int,
        save_depth: bool = False,
        save_sem: bool = False,
    ) -> Dict:
        """
        Steps agent along the path, captures observations, computes labels, and writes episode files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        img_dir = output_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        depth_dir = output_dir / "images_depth" if save_depth else None
        sem_dir = output_dir / "images_sem" if save_sem else None
        if depth_dir:
            depth_dir.mkdir(parents=True, exist_ok=True)
        if sem_dir:
            sem_dir.mkdir(parents=True, exist_ok=True)

        p_goal = np.array([p_goal[0], p_goal[1], p_goal[2]], dtype=np.float32)
        p_start = np.array([p_start[0], p_start[1], p_start[2]], dtype=np.float32)
        discrete_pts = np.asarray(self.discretize_path(raw_path), dtype=np.float32)
        num_steps = len(discrete_pts)

        agent_states_records = []
        action_labels = []

        agent_state = habitat_sim.AgentState()

        for step in range(num_steps):
            curr_pos = np.asarray(discrete_pts[step], dtype=np.float32)

            # Orientation: at step 0 and onwards, face directly toward next waypoint along path
            if step < num_steps - 1:
                next_pt = np.asarray(discrete_pts[step + 1], dtype=np.float32)
            else:
                next_pt = np.asarray(p_goal, dtype=np.float32)
                diff_2d = next_pt[[0, 2]] - curr_pos[[0, 2]]
                if np.linalg.norm(diff_2d) < 1e-3:
                    if num_steps > 1:
                        next_pt = curr_pos + (discrete_pts[-1] - discrete_pts[-2])
                    else:
                        next_pt = curr_pos + np.array([0, 0, -1], dtype=np.float32)

            # Planar horizontal target point (eye level) to avoid vertical pitch/roll tilt
            target_pt = np.array([float(next_pt[0]), float(curr_pos[1]), float(next_pt[2])], dtype=np.float32)
            tangent_2d = target_pt[[0, 2]] - curr_pos[[0, 2]]
            if np.linalg.norm(tangent_2d) > 1e-4:
                rot_mat = mn.Matrix4.look_at(
                    mn.Vector3(float(curr_pos[0]), float(curr_pos[1]), float(curr_pos[2])),
                    mn.Vector3(float(target_pt[0]), float(target_pt[1]), float(target_pt[2])),
                    mn.Vector3(0, 1.0, 0),
                )
                rotation_q = mn.Quaternion.from_matrix(rot_mat.rotation())
            else:
                rotation_q = mn.Quaternion()

            agent_rot = habitat_sim.utils.common.quat_from_magnum(rotation_q)
            agent_state.position = curr_pos
            agent_state.rotation = agent_rot
            agent.set_state(agent_state)

            # Compute planar yaw angle around Habitat UP axis (Y)
            rot_xyzw = [agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w]
            yaw = float(Rotation.from_quat(rot_xyzw).as_euler("xyz")[1])

            # Capture Sensor Observations
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][:, :, :3]  # (H, W, 3) uint8

            # Save RGB
            img_path = img_dir / f"{step:05d}.png"
            Image.fromarray(rgb).save(img_path)

            if save_depth and "depth_sensor" in obs:
                depth = obs["depth_sensor"].astype(np.float16)
                np.save(depth_dir / f"{step:05d}.npy", depth)

            if save_sem and "semantic_sensor" in obs:
                sem = obs["semantic_sensor"].astype(np.uint16)
                np.save(sem_dir / f"{step:05d}.npy", sem)

            # Compute ground truth action label (10 future lookahead waypoints)
            action_label = compute_action_label(
                sim=sim,
                curr_pos=curr_pos,
                curr_yaw=yaw,
                goal_pos=p_goal,
                num_wps=10,
                spacing=0.5,
            )
            action_labels.append(action_label)

            agent_states_records.append(
                {
                    "step": step,
                    "position": np.array(curr_pos, dtype=np.float32).tolist(),
                    "rotation": [agent_rot.w, agent_rot.x, agent_rot.y, agent_rot.z] if hasattr(agent_rot, "w") else list(agent_rot),
                    "yaw_rad": float(yaw),
                }
            )

        # Save metadata and arrays
        np.save(output_dir / "path.npy", np.array(discrete_pts, dtype=np.float32))
        np.save(output_dir / "agent_states.npy", agent_states_records)
        np.save(output_dir / "action_labels.npy", np.array(action_labels, dtype=np.float32))
        np.save(output_dir / "goal_position.npy", np.array(p_goal, dtype=np.float32))

        # Save standard ViNT / ObjectReact odometry file
        traj_data = {
            "position": np.array(discrete_pts, dtype=np.float32)[:, [0, 2]],
            "yaw": np.array([r["yaw_rad"] for r in agent_states_records], dtype=np.float32),
        }
        with open(output_dir / "traj_data.pkl", "wb") as f:
            pickle.dump(traj_data, f)

        # Render Top-Down Map
        try:
            topdown_map = maps.get_topdown_map(sim.pathfinder, height=p_goal[1], meters_per_pixel=0.05)
            fig_map = utils_visualize_path_on_topdown(
                pathfinder=sim.pathfinder,
                topdown_map=topdown_map,
                path_points=discrete_pts,
                goal_pos=p_goal,
                meters_per_pixel=0.05,
            )
            if fig_map is not None:
                cv2.imwrite(str(output_dir / "trajectory_topdown.png"), fig_map)
        except Exception as e:
            logger.debug(f"Top-down map visualization skipped: {e}")

        metadata = {
            "episode_idx": episode_idx,
            "target_category": target_info["category"],
            "target_instance_id": target_info["instance_id"],
            "target_center": np.array(target_info["center"], dtype=np.float32).tolist(),
            "goal_interaction_point": np.array(p_goal, dtype=np.float32).tolist(),
            "start_position": np.array(p_start, dtype=np.float32).tolist(),
            "initial_geodesic_distance": float(np.linalg.norm(discrete_pts[-1] - discrete_pts[0])),
            "num_steps": num_steps,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata


def utils_visualize_path_on_topdown(
    pathfinder: habitat_sim.PathFinder,
    topdown_map: np.ndarray,
    path_points: np.ndarray,
    goal_pos: np.ndarray,
    meters_per_pixel: float = 0.05,
) -> Optional[np.ndarray]:
    """Draws trajectory polyline on 2D floor plan."""
    if topdown_map is None:
        return None
    # Recolor: free space = white (255), walls = dark gray (50)
    canvas = np.zeros((*topdown_map.shape, 3), dtype=np.uint8)
    canvas[topdown_map == 1] = [230, 230, 230]
    canvas[topdown_map == 0] = [40, 40, 40]

    grid_dims = canvas.shape[:2]
    pts_grid = []
    for pt in path_points:
        gx, gz = maps.to_grid(pt[2], pt[0], grid_dims, pathfinder=pathfinder)
        pts_grid.append((int(gz), int(gx)))

    # Draw path
    for i in range(len(pts_grid) - 1):
        cv2.line(canvas, pts_grid[i], pts_grid[i + 1], (40, 160, 240), 2, cv2.LINE_AA)

    # Start circle (Green)
    if pts_grid:
        cv2.circle(canvas, pts_grid[0], 5, (40, 220, 40), -1)
    # Goal star (Red)
    gx_goal, gz_goal = maps.to_grid(goal_pos[2], goal_pos[0], grid_dims, pathfinder=pathfinder)
    cv2.circle(canvas, (int(gz_goal), int(gx_goal)), 6, (40, 40, 240), -1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Modal Trajectories from HM3D Meshes")
    parser.add_argument("--hm3d_root", type=str, default="data/hm3d_v0.2", help="Path to HM3D dataset directory")
    parser.add_argument("--scene_id", type=str, default="00800-TEEsavR23oF", help="Specific scene ID, or 'all_val', 'all_train'")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "minival"], help="Dataset split")
    parser.add_argument("--categories", type=str, default="chair,bed,sofa,tv_monitor,plant,toilet", help="Comma-separated target categories")
    parser.add_argument("--num_trajectories", type=int, default=10, help="Number of trajectories to generate")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "custom"], help="Difficulty mode: easy (1.5-3.5m), medium (3.5-8.0m), hard (8.0-15.0m)")
    parser.add_argument("--min_dist", type=float, default=None, help="Minimum geodesic distance override (meters)")
    parser.add_argument("--max_dist", type=float, default=None, help="Maximum geodesic distance override (meters)")
    parser.add_argument("--step_size", type=float, default=0.25, help="Step size between camera frames (meters)")
    parser.add_argument("--output_dir", type=str, default="data/my_new_dataset", help="Directory to save output")
    parser.add_argument("--gpu", type=str, default="1", help="CUDA GPU ID (default: 1)")
    parser.add_argument("--save_depth", action="store_true", default=False, help="Save depth frames as NPY")
    parser.add_argument("--save_sem", action="store_true", default=False, help="Save semantic segmentation frames as NPY")
    parser.add_argument("--starts_per_target", type=int, default=1, help="Number of distinct starting locations to sample per target object")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Apply difficulty presets
    difficulty_presets = {
        "easy": (1.5, 3.5),
        "medium": (3.5, 8.0),
        "hard": (8.0, 15.0),
    }
    if args.difficulty in difficulty_presets:
        preset_min, preset_max = difficulty_presets[args.difficulty]
        min_dist = args.min_dist if args.min_dist is not None else preset_min
        max_dist = args.max_dist if args.max_dist is not None else preset_max
    else:
        min_dist = args.min_dist if args.min_dist is not None else 3.5
        max_dist = args.max_dist if args.max_dist is not None else 8.0

    args.min_dist = min_dist
    args.max_dist = max_dist

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)

    hm3d_root = Path(args.hm3d_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    logger.info(f"Difficulty mode: {args.difficulty} (Distance range: [{args.min_dist}m, {args.max_dist}m])")
    logger.info(f"Target categories: {categories}")

    # Resolve Scenes
    if args.scene_id in ["all_val", "all_train", "all"]:
        target_split = "val" if "val" in args.scene_id else "train"
        scene_ids = get_available_scene_ids(hm3d_root, split=target_split)
    else:
        scene_ids = [args.scene_id]

    logger.info(f"Generating {args.num_trajectories} total trajectories across {len(scene_ids)} scene(s)...")

    generator = HM3DTrajectoryGenerator(
        hm3d_root=hm3d_root,
        step_size=args.step_size,
    )

    total_generated = 0
    manifest_rows = []

    pbar = tqdm(total=args.num_trajectories, desc="Generating Trajectories", unit="traj")

    starts_per_target = max(1, args.starts_per_target)
    estimated_objects_total = len(scene_ids) * 9
    if args.num_trajectories > estimated_objects_total and starts_per_target == 1:
        starts_per_target = int(np.ceil(args.num_trajectories / max(1, estimated_objects_total)))
        logger.info(
            f"Auto-scaling starts_per_target to {starts_per_target} to reach requested {args.num_trajectories} trajectories."
        )

    pass_idx = 0
    max_passes = 10

    while total_generated < args.num_trajectories and pass_idx < max_passes:
        pass_idx += 1
        if pass_idx > 1:
            logger.info(f"\n--- Starting Pass {pass_idx}/{max_passes} to reach {args.num_trajectories} trajectories ---")

        for scene_id in scene_ids:
            if total_generated >= args.num_trajectories:
                break

            scene_glb = find_scene_glb(hm3d_root, scene_id)
            if not scene_glb or not scene_glb.exists():
                logger.warning(f"Scene GLB not found for: {scene_id}")
                continue

            logger.info(f"\nLoading Scene: {scene_id} -> {scene_glb.name}")
            sim, agent = generator.init_sim(scene_glb)

            try:
                # 1. Discover target objects
                targets = generator.find_target_objects(sim, desired_categories=categories)
                if not targets:
                    logger.warning(f"No matching target objects found in {scene_id}")
                    continue

                logger.info(f"Found {len(targets)} candidate target object instances in {scene_id}")
                random.shuffle(targets)

                # Sample episodes in this scene
                for target in targets:
                    if total_generated >= args.num_trajectories:
                        break

                    # 2. Viewpoint at goal
                    p_goal = generator.find_viewpoint_for_target(sim, target)
                    if p_goal is None:
                        continue

                    for _ in range(starts_per_target):
                        if total_generated >= args.num_trajectories:
                            break

                        # 3. Sample start position
                        sample_res = generator.sample_trajectory_points(
                            sim=sim,
                            p_goal=p_goal,
                            min_dist=args.min_dist,
                            max_dist=args.max_dist,
                        )
                        if not sample_res:
                            continue

                        p_start, raw_path, geo_dist = sample_res

                        # 4. Generate episode
                        ep_name = f"{scene_id}_{total_generated:05d}_{target['category']}_{target['instance_id']}"
                        ep_dir = out_dir / ep_name

                        metadata = generator.generate_single_trajectory(
                            sim=sim,
                            agent=agent,
                            target_info=target,
                            p_goal=p_goal,
                            p_start=p_start,
                            raw_path=raw_path,
                            output_dir=ep_dir,
                            episode_idx=total_generated,
                            save_depth=args.save_depth,
                            save_sem=args.save_sem,
                        )

                        total_generated += 1
                        pbar.update(1)
                        manifest_rows.append(metadata)
                        logger.info(
                            f"Generated [{total_generated}/{args.num_trajectories}]: {ep_name} "
                            f"({geo_dist:.2f}m, {metadata['num_steps']} steps)"
                        )

            finally:
                sim.close()

    pbar.close()

    # Save summary manifest
    manifest_path = out_dir / "generated_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_rows, f, indent=2)

    logger.info(f"\n Generation Complete!")
    logger.info(f"Total Trajectories: {total_generated}")
    logger.info(f"Saved to Directory:  {out_dir}")
    logger.info(f"Manifest File:     {manifest_path}")


if __name__ == "__main__":
    main()
