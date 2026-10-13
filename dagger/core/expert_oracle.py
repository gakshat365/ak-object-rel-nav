"""
Habitat Pathfinder Expert Oracle with ObjectReact Target Mapping.

Computes geodesic shortest paths using Habitat-Sim's navmesh Pathfinder API,
samples forward lookahead waypoints, and transforms them into the exact
normalized egocentric coordinate format expected by ObjectReact and vint_dataset.py.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import habitat_sim

logger = logging.getLogger("[ExpertOracle]")


def quat_to_yaw(x, y, z, w) -> float:
    """
    Computes counterclockwise yaw angle (radians) from quaternion components.
    Matches libs/control/object_react/train/vint_train/process_data/process_data_utils.py.
    """
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


def to_local_coords_2d(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    """
    Converts 2D world coordinates (x, z) into robot local coordinates.
    Matches vint_train.data.data_utils.to_local_coords.
    """
    rotmat = yaw_rotmat_2d(curr_yaw)
    diff = positions - curr_pos
    # Local coordinate projection
    return diff @ rotmat


@dataclass
class OracleOutput:
    success: bool
    geodesic_distance: float
    # Target action tensor in ObjectReact format (len_traj_pred, action_dim)
    action_label: Optional[np.ndarray]
    # Local waypoint lookaheads for immediate control (v, w)
    linear_velocity: float
    angular_velocity: float
    raw_path_points: Optional[np.ndarray]
    error_message: Optional[str] = None


class HabitatExpertOracle:
    """
    Algorithmic navigation oracle computing optimal actions from current states to goals.
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        len_traj_pred: int = 10,
        learn_angle: bool = True,
        normalize: bool = True,
        metric_waypoint_spacing: float = 0.5,
        waypoint_spacing: int = 1,
        max_v: float = 0.05,
        max_w: float = 0.1,
    ):
        self.sim = sim
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.normalize = normalize
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.waypoint_spacing = waypoint_spacing
        self.scale_factor = self.metric_waypoint_spacing * self.waypoint_spacing
        self.max_v = max_v
        self.max_w = max_w

    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Queries Habitat's Pathfinder for geodesic shortest path points.
        """
        # Snap start and goal points to NavMesh if slightly off
        if not self.sim.pathfinder.is_navigable(start_pos):
            start_pos = self.sim.pathfinder.snap_point(start_pos)
        if not self.sim.pathfinder.is_navigable(goal_pos):
            goal_pos = self.sim.pathfinder.snap_point(goal_pos)

        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = goal_pos

        found = self.sim.pathfinder.find_path(path)
        dist = path.geodesic_distance

        if not found or math.isinf(dist) or len(path.points) < 2:
            return False, float("inf"), None

        return True, dist, np.array(path.points)

    def resample_path(self, path_points: np.ndarray, num_points: int, spacing: float) -> np.ndarray:
        """
        Interpolates raw path vertices along arc length at fixed spatial intervals.
        """
        # Distances between consecutive raw path points in 2D ground plane (X, Z)
        seg_diffs = np.diff(path_points[:, [0, 2]], axis=0)
        seg_lens = np.linalg.norm(seg_diffs, axis=1)
        cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
        total_len = cum_lens[-1]

        target_distances = np.arange(1, num_points + 1) * spacing

        resampled = []
        for td in target_distances:
            if td >= total_len:
                # Clamp to final goal point
                resampled.append(path_points[-1, [0, 2]])
            else:
                idx = np.searchsorted(cum_lens, td) - 1
                idx = max(0, min(idx, len(seg_lens) - 1))
                seg_ratio = (td - cum_lens[idx]) / max(seg_lens[idx], 1e-6)
                pt2d = path_points[idx, [0, 2]] + seg_ratio * seg_diffs[idx]
                resampled.append(pt2d)

        return np.array(resampled)

    def compute_expert_action(
        self,
        current_position: np.ndarray,
        current_rotation,
        goal_position: np.ndarray,
    ) -> OracleOutput:
        """
        Computes the expert action label and continuous velocity commands.
        
        Args:
            current_position: [x, y, z] agent world coordinates.
            current_rotation: agent rotation (quaternion with w, x, y, z or np.quaternion).
            goal_position: [x, y, z] target world coordinates.
        """
        found, dist, path_points = self.plan_path(current_position, goal_position)
        if not found:
            return OracleOutput(
                success=False,
                geodesic_distance=float("inf"),
                action_label=None,
                linear_velocity=0.0,
                angular_velocity=0.0,
                raw_path_points=None,
                error_message="Pathfinder could not find a valid navigable path to goal",
            )

        # Extract current yaw in radians
        if isinstance(current_rotation, np.ndarray) or isinstance(current_rotation, (list, tuple)):
            # Formatted as [w, x, y, z]
            w_val, x_val, y_val, z_val = current_rotation[0], current_rotation[1], current_rotation[2], current_rotation[3]
            curr_yaw = quat_to_yaw(x_val, y_val, z_val, w_val)
        elif hasattr(current_rotation, "w"):
            curr_yaw = quat_to_yaw(
                current_rotation.x, current_rotation.y, current_rotation.z, current_rotation.w
            )
        elif hasattr(current_rotation, "components"):
            c = current_rotation.components
            curr_yaw = quat_to_yaw(c[0], c[1], c[2], current_rotation.scalar)
        else:
            w_val, x_val, y_val, z_val = current_rotation[0], current_rotation[1], current_rotation[2], current_rotation[3]
            curr_yaw = quat_to_yaw(x_val, y_val, z_val, w_val)

        curr_pos_2d = current_position[[0, 2]]

        # Resample lookahead positions along the geodesic path
        future_positions_2d = self.resample_path(
            path_points,
            num_points=self.len_traj_pred,
            spacing=self.scale_factor,
        )

        # Transform to robot local coordinates (egocentric)
        local_waypoints = to_local_coords_2d(future_positions_2d, curr_pos_2d, curr_yaw)

        # Calculate yaw along trajectory
        if self.learn_angle:
            diffs = np.vstack([future_positions_2d[0] - curr_pos_2d, np.diff(future_positions_2d, axis=0)])
            waypoint_yaws = np.arctan2(diffs[:, 1], diffs[:, 0])
            yaw_deltas = waypoint_yaws - curr_yaw
            # Normalize angles to [-pi, pi]
            yaw_deltas = (yaw_deltas + np.pi) % (2 * np.pi) - np.pi
            action_label = np.concatenate([local_waypoints, yaw_deltas[:, None]], axis=-1)
        else:
            action_label = local_waypoints.copy()

        # Normalize action coordinates by metric spacing scale factor
        if self.normalize:
            action_label[:, :2] /= self.scale_factor

        # Controller commands: Compute linear velocity (v) and steering (w)
        # First lookahead waypoint
        immediate_wp = local_waypoints[0]
        dx, dy = immediate_wp[0], immediate_wp[1]
        
        # Heading angle to next waypoint
        w = float(np.arctan2(dy, dx))
        w = float(np.clip(w, -self.max_w, self.max_w))
        v = float(min(max(dx, 0.0), self.max_v))

        return OracleOutput(
            success=True,
            geodesic_distance=dist,
            action_label=action_label.astype(np.float32),
            linear_velocity=v,
            angular_velocity=-w,  # habitat steer convention
            raw_path_points=path_points,
        )
