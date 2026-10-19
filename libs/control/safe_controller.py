"""
Adaptive Safe Turn Controller for Object-Relative Navigation (v3).

Key Design Principles:
1. Continuous Forward Driving for Normal Turning (|Δθ| <= 55°):
   - For all normal turns up to 55° (~0.95 rad), the robot drives forward smoothly
     while steering: v = max(0.02, 0.05 * cos(Δθ)), w = clip(0.80 * Δθ, -0.20, 0.20).
   - Never freezes forward motion for gentle turns or room navigation!
2. Turn-in-Place ONLY for Severe / Wall Reversals (|Δθ| > 55°):
   - Pure in-place rotation (v = 0.0, w = ±0.35 rad) is reserved exclusively for major
     pivots (e.g. facing a wall at 90° or 180°).
   - Once heading aligns within 28° (0.50 rad), it immediately exits and resumes forward driving.
3. Forward Drive Cooldown (Anti-Looping Guarantee):
   - After any in-place turn, the robot enters a mandatory forward-driving cooldown (10 steps)
     where v > 0 is enforced and in-place turning is temporarily blocked.
   - Mathematically prevents infinite spinning traps!
4. Adaptive Lookahead (0.6m - 1.8m):
   - Tracks smooth waypoints 3-4 (1.5m - 1.8m) on straight paths, collapses to waypoint 0-1 on corners.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional, Any

from libs.control.objectreact import ObjRelLearntController, visualize_prediction

logger = logging.getLogger("[SafeTurnController]")


class SafeTurnController(ObjRelLearntController):
    """
    Smooth forward-driving controller with high-threshold in-place recovery
    and mandatory forward drive cooldown.
    100% non-destructive: base ObjectReact weights and configs are unchanged.
    """

    def __init__(
        self,
        config: Any,
        turn_enter_thresh: float = 0.95,   # ~55° heading error to initiate emergency turn-in-place
        turn_exit_thresh: float = 0.50,    # ~28° heading error to exit turn-in-place
        max_turn_w: float = 0.35,          # ~20°/step max angular velocity
        min_lookahead_dist: float = 1.2,   # Min lookahead distance (meters, prevents short pure pursuit instability)
        max_lookahead_dist: float = 2.8,   # Max lookahead distance (meters, spans full long-horizon path)
        cooldown_steps: int = 10,          # Mandatory forward steps after turn-in-place
        heading_weight: float = 0.15,      # 15% weight to [cos, sin], 85% to spatial (x, y)
        use_adaptive_lookahead: bool = True,  # Toggle dynamic waypoint lookahead
        use_turn_in_place: bool = True,       # Toggle emergency turn-in-place
        lookahead_ema_alpha: float = 0.35, # Temporal smoothing factor for target lookahead
        **kwargs,
    ):
        self.turn_enter_thresh = turn_enter_thresh
        self.turn_exit_thresh = turn_exit_thresh
        self.max_turn_w = max_turn_w
        self.min_lookahead_dist = min_lookahead_dist
        self.max_lookahead_dist = max_lookahead_dist
        self.cooldown_steps = cooldown_steps
        self.heading_weight = heading_weight
        self.use_adaptive_lookahead = use_adaptive_lookahead
        self.use_turn_in_place = use_turn_in_place
        self.lookahead_ema_alpha = lookahead_ema_alpha

        # State tracking initialized before super call (super().__init__ calls reset_params)
        self.is_turning_in_place = False
        self.turn_in_place_steps = 0
        self.forward_cooldown_remaining = 0
        self.locked_turn_sign = 0.0
        self.current_lookahead_idx = -1
        self.current_lookahead_dist = max_lookahead_dist
        self.smoothed_lookahead_dist: Optional[float] = None

        super().__init__(config, **kwargs)

        logger.info(
            f"Adaptive SafeTurnController (v3-optimized) initialized: "
            f"lookahead=[{self.min_lookahead_dist:.1f}m - {self.max_lookahead_dist:.1f}m], "
            f"enter={np.degrees(self.turn_enter_thresh):.1f}°, "
            f"exit={np.degrees(self.turn_exit_thresh):.1f}°, "
            f"cooldown={self.cooldown_steps} steps"
        )

    def reset_params(self):
        super().reset_params()
        self.is_turning_in_place = False
        self.turn_in_place_steps = 0
        self.forward_cooldown_remaining = 0
        self.locked_turn_sign = 0.0
        self.current_lookahead_idx = -1
        self.current_lookahead_dist = getattr(self, "max_lookahead_dist", 2.8)
        self.smoothed_lookahead_dist = None

    def _compute_adaptive_lookahead(self, waypoints: np.ndarray) -> Tuple[int, float]:
        """
        Dynamically adjusts lookahead distance based on true spatial curvature and
        open-corridor clearance.
        
        Design:
        - In open corridors & expansive rooms: maintains full long-horizon lookahead (2.5m - 2.8m)
          for ballistic, stable transit toward distant goals (matching Flag OFF).
        - In tight bottlenecks & 90° corners: dynamically shortens lookahead (down to 1.2m)
          so the robot rounds door jambs and corners without clipping walls.
        """
        num_wps = len(waypoints)
        if num_wps == 0:
            return 0, self.min_lookahead_dist

        wp_distances = np.linalg.norm(waypoints[:, :2], axis=-1)

        # 1. During emergency in-place turning: track a stable intermediate pivot waypoint (~0.6m)
        # Avoids immediate waypoint-0 noise that causes sign chattering during pivots
        if self.is_turning_in_place:
            pivot_idx = min(2, num_wps - 1)
            return pivot_idx, float(wp_distances[pivot_idx])

        # 2. Extract near (0.8m) and far (2.4m - 2.8m) horizon samples
        idx_near = min(2, num_wps - 1)
        idx_mid = min(5, num_wps - 1)
        idx_far = min(8, num_wps - 1)

        x_near, y_near = waypoints[idx_near, 0], waypoints[idx_near, 1]
        x_far, y_far = waypoints[idx_far, 0], waypoints[idx_far, 1]

        theta_near = np.arctan2(y_near, max(x_near, 0.05))
        theta_far = np.arctan2(y_far, max(x_far, 0.05))

        # Spatial arc distance between near and far samples
        s_near = float(wp_distances[idx_near])
        s_far = float(wp_distances[idx_far])
        ds = max(s_far - s_near, 0.5)

        # Spatial curvature: rate of heading change per meter of path (rad/m)
        path_bend = abs(theta_far - theta_near)
        spatial_curvature = path_bend / ds

        # Lateral aspect ratio across the forward path
        max_lateral = float(np.max(np.abs(waypoints[:idx_mid + 1, 1])))
        max_forward = float(np.max(waypoints[:idx_mid + 1, 0]))
        lateral_aspect = max_lateral / max(max_forward, 0.20)

        # 3. Open-Corridor Gating:
        # If the path ahead has forward clearance and near-path is reasonably aligned (< 22°),
        # lock lookahead to max_lookahead_dist (2.8m, wp[9]) for ballistic momentum.
        near_aligned = abs(theta_near) < np.radians(22.0)
        forward_clearing = x_far > 1.6 and lateral_aspect < 0.45

        if near_aligned and forward_clearing:
            corner_intensity = 0.0
        else:
            # True corner bottleneck: activate penalty if near path turns sharply (> 22°),
            # spatial curvature is high (> 0.40 rad/m), or lateral deviation is significant (> 0.45)
            sharp_near = min(1.0, max(0.0, (abs(theta_near) - np.radians(22.0)) / np.radians(35.0)))
            sharp_curve = min(1.0, max(0.0, (spatial_curvature - 0.40) / 0.50))
            lateral_dev = min(1.0, max(0.0, (lateral_aspect - 0.45) / 0.40))
            corner_intensity = max(sharp_near, sharp_curve, lateral_dev)

        # Smooth scaling between min_lookahead_dist (1.2m) and max_lookahead_dist (2.8m)
        raw_target_dist = self.max_lookahead_dist - (self.max_lookahead_dist - self.min_lookahead_dist) * corner_intensity

        # 4. Temporal Exponential Moving Average (EMA) smoothing
        if self.smoothed_lookahead_dist is None:
            self.smoothed_lookahead_dist = raw_target_dist
        else:
            self.smoothed_lookahead_dist = (
                self.lookahead_ema_alpha * raw_target_dist
                + (1.0 - self.lookahead_ema_alpha) * self.smoothed_lookahead_dist
            )

        # Select closest waypoint matching the smoothed lookahead distance
        best_idx = int(np.argmin(np.abs(wp_distances - self.smoothed_lookahead_dist)))
        best_idx = max(1, min(best_idx, num_wps - 1))

        return best_idx, float(wp_distances[best_idx])

    def predict(
        self, rgb: np.ndarray, goal_data: Any, visualize: bool = False
    ) -> Tuple[float, float, Optional[np.ndarray]]:
        """
        Predict linear velocity (v) and angular velocity (w) with continuous forward driving,
        high-threshold in-place escape, and forward drive cooldown.
        """
        self.iter += 1
        with torch.no_grad():
            obs_image = self.ready_obs(rgb)
            goal_image = self.ready_goal(goal_data)

            model_outputs = self.model(obs_image, goal_image)
            _, action_pred = model_outputs
            self.action_pred = action_pred[0].float().cpu().numpy()

            # Rollout telemetry for parity
            w_rollout = np.arctan2(self.action_pred[:, 1], self.action_pred[:, 0])
            w_rollout = np.insert(w_rollout, 0, 0)
            self.w_rollout = -(w_rollout[1:] - w_rollout[:-1])
            v_rollout = 0.2 * self.action_pred[:, 0]
            v_rollout = np.insert(v_rollout, 0, 0)
            self.v_rollout = v_rollout[1:] - v_rollout[:-1]

            if self.use_adaptive_lookahead:
                self.current_lookahead_idx, self.current_lookahead_dist = self._compute_adaptive_lookahead(self.action_pred)
            else:
                self.current_lookahead_idx = self.waypoint_index
                wp_dist = np.linalg.norm(self.action_pred[self.current_lookahead_idx, :2])
                self.current_lookahead_dist = float(wp_dist)
            target_wp = self.action_pred[self.current_lookahead_idx]

            dx = target_wp[0]
            dy = target_wp[1]

            # Spatial heading angle
            theta_pos = np.arctan2(dy, max(dx, 0.05) if dx >= 0 else dx)

            # Robust 4D Heading: blend body orientation gently
            if self.config.get("learn_angle", True) and target_wp.shape[0] >= 4:
                cos_th = target_wp[2]
                sin_th = target_wp[3]
                theta_pred_body = np.arctan2(sin_th, cos_th)
                delta_theta = (1.0 - self.heading_weight) * theta_pos + self.heading_weight * theta_pred_body
            else:
                delta_theta = theta_pos

            # -------------------------------------------------------------
            # 2. State Machine: In-Place Turn vs. Forward Drive
            # -------------------------------------------------------------
            if not self.use_turn_in_place:
                self.is_turning_in_place = False
                self.turn_in_place_steps = 0
            elif self.forward_cooldown_remaining > 0:
                self.forward_cooldown_remaining -= 1
                # In cooldown: strictly forbid in-place turning to force forward movement
                self.is_turning_in_place = False
                self.turn_in_place_steps = 0
            elif self.is_turning_in_place:
                self.turn_in_place_steps += 1
                # Exit turn-in-place if aligned within 28° OR after 6 steps
                if abs(delta_theta) < self.turn_exit_thresh or self.turn_in_place_steps >= 6:
                    self.is_turning_in_place = False
                    self.turn_in_place_steps = 0
                    self.forward_cooldown_remaining = self.cooldown_steps
                    logger.info(f"[SafeTurn] Exiting turn-in-place -> starting {self.cooldown_steps}-step forward drive")
            else:
                # Only enter turn-in-place for SEVERE heading error (> 55°)
                if abs(delta_theta) > self.turn_enter_thresh:
                    self.is_turning_in_place = True
                    self.turn_in_place_steps = 1
                    self.locked_turn_sign = 1.0 if delta_theta > 0 else -1.0
                    logger.info(f"[SafeTurn] Severe error ({np.degrees(delta_theta):.1f}° > 55°) -> entering turn-in-place")

            # -------------------------------------------------------------
            # 3. Velocity Computation
            # -------------------------------------------------------------
            if self.is_turning_in_place:
                # --- EMERGENCY TURN IN PLACE ---
                v = 0.0
                turn_magnitude = min(self.max_turn_w, max(0.15, 0.75 * abs(delta_theta)))
                w = self.locked_turn_sign * turn_magnitude
                self.action_history.clear()
            else:
                # --- CONTINUOUS FORWARD DRIVING ---
                # Drive forward while curving: full speed when straight, gentle modulation on turns
                alignment_factor = max(np.cos(delta_theta), 0.35)  # Maintain forward momentum through curves
                base_v = 0.05 if self.forward_cooldown_remaining > 0 else 0.045
                v = base_v * alignment_factor

                # Smooth pursuit steering
                w = np.clip(0.85 * delta_theta, -0.22, 0.22)

                if self.use_vel_filter:
                    v, w = self.filter_vel([v, w])

            logger.info(
                f"[AdaptiveSafeTurn] lin: {v:.3f}, ang: {w:.3f} | delta_th: {np.degrees(delta_theta):.1f}° | "
                f"wp[{self.current_lookahead_idx}] ({self.current_lookahead_dist:.2f}m) | "
                f"tip={self.is_turning_in_place} | cooldown={self.forward_cooldown_remaining}"
            )

            vis_img = None
            if visualize:
                vis_img = visualize_prediction(
                    rgb,
                    self.action_pred,
                    self.goal_mask_vis,
                    save_path=None,
                    get_plot_img=True,
                )

            self.controller_logs.append(
                {
                    "action_pred": self.action_pred,
                    "v_rollout": self.v_rollout,
                    "w_rollout": self.w_rollout,
                    "is_turning_in_place": self.is_turning_in_place,
                    "delta_theta": delta_theta,
                    "lookahead_idx": self.current_lookahead_idx,
                    "lookahead_dist": self.current_lookahead_dist,
                }
            )

        # Base controller convention: return -w so caller's steer = -(-w) = +w
        return v, -w, vis_img
