"""
Safe Turn Controller for Object-Relative Navigation.

Extends ObjRelLearntController with:
1. Pure-Pursuit Turn-in-Place Heading Gate:
   - When heading error > threshold (default 20° / 0.35 rad), forces v = 0.0 to prevent
     ramming forward into walls/obstacles, and allows faster in-place rotation (up to ±0.4 rad/step).
2. 4D Waypoint & Orientation Utilization:
   - Evaluates immediate waypoints (steps 0 and 1) rather than skipping to waypoint -1 (5m ahead).
   - Ingests predicted [cos(theta), sin(theta)] heading angles from the neural network rather than discarding them.
3. Velocity Filter Lag Bypass:
   - Flushes action history during sharp turns so evasive rotation takes effect instantly
     without the ~17cm forward momentum drift caused by rolling average smoothing.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional, Any

from libs.control.objectreact import ObjRelLearntController, visualize_prediction

logger = logging.getLogger("[SafeTurnController]")


class SafeTurnController(ObjRelLearntController):
    """
    Non-destructive subclass of ObjRelLearntController adding in-place turning,
    full 4D waypoint usage, and lag-free evasive maneuvers.
    """

    def __init__(
        self,
        config: Any,
        turn_threshold: float = 0.35,  # ~20 degrees
        max_turn_w: float = 0.4,       # ~23 degrees per step
        immediate_wp_idx: int = 1,     # Look at step 1 (~0.5m ahead) instead of -1 (5m ahead)
        heading_weight: float = 0.5,   # Balance between positional angle and predicted [cos, sin] heading
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.turn_threshold = turn_threshold
        self.max_turn_w = max_turn_w
        self.immediate_wp_idx = immediate_wp_idx
        self.heading_weight = heading_weight
        self.is_turning_in_place = False
        logger.info(
            f"SafeTurnController initialized (turn_thresh={self.turn_threshold:.2f} rad, "
            f"max_turn_w={self.max_turn_w:.2f} rad, wp_idx={self.immediate_wp_idx})"
        )

    def predict(
        self, rgb: np.ndarray, goal_data: Any, visualize: bool = False
    ) -> Tuple[float, float, Optional[np.ndarray]]:
        """
        Predict linear velocity (v) and angular velocity (w) using the full 4D trajectory
        with turn-in-place gating.
        """
        self.iter += 1
        with torch.no_grad():
            obs_image = self.ready_obs(rgb)
            goal_image = self.ready_goal(goal_data)

            model_outputs = self.model(obs_image, goal_image)
            _, action_pred = model_outputs
            self.action_pred = action_pred[0].float().cpu().numpy()

            # Rollout logs for debugging / parity with base controller
            w_rollout = np.arctan2(self.action_pred[:, 1], self.action_pred[:, 0])
            w_rollout = np.insert(w_rollout, 0, 0)
            self.w_rollout = -(w_rollout[1:] - w_rollout[:-1])
            v_rollout = 0.2 * self.action_pred[:, 0]
            v_rollout = np.insert(v_rollout, 0, 0)
            self.v_rollout = v_rollout[1:] - v_rollout[:-1]

            # -------------------------------------------------------------
            # Step 1 Logic: Multi-Waypoint 4D Heading Error Computation
            # -------------------------------------------------------------
            wp_idx = min(self.immediate_wp_idx, len(self.action_pred) - 1)
            immediate_wp = self.action_pred[wp_idx]

            dx = immediate_wp[0]
            dy = immediate_wp[1]

            # 1. Positional heading error (direction of immediate translation)
            theta_pos = np.arctan2(dy, max(dx, 1e-4) if dx >= 0 else dx)

            # 2. Predicted orientation angle [cos(theta), sin(theta)] from network
            if self.config.get("learn_angle", True) and immediate_wp.shape[0] >= 4:
                cos_th = immediate_wp[2]
                sin_th = immediate_wp[3]
                theta_pred_heading = np.arctan2(sin_th, cos_th)
                # Composite error combining spatial direction and desired body orientation
                delta_theta = (
                    (1.0 - self.heading_weight) * theta_pos
                    + self.heading_weight * theta_pred_heading
                )
            else:
                delta_theta = theta_pos

            # -------------------------------------------------------------
            # Turn-in-Place Heading Gate
            # -------------------------------------------------------------
            if abs(delta_theta) > self.turn_threshold:
                # RECOVERY / SHARP TURN REGIME:
                # 1. Zero out linear velocity completely to avoid ramming obstacles
                v = 0.0
                # 2. Rotate cleanly in place with proportional control up to max_turn_w
                w = np.clip(delta_theta * 1.5, -self.max_turn_w, self.max_turn_w)
                # 3. Flush filter lag so the stop & turn take effect immediately
                self.action_history.clear()
                self.is_turning_in_place = True
            else:
                # ALIGNED FORWARD DRIVE REGIME:
                # 1. Linear velocity scales with forward clearance and alignment
                alignment_factor = max(np.cos(delta_theta), 0.0)
                v = min(max(dx / 100.0, 0.01), 0.05) * alignment_factor
                # 2. Gentle steering while moving forward
                w = np.clip(delta_theta, -0.1, 0.1)
                self.is_turning_in_place = False

                # Apply smoothing filter ONLY when driving forward in aligned regime
                if self.use_vel_filter:
                    v, w = self.filter_vel([v, w])

            logger.info(
                f"[SafeTurn] lin: {v:.3f}, ang: {w:.3f} (delta_th: {np.degrees(delta_theta):.1f}°, "
                f"turn_in_place={self.is_turning_in_place})"
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
                }
            )

        # Consistent with base controller sign convention: returns -w so steer = -(-w) = +w
        return v, -w, vis_img
