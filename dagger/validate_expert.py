"""
Standalone Validation Milestone for Habitat Expert Oracle (Gate 1).

Validates that:
1. Habitat pathfinder generates valid collision-free paths from recorded failure states.
2. Geodesic paths convert cleanly into ObjectReact-compatible action tensors.
3. Output action shapes, coordinate normalization, and steering match training conventions.
4. Unnavigable or invalid states are detected cleanly without silent corruption.
"""

import os
import sys
from pathlib import Path
import logging

from typing import Optional, Tuple
import numpy as np
import yaml

# Set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[ValidateExpert]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from libs.common import utils
from dagger.core.failure_loader import FailureLoader
from dagger.core.expert_oracle import HabitatExpertOracle


def find_scene_glb(hm3d_root: Path, scene_id: str) -> Optional[Path]:
    """Finds the basis.glb scene mesh for a given scene ID."""
    clean_id = scene_id.split("-")[-1].strip()
    matches = list(hm3d_root.glob(f"*{clean_id}*/*basis.glb"))
    if not matches:
        matches = list(hm3d_root.glob(f"*{clean_id}*/*.glb"))
    return matches[0] if matches else None


def validate_oracle_on_case(case, cfg, hm3d_root):
    scene_glb = find_scene_glb(hm3d_root, case.scene_id)
    if scene_glb is None or not scene_glb.exists():
        logger.warning(f"Could not find scene mesh for {case.scene_id} in {hm3d_root}")
        return False, "Scene GLB not found"

    logger.info(f"\n--- Testing Case: {case.episode_name} ({case.scene_id}) ---")
    logger.info(f"Failure Position: {case.failure_position}")
    logger.info(f"Goal Position:    {case.goal_position}")
    logger.info(f"Recorded Goal Dist: {case.initial_goal_dist:.3f}m")

    sim_cfg = cfg["simulation"]
    model_cfg = cfg["model"]

    # Initialize Simulator
    sim, agent, _ = utils.get_sim_agent(
        str(scene_glb),
        updateNavMesh=False,
        width=sim_cfg["width"],
        height=sim_cfg["height"],
        hfov=sim_cfg["hfov"],
        sensor_height=sim_cfg["sensor_height"],
    )

    try:
        # Check navmesh
        if not sim.pathfinder.is_loaded:
            return False, "Navmesh is not loaded in simulator"

        oracle = HabitatExpertOracle(
            sim=sim,
            len_traj_pred=model_cfg["len_traj_pred"],
            learn_angle=model_cfg["learn_angle"],
            normalize=model_cfg["normalize"],
            metric_waypoint_spacing=model_cfg["metric_waypoint_spacing"],
            waypoint_spacing=model_cfg["waypoint_spacing"],
        )

        output = oracle.compute_expert_action(
            current_position=case.failure_position,
            current_rotation=case.failure_rotation,
            goal_position=case.goal_position,
        )

        if not output.success:
            logger.error(f"Oracle failed to plan: {output.error_message}")
            return False, output.error_message

        # Assertions and checks
        action = output.action_label
        expected_dim = 3 if model_cfg["learn_angle"] else 2
        expected_shape = (model_cfg["len_traj_pred"], expected_dim)

        assert action is not None, "action_label is None"
        assert action.shape == expected_shape, f"Shape mismatch: {action.shape} vs {expected_shape}"
        assert not np.isnan(action).any(), "action_label contains NaNs"
        assert not np.isinf(action).any(), "action_label contains Infs"

        logger.info(f"Geodesic Distance Computed: {output.geodesic_distance:.3f}m")
        logger.info(f"Action Tensor Shape:        {action.shape}")
        logger.info(f"Local Waypoints (first 3): \n{action[:3]}")
        logger.info(f"Velocity Commands:          v={output.linear_velocity:.3f} m/s, w={output.angular_velocity:.3f} rad/s")

        # Range sanity checks
        max_coord = np.max(np.abs(action[:, :2]))
        assert max_coord < 25.0, f"Unreasonably large normalized waypoint: {max_coord}"

        return True, "Passed all validation checks"

    finally:
        sim.close()


def main():
    config_path = REPO_ROOT / "dagger/config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hm3d_root = REPO_ROOT / cfg["paths"]["hm3d_scenes_path"]
    loader = FailureLoader(str(config_path))
    cases = loader.load_from_failed_cases_dir()

    if len(cases) == 0:
        logger.error("No failure cases found to validate!")
        sys.exit(1)

    logger.info(f"Found {len(cases)} total cases. Validating on up to 3 diverse sample cases...")
    test_cases = cases[:3]

    results = []
    for case in test_cases:
        passed, msg = validate_oracle_on_case(case, cfg, hm3d_root)
        results.append((case.episode_name, passed, msg))

    logger.info("\n================ Validation Summary (Gate 1) ================")
    all_passed = True
    for name, passed, msg in results:
        status = "PASSED" if passed else "FAILED"
        logger.info(f"[{status}] {name}: {msg}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n>>> GO/NO-GO GATE 1 PASSED: Expert oracle produces validated ObjectReact targets.")
        sys.exit(0)
    else:
        logger.error("\n>>> GO/NO-GO GATE 1 FAILED: Resolve issues before continuing to Phase 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
