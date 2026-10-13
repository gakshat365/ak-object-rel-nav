"""
Interleaved Rollout Collector for DAgger Failure Recovery.

Executes rollouts from recorded failure states towards the original goal:
1. Student policy predicts action π_θ(s_t).
2. Expert oracle computes optimal action a_t* = π*(s_t, g).
3. Executed action is sampled via Bernoulli(β_i):
   - With probability β_i: expert acts (action_source = "expert")
   - With probability (1 - β_i): student acts (action_source = "student")
4. Visited state s_t is labeled with ground-truth expert target a_t* (100% labeling).
5. State, observation context, goal masks, and action labels are stored as sample bundles.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import quaternion
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[RolloutCollector]")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "libs" / "control" / "object_react" / "train"))

from libs.common import utils
from libs.common import utils_sim_traj as ust
from dagger.core.failure_loader import FailureLoader, FailureCase
from dagger.core.expert_oracle import HabitatExpertOracle, OracleOutput, quat_to_yaw
from dagger.validate_expert import find_scene_glb


class RolloutCollector:
    """
    Rollout engine orchestrating beta-interleaved trajectory execution and expert data collection.
    """

    def __init__(self, config_path: str = "dagger/config.yaml"):
        config_file = REPO_ROOT / config_path
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.hm3d_root = REPO_ROOT / self.cfg["paths"]["hm3d_scenes_path"]
        self.sim_cfg = self.cfg["simulation"]
        self.model_cfg = self.cfg["model"]
        self.threshold_dist = float(self.sim_cfg["threshold_goal_distance"])
        self.max_steps = int(self.sim_cfg["max_rollout_steps"])
        self.time_delta = float(self.sim_cfg["time_delta"])

    def collect_case(
        self,
        case: FailureCase,
        controller,
        beta: float,
        iteration_idx: int,
        output_dir: Path,
        episode_idx: int,
        max_steps_override: Optional[int] = None,
    ) -> List[Dict]:
        """
        Runs an interleaved rollout for a single failure case and logs samples to disk.
        """
        scene_glb = find_scene_glb(self.hm3d_root, case.scene_id)
        if scene_glb is None or not scene_glb.exists():
            logger.warning(f"Skipping {case.episode_name}: scene mesh not found in {self.hm3d_root}")
            return []

        sim, agent, _ = utils.get_sim_agent(
            str(scene_glb),
            updateNavMesh=False,
            width=self.sim_cfg["width"],
            height=self.sim_cfg["height"],
            hfov=self.sim_cfg["hfov"],
            sensor_height=self.sim_cfg["sensor_height"],
        )

        import habitat_sim
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True

        records = []
        try:
            # Initialize agent at recorded failure state
            start_state = agent.get_state()
            start_state.position = case.failure_position
            q = case.failure_rotation
            start_state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
            agent.set_state(start_state)

            # Initialize oracle and reset controller
            oracle = HabitatExpertOracle(
                sim=sim,
                len_traj_pred=self.model_cfg["len_traj_pred"],
                learn_angle=self.model_cfg["learn_angle"],
                normalize=self.model_cfg["normalize"],
                metric_waypoint_spacing=self.model_cfg["metric_waypoint_spacing"],
                waypoint_spacing=self.model_cfg["waypoint_spacing"],
            )
            controller.reset_params()

            initial_dist = ust.find_shortest_path(sim, case.failure_position, case.goal_position)[0]
            step_budget = max_steps_override or self.max_steps

            for step in range(step_budget):
                curr_pos = agent.get_state().position
                curr_dist = ust.find_shortest_path(sim, curr_pos, case.goal_position)[0]

                # Goal reached check
                if curr_dist <= self.threshold_dist:
                    logger.info(
                        f"Episode {case.episode_name} achieved goal at step {step} "
                        f"(dist={curr_dist:.3f}m <= {self.threshold_dist}m)"
                    )
                    break

                # 1. Capture sensor observations
                obs = sim.get_sensor_observations()
                display_img, depth, semantic = utils.split_observations(obs)

                # 2. Prepare ObjectReact goal representation
                unique_classes = np.unique(semantic)
                masks = (semantic[None, ...] == unique_classes[:, None, None])
                pls = np.array([curr_dist] * len(masks), dtype=np.float32)
                control_input = [masks, pls]

                # 3. Query student policy
                v_s, w_s, _ = controller.predict(display_img, control_input)
                student_action = controller.action_pred.copy()

                # Extract observation context history (context_size + 1 images)
                # controller.image_history contains torch tensors of shape (3, H, W)
                if len(controller.image_history) > 0:
                    obs_context = np.stack([img.cpu().numpy() for img in controller.image_history])
                else:
                    obs_context = np.zeros((1, 3, self.model_cfg["image_size"][1], self.model_cfg["image_size"][0]), dtype=np.float32)

                # 4. Query expert oracle on current state
                agent_state = agent.get_state()
                oracle_out: OracleOutput = oracle.compute_expert_action(
                    current_position=agent_state.position,
                    current_rotation=agent_state.rotation,
                    goal_position=case.goal_position,
                )

                if not oracle_out.success or oracle_out.action_label is None:
                    logger.warning(
                        f"Step {step}: Expert oracle failed for {case.episode_name} "
                        f"({oracle_out.error_message}). Ending episode."
                    )
                    break

                expert_action = oracle_out.action_label.copy()
                v_e = oracle_out.linear_velocity
                w_e = oracle_out.angular_velocity

                # 5. Interleaved action selection: sample Bernoulli(beta)
                u = np.random.rand()
                if u < beta:
                    action_source = "expert"
                    v_exec, w_exec = v_e, w_e
                else:
                    action_source = "student"
                    v_exec, w_exec = v_s, w_s

                # 6. Save sample bundle to disk
                sample_id = f"ep_{episode_idx:03d}_step_{step:04d}"
                sample_dir = output_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Save raw RGB image
                Image.fromarray(display_img).save(sample_dir / "rgb.png")

                # Save tensors
                np.save(sample_dir / "obs_context.npy", obs_context.astype(np.float32))
                np.save(sample_dir / "goal_masks.npy", masks.astype(bool))
                np.save(sample_dir / "goal_pls.npy", pls.astype(np.float32))
                np.save(sample_dir / "expert_action.npy", expert_action.astype(np.float32))
                np.save(sample_dir / "student_action.npy", student_action.astype(np.float32))
                np.save(sample_dir / "position.npy", agent_state.position.astype(np.float32))

                # Current yaw
                curr_yaw = quat_to_yaw(
                    agent_state.rotation.x,
                    agent_state.rotation.y,
                    agent_state.rotation.z,
                    agent_state.rotation.w,
                )
                np.save(sample_dir / "yaw.npy", np.float32(curr_yaw))

                # Metadata record
                sample_meta = {
                    "sample_id": sample_id,
                    "sample_dir": str(sample_dir),
                    "episode_name": case.episode_name,
                    "scene_id": case.scene_id,
                    "iteration": iteration_idx,
                    "step": step,
                    "action_source": action_source,
                    "beta": float(beta),
                    "distance_to_goal": float(curr_dist),
                    "linear_velocity": float(v_exec),
                    "angular_velocity": float(w_exec),
                }
                with open(sample_dir / "metadata.json", "w") as f:
                    json.dump(sample_meta, f, indent=2)

                records.append(sample_meta)

                # 7. Apply selected action in simulation
                agent, sim, _ = utils.apply_velocity(
                    vel_control=vel_control,
                    agent=agent,
                    sim=sim,
                    velocity=v_exec,
                    steer=-w_exec,
                    time_step=self.time_delta,
                )

        finally:
            sim.close()

        return records

    def collect_iteration(
        self,
        cases: List[FailureCase],
        checkpoint_path: str,
        beta: float,
        iteration_idx: int,
        output_dir: Optional[Path] = None,
        max_episodes: Optional[int] = None,
        max_steps_per_episode: Optional[int] = None,
    ) -> List[Dict]:
        """
        Runs rollouts across cases for an entire DAgger iteration.
        """
        if output_dir is None:
            output_dir = REPO_ROOT / self.cfg["paths"]["data_dir"] / f"iter_{iteration_idx}" / "samples"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Preload Controller
        from libs.control.objectreact import ObjRelLearntController

        ctrl_cfg_path = REPO_ROOT / self.cfg["paths"]["controller_config"]
        with open(ctrl_cfg_path, "r") as f:
            ctrl_cfg = yaml.safe_load(f)

        ctrl_cfg["load_run"] = str(checkpoint_path)
        logger.info(f"Initializing controller from checkpoint: {checkpoint_path}")
        controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")

        target_cases = cases[:max_episodes] if max_episodes is not None else cases
        logger.info(
            f"Starting DAgger Iteration {iteration_idx} Rollouts: "
            f"{len(target_cases)} cases, beta={beta:.2f}, output={output_dir}"
        )

        all_records = []
        for ep_idx, case in enumerate(tqdm(target_cases, desc=f"Collecting Iter {iteration_idx} (beta={beta})")):
            case_records = self.collect_case(
                case=case,
                controller=controller,
                beta=beta,
                iteration_idx=iteration_idx,
                output_dir=output_dir,
                episode_idx=ep_idx,
                max_steps_override=max_steps_per_episode,
            )
            all_records.extend(case_records)

        logger.info(
            f"Finished Iteration {iteration_idx} Collection: "
            f"{len(all_records)} total sample states recorded across {len(target_cases)} episodes."
        )
        return all_records


def main():
    parser = argparse.ArgumentParser(description="Collect Interleaved DAgger Rollouts")
    parser.add_argument("--config", type=str, default="dagger/config.yaml", help="Path to dagger config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint (.pth)")
    parser.add_argument("--beta", type=float, default=0.7, help="Intervention parameter beta")
    parser.add_argument("--iteration", type=int, default=0, help="DAgger iteration index")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to run (for pilot/testing)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode override")
    parser.add_argument("--out_dir", type=str, default=None, help="Output samples directory")
    args = parser.parse_args()

    collector = RolloutCollector(args.config)
    checkpoint = args.checkpoint or collector.cfg["paths"]["base_checkpoint"]

    loader = FailureLoader(args.config)
    all_cases = loader.load_from_failed_cases_dir()
    train_c, val_c, test_c = loader.split_cases(all_cases)

    case_splits = {
        "train": train_c,
        "val": val_c,
        "test": test_c,
        "all": all_cases,
    }
    selected_cases = case_splits[args.split]

    out_path = Path(args.out_dir) if args.out_dir else None
    records = collector.collect_iteration(
        cases=selected_cases,
        checkpoint_path=checkpoint,
        beta=args.beta,
        iteration_idx=args.iteration,
        output_dir=out_path,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
    )
    print(f"Collected {len(records)} samples successfully.")


if __name__ == "__main__":
    main()
