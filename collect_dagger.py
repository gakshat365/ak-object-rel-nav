import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from libs.common import utils
from libs.common import utils_sim_traj
from libs.experiments import task_setup
from libs.control.object_react.train.vint_train.data.data_utils import (
    to_local_coords,
)


REPO_ROOT = Path(__file__).resolve().parent

FAILURE_CSV = REPO_ROOT / "out/dagger/failure_trajectories_1300.csv"
OUTPUT_ROOT = REPO_ROOT / "out/dagger/pilot"
SAMPLES_ROOT = OUTPUT_ROOT / "samples"

CHECKPOINT = (
    REPO_ROOT
    / "libs/control/object_react/train/logs/object_rel_nav/"
      "object_react_hm3d_2026_10_07_20_54_34_/latest.pth"
)


def quaternion_to_yaw(rotation):
    """Same yaw convention used by ObjectReact preprocessing."""
    x = float(rotation.x)
    y = float(rotation.y)
    z = float(rotation.z)
    w = float(rotation.w)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(t3, t4)


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sample_path(path_points, spacing=0.4, num_waypoints=10):
    """
    Sample 10 future points from the Habitat shortest path.

    The first point is the current robot position.
    Points are approximately 0.4 m apart.
    """

    path_points = np.asarray(path_points, dtype=np.float32)

    if path_points.ndim != 2 or len(path_points) < 2:
        return None

    segment_lengths = np.linalg.norm(
        np.diff(path_points, axis=0),
        axis=1,
    )

    cumulative = np.concatenate(
        [
            np.array([0.0], dtype=np.float32),
            np.cumsum(segment_lengths),
        ]
    )

    total_length = float(cumulative[-1])

    if total_length <= 1e-6:
        return None

    desired = np.arange(
        num_waypoints + 1,
        dtype=np.float32,
    ) * spacing

    desired = np.minimum(desired, total_length)

    sampled = []

    for d in desired:
        idx = np.searchsorted(
            cumulative,
            d,
            side="right",
        ) - 1

        idx = max(
            0,
            min(idx, len(path_points) - 2),
        )

        d0 = cumulative[idx]
        d1 = cumulative[idx + 1]

        if d1 - d0 < 1e-8:
            alpha = 0.0
        else:
            alpha = float((d - d0) / (d1 - d0))

        point = (
            path_points[idx]
            + alpha
            * (path_points[idx + 1] - path_points[idx])
        )

        sampled.append(point)

    return np.asarray(sampled, dtype=np.float32)


def make_expert_action(
    path_points,
    current_position,
    current_yaw,
):
    """
    Produce the exact 10x4 ObjectReact action representation.

    Training:
        waypoint spacing = 0.2 * 2 = 0.4 m
        local XY / 0.4
        relative yaw -> cos/sin
    """

    spacing = 0.4
    num_waypoints = 10

    sampled = sample_path(
        path_points,
        spacing=spacing,
        num_waypoints=num_waypoints,
    )

    if sampled is None:
        return None

    # --------------------------------------------------------
    # Position convention
    #
    # Habitat ground plane = X/Z.
    # --------------------------------------------------------

    positions = sampled[:, [0, 2]]

    current_xz = np.asarray(
        [
            current_position[0],
            current_position[2],
        ],
        dtype=np.float32,
    )

    local_positions = to_local_coords(
        positions,
        current_xz,
        current_yaw,
    )

    # --------------------------------------------------------
    # Derive expert path heading.
    # --------------------------------------------------------

    headings = []

    for i in range(num_waypoints):
        delta = sampled[i + 1] - sampled[i]

        dx = float(delta[0])
        dz = float(delta[2])

        if abs(dx) + abs(dz) < 1e-8:
            if headings:
                heading = headings[-1]
            else:
                heading = current_yaw
        else:
            heading = math.atan2(dz, dx)

        headings.append(heading)

    headings = np.asarray(
        headings,
        dtype=np.float32,
    )

    relative_yaw = np.asarray(
        [
            wrap_angle(
                heading - current_yaw
            )
            for heading in headings
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Exact training normalization.
    # --------------------------------------------------------

    local_xy = (
        local_positions[1:]
        / spacing
    )

    action = np.zeros(
        (num_waypoints, 4),
        dtype=np.float32,
    )

    action[:, :2] = local_xy
    action[:, 2] = np.cos(relative_yaw)
    action[:, 3] = np.sin(relative_yaw)

    return action


def resolve_scene(
    scene_root,
    trajectory_name,
):
    episode_name = trajectory_name.split("_")[0]

    matches = sorted(
        scene_root.glob(
            f"*{episode_name}"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No scene found for {trajectory_name}"
        )

    basis_files = sorted(
        matches[0].glob("*basis.glb")
    )

    if not basis_files:
        raise FileNotFoundError(
            f"No basis.glb found in {matches[0]}"
        )

    return basis_files[0]


def save_sample(
    sample_id,
    rgb,
    goal_masks,
    goal_pls,
    student_action,
    expert_action,
    state,
    distance_to_goal,
    trajectory,
    scene,
    step,
    image_history,
):
    sample_dir = (
        SAMPLES_ROOT
        / sample_id
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        rgb
    ).save(
        sample_dir / "rgb.png"
    )

    # Save the exact 5-frame observation history used by ObjectReact.
    # image_history contains the preprocessed C,H,W tensors maintained
    # internally by the controller.
    obs_context = torch.stack(
        [
            frame.detach().cpu()
            if torch.is_tensor(frame)
            else torch.as_tensor(frame)
            for frame in image_history
        ]
    ).numpy()

    np.save(
        sample_dir / "obs_context.npy",
        obs_context,
    )

    np.save(
        sample_dir / "goal_masks.npy",
        np.asarray(goal_masks),
    )

    np.save(
        sample_dir / "goal_pls.npy",
        np.asarray(goal_pls),
    )

    np.save(
        sample_dir / "student_action.npy",
        np.asarray(
            student_action,
            dtype=np.float32,
        ),
    )

    np.save(
        sample_dir / "expert_action.npy",
        np.asarray(
            expert_action,
            dtype=np.float32,
        ),
    )

    np.save(
        sample_dir / "position.npy",
        np.asarray(
            state.position,
            dtype=np.float32,
        ),
    )

    yaw = quaternion_to_yaw(
        state.rotation
    )

    np.save(
        sample_dir / "yaw.npy",
        np.asarray(
            yaw,
            dtype=np.float32,
        ),
    )

    with open(
        sample_dir / "metadata.txt",
        "w",
    ) as f:
        f.write(
            f"trajectory={trajectory}\n"
        )
        f.write(
            f"scene={scene}\n"
        )
        f.write(
            f"step={step}\n"
        )
        f.write(
            f"distance_to_goal={distance_to_goal}\n"
        )
        f.write(
            f"yaw={yaw}\n"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--start-trajectory",
        type=int,
        default=0,
        help="Index into the failure list from which to start collecting.",
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/dagger/pilot",
        help="Directory where DAgger samples and metadata are written.",
    )

    args_cli = parser.parse_args()

    # --------------------------------------------------------
    # Validate files.
    # --------------------------------------------------------

    if not FAILURE_CSV.exists():
        raise FileNotFoundError(
            FAILURE_CSV
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            CHECKPOINT
        )

    globals()["OUTPUT_ROOT"] = REPO_ROOT / args_cli.output_dir
    globals()["SAMPLES_ROOT"] = globals()["OUTPUT_ROOT"] / "samples"

    global OUTPUT_ROOT, SAMPLES_ROOT

    OUTPUT_ROOT = REPO_ROOT / args_cli.output_dir
    SAMPLES_ROOT = OUTPUT_ROOT / "samples"

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load exactly the same base configuration as main.py.
    # --------------------------------------------------------

    parser_config = REPO_ROOT / "configs/defaults.yaml"

    with open(
        parser_config,
        "r",
    ) as f:
        config = yaml.safe_load(f)

    class Args:
        pass

    args = Args()

    for key, value in config.items():
        setattr(
            args,
            key,
            value,
        )

    # --------------------------------------------------------
    # HM3D ObjectReact evaluation overrides.
    # --------------------------------------------------------

    args.start_idx = 0
    args.step_idx = 1
    args.end_idx = 999999

    args.task_type = "original"
    args.use_gt_localization = False

    args.env = "sim"
    args.split = "train"

    args.threshold_goal_distance = 1.0
    args.max_start_distance = "hard"
    args.max_steps = 300

    args.method = "learnt"
    args.goal_source = "gt_metric"

    args.infer_depth = False
    args.infer_traversable = False

    args.log_wandb = False
    args.log_robot = True

    args.plot = False
    args.save_vis = False

    args.except_exit = True

    args.cull_map_instances = False

    args.exp_name = "dagger_pilot"

    # --------------------------------------------------------
    # Same traversable classes as main.py.
    # --------------------------------------------------------

    args.traversable_class_names = [
        "floor",
        "flooring",
        "floor mat",
        "floor vent",
        "carpet",
        "mat",
        "rug",
        "doormat",
        "shower floor",
        "pavement",
        "ground",
        "tiles",
    ]

    # --------------------------------------------------------
    # Use the trained ObjectReact checkpoint.
    # --------------------------------------------------------

    args.controller = dict(
        args.controller
    )

    args.controller[
        "config_file"
    ] = "configs/controller/object_react_controller.yaml"

    args.controller[
        "load_run"
    ] = str(
        CHECKPOINT
    )

    # --------------------------------------------------------
    # Load failed trajectories.
    # --------------------------------------------------------

    with open(
        FAILURE_CSV,
        newline="",
    ) as f:
        failures = list(
            csv.DictReader(f)
        )

    failures = [
        row
        for row in failures
        if row["status"]
        == "exceeded_steps"
    ]

    failures = failures[
        args_cli.start_trajectory:
        args_cli.start_trajectory + args_cli.num_trajectories
    ]

    print(
        f"Using {len(failures)} failed trajectories"
    )

    # --------------------------------------------------------
    # Dataset paths.
    # --------------------------------------------------------

    path_dataset = Path(
        args.path_dataset
    )

    trajectory_root = (
        path_dataset
        / "hm3d_iin_train"
    )

    scene_root = (
        path_dataset
        / "hm3d_v0.2"
        / "train"
    )

    results_root = (
        OUTPUT_ROOT
        / "episode_results"
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load models exactly as main.py.
    # --------------------------------------------------------

    print(
        "Loading ObjectReact controller..."
    )

    preload_data = (
        task_setup.preload_models(
            args
        )
    )

    print(
        "Controller loaded."
    )

    # --------------------------------------------------------
    # Metadata CSV.
    # --------------------------------------------------------

    metadata_path = (
        OUTPUT_ROOT
        / "metadata.csv"
    )

    with open(
        metadata_path,
        "w",
        newline="",
    ) as metadata_file:

        writer = csv.DictWriter(
            metadata_file,
            fieldnames=[
                "sample_id",
                "trajectory",
                "scene",
                "step",
                "distance_to_goal",
                "mse",
                "xy_mse",
                "yaw_mse",
            ],
        )

        writer.writeheader()

        # ====================================================
        # DAgger trajectories
        # ====================================================

        for trajectory_idx, row in enumerate(
            failures
        ):

            trajectory_name = (
                row["trajectory"]
            )

            print(
                f"\n[{trajectory_idx + 1}/"
                f"{len(failures)}] "
                f"{trajectory_name}"
            )

            episode_path = (
                trajectory_root
                / trajectory_name
            )

            if not episode_path.exists():
                print(
                    f"SKIP: {episode_path}"
                )
                continue

            scene_path = resolve_scene(
                scene_root,
                trajectory_name,
            )

            print(
                f"Scene: {scene_path}"
            )

            episode_runner = None

            try:

                # --------------------------------------------
                # EXACT SAME Episode construction as main.py.
                # --------------------------------------------

                episode_runner = (
                    task_setup.Episode(
                        args,
                        episode_path,
                        str(scene_path),
                        results_root,
                        preload_data,
                    )
                )

                for step in range(
                    args.max_steps
                ):

                    if (
                        episode_runner.is_done()
                    ):
                        print(
                            f"Reached goal at "
                            f"step {step}"
                        )
                        break

                    # ----------------------------------------
                    # Observation
                    # ----------------------------------------

                    observations = (
                        episode_runner.sim
                        .get_sensor_observations()
                    )

                    (
                        display_img,
                        depth,
                        semantic_instance_sim,
                    ) = utils.split_observations(
                        observations
                    )

                    # ----------------------------------------
                    # Goal
                    # ----------------------------------------

                    episode_runner.get_goal(
                        display_img,
                        depth,
                        semantic_instance_sim,
                    )

                    # ----------------------------------------
                    # Traversability
                    # ----------------------------------------

                    if not args.infer_traversable:

                        episode_runner.traversable_mask = (
                            utils.get_traversibility(
                                torch.from_numpy(
                                    semantic_instance_sim
                                ),
                                episode_runner
                                .traversable_class_indices,
                            ).numpy()
                        )

                    # ----------------------------------------
                    # Student policy
                    # ----------------------------------------

                    episode_runner.get_control_signal(
                        step,
                        display_img,
                        depth,
                    )

                    student_action = (
                        episode_runner
                        .goal_controller
                        .action_pred
                        .copy()
                    )

                    # ----------------------------------------
                    # Expert policy
                    # ----------------------------------------

                    state = (
                        episode_runner
                        .agent
                        .get_state()
                    )

                    current_position = np.asarray(
                        state.position,
                        dtype=np.float32,
                    )

                    current_yaw = (
                        quaternion_to_yaw(
                            state.rotation
                        )
                    )

                    (
                        expert_distance,
                        expert_path,
                    ) = (
                        utils_sim_traj
                        .find_shortest_path(
                            episode_runner.sim,
                            current_position,
                            episode_runner
                            .final_goal_position,
                        )
                    )

                    expert_action = (
                        make_expert_action(
                            expert_path,
                            current_position,
                            current_yaw,
                        )
                    )

                    if expert_action is None:
                        print(
                            f"WARNING: no expert "
                            f"action at step {step}"
                        )
                    else:

                        # ------------------------------------
                        # Save every Nth state.
                        # ------------------------------------

                        if (
                            step
                            % args_cli.sample_every
                            == 0
                        ):

                            sample_id = (
                                f"{trajectory_idx:03d}_"
                                f"{step:04d}"
                            )

                            goal_masks = (
                                episode_runner
                                .control_input_learnt[0]
                            )

                            goal_pls = (
                                episode_runner
                                .control_input_learnt[1]
                            )

                            save_sample(
                                sample_id,
                                display_img,
                                goal_masks,
                                goal_pls,
                                student_action,
                                expert_action,
                                state,
                                expert_distance,
                                trajectory_name,
                                scene_path.parent.name,
                                step,
                                episode_runner.goal_controller.image_history,
                            )

                            diff = (
                                student_action
                                - expert_action
                            )

                            mse = float(
                                np.mean(
                                    diff ** 2
                                )
                            )

                            xy_mse = float(
                                np.mean(
                                    diff[:, :2] ** 2
                                )
                            )

                            yaw_mse = float(
                                np.mean(
                                    diff[:, 2:4] ** 2
                                )
                            )

                            writer.writerow(
                                {
                                    "sample_id":
                                        sample_id,
                                    "trajectory":
                                        trajectory_name,
                                    "scene":
                                        scene_path.parent.name,
                                    "step":
                                        step,
                                    "distance_to_goal":
                                        float(
                                            expert_distance
                                        ),
                                    "mse":
                                        mse,
                                    "xy_mse":
                                        xy_mse,
                                    "yaw_mse":
                                        yaw_mse,
                                }
                            )

                            metadata_file.flush()

                            print(
                                f"  step={step:3d} "
                                f"dist={expert_distance:.3f} "
                                f"mse={mse:.5f}"
                            )

                    # ----------------------------------------
                    # Student executes its action.
                    # ----------------------------------------

                    episode_runner.execute_action()

                print(
                    "Trajectory complete."
                )

            except Exception:
                import traceback

                traceback.print_exc()

                if args.except_exit:
                    raise

            finally:
                if episode_runner is not None:
                    try:
                        episode_runner.close(
                            step
                        )
                    except Exception:
                        pass

    print(
        "\nDAgger pilot collection complete."
    )

    print(
        f"Samples: {SAMPLES_ROOT}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()
