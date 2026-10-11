import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np
import yaml

from libs.experiments import task_setup
from libs.common import utils


def resolve_scene(path_scenes_root_hm3d, episode_path):
    """
    Resolve a trajectory to its HM3D .basis.glb scene using the
    same naming convention as main.py.
    """
    episode_name = episode_path.parts[-1].split("_")[0]

    matches = sorted(path_scenes_root_hm3d.glob(f"*{episode_name}"))

    if not matches:
        raise FileNotFoundError(
            f"No HM3D scene found for episode {episode_path.name} "
            f"(scene id: {episode_name})"
        )

    scene_files = sorted(matches[0].glob("*basis.glb"))

    if not scene_files:
        raise FileNotFoundError(
            f"No basis.glb found in scene directory {matches[0]}"
        )

    return str(scene_files[0])


def evaluate_episode(
    args,
    episode_path,
    scene_path,
    preload_data,
    results_dir,
):
    """
    Run one trajectory using the existing Episode evaluation machinery.
    """

    episode_runner = None

    result = {
        "trajectory": episode_path.name,
        "scene": Path(scene_path).parent.name,
        "status": "error",
        "steps": 0,
        "final_distance": None,
        "error": "",
    }

    try:
        episode_runner = task_setup.Episode(
            args,
            episode_path,
            scene_path,
            results_dir,
            preload_data,
        )


        for step in range(args.max_steps):

            # Check whether we have already reached the goal.
            if episode_runner.is_done():
                break

            # Get simulator observation.
            observations = episode_runner.sim.get_sensor_observations()

            display_img, depth, semantic_instance_sim = (
                utils.split_observations(observations)
            )

            # Only used if explicitly enabled.
            if args.infer_depth:
                depth = (
                    preload_data["depth_model"].infer(display_img)
                    * 0.44
                )

            # Generate the gt_metric goal input used by ObjectReact.
            episode_runner.get_goal(
                display_img,
                depth,
                semantic_instance_sim,
            )

            # ObjectReact inference.
            episode_runner.get_control_signal(
                step,
                display_img,
                depth,
            )

            # Execute predicted velocity.
            episode_runner.execute_action()

            result["steps"] = step + 1

        # If the loop ended because max_steps was reached,
        # perform the final success check.
        if episode_runner.success_status != "success":
            episode_runner.is_done()

        result["status"] = episode_runner.success_status

        if episode_runner.distance_to_goal is not None:
            result["final_distance"] = float(
                episode_runner.distance_to_goal
            )

    except Exception as e:

        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"

        print(
            f"\nERROR: {episode_path.name}\n"
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

    finally:

        if episode_runner is not None:
            try:
                episode_runner.close(result["steps"])
            except Exception as e:
                print(
                    f"Warning: failed to close "
                    f"{episode_path.name}: {e}"
                )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/object_react.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--trajectory-root",
        default="data/hm3d_iin_train",
    )

    parser.add_argument(
        "--scene-root",
        default="data/hm3d_v0.2/train",
    )

    parser.add_argument(
        "--output-dir",
        default="out/hm3d_objectreact_eval",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
    )

    args_cli = parser.parse_args()

    # ---------------------------------------------------------
    # Load ObjectReact configuration.
    # ---------------------------------------------------------

    with open(args_cli.config, "r") as f:
        config = yaml.safe_load(f)

    args = argparse.Namespace(**config)

    # These must remain dictionaries because the existing
    # ObjectReact code accesses them using ["key"].
    args.controller = config["controller"]
    args.goal_gen = config["goal_gen"]
    args.sim = config["sim"]

    # ---------------------------------------------------------
    # Evaluation configuration.
    # ---------------------------------------------------------

    args.method = "learnt"
    args.goal_source = "gt_metric"
    args.task_type = "original"

    args.env = "sim"
    args.split = "train"

    args.max_steps = 300
    args.threshold_goal_distance = 1.0

    args.path_dataset = (
        args_cli.trajectory_root.rsplit(
            "/hm3d_iin_train",
            1,
        )[0]
        + "/"
    )

    args.path_models = config.get("path_models", "")
    args.path_results = args_cli.output_dir

    args.run_list = ""
    args.path_run = ""

    args.start_idx = args_cli.start

    args.end_idx = (
        args_cli.end
        if args_cli.end is not None
        else 2723
    )

    args.step_idx = 1

    args.exp_name = "hm3d_objectreact_eval"

    args.reverse = False
    args.use_gt_localization = False

    args.infer_depth = False
    args.infer_traversable = False

    args.log_wandb = False

    # IMPORTANT:
    # We don't need per-step robot logging for this evaluation.
    args.log_robot = True

    args.debug = False
    args.except_exit = False

    # Same list used by main.py.
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

    args.plot = False
    args.save_vis = False

    args.cull_qry_instances = False
    args.cull_map_instances = False

    # ObjectReact checkpoint.
    args.controller["load_run"] = args_cli.checkpoint

    # ---------------------------------------------------------
    # Paths.
    # ---------------------------------------------------------

    trajectory_root = Path(args_cli.trajectory_root)
    scene_root = Path(args_cli.scene_root)
    output_dir = Path(args_cli.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir = output_dir / "episodes"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Find all trajectories.
    # ---------------------------------------------------------

    trajectories = sorted(
        p
        for p in trajectory_root.iterdir()
        if p.is_dir()
    )

    total = len(trajectories)

    print(f"Found {total} trajectories.")

    end = min(
        args_cli.end
        if args_cli.end is not None
        else total,
        total,
    )

    trajectories = trajectories[
        args_cli.start:end
    ]

    print(
        f"Evaluating trajectories "
        f"[{args_cli.start}:{end}] "
        f"({len(trajectories)} episodes)"
    )

    # ---------------------------------------------------------
    # Load existing results so interrupted runs can resume.
    # ---------------------------------------------------------

    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"

    completed = set()

    if csv_path.exists():

        with open(
            csv_path,
            "r",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                if (
                    row.get("trajectory")
                    and row.get("status") in ("success", "exceeded_steps")
                ):
                    completed.add(row["trajectory"])

    print(
        f"Already completed: "
        f"{len(completed)} trajectories"
    )

    # ---------------------------------------------------------
    # Preload ObjectReact once.
    # ---------------------------------------------------------

    print("Loading ObjectReact controller...")

    preload_data = task_setup.preload_models(args)

    print("Controller loaded.")

    # ---------------------------------------------------------
    # Open results CSV in append mode.
    # ---------------------------------------------------------

    file_exists = csv_path.exists()

    fieldnames = [
        "trajectory",
        "scene",
        "status",
        "steps",
        "final_distance",
        "error",
    ]

    with open(
        csv_path,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()
            f.flush()

        # -----------------------------------------------------
        # Evaluate trajectories.
        # -----------------------------------------------------

        for i, episode_path in enumerate(
            trajectories
        ):

            # Skip episodes already present in CSV.
            if episode_path.name in completed:

                print(
                    f"\n[{i + args_cli.start + 1}/{total}] "
                    f"{episode_path.name} "
                    f"[SKIP - already completed]"
                )

                continue

            print(
                f"\n[{i + args_cli.start + 1}/{total}] "
                f"{episode_path.name}"
            )

            # -------------------------------------------------
            # Resolve HM3D scene.
            # -------------------------------------------------

            try:

                scene_path = resolve_scene(
                    scene_root,
                    episode_path,
                )

            except Exception as e:

                result = {
                    "trajectory": episode_path.name,
                    "scene": "",
                    "status": "error",
                    "steps": 0,
                    "final_distance": None,
                    "error": (
                        f"{type(e).__name__}: {e}"
                    ),
                }

                print(
                    f"  SCENE ERROR: "
                    f"{result['error']}"
                )

                writer.writerow(result)
                f.flush()

               

                continue

            # -------------------------------------------------
            # Run episode.
            # -------------------------------------------------

            result = evaluate_episode(
                args,
                episode_path,
                scene_path,
                preload_data,
                results_dir,
            )

            # Write immediately so an interrupted run
            # doesn't lose the current result.
            writer.writerow(result)
            f.flush()

            if result["status"] in ("success", "exceeded_steps"):
                completed.add(episode_path.name)

            # -------------------------------------------------
            # Print result.
            # -------------------------------------------------

            if result["status"] == "success":

                print(
                    f"  SUCCESS | "
                    f"steps={result['steps']} | "
                    f"distance="
                    f"{result['final_distance']:.3f}m"
                )

            elif result["status"] == "exceeded_steps":

                distance = result["final_distance"]

                if distance is not None:

                    print(
                        f"  FAILED | "
                        f"steps={result['steps']} | "
                        f"distance="
                        f"{distance:.3f}m"
                    )

                else:

                    print(
                        f"  FAILED | "
                        f"steps={result['steps']}"
                    )

            else:

                print(
                    f"  ERROR | "
                    f"{result['error']}"
                )

    # ---------------------------------------------------------
    # Recalculate summary from the COMPLETE CSV.
    # ---------------------------------------------------------

    all_results = []

    if csv_path.exists():

        with open(
            csv_path,
            "r",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                all_results.append(row)

    success_count = sum(
        r["status"] == "success"
        for r in all_results
    )

    failure_count = sum(
        r["status"] == "exceeded_steps"
        for r in all_results
    )

    error_count = sum(
        r["status"] == "error"
        for r in all_results
    )

    evaluated = len(all_results)

    summary = {
        "total_trajectories_found": total,
        "evaluated": evaluated,
        "success": success_count,
        "failure": failure_count,
        "error": error_count,
        "success_rate_over_evaluated": (
            success_count / evaluated
            if evaluated > 0
            else 0.0
        ),
        "checkpoint": args_cli.checkpoint,
        "goal_source": args.goal_source,
        "method": args.method,
        "task_type": args.task_type,
        "threshold_goal_distance": (
            args.threshold_goal_distance
        ),
        "max_steps": args.max_steps,
    }

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("HM3D OBJECTREACT EVALUATION COMPLETE")
    print("=" * 60)

    print(f"Found:     {total}")
    print(f"Evaluated: {evaluated}")
    print(f"Success:   {success_count}")
    print(f"Failure:   {failure_count}")
    print(f"Errors:    {error_count}")

    if evaluated:

        print(
            f"Success rate: "
            f"{100 * success_count / evaluated:.2f}%"
        )

    print()
    print(f"Results: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()