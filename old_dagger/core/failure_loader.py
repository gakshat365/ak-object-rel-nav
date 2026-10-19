"""
Failure Cases Loader and Dataset Splitter for DAgger.

Loads failure episodes from evaluation outputs (e.g., out/failed_cases or
results CSVs), extracts the exact robot failure pose, rotation, and original
goal, and partitions them into Train, Validation, and Held-out Test splits
by scene ID to prevent data leakage.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger("[FailureLoader]")


_SCENE_GLB_CACHE: Dict[Tuple[str, str], Optional[Path]] = {}


def find_scene_glb(hm3d_root: Path, scene_id: str) -> Optional[Path]:
    """Finds the basis.glb scene mesh for a given scene ID across HM3D scenes directories with caching."""
    key = (str(hm3d_root), scene_id)
    if key in _SCENE_GLB_CACHE:
        return _SCENE_GLB_CACHE[key]

    clean_id = scene_id.split("-")[-1].strip()

    # Fast direct path check first (avoids expensive NFS globbing)
    candidates = [hm3d_root]
    for sub in ["train", "val", "minival"]:
        candidates.append(hm3d_root / sub)
    if hm3d_root.parent.exists() and hm3d_root.name != "data":
        for sub in ["train", "val", "minival"]:
            candidates.append(hm3d_root.parent / sub)

    # 1. Direct O(1) file lookups
    for base in candidates:
        direct_path = base / scene_id / f"{clean_id}.basis.glb"
        if direct_path.exists():
            _SCENE_GLB_CACHE[key] = direct_path
            return direct_path
        direct_folder = base / scene_id
        if direct_folder.is_dir():
            for p in direct_folder.iterdir():
                if p.name.endswith(".basis.glb") or (p.suffix == ".glb" and not p.name.endswith(".semantic.glb")):
                    _SCENE_GLB_CACHE[key] = p
                    return p

    # 2. Targeted shallow folder search if scene_id folder differed slightly
    for base in candidates:
        if not base.is_dir():
            continue
        matches = list(base.glob(f"*{clean_id}*/*basis.glb"))
        if matches:
            _SCENE_GLB_CACHE[key] = matches[0]
            return matches[0]

    _SCENE_GLB_CACHE[key] = None
    return None


@dataclass
class FailureCase:
    episode_name: str
    scene_id: str
    failure_position: np.ndarray  # [x, y, z]
    failure_rotation: np.ndarray  # [w, x, y, z] (quaternion)
    goal_position: np.ndarray     # [x, y, z]
    initial_goal_dist: float
    metadata: Dict

    def to_dict(self):
        d = asdict(self)
        d["failure_position"] = self.failure_position.tolist()
        d["failure_rotation"] = self.failure_rotation.tolist()
        d["goal_position"] = self.goal_position.tolist()
        return d


class FailureLoader:
    def __init__(self, config_path: str = "dagger/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.repo_root = Path(self.cfg["paths"].get("repo_root", ".")).resolve()
        self.failed_cases_dir = self.repo_root / self.cfg["paths"].get("failures_cases_dir", "dagger/data/trapped_cases")
        self.failures_source_csv = self.repo_root / self.cfg["paths"].get("failures_source_csv", "out/dagger/failure_trajectories_1300.csv")
        self.hm3d_eval_dir = self.repo_root / self.cfg["paths"].get("hm3d_eval_dir", "./out/hm3d_objectreact_eval")
        self.val_data_dir = self.repo_root / "data/hm3d_iin_val"
        self.train_data_dir = self.repo_root / "data/hm3d_iin_train"
        self.failure_source = self.cfg["dagger"].get("failure_source", "csv")
        self.split_ratios = self.cfg["dagger"]["split_ratios"]
        self.split_by_scene = self.cfg["dagger"].get("split_by_scene", True)
        self.seed = self.cfg["dagger"].get("seed", 42)

    def load_from_csv(self, csv_path: Optional[str] = None) -> List[FailureCase]:
        """
        Loads failure cases from evaluation CSV (e.g., failure_trajectories_1300.csv).
        Resolves failure poses from hm3d_eval_dir/episodes and goals from dataset teacher trajectories.
        """
        import csv
        if csv_path is None:
            csv_path = self.failures_source_csv
        path = Path(csv_path)
        if not path.exists():
            logger.warning(f"Failure source CSV {path} does not exist.")
            return []

        cases = []
        eval_episodes_dir = self.hm3d_eval_dir / "episodes"

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                traj = row.get("trajectory", "").strip()
                if not traj:
                    continue
                scene_col = row.get("scene", "").strip()
                scene_id = scene_col.split("-")[-1] if "-" in scene_col else scene_col

                clean_traj = traj.rstrip("_")
                candidates = list(eval_episodes_dir.glob(f"{clean_traj}*"))
                if not candidates:
                    continue

                ep_eval_dir = candidates[0]
                npz_path = ep_eval_dir / "results_dict.npz"
                csv_path_ep = ep_eval_dir / "results.csv"

                pos = None
                rot_quat = None
                final_dist = float(row.get("final_distance", 0.0) or 0.0)

                if npz_path.exists():
                    try:
                        data = np.load(npz_path, allow_pickle=True)
                        agent_states = data["agent_states"]
                        last_state = agent_states[-1]
                        pos = np.array(last_state.position, dtype=np.float32)
                        rot = last_state.rotation
                        if hasattr(rot, "w"):
                            rot_quat = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float32)
                        elif hasattr(rot, "real"):
                            rot_quat = np.array([rot.real, rot.imag[0], rot.imag[1], rot.imag[2]], dtype=np.float32)
                        else:
                            rot_quat = np.array([rot[0], rot[1], rot[2], rot[3]], dtype=np.float32)
                    except Exception as e:
                        logger.debug(f"Could not load npz from {npz_path}: {e}")

                if pos is None and csv_path_ep.exists():
                    try:
                        with open(csv_path_ep, "r") as cf:
                            lines = [l.strip() for l in cf if l.strip()]
                            if len(lines) > 1:
                                last_row = lines[-1].split(",")
                                pos = np.array([float(last_row[1]), float(last_row[2]), float(last_row[3])], dtype=np.float32)
                                yaw_deg = float(last_row[4])
                                yaw_rad = np.radians(yaw_deg)
                                rot_quat = np.array([np.cos(yaw_rad / 2), 0.0, np.sin(yaw_rad / 2), 0.0], dtype=np.float32)
                    except Exception as e:
                        logger.debug(f"Could not load csv from {csv_path_ep}: {e}")

                if pos is None or rot_quat is None:
                    continue

                goal_pos = None
                for search_dir in [self.train_data_dir, self.val_data_dir]:
                    ep_dir = search_dir / traj
                    if not ep_dir.exists():
                        ep_dir = search_dir / f"{clean_traj}_"
                    if not ep_dir.exists():
                        m = list(search_dir.glob(f"{clean_traj}*"))
                        if m:
                            ep_dir = m[0]
                    if ep_dir.exists() and (ep_dir / "agent_states.npy").exists():
                        try:
                            teach_states = np.load(ep_dir / "agent_states.npy", allow_pickle=True)
                            goal_pos = np.array(teach_states[-1].position, dtype=np.float32)
                            break
                        except Exception:
                            pass

                if goal_pos is None:
                    continue

                fc = FailureCase(
                    episode_name=clean_traj,
                    scene_id=scene_id,
                    failure_position=pos,
                    failure_rotation=rot_quat,
                    goal_position=goal_pos,
                    initial_goal_dist=final_dist,
                    metadata={"source_csv": str(path), "eval_dir": str(ep_eval_dir)},
                )
                cases.append(fc)

        logger.info(f"Loaded {len(cases)} failure cases from {path}")
        return cases

    def load_from_trapped_manifest(self, manifest_csv: Optional[str] = None) -> List[FailureCase]:
        """
        Loads trapped failure cases with 50-step-back recovery poses from trapped cases CSV manifest.
        """
        import csv
        if manifest_csv is None:
            manifest_csv = self.repo_root / self.cfg["paths"].get("trapped_cases_csv", "dagger/data/trapped_cases_manifest.csv")
        path = Path(manifest_csv)
        if not path.exists():
            logger.warning(f"Trapped manifest CSV {path} does not exist.")
            return []

        cases = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep_name = row.get("episode_name", "").strip()
                scene_id = row.get("scene_id", "").strip()
                if not ep_name or not scene_id:
                    continue
                pos = np.array([
                    float(row["rec_pos_x"]),
                    float(row["rec_pos_y"]),
                    float(row["rec_pos_z"])
                ], dtype=np.float32)
                rot = np.array([
                    float(row["rot_w"]),
                    float(row["rot_x"]),
                    float(row["rot_y"]),
                    float(row["rot_z"])
                ], dtype=np.float32)
                rot_norm = float(np.linalg.norm(rot))
                if rot_norm > 1e-6:
                    rot = (rot / rot_norm).astype(np.float32)
                else:
                    rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                goal = np.array([
                    float(row["goal_x"]),
                    float(row["goal_y"]),
                    float(row["goal_z"])
                ], dtype=np.float32)
                init_dist = float(row.get("initial_dist", 0.0) or 0.0)

                fc = FailureCase(
                    episode_name=ep_name,
                    scene_id=scene_id,
                    failure_position=pos,
                    failure_rotation=rot,
                    goal_position=goal,
                    initial_goal_dist=init_dist,
                    metadata={
                        "trap_step": int(row.get("trap_step", -1)),
                        "recovery_step": int(row.get("recovery_step", -1)),
                        "final_dist": float(row.get("final_dist", 0.0)),
                    }
                )
                cases.append(fc)

        logger.info(f"Loaded {len(cases)} trapped recovery cases from {path}")
        return cases

    def load_cases(self) -> List[FailureCase]:
        """Loads failure cases according to config failure_source ('trapped_csv', 'csv', or 'dir')."""
        if self.failure_source == "trapped_csv":
            cases = self.load_from_trapped_manifest()
            if len(cases) > 0:
                return cases
            logger.warning("No cases loaded from trapped CSV; falling back to dir loading.")
        elif self.failure_source == "csv":
            cases = self.load_from_csv()
            if len(cases) > 0:
                return cases
            logger.warning("No cases loaded from CSV; falling back to directory loading.")
        return self.load_from_failed_cases_dir(force_dir=True)

    def load_from_failed_cases_dir(self, force_dir: bool = False) -> List[FailureCase]:
        """
        Loads recorded failure cases from out/failed_cases directory.
        Each subdirectory contains a case.npz with full AgentState history.
        """
        if not force_dir and self.failure_source == "trapped_csv":
            cases = self.load_from_trapped_manifest()
            if len(cases) > 0:
                return cases
            logger.warning("No cases loaded from trapped CSV; falling back to failed_cases_dir.")
        elif not force_dir and self.failure_source == "csv":
            cases = self.load_from_csv()
            if len(cases) > 0:
                return cases
            logger.warning("No cases loaded from CSV; falling back to failed_cases_dir.")

        cases = []
        if not self.failed_cases_dir.exists():
            logger.warning(f"Failed cases dir {self.failed_cases_dir} does not exist.")
            return cases

        for case_dir in sorted(self.failed_cases_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            npz_path = case_dir / "case.npz"
            if not npz_path.exists():
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)
                episode_name = str(data["episode_name"])
                scene_id = str(data["scene_id"])

                # Check if failure/recovery pose is directly stored
                if "failure_position" in data and "failure_rotation" in data:
                    pos = np.array(data["failure_position"], dtype=np.float32)
                    rot_quat = np.array(data["failure_rotation"], dtype=np.float32)
                else:
                    agent_states = data["agent_states"]
                    last_state = agent_states[-1]
                    pos = np.array(last_state.position, dtype=np.float32)
                    rot = last_state.rotation
                    if hasattr(rot, "w"):
                        rot_quat = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float32)
                    elif hasattr(rot, "real"):
                        rot_quat = np.array([rot.real, rot.imag[0], rot.imag[1], rot.imag[2]], dtype=np.float32)
                    else:
                        rot_quat = np.array([rot[0], rot[1], rot[2], rot[3]], dtype=np.float32)

                # Resolve goal position from trajectory mapping data
                clean_ep_name = episode_name.replace("__learnt_gt_topometric", "")
                ep_dir = self.val_data_dir / clean_ep_name
                if not ep_dir.exists():
                    ep_dir = self.val_data_dir / f"{clean_ep_name}_"
                if not ep_dir.exists():
                    matches = list(self.val_data_dir.glob(f"*{clean_ep_name}*"))
                    if matches:
                        ep_dir = matches[0]

                goal_pos = None

                if "goal_position" in data:
                    goal_pos = np.array(data["goal_position"], dtype=np.float32)
                elif ep_dir.exists() and (ep_dir / "agent_states.npy").exists():
                    teach_states = np.load(ep_dir / "agent_states.npy", allow_pickle=True)
                    goal_pos = np.array(teach_states[-1].position, dtype=np.float32)
                else:
                    # Fallback to metadata
                    meta_json_path = case_dir / "metadata.json"
                    if meta_json_path.exists():
                        with open(meta_json_path, "r") as mf:
                            meta_dict = json.load(mf)
                            if "goal_position" in meta_dict:
                                goal_pos = np.array(meta_dict["goal_position"], dtype=np.float32)
                            else:
                                goal_pos = pos.copy()

                if goal_pos is None:
                    continue

                dists = data["distance_to_goal"]
                final_dist = float(dists[-1]) if len(dists) > 0 else 0.0

                fc = FailureCase(
                    episode_name=clean_ep_name,
                    scene_id=scene_id,
                    failure_position=pos,
                    failure_rotation=rot_quat,
                    goal_position=goal_pos,
                    initial_goal_dist=final_dist,
                    metadata={"source_dir": str(case_dir)},
                )
                cases.append(fc)

            except Exception as e:
                logger.error(f"Error reading {npz_path}: {e}")
                continue

        logger.info(f"Loaded {len(cases)} failure cases from {self.failed_cases_dir}")
        return cases

    def split_cases(
        self, cases: List[FailureCase]
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """
        Partitions cases into Train, Validation, and Held-out Test sets.
        Splits by scene_id to guarantee no scene overlap (leakage prevention).
        """
        if len(cases) == 0:
            return [], [], []

        np.random.seed(self.seed)

        if self.split_by_scene:
            unique_scenes = sorted(list(set(c.scene_id for c in cases)))
            np.random.shuffle(unique_scenes)

            n_scenes = len(unique_scenes)
            n_train = max(1, int(n_scenes * self.split_ratios["train"]))
            n_val = max(1, int(n_scenes * self.split_ratios["val"]))

            train_scenes = set(unique_scenes[:n_train])
            val_scenes = set(unique_scenes[n_train : n_train + n_val])
            test_scenes = set(unique_scenes[n_train + n_val :])
            if len(test_scenes) == 0:
                # If very few scenes, share val or reserve last scene
                test_scenes = {unique_scenes[-1]}

            train_cases = [c for c in cases if c.scene_id in train_scenes]
            val_cases = [c for c in cases if c.scene_id in val_scenes]
            test_cases = [c for c in cases if c.scene_id in test_scenes]

        else:
            shuffled = list(cases)
            np.random.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * self.split_ratios["train"])
            n_val = int(n * self.split_ratios["val"])

            train_cases = shuffled[:n_train]
            val_cases = shuffled[n_train : n_train + n_val]
            test_cases = shuffled[n_train + n_val :]

        logger.info(
            f"Split {len(cases)} failure cases -> Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}"
        )
        return train_cases, val_cases, test_cases
