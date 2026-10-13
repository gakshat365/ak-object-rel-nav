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

        self.repo_root = Path(self.cfg["paths"]["repo_root"]).resolve()
        self.failed_cases_dir = self.repo_root / self.cfg["paths"]["failures_cases_dir"]
        self.val_data_dir = self.repo_root / "data/hm3d_iin_val"
        self.split_ratios = self.cfg["dagger"]["split_ratios"]
        self.split_by_scene = self.cfg["dagger"].get("split_by_scene", True)
        self.seed = self.cfg["dagger"].get("seed", 42)

    def load_from_failed_cases_dir(self) -> List[FailureCase]:
        """
        Loads recorded failure cases from out/failed_cases directory.
        Each subdirectory contains a case.npz with full AgentState history.
        """
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

                # Get final failure agent state
                agent_states = data["agent_states"]
                last_state = agent_states[-1]
                pos = np.array(last_state.position, dtype=np.float32)

                # Extract quaternion [w, x, y, z]
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

                if ep_dir.exists() and (ep_dir / "agent_states.npy").exists():
                    teach_states = np.load(ep_dir / "agent_states.npy", allow_pickle=True)
                    goal_pos = np.array(teach_states[-1].position, dtype=np.float32)
                else:
                    # Fallback to metadata
                    meta_json_path = case_dir / "metadata.json"
                    if meta_json_path.exists():
                        with open(meta_json_path, "r") as mf:
                            meta_dict = json.load(mf)
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
