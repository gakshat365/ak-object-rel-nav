import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import quaternion
import yaml
import sys
from pathlib import Path

REPO_ROOT = Path("/scratch2/akshat.gupta/ak-object-rel-nav")
sys.path.insert(0, str(REPO_ROOT))

from libs.common import utils
from dagger.core.failure_loader import FailureLoader, find_scene_glb
from libs.control.objectreact import ObjRelLearntController

loader = FailureLoader(str(REPO_ROOT / "dagger/config.yaml"))
all_cases = loader.load_from_csv()
_, _, test_cases = loader.split_cases(all_cases)
case = test_cases[0]

with open(REPO_ROOT / "dagger/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

hm3d_root = REPO_ROOT / cfg["paths"]["hm3d_scenes_path"]
scene_glb = find_scene_glb(hm3d_root, case.scene_id)

sim, agent, _ = utils.get_sim_agent(
    str(scene_glb),
    updateNavMesh=False,
    width=320,
    height=240,
    hfov=120,
    sensor_height=0.4,
)

ctrl_cfg_path = REPO_ROOT / cfg["paths"]["controller_config"]
with open(ctrl_cfg_path, "r") as f:
    ctrl_cfg = yaml.safe_load(f)
ctrl_cfg["load_run"] = str(REPO_ROOT / "dagger/checkpoints/policy_iter_2.pth")
controller = ObjRelLearntController(ctrl_cfg, goal_source="gt_topometric")

# Set failure state
start_state = agent.get_state()
start_state.position = case.failure_position
q = case.failure_rotation
start_state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
agent.set_state(start_state)

print(f"\n--- Testing what neural policy predicts as robot rotates 360 degrees ---")
for rot_step in range(12):
    angle_deg = rot_step * 30
    # Apply rotation
    curr_state = agent.get_state()
    rot = quaternion.from_rotation_vector([0, np.radians(angle_deg), 0])
    curr_state.rotation = rot * quaternion.quaternion(q[0], q[1], q[2], q[3])
    agent.set_state(curr_state)

    obs = sim.get_sensor_observations()
    display_img, depth, semantic = utils.split_observations(obs)

    curr_dist = 5.0
    unique_classes = np.unique(semantic)
    masks = semantic[None, ...] == unique_classes[:, None, None]
    pls = np.linspace(curr_dist, curr_dist + 4.0, len(unique_classes), dtype=np.float32)
    control_input = [masks, pls]

    v, w, _ = controller.predict(display_img, control_input)
    preds = controller.action_pred

    wp0 = preds[0]
    wp1 = preds[1]
    wp9 = preds[-1]

    th0 = np.degrees(np.arctan2(wp0[1], wp0[0]))
    th1 = np.degrees(np.arctan2(wp1[1], wp1[0]))
    th9 = np.degrees(np.arctan2(wp9[1], wp9[0]))

    print(f"Robot Rot {angle_deg:3d}°: wp0=[{wp0[0]:.2f}, {wp0[1]:.2f}] (th0={th0:5.1f}°) | "
          f"wp1=[{wp1[0]:.2f}, {wp1[1]:.2f}] (th1={th1:5.1f}°) | "
          f"wp9=[{wp9[0]:.2f}, {wp9[1]:.2f}] (th9={th9:5.1f}°)")

sim.close()
