"""deploy_pi05.py — real-robot client for the pi0.5 multi-task policy.

The policy is served by openpi's websocket server. This script talks to it,
gets an action chunk of shape (10, 7) per call, and applies each step on
the robot at ~15 Hz (the training fps).

Action layout (each row of the chunk):
    [dx_eef, dy_eef, dz_eef, drx_eef, dry_eef, drz_eef, gripper]
EE-frame delta + gripper, same convention as the training data
(`pycontroller_template.controls.chunk_compose.decompose_to_eef_deltas`).

Replace the four `# TODO` hooks (cameras, joints, EE-delta apply) with your
real hardware before running.
"""
from __future__ import annotations

import collections
import time

import numpy as np
from openpi_client import websocket_client_policy


# --------------------------------------------------------------------- config
SERVER_HOST = "127.0.0.1"      # GPU machine IP (or localhost if same box)
SERVER_PORT = 8000
CONTROL_HZ = 15                # match training fps (do NOT change unless dataset fps differs)
REPLAN_EVERY = 5               # run only the first N steps of each (10,) chunk, then re-infer

# 4 tasks the multi-task model was trained on (exact strings from tasks.parquet).
TASK_PROMPTS = {
    "pick_and_place": "Pick up the cube and place it on the plate.",
    "chocomilk":      "Pick up the chocolate milk and place it on top of the white milk, then pick up the cube and stack it on top of the chocolate milk.",
    "pot":            "Open the lid of the pot, put the yellow cube inside the pot, and close the lid again.",
    "kitchen":        "Pick up the pot and place it on the bottom section of the induction cooktop, then pick up the pan and place it on the top section of the induction cooktop.",
}
TASK = "pick_and_place"        # which task to execute


# ---------------------------------------------------------- hardware hooks (TODO)
def get_external_image() -> np.ndarray:
    """Return RGB uint8 ndarray (H, W, 3). Any size — the server resizes to 224x224."""
    raise NotImplementedError("hook your external (3rd-person) RGB camera here")


def get_wrist_image() -> np.ndarray:
    """Return RGB uint8 ndarray (H, W, 3)."""
    raise NotImplementedError("hook your wrist RGB camera here")


def get_joint_angles() -> np.ndarray:
    """Return current Panda joint angles, shape (7,) float32, radians."""
    raise NotImplementedError("hook your robot joint encoder read here")


def apply_ee_delta(delta_xyz_rpy: np.ndarray, gripper: float) -> None:
    """Apply one step of EE-frame action to the robot.

    delta_xyz_rpy : ndarray (6,)
        dx, dy, dz in metres + axis-angle rotation delta (drx, dry, drz) in radians,
        expressed in the *current* EE frame.
    gripper : float
        in [0, 1] (training convention: 0 = open, 1 = close).

    Typical implementation:
        current_pose = robot.read_ee_pose()                  # SE(3)
        target_pose  = compose_ee_delta(current_pose, delta) # inverse of decompose_to_eef_deltas
        joint_cmd    = ik_solver(target_pose)
        robot.servo(joint_cmd)
        robot.set_gripper(gripper)
    """
    raise NotImplementedError("hook your EE-delta integrator + IK + gripper here")


# ------------------------------------------------------------------------ main
def main() -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
    print(f"[deploy_pi05] connected to ws://{SERVER_HOST}:{SERVER_PORT}; metadata={client.get_server_metadata()}")

    prompt = TASK_PROMPTS[TASK]
    plan: collections.deque = collections.deque()
    period = 1.0 / CONTROL_HZ

    while True:
        loop_t = time.time()

        if not plan:
            obs = {
                "observation/image":       get_external_image(),
                "observation/wrist_image": get_wrist_image(),
                "observation/state":       np.asarray(get_joint_angles(), dtype=np.float32),
                "prompt":                  prompt,
            }
            chunk = np.asarray(client.infer(obs)["actions"], dtype=np.float32)  # (10, 7)
            assert chunk.shape == (10, 7), f"unexpected chunk shape {chunk.shape}"
            plan.extend(chunk[:REPLAN_EVERY])

        action = plan.popleft()  # (7,)
        apply_ee_delta(action[:6], gripper=float(action[6]))

        # pace the control loop
        slept = time.time() - loop_t
        if slept < period:
            time.sleep(period - slept)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[deploy_pi05] stopped by user")
