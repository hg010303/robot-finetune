"""deploy_sf.py — real-robot client for the Spatial-Forcing (pi0 + VGGT alignment) policy.

Same client/server pattern as deploy_pi05.py: the SF policy is served by the
same openpi websocket server (`scripts/serve_policy.py`), so the client side
code is identical — only the checkpoint and config name differ.

Action layout (per step of the (10, 7) chunk):
    [dx_eef, dy_eef, dz_eef, drx_eef, dry_eef, drz_eef, gripper]
"""
from __future__ import annotations

import collections
import time

import numpy as np
from openpi_client import websocket_client_policy


# --------------------------------------------------------------------- config
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000             # different port than pi0.5 if you serve both at once
CONTROL_HZ = 15
REPLAN_EVERY = 5

TASK_PROMPTS = {
    "pick_and_place": "Pick up the cube and place it on the plate.",
    "chocomilk":      "Pick up the chocolate milk and place it on top of the white milk, then pick up the cube and stack it on top of the chocolate milk.",
    "pot":            "Open the lid of the pot, put the yellow cube inside the pot, and close the lid again.",
    "kitchen":        "Pick up the pot and place it on the bottom section of the induction cooktop, then pick up the pan and place it on the top section of the induction cooktop.",
}
TASK = "pick_and_place"


# ---------------------------------------------------------- hardware hooks (TODO)
def get_external_image() -> np.ndarray:
    raise NotImplementedError("hook your external camera here")


def get_wrist_image() -> np.ndarray:
    raise NotImplementedError("hook your wrist camera here")


def get_joint_angles() -> np.ndarray:
    raise NotImplementedError("hook your robot joint encoders here")


def apply_ee_delta(delta_xyz_rpy: np.ndarray, gripper: float) -> None:
    """See deploy_pi05.py.apply_ee_delta — identical semantics."""
    raise NotImplementedError("hook your EE-delta integrator + IK + gripper here")


# ------------------------------------------------------------------------ main
def main() -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
    print(f"[deploy_sf] connected to ws://{SERVER_HOST}:{SERVER_PORT}; metadata={client.get_server_metadata()}")

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

        action = plan.popleft()
        apply_ee_delta(action[:6], gripper=float(action[6]))

        slept = time.time() - loop_t
        if slept < period:
            time.sleep(period - slept)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[deploy_sf] stopped by user")
