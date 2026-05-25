"""deploy_openvla.py — real-robot client for the OpenVLA policy.

OpenVLA, unlike openpi/SF, doesn't ship with a separate websocket server in
this repo: the model (~14 GB) is loaded directly into this script's GPU
memory and queried in-process. For a server/client split, wrap
`vla.predict_action` in your own FastAPI/socket server.

OpenVLA returns a single 7-dim action per query (no chunking), so the control
loop calls `predict_action` at every step (~15 Hz). Wrist camera is not used
by OpenVLA (single-image model); train a separate run with `--image_video_key
observation.images.wrist_image` if you want a wrist-only policy.

Action layout:  [dx_eef, dy_eef, dz_eef, drx_eef, dry_eef, drz_eef, gripper].
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# openvla repo's HF-registered classes (must be importable in this env)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# --------------------------------------------------------------------- config
RUN_DIR = Path("/home/cvlab/project/realsangbeom/robot/openvla/runs/<exp_id>--lerobot")
# point at the merged full model dir produced by finetune_lerobot.py (NOT the LoRA adapter)
DATASET_NAME = "local__jisang_combined"  # the --dataset_name used at training time
DEVICE = "cuda:0"
CONTROL_HZ = 15

TASK_PROMPTS = {
    "pick_and_place": "Pick up the cube and place it on the plate.",
    "chocomilk":      "Pick up the chocolate milk and place it on top of the white milk, then pick up the cube and stack it on top of the chocolate milk.",
    "pot":            "Open the lid of the pot, put the yellow cube inside the pot, and close the lid again.",
    "kitchen":        "Pick up the pot and place it on the bottom section of the induction cooktop, then pick up the pan and place it on the top section of the induction cooktop.",
}
TASK = "pick_and_place"


# ---------------------------------------------------------- hardware hooks (TODO)
def get_external_image() -> Image.Image:
    """Return a PIL.Image (RGB). The processor handles resize/normalisation."""
    raise NotImplementedError("hook your external RGB camera here (return PIL.Image)")


def get_joint_angles() -> np.ndarray:
    """Currently unused by single-image OpenVLA, but kept here for symmetry / logging."""
    raise NotImplementedError("hook your robot joint encoders here")


def apply_ee_delta(delta_xyz_rpy: np.ndarray, gripper: float) -> None:
    """See deploy_pi05.py.apply_ee_delta — identical semantics."""
    raise NotImplementedError("hook your EE-delta integrator + IK + gripper here")


# ---------------------------------------------------------- model loading
def load_model():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(RUN_DIR, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        RUN_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla.eval()

    # dataset_statistics.json is saved next to the model dir by finetune_lerobot.py;
    # OpenVLA's predict_action looks it up via `unnorm_key` so we just check it exists.
    stats_path = RUN_DIR / "dataset_statistics.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"missing {stats_path}; the merged model must keep dataset_statistics.json "
            f"so that predict_action(..., unnorm_key='{DATASET_NAME}') works"
        )
    stats = json.loads(stats_path.read_text())
    if DATASET_NAME not in stats:
        raise KeyError(
            f"dataset key '{DATASET_NAME}' not found in {stats_path}; "
            f"available: {list(stats.keys())}"
        )
    return processor, vla


# ------------------------------------------------------------------------ main
def main() -> None:
    print(f"[deploy_openvla] loading {RUN_DIR} on {DEVICE} ...")
    processor, vla = load_model()
    print("[deploy_openvla] ready")

    prompt = TASK_PROMPTS[TASK].lower()
    period = 1.0 / CONTROL_HZ

    while True:
        loop_t = time.time()

        img = get_external_image()
        text = f"In: What action should the robot take to {prompt}?\nOut:"
        inputs = processor(text, img).to(DEVICE, dtype=torch.bfloat16)

        with torch.no_grad():
            action = vla.predict_action(**inputs, unnorm_key=DATASET_NAME, do_sample=False)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape == (7,), f"unexpected action shape {action.shape}"

        apply_ee_delta(action[:6], gripper=float(action[6]))

        slept = time.time() - loop_t
        if slept < period:
            time.sleep(period - slept)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[deploy_openvla] stopped by user")
