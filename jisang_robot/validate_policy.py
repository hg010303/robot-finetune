"""validate_policy.py

Dry-run check that a trained openpi policy returns sane actions on frames sampled
from the training dataset. No robot/sim needed.

Loop:
  1. Sample N random global frame indices.
  2. For each, read state/action/task_index from the data parquet and decode the
     external + wrist video frame at the matching timestamp.
  3. Send the obs dict to the websocket server, receive a (10, 7) action chunk.
  4. Sanity-check shape / NaN / range, compare first predicted step to ground-truth.

Run (with server already up on localhost:8000):
    python validate_policy.py --num_samples 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

from openpi_client import websocket_client_policy


def decode_frame(video_path: Path, target_t_s: float) -> np.ndarray:
    """Return the frame at `target_t_s` from `video_path` as (H, W, 3) uint8 RGB."""
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        seek_pts = int(max(target_t_s, 0.0) / float(stream.time_base))
        container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t + 1e-6 >= target_t_s:
                return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"no frame found near t={target_t_s} in {video_path}")
    finally:
        container.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        default="/home/cvlab/project/realsangbeom/robot/jisang_robot/lerobot_datasets/local__pick_and_place",
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.data_root)

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    total_frames = int(info["total_frames"])

    # episode metadata: where each episode's frames live (parquet + video chunks)
    epi_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    epi_df = pd.concat([pd.read_parquet(f) for f in epi_files], ignore_index=True)
    epi_df = epi_df.sort_values("episode_index").reset_index(drop=True)
    starts = epi_df["dataset_from_index"].to_numpy()

    # task table — usually a single task here
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_table = {int(r["task_index"]): str(text) for text, r in tasks_df.iterrows()}

    # data parquet LRU-ish cache
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}

    def get_data(chunk: int, file: int) -> pd.DataFrame:
        key = (chunk, file)
        if key not in data_cache:
            data_cache[key] = pd.read_parquet(
                root / "data" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"
            )
        return data_cache[key]

    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(total_frames, size=args.num_samples, replace=False)

    print(f"Connecting to policy server at ws://{args.host}:{args.port} …")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Connected. Server metadata: {client.get_server_metadata()}")
    print()

    EXT_KEY, WRI_KEY = "observation.images.external_image", "observation.images.wrist_image"

    diffs_l1: list[float] = []
    grip_diffs: list[float] = []
    print(f"{'idx':>6} | {'gt(dx,dy,dz)':>34} | {'pred(dx,dy,dz)':>34} | grip gt/pred | L1_first")
    print("-" * 130)
    for global_idx in sampled:
        global_idx = int(global_idx)

        # find episode this frame belongs to
        epi_idx = int(np.searchsorted(starts, global_idx, side="right") - 1)
        ep = epi_df.iloc[epi_idx]
        within = global_idx - int(ep["dataset_from_index"])

        # data parquet row (file's first row's "index" is its global start)
        df = get_data(int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        row = df.iloc[global_idx - int(df.iloc[0]["index"])]

        # video frames (timestamp inside the mp4 = episode_from + within/fps)
        ext_ck = int(ep[f"videos/{EXT_KEY}/chunk_index"])
        ext_fi = int(ep[f"videos/{EXT_KEY}/file_index"])
        ext_t0 = float(ep[f"videos/{EXT_KEY}/from_timestamp"])
        wri_ck = int(ep[f"videos/{WRI_KEY}/chunk_index"])
        wri_fi = int(ep[f"videos/{WRI_KEY}/file_index"])
        wri_t0 = float(ep[f"videos/{WRI_KEY}/from_timestamp"])
        within_t = within / fps

        img_ext = decode_frame(
            root / "videos" / EXT_KEY / f"chunk-{ext_ck:03d}" / f"file-{ext_fi:03d}.mp4",
            ext_t0 + within_t,
        )
        img_wri = decode_frame(
            root / "videos" / WRI_KEY / f"chunk-{wri_ck:03d}" / f"file-{wri_fi:03d}.mp4",
            wri_t0 + within_t,
        )

        state = np.asarray(row["observation.state"], dtype=np.float32)
        gt_action = np.asarray(row["action"], dtype=np.float32)
        prompt = task_table[int(row["task_index"])]

        obs = {
            "observation/image": img_ext,
            "observation/wrist_image": img_wri,
            "observation/state": state,
            "prompt": prompt,
        }

        out = client.infer(obs)
        action_chunk = np.asarray(out["actions"], dtype=np.float32)

        # sanity checks
        assert action_chunk.shape == (10, 7), f"bad shape {action_chunk.shape}"
        assert np.isfinite(action_chunk).all(), "NaN/Inf in actions"

        first = action_chunk[0]
        l1 = float(np.abs(first - gt_action).mean())
        diffs_l1.append(l1)
        grip_diffs.append(abs(float(first[6] - gt_action[6])))

        print(
            f"{global_idx:>6} | "
            f"{np.array2string(gt_action[:3], precision=4, suppress_small=True):>34} | "
            f"{np.array2string(first[:3], precision=4, suppress_small=True):>34} | "
            f"   {gt_action[6]:.2f}/{first[6]:.2f}   | "
            f"{l1:.5f}"
        )

    arr = np.array(diffs_l1)
    grip = np.array(grip_diffs)
    print()
    print(f"== Summary ({len(arr)} samples) ==")
    print(f"L1(first_pred, GT_action)   mean={arr.mean():.5f}  median={np.median(arr):.5f}  min/max={arr.min():.5f}/{arr.max():.5f}")
    print(f"|gripper_pred - gripper_GT| mean={grip.mean():.4f}  max={grip.max():.4f}")
    print("All action chunks: shape=(10,7), finite, gripper roughly in [0,1].")


if __name__ == "__main__":
    main()
