#!/usr/bin/env python3
"""Convert a teleop rosbag (recorded by data_collection_node) to a LeRobot
dataset episode.

The script reads the rosbag, resamples observations to ``--fps`` using
nearest-neighbour in time, recovers the EE pose by looking it up from the
recorded ``/tf`` + ``/tf_static`` stream (default: ``panda_link0`` →
``panda_link8``), then computes per-step EE-frame action deltas using the
DROID-style convention shared with rollout
(:func:`pycontroller_template.controls.chunk_compose.decompose_to_eef_deltas`),
and appends one episode to the LeRobot dataset at ``--output``.

Usage
-----
    python scripts/bag_to_lerobot.py \\
        --bag /tmp/pycontroller_episodes/episode_<id>/rollout.bag \\
        --output /tmp/lerobot_datasets \\
        --repo_id local/pycontroller_pickplace \\
        --fps 15

If ``--task`` is omitted, the instruction is read from ``metadata.json``
next to the bag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Heavy ROS / LeRobot imports are inside main() so `--help` works in any env.


TOPIC_EXTERNAL_IMG = "/camera/color/image_raw"
TOPIC_WRIST_IMG = "/zedm/zed_node/rgb/image_rect_color"
TOPIC_JOINTS = "/joint_states"
TOPIC_GRIPPER = "/gripper/joint_commands"
TOPIC_TF = "/tf"
TOPIC_TF_STATIC = "/tf_static"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bag", required=True, type=Path)
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Root directory containing one or more LeRobot datasets.",
    )
    p.add_argument(
        "--repo_id",
        required=True,
        help="LeRobot dataset identifier, e.g. 'local/pycontroller_pickplace'.",
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument(
        "--task",
        help="Override language instruction. Default: read metadata.json next to the bag.",
    )
    p.add_argument(
        "--base_frame",
        default="panda_link0",
        help="TF parent frame for the EE pose lookup.",
    )
    p.add_argument(
        "--ee_frame",
        default="panda_link8",
        help="TF child frame whose pose is recorded as the EE proprio + action source.",
    )
    p.add_argument(
        "--invert-gripper",
        action="store_true",
        default=True,
        help="Invert the recorded gripper command (1 - x) back to policy "
        "convention (0=open, 1=close). The trajectory tracker publishes "
        "the inverted value, so this defaults to True.",
    )
    p.add_argument("--no-invert-gripper", action="store_false", dest="invert_gripper")
    return p.parse_args()


def resolve_task(args: argparse.Namespace) -> str:
    if args.task:
        return args.task
    metadata_path = args.bag.parent / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(
            f"No --task provided and no metadata.json at {metadata_path}."
        )
    return json.loads(metadata_path.read_text())["task"]


def load_messages(bag_path: Path, topics: list[str]) -> dict[str, list]:
    import rosbag

    out: dict[str, list] = {t: [] for t in topics}
    with rosbag.Bag(str(bag_path)) as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            if hasattr(msg, "header") and msg.header.stamp.to_sec() > 0:
                ts = msg.header.stamp.to_sec()
            else:
                ts = t.to_sec()
            out[topic].append((ts, msg))
    for t in topics:
        out[t].sort(key=lambda x: x[0])
    return out


def resample_nearest(msgs: list, ts_grid: np.ndarray) -> list:
    if not msgs:
        raise ValueError("Cannot resample an empty topic.")
    times = np.array([m[0] for m in msgs])
    idx = np.searchsorted(times, ts_grid)
    idx = np.clip(idx, 1, len(times) - 1)
    left_dist = np.abs(times[idx - 1] - ts_grid)
    right_dist = np.abs(times[idx] - ts_grid)
    picks = np.where(left_dist <= right_dist, idx - 1, idx)
    return [msgs[i][1] for i in picks]


def build_tf_buffer(bag_path: Path, cache_seconds: float):
    """Replay /tf + /tf_static from the bag into a ``tf2_ros.BufferCore``."""
    import rosbag
    import rospy
    import tf2_ros

    buffer = tf2_ros.BufferCore(rospy.Duration(cache_seconds))
    n_dyn = 0
    n_static = 0
    with rosbag.Bag(str(bag_path)) as bag:
        for topic, msg, _t in bag.read_messages(topics=[TOPIC_TF, TOPIC_TF_STATIC]):
            for transform in msg.transforms:
                if topic == TOPIC_TF_STATIC:
                    buffer.set_transform_static(transform, "bag")
                    n_static += 1
                else:
                    buffer.set_transform(transform, "bag")
                    n_dyn += 1
    print(f"  /tf: {n_dyn} transforms, /tf_static: {n_static} transforms")
    if n_dyn == 0 and n_static == 0:
        raise SystemExit(
            "No transforms found in the bag — was /tf recorded? "
            "Check `record_topics` in data_collection.yaml."
        )
    return buffer


def lookup_pose(buffer, base_frame: str, ee_frame: str, time_sec: float):
    """Look up ``ee_frame`` pose expressed in ``base_frame`` at ``time_sec``.

    Returns ``(pos[3], quat_xyzw[4])`` or ``None`` if the lookup fails.
    """
    import rospy
    import tf2_ros

    try:
        tr = buffer.lookup_transform_core(
            base_frame, ee_frame, rospy.Time.from_sec(time_sec)
        ).transform
    except (
        tf2_ros.LookupException,
        tf2_ros.ExtrapolationException,
        tf2_ros.ConnectivityException,
        tf2_ros.InvalidArgumentException,
    ):
        return None
    pos = np.array([tr.translation.x, tr.translation.y, tr.translation.z])
    quat = np.array(
        [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]
    )
    return pos, quat


def trim_to_valid_range(
    ts_grid: np.ndarray, lookups: list
) -> tuple[np.ndarray, list[int]]:
    """Restrict to the longest contiguous range over which TF lookups succeed.

    Returns the trimmed grid and the indices into the original grid that survived.
    """
    valid = np.array([x is not None for x in lookups])
    if valid.all():
        return ts_grid, list(range(len(ts_grid)))
    if not valid.any():
        raise SystemExit(
            "TF lookup failed at every timestamp; check --base_frame/--ee_frame."
        )

    # Longest contiguous run of valid.
    best_start = best_end = -1
    best_len = 0
    i = 0
    while i < len(valid):
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < len(valid) and valid[j]:
            j += 1
        if (j - i) > best_len:
            best_len = j - i
            best_start, best_end = i, j
        i = j

    dropped = len(valid) - best_len
    print(
        f"Warning: dropped {dropped} of {len(valid)} frames due to TF lookup "
        f"failures (kept range {best_start}:{best_end})."
    )
    keep = list(range(best_start, best_end))
    return ts_grid[keep], keep


def main() -> None:
    args = parse_args()
    task = resolve_task(args)
    print(f"Task: {task}")
    print(f"EE frame: {args.base_frame} → {args.ee_frame}")

    import cv_bridge

    from pycontroller_template.controls.chunk_compose import decompose_to_eef_deltas

    topics = [TOPIC_EXTERNAL_IMG, TOPIC_WRIST_IMG, TOPIC_JOINTS, TOPIC_GRIPPER]
    print(f"Loading {args.bag}...")
    msgs = load_messages(args.bag, topics)
    for topic in topics:
        n = len(msgs[topic])
        print(f"  {topic}: {n} messages")
        if n == 0:
            raise SystemExit(f"Topic {topic} has no messages.")

    # /joint_states is published continuously during teleop — anchor on it.
    joint_times = np.array([m[0] for m in msgs[TOPIC_JOINTS]])
    t_start, t_end = joint_times[0], joint_times[-1]
    duration = t_end - t_start
    n_frames = int(duration * args.fps)
    if n_frames < 2:
        raise SystemExit(
            f"Bag too short for fps={args.fps}: only {n_frames} frames."
        )
    ts_grid = t_start + np.arange(n_frames) / args.fps

    print("Building TF buffer...")
    buffer = build_tf_buffer(args.bag, cache_seconds=max(60.0, duration + 10.0))

    print(f"Looking up {args.ee_frame} pose at {n_frames} timestamps...")
    lookups = [
        lookup_pose(buffer, args.base_frame, args.ee_frame, ts) for ts in ts_grid
    ]
    ts_grid, keep_idx = trim_to_valid_range(ts_grid, lookups)
    n_frames = len(ts_grid)
    positions = np.array([lookups[i][0] for i in keep_idx])
    quats = np.array([lookups[i][1] for i in keep_idx])
    print(f"Resampling {n_frames} frames at {args.fps} Hz " 
          f"({ts_grid[-1] - ts_grid[0]:.2f} s of trajectory).")

    bridge = cv_bridge.CvBridge()
    external_imgs = np.stack(
        [
            bridge.imgmsg_to_cv2(m, desired_encoding="rgb8")
            for m in resample_nearest(msgs[TOPIC_EXTERNAL_IMG], ts_grid)
        ]
    )
    wrist_imgs = np.stack(
        [
            bridge.imgmsg_to_cv2(m, desired_encoding="rgb8")
            for m in resample_nearest(msgs[TOPIC_WRIST_IMG], ts_grid)
        ]
    )
    joints = np.array(
        [
            list(m.position)
            for m in resample_nearest(msgs[TOPIC_JOINTS], ts_grid)
        ],
        dtype=np.float32,
    )

    gripper_msgs = resample_nearest(msgs[TOPIC_GRIPPER], ts_grid)
    grippers = np.array(
        [float(m.position[0]) if m.position else 0.0 for m in gripper_msgs],
        dtype=np.float32,
    )
    if args.invert_gripper:
        grippers = np.clip(1.0 - grippers, 0.0, 1.0)

    # Action: EE-frame delta to the *next* frame, plus current commanded gripper.
    deltas = decompose_to_eef_deltas(positions, quats)  # (n_frames - 1, 6)
    actions = np.zeros((n_frames, 7), dtype=np.float32)
    actions[:-1, :6] = deltas
    actions[:, 6] = grippers

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.external_image": {
            "dtype": "video",
            "shape": (external_imgs.shape[1], external_imgs.shape[2], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": (wrist_imgs.shape[1], wrist_imgs.shape[2], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (joints.shape[1],),
            "names": [f"joint_{i + 1}" for i in range(joints.shape[1])],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "dx_eef", "dy_eef", "dz_eef",
                "drx_eef", "dry_eef", "drz_eef",
                "gripper",
            ],
        },
    }

    dataset_root = args.output / args.repo_id.replace("/", "__")
    if dataset_root.exists():
        print(f"Appending to existing dataset at {dataset_root}")
        dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_root)
    else:
        print(f"Creating new dataset at {dataset_root}")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=dataset_root,
            features=features,
            use_videos=True,
        )

    for i in range(n_frames):
        dataset.add_frame(
            {
                "observation.images.external_image": external_imgs[i],
                "observation.images.wrist_image": wrist_imgs[i],
                "observation.state": joints[i],
                "action": actions[i],
                "task": task,
            }
        )
    dataset.save_episode()
    print(f"Saved episode with {n_frames} frames to {dataset_root}")


if __name__ == "__main__":
    main()