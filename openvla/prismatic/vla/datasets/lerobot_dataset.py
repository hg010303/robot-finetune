"""
lerobot_dataset.py

Self-contained reader + PyTorch Dataset for a local LeRobot v3.0 dataset
(https://github.com/huggingface/lerobot), producing the {pixel_values, input_ids, labels,
dataset_name} dict expected by OpenVLA's collator/models.

Why not depend on the `lerobot` package?  Its v0.4.x release line pulls torch>=2.5 and
numpy>=2, which conflicts with OpenVLA's torch 2.2 / tensorflow 2.15 environment. We only
need to (a) iterate frames, (b) decode one video frame at a time, and (c) read the action /
task fields; all of that is doable with pandas + PyAV directly.

Action normalization mirrors LeRobot/OpenVLA's BOUNDS_Q99 scheme:

    normalized = clip(2 * (x - q01) / (q99 - q01 + 1e-8) - 1, -1, 1)
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------- v3.0 index


@dataclass(frozen=True)
class _EpisodeLoc:
    """Per-episode lookup row distilled from `meta/episodes/.../*.parquet`."""

    dataset_from_index: int      # global frame index where this episode starts (inclusive)
    dataset_to_index: int        # global frame index where this episode ends (exclusive)
    data_chunk: int
    data_file: int
    video_locs: Dict[str, Tuple[int, int, float, float]]  # video_key -> (chunk, file, from_t, to_t)


class LeRobotV3Index:
    """Resolves a global frame index to (data parquet row, video file, video time)."""

    def __init__(self, root: Path, video_keys: Tuple[str, ...]):
        self.root = Path(root)
        self.video_keys = video_keys

        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        self.fps: float = float(self.info["fps"])
        self.total_episodes: int = int(self.info["total_episodes"])
        self.total_frames: int = int(self.info["total_frames"])
        self.data_path_tmpl: str = self.info["data_path"]
        self.video_path_tmpl: str = self.info["video_path"]

        self.tasks = self._load_tasks()
        self.episodes = self._load_episode_meta()

        # Sorted starts for fast bisect from global frame idx -> episode.
        self._episode_starts = np.fromiter(
            (ep.dataset_from_index for ep in self.episodes), dtype=np.int64, count=len(self.episodes)
        )

    # --------------------------------------------------------------------- helpers

    def _load_tasks(self) -> Dict[int, str]:
        tasks_df = pd.read_parquet(self.root / "meta" / "tasks.parquet")
        # The parquet uses task text as the index and `task_index` as the only column.
        return {int(row["task_index"]): str(text) for text, row in tasks_df.iterrows()}

    def _load_episode_meta(self) -> list[_EpisodeLoc]:
        meta_dir = self.root / "meta" / "episodes"
        files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata parquet files under {meta_dir}")
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        df = df.sort_values("episode_index").reset_index(drop=True)

        episodes: list[_EpisodeLoc] = []
        for _, r in df.iterrows():
            video_locs = {
                k: (
                    int(r[f"videos/{k}/chunk_index"]),
                    int(r[f"videos/{k}/file_index"]),
                    float(r[f"videos/{k}/from_timestamp"]),
                    float(r[f"videos/{k}/to_timestamp"]),
                )
                for k in self.video_keys
            }
            episodes.append(
                _EpisodeLoc(
                    dataset_from_index=int(r["dataset_from_index"]),
                    dataset_to_index=int(r["dataset_to_index"]),
                    data_chunk=int(r["data/chunk_index"]),
                    data_file=int(r["data/file_index"]),
                    video_locs=video_locs,
                )
            )
        return episodes

    # --------------------------------------------------------------------- public API

    def __len__(self) -> int:
        return self.total_frames

    def resolve(self, global_index: int):
        """Returns (episode_loc, frame_within_episode) for the given global frame index."""
        if not 0 <= global_index < self.total_frames:
            raise IndexError(f"global_index {global_index} out of range")
        epi_idx = int(bisect.bisect_right(self._episode_starts, global_index) - 1)
        ep = self.episodes[epi_idx]
        return ep, global_index - ep.dataset_from_index

    def data_parquet_path(self, chunk: int, file: int) -> Path:
        return self.root / self.data_path_tmpl.format(chunk_index=chunk, file_index=file)

    def video_path(self, video_key: str, chunk: int, file: int) -> Path:
        return self.root / self.video_path_tmpl.format(
            video_key=video_key, chunk_index=chunk, file_index=file
        )


# ---------------------------------------------------------------------------- decoders


class _DataParquetCache:
    """Tiny LRU cache over data parquet files; keeps the most-recently-used N in memory."""

    def __init__(self, index: LeRobotV3Index, max_size: int = 4) -> None:
        self._index = index
        self._max_size = max_size
        self._cache: Dict[Tuple[int, int], pd.DataFrame] = {}
        self._order: list[Tuple[int, int]] = []

    def get(self, chunk: int, file: int) -> pd.DataFrame:
        key = (chunk, file)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        df = pd.read_parquet(self._index.data_parquet_path(chunk, file))
        self._cache[key] = df
        self._order.append(key)
        while len(self._order) > self._max_size:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)
        return df


def _decode_video_frame(path: Path, target_time_s: float) -> Image.Image:
    """Seek to `target_time_s` in the mp4 at `path` and return the closest frame as a PIL RGB image."""
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        # PyAV seek is in stream.time_base ticks; convert seconds to ticks.
        seek_pts = int(max(target_time_s, 0.0) / float(stream.time_base))
        container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            if frame_time + 1e-6 >= target_time_s:
                return frame.to_image()
        # Fallback: return the last decoded frame.
        frame = next(container.decode(stream), None)
        if frame is None:
            raise RuntimeError(f"Could not decode any frame from {path} near t={target_time_s}")
        return frame.to_image()
    finally:
        container.close()


# ---------------------------------------------------------------------------- dataset


@dataclass
class LeRobotConfig:
    """Configuration knobs for `LeRobotDatasetForOpenVLA`."""

    data_root: Path
    dataset_name: str = "lerobot_local"
    image_video_key: str = "observation.images.image"           # primary camera video key
    state_column: str = "observation.state"                     # currently unused but kept for parity
    action_column: str = "action"
    task_column: str = "task_index"
    # If set (e.g. (256, 256)), the decoded video frame is resized to this (W, H) before
    # being passed to the model's `image_transform`. Useful when the underlying mp4 is
    # at a high resolution (e.g. 1280x720) and we want to skip the cost of decoding the
    # full-size frame all the way through the backbone preprocessor.
    image_resize_hw: Optional[Tuple[int, int]] = None
    # Optional fixed prompt to use when the dataset has only a single task. If None, the
    # prompt is taken from the dataset's `tasks` table indexed by `task_index`.
    fixed_prompt: Optional[str] = None
    predict_stop_token: bool = True
    normalize_actions: bool = True
    stats_max_frames: Optional[int] = 50_000
    stats_seed: int = 0
    data_parquet_cache_size: int = 4


class LeRobotDatasetForOpenVLA(Dataset):
    """Map-style PyTorch dataset over a local LeRobot v3.0 dataset for OpenVLA fine-tuning."""

    def __init__(
        self,
        config: LeRobotConfig,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.cfg = config
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.index = LeRobotV3Index(self.cfg.data_root, video_keys=(self.cfg.image_video_key,))
        self._data_cache = _DataParquetCache(self.index, max_size=self.cfg.data_parquet_cache_size)

        # Compute action quantiles on a random subset of frames.
        q01, q99 = self._compute_action_quantiles()
        self._action_q01 = q01.astype(np.float32)
        self._action_q99 = q99.astype(np.float32)
        self.dataset_statistics = {
            self.cfg.dataset_name: {
                "action": {
                    "q01": self._action_q01.tolist(),
                    "q99": self._action_q99.tolist(),
                    "mask": [True] * len(self._action_q01),
                }
            }
        }

    # ----------------------------------------------------------------- internals

    def _fetch_row(self, global_index: int) -> Tuple[_EpisodeLoc, int, pd.Series]:
        ep, within = self.index.resolve(global_index)
        df = self._data_cache.get(ep.data_chunk, ep.data_file)
        row = df.iloc[global_index - df.iloc[0]["index"]]
        return ep, within, row

    def _action_array(self, row: pd.Series) -> np.ndarray:
        a = row[self.cfg.action_column]
        return np.asarray(a, dtype=np.float32)

    def _compute_action_quantiles(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.index)
        n_sample = min(n, self.cfg.stats_max_frames or n)
        rng = np.random.default_rng(self.cfg.stats_seed)
        indices = rng.choice(n, size=n_sample, replace=False) if n_sample < n else np.arange(n)

        # Avoid per-row video reads by sampling actions only.
        actions: list[np.ndarray] = []
        for idx in tqdm(indices, desc=f"Computing action stats ({n_sample} frames)"):
            _, _, row = self._fetch_row(int(idx))
            actions.append(self._action_array(row))
        actions_arr = np.stack(actions, axis=0)
        return np.quantile(actions_arr, 0.01, axis=0), np.quantile(actions_arr, 0.99, axis=0)

    def _normalize(self, action: np.ndarray) -> np.ndarray:
        denom = (self._action_q99 - self._action_q01) + 1e-8
        return np.clip(2.0 * (action - self._action_q01) / denom - 1.0, -1.0, 1.0)

    def _decode_image(self, ep: _EpisodeLoc, frame_within: int) -> Image.Image:
        v_chunk, v_file, from_t, _ = ep.video_locs[self.cfg.image_video_key]
        target_t = from_t + frame_within / self.index.fps
        img = _decode_video_frame(self.index.video_path(self.cfg.image_video_key, v_chunk, v_file), target_t)
        if self.cfg.image_resize_hw is not None:
            h, w = self.cfg.image_resize_hw
            img = img.resize((w, h), Image.BILINEAR)
        return img

    # ----------------------------------------------------------------- Dataset

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep, within, row = self._fetch_row(idx)

        img = self._decode_image(ep, within)

        action = self._action_array(row)
        if self.cfg.normalize_actions:
            action = self._normalize(action)

        if self.cfg.fixed_prompt is not None:
            instruction = self.cfg.fixed_prompt.lower()
        else:
            task_index = int(row[self.cfg.task_column])
            instruction = self.index.tasks[task_index].lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.cfg.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=self.cfg.dataset_name,
        )
