"""Offline action prediction eval for a trained openpi pi0/pi0.5 checkpoint.

For each sampled frame in the configured LeRobot dataset, this script:
  1. Loads the raw frame (image, state, task) via openpi's training data loader.
  2. Runs `policy.infer()` to get a predicted action chunk of shape (H, action_dim).
  3. Compares against the GT action chunk and accumulates MAE/RMSE per dim.

Run from the openpi repo root:

    uv run scripts/eval_offline.py \\
        --config_name=pi05_libero_local \\
        --checkpoint_dir=checkpoints/pi05_libero_local/panda_libero_run0/4999 \\
        --num_samples=200

Optional flags:
    --plot_dir=eval_plots  Save GT vs. predicted action curves for the first 8 samples.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import tqdm
import tyro

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass
class Args:
    # Name of a TrainConfig registered in `openpi.training.config`.
    config_name: str
    # Path to the trained checkpoint step directory (e.g. `.../4999`).
    checkpoint_dir: Path
    # Number of frames to evaluate (randomly sampled, no replacement).
    num_samples: int = 200
    # RNG seed for sampling and for the policy's diffusion noise (when applicable).
    seed: int = 0
    # If set, save GT-vs-pred action curve PNGs for the first 8 samples to this dir.
    plot_dir: Path | None = None
    # If set, write the per-sample errors + summary metrics as JSON to this path.
    output_json: Path | None = None


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _save_plot(plot_dir: Path, sample_idx: int, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    horizon, dim = gt.shape
    fig, axes = plt.subplots(dim, 1, figsize=(8, 1.4 * dim), sharex=True)
    if dim == 1:
        axes = [axes]
    t = np.arange(horizon)
    for d in range(dim):
        axes[d].plot(t, gt[:, d], "o-", label="GT", color="tab:blue")
        axes[d].plot(t, pred[:, d], "x--", label="Pred", color="tab:orange")
        axes[d].fill_between(t, gt[:, d], pred[:, d], color="gray", alpha=0.15)
        axes[d].set_ylabel(f"a[{d}]")
        if (~mask).any():
            for ti, m in enumerate(mask):
                if not m:
                    axes[d].axvspan(ti - 0.5, ti + 0.5, color="red", alpha=0.05)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("action step")
    fig.suptitle(f"sample idx={sample_idx}")
    fig.tight_layout()
    out = plot_dir / f"sample_{sample_idx:06d}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(args: Args) -> None:
    cfg = _config.get_config(args.config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

    print(f"Config: {cfg.name}  model_type={cfg.model.model_type}  "
          f"action_horizon={cfg.model.action_horizon}  action_dim={cfg.model.action_dim}")
    print(f"Checkpoint: {args.checkpoint_dir}")

    raw_ds = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    print(f"Dataset size: {len(raw_ds)} frames")

    # The repack transforms map raw LeRobot keys (e.g. `observation.images.image`) to
    # the keys expected by the libero policy inputs (e.g. `observation/image`).
    policy = _policy_config.create_trained_policy(
        cfg,
        str(args.checkpoint_dir),
        repack_transforms=data_config.repack_transforms,
    )
    print("Policy loaded.\n")

    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, len(raw_ds))
    indices = rng.choice(len(raw_ds), size=n, replace=False)

    mae_rows: list[np.ndarray] = []
    rmse_rows: list[np.ndarray] = []
    timings: list[float] = []
    per_sample = []

    for plot_count, idx in enumerate(tqdm.tqdm(indices.tolist(), desc="eval")):
        sample = raw_ds[idx]
        obs = {k: _to_numpy(v) for k, v in sample.items()}

        # Keep `action` in obs — the policy's RepackTransform was registered with
        # `actions <- action` at training time and will key-error if it's missing.
        gt = obs["action"].astype(np.float32)  # (H, action_dim)
        pad_mask_full = obs.get("action_is_pad", np.zeros(gt.shape[0], dtype=bool))
        keep = ~np.asarray(pad_mask_full).astype(bool)
        if not keep.any():
            continue

        result = policy.infer(obs)
        pred = np.asarray(result["actions"], dtype=np.float32)  # (H, action_dim_model)
        pred = pred[:, : gt.shape[-1]]  # drop padded dims (action_dim is padded to 32 in pi05)

        gt_v = gt[keep]
        pred_v = pred[keep]

        mae_rows.append(np.mean(np.abs(pred_v - gt_v), axis=0))
        rmse_rows.append(np.sqrt(np.mean((pred_v - gt_v) ** 2, axis=0)))
        timings.append(result["policy_timing"]["infer_ms"])

        per_sample.append({
            "idx": int(idx),
            "mae": float(np.mean(np.abs(pred_v - gt_v))),
            "rmse": float(np.sqrt(np.mean((pred_v - gt_v) ** 2))),
            "horizon_kept": int(keep.sum()),
        })

        if args.plot_dir is not None and plot_count < 8:
            _save_plot(Path(args.plot_dir), int(idx), gt, pred, keep)

    if not mae_rows:
        print("No valid samples (every chunk was fully padded?). Aborting.")
        return

    mae = np.stack(mae_rows).mean(axis=0)
    rmse = np.stack(rmse_rows).mean(axis=0)

    print(f"\n=== Evaluation over {len(mae_rows)} frames ===")
    print(f"Per-dim MAE  : {np.array2string(mae, precision=4)}")
    print(f"Per-dim RMSE : {np.array2string(rmse, precision=4)}")
    print(f"Overall MAE  : {mae.mean():.4f}")
    print(f"Overall RMSE : {rmse.mean():.4f}")
    print(f"Median infer time: {np.median(timings):.1f} ms  (mean {np.mean(timings):.1f} ms)")

    if args.output_json is not None:
        out = {
            "config_name": cfg.name,
            "checkpoint_dir": str(args.checkpoint_dir),
            "num_samples": len(mae_rows),
            "per_dim_mae": mae.tolist(),
            "per_dim_rmse": rmse.tolist(),
            "overall_mae": float(mae.mean()),
            "overall_rmse": float(rmse.mean()),
            "median_infer_ms": float(np.median(timings)),
            "per_sample": per_sample,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
