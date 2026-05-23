# Robot fine-tuning bundle (LeRobot v3.0)

Three vendored projects with our local LeRobot v3.0 dataset patches applied.
History from each upstream is intentionally not preserved — this bundle is for
moving the local-modified codebases to other machines, not for upstream PRs.

| Subdir | Upstream | What we added |
|---|---|---|
| `openpi/` | https://github.com/Physical-Intelligence/openpi | `pi05_libero_local` TrainConfig, `local_data_root` support, `scripts/eval_offline.py`, `FINETUNING.md` |
| `openvla/` | https://github.com/openvla/openvla | self-contained LeRobot v3.0 reader (`prismatic/vla/datasets/lerobot_dataset.py`) + `vla-scripts/finetune_lerobot.py` |
| `Spatial-Forcing/` | https://github.com/OpenHelix-Team/Spatial-Forcing | `pi0_align_libero_local` TrainConfig under `openpi-SF/` |

See `openpi/FINETUNING.md` for end-to-end run instructions covering pi0.5,
OpenVLA, and Spatial-Forcing.

Heavy `third_party/` vendor directories were stripped from `openpi/` to keep
the bundle small; if you need them on another machine, re-clone the upstream
openpi repo separately.
