# jisang_robot — Project Handoff

이 문서는 **컨텍스트가 전혀 없는 사람 (또는 새 Claude session) 이 이 프로젝트를
이어받아 fine-tuning / inference / 추가 데이터 수집을 진행할 수 있도록** 작성됐습니다.
세 가지 큰 축 — (1) 데이터 포맷, (2) 학습 컨벤션, (3) 배포 — 을 모두 이 문서에서
다룹니다. 더 자세한 내용은 같은 디렉토리의 `FINETUNING_JISANG.md`,
`DATASET_COMPARISON.md`, `DEPLOY.md` 를 참조.

---

## 0. 한눈에 보기

| 항목 | 값 |
|---|---|
| 로봇 | Franka Panda (7-DOF) |
| 카메라 | external (3rd-person) + wrist, 둘 다 1280×720 RGB |
| 학습 fps | **15 Hz** |
| State | **7D joint angles** (rad) — Panda 관절 7개 |
| Action | **7D EE-frame delta + gripper** = `[dx, dy, dz, drx, dry, drz, gripper]` |
| Action chunk (학습) | 10 step (pi0.5 / SF), 1 step (OpenVLA) |
| Task 수 | 4 (`pick_and_place`, `chocomilk`, `kitchen`, `pot`) |
| 정규화 | quantile-based (q01/q99 → `[-1, 1]`) |
| 데이터셋 포맷 | **LeRobot v3.0** |

---

## 1. 디렉토리 구조

이 monorepo 가 (거의) 전부입니다:

```
robot-monorepo/
├── openpi/                  # pi0.5 학습 + websocket inference server
├── Spatial-Forcing/openpi-SF/   # SF (pi0 + VGGT alignment) 학습
├── openvla/                 # OpenVLA fine-tune
└── jisang_robot/            # ← 이 프로젝트 고유 코드/문서
    ├── HANDOFF.md           # 이 문서
    ├── FINETUNING_JISANG.md # 3 모델 학습 절차
    ├── DATASET_COMPARISON.md  # reference 데이터셋과의 차이
    ├── DEPLOY.md            # 배포 가이드
    ├── convert.py           # rosbag → LeRobot v3.0 변환 스크립트
    ├── convert_*.sh         # 데이터셋별 변환 셸 스크립트
    ├── validate_policy.py   # 학습 데이터로 정책 dry-run
    └── deploy/
        ├── deploy_pi05.py
        ├── deploy_sf.py
        └── deploy_openvla.py
```

`jisang_robot/lerobot_datasets/` (변환된 LeRobot 데이터셋들) 과 `jisang_robot/robot/`
(원본 rosbag) 은 **이 monorepo 에는 포함되지 않습니다** — 크기가 커서. 실제 디스크
위치는 `/home/cvlab/project/realsangbeom/robot/jisang_robot/{lerobot_datasets,robot}/`.

---

## 2. 데이터 포맷 — LeRobot v3.0

### 2-1. 원본 (rosbag)
- 위치: `jisang_robot/robot/<task_name>/recording_*/recording.bag`
- 각 bag 한 episode 분량 (teleop 으로 수집)
- ROS topics:
  - `/camera/color/image_raw` → external image
  - `/zedm/zed_node/rgb/image_rect_color` → wrist image
  - `/joint_states` → Panda 7-DOF joint
  - `/gripper/joint_commands` → gripper
  - `/tf` + `/tf_static` → EE pose (panda_link0 → panda_link8)

### 2-2. 변환 — `convert.py`
- bag 을 읽어 15 Hz 로 resample
- EE pose 를 TF lookup 으로 복구
- **DROID-style EE-frame action delta** 를 `decompose_to_eef_deltas` 로 계산
  (`pycontroller_template-dev-corl26/src/pycontroller/controls/chunk_compose.py`)
- gripper: `--invert-gripper` 기본 True (trajectory tracker 가 1-x 형태로 publish 하므로 되돌림)
- 결과 episode 를 `--output` 디렉토리의 LeRobot dataset 에 append

### 2-3. 변환된 데이터셋 구조 (v3.0)
```
<dataset_root>/
├── meta/
│   ├── info.json                          # total_episodes, frames, fps, schema
│   ├── tasks.parquet                      # index=task_text, col=task_index
│   ├── stats.json                         # 학습 시 별도 norm_stats 가 우선 사용됨
│   └── episodes/chunk-XXX/file-YYY.parquet # 각 file = 1 episode 의 메타
├── data/chunk-XXX/file-YYY.parquet        # 각 file = 1 episode 의 모든 frame
└── videos/<video_key>/chunk-XXX/file-YYY.mp4   # 각 file = 1 episode 의 비디오
```

`data/.../file-YYY.parquet` 의 컬럼:
- `observation.state` : `list<float>[7]` (joint angles rad)
- `action` : `list<float>[7]` (EE delta + gripper)
- `timestamp`, `frame_index`, `episode_index`, `index` (global frame idx), `task_index`

`videos/` 의 키:
- `observation.images.external_image` (720×1280 AV1/yuv420p, 15Hz)
- `observation.images.wrist_image` (720×1280 AV1/yuv420p, 15Hz)

### 2-4. 4 개 단일-task 데이터셋
| 디렉토리 | task text |
|---|---|
| `local__pick_and_place/` (284 ep / 43328 frame) | `"Pick up the cube and place it on the plate."` |
| `local__chocomilk/` (202 ep / 84435 frame)      | `"Pick up the chocolate milk and place it on top of the white milk, then pick up the cube and stack it on top of the chocolate milk."` |
| `local__pot/` (169 ep / 63002 frame)            | `"Open the lid of the pot, put the yellow cube inside the pot, and close the lid again."` |
| `local__kitchen/` (183 ep / 64182 frame)        | `"Pick up the pot and place it on the bottom section of the induction cooktop, then pick up the pan and place it on the top section of the induction cooktop."` |

### 2-5. 통합 multi-task 데이터셋 — `local__jisang_combined/`
- 위 4개를 하나로 merge (838 episode / 254947 frame / **4 task**)
- merge 방식: source 별 `chunk-{src_id:03d}` (0,1,2,3) 로 분리해서 data/video/episode meta 옮기고
  - `task_index` 를 0,1,2,3 으로 재할당
  - 모든 global 인덱스 (`index`, `episode_index`, `dataset_from_index/to_index`) 를 누적 offset 으로 재계산
  - video 파일은 **hardlink** (디스크 절약)
- 멀티 태스크 학습은 이 통합 데이터셋을 그대로 single dataset 으로 학습 — 데이터로더가 episode_index → task_index → task text 를 자동 변환.

---

## 3. Action / State 컨벤션 (반드시 일치시켜야 함)

### 3-1. State (7D)
- **Panda joint angles in radians**, shape `(7,)`, float32.
- 학습 시 model 내부에서 max_state_dim (32) 까지 zero-pad 됨 (LiberoInputs 가 통과시키고 PadStatesAndActions transform 이 수행).
- inference 시에도 7D 그대로 obs 에 넣음. 서버가 알아서 pad.

### 3-2. Action (7D)
- `[dx_eef, dy_eef, dz_eef, drx_eef, dry_eef, drz_eef, gripper]`
- 처음 6: **현재 EE frame 기준 변위** (xyz 미터, 회전 axis-angle 라디안)
- 마지막 1: gripper, `[0, 1]` 범위 (training convention: **0 = open, 1 = close**).
- 한 step = **1/15 초** 분량의 EE-delta.

### 3-3. EE-delta 계산 (학습 데이터 생성 시)
```python
# pycontroller_template-dev-corl26/src/pycontroller/controls/chunk_compose.py
delta = decompose_to_eef_deltas(prev_ee_pose, curr_ee_pose)
# delta[:3]  = T_prev_ee^-1 · (curr_pos - prev_pos), 즉 prev EE frame 기준 xyz delta
# delta[3:6] = log_map(T_prev_ee^-1 · R_curr), axis-angle in prev EE frame
```

### 3-4. EE-delta 적용 (inference / deploy 시)
`decompose_to_eef_deltas` 의 **역연산** 이 필요합니다 — 같은 파일에 `compose_ee_delta`
(또는 호출 패턴은 자유롭게 짜도 됨, 핵심은 SE(3) 연산):
```python
current_pose = robot.read_ee_pose()                 # SE(3)
target_pose  = compose_ee_delta(current_pose, delta)
joint_cmd    = ik_solver(target_pose)
robot.servo(joint_cmd, dt=1/15)
robot.set_gripper(gripper)
```

**중요**: control loop 는 정확히 15 Hz 로 돌려야 학습 EE-delta 의 의미가 맞습니다.
30 Hz 면 한 delta 가 두 번 누적되어 두 배 빠르게 움직임.

---

## 4. 데이터로더 흐름 — 모델별

### 4-1. pi0.5 / SF (openpi 계열)
1. **`lerobot.LeRobotDataset`** 이 episode_index → frame 단위로 sample. video 디코딩 (PyAV), state/action/task_index/task_text 반환.
2. **`RepackTransform`** (`LeRobotJisangDataConfig` 안):
   - `observation.images.external_image` → `observation/image`
   - `observation.images.wrist_image`    → `observation/wrist_image`
   - `observation.state`                  → `observation/state`
   - `action`                             → `actions`
   - `task`                                → `prompt`
3. **`LiberoInputs`** (`libero_policy.LiberoInputs`):
   - state / action 7D 를 그대로 통과 (모델 내부에서 pad)
   - image dict 구조 정리 (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`)
4. **`ModelTransformFactory`**:
   - `ResizeImages(224, 224)` — 모델 입력 크기로 letterbox-resize
   - `TokenizePrompt` — 텍스트 prompt → 200 token
   - `PadStatesAndActions(model_action_dim=32)` — 7D → 32D zero-pad
5. **Normalize** (auto): `norm_stats.json` 의 q01/q99 로 action 을 `[-1, 1]` 로 변환.
6. 모델 forward, action loss 계산.

### 4-2. OpenVLA (self-contained reader)
- `openvla/prismatic/vla/datasets/lerobot_dataset.py` 에 별도 reader (`LeRobotV3Index`, `LeRobotDatasetForOpenVLA`) 가 있음 — lerobot 패키지 의존성 회피용.
- task text: `tasks.parquet` 을 시작 시 dict 로 로드, `__getitem__` 에서 `task_index` → text 직접 lookup. `--fixed_prompt` 인자로 강제 override 가능.
- image resize: `--image_resize_h/--image_resize_w` 옵션 → 디코딩 직후 PIL.Image.resize 적용 (서버측 부담 줄임).
- action normalization: **BOUNDS_Q99** 동일 공식 (`clip(2*(x-q01)/(q99-q01) - 1, -1, 1)`). 학습 시 quantile 직접 계산 후 `dataset_statistics.json` 에 저장.

---

## 5. Action Normalization (반드시 알아야 함)

### 5-1. 정규화 공식 (모든 모델 공통)
```
정규화:   x_norm = clip(2 * (x - q01) / (q99 - q01) - 1, -1, 1)
역정규화: x      = (x_norm + 1) / 2 * (q99 - q01) + q01
```
- `q01`, `q99` 는 dataset action 분포의 1% / 99% quantile.
- 학습 시작 시 무작위 N frame sample 해서 계산.
- 학습 / inference 시 **반드시 같은 q01/q99 사용**. 안 맞으면 action 이 무의미.

### 5-2. norm_stats 저장 위치
| 모델 | 경로 |
|---|---|
| pi0.5 / SF | `<openpi-root>/assets/<config_name>/<asset_id>/norm_stats.json`, 학습 후엔 `checkpoints/<config>/<exp>/<step>/assets/...` 안에도 복사됨 |
| OpenVLA | `runs/<exp_id>--lerobot/dataset_statistics.json` (key: dataset_name) |

### 5-3. 자동 사용
- pi0.5 / SF: `serve_policy.py` 가 체크포인트의 `assets/` 에서 자동 로드.
- OpenVLA: `vla.predict_action(..., unnorm_key=DATASET_NAME)` 가 모델 dir 의 `dataset_statistics.json` 을 자동 lookup.

→ **사용자가 따로 정규화/역정규화 코드를 짜지 않아도 됨**. obs/action 은 raw scale 로만 주고받음.

---

## 6. 학습된 모델

### 6-1. pi0.5 multi-task — ✅ 학습 완료, HF 업로드 완료
- **Config**: `pi05_jisang_combined` (`openpi/src/openpi/training/config.py`)
- **체크포인트 (local)**: `/home/cvlab/project/realsangbeom/robot/openpi/checkpoints/pi05_jisang_combined/combined_run0/{5000, 10000, …, 95000, 99999}`
- **HF**: [`honggyuAn/jisang_robot`](https://huggingface.co/honggyuAn/jisang_robot) 의 `pi05_combined/99999/`
- **세팅**: GPU 4 FSDP, batch_size=16, num_train_steps=100K, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`
- **Final**: `loss=0.0305`, `grad_norm=0.13`, `param_norm=1892`
- **wandb**: [bags/openpi/runs/5xp5wsng](https://wandb.ai/bags/openpi/runs/5xp5wsng)

### 6-2. SF (Spatial-Forcing) multi-task — 🟡 학습 중
- **Config**: `pi0_align_jisang_combined` (`openpi-SF/.../config.py`)
- 체크포인트 위치: `openpi-SF/checkpoints/pi0_align_jisang_combined/sf_combined_run0/<STEP>`
- **세팅**: GPU 4 DDP (PyTorch), batch=16, num_train_steps=100K
- alignment loss: VGGT-1B 의 feature 와 paligemma 12번째 layer feature 사이 alignment (coeff 0.5)
- **wandb**: [bags/openpi/runs/gf4hyvss](https://wandb.ai/bags/openpi/runs/gf4hyvss)

### 6-3. OpenVLA multi-task — 🟡 학습 중
- 학습 스크립트: `openvla/vla-scripts/finetune_lerobot.py`
- LoRA r=32, batch=16, num_train_steps=100K
- 결과 (머지된 풀 모델): `openvla/runs/openvla-7b+local__jisang_combined+...--lerobot/`

### 6-4. 이전 (단일 task) 모델들
- `pi05_jisang_pick_and_place` (`jisang_run0/100000`) — 단일 task pick_and_place 만 학습. 비교용으로 남겨둠.

---

## 7. Inference / Deploy

### 7-1. 입력 obs 형식 (pi0.5 / SF — websocket 클라이언트 측)
```python
{
    "observation/image":       np.ndarray (H, W, 3) uint8 RGB,   # 임의 해상도
    "observation/wrist_image": np.ndarray (H, W, 3) uint8 RGB,
    "observation/state":       np.ndarray (7,) float32,           # Panda joint rad
    "prompt":                  str,                                # 4 task 중 하나의 exact text
}
```

### 7-2. 출력 action 형식
- pi0.5 / SF: 응답 dict `{"actions": np.ndarray (10, 7) float32}` — 각 행 = 1 step EE-delta+gripper, **이미 denormalize 된 raw scale**.
- OpenVLA: `predict_action()` 반환 `np.ndarray (7,)` — 매 step 단발 inference, **이미 denormalize**.

### 7-3. 서버 시작 (pi0.5 예시)
```bash
cd openpi
conda activate openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_jisang_combined \
    --policy.dir=checkpoints/pi05_jisang_combined/combined_run0/99999
# 또는 HF 에서 받은 디렉토리를 --policy.dir 로
```

### 7-4. 클라이언트 (3 모델 deploy 스크립트)
- `jisang_robot/deploy/deploy_pi05.py`
- `jisang_robot/deploy/deploy_sf.py`
- `jisang_robot/deploy/deploy_openvla.py`

공통 4 hook 만 채우면 됨:
- `get_external_image()` → uint8 RGB ndarray (OpenVLA 는 PIL.Image)
- `get_wrist_image()` → uint8 RGB ndarray (OpenVLA 미사용)
- `get_joint_angles()` → (7,) float32 rad
- `apply_ee_delta(delta_xyz_rpy, gripper)` → 로봇에 적용

자세한 deploy 절차는 **`DEPLOY.md`** 참조.

### 7-5. 로봇 없이 검증 — `validate_policy.py`
```bash
# 1) 서버 띄우고
# 2) 학습 데이터에서 10 frame 샘플해서 client.infer 호출, action shape/range/finite 검사
python jisang_robot/validate_policy.py --num_samples 10
```
이게 통과해야 로봇 연결 의미 있음.

---

## 8. 학습 인프라 / 환경

### 8-1. conda envs
- `openpi` (Python 3.11) — pi0.5 학습/inference, uv-managed `.venv`
- `openpi-SF` (Python 3.11) — Spatial-Forcing 학습/inference
- `openvla` (Python 3.10) — OpenVLA 학습
- `bag_convert` — rosbag → LeRobot 변환

### 8-2. 주요 코드 수정 위치 (vs reference)
| 파일 | 변경 |
|---|---|
| `openpi/src/openpi/training/config.py` | `LeRobotJisangDataConfig` + 4 TrainConfig 추가 |
| `openpi/scripts/train.py:60` | `wandb.init(... resume="must" → "allow")` 패치 |
| `openpi-SF/src/openpi/training/config.py` | 동일하게 jisang DataConfig + TrainConfig |
| `openpi-SF/.venv/.../orbax/checkpoint/_src/path/step.py` | GCS finalize check 무력화 (`commit_success.txt` 누락 우회) |
| `openvla/prismatic/vla/datasets/lerobot_dataset.py` | `image_resize_hw`, `fixed_prompt` 옵션 |
| `openvla/vla-scripts/finetune_lerobot.py` | CLI 인자 노출 (`--image_resize_h/w`, `--fixed_prompt`) |

### 8-3. 학습 명령 (실제로 돌렸던 것)
```bash
# pi0.5 multi-task (완료된 학습)
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
    uv run scripts/train.py pi05_jisang_combined \
        --exp_name=combined_run0 --fsdp_devices=4 \
        --batch_size=16 --num_train_steps=100000

# SF multi-task
CUDA_VISIBLE_DEVICES=4,5,6,7 \
    uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_align_pytorch.py pi0_align_jisang_combined \
        --exp_name=sf_combined_run0 \
        --batch_size=16 --num_train_steps=100000 \
        --wandb-enabled

# OpenVLA multi-task
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nnodes 1 --nproc-per-node 4 \
        vla-scripts/finetune_lerobot.py \
        --vla_path openvla/openvla-7b \
        --lerobot_root .../local__jisang_combined \
        --dataset_name local__jisang_combined \
        --image_video_key observation.images.external_image \
        --image_resize_h 256 --image_resize_w 256 \
        --batch_size 16 --learning_rate 5e-4 --max_steps 100000 \
        --wandb_project openvla --wandb_entity bags
```

---

## 9. 알려진 함정 (반드시 읽어볼 것)

### 9-1. 표현 호환성
- inference 시 obs 의 state/action **차원과 의미** 가 학습과 완전히 동일해야 합니다.
- 한 차원이라도 의미가 다르면 (예: rad → deg, EE-delta → 절대 좌표) 모델 출력 무의미.

### 9-2. fps
- 학습 = 15 Hz. control loop 도 정확히 15 Hz. EE-delta 의 누적 의미가 시간 의존.

### 9-3. norm_stats 일관성
- 체크포인트 디렉토리에 함께 저장된 norm_stats 그대로 사용. **본인이 다시 계산하면
  안 됨** (sample 이 달라서 q01/q99 가 약간 어긋나면 inference 결과 망가짐).

### 9-4. multi-task prompt
- 학습 시 사용한 4 task text 와 **정확히** 같은 문자열을 prompt 로 보내야 합니다.
  (대소문자, 마침표, 공백까지) — 다른 텍스트면 OOD 가 되어 동작 보장 안 됨.

### 9-5. GCS 의 pi0_base 체크포인트
- `gs://openpi-assets/checkpoints/pi0_base/params` 에는 `commit_success.txt` 마커가
  누락되어 있어서 orbax 가 `Found incomplete checkpoint` 던집니다. 이 monorepo 의
  openpi-SF venv 에는 그 검사를 무력화한 패치가 포함됐습니다 (코드 위치는 §8-2 표
  참조). 만약 새 env 를 처음부터 만들면 같은 패치가 다시 필요합니다.

### 9-6. wandb resume
- `train.py` 의 init_wandb 는 원래 `resume="must"` 였는데, wandb 의 일시적 network
  timeout 이나 run 삭제 시 학습이 죽었습니다. `"allow"` 로 패치되어 있어서 이제 새 run
  으로 자연 생성됩니다.

### 9-7. 디스크 압박
- 체크포인트가 매 5000 step 마다 ~31 GB. 100K step 학습이면 약 ~620 GB. 학습 도중
  디스크 가득 차면 체크포인트 저장 단계에서 죽습니다 (실제로 한 번 발생). 학습 시작
  전 `df -h /home` 확인 권장.

---

## 10. 다음에 할 만한 것

1. **SF / OpenVLA 학습 완료 후 HF 업로드** (현재는 pi0.5 만 완료)
2. **실제 로봇에서 4 task 평가**: deploy 스크립트의 4 hook 채우고 `validate_policy.py` 통과 후 real-world rollout.
3. **새 task 추가**: rosbag 수집 → `convert.py` → 새 lerobot dataset → `local__jisang_combined` 에 추가 (merge 스크립트 재실행) → 새 TrainConfig 또는 기존 combined 재학습.
4. **action_horizon 늘리기**: 현재 10. 30 ~ 50 까지 늘리면 OpenVLA 외 모델들의 시간적 일관성 향상 가능.
5. **norm stats 를 full dataset (max-frames=None) 으로 다시 계산**: 현재 2048 frame sample 로 학습됨. 결과 약간 더 정확해질 수 있음 (단 재학습 필요).

---

## 11. 자주 묻는 질문

**Q. 어디서부터 시작하면 됩니까?**
1. 이 문서 §0, §2, §3, §5 를 먼저 읽어 데이터 / action / 정규화 컨벤션 파악.
2. `DEPLOY.md` 보고 서버-클라이언트 구조 이해.
3. `validate_policy.py` 로 학습 데이터 dry-run → 모델이 살아있는지 확인.
4. 로봇에 연결할 준비가 되면 `deploy/deploy_*.py` 의 4 hook 을 본인 환경에 맞춰 채움.

**Q. HF 에서 pi0.5 체크포인트 받는 법은?**
```bash
huggingface-cli download honggyuAn/jisang_robot \
    --include "pi05_combined/99999/*" \
    --local-dir ./checkpoints_pi05
# 그 후 serve_policy.py 의 --policy.dir 로 ./checkpoints_pi05/pi05_combined/99999 지정
```

**Q. 새 dataset 변환 시 task text 가 다르면?**
- `convert.py` 의 `--task "..."` 를 다르게 주고 변환.
- 합쳐서 학습하려면 `local__jisang_combined` 의 merge 스크립트 재실행 (해당 작업 코드는 conversation history 에 있고 향후 필요하면 `merge_combined.py` 로 따로 코드화 권장).

**Q. q01/q99 가 어떤 값인지 보고 싶어요.**
```bash
cat openpi/assets/pi05_jisang_combined/local/jisang_combined/norm_stats.json | python -m json.tool
```

**Q. 학습 중 OOM 나면?**
- batch_size 줄이기 (8 → 4) 또는 GPU 늘리기. `--fsdp_devices=N` 으로 sharding 강도 조절.
- pi0.5 는 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` 환경변수로 JAX preallocate 줄여서 다른
  사용자 프로세스와 공존 가능.

---

**작성일**: 2026-05-25
**작성 컨텍스트**: 4 데이터셋 multi-task 학습 완료 (pi0.5) / 진행 중 (SF, OpenVLA), pi0.5 HF 업로드 완료
