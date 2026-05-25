# jisang_robot LeRobot v3.0 → openpi pi0.5 / OpenVLA / openpi-SF Fine-Tuning

이 문서는 `/home/cvlab/project/realsangbeom/robot/jisang_robot/lerobot_datasets/local__pick_and_place`
(LeRobot v3.0, 1 task, 15Hz, 7D joint state, 7D EE-delta+gripper action, 1280×720 dual-cam)
데이터로 세 모델을 fine-tuning하는 절차입니다.

기존 `/home/cvlab/project/realsangbeom/robot/FINETUNING.md` (reference `/robot/lerobot` 데이터셋용)
과 데이터 표현이 다르므로, 별도 config / loader 옵션을 추가했습니다.

데이터셋 키 매핑 (양쪽 공통, 세 모델 모두):

| 데이터셋 키 | 의미 | 모델 입력 |
|---|---|---|
| `observation.images.external_image` | 3인칭 외부 카메라 (720×1280) | primary image (`image` / `base_0_rgb`) |
| `observation.images.wrist_image` | 그리퍼 시점 (720×1280) | wrist image (`wrist_image` / `left_wrist_0_rgb`) |
| `observation.state` | 7D joint angles | state (모델 내부에서 max_state_dim까지 zero-pad) |
| `action` | 7D (6D EE delta + gripper) | action |
| `task` / `task_index` | 언어 명령 (1개, `"pick and place the object"`) | prompt |

표현이 reference와 다른 부분:

| 항목 | reference (`/robot/lerobot`) | 이 데이터셋 |
|---|---|---|
| state | 8D EE pose + gripper | 7D joint angles |
| action | 7D absolute | 7D EE-delta + gripper |
| fps | 10 | 15 |
| 이미지 키 | `image` / `image2` | `external_image` / `wrist_image` |
| tasks | 40개 LIBERO | 1개 pick-and-place |

→ reference 체크포인트와 inference 호환되지 않습니다. norm stats 등 학습 결과물도 별도 디렉토리에 떨어집니다.

---

## 1. openpi pi0.5

새 config: `pi05_jisang_pick_and_place` (`openpi/src/openpi/training/config.py`)
새 DataConfig: `LeRobotJisangDataConfig` (동일 파일)

### 1-1. norm stats

```bash
conda activate openpi
cd /home/cvlab/project/realsangbeom/robot/openpi

# 빠른 스모크 테스트
uv run scripts/compute_norm_stats.py --config-name=pi05_jisang_pick_and_place --max-frames=2048

# 본 학습용 (이 데이터셋은 16710 frame이므로 전부 써도 됨)
uv run scripts/compute_norm_stats.py --config-name=pi05_jisang_pick_and_place
```

결과: `assets/pi05_jisang_pick_and_place/local/jisang_pick_and_place/norm_stats.json`

### 1-2. 학습

```bash
cd /home/cvlab/project/realsangbeom/robot/openpi
uv run scripts/train.py pi05_jisang_pick_and_place --exp_name=jisang_run0
```

- 체크포인트: `checkpoints/pi05_jisang_pick_and_place/jisang_run0/`
- 다중 GPU FSDP: `--fsdp_devices=N`
- PyTorch backend: `uv run scripts/train_pytorch.py ...`

---

## 2. OpenVLA

self-contained reader (`prismatic/vla/datasets/lerobot_dataset.py`) 의 `LeRobotConfig`에
`image_resize_hw` / `fixed_prompt` 옵션을 추가했고, `vla-scripts/finetune_lerobot.py` 에
`--image_resize_h/--image_resize_w` / `--fixed_prompt` CLI 인자를 노출했습니다.

### 2-1. 스모크 테스트

```bash
conda activate openvla
cd /home/cvlab/project/realsangbeom/robot/openvla

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_lerobot.py \
    --vla_path openvla/openvla-7b \
    --lerobot_root /home/cvlab/project/realsangbeom/robot/jisang_robot/lerobot_datasets/local__pick_and_place \
    --dataset_name local__pick_and_place \
    --image_video_key observation.images.external_image \
    --image_resize_h 256 --image_resize_w 256 \
    --fixed_prompt "pick and place the object" \
    --run_root_dir runs --adapter_tmp_dir adapter-tmp \
    --batch_size 4 --num_workers 2 \
    --max_steps 10 --save_steps 1000 \
    --stats_max_frames 2048 \
    --wandb_project openvla --wandb_entity <YOUR_WANDB_ENTITY>
```

### 2-2. 본 학습 (단일 GPU LoRA)

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_lerobot.py \
    --vla_path openvla/openvla-7b \
    --lerobot_root /home/cvlab/project/realsangbeom/robot/jisang_robot/lerobot_datasets/local__pick_and_place \
    --dataset_name local__pick_and_place \
    --image_video_key observation.images.external_image \
    --image_resize_h 256 --image_resize_w 256 \
    --fixed_prompt "pick and place the object" \
    --run_root_dir runs --adapter_tmp_dir adapter-tmp \
    --batch_size 16 --learning_rate 5e-4 \
    --max_steps 50000 --save_steps 5000 --num_workers 4 \
    --stats_max_frames 16710 \
    --wandb_project openvla --wandb_entity <YOUR_WANDB_ENTITY>
```

- wrist 카메라로 학습하려면 `--image_video_key observation.images.wrist_image` 로 별도 run.
- 다중 GPU: `--nproc-per-node N`.

### 2-3. 출력

- LoRA adapter: `adapter-tmp/<exp_id>--lerobot/`
- 머지된 풀 모델 + dataset_statistics.json: `runs/<exp_id>--lerobot/`

---

## 3. openpi-SF (Spatial-Forcing, real-world 권장)

새 config: `pi0_align_jisang_pick_and_place` (`Spatial-Forcing/openpi-SF/src/openpi/training/config.py`)
새 DataConfig: `LeRobotJisangDataConfig` (동일 파일)

### 3-1. 사전 준비 (한 번)

```bash
conda activate openpi-SF
cd /home/cvlab/project/realsangbeom/robot/Spatial-Forcing/openpi-SF

# (a) pi0_base 를 JAX → PyTorch 변환
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi0_base \
    --config_name pi0_align_jisang_pick_and_place \
    --output_path ./checkpoints/pi0_base_full_torch

# (b) VGGT-1B 체크포인트
mkdir -p ./checkpoints/vggt
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/VGGT-1B', filename='model.pt', local_dir='./checkpoints/vggt')"
```

기존 reference 학습 후 (a)를 이미 한 번 했다면 그대로 재사용 가능합니다.

### 3-2. norm stats

```bash
uv run scripts/compute_norm_stats.py --config-name=pi0_align_jisang_pick_and_place --max-frames=2048   # 스모크
uv run scripts/compute_norm_stats.py --config-name=pi0_align_jisang_pick_and_place                    # 본 학습용
```

결과: `assets/pi0_align_jisang_pick_and_place/local/jisang_pick_and_place/norm_stats.json`

### 3-3. 학습

```bash
# 단일 GPU
uv run scripts/train_align_pytorch.py pi0_align_jisang_pick_and_place --exp_name=sf_jisang_run0

# 단일 노드 4 GPU
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_align_pytorch.py pi0_align_jisang_pick_and_place --exp_name=sf_jisang_run0
```

SF alignment 하이퍼파라 (`vla_layers_align=12`, `vggt_layers_align=-1`, `pooling_func="bilinear"`,
`align_loss_coeff=0.5` 등) 는 reference config 와 동일하게 잡혀 있습니다. 바꾸려면
`pi0_align_jisang_pick_and_place` TrainConfig 항목에서 직접 수정.

### 3-4. 서버-클라이언트 inference

학습 후 (예: step 20000):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_align_jisang_pick_and_place \
    --policy.dir=checkpoints/pi0_align_jisang_pick_and_place/sf_jisang_run0/20000
```

클라이언트 쪽에서는 **7D joint state + 7D EE-delta+gripper action** 표현으로 obs/action 을
주고받아야 합니다 (학습 시 표현과 일치해야 함).

---

## 4. 주의사항

### 4-1. 표현 호환성

- 학습 시 사용한 state(7D joints) / action(7D EE delta + gripper) 정의가 inference 시
  클라이언트가 보내는 obs / 받는 action 과 같아야 합니다. reference (`/robot/lerobot`)
  로 학습한 체크포인트와 **호환되지 않습니다** — norm stats / state dim / action 의미가
  모두 다릅니다.

### 4-2. state / action zero-padding

- `LiberoInputs` 는 state / action 을 그대로 통과시키고, pi0 모델 내부에서
  `max_state_dim` / `max_action_dim` 까지 zero-pad 합니다. 즉 7D 입력이어도 학습
  자체는 문제 없이 됩니다. inference 시 `LiberoOutputs` 가 첫 7 차원만 잘라서
  반환합니다.

### 4-3. 이미지 해상도 / fps

- openpi / openpi-SF: `ModelTransformFactory` 가 어차피 224×224 로 resize 하므로 추가
  256×256 resize 는 적용하지 않았습니다. 다만 비디오 디코딩은 native 720×1280 에서
  일어나므로 dataloader CPU 부담이 큽니다. 필요하다면 `LeRobotDataset(..., image_transforms=...)`
  인자를 `data_loader.py` 에 노출해 디코딩 직후 resize 하도록 추가 패치 가능.
- OpenVLA: self-contained reader 이므로 `--image_resize_h 256 --image_resize_w 256`
  옵션으로 디코딩 직후 PIL resize 가 적용됩니다.
- fps 는 native 15Hz 그대로 사용합니다. pi0.5 의 `action_horizon=10` 은 chunk 길이
  (10 frame = 약 0.67 s @ 15Hz) 로 해석됩니다 — reference (10 frame = 1 s @ 10Hz) 와
  시간 길이가 다르니 inference 시 control loop 주기를 데이터셋 fps 에 맞춰야 합니다.

### 4-4. 단일 task

- `task_index` 가 항상 0 이고 prompt 가 항상 `"pick and place the object"` 입니다.
  pi0.5 / openpi-SF 는 `prompt: "task"` repack 으로 frame 마다 같은 문자열을 받습니다.
  OpenVLA 는 `--fixed_prompt "pick and place the object"` 로 강제할 수 있습니다.

### 4-5. 코드 변경 위치 요약

| 파일 | 변경 |
|---|---|
| `openpi/src/openpi/training/config.py` | `LeRobotJisangDataConfig` 클래스 + `pi05_jisang_pick_and_place` TrainConfig 추가 |
| `Spatial-Forcing/openpi-SF/src/openpi/training/config.py` | `LeRobotJisangDataConfig` 클래스 + `pi0_align_jisang_pick_and_place` TrainConfig 추가 |
| `openvla/prismatic/vla/datasets/lerobot_dataset.py` | `LeRobotConfig` 에 `image_resize_hw` / `fixed_prompt` 필드 + `_decode_image` 에서 PIL resize 적용 |
| `openvla/vla-scripts/finetune_lerobot.py` | `--image_resize_h/--image_resize_w/--fixed_prompt` CLI 인자 노출 |
