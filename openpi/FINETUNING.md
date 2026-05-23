# LeRobot v3.0 Dataset → openpi pi0.5 / OpenVLA / Spatial-Forcing Fine-Tuning

이 문서는 `/home/cvlab/project/realsangbeom/robot/lerobot` (LeRobot v3.0, 1693 episodes / 273465 frames / 40 tasks / panda / fps=10) 데이터로 세 모델을 fine-tuning하는 절차를 정리한 것입니다.

> **real robot 배포가 목표라면 openpi-SF (§3) 가 공식 추천 경로**입니다. openvla-SF는 LIBERO 시뮬 재현용입니다.

데이터셋 키 매핑 (양쪽 공통):

| 데이터셋 키 | 의미 | 모델 입력 |
|---|---|---|
| `observation.images.image` | 3인칭 외부 카메라 (256x256) | primary image (`image` / `base_0_rgb`) |
| `observation.images.image2` | 그리퍼 시점 (wrist) | wrist image (`wrist_image` / `left_wrist_0_rgb`) |
| `observation.state` | 8-dim 상태 | state |
| `action` | 7-dim 액션 | action |
| `task` / `task_index` | 언어 명령 (40개) | prompt |

---

## 1. openpi pi0.5 fine-tuning

환경: 기존 conda env `openpi` (Python 3.13, uv 관리).

### 1-1. 환경 동기화

```bash
conda activate openpi
cd /home/cvlab/project/realsangbeom/robot/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

이 단계에서 다음 핀들이 적용됩니다 (이미 수정 완료):
- `lerobot` → `v0.4.4` (v3.0 codebase 지원)
- `numpy` 상한 제거, `override-dependencies = ["ml-dtypes>=0.4.0", "tensorstore==0.1.74", "numpy>=1.26.0"]`
- `src/openpi/training/data_loader.py` 의 import → `lerobot.datasets.lerobot_dataset`
- `DataConfig.local_data_root` 필드 추가로 로컬 데이터셋 지원

### 1-2. 등록된 TrainConfig

`src/openpi/training/config.py` 에 추가된 항목:

| 이름 | 내용 |
|---|---|
| `LeRobotLocalLiberoDataConfig` | 로컬 v3.0 LeRobot 데이터셋용 DataConfigFactory. 기존 LIBERO 키 매핑 대신 `observation.images.image{,2}` / `action` / `task` 를 LIBERO inputs 형식으로 repack. `action_sequence_keys=("action",)`. |
| `pi05_libero_local` TrainConfig | model=`Pi0Config(pi05=True, action_horizon=10)`, weight=pi05_base, batch_size=32. `local_data_root=/home/cvlab/project/realsangbeom/robot/lerobot`. |

### 1-3. norm stats 계산

```bash
cd /home/cvlab/project/realsangbeom/robot/openpi

# 빠른 스모크 테스트 (수십 초)
uv run scripts/compute_norm_stats.py --config-name=pi05_libero_local --max-frames=2048

# 본 학습용 (모든 프레임, 수 분 ~ 수십 분 소요)
uv run scripts/compute_norm_stats.py --config-name=pi05_libero_local
```

결과: `assets/pi05_libero_local/local/scannet_panda/norm_stats.json`

### 1-4. 학습 실행

```bash
cd /home/cvlab/project/realsangbeom/robot/openpi

uv run scripts/train.py pi05_libero_local --exp_name=panda_libero_run0
```

참고:
- pi0.5 베이스 체크포인트가 `gs://openpi-assets/checkpoints/pi05_base/params` 에서 자동 다운로드됩니다 (GCS 익명 read 가능해야 함).
- 체크포인트는 `checkpoints/pi05_libero_local/<exp_name>/` 에 저장.
- 다중 GPU FSDP는 `--fsdp_devices=N` 옵션.
- PyTorch backend로 학습하려면 `uv run scripts/train_pytorch.py ...`.

---

## 2. OpenVLA fine-tuning

환경: 새로 만든 conda env `openvla` (Python 3.10, torch 2.2.1, flash-attn 2.5.5).

### 2-1. 환경 (이미 구축 완료)

```bash
conda activate openvla
# torch==2.2.1, transformers==4.40.1, flash-attn==2.5.5, PyAV==15.1.0, numpy==1.26.4
```

만약 환경을 처음부터 재구축해야 한다면:

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
cd /home/cvlab/project/realsangbeom/robot/openvla
pip install -e .
pip install --upgrade "torch==2.2.1" "torchvision==0.17.1" "torchaudio==2.2.1"
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
# PyAV 는 별도로 필요 — lerobot 패키지에 의존하지 않고 mp4 디코딩에 사용
pip install "av==15.1.0"
```

LeRobot 파이썬 패키지는 **설치하지 않습니다** — openpi 환경과 달리 openvla 환경에서는 torch/transformers 핀 충돌 때문에 lerobot 라이브러리 대신 v3.0 포맷을 직접 읽는 self-contained reader를 추가했습니다 (`prismatic/vla/datasets/lerobot_dataset.py`).

### 2-2. 추가된 코드

| 파일 | 내용 |
|---|---|
| `prismatic/vla/datasets/lerobot_dataset.py` (신규) | `LeRobotV3Index` (메타 + 인덱스 해석), PyAV 기반 비디오 디코더, `LeRobotDatasetForOpenVLA(Dataset)` map-style. BOUNDS_Q99 액션 정규화. |
| `prismatic/vla/datasets/__init__.py` | `LeRobotConfig`, `LeRobotDatasetForOpenVLA` export |
| `vla-scripts/finetune_lerobot.py` (신규) | 원본 `finetune.py` 를 LeRobot 용으로 포팅. RLDSDataset → LeRobotDatasetForOpenVLA, `DistributedSampler` + `num_workers>0`, epoch loop. |
| `pyproject.toml` | torch 핀 `2.2.0` → `2.2.1` |

### 2-3. 빠른 스모크 테스트

```bash
conda activate openvla
cd /home/cvlab/project/realsangbeom/robot/openvla

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_lerobot.py \
    --vla_path openvla/openvla-7b \
    --lerobot_root /home/cvlab/project/realsangbeom/robot/lerobot \
    --dataset_name scannet_panda \
    --run_root_dir runs \
    --adapter_tmp_dir adapter-tmp \
    --batch_size 4 --num_workers 2 \
    --max_steps 10 --save_steps 1000 \
    --stats_max_frames 2048 \
    --wandb_project openvla --wandb_entity <YOUR_WANDB_ENTITY>
```

처음 한 번은 openvla-7b 체크포인트 (~14GB)와 action quantile 계산이 일어나므로 수 분 ~ 십수 분 소요.

### 2-4. 본 학습 (단일 GPU LoRA)

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_lerobot.py \
    --vla_path openvla/openvla-7b \
    --lerobot_root /home/cvlab/project/realsangbeom/robot/lerobot \
    --dataset_name scannet_panda \
    --run_root_dir runs \
    --adapter_tmp_dir adapter-tmp \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --max_steps 50000 \
    --save_steps 5000 \
    --num_workers 4 \
    --stats_max_frames 50000 \
    --wandb_project openvla --wandb_entity <YOUR_WANDB_ENTITY>
```

다중 GPU는 `--nproc-per-node N` 으로 설정.

### 2-5. wrist 카메라로 학습

`--image_video_key observation.images.image2` 를 추가.

### 2-6. 체크포인트와 norm stats 위치

- LoRA adapter: `adapter-tmp/<exp_id>--lerobot/`
- 머지된 풀 모델: `runs/<exp_id>--lerobot/`
- Inference용 정규화 통계: `runs/<exp_id>--lerobot/dataset_statistics.json`
  - 키: `{ "<dataset_name>": { "action": { "q01": [...], "q99": [...], "mask": [...] } } }`

---

## 3. Spatial-Forcing (openpi-SF) — real-world 권장

`Spatial-Forcing/openpi-SF` 는 openpi 의 `train_pytorch.py` 위에 **VGGT 3D foundation model 의 geometric feature 를 VLA 의 중간 visual embedding 과 정렬 (alignment loss)** 시키는 학습 스크립트 `train_align_pytorch.py` 를 추가한 변형입니다. 베이스 모델은 `pi0_base` PyTorch 버전.

### 3-1. 환경 (이미 구축 완료)

conda env `openpi-SF` (Python 3.11) + uv venv (`./openpi-SF/.venv`):

| 패키지 | 버전 |
|---|---|
| torch | 2.7.1 |
| lerobot | 0.4.4 (v3.0 codebase 지원) |
| torchcodec | 0.4.0 (사용 안 함, PyAV 로 fallback) |
| PyAV | 14.x (lerobot deps) |
| chex | 0.1.90 (수동 추가) |
| transformers | 4.53.2 (`src/openpi/models_pytorch/transformers_replace` 패치됨) |

재구축 시:

```bash
conda create -n openpi-SF python=3.11 -y
conda activate openpi-SF
cd /home/cvlab/project/realsangbeom/robot/Spatial-Forcing/openpi-SF

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

### 3-2. 패치된 파일

`openpi-SF/pyproject.toml`:
- `lerobot` 핀 → `tag = "v0.4.4"`
- `numpy>=1.26.0`, override-dependencies 에 `ml-dtypes>=0.4.0`, `numpy>=1.26.0` 추가
- `chex==0.1.90` 추가

`openpi-SF/packages/openpi-client/pyproject.toml`:
- `numpy>=1.26.0`

`openpi-SF/src/openpi/training/data_loader.py`:
- import 경로 → `lerobot.datasets.lerobot_dataset`
- `LeRobotDataset(..., root=local_root, video_backend="pyav")` — 시스템 ffmpeg 가 v4 라서 torchcodec ABI 충돌 회피용 PyAV 강제

`openpi-SF/src/openpi/training/config.py`:
- `DataConfig.local_data_root` 필드 추가
- `LeRobotLocalLiberoDataConfig` 클래스 추가 (openpi 와 동일)
- 새 `TrainConfig name="pi0_align_libero_local"` 추가 (Panda single-arm LIBERO-style + SF alignment fields)

### 3-3. 추가 체크포인트 다운로드 (학습 전 한 번)

```bash
conda activate openpi-SF
cd /home/cvlab/project/realsangbeom/robot/Spatial-Forcing/openpi-SF

# (a) pi0_base 를 JAX → PyTorch 변환
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi0_base \
    --config_name pi0_align_libero_local \
    --output_path ./checkpoints/pi0_base_full_torch

# (b) VGGT-1B 체크포인트 (face/VGGT-1B 사용)
mkdir -p ./checkpoints/vggt
# https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt 에서 받아서 위 디렉토리에 배치
# (huggingface_hub 으로 받으려면)
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/VGGT-1B', filename='model.pt', local_dir='./checkpoints/vggt')"
```

### 3-4. norm stats

```bash
# 스모크 테스트 (smoke test 검증 완료, 약 13분 소요)
uv run scripts/compute_norm_stats.py --config-name=pi0_align_libero_local --max-frames=512

# 본 학습용 (PyAV CPU 디코딩이라 시간 많이 소요 — full 273K 프레임 권장하지 않음. 50K 정도가 합리적)
uv run scripts/compute_norm_stats.py --config-name=pi0_align_libero_local --max-frames=50000
```

결과: `assets/pi0_align_libero_local/local/scannet_panda/norm_stats.json`

### 3-5. 학습 실행

```bash
# 단일 GPU
uv run scripts/train_align_pytorch.py pi0_align_libero_local --exp_name=sf_run0

# 단일 노드 다중 GPU (4 GPU)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_align_pytorch.py pi0_align_libero_local --exp_name=sf_run0

# 재개 / 덮어쓰기
uv run scripts/train_align_pytorch.py pi0_align_libero_local --exp_name=sf_run0 --resume
uv run scripts/train_align_pytorch.py pi0_align_libero_local --exp_name=sf_run0 --overwrite
```

`TrainConfig` 의 핵심 SF 필드 (값을 바꾸려면 `src/openpi/training/config.py` 의 `pi0_align_libero_local` 항목 수정):

| 필드 | 기본값 | 의미 |
|---|---|---|
| `vla_layers_align` | 12 | paligemma-2b 의 어느 레이어 출력을 align 대상 (총 18 레이어) |
| `vggt_layers_align` | -1 | VGGT 의 어느 레이어 feature 와 매칭 (총 24, -1 = 마지막) |
| `pooling_func` | "bilinear" | VGGT feature 를 VLA token grid 에 맞추는 풀링 |
| `use_vggt_pe` | True | VGGT positional encoding 사용 여부 |
| `use_vlm_norm` | True | VLM-side feature 정규화 |
| `align_loss_coeff` | 0.5 | alignment loss 가중치 (총 loss = action_loss + α·align_loss) |

### 3-6. 서버-클라이언트 inference

학습 후 (예: step 20000 체크포인트):

```bash
# 서버
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_align_libero_local \
    --policy.dir=checkpoints/pi0_align_libero_local/sf_run0/20000

# 클라이언트 (자기 로봇용 스크립트 직접 작성 필요 — examples/simple_client/main.py 참고)
uv run examples/simple_client/main.py --env <your_env>
```

### 3-7. 주의사항

- **PyAV 백엔드는 GPU 디코딩 불가** — torchcodec 0.4.0+0.10.0 모두 시스템 ffmpeg 4 / torch ABI 와 충돌해서 PyAV 로 강제 fallback. 학습 속도가 보장 안 되면 데이터로더 워커 수 (`--num_workers` 같은 인자) 를 올리거나, 시스템 ffmpeg 5+ 설치 후 torchcodec 동작 재확인이 필요.
- **GPU OOM 경고** (`CUDA_ERROR_OUT_OF_MEMORY` for CUDA:4-7) — norm stats 계산 중 다른 사용자가 GPU 4-7 점유 중일 때 발생. CPU bound 작업이라 학습엔 영향 없음. GPU 0-3 사용 가능 여부만 확인.
- **`pi05` 가능 여부**: SF 코드는 pi05 분기를 가지지만 공식 example 은 pi0 만 검증됨. pi05 로 가려면 `model=pi0_config.Pi0Config(pi05=True)` + `weight_loader` 를 `pi05_base/params` 로 바꾸고 추가 하이퍼파라 튜닝 가능성 있음.

---

## 4. 공통 주의사항

- **카메라 매핑 검증 완료**: `observation.images.image` = 3인칭 외부 카메라, `observation.images.image2` = 그리퍼 시점 (wrist). 데이터 출처는 LIBERO 계열로 추정 (40 tasks 의 프롬프트가 LIBERO 표준).
- **액션 차원**: 7-dim (xyz + rpy + gripper). state 는 8-dim.
- **action_horizon**: pi0.5 는 10 step chunk 학습 (`pi05_libero_local`). OpenVLA 는 single-step (chunking 없음).
- **카메라 두 개 모두** 학습에 쓰고 싶다면 pi0.5 는 LIBERO 의 base/wrist 슬롯에 동시 입력 (현재 config 가 그렇게 설정됨). OpenVLA 는 single image 모델이라 한 번에 하나만 사용 — wrist 학습이 필요하면 별도 run.
- **norm stats 일관성**: 학습 시 사용한 q01/q99 와 inference 시 denormalize 통계가 같아야 합니다. 체크포인트 디렉토리에 자동 저장되므로 inference 시 그걸 불러오면 됩니다.
- **GCS 접근** (openpi pi05_base 체크포인트용): 익명 read 가 가능한 버킷이지만, 회사 네트워크에서 GCS 가 막혀있다면 사전에 `gsutil cp -r gs://openpi-assets/checkpoints/pi05_base/ <local_path>` 로 받아두고 `weight_loader` 경로를 수정해야 함.
