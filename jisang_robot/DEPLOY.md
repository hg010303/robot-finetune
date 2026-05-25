# Real-Robot Deploy Guide — pi0.5 / Spatial-Forcing / OpenVLA

세 모델 모두 같은 입출력 표현 (Panda 7-DOF joint state 7D + EE-frame delta action
7D + gripper) 으로 학습되어 있어 클라이언트 측 로봇 인터페이스는 **공통**입니다.
다른 점은 (a) 모델 서빙 방식, (b) action chunk 길이.

| 항목 | pi0.5 (`pi05_jisang_combined`) | SF (`pi0_align_jisang_combined`) | OpenVLA (`local__jisang_combined`) |
|---|---|---|---|
| 서빙 방식 | openpi websocket server | openpi websocket server | 인-프로세스 (HF transformers) |
| Action chunk | (10, 7) | (10, 7) | (1, 7) per query |
| 카메라 | external + wrist | external + wrist | external (single-image) |
| Replan 주기 | 5 step (chunk 의 앞 5) | 5 step | 매 step |

학습된 4 task 의 prompt 그대로 보내야 하며, deploy 스크립트의 `TASK_PROMPTS` 에
4개 다 들어있습니다:

```python
TASK_PROMPTS = {
    "pick_and_place": "Pick up the cube and place it on the plate.",
    "chocomilk":      "Pick up the chocolate milk and place it on top of the white milk, ...",
    "pot":            "Open the lid of the pot, put the yellow cube inside the pot, ...",
    "kitchen":        "Pick up the pot and place it on the bottom section of the induction cooktop, ...",
}
```

수행할 태스크는 각 스크립트 상단의 `TASK = "..."` 한 줄만 바꿉니다.

---

## 1. 공통 — 클라이언트가 채워야 할 4 hook

세 deploy 스크립트 (`deploy_pi05.py`, `deploy_sf.py`, `deploy_openvla.py`) 모두
다음 placeholder 를 사용자가 본인 환경에 맞게 채워야 합니다:

| Hook | 반환 | 비고 |
|---|---|---|
| `get_external_image()` | RGB uint8 ndarray (또는 OpenVLA 는 PIL.Image) | 임의 해상도 OK — 서버/processor 가 224×224 resize |
| `get_wrist_image()` | RGB uint8 ndarray | OpenVLA 는 미사용 |
| `get_joint_angles()` | (7,) float32, radians | Panda joint encoder |
| `apply_ee_delta(delta_xyz_rpy, gripper)` | — | EE-frame delta 한 step 을 로봇에 적용 |

`apply_ee_delta` 구현 예 (pseudo):

```python
def apply_ee_delta(delta, gripper):
    current = robot.read_ee_pose()                       # SE(3)
    target  = compose_ee_delta(current, delta)           # inverse of decompose_to_eef_deltas
    joint_q = ik_solver(target)
    robot.servo(joint_q, dt=1/15)                        # match CONTROL_HZ=15
    robot.set_gripper(gripper)                           # 0=open, 1=close
```

`compose_ee_delta` 는 학습 데이터 변환 시 사용한
`pycontroller_template.controls.chunk_compose.decompose_to_eef_deltas` 의 **역연산**
입니다 (`jisang_robot/pycontroller_template-dev-corl26/` 안에 있으니 그대로 import).

**중요**: `CONTROL_HZ` 는 학습 데이터 fps (15Hz) 와 같이 맞춰야 합니다. 너무 빠르거나
느리면 EE-delta 의 누적 의미가 어긋납니다.

---

## 2. pi0.5 멀티태스크 정책 배포

### 2-1. GPU 머신에서 서버 시작

```bash
cd /home/cvlab/project/realsangbeom/robot/openpi
conda activate openpi

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_jisang_combined \
    --policy.dir=checkpoints/pi05_jisang_combined/combined_run0/<STEP>
```

`<STEP>` 자리에 사용할 체크포인트 step (예: `100000` 또는 가장 최신) 을 넣습니다.

서버가 `Listening on 0.0.0.0:8000` 까지 출력되면 준비 완료. norm_stats 와 dataset
config 는 체크포인트 안에 같이 저장돼 있어 자동 로드됩니다.

### 2-2. 로봇 PC 에서 클라이언트

```bash
# openpi-client 패키지 설치 (한 번만)
pip install /home/cvlab/project/realsangbeom/robot/openpi/packages/openpi-client

# 스크립트의 SERVER_HOST/PORT, TASK, hook 4개를 채운 뒤:
python jisang_robot/deploy/deploy_pi05.py
```

기본 옵션:
- `CONTROL_HZ = 15`, `REPLAN_EVERY = 5`
- 10-step chunk 의 앞 5 step 만 실행하고 다시 inference (책상 위 동작에 reactive)

---

## 3. Spatial-Forcing (pi0 + VGGT alignment) 배포

OpenPi 와 **동일한 서버/클라이언트 인프라**를 그대로 씁니다 (SF 는 alignment loss 가
학습 시점에만 작용, inference 시엔 일반 pi0 와 동일한 forward).

### 3-1. 서버

```bash
cd /home/cvlab/project/realsangbeom/robot/Spatial-Forcing/openpi-SF
conda activate openpi-SF

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_align_jisang_combined \
    --policy.dir=checkpoints/pi0_align_jisang_combined/sf_combined_run0/<STEP>
```

pi0.5 와 다른 점은 conda env (`openpi-SF`) 와 config 이름 (`pi0_align_jisang_combined`)
뿐입니다.

### 3-2. 클라이언트

```bash
python jisang_robot/deploy/deploy_sf.py    # pi0.5 와 동일한 obs/action 구조
```

pi0.5 와 SF 를 **동시에** 비교 배포하려면 서버 두 개를 다른 port (예: 8000, 8001) 로
띄우고 클라이언트의 `SERVER_PORT` 만 바꿉니다.

---

## 4. OpenVLA 배포

OpenVLA 는 별도 서버 없이 **클라이언트 프로세스가 직접 모델을 GPU 에 로드**합니다
(~14 GB). 로봇 PC 자체가 GPU 머신이 아니면 사용자가 직접 FastAPI / socket 래퍼를
짜야 합니다.

### 4-1. 사전 준비

학습 완료 후 머지된 풀 모델이 `runs/<exp_id>--lerobot/` 에 저장돼 있습니다 (LoRA
adapter 가 base 와 머지된 결과). `dataset_statistics.json` 도 같은 폴더에 있어야
합니다 — 없으면 `runs/.../dataset_statistics.json` 을 검색하거나 학습 시
`save_dataset_statistics(...)` 호출이 빠진 게 아닌지 확인.

### 4-2. 실행

```bash
conda activate openvla

# deploy_openvla.py 의 RUN_DIR 을 실제 머지된 모델 폴더로 바꾼 뒤:
python jisang_robot/deploy/deploy_openvla.py
```

OpenVLA 는 single-image 모델이라 wrist 카메라는 보지 않습니다. wrist-only 정책이
필요하면 학습 시 `--image_video_key observation.images.wrist_image` 로 별도 run.

### 4-3. (선택) 서버화

직접 서버화하려면 다음 패턴:

```python
from fastapi import FastAPI
# 위 deploy_openvla.py 의 load_model() 호출해서 processor, vla 보관
app = FastAPI()

@app.post("/infer")
def infer(req: dict):
    img = decode_image(req["image"])           # base64 또는 raw bytes
    inputs = processor(req["prompt"], img).to(DEVICE, dtype=torch.bfloat16)
    a = vla.predict_action(**inputs, unnorm_key=DATASET_NAME, do_sample=False)
    return {"action": np.asarray(a).tolist()}
```

---

## 5. 공통 주의사항

### 5-1. 표현 호환성
- 학습 시 obs/action 정의 (7D joint state, 7D EE delta + gripper, 256×256 dual-cam,
  fps=15) 와 동일하게 보내야 합니다. 한 dimension 이라도 다르면 모델 출력이
  무의미합니다.

### 5-2. norm_stats 일관성
- pi0.5 / SF: 체크포인트 디렉토리의 `assets/local/jisang_combined/norm_stats.json`
  가 자동 사용됩니다. 별도 작업 없음.
- OpenVLA: `runs/<exp_id>--lerobot/dataset_statistics.json` 이 `predict_action(...,
  unnorm_key=...)` 에서 자동 사용됩니다.

### 5-3. 안전
- 첫 step 에서 action 이 NaN 이거나 매우 큰 값이면 **즉시 정지**. deploy 스크립트의
  `apply_ee_delta` 안에서 `np.isfinite(delta).all() and np.abs(delta).max() < SAFE_MAX`
  같은 가드를 추가하는 게 안전합니다 (학습 데이터 q99 가 약 ±0.008 m / ±0.003 rad
  이므로 SAFE_MAX 0.05 정도면 충분).
- KeyboardInterrupt 로 깔끔하게 빠져나오게 메인 try/except 가 이미 들어 있습니다.

### 5-4. 디버깅 — 로봇 없이 dry run
- `jisang_robot/validate_policy.py` 가 학습 데이터에서 N 개 frame 을 샘플해서 서버에
  보내는 dry-run 스크립트입니다. pi0.5 / SF deploy 가 정상인지 로봇 연결 전에
  확인할 수 있습니다.

### 5-5. 코드 변경 위치 요약

| 모델 | deploy 코드 | 사전 준비 |
|---|---|---|
| pi0.5 | `jisang_robot/deploy/deploy_pi05.py` | openpi env, `openpi-client` pip 설치, 서버 띄우기 |
| SF | `jisang_robot/deploy/deploy_sf.py` | openpi-SF env, `openpi-client` 동일, 서버 띄우기 |
| OpenVLA | `jisang_robot/deploy/deploy_openvla.py` | openvla env, 머지된 모델 폴더 경로 |
