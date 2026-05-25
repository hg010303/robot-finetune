# LeRobot 데이터셋 구조 비교

`/home/cvlab/project/realsangbeom/robot/lerobot` (reference)와
`/home/cvlab/project/realsangbeom/robot/jisang_robot/lerobot_datasets/local__pick_and_place`
(`convert.py`로 생성 중인 출력)의 구조 비교.

## 비슷한 점

- `codebase_version`: `v3.0` (동일)
- 디렉토리 구조 동일
  - `meta/`
  - `data/chunk-000/file-*.parquet`
  - `videos/<video_key>/chunk-000/file-*.mp4`
- meta 파일 구성 동일
  - `info.json`, `stats.json`, `tasks.parquet`, `episodes/chunk-000/*.parquet`
- 비디오 인코딩 동일
  - AV1 / yuv420p
  - episode 당 1개의 `.mp4` 파일
- parquet 공통 컬럼
  - `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`
- 경로 패턴 동일
  - `data_path`: `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
  - `video_path`: `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`

## 다른 점

| 항목 | reference (`/robot/lerobot`) | 우리 출력 (`local__pick_and_place`) |
|---|---|---|
| `robot_type` | `"panda"` | `null` |
| `fps` | 10 | 15 |
| 이미지 키 이름 | `observation.images.image`, `observation.images.image2` | `observation.images.external_image`, `observation.images.wrist_image` |
| 이미지 해상도 | 256×256 (리사이즈됨) | **720×1280 (원본 그대로)** |
| `observation.state` shape | (8,), names=`["state"]` | (7,), names=`["joint_1"..."joint_7"]` |
| `action` shape | (7,), names=`["actions"]` | (7,), names=`["dx_eef","dy_eef","dz_eef","drx_eef","dry_eef","drz_eef","gripper"]` |
| action 의미 | 절대 EE pose 등 7D 벡터 (예: `[0.016, 0, -0, 0, 0, -0, -1]`) | **EE-frame delta + gripper** (DROID 스타일, `decompose_to_eef_deltas`) |
| 각 feature `fps` 키 | image/state/action 모두 명시 | image에만 명시 |
| parquet 컬럼 타입 | `list<float>` (variable size) | `fixed_size_list<float>[7]` |
| `tasks` 개수 | 40개 (LIBERO 스타일 다양한 task) | 1개 (`"pick and place the object"`) |
| `total_episodes` | 1693 | 299 (예정, 변환 중) |
| `video_files_size_in_mb` | 500 | 200 |

## 주요 시사점

1. **action 정의가 다릅니다**
   - reference: 절대 좌표 7D 벡터 (예: `[0.016, 0, ..., -1]` — XYZ 위치 + 어떤 표현 + gripper로 추정)
   - 우리: EE 프레임 상대 변위(`dx, dy, dz, drx, dry, drz`) + gripper
   - **동일한 policy로 학습/추론 호환 불가**. 같은 모델에 넣으려면 변환 필요.

2. **state 차원/이름이 다릅니다**
   - reference: 8D `"state"` (EE pose 7D + gripper 1D로 추정)
   - 우리: 7D joint angles (Panda 관절 각도)
   - 의미가 완전히 다른 representation.

3. **이미지 리사이즈 없음**
   - 720×1280 그대로 저장되어 비디오 파일이 큽니다.
   - reference처럼 256×256으로 줄이려면 `convert.py`에 `cv2.resize`를 추가해야 함.

4. **fps 차이**
   - reference: 10Hz, 우리: 15Hz
   - 동일 policy 학습 시 통일해야 함.

5. **parquet 컬럼 타입**
   - `fixed_size_list` vs `list`는 lerobot 신/구 버전 차이로 보이며 호환 대부분 OK.

## reference 포맷에 맞추려면 `convert.py`에서 수정해야 할 것

- `--fps 10`으로 변환
- 이미지 키 이름을 `observation.images.image`, `observation.images.image2`로 변경 (또는 호환 layer)
- 이미지 리사이즈 256×256 적용 (`cv2.resize` 추가)
- `robot_type="panda"` 설정 — `LeRobotDataset.create()`에 `robot_type` 인자 전달
- `observation.state`를 EE pose(7D) + gripper(1D) 형태로 만들어 8D `"state"`로 저장
- `action`을 EE delta 대신 reference와 동일한 정의로 재계산
- features의 각 entry에 `"fps"` 키 명시
