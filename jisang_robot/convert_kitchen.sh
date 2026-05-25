#!/bin/bash
set -u

PROJECT=/home/cvlab/project/realsangbeom/robot/jisang_robot
BAG_DIR=$PROJECT/robot/kitchen
OUT=$PROJECT/lerobot_datasets
LOG=$PROJECT/convert_kitchen.log
PY=/home/cvlab/miniconda3/envs/bag_convert/bin/python
export PYTHONPATH=$PROJECT/pycontroller_template-dev-corl26/src

# NOTE: deliberately NOT removing $OUT — other datasets (pick_and_place, chocomilk) live there.
: > "$LOG"

bags=( "$BAG_DIR"/recording_*/recording.bag )
total=${#bags[@]}
echo "[$(date '+%F %T')] Converting $total kitchen bags" | tee -a "$LOG"

i=0
failed=0
for bag in "${bags[@]}"; do
  i=$((i+1))
  echo "[$(date '+%F %T')] ($i/$total) $bag" >> "$LOG"
  "$PY" "$PROJECT/convert.py" \
    --bag "$bag" \
    --output "$OUT" \
    --repo_id local/kitchen \
    --fps 15 \
    --task "pick and place the kitchen item" \
    >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    failed=$((failed+1))
    echo "[$(date '+%F %T')] FAILED rc=$rc: $bag" | tee -a "$LOG"
  fi
done

echo "[$(date '+%F %T')] DONE. converted=$((i-failed)) failed=$failed" | tee -a "$LOG"
