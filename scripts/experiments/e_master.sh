#!/bin/zsh
# Master driver for the fragility campaign — designed to be nohup'd so it
# survives session interruptions. Three lanes in parallel:
#   lane A (MPS): e5a_warmcrash -> E6 EMA x6 -> E9 sweep x8 -> E5b mid x6 -> E8 retrains
#   lane B (CPU): dagger_collect -> e8 mix (feeds the tail of lane A)
#   lane C (CPU): e34 three-arm training -> wide-drone fly test
# Every train/fly is skipped if its artifact already exists (idempotent restart).
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
FM=artifacts/skyjepa_rotor_mix_recovery/dataset.pt
RR=web-demo/racer
SCRATCH=artifacts
mkdir -p artifacts/logs
COMMON=(--action-mode rotor_force --pos-mode relative --width-mult 2 --batch 256 --stride 5 --device mps)

train() {
  local name=$1 data=$2 s1=$3 s2=$4; shift 4
  [ -f artifacts/$name.pt ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%H:%M)] TRAIN $name"
  $PY -m drone_jepa.train --data $data $COMMON --stage1-steps $s1 --stage2-steps $s2 \
      --out artifacts/$name.pt.tmp "$@" > artifacts/logs/$name.train.log 2>&1 \
    && mv artifacts/$name.pt.tmp artifacts/$name.pt
}
fly() {
  local name=$1
  [ -s artifacts/logs/$name.rotor_fly.log ] && grep -q "==" artifacts/logs/$name.rotor_fly.log && { echo "[skip-fly] $name"; return 0; }
  $PY scripts/export_jepa.py artifacts/$name.pt $name > /dev/null 2>&1
  $PY scripts/export_jepa_blob.py $name > /dev/null 2>&1
  ( cd $RR && ROTOR_BLOB=assets/$name.jblob ./target/release/examples/rotor_fly \
      > ../../artifacts/logs/$name.rotor_fly.log 2>/dev/null ) &
  echo "[$(date +%H:%M)] FLY-> $name (bg)"
}

lane_A() {
  train e5a_warmcrash_d1 $FM 8000 8000 --warm-start artifacts/blog_sep_i0_d0_w2.pt --data-seed 1 && fly e5a_warmcrash_d1
  for s in 41 0 1 2 3 4; do
    train e6_ema_s$s $FM 8000 8000 --seed $s --ema-tau 0.99 && fly e6_ema_s$s
  done
  for s in 1 2; do
    train e9_sm500_s$s artifacts/e9_mix_sm500.pt 8000 8000 --seed $s && fly e9_sm500_s$s
  done
  for s in 0 1 2; do
    train e9_sm1000_s$s artifacts/e9_mix_sm1000.pt 8000 8000 --seed $s && fly e9_sm1000_s$s
  done
  for s in 0 1 2; do
    train e9_sm1500_s$s artifacts/e9_mix_sm1500.pt 8000 8000 --seed $s && fly e9_sm1500_s$s
  done
  for s in 41 0 1 2 3 4; do
    train e5b_mid_s$s $FM 2000 1500 --seed $s
  done
  # E8 tail — wait for lane B's mixed dataset
  while [ ! -f artifacts/e8_fullmix40_dagger.pt ]; do sleep 60; done
  train e8_ctrl_s0    artifacts/e8_fullmix40.pt        8000 8000 --seed 0  && fly e8_ctrl_s0
  train e8_dagger_s0  artifacts/e8_fullmix40_dagger.pt 8000 8000 --seed 0  && fly e8_dagger_s0
  train e8_dagger_s1  artifacts/e8_fullmix40_dagger.pt 8000 8000 --seed 1  && fly e8_dagger_s1
  train e8_dagger_s41 artifacts/e8_fullmix40_dagger.pt 8000 8000 --seed 41 && fly e8_dagger_s41
  echo "[$(date +%H:%M)] LANE-A DONE"
}

lane_B() {
  if [ ! -f artifacts/dagger_s0.bin ]; then
    ( cd $RR && ROTOR_BLOB=assets/blog_fullmix_w2_s0.jblob \
        ./target/release/examples/dagger_collect ../../artifacts/dagger_s0.bin 2500 0 \
        > ../../artifacts/logs/dagger_collect.log 2>&1 )
  fi
  [ -f artifacts/e8_fullmix40_dagger.pt ] || $PY $SCRATCH/e8_mix.py > artifacts/logs/e8_mix.log 2>&1
  echo "[$(date +%H:%M)] LANE-B DONE"
}

lane_C() {
  local missing=""
  for arm in uncond cond mass; do
    [ -f artifacts/e34_$arm.pt ] || missing="$missing,$arm"
  done
  missing=${missing#,}
  if [ -n "$missing" ]; then
    $PY -m drone_jepa.train_pc --bin artifacts/racing_pc.bin --device cpu \
        --s1 8000 --s2 8000 --arms $missing --save-prefix artifacts/e34 \
        > artifacts/logs/e34.train.log 2>&1
  fi
  $PY $SCRATCH/e34_flytest.py 10 > artifacts/logs/e34.flytest.log 2>&1
  echo "[$(date +%H:%M)] LANE-C DONE"
}

lane_A > artifacts/logs/laneA.log 2>&1 &
lane_B > artifacts/logs/laneB.log 2>&1 &
lane_C > artifacts/logs/laneC.log 2>&1 &
wait
echo "[$(date +%H:%M)] MASTER DONE"
