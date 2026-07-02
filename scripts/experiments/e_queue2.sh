#!/bin/zsh
# Follow-up queue: E8 DAgger retrains + E5b mid-training shorts (step-2000 probe).
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
FM=artifacts/skyjepa_rotor_mix_recovery/dataset.pt
RR=web-demo/racer
SCRATCH=artifacts
COMMON=(--action-mode rotor_force --pos-mode relative --width-mult 2 --batch 256 --stride 5 --device mps)

train() {
  local name=$1 data=$2 s1=$3 s2=$4; shift 4
  echo "[$(date +%H:%M)] TRAIN $name"
  $PY -m drone_jepa.train --data $data $COMMON --stage1-steps $s1 --stage2-steps $s2 \
      --out artifacts/$name.pt "$@" > artifacts/logs/$name.train.log 2>&1
}
fly() {
  local name=$1
  $PY scripts/export_jepa.py artifacts/$name.pt $name > /dev/null 2>&1
  $PY scripts/export_jepa_blob.py $name > /dev/null 2>&1
  ( cd $RR && ROTOR_BLOB=assets/$name.jblob ./target/release/examples/rotor_fly \
      > ../../artifacts/logs/$name.rotor_fly.log 2>/dev/null ) &
  echo "[$(date +%H:%M)] FLY-> $name (bg)"
}

# ---- E5b: mid-training shorts (== full run's own step 2000) ----
for s in 41 0 1 2 3 4; do
  train e5b_mid_s$s $FM 2000 1500 --seed $s
done

# ---- E8: DAgger mix + retrains ----
$PY $SCRATCH/e8_mix.py > artifacts/logs/e8_mix.log 2>&1
train e8_ctrl_s0   artifacts/e8_fullmix40.pt        8000 8000 --seed 0  && fly e8_ctrl_s0
train e8_dagger_s0 artifacts/e8_fullmix40_dagger.pt 8000 8000 --seed 0  && fly e8_dagger_s0
train e8_dagger_s1 artifacts/e8_fullmix40_dagger.pt 8000 8000 --seed 1  && fly e8_dagger_s1
train e8_dagger_s41 artifacts/e8_fullmix40_dagger.pt 8000 8000 --seed 41 && fly e8_dagger_s41

wait
echo "[$(date +%H:%M)] QUEUE2 DONE"
