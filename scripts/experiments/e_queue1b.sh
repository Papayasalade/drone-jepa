#!/bin/zsh
# Restart of the remaining queue after interruption: e5a_warmcrash, E6 EMA arm,
# E9 smooth sweep. Same conventions as e_queue.sh.
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
FM=artifacts/skyjepa_rotor_mix_recovery/dataset.pt
RR=web-demo/rotor-rs
mkdir -p artifacts/logs
COMMON=(--action-mode rotor_force --pos-mode relative --width-mult 2 --batch 256 --stride 5 --device mps)

train() {
  local name=$1 data=$2 s1=$3 s2=$4; shift 4
  [ -f artifacts/$name.pt ] && { echo "[skip] $name"; return 0; }
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

wait
echo "[$(date +%H:%M)] QUEUE DONE"
