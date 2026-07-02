#!/bin/zsh
# Fragility-campaign training queue (E5a warm-start, E5b early-probe shorts,
# E6 EMA-target, E9 smooth-volume sweep). MPS is serialized; rotor_fly runs
# in the background on CPU after each export.
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
FM=artifacts/skyjepa_rotor_mix_recovery/dataset.pt
RR=web-demo/racer
mkdir -p artifacts/logs
COMMON=(--action-mode rotor_force --pos-mode relative --width-mult 2 --batch 256 --stride 5 --device mps)

train() { # name data steps1 steps2 extra...
  local name=$1 data=$2 s1=$3 s2=$4; shift 4
  echo "[$(date +%H:%M)] TRAIN $name"
  $PY -m drone_jepa.train --data $data $COMMON --stage1-steps $s1 --stage2-steps $s2 \
      --out artifacts/$name.pt "$@" > artifacts/logs/$name.train.log 2>&1
}

fly() { # name  (export + rotor_fly in background)
  local name=$1
  $PY scripts/export_jepa.py artifacts/$name.pt $name > /dev/null 2>&1
  $PY scripts/export_jepa_blob.py $name > /dev/null 2>&1
  ( cd $RR && ROTOR_BLOB=assets/$name.jblob ./target/release/examples/rotor_fly \
      > ../../artifacts/logs/$name.rotor_fly.log 2>/dev/null ) &
  echo "[$(date +%H:%M)] FLY-> $name (bg)"
}

# ---- E5b: early-probe shorts (600 s1 steps == the full run's own step 600) ----
for s in 41 0 1 2 3 4; do
  train e5b_early_s$s $FM 600 1500 --seed $s
done

# ---- E5a: warm-start basin-survival ----
train e5a_warmfly_d1  $FM 8000 8000 --warm-start artifacts/blog_fullmix_w2_s2.pt --data-seed 1 && fly e5a_warmfly_d1
train e5a_warmfly_d2  $FM 8000 8000 --warm-start artifacts/blog_fullmix_w2_s2.pt --data-seed 2 && fly e5a_warmfly_d2
train e5a_warmcrash_d1 $FM 8000 8000 --warm-start artifacts/blog_sep_i0_d0_w2.pt --data-seed 1 && fly e5a_warmcrash_d1

# ---- E6: EMA-target arm, seeds matched to the labeled default arm ----
for s in 41 0 1 2 3 4; do
  train e6_ema_s$s $FM 8000 8000 --seed $s --ema-tau 0.99 && fly e6_ema_s$s
done

# ---- E9: smooth-volume sweep at rec=800 (sm2000 == full mix, already labeled) ----
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
