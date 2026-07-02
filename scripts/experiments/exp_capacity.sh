#!/bin/zsh
# Capacity sweep: does a bigger JEPA fly the WIDE fleet? Train widths 1.25/1.5/2
# (14K/20K/34K) on the full wide data, export, and wide-fleet-test each.
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
RR=web-demo/rotor-rs
LOG=/tmp/exp_capacity.log
exec >> $LOG 2>&1
echo "=== capacity sweep started $(date +%H:%M) ==="
for W in 1.25 1.5 2; do
  tag="w${W//./}"  # w125 w15 w2
  echo "[$(date +%H:%M)] === TRAIN width $W ($tag) ==="
  $PY -m drone_jepa.train --data artifacts/racing_v2_ctbr.pt \
      --action-mode ctbr --pos-mode relative --width-mult $W \
      --stage1-steps 8000 --stage2-steps 8000 --batch 256 --device mps \
      --out artifacts/exp_$tag.pt 2>&1 | grep -E "model:|val_pos" | tail -2
  $PY scripts/export_jepa.py artifacts/exp_$tag.pt exp_$tag 2>&1 | grep -i "wrote.*safet" | head -1
  $PY scripts/export_jepa_blob.py exp_$tag 2>&1 | grep -i "wrote.*jblob"
  echo "--- fleet (wide) width $W ---"
  (cd $RR && JTEST=exp_$tag.jblob ./target/release/examples/fleet_fly 2>/dev/null \
      | grep -E "true-MPPI|RL-v2|JEPA-test")
done
echo "=== CAPACITY SWEEP DONE $(date +%H:%M) ==="
