#!/bin/zsh
# Orchestrate the JEPA-fleet experiments: export + wide/narrow fleet-test each model.
# MPS is serialized (narrow-train waits for train-longer to finish).
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
RR=web-demo/rotor-rs
FLEET=$RR/target/release/examples/fleet_fly
LOG=/tmp/exp_results.log
exec >> $LOG 2>&1
echo "=== exp_orchestrate started $(date +%H:%M) ==="

# 1. train-longer (wide, 16k/16k) -> export -> wide fleet test
while pgrep -f "out artifacts/exp_long.pt" >/dev/null; do sleep 20; done
echo "[$(date +%H:%M)] === EXP train-longer (wide 16k/16k) ==="
grep -E "\[s2  (14|15)[0-9]00" /tmp/exp_long.log | tail -1
$PY scripts/export_jepa.py artifacts/exp_long.pt exp_long 2>&1 | grep -i wrote | head -1
$PY scripts/export_jepa_blob.py exp_long 2>&1 | grep -i wrote
(cd $RR && JTEST=exp_long.jblob ./target/release/examples/fleet_fly 2>/dev/null | grep -E "over wide|true-MPPI|RL-v2|JEPA-test")

# 2. narrow gen -> (MPS now free) convert + train + test on MATCHING narrow fleet
while pgrep -f "gen_dataset 60000" >/dev/null; do sleep 20; done
echo "[$(date +%H:%M)] === EXP narrow (SYM, mass 0.4-0.9, arm 0.12-0.22) ==="
$PY scripts/convert_dataset.py artifacts/racing_narrow.bin artifacts/racing_narrow 2>&1 | grep -i ctbr | tail -1
$PY -m drone_jepa.train --data artifacts/racing_narrow_ctbr.pt \
    --action-mode ctbr --pos-mode relative --prober-hidden 40 \
    --stage1-steps 8000 --stage2-steps 8000 --batch 256 --device mps \
    --out artifacts/exp_narrow.pt 2>&1 | grep -E "val_pos" | tail -1
$PY scripts/export_jepa.py artifacts/exp_narrow.pt exp_narrow 2>&1 | grep -i wrote | head -1
$PY scripts/export_jepa_blob.py exp_narrow 2>&1 | grep -i wrote
echo "--- narrow JEPA on MATCHING narrow fleet (RL-v2/true also on narrow fleet) ---"
(cd $RR && SYM=1 ARM_LO=0.12 ARM_HI=0.22 MASS_LO=0.4 MASS_HI=0.9 \
    JTEST=exp_narrow.jblob ./target/release/examples/fleet_fly 2>/dev/null | grep -E "over wide|true-MPPI|RL-v2|JEPA-test")
echo "=== EXPERIMENTS DONE $(date +%H:%M) ==="
