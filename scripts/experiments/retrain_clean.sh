#!/bin/zsh
# Clean retrain: NEW flyable drone distribution, NO augmentation (expert flight only).
# Isolates distribution from the (harmful) dive/stall/tilt augmentation. Trains ctbr
# (relative pos) + rotor-force (zero pos) to v2clean staging names. Does NOT deploy.
set -e
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
LOG=/tmp/retrain_clean.log
exec > >(tee -a $LOG) 2>&1
echo "=== retrain_clean started $(date) ==="

while pgrep -f "gen_dataset 60000" >/dev/null; do sleep 30; done
echo "gen finished: $(ls -la artifacts/racing_v2.bin | awk '{print $5}') bytes"

$PY scripts/convert_dataset.py artifacts/racing_v2.bin artifacts/racing_v2

$PY -m drone_jepa.train --data artifacts/racing_v2_ctbr.pt \
    --action-mode ctbr --pos-mode relative --prober-hidden 40 \
    --stage1-steps 8000 --stage2-steps 8000 --batch 256 --device mps \
    --out artifacts/skyjepa_ctbr_v2clean.pt

$PY -m drone_jepa.train --data artifacts/racing_v2_rotor.pt \
    --action-mode rotor_force --pos-mode relative --prober-hidden 40 \
    --stage1-steps 8000 --stage2-steps 8000 --batch 256 --device mps \
    --out artifacts/skyjepa_rotor_v2clean.pt

echo "=== retrain_clean done $(date) — staged: skyjepa_ctbr_v2clean.pt, skyjepa_rotor_v2clean.pt ==="
