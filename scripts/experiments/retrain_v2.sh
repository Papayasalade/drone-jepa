#!/bin/zsh
# Retrain JEPA on the dive/fast-recovery-augmented racing_v2 data.
# Waits for the gen job to finish, converts, then trains ctbr (relative pos) +
# rotor-force (zero pos) JEPA to STAGING names. Does NOT touch the live blob —
# we verify the floor-diving fix before deploying.
set -e
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
LOG=/tmp/retrain_v2.log
exec > >(tee -a $LOG) 2>&1
echo "=== retrain_v2 started $(date) ==="

# 1. wait for gen (racing_v2.bin stops growing + gen process gone)
while pgrep -f "gen_dataset 60000" >/dev/null; do sleep 30; done
echo "gen finished: $(ls -la artifacts/racing_v2.bin | awk '{print $5}') bytes"

# 2. convert dual-label bin -> racing_v2_ctbr.pt + racing_v2_rotor.pt
$PY scripts/convert_dataset.py artifacts/racing_v2.bin artifacts/racing_v2

# 3. train ctbr (relative pos — matches deployed JEPA drone)
$PY -m drone_jepa.train --data artifacts/racing_v2_ctbr.pt \
    --action-mode ctbr --pos-mode relative --prober-hidden 40 \
    --stage1-steps 8000 --stage2-steps 8000 --batch 256 --device mps \
    --out artifacts/skyjepa_ctbr_v2dive.pt

# 4. train rotor-force (relative pos — matches the deployed rotor model)
$PY -m drone_jepa.train --data artifacts/racing_v2_rotor.pt \
    --action-mode rotor_force --pos-mode relative --prober-hidden 40 \
    --stage1-steps 8000 --stage2-steps 8000 --batch 256 --device mps \
    --out artifacts/skyjepa_rotor_v2dive.pt

echo "=== retrain_v2 done $(date) — staged: skyjepa_ctbr_v2dive.pt, skyjepa_rotor_v2dive.pt ==="
echo "NOT YET DEPLOYED. Verify, then export_jepa + export_jepa_blob + rebuild wasm."
