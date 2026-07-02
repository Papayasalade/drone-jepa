#!/bin/zsh
# First production run of the adopted train_select workflow, on the two
# rotor-force recipes: full-mix w2 (lottery-prone) and the DAgger mixture.
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
mkdir -p artifacts/logs

$PY scripts/train_select.py --stem skyjepa_fullmix_w2 \
    --data artifacts/skyjepa_rotor_mix_recovery/dataset.pt \
    --n-seeds 4 --device mps -- \
    --action-mode rotor_force --pos-mode relative --width-mult 2 \
    --batch 256 --stride 5 --stage1-steps 8000 --stage2-steps 8000 \
    > artifacts/logs/select_fullmix.log 2>&1
echo "[$(date +%H:%M)] fullmix select done (rc=$?)"

$PY scripts/train_select.py --stem skyjepa_rf_dagger \
    --data artifacts/e8_fullmix40_dagger.pt \
    --n-seeds 4 --device mps -- \
    --action-mode rotor_force --pos-mode relative --width-mult 2 \
    --batch 256 --stride 5 --stage1-steps 8000 --stage2-steps 8000 \
    > artifacts/logs/select_dagger.log 2>&1
echo "[$(date +%H:%M)] dagger select done (rc=$?)"
echo "SELECT-ALL DONE"
