#!/usr/bin/env bash
set -euo pipefail

# Mixed rotor-force JEPA experiment:
#   existing rotor-force dataset + smooth SE3 tracking + realistic SE3 recovery.
#
# Defaults assume artifacts/dataset_rf_mppi.pt already exists. Override env vars
# for a larger/smaller run:
#   DEVICE=mps SMOOTH_N=3000 RECOVERY_N=1000 scripts/exp_mixed_recovery_jepa.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXP="${EXP:-skyjepa_rotor_mix_recovery}"
BASE_DATA="${BASE_DATA:-artifacts/dataset_rf_mppi.pt}"
STEPS="${STEPS:-200}"
SMOOTH_N="${SMOOTH_N:-2000}"
RECOVERY_N="${RECOVERY_N:-800}"
SEED="${SEED:-41}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-256}"
STAGE1_STEPS="${STAGE1_STEPS:-8000}"
STAGE2_STEPS="${STAGE2_STEPS:-8000}"
STRIDE="${STRIDE:-5}"
WIDTH_MULT="${WIDTH_MULT:-1.0}"

MIX_DIR="artifacts/${EXP}"
SMOOTH_BIN="${MIX_DIR}/smooth_se3.bin"
RECOVERY_BIN="${MIX_DIR}/recovery_se3.bin"
SMOOTH_STEM="${MIX_DIR}/smooth_se3"
RECOVERY_STEM="${MIX_DIR}/recovery_se3"
MIX_DATA="${MIX_DIR}/dataset.pt"
CKPT="artifacts/${EXP}.pt"

mkdir -p "$MIX_DIR"

if [[ ! -f "$BASE_DATA" ]]; then
  echo "missing BASE_DATA=$BASE_DATA"
  echo "Set BASE_DATA to an existing rotor-force .pt dataset, or build artifacts/dataset_rf_mppi.pt first."
  exit 2
fi

echo "== generating smooth SE3 add-on =="
cargo run --release --manifest-path web-demo/racer/Cargo.toml \
  --example gen_dataset_se3 -- "$SMOOTH_N" "$STEPS" "$SMOOTH_BIN" "$SEED" fourier smooth
python scripts/convert_dataset.py "$SMOOTH_BIN" "$SMOOTH_STEM"

echo "== generating realistic recovery SE3 add-on =="
cargo run --release --manifest-path web-demo/racer/Cargo.toml \
  --example gen_dataset_se3 -- "$RECOVERY_N" "$STEPS" "$RECOVERY_BIN" "$((SEED + 1))" gp recovery
python scripts/convert_dataset.py "$RECOVERY_BIN" "$RECOVERY_STEM"

echo "== mixing datasets =="
python scripts/concat_datasets.py "$MIX_DATA" \
  "$BASE_DATA" \
  "${SMOOTH_STEM}_rotor.pt" \
  "${RECOVERY_STEM}_rotor.pt"

echo "== coverage summary =="
python scripts/dataset_coverage.py "$BASE_DATA" "$MIX_DATA"

echo "== training $EXP =="
python -m drone_jepa.train \
  --data "$MIX_DATA" \
  --action-mode rotor_force \
  --pos-mode relative \
  --width-mult "$WIDTH_MULT" \
  --stage1-steps "$STAGE1_STEPS" \
  --stage2-steps "$STAGE2_STEPS" \
  --batch "$BATCH" \
  --stride "$STRIDE" \
  --device "$DEVICE" \
  --out "$CKPT"

echo "== exporting Rust/WASM checkpoint =="
python scripts/export_jepa.py "$CKPT" "$EXP"
python scripts/export_jepa_blob.py "$EXP"

if [[ "${RUN_RACE_EVAL:-0}" == "1" ]]; then
  echo "== rotor-force race sanity =="
  ROTOR_BLOB="web-demo/racer/assets/${EXP}.jblob" cargo run --release --features jepa \
    --manifest-path web-demo/racer/Cargo.toml \
    --example rotor_fly | tee "${MIX_DIR}/rotor_fly.log"
fi

echo "done:"
echo "  dataset: $MIX_DATA"
echo "  checkpoint: $CKPT"
echo "  wasm blob: web-demo/racer/assets/${EXP}.jblob"
