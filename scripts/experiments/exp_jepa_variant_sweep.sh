#!/usr/bin/env bash
set -euo pipefail

# Controlled rotor-force JEPA variants. This script assumes the mixed experiment
# has already produced:
#   artifacts/skyjepa_rotor_mix_recovery/smooth_se3_rotor.pt
#   artifacts/skyjepa_rotor_mix_recovery/recovery_se3_rotor.pt
#
# Default is a quick CPU sweep. Increase STAGE*_STEPS for full runs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_DATA="${BASE_DATA:-artifacts/dataset_rf_mppi.pt}"
ADDON_DIR="${ADDON_DIR:-artifacts/skyjepa_rotor_mix_recovery}"
SMOOTH_DATA="${SMOOTH_DATA:-${ADDON_DIR}/smooth_se3_rotor.pt}"
RECOVERY_DATA="${RECOVERY_DATA:-${ADDON_DIR}/recovery_se3_rotor.pt}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-256}"
STAGE1_STEPS="${STAGE1_STEPS:-2000}"
STAGE2_STEPS="${STAGE2_STEPS:-2000}"
STRIDE="${STRIDE:-5}"
WIDTH_MULT="${WIDTH_MULT:-1.0}"
SEED="${SEED:-501}"
VARIANTS="${VARIANTS:-base,recovery,smooth,mix}"
EXP_SUFFIX="${EXP_SUFFIX:-}"

want_variant() {
  local needle="$1"
  [[ ",${VARIANTS}," == *",${needle},"* ]]
}

run_variant() {
  local exp="$1"
  local init_from="$2"
  shift 2
  local out_dir="artifacts/${exp}"
  local mix_data="${out_dir}/dataset.pt"
  local ckpt="artifacts/${exp}.pt"
  mkdir -p "$out_dir"

  echo
  echo "========================================================================"
  echo "== $exp"
  echo "========================================================================"

  python scripts/concat_datasets.py "$mix_data" "$@"
  python scripts/dataset_coverage.py "$mix_data"

  if [[ "$init_from" != "-" ]]; then
    python -m drone_jepa.train \
      --data "$mix_data" \
      --action-mode rotor_force \
      --pos-mode relative \
      --width-mult "$WIDTH_MULT" \
      --stage1-steps "$STAGE1_STEPS" \
      --stage2-steps "$STAGE2_STEPS" \
      --batch "$BATCH" \
      --stride "$STRIDE" \
      --device "$DEVICE" \
      --out "$ckpt" \
      --init-from "$init_from"
  else
    python -m drone_jepa.train \
      --data "$mix_data" \
      --action-mode rotor_force \
      --pos-mode relative \
      --width-mult "$WIDTH_MULT" \
      --stage1-steps "$STAGE1_STEPS" \
      --stage2-steps "$STAGE2_STEPS" \
      --batch "$BATCH" \
      --stride "$STRIDE" \
      --device "$DEVICE" \
      --out "$ckpt"
  fi

  python scripts/export_jepa.py "$ckpt" "$exp"
  python scripts/export_jepa_blob.py "$exp"

  ROTOR_BLOB="web-demo/racer/assets/${exp}.jblob" cargo run --release --features jepa \
    --manifest-path web-demo/racer/Cargo.toml \
    --example rotor_fly | tee "${out_dir}/rotor_fly.log"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing $1"
    exit 2
  fi
}

require_file "$BASE_DATA"
require_file "$SMOOTH_DATA"
require_file "$RECOVERY_DATA"

mkdir -p artifacts/variant_inputs
python scripts/subsample_dataset.py "$SMOOTH_DATA" artifacts/variant_inputs/smooth_500.pt 500 "$((SEED + 1))"
python scripts/subsample_dataset.py "$SMOOTH_DATA" artifacts/variant_inputs/smooth_1000.pt 1000 "$((SEED + 2))"
python scripts/subsample_dataset.py "$RECOVERY_DATA" artifacts/variant_inputs/recovery_200.pt 200 "$((SEED + 3))"
python scripts/subsample_dataset.py "$RECOVERY_DATA" artifacts/variant_inputs/recovery_500.pt 500 "$((SEED + 4))"

if want_variant base; then
  run_variant "skyjepa_rotor_var_base_relative${EXP_SUFFIX}" - \
    "$BASE_DATA"
fi

if want_variant recovery; then
  run_variant "skyjepa_rotor_var_recovery_relative_small${EXP_SUFFIX}" - \
    "$BASE_DATA" artifacts/variant_inputs/recovery_200.pt
fi

if want_variant smooth; then
  run_variant "skyjepa_rotor_var_smooth_relative_small${EXP_SUFFIX}" - \
    "$BASE_DATA" artifacts/variant_inputs/smooth_500.pt
fi

if want_variant mix; then
  run_variant "skyjepa_rotor_var_mix_relative_small${EXP_SUFFIX}" - \
    "$BASE_DATA" artifacts/variant_inputs/smooth_500.pt artifacts/variant_inputs/recovery_200.pt
fi

if want_variant init; then
  run_variant "skyjepa_rotor_var_mix_relative_init_existing${EXP_SUFFIX}" artifacts/skyjepa_rotor_v2clean.pt \
    "$BASE_DATA" artifacts/variant_inputs/smooth_500.pt artifacts/variant_inputs/recovery_200.pt
fi

echo
echo "Variant logs:"
find artifacts -path '*/rotor_fly.log' -name rotor_fly.log | sort | rg 'skyjepa_rotor_var_' || true
python scripts/summarize_rotor_fly.py artifacts/skyjepa_rotor_var_*/rotor_fly.log
