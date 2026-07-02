#!/bin/zsh
# Forecast-demo model for the bigquad: the demo drives with the SE3 controller,
# so the model needs SE3-style actions in training. base(chopped) + dagger +
# 1000 smooth SE3 (E9-safe volume), warm-started from bigquad_rf_v2, one race
# gate, then swap into the wasm bundle.
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
RR=web-demo/rotor-rs
export DRONE_MASS=1.2 DRONE_IXX=0.0189 DRONE_IYY=0.0191 DRONE_IZZ=0.0365 \
       DRONE_ARM=0.25 DRONE_K_ETA=3.92e-6 DRONE_K_M=9.6e-8 DRONE_TAU_M=0.02 DRONE_K_W=16

echo "[$(date +%H:%M)] smooth SE3 add-on (1000 traj)"
( cd $RR && cargo build --release --features jepa --example gen_dataset_se3 > /dev/null 2>&1 )
[ -f artifacts/bigquad_se3_rotor.pt ] || {
  $RR/target/release/examples/gen_dataset_se3 1000 200 artifacts/bigquad_se3.bin 21 fourier smooth \
      > artifacts/logs/bigquad_se3_gen.log 2>&1
  $PY scripts/convert_dataset.py artifacts/bigquad_se3.bin artifacts/bigquad_se3 \
      >> artifacts/logs/bigquad_se3_gen.log 2>&1
}

echo "[$(date +%H:%M)] mix base40+dagger+se3"
[ -f artifacts/bigquad_forecast_mix.pt ] || $PY - <<'EOF' > artifacts/logs/bigquad_forecast_mix.log 2>&1
import torch, sys
sys.path.insert(0, ".")
base = torch.load("artifacts/bigquad_rf_dagger_mix.pt", weights_only=False)  # base40 + dagger
se3 = torch.load("artifacts/bigquad_se3_rotor.pt", weights_only=False)
S3, A3, d3 = se3["states"], se3["actions"], se3["domain"]
k = S3.shape[1] // 40
S3 = S3.reshape(-1, 40, 18); A3 = A3.reshape(-1, 40, 4)
d3 = (d3.repeat_interleave(k) + int(base["domain"].max()) + 1000)
out = {"states": torch.cat([base["states"], S3]), "actions": torch.cat([base["actions"], A3]),
       "domain": torch.cat([base["domain"], d3]), "dt": base["dt"]}
torch.save(out, "artifacts/bigquad_forecast_mix.pt")
print("forecast mix:", out["states"].shape)
EOF

echo "[$(date +%H:%M)] warm retrain from bigquad_rf_v2"
[ -f artifacts/bigquad_forecast.pt ] || \
  $PY -m drone_jepa.train --data artifacts/bigquad_forecast_mix.pt \
      --action-mode rotor_force --pos-mode relative --batch 256 --stride 5 \
      --stage1-steps 8000 --stage2-steps 8000 --device mps \
      --warm-start artifacts/bigquad_rf_v2.pt \
      --out artifacts/bigquad_forecast.pt > artifacts/logs/bigquad_forecast.train.log 2>&1

echo "[$(date +%H:%M)] gate (race must not regress) + probe"
$PY scripts/export_jepa.py artifacts/bigquad_forecast.pt bigquad_forecast > /dev/null 2>&1
$PY scripts/export_jepa_blob.py bigquad_forecast > /dev/null 2>&1
( cd $RR && ROTOR_BLOB=assets/bigquad_forecast.jblob ./target/release/examples/rotor_fly \
    > ../../artifacts/logs/bigquad_forecast.race.log 2>&1 )
tail -1 artifacts/logs/bigquad_forecast.race.log
$PY -m drone_jepa.eval.probe --drone drones/bigquad.json bigquad_forecast 2>/dev/null | tail -1
echo "[$(date +%H:%M)] FORECAST MODEL DONE"
