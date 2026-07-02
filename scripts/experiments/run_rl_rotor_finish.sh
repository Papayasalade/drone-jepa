#!/bin/zsh
# Finish the RL-rotor pipeline once PPO training lands: export the blob,
# validate natively on the bigquad (rl_fly RL_ROTOR=1), rebuild the WASM demo.
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
RR=web-demo/rotor-rs
export DRONE_MASS=1.2 DRONE_IXX=0.0189 DRONE_IYY=0.0191 DRONE_IZZ=0.0365 \
       DRONE_ARM=0.25 DRONE_K_ETA=3.92e-6 DRONE_K_M=9.6e-8 DRONE_TAU_M=0.02 DRONE_K_W=16

until [ -f artifacts/skyrl_rotor_bigquad.pt ]; do sleep 60; done
echo "[$(date +%H:%M)] export blob"
$PY scripts/export_rl_blob.py artifacts/skyrl_rotor_bigquad.pt skyrl_rotor_bigquad > artifacts/logs/skyrl_rotor.export.log 2>&1
ls $RR/assets/ | grep -i rotor_bigquad || { echo "blob export failed"; exit 1; }

echo "[$(date +%H:%M)] native validation (rl_fly rotor mode, bigquad)"
( cd $RR && RL_ROTOR=1 RL_BLOB=assets/skyrl_rotor_bigquad.rlb \
    ./target/release/examples/rl_fly > ../../artifacts/logs/rl_rotor_fly.log 2>&1 )
tail -1 artifacts/logs/rl_rotor_fly.log

echo "[$(date +%H:%M)] rebuild wasm"
( cd $RR && wasm-pack build --target web --out-dir ../web/pkg -- --features wasm \
    > ../../artifacts/logs/wasm_build.log 2>&1 )
tail -2 artifacts/logs/wasm_build.log
echo "[$(date +%H:%M)] RL-ROTOR PIPELINE DONE"
