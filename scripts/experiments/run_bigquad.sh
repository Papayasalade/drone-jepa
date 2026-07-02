#!/bin/zsh
# From-scratch benchmark on a NEW drone (drones/bigquad.json: 1.2 kg, 25 cm
# arms, TWR 3, slow motors). No reuse of any previously-trained model:
# fresh data (both action modes), fresh AR baseline, fresh recipe seeds.
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
RR=web-demo/racer
mkdir -p artifacts/logs
export DRONE_MASS=1.2 DRONE_IXX=0.0189 DRONE_IYY=0.0191 DRONE_IZZ=0.0365 \
       DRONE_ARM=0.25 DRONE_K_ETA=3.92e-6 DRONE_K_M=9.6e-8 DRONE_TAU_M=0.02 \
       DRONE_TAU_M_LO=0.015 DRONE_TAU_M_HI=0.03 DRONE_K_W=16

echo "[$(date +%H:%M)] 1/6 rotor-force data (rf-native MPPI expert)"
[ -f artifacts/bigquad_rf.pt ] || {
  $RR/target/release/examples/gen_dataset_rf 8000 200 artifacts/bigquad_rf.bin 7 \
      > artifacts/logs/bigquad_gen_rf.log 2>&1
  $PY scripts/convert_dataset.py artifacts/bigquad_rf.bin artifacts/bigquad_rf \
      >> artifacts/logs/bigquad_gen_rf.log 2>&1
}

echo "[$(date +%H:%M)] 2/6 dual-action data (ctbr MPPI expert, narrow band around bigquad)"
[ -f artifacts/bigquad_dual_ctbr.pt ] || {
  MASS_LO=0.9 MASS_HI=1.5 SYM=1 ARM_LO=0.2 ARM_HI=0.3 \
      $RR/target/release/examples/gen_dataset 8000 200 artifacts/bigquad_dual.bin 7 \
      > artifacts/logs/bigquad_gen_dual.log 2>&1
  $PY scripts/convert_dataset.py artifacts/bigquad_dual.bin artifacts/bigquad_dual \
      >> artifacts/logs/bigquad_gen_dual.log 2>&1
}

echo "[$(date +%H:%M)] 3/6 fresh AR baseline (benchmark reference)"
[ -f artifacts/bigquad_baseline.pt ] || \
  $PY -m drone_jepa.train_baseline --data artifacts/bigquad_rf.pt --steps 8000 \
      --device mps --out artifacts/bigquad_baseline.pt \
      > artifacts/logs/bigquad_baseline.train.log 2>&1

echo "[$(date +%H:%M)] 4/6 recipe: rotor-force (4 seeds -> probe -> race gate)"
$PY -m drone_jepa.train_recipe --data artifacts/bigquad_rf.pt --stem bigquad_rf \
    --drone drones/bigquad.json --n-seeds 4 --device mps \
    > artifacts/logs/bigquad_recipe_rf.log 2>&1
echo "  rf recipe rc=$?"

echo "[$(date +%H:%M)] 5/6 recipe: ctbr (4 seeds -> probe rank -> jepa_fly gate)"
$PY -m drone_jepa.train_recipe --data artifacts/bigquad_dual_ctbr.pt --stem bigquad_ctbr \
    --drone drones/bigquad.json --action-mode ctbr --n-seeds 4 --device mps \
    > artifacts/logs/bigquad_recipe_ctbr.log 2>&1
echo "  ctbr recipe rc=$?"

echo "[$(date +%H:%M)] 6/6 benchmark + race sims"
$PY -m drone_jepa.eval.openloop --jepa artifacts/bigquad_rf.pt \
    --baseline artifacts/bigquad_baseline.pt --data artifacts/bigquad_rf.pt \
    --out-png artifacts/compounding_bigquad.png \
    > artifacts/logs/bigquad_benchmark.log 2>&1
$PY -m drone_jepa.eval.probe --drone drones/bigquad.json bigquad_rf \
    >> artifacts/logs/bigquad_benchmark.log 2>&1
$PY -m drone_jepa.eval.probe --drone drones/bigquad.json --mode ctbr bigquad_ctbr \
    >> artifacts/logs/bigquad_benchmark.log 2>&1
( cd $RR && ROTOR_BLOB=assets/bigquad_rf.jblob ./target/release/examples/rotor_fly \
    > ../../artifacts/logs/bigquad_rf.race.log 2>&1 )
( cd $RR && JBLOB=assets/bigquad_ctbr.jblob ./target/release/examples/jepa_fly \
    > ../../artifacts/logs/bigquad_ctbr.race.log 2>&1 )
echo "[$(date +%H:%M)] BIGQUAD DONE"
