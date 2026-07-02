#!/bin/zsh
# Bigquad round 2 (after the base run): DAgger refinement + a vehicle-difficulty
# reference. Waits for BIGQUAD DONE, then:
#   A. true-MPPI reference on a bigquad-pinned fleet (how hard is this vehicle?)
#   B. DAgger round: collect with the confirmed rf winner -> mix -> retrain
#      2 warm-started + 2 fresh seeds -> select (fair comparison with the
#      hummingbird's dagger-refined 12/12 model).
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
RR=web-demo/rotor-rs
export DRONE_MASS=1.2 DRONE_IXX=0.0189 DRONE_IYY=0.0191 DRONE_IZZ=0.0365 \
       DRONE_ARM=0.25 DRONE_K_ETA=3.92e-6 DRONE_K_M=9.6e-8 DRONE_TAU_M=0.02 DRONE_K_W=16

echo "[$(date +%H:%M)] B0. naive 1-step baseline (paper-claim benchmark reference)"
[ -f artifacts/bigquad_baseline_1step.pt ] || \
  $PY -m drone_jepa.train_baseline --data artifacts/bigquad_rf_data.pt --steps 8000 \
      --horizon 1 --device mps --out artifacts/bigquad_baseline_1step.pt \
      > artifacts/logs/bigquad_baseline1.train.log 2>&1
$PY -m drone_jepa.eval.openloop --jepa artifacts/bigquad_rf.pt \
    --baseline artifacts/bigquad_baseline_1step.pt --data artifacts/bigquad_rf_data.pt \
    --out-png artifacts/compounding_bigquad_naive.png \
    > artifacts/logs/bigquad_benchmark_naive.log 2>&1

echo "[$(date +%H:%M)] B1. DAgger collection with the rf winner"
( cd $RR && cargo build --release --features jepa --example dagger_collect > /dev/null 2>&1 )
[ -f artifacts/bigquad_dagger.bin ] || \
  ( cd $RR && ROTOR_BLOB=assets/bigquad_rf.jblob \
      ./target/release/examples/dagger_collect ../../artifacts/bigquad_dagger.bin 2500 0 \
      > ../../artifacts/logs/bigquad_dagger_collect.log 2>&1 )
[ -f artifacts/bigquad_rf_dagger_mix.pt ] || \
  $PY scripts/dagger_mix.py artifacts/bigquad_rf_data.pt artifacts/bigquad_dagger.bin \
      artifacts/bigquad_rf_dagger_mix.pt > artifacts/logs/bigquad_dagger_mix.log 2>&1

echo "[$(date +%H:%M)] B2. retrain on base+dagger: 2 warm-started seeds"
for s in 0 1; do
  [ -f artifacts/bigquad_rfd_warm$s.pt ] || \
    $PY -m drone_jepa.train --data artifacts/bigquad_rf_dagger_mix.pt \
        --action-mode rotor_force --pos-mode relative --batch 256 --stride 5 \
        --stage1-steps 8000 --stage2-steps 8000 --device mps --data-seed $s \
        --warm-start artifacts/bigquad_rf.pt \
        --out artifacts/bigquad_rfd_warm$s.pt > artifacts/logs/bigquad_rfd_warm$s.log 2>&1
done

echo "[$(date +%H:%M)] B3. select across warm-started + the original winner"
$PY - <<'EOF' > artifacts/logs/bigquad_round2_select.log 2>&1
import json, subprocess, sys, shutil
sys.path.insert(0, ".")
from drone_jepa.train_recipe import race, passes_gate
from drone_jepa.eval.probe import get_gt, probe_checkpoint
spec = json.load(open("drones/bigquad.json"))
gt = get_gt(drone=spec, action_mode="rotor_force", drone_name="bigquad")
cands = ["bigquad_rfd_warm0", "bigquad_rfd_warm1"]
best, best_f = None, None
for c in cands:
    r = probe_checkpoint(f"artifacts/{c}.pt", gt)
    f = race(c, spec, "rotor_force")
    print(f"{c}: probe={r['plan_rot']:.1f}deg race={f}")
    if passes_gate(f, "rotor_force") and (best_f is None or f["gates"] > best_f["gates"]
                                          or (f["gates"] == best_f["gates"] and f["wins"] > best_f["wins"])):
        best, best_f = c, f
if best:
    shutil.copy(f"artifacts/{best}.pt", "artifacts/bigquad_rf_v2.pt")
    print(f"ROUND2 WINNER {best} -> artifacts/bigquad_rf_v2.pt  {best_f}")
else:
    print("ROUND2: no dagger candidate beat the gate; keeping bigquad_rf.pt")
EOF
echo "[$(date +%H:%M)] BIGQUAD ROUND2 DONE"
