#!/bin/zsh
# ONE unified JEPA for both demos (race + follow/forecast): candidates trained
# on the union mix (base + dagger + SE3), judged on BOTH roles:
#   race gate  = rotor_fly (bigquad)          [deployment: race demo]
#   follow fit = open-loop pos RMSE on a FRESH SE3-driven test set
#                                              [deployment: forecast demo]
set -u
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
RR=web-demo/racer
export DRONE_MASS=1.2 DRONE_IXX=0.0189 DRONE_IYY=0.0191 DRONE_IZZ=0.0365 \
       DRONE_ARM=0.25 DRONE_K_ETA=3.92e-6 DRONE_K_M=9.6e-8 DRONE_TAU_M=0.02 DRONE_K_W=16

echo "[$(date +%H:%M)] fresh SE3 test set (never used in training)"
[ -f artifacts/bigquad_se3_test_rotor.pt ] || {
  $RR/target/release/examples/gen_dataset_se3 400 200 artifacts/bigquad_se3_test.bin 99 fourier smooth \
      > artifacts/logs/bigquad_se3_test.log 2>&1
  $PY scripts/convert_dataset.py artifacts/bigquad_se3_test.bin artifacts/bigquad_se3_test \
      >> artifacts/logs/bigquad_se3_test.log 2>&1
}

echo "[$(date +%H:%M)] 2 more unified candidates (warm from v2, fresh data order)"
for s in 1 2; do
  [ -f artifacts/bigquad_uni_d$s.pt ] || \
    $PY -m drone_jepa.train --data artifacts/bigquad_forecast_mix.pt \
        --action-mode rotor_force --pos-mode relative --batch 256 --stride 5 \
        --stage1-steps 8000 --stage2-steps 8000 --device mps --data-seed $s \
        --warm-start artifacts/bigquad_rf_v2.pt \
        --out artifacts/bigquad_uni_d$s.pt > artifacts/logs/bigquad_uni_d$s.log 2>&1
done

echo "[$(date +%H:%M)] dual-role evaluation"
$PY - <<'EOF' > artifacts/logs/bigquad_unified_select.log 2>&1
import json, shutil, subprocess, sys
import torch
sys.path.insert(0, ".")
from drone_jepa.train_recipe import race, passes_gate
from drone_jepa.eval.probe import get_gt, probe_checkpoint
from drone_jepa.eval.openloop import _long_windows
from drone_jepa.model.jepa import SkyJEPA

spec = json.load(open("drones/bigquad.json"))
gt = get_gt(drone=spec, action_mode="rotor_force", drone_name="bigquad")

d = torch.load("artifacts/bigquad_se3_test_rotor.pt", weights_only=False)
Xw, Aw = _long_windows(d["states"], d["actions"], 10, 40, stride=20)
i = torch.randperm(Xw.shape[0], generator=torch.Generator().manual_seed(0))[:512]
Xw, Aw = Xw[i], Aw[i]

def se3_err(path):
    m, _ = SkyJEPA.from_checkpoint(path); m.eval()
    with torch.no_grad():
        p = m.predict(Xw[:, :10], Aw, horizon=40)
    e20 = torch.linalg.norm(p[:, 19, :3] - Xw[:, 29, :3], dim=-1).mean().item()
    e40 = torch.linalg.norm(p[:, 39, :3] - Xw[:, 49, :3], dim=-1).mean().item()
    return e20, e40

cands = ["bigquad_rf_v2", "bigquad_forecast", "bigquad_uni_d1", "bigquad_uni_d2"]
rows = []
for c in cands:
    e20, e40 = se3_err(f"artifacts/{c}.pt")
    pr = probe_checkpoint(f"artifacts/{c}.pt", gt)
    f = race(c, spec, "rotor_force")
    ok = passes_gate(f, "rotor_force")
    rows.append((c, e20, e40, pr["plan_rot"], f, ok))
    print(f"{c}: se3@1s={e20:.3f}m se3@2s={e40:.3f}m probe={pr['plan_rot']:.1f} race={f} gate={'PASS' if ok else 'fail'}")

# among gate-passers, pick the lowest follow-error (the race gate already
# guarantees racing quality; the follow role is the differentiator)
passers = [r for r in rows if r[5]]
if not passers:
    print("NO unified candidate passes the race gate; keeping split models")
else:
    best = min(passers, key=lambda r: r[1])
    shutil.copy(f"artifacts/{best[0]}.pt", "artifacts/bigquad_unified.pt")
    print(f"UNIFIED WINNER {best[0]} -> artifacts/bigquad_unified.pt")
    subprocess.run([sys.executable, "scripts/export_jepa.py",
                    "artifacts/bigquad_unified.pt", "bigquad_unified"], capture_output=True)
    subprocess.run([sys.executable, "scripts/export_jepa_blob.py", "bigquad_unified"],
                   capture_output=True)
EOF
tail -8 artifacts/logs/bigquad_unified_select.log
echo "[$(date +%H:%M)] UNIFIED DONE"
