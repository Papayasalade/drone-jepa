"""Default training workflow (fragility-campaign playbook, adopted 2026-07-02):

  1. train N seeds of the same recipe,
  2. rank them with the deployment-action plan probe (drone_jepa.eval.probe),
  3. CONFIRM candidates with one rotor_fly each (best-ranked first; the probe
     misses ~4/21 milder crash modes, so a flight gate is mandatory),
  4. save the confirmed winner as artifacts/<stem>.pt — the canonical
     checkpoint to deploy AND to `--warm-start` all future retrains from
     (basins are heritable under continued training).

Usage (extra args go straight to drone_jepa.train):

  .venv/bin/python scripts/train_select.py --stem skyjepa_rf \\
      --data artifacts/dataset_rf_mppi.pt --n-seeds 4 --device mps -- \\
      --action-mode rotor_force --pos-mode relative --width-mult 2 \\
      --stage1-steps 8000 --stage2-steps 8000

Flight gate: 0 hard crashes AND <5% steps flipped AND >=4 gates/race (the last
rejects "timid" basins that hover safely but don't race). rotor_fly is
rotor-force only — for other action modes pass --no-fly and gate manually.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "web-demo" / "racer"
SUMMARY = re.compile(
    r"== (?P<wins>\d+)/(?P<trials>\d+) WON, (?P<crashes>\d+) hard-crashed, "
    r"(?P<respawns>\d+) respawns, (?P<gates>[0-9.]+) gates/race, "
    r"(?P<flipped>[0-9.]+)% steps flipped ==")


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, **kw)


def fly(name: str) -> dict | None:
    sh([sys.executable, "scripts/export_jepa.py", f"artifacts/{name}.pt", name],
       capture_output=True)
    sh([sys.executable, "scripts/export_jepa_blob.py", name], capture_output=True)
    out = subprocess.run([str(RR / "target/release/examples/rotor_fly")],
                         cwd=RR, capture_output=True, text=True,
                         env={"ROTOR_BLOB": f"assets/{name}.jblob", "PATH": "/usr/bin:/bin"})
    m = SUMMARY.search(out.stdout)
    return {k: float(v) for k, v in m.groupdict().items()} if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="canonical output artifacts/<stem>.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-fly", action="store_true",
                    help="skip the flight gate (non-rotor-force models)")
    ap.add_argument("train_args", nargs="*",
                    help="passed through to drone_jepa.train (after --)")
    args = ap.parse_args()

    from drone_jepa.eval.probe import FLY_THRESHOLD_DEG, get_gt, probe_checkpoint

    names = []
    for s in range(args.n_seeds):
        name = f"{args.stem}_seed{s}"
        names.append(name)
        if (ROOT / f"artifacts/{name}.pt").exists():
            print(f"[skip] {name} exists")
            continue
        print(f"[train] {name}", flush=True)
        r = sh([sys.executable, "-m", "drone_jepa.train", "--data", args.data,
                "--device", args.device, "--seed", str(s),
                "--out", f"artifacts/{name}.pt", *args.train_args],
               capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"training failed for {name}")

    gt = get_gt()
    scored = []
    for name in names:
        r = probe_checkpoint(f"artifacts/{name}.pt", gt)
        scored.append((r["plan_rot"], name, r))
        print(f"[probe] {name}: plan_rot={r['plan_rot']:.1f} inv={r['inv']*100:.1f}% "
              f"rank={r['rank']:+.2f}", flush=True)
    scored.sort()

    winner = None
    for prot, name, r in scored:
        if prot > FLY_THRESHOLD_DEG:
            print(f"[gate] {name} has the crash signature ({prot:.1f} deg) — skipping")
            continue
        if args.no_fly:
            winner = name
            break
        print(f"[fly] confirming {name} ...", flush=True)
        f = fly(name)
        print(f"[fly] {name}: {f}", flush=True)
        if f and f["crashes"] == 0 and f["flipped"] < 5.0 and f["gates"] >= 4.0:
            winner = name
            break
        print(f"[gate] {name} failed the flight gate — trying next candidate")

    if winner is None:
        raise SystemExit("NO candidate passed both gates — train more seeds, or "
                         "warm-start from a known flyer (see basin playbook).")
    shutil.copy(ROOT / f"artifacts/{winner}.pt", ROOT / f"artifacts/{args.stem}.pt")
    print(f"\nWINNER {winner} -> artifacts/{args.stem}.pt")
    print(f"Deploy this, and warm-start future retrains from it:\n"
          f"  python -m drone_jepa.train --warm-start artifacts/{args.stem}.pt ...")


if __name__ == "__main__":
    main()
