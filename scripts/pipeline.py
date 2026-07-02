"""ONE command for the whole adopted workflow: train -> select -> eval -> deploy.

    .venv/bin/python scripts/pipeline.py --stem skyjepa_rf \\
        --data artifacts/dataset_rf_mppi.pt [--n-seeds 4] [--device mps] \\
        [--baseline artifacts/baseline.pt] [-- extra drone_jepa.train args]

Steps (each idempotent, artifacts under artifacts/):
  1. SELECT  scripts/train_select.py — train N seeds, probe-rank
             (drone_jepa.eval.probe), flight-gate candidates (0 hard crashes,
             <5% flipped, >=4 gates/race), save winner as artifacts/<stem>.pt.
  2. EVAL    drone_jepa.eval.openloop --probe on the winner (RMSE table + the
             flight-predicting deployment-action probe).
  3. DEPLOY  export_jepa.py + export_jepa_blob.py — WASM/Rust-ready
             web-demo/racer/assets/<stem>.jblob.

Warm-start reminder: future retrains should start from the confirmed winner:
    python -m drone_jepa.train --warm-start artifacts/<stem>.pt ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(step: str, cmd: list[str]) -> None:
    print(f"\n=== [{step}] {' '.join(cmd)} ===", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[{step}] failed (rc={r.returncode}) — pipeline stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--baseline", default="artifacts/baseline.pt",
                    help="baseline ckpt for the eval step (skipped if missing)")
    ap.add_argument("--no-fly", action="store_true",
                    help="skip flight gating (non-rotor-force models)")
    ap.add_argument("train_args", nargs="*",
                    help="passed through to drone_jepa.train (after --)")
    args = ap.parse_args()
    py = sys.executable

    sel = [py, "scripts/train_select.py", "--stem", args.stem, "--data", args.data,
           "--n-seeds", str(args.n_seeds), "--device", args.device]
    if args.no_fly:
        sel.append("--no-fly")
    run("SELECT", sel + ["--", *args.train_args])

    if (ROOT / args.baseline).exists():
        run("EVAL", [py, "-m", "drone_jepa.eval.openloop",
                     "--jepa", f"artifacts/{args.stem}.pt",
                     "--baseline", args.baseline, "--data", args.data,
                     "--out-png", f"artifacts/compounding_{args.stem}.png",
                     "--probe"])
    else:
        print(f"[EVAL] baseline {args.baseline} missing — skipping eval step")

    run("DEPLOY", [py, "scripts/export_jepa.py", f"artifacts/{args.stem}.pt", args.stem])
    run("DEPLOY", [py, "scripts/export_jepa_blob.py", args.stem])

    print(f"\nPIPELINE DONE: artifacts/{args.stem}.pt  +  "
          f"web-demo/racer/assets/{args.stem}.jblob")
    print(f"Warm-start future retrains:  python -m drone_jepa.train "
          f"--warm-start artifacts/{args.stem}.pt ...")


if __name__ == "__main__":
    main()
