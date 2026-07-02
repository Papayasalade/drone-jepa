"""Export a trained SkyJEPA checkpoint for the Candle (Rust) inference port:
  - weights  -> safetensors (Candle loads these natively),
  - config   -> JSON,
  - goldens  -> JSON: predict() inputs+outputs, to assert Candle == PyTorch.

    .venv/bin/python scripts/export_jepa.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from scipy.spatial.transform import Rotation

from drone_jepa.model.jepa import SkyJEPA

# usage: export_jepa.py [checkpoint.pt] [out_stem]
CKPT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/skyjepa_ctbr_1x.pt"
STEM = sys.argv[2] if len(sys.argv) > 2 else "skyjepa_ctbr_1x"
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "web-demo" / "jepa-rs" / "weights"
FIX = ROOT / "web-demo" / "fixtures" / "jepa"


def random_state(rng: np.random.Generator) -> np.ndarray:
    """One valid 18-dim state (pos, vel, R row-major, omega)."""
    R = Rotation.from_euler("xyz", rng.uniform(-0.5, 0.5, 3)).as_matrix()
    return np.concatenate([
        rng.uniform(-1.0, 1.0, 3),   # pos
        rng.uniform(-1.0, 1.0, 3),   # vel
        R.reshape(9),                # row-major
        rng.uniform(-1.0, 1.0, 3),   # omega
    ]).astype(np.float64)


def main() -> None:
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    FIX.mkdir(parents=True, exist_ok=True)

    model, cfg = SkyJEPA.from_checkpoint(CKPT, device="cpu")
    model.eval()
    H, T = model.H, model.T

    # --- weights -> safetensors (keep PyTorch state_dict key names) ---
    sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    st_path = WEIGHTS / f"{STEM}.safetensors"
    save_file(sd, str(st_path))

    full_cfg = {
        "stem": STEM,
        "action_mode": cfg.get("action_mode", "rotor_force"),
        "pos_mode": cfg.get("pos_mode", "zero"),
        "history": int(H),
        "horizon": int(T),
        "width_mult": float(cfg.get("width_mult", 1)),
        "prober_hidden": int(cfg.get("prober_hidden", 40)),
        "dt": float(model.dki.dt),
        "m_nominal": float(model.dki.m_nominal),
        "g": float(model.dki.g),
        "state_dim": 18,
        "action_dim": 4,
    }
    (WEIGHTS / f"{STEM}.json").write_text(json.dumps(full_cfg, indent=2))

    # --- goldens: predict() on a few random (history, action-window) inputs ---
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(4):
        state_hist = np.stack([random_state(rng) for _ in range(H)])  # (H,18)
        # CTBR action window around hover: thrust ~ m*g, modest body rates.
        aw = np.zeros((H + T, 4))
        aw[:, 0] = model.dki.m_nominal * 9.81 + rng.uniform(-1.0, 1.0, H + T)
        aw[:, 1:] = rng.uniform(-2.0, 2.0, (H + T, 3))

        sh = torch.tensor(state_hist, dtype=torch.float32).unsqueeze(0)
        awt = torch.tensor(aw, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model.predict(sh, awt, horizon=T)  # (1,T,18)
        cases.append({
            "state_hist": state_hist.tolist(),
            "action_window": aw.tolist(),
            "pred": pred.squeeze(0).double().numpy().tolist(),  # (T,18)
        })

    (FIX / "goldens.json").write_text(json.dumps({"config": full_cfg, "cases": cases}))

    n_params = sum(v.numel() for v in sd.values() if v.dtype.is_floating_point)
    print(f"wrote {st_path.name} ({n_params} params), config, and {len(cases)} goldens")
    print(f"  action_mode={full_cfg['action_mode']} H={H} T={T} dt={full_cfg['dt']}")
    print(f"  weights: {st_path}")
    print(f"  goldens: {FIX / 'goldens.json'}")


if __name__ == "__main__":
    main()
