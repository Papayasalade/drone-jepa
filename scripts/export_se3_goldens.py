"""Golden fixtures for the Rust SE3 controller port: feed identical (params, state,
flat reference) into RotorPy's SE3Control and dump its cmd_motor_thrusts, so the
Rust Se3Control can be asserted equal.

    .venv/bin/python scripts/export_se3_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control

from drone_jepa.data_gen.sim import sample_params

OUT = Path(__file__).resolve().parents[1] / "web-demo" / "rotor-rs" / "fixtures" / "se3"


def vec(a):
    return [float(x) for x in np.asarray(a).ravel()]


def params_json(p):
    return {
        "mass": float(p["mass"]),
        "Ixx": float(p["Ixx"]), "Iyy": float(p["Iyy"]), "Izz": float(p["Izz"]),
        "Ixy": float(p["Ixy"]), "Iyz": float(p["Iyz"]), "Ixz": float(p["Ixz"]),
        "rotor_pos": [vec(p["rotor_pos"][k]) for k in p["rotor_pos"]],
        "rotor_directions": vec(p["rotor_directions"]),
        "c_Dx": float(p["c_Dx"]), "c_Dy": float(p["c_Dy"]), "c_Dz": float(p["c_Dz"]),
        "k_eta": float(p["k_eta"]), "k_m": float(p["k_m"]),
        "k_d": float(p.get("k_d", 0.0)), "k_z": float(p.get("k_z", 0.0)),
        "k_h": float(p.get("k_h", 0.0)), "k_flap": float(p.get("k_flap", 0.0)),
        "tau_m": float(p["tau_m"]),
        "rotor_speed_min": float(p["rotor_speed_min"]),
        "rotor_speed_max": float(p["rotor_speed_max"]),
        "k_w": float(p.get("k_w", 1.0)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    cases = []
    for _ in range(40):
        p = sample_params(rng)
        ctrl = SE3Control(p)
        state = {
            "x": rng.uniform(-2, 2, 3),
            "v": rng.uniform(-2, 2, 3),
            "q": Rotation.from_euler("xyz", rng.uniform(-0.8, 0.8, 3)).as_quat(),
            "w": rng.uniform(-1.5, 1.5, 3),
        }
        flat = {
            "x": rng.uniform(-2, 2, 3),
            "x_dot": rng.uniform(-1, 1, 3),
            "x_ddot": rng.uniform(-2, 2, 3),
            "x_dddot": np.zeros(3), "x_ddddot": np.zeros(3),
            "yaw": float(rng.uniform(-np.pi, np.pi)),
            "yaw_dot": float(rng.uniform(-1, 1)),
        }
        u = ctrl.update(0.0, state, flat)
        cases.append({
            "params": params_json(p),
            "state": {"x": vec(state["x"]), "v": vec(state["v"]),
                      "q": vec(state["q"]), "w": vec(state["w"])},
            "flat": {"x": vec(flat["x"]), "x_dot": vec(flat["x_dot"]),
                     "x_ddot": vec(flat["x_ddot"]),
                     "yaw": flat["yaw"], "yaw_dot": flat["yaw_dot"]},
            "forces": vec(u["cmd_motor_thrusts"]),
        })
    (OUT / "goldens.json").write_text(json.dumps({"cases": cases}))
    print(f"wrote {OUT / 'goldens.json'}: {len(cases)} SE3 cases")


if __name__ == "__main__":
    main()
