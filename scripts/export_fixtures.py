"""Export golden fixtures for the Rust RotorPy port (`web-demo/rotor-rs`).

RotorPy ships no tests, so these JSON fixtures ARE the spec. We drive RotorPy's
`Multirotor` at the fine control rate (mirroring `drone_jepa/data_gen/sim.py`)
and dump, per fine step:

  - the exact command dict passed to `vehicle.step` (post-clip),
  - the full resulting state (x, v, q, w, wind, rotor_speeds),
  - the full 20-dim state derivative `s_dot` at that step (integrator-independent,
    for the exact-math gate in the Rust test).

Coverage: both control abstractions (cmd_motor_thrusts, cmd_ctbr), nominal + OOD
+ edge domains (slow/fast tau_m, high tilt, near motor saturation), and nonzero
constant wind. Run:

    .venv/bin/python scripts/export_fixtures.py
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.multirotor import Multirotor

from drone_jepa.data_gen.references import RandomFourierTrajectory
from drone_jepa.data_gen.sim import (
    DRRanges,
    OOD_RANGES,
    RATE_MAX,
    random_initial_state,
    rotor_force_limit,
    sample_params,
)

OUT_DIR = Path(os.environ.get("ROTOR_RS_DIR", Path(__file__).resolve().parents[2] / "rotor-rs")) / "fixtures" / "sim"


# --------------------------------------------------------------------------- #
# JSON serialization helpers
# --------------------------------------------------------------------------- #
def _vec(a) -> list:
    return [float(x) for x in np.asarray(a).ravel()]


def params_to_json(p: dict) -> dict:
    """RotorPy quad_params -> JSON-friendly dict the Rust side can rebuild from."""
    return {
        "mass": float(p["mass"]),
        "Ixx": float(p["Ixx"]), "Iyy": float(p["Iyy"]), "Izz": float(p["Izz"]),
        "Ixy": float(p["Ixy"]), "Iyz": float(p["Iyz"]), "Ixz": float(p["Ixz"]),
        "num_rotors": int(p["num_rotors"]),
        # rotor positions as an (num_rotors, 3) list, in dict order r1..rN
        "rotor_pos": [_vec(p["rotor_pos"][k]) for k in p["rotor_pos"]],
        "rotor_directions": _vec(p["rotor_directions"]),
        "c_Dx": float(p["c_Dx"]), "c_Dy": float(p["c_Dy"]), "c_Dz": float(p["c_Dz"]),
        "k_eta": float(p["k_eta"]), "k_m": float(p["k_m"]),
        "k_d": float(p.get("k_d", 0.0)), "k_z": float(p.get("k_z", 0.0)),
        "k_h": float(p.get("k_h", 0.0)), "k_flap": float(p.get("k_flap", 0.0)),
        "tau_m": float(p["tau_m"]),
        "rotor_speed_min": float(p["rotor_speed_min"]),
        "rotor_speed_max": float(p["rotor_speed_max"]),
        "k_w": float(p.get("k_w", 1.0)),
    }


def state_to_json(s: dict) -> dict:
    return {
        "x": _vec(s["x"]), "v": _vec(s["v"]), "q": _vec(s["q"]),
        "w": _vec(s["w"]), "wind": _vec(s["wind"]),
        "rotor_speeds": _vec(s["rotor_speeds"]),
    }


def full_sdot(vehicle: Multirotor, state: dict, control: dict) -> list:
    """Replicate Multirotor.step's pre-integration setup to get the full 20-dim
    derivative (RotorPy's public `statedot` only returns vdot/wdot)."""
    cmd_rotor_speeds = vehicle.get_cmd_motor_speeds(state, control)
    cmd_rotor_speeds = np.clip(
        cmd_rotor_speeds, vehicle.rotor_speed_min, vehicle.rotor_speed_max
    )
    s = Multirotor._pack_state(state)
    return _vec(vehicle._s_dot_fn(0.0, s, cmd_rotor_speeds))


# --------------------------------------------------------------------------- #
# Fixture generation
# --------------------------------------------------------------------------- #
def make_fixture(name: str, *, abstraction: str, params: dict,
                 init_state: dict, t_final: float = 2.0, dt_control: float = 0.005,
                 seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ctbr = abstraction == "cmd_ctbr"
    vehicle = Multirotor(params, control_abstraction=abstraction, aero=True)
    controller = SE3Control(params)
    traj = RandomFourierTrajectory(rng, center=(0.0, 0.0, 1.5))

    f_max = rotor_force_limit(params)
    total_thrust_max = 4 * f_max

    state = copy.deepcopy(init_state)
    n = int(round(t_final / dt_control))
    steps = []
    for i in range(n):
        flat = traj.update(i * dt_control)
        u = controller.update(i * dt_control, state, flat)
        if ctbr:
            thrust = float(np.clip(u["cmd_thrust"], 0.0, total_thrust_max))
            w_cmd = np.clip(u["cmd_w"], -RATE_MAX, RATE_MAX)
            control = {"cmd_thrust": thrust, "cmd_w": w_cmd}
            cmd_json = {"cmd_thrust": thrust, "cmd_w": _vec(w_cmd)}
        else:
            force = np.clip(u["cmd_motor_thrusts"], 0.0, f_max)
            control = {"cmd_motor_thrusts": force}
            cmd_json = {"cmd_motor_thrusts": _vec(force)}

        sdot = full_sdot(vehicle, state, control)
        state = vehicle.step(state, control, dt_control)
        steps.append({"cmd": cmd_json, "sdot": sdot, "state": state_to_json(state)})

    return {
        "name": name,
        "abstraction": abstraction,
        "dt": dt_control,
        "params": params_to_json(params),
        "initial_state": state_to_json(init_state),
        "steps": steps,
    }


def build_all() -> list[dict]:
    fixtures = []
    base_rng = np.random.default_rng(20240625)

    # nominal + OOD domains, both abstractions
    domain_specs = [
        ("nominal", DRRanges()),
        ("ood", OOD_RANGES),
    ]
    for dname, ranges in domain_specs:
        for abstraction in ("cmd_motor_thrusts", "cmd_ctbr"):
            params = sample_params(base_rng, ranges)
            init = random_initial_state(base_rng)
            fixtures.append(make_fixture(
                f"{dname}_{abstraction}", abstraction=abstraction,
                params=params, init_state=init, seed=int(base_rng.integers(1 << 30)),
            ))

    # edge: slow motors (large tau_m)
    p = sample_params(base_rng)
    p["tau_m"] = 0.05
    fixtures.append(make_fixture(
        "edge_slow_tau", abstraction="cmd_motor_thrusts", params=p,
        init_state=random_initial_state(base_rng), seed=11,
    ))

    # edge: fast motors (small tau_m)
    p = sample_params(base_rng)
    p["tau_m"] = 0.008
    fixtures.append(make_fixture(
        "edge_fast_tau", abstraction="cmd_ctbr", params=p,
        init_state=random_initial_state(base_rng), seed=12,
    ))

    # edge: high initial tilt + spin (stresses SO(3)/wrench path)
    p = sample_params(base_rng)
    init = random_initial_state(base_rng)
    from scipy.spatial.transform import Rotation
    init["q"] = Rotation.from_euler("xyz", [0.7, -0.6, 0.4]).as_quat()
    init["w"] = np.array([1.5, -1.2, 0.8])
    fixtures.append(make_fixture(
        "edge_high_tilt", abstraction="cmd_motor_thrusts", params=p,
        init_state=init, seed=13,
    ))

    # wind: nonzero constant wind exercises R.T @ (v - wind) airspeed path
    for abstraction, wind in (
        ("cmd_motor_thrusts", [4.0, -2.0, 1.0]),
        ("cmd_ctbr", [-3.0, 3.0, -1.5]),
    ):
        p = sample_params(base_rng)
        init = random_initial_state(base_rng)
        init["wind"] = np.array(wind, dtype=float)
        fixtures.append(make_fixture(
            f"wind_{abstraction}", abstraction=abstraction, params=p,
            init_state=init, seed=14,
        ))

    return fixtures


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = build_all()
    for fx in fixtures:
        path = OUT_DIR / f"{fx['name']}.json"
        with open(path, "w") as f:
            json.dump(fx, f)
        n = len(fx["steps"])
        last = fx["steps"][-1]["state"]
        finite = np.isfinite(np.array(last["x"])).all()
        print(f"  wrote {path.name:28s}  steps={n:4d}  final_pos={last['x']}  finite={finite}")
    print(f"\n{len(fixtures)} fixtures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
