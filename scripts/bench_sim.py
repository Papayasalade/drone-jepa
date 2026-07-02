"""Benchmark Python RotorPy: cost of one `Multirotor.step` (the ported unit) and
end-to-end trajectory throughput. Pair with `cargo run --release --example bench`.

    .venv/bin/python scripts/bench_sim.py
"""

from __future__ import annotations

import time

import numpy as np

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.hummingbird_params import quad_params as BASE

from drone_jepa.data_gen.sim import simulate_trajectory


def bench_step(n: int = 50_000) -> None:
    veh = Multirotor(BASE, control_abstraction="cmd_motor_thrusts", aero=True)
    hover_f = BASE["mass"] * 9.81 / 4.0
    cmd = {"cmd_motor_thrusts": np.full(4, hover_f)}
    state = {
        "x": np.array([0.0, 0.0, 1.5]), "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]), "w": np.zeros(3),
        "wind": np.zeros(3), "rotor_speeds": np.full(4, 500.0),
    }
    dt = 0.005

    # warmup
    for _ in range(200):
        state = veh.step(state, cmd, dt)

    t0 = time.perf_counter()
    for _ in range(n):
        state = veh.step(state, cmd, dt)
    el = time.perf_counter() - t0

    sps = n / el
    print(f"[python] Multirotor.step  : {sps:>12,.0f} steps/s   {el / n * 1e6:8.2f} us/step")
    print(f"[python]   real-time @200Hz: {sps / 200:>8.0f}x")


def bench_traj(n: int = 20) -> None:
    rng = np.random.default_rng(0)
    # warmup
    simulate_trajectory(rng, t_final=2.0)
    t0 = time.perf_counter()
    for _ in range(n):
        simulate_trajectory(rng, t_final=10.0)
    el = time.perf_counter() - t0
    print(f"[python] simulate_trajectory (10s, SE3@200Hz): {n / el:6.2f} traj/s   {el / n * 1e3:7.1f} ms/traj")


if __name__ == "__main__":
    bench_step()
    bench_traj()
