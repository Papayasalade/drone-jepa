"""Diagnose WHERE Python RotorPy's per-step time goes, to check the Rust
comparison is apples-to-apples (same physics) and not scipy doing something
pathological.

    .venv/bin/python scripts/bench_diag.py
"""

from __future__ import annotations

import time

import numpy as np
import scipy.integrate

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.hummingbird_params import quad_params as BASE


def make():
    veh = Multirotor(BASE, control_abstraction="cmd_motor_thrusts", aero=True)
    hover_f = BASE["mass"] * 9.81 / 4.0
    cmd = {"cmd_motor_thrusts": np.full(4, hover_f)}
    state = {
        "x": np.array([0.0, 0.0, 1.5]), "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]), "w": np.zeros(3),
        "wind": np.zeros(3), "rotor_speeds": np.full(4, 500.0),
    }
    return veh, cmd, state


def time_it(fn, n):
    for _ in range(max(50, n // 20)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    veh, cmd, state = make()
    dt = 0.005

    # how many derivative evals does scipy's adaptive RK45 actually do per step?
    cmd_speeds = veh.get_cmd_motor_speeds(state, cmd)
    cmd_speeds = np.clip(cmd_speeds, veh.rotor_speed_min, veh.rotor_speed_max)
    s = Multirotor._pack_state(state)

    def s_dot_fn(t, y):
        return veh._s_dot_fn(t, y, cmd_speeds)

    sol = scipy.integrate.solve_ivp(s_dot_fn, (0.0, dt), s, method="RK45")
    print(f"scipy RK45 per 5 ms step:  nfev={sol.nfev}  internal_steps={len(sol.t) - 1}")

    # cost of a single derivative evaluation (the actual physics)
    t_sdot = time_it(lambda: veh._s_dot_fn(0.0, s, cmd_speeds), 200_000)
    print(f"one _s_dot_fn eval         : {t_sdot * 1e6:8.2f} us")

    # cost of just scipy.solve_ivp overhead vs the math it dispatches
    t_step = time_it(lambda: veh.step(state, cmd, dt), 5_000)
    print(f"full veh.step (solve_ivp)  : {t_step * 1e6:8.2f} us   (= {t_step / t_sdot:.0f} x one eval)")
    print(f"  -> solve_ivp did ~{sol.nfev} evals; overhead beyond them = "
          f"{(t_step - sol.nfev * t_sdot) * 1e6:.1f} us")

    # apples-to-apples: SAME integrator as Rust (fixed RK4, N_SUB=8) in numpy,
    # calling the SAME _s_dot_fn. Isolates language/numpy overhead from method.
    def rk4_fixed(n_sub=8):
        y = s.copy()
        h = dt / n_sub
        for _ in range(n_sub):
            k1 = veh._s_dot_fn(0.0, y, cmd_speeds)
            k2 = veh._s_dot_fn(0.0, y + 0.5 * h * k1, cmd_speeds)
            k3 = veh._s_dot_fn(0.0, y + 0.5 * h * k2, cmd_speeds)
            k4 = veh._s_dot_fn(0.0, y + h * k3, cmd_speeds)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    t_rk4 = time_it(rk4_fixed, 5_000)
    print(f"numpy fixed RK4 (N_SUB=8)  : {t_rk4 * 1e6:8.2f} us/step   "
          f"({1 / t_rk4:,.0f} steps/s)   [same integrator as Rust]")


if __name__ == "__main__":
    main()
