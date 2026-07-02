"""RotorPy wrapper + domain-randomization sampler (Table I).

We drive RotorPy's `Multirotor` directly (bypassing the Environment runner) in a
tight 20 Hz loop with `control_abstraction='cmd_motor_thrusts'`, so the control
input is literally the four rotor forces SkyJEPA uses as its action. An
SE3 geometric controller (built with the *true* randomized params, so it stays
stable across domains) tracks a random reference trajectory to produce diverse
closed-loop data.

State is logged in the canonical 18-dim layout (pos, vel, R-flat, omega-body);
the action logged is the commanded rotor-force vector, clamped to motor limits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.hummingbird_params import quad_params as BASE_PARAMS

from ..state import POS, VEL
from .references import RandomFourierTrajectory


# ---- Table I domain randomization ranges ----------------------------------
# Multiplicative ranges are factors on the base value; absolute ranges replace.
@dataclass
class DRRanges:
    mass: tuple = (0.5, 1.5)      # +-50%
    inertia: tuple = (0.7, 1.3)   # +-30% (Ixx,Iyy,Izz)
    tau_m: tuple = (0.01, 0.03)   # seconds (paper: 0.01-0.1; capped because the
                                  # bundled SE3 controller does not model motor lag
                                  # and oscillates for slow motors — see NOTES.md)
    drag: tuple = (0.05, 0.30)    # absolute c_Dx/y/z  (paper: 0.1-0.5; tamed for sim stability)
    k_eta: tuple = (0.5, 1.5)     # thrust coeff +-50%
    k_m: tuple = (0.5, 1.5)       # torque coeff +-50%


# Out-of-distribution ranges: domains BEYOND the training support, to test
# zero-shot generalization (the paper's actual regime — sim-to-real is OOD).
OOD_RANGES = DRRanges(
    mass=(1.5, 2.2),       # heavier than any training domain (>0.75 kg)
    inertia=(1.3, 1.7),
    tau_m=(0.03, 0.05),    # slower motors than training (still controllable)
    drag=(0.30, 0.45),     # heavier drag than training
    k_eta=(0.4, 0.6),      # weaker thrust than training
    k_m=(0.4, 0.6),
)


def sample_params(rng: np.random.Generator, ranges: DRRanges = DRRanges()) -> dict:
    """Draw one domain's quad_params by perturbing the base PX4 1 kg quad."""
    p = copy.deepcopy(BASE_PARAMS)
    p["mass"] = BASE_PARAMS["mass"] * rng.uniform(*ranges.mass)
    for k in ("Ixx", "Iyy", "Izz"):
        p[k] = BASE_PARAMS[k] * rng.uniform(*ranges.inertia)
    p["tau_m"] = rng.uniform(*ranges.tau_m)
    for k in ("c_Dx", "c_Dy", "c_Dz"):
        p[k] = rng.uniform(*ranges.drag)
    p["k_eta"] = BASE_PARAMS["k_eta"] * rng.uniform(*ranges.k_eta)
    p["k_m"] = BASE_PARAMS["k_m"] * rng.uniform(*ranges.k_m)
    return p


def rotor_force_limit(params: dict) -> float:
    """Max physical force a single rotor can produce: k_eta * w_max^2."""
    return params["k_eta"] * params["rotor_speed_max"] ** 2


def _state_to_vec(state: dict) -> np.ndarray:
    """RotorPy state dict -> canonical 18-dim vector."""
    R = Rotation.from_quat(state["q"]).as_matrix()  # body->world (q is [i,j,k,w])
    return np.concatenate([state["x"], state["v"], R.reshape(9), state["w"]])


def random_initial_state(rng: np.random.Generator, center=(0.0, 0.0, 1.5),
                         hover_speed: float = 500.0) -> dict:
    """Hover-ish start with small random perturbations."""
    att = Rotation.from_euler("xyz", rng.uniform(-0.2, 0.2, 3))
    return {
        "x": np.asarray(center, float) + rng.uniform(-0.3, 0.3, 3),
        "v": rng.uniform(-0.3, 0.3, 3),
        "q": att.as_quat(),
        "w": rng.uniform(-0.2, 0.2, 3),
        "wind": np.zeros(3),
        "rotor_speeds": np.full(4, hover_speed),
    }


@dataclass
class TrajResult:
    states: np.ndarray   # (N, 18)
    actions: np.ndarray  # (N, 4)  rotor forces commanded at each step
    params: dict = field(repr=False, default=None)

    def is_valid(self, center=(0.0, 0.0, 1.5), max_pos_dev: float = 12.0,
                 max_speed: float = 15.0) -> bool:
        """Reject diverged / unstable rollouts (NaNs, runaway, or crash)."""
        if not np.isfinite(self.states).all():
            return False
        pos = self.states[:, POS] - np.asarray(center)
        speed = np.linalg.norm(self.states[:, VEL], axis=1)
        return bool(np.abs(pos).max() < max_pos_dev and speed.max() < max_speed)


RATE_MAX = 12.0  # body-rate command clamp [rad/s] for CTBR action mode


def simulate_trajectory(rng: np.random.Generator, params: dict | None = None,
                        t_final: float = 10.0, dt: float = 0.05,
                        dt_control: float = 0.005,
                        ranges: DRRanges = DRRanges(),
                        action_noise: float = 0.0,
                        action_mode: str = "rotor_force") -> TrajResult:
    """Collect one closed-loop trajectory under one (possibly random) domain.

    The SE(3) controller runs at the fine rate `dt_control` (e.g. 200 Hz) and we
    subsample state/action to the `dt`=20 Hz model rate — exactly the paper's
    "resampled to 20 Hz". This is critical: running SE(3) feedback directly at
    20 Hz produces violently chattering rotor-force commands (≈2.5 N step-to-step
    on a ≈3.8 N signal) that, with motor lag, make the dynamics near-unpredictable
    for a tiny model. Fine-rate control is smooth; the logged 20 Hz action is the
    *mean* fine command over each window (the best zero-order-hold equivalent).

    action_noise (N, std): optional Gaussian exploration noise on the executed
    fine command, to broaden the state/action distribution for closed-loop use.
    """
    if params is None:
        params = sample_params(rng, ranges)

    ctbr = action_mode == "ctbr"
    abstraction = "cmd_ctbr" if ctbr else "cmd_motor_thrusts"
    vehicle = Multirotor(params, control_abstraction=abstraction, aero=True)
    controller = SE3Control(params)
    center = np.array([0.0, 0.0, 1.5])
    traj = RandomFourierTrajectory(rng, center=tuple(center))

    state = random_initial_state(rng, center=center)
    f_max = rotor_force_limit(params)
    total_thrust_max = 4 * f_max

    decim = max(1, int(round(dt / dt_control)))
    n_log = int(round(t_final / dt))
    states = np.zeros((n_log, 18))
    actions = np.zeros((n_log, 4))

    win = []  # fine actions accumulated over the current 20 Hz window
    k = 0
    for i in range(n_log * decim):
        if i % decim == 0:
            states[k] = _state_to_vec(state)  # state at window start
            win = []
        flat = traj.update(i * dt_control)
        u = controller.update(i * dt_control, state, flat)
        if ctbr:
            # action = [collective thrust, body rates] (real-drone CTBR command);
            # rates clamped to ±RATE_MAX to drop the rare initial-tilt transient.
            thrust = float(np.clip(u["cmd_thrust"], 0.0, total_thrust_max))
            w_cmd = np.clip(u["cmd_w"], -RATE_MAX, RATE_MAX)
            if action_noise > 0.0:
                thrust = thrust + rng.normal(0.0, action_noise)
                w_cmd = w_cmd + rng.normal(0.0, action_noise, size=3)
                w_cmd = np.clip(w_cmd, -RATE_MAX, RATE_MAX)
            act = np.concatenate([[thrust], w_cmd])
            state = vehicle.step(state, {"cmd_thrust": thrust, "cmd_w": w_cmd}, dt_control)
        else:
            force = u["cmd_motor_thrusts"]
            if action_noise > 0.0:
                force = force + rng.normal(0.0, action_noise, size=4)
            act = np.clip(force, 0.0, f_max)
            state = vehicle.step(state, {"cmd_motor_thrusts": act}, dt_control)
        win.append(act)
        if (i + 1) % decim == 0:
            actions[k] = np.mean(win, axis=0)  # ZOH-equivalent mean command
            k += 1

    return TrajResult(states=states, actions=actions, params=params)
