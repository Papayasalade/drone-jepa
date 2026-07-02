"""Interactive UI for drone-jepa — compare world models head-to-head.

Run with:   streamlit run app.py

Models compared across every tab (pick in the sidebar; 1x + 2x + strong AR on by
default): 2x JEPA, 1x JEPA, strong autoregressive baseline, naive 1-step AR.

Tabs:
  1. Prediction explorer — pick a moment in a held-out flight, drag a horizon
     slider, see each model's predicted path vs ground truth in 3D.
  2. Compounding curves — error-vs-horizon for each model, averaged over windows.
  3. Fly with MPPI — race each model as the world model inside the MPPI controller.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from drone_jepa.model.baseline import DirectBaseline
from drone_jepa.model.jepa import SkyJEPA
from drone_jepa.state import OMEGA, POS, ROT, VEL

DATA = "artifacts/dataset.pt"
TRUTH_COLOR = "#2ca02c"
# label -> (checkpoint, kind, color, default_on)
MODEL_SPEC = {
    "2x JEPA": ("artifacts/skyjepa.pt", "jepa", "#1f77b4", True),
    "1x JEPA": ("artifacts/skyjepa_1x.pt", "jepa", "#9467bd", True),
    "CTBR JEPA (2x)": ("artifacts/skyjepa_ctbr.pt", "jepa", "#17becf", True),
    "CTBR JEPA (1x)": ("artifacts/skyjepa_ctbr_1x.pt", "jepa", "#7f7f7f", True),
    "AR (strong)": ("artifacts/baseline.pt", "baseline", "#d62728", True),
    "AR (naive 1-step)": ("artifacts/baseline_1step.pt", "baseline", "#ff7f0e", False),
}

st.set_page_config(page_title="drone-jepa", layout="wide")


@st.cache_resource
def load_models():
    models, meta = {}, {}
    for label, (path, kind, color, _) in MODEL_SPEC.items():
        if not os.path.exists(path):
            continue
        if kind == "jepa":
            m, cfg = SkyJEPA.from_checkpoint(path)
            n = sum(p.numel() for p in m.parameters())
        else:
            ck = torch.load(path, weights_only=False)
            cfg = ck["config"]
            m = DirectBaseline(history=cfg["history"], horizon=cfg["horizon"])
            m.load_state_dict(ck["model"]); m.eval()
            n = sum(p.numel() for p in m.parameters())
        models[label] = m
        meta[label] = {"color": color, "kind": kind, "params": n,
                       "H": cfg["history"], "T": cfg["horizon"],
                       "action_mode": cfg.get("action_mode", "rotor_force")}
    return models, meta


@st.cache_data
def load_test_trajectories():
    data = torch.load(DATA, weights_only=False)
    states, actions = data["states"], data["actions"]
    N = states.shape[0]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(N, generator=g)[:int(N * 0.1)]
    return states[idx], actions[idx]


@torch.no_grad()
def model_predict(model, X_win, A_win, H, L):
    return model.predict(X_win[None, :H], A_win[None], horizon=L)[0].numpy()


def pos_err(pred, truth):
    return np.linalg.norm(pred[:, POS] - truth[:, POS], axis=1)


models, meta = load_models()
states, actions = load_test_trajectories()
n_test, traj_len, _ = states.shape
# any JEPA's H/T (all share H=10, T=20)
H = next(v["H"] for v in meta.values())
T = next(v["T"] for v in meta.values())

st.title("🚁 drone-jepa — world-model comparison")
st.caption(f"Held-out test set: {n_test} flights × {traj_len} steps @ 20 Hz. "
           f"Models trained on horizon T={T} (history H={H}).")

# race-only controllers (no world model -> only appear in the MPPI/fly race)
import os as _os
HAS_RL = _os.path.exists("artifacts/rl_policy.zip")
CONTROLLERS = {}                       # label -> ("rl"|"pid", color)
if HAS_RL:
    CONTROLLERS["RL policy (PPO)"] = ("rl", "#e377c2")
CONTROLLERS["PID (classical)"] = ("pid", "#bcbd22")
if os.path.exists("artifacts/skyjepa_ctbr.pt"):
    CONTROLLERS["Gradient-MPC (CTBR)"] = ("gradmpc", "#393b79")
for _lab, (_k, _c) in CONTROLLERS.items():
    meta[_lab] = {"color": _c, "kind": _k}

# ---- global selection ----------------------------------------------------- #
with st.sidebar:
    st.header("World models")
    st.caption("Predict-capable — appear in every tab.")
    chosen = []
    for label, (_, _, _, default_on) in MODEL_SPEC.items():
        if label in models:
            if st.checkbox(f"{label}  ({meta[label]['params']/1000:.1f}K)", value=default_on):
                chosen.append(label)
    st.header("Race controllers")
    st.caption("No world model — only in the fly race.")
    chosen_ctrl = []
    for label, (kind, _) in CONTROLLERS.items():
        if st.checkbox(label, value=True):
            chosen_ctrl.append(label)

tab_pred, tab_curve, tab_mppi = st.tabs(
    ["🔮 Prediction explorer", "📈 Compounding curves", "🎮 Fly race"])

# ============================ TAB 1: prediction ============================ #
with tab_pred:
    c = st.columns([1, 3])
    with c[0]:
        ti = st.slider("Flight (test trajectory)", 0, n_test - 1, 0)
        tau, act = states[ti], actions[ti]
        t_now = st.slider("Current step t", H - 1, traj_len - 2, min(40, traj_len - 2))
        max_L = traj_len - 1 - t_now
        L = st.slider("Horizon (steps × 0.05 s)", 1, min(100, max_L), min(40, max_L))
        st.caption(f"Predicting t+1 … t+{L}  ({L*0.05:.2f} s ahead).")

    start = t_now - H + 1
    X_win = tau[start:start + H + L]
    A_win = act[start:start + H + L]
    truth = X_win[H:H + L].numpy()
    hist_pos = tau[start:t_now + 1, POS].numpy()
    preds = {lab: model_predict(models[lab], X_win, A_win, H, L) for lab in chosen}

    with c[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=hist_pos[:, 0], y=hist_pos[:, 1], z=hist_pos[:, 2],
                      mode="lines", line=dict(color="gray", width=4), name="history"))
        fig.add_trace(go.Scatter3d(x=truth[:, 0], y=truth[:, 1], z=truth[:, 2],
                      mode="lines", line=dict(color=TRUTH_COLOR, width=8), name="ground truth"))
        for lab, p in preds.items():
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                          line=dict(color=meta[lab]["color"], width=5, dash="dash"), name=lab))
        fig.add_trace(go.Scatter3d(x=[hist_pos[-1, 0]], y=[hist_pos[-1, 1]], z=[hist_pos[-1, 2]],
                      mode="markers", marker=dict(size=4, color="black"), name="now"))
        fig.update_layout(height=540, margin=dict(l=0, r=0, t=10, b=0),
                          scene=dict(aspectmode="data", xaxis_title="x [m]",
                                     yaxis_title="y [m]", zaxis_title="z [m]"),
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, use_container_width=True)

    cc = st.columns([3, 1])
    with cc[0]:
        ef = go.Figure()
        h = np.arange(1, L + 1) * 0.05
        for lab, p in preds.items():
            ef.add_trace(go.Scatter(x=h, y=pos_err(p, truth), mode="lines",
                         line=dict(color=meta[lab]["color"]), name=lab))
        ef.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis_title="seconds ahead", yaxis_title="position error [m]",
                         legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(ef, use_container_width=True)
    with cc[1]:
        st.markdown(f"**Error at t+{L} ({L*0.05:.2f}s)**")
        for lab, p in preds.items():
            st.metric(lab, f"{pos_err(p, truth)[-1]:.2f} m")

# ============================ TAB 2: curves ================================ #
with tab_curve:
    st.markdown("Average **position error vs horizon** over many held-out windows.")
    cc = st.columns([1, 3])
    with cc[0]:
        L2 = st.slider("Horizon (steps)", 10, 100, 40, key="curve_L")
        n_win = st.slider("Windows to average", 50, 800, 250, step=50)
    rng = np.random.default_rng(0)
    acc = {lab: None for lab in chosen}
    nwin = 0
    for _ in range(n_win):
        ti = int(rng.integers(0, n_test))
        smax = traj_len - (H + L2)
        if smax <= 0:
            break
        s = int(rng.integers(0, smax))
        Xw, Aw = states[ti, s:s + H + L2], actions[ti, s:s + H + L2]
        tr = Xw[H:H + L2].numpy()
        for lab in chosen:
            e = pos_err(model_predict(models[lab], Xw, Aw, H, L2), tr) ** 2
            acc[lab] = e if acc[lab] is None else acc[lab] + e
        nwin += 1
    with cc[1]:
        ef = go.Figure()
        h = np.arange(1, L2 + 1) * 0.05
        ef.add_vline(x=T * 0.05, line=dict(color="gray", dash="dot"),
                     annotation_text="train horizon")
        for lab in chosen:
            ef.add_trace(go.Scatter(x=h, y=np.sqrt(acc[lab] / nwin), mode="lines",
                         line=dict(color=meta[lab]["color"]), name=lab))
        ef.update_layout(height=440, xaxis_title="seconds ahead",
                         yaxis_title="position RMSE [m]", margin=dict(l=0, r=0, t=10, b=0),
                         legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(ef, use_container_width=True)

# ============================ TAB 3: MPPI race ============================= #
with tab_mppi:
    st.markdown("Race each selected model as the **world model inside MPPI**, flying "
                "a RotorPy drone (same held-out domain & reference) — tests *closed-loop* "
                "control, which differs from open-loop accuracy.")
    st.info("⚠️ Closed-loop control is the hard, data-limited part of this project; "
            "tracking is loose and some models drift. The comparison is what's interesting.")
    cols = st.columns(6)
    seconds = cols[0].slider("Seconds", 4.0, 14.0, 8.0)
    samples = cols[1].slider("MPPI samples", 128, 512, 384, step=128)
    sigma = cols[2].slider("Action noise σ [N]", 0.2, 1.5, 0.6)
    smooth = cols[3].slider("Noise smoothing", 0.0, 0.95, 0.8,
                            help="temporal smoothing of candidate actions; "
                                 "higher = smoother control, ~halves tracking error")
    ki = cols[4].slider("Integral gain kᵢ", 0.0, 0.4, 0.1,
                        help="integral action — drives steady-state offset "
                             "(e.g. altitude) to zero, like a real flight controller")
    seed = cols[5].number_input("Seed", 0, 9999, 3)
    sampler = st.radio("Sampler", ["gaussian", "icem"], horizontal=True,
                       help="gaussian = robust EMA-smoothed MPPI (recommended). "
                            "icem = colored-noise + spline + elite refit — on this "
                            "imperfect model it exploits model bias and DIVERGES "
                            "(an instructive failure: aggressive planner + imperfect model).")

    if st.button("▶ Fly & compare", type="primary"):
        from rotorpy.vehicles.multirotor import Multirotor

        from drone_jepa.control import MPPIController
        from drone_jepa.data_gen.sim import (_state_to_vec, random_initial_state,
                                             rotor_force_limit, sample_params)

        def ref_state(t, L, dt, center, radius=1.2, freq=0.15):
            out = np.zeros((L, 18)); out[:, 6:15] = np.eye(3).reshape(9)
            for k in range(L):
                tk = t + (k + 1) * dt; w = 2 * np.pi * freq
                out[k, POS] = center + np.array(
                    [radius * np.cos(w * tk) - radius, radius * np.sin(w * tk), 0.0])
                out[k, 3:6] = np.array(
                    [-radius * w * np.sin(w * tk), radius * w * np.cos(w * tk), 0.0])
            return out

        # one domain, one start, shared by all models
        rng = np.random.default_rng(int(seed))
        params = sample_params(rng)
        f_max = rotor_force_limit(params)
        a_hover = params["mass"] * 9.81 / 4
        center = np.array([0.0, 0.0, 1.5]); dt = 0.05
        init0 = random_initial_state(rng, center=center)
        n_steps = int(seconds / dt)

        thrust_max = 4 * f_max
        results = {}
        with st.spinner(f"flying {len(chosen) + len(chosen_ctrl)} controllers…"):
            for lab in chosen:
                model = models[lab]
                ctbr = meta[lab]["action_mode"] == "ctbr"
                if ctbr:
                    # real-drone cascaded control: command [thrust, body-rates],
                    # RotorPy's inner rate loop drives the motors.
                    veh = Multirotor(params, control_abstraction="cmd_ctbr", aero=True)
                    hover = np.array([params["mass"] * 9.81, 0.0, 0.0, 0.0])
                    mppi = MPPIController(
                        model, horizon=min(15, meta[lab]["T"]), samples=int(samples),
                        sigma=torch.tensor([float(sigma), 1.5, 1.5, 1.5]),
                        action_low=torch.tensor([0.0, -12., -12., -12.]),
                        action_high=torch.tensor([thrust_max, 12., 12., 12.]),
                        temperature=1.0, noise_smooth=float(smooth), ki=float(ki), sampler=sampler)
                    apply = lambda st, a: veh.step(
                        st, {"cmd_thrust": float(a[0]), "cmd_w": a[1:].numpy()}, dt)
                else:
                    veh = Multirotor(params, control_abstraction="cmd_motor_thrusts", aero=True)
                    hover = np.full(4, a_hover)
                    mppi = MPPIController(model, horizon=min(15, meta[lab]["T"]),
                                         samples=int(samples), sigma=float(sigma),
                                         f_max=f_max, temperature=1.0,
                                         noise_smooth=float(smooth), ki=float(ki), sampler=sampler)
                    apply = lambda st, a: veh.step(
                        st, {"cmd_motor_thrusts": a.numpy().clip(0, f_max)}, dt)
                mppi.nominal = torch.tensor(hover, dtype=torch.float32).expand(mppi.T, 4).clone()
                stt = copy.deepcopy(init0)
                x0 = _state_to_vec(stt)
                hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)
                ahist = torch.tensor(np.tile(hover, (H, 1)), dtype=torch.float32)
                P, R, diverged = [], [], None
                for i in range(n_steps):
                    ref = torch.tensor(ref_state(i * dt, T, dt, center), dtype=torch.float32)
                    a = mppi.step(hist, ahist, ref)
                    stt = apply(stt, a)
                    x = _state_to_vec(stt)
                    if not np.isfinite(x).all():
                        diverged = i; break
                    hist = torch.cat([hist[1:], torch.tensor(x, dtype=torch.float32)[None]], 0)
                    ahist = torch.cat([ahist[1:], a[None]], 0)
                    P.append(x[POS]); R.append(ref[0, POS].numpy())
                results[lab] = (np.array(P), np.array(R), diverged)

            # ---- race-only controllers (RL policy / classical PID / grad-MPC) ----
            for lab in chosen_ctrl:
                kind = CONTROLLERS[lab][0]
                if kind == "gradmpc":
                    from drone_jepa.control.gradient_mpc import GradientMPC
                    gm = models.get("CTBR JEPA (2x)")
                    if gm is None:
                        continue
                    veh = Multirotor(params, control_abstraction="cmd_ctbr", aero=True)
                    hover = np.array([params["mass"] * 9.81, 0., 0., 0.])
                    ctrl = GradientMPC(gm, horizon=min(15, gm.T), iters=12, lr=0.3,
                                       ki=float(ki),
                                       action_low=torch.tensor([0., -12, -12, -12]),
                                       action_high=torch.tensor([thrust_max, 12., 12., 12.]))
                    ctrl.nominal = torch.tensor(hover, dtype=torch.float32).expand(ctrl.T, 4).clone()
                    stt = copy.deepcopy(init0); x0 = _state_to_vec(stt)
                    hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)
                    ahist = torch.tensor(np.tile(hover, (H, 1)), dtype=torch.float32)
                    P, R, diverged = [], [], None
                    for i in range(n_steps):
                        ref = torch.tensor(ref_state(i * dt, T, dt, center), dtype=torch.float32)
                        a = ctrl.step(hist, ahist, ref)
                        stt = veh.step(stt, {"cmd_thrust": float(a[0]), "cmd_w": a[1:].numpy()}, dt)
                        x = _state_to_vec(stt)
                        if not np.isfinite(x).all():
                            diverged = i; break
                        hist = torch.cat([hist[1:], torch.tensor(x, dtype=torch.float32)[None]], 0)
                        ahist = torch.cat([ahist[1:], a[None]], 0)
                        P.append(x[POS]); R.append(ref[0, POS].numpy())
                    results[lab] = (np.array(P), np.array(R), diverged)
                    continue
                if kind == "rl":
                    from drone_jepa.rl.policy import RLController
                    from drone_jepa.rl.env import PREVIEW
                    ctrl = RLController("artifacts/rl_policy.zip",
                                        "artifacts/rl_policy_vecnorm.pkl", f_max)
                    veh = Multirotor(params, control_abstraction="cmd_motor_thrusts", aero=True)
                    apply = lambda st, a: veh.step(st, {"cmd_motor_thrusts": np.clip(a, 0, f_max)}, dt)
                else:
                    from drone_jepa.control.pid import PIDController
                    PREVIEW = 1
                    ctrl = PIDController()
                    veh = Multirotor(params, control_abstraction="cmd_ctbr", aero=True)
                    apply = lambda st, a: veh.step(
                        st, {"cmd_thrust": float(np.clip(a[0], 0, thrust_max)),
                             "cmd_w": np.clip(a[1:], -12, 12)}, dt)
                stt = copy.deepcopy(init0)
                P, R, diverged = [], [], None
                for i in range(n_steps):
                    x = _state_to_vec(stt)
                    fut = ref_state(i * dt, T, dt, center)
                    if kind == "rl":
                        a = ctrl.act(x, [fut[k, POS] for k in range(PREVIEW)])
                    else:
                        a = ctrl.act(x, fut[0, POS], fut[0, 3:6])
                    stt = apply(stt, a)
                    xx = _state_to_vec(stt)
                    if not np.isfinite(xx).all():
                        diverged = i; break
                    P.append(xx[POS]); R.append(fut[0, POS])
                results[lab] = (np.array(P), np.array(R), diverged)

        Rf = next((r for _, r, _ in results.values() if len(r) > 1), None)
        st.markdown(f"**Domain:** mass={params['mass']:.2f} kg, f_max/rotor={f_max:.1f} N")
        # RMSE table
        tcols = st.columns(len(results))
        for col, (lab, (P, R, dv)) in zip(tcols, results.items()):
            with col:
                if len(P) < 2:
                    st.metric(lab, "diverged early")
                else:
                    rmse = np.sqrt(((P - R) ** 2).sum(1).mean())
                    note = f"diverged @{dv*0.05:.1f}s" if dv else "stable"
                    st.metric(f"{lab} RMSE", f"{rmse:.2f} m", note)

        span = 3.0
        for P, _, _ in results.values():
            if len(P) > 1:
                span = max(span, np.abs(P - center).max())
        span = float(span) * 1.1
        fig = go.Figure()
        if Rf is not None:
            fig.add_trace(go.Scatter3d(x=Rf[:, 0], y=Rf[:, 1], z=Rf[:, 2], mode="lines",
                          line=dict(color="black", dash="dash"), name="reference"))
        for lab, (P, R, dv) in results.items():
            if len(P) > 1:
                fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines",
                              line=dict(color=meta[lab]["color"], width=5), name=lab))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                          scene=dict(aspectmode="cube",
                                     xaxis=dict(title="x [m]", range=[center[0]-span, center[0]+span]),
                                     yaxis=dict(title="y [m]", range=[center[1]-span, center[1]+span]),
                                     zaxis=dict(title="z [m]", range=[center[2]-span, center[2]+span])),
                          legend=dict(orientation="h", y=-0.02))
        st.plotly_chart(fig, use_container_width=True)
