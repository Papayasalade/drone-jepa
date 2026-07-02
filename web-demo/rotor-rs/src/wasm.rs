//! Browser bindings (`--features wasm`). Wraps the scalar-`f64` dynamics in a
//! single advancing "ground truth" drone for the web demo.
//!
//! The control abstraction is chosen per-call (`step_rotor_force` / `step_ctbr`)
//! rather than via the type-level generic, so JS can switch at runtime off one
//! mutable `QuadParams`. Dynamics params are mutable live (`set_mass`, `set_wind`)
//! for the demo's "break it" knobs.

use serde::Deserialize;
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

use crate::control::{Ctbr, CtbrCmd, ControlLaw, RotorForce};
use crate::linalg::{Quat, Vec3};
use crate::multirotor::{clip_speeds, integrate};
use crate::params::{QuadParams, QuadParamsInput, GRAVITY, NUM_ROTORS};
use crate::se3::{FlatRef, Se3Control};
use crate::state::State;

/// JS-facing parameter object. Every field optional (`#[serde(default)]` +
/// hummingbird defaults), so JS can pass `{}` for a stock drone or
/// `{ mass: 0.8, kEta: 7e-6 }` to override just a few. camelCase keys.
#[derive(Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct WasmParams {
    mass: f64,
    ixx: f64,
    iyy: f64,
    izz: f64,
    ixy: f64,
    iyz: f64,
    ixz: f64,
    rotor_pos: [[f64; 3]; NUM_ROTORS],
    rotor_directions: [f64; NUM_ROTORS],
    c_dx: f64,
    c_dy: f64,
    c_dz: f64,
    k_eta: f64,
    k_m: f64,
    k_d: f64,
    k_z: f64,
    k_h: f64,
    k_flap: f64,
    tau_m: f64,
    rotor_speed_min: f64,
    rotor_speed_max: f64,
    k_w: f64,
}

impl Default for WasmParams {
    fn default() -> Self {
        // AscTec Hummingbird (rotorpy/vehicles/hummingbird_params.py)
        let d = 0.17 * std::f64::consts::FRAC_1_SQRT_2;
        Self {
            mass: 0.5,
            ixx: 3.65e-3,
            iyy: 3.68e-3,
            izz: 7.03e-3,
            ixy: 0.0,
            iyz: 0.0,
            ixz: 0.0,
            rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
            rotor_directions: [1.0, -1.0, 1.0, -1.0],
            c_dx: 0.5e-2,
            c_dy: 0.5e-2,
            c_dz: 1e-2,
            k_eta: 5.57e-6,
            k_m: 1.36e-7,
            k_d: 1.19e-4,
            k_z: 2.32e-4,
            k_h: 3.39e-3,
            k_flap: 0.0,
            tau_m: 0.005,
            rotor_speed_min: 0.0,
            rotor_speed_max: 1500.0,
            k_w: 1.0,
        }
    }
}

impl WasmParams {
    fn to_input(&self) -> QuadParamsInput {
        QuadParamsInput {
            mass: self.mass,
            ixx: self.ixx, iyy: self.iyy, izz: self.izz,
            ixy: self.ixy, iyz: self.iyz, ixz: self.ixz,
            rotor_pos: self.rotor_pos,
            rotor_directions: self.rotor_directions,
            c_dx: self.c_dx, c_dy: self.c_dy, c_dz: self.c_dz,
            k_eta: self.k_eta, k_m: self.k_m, k_d: self.k_d, k_z: self.k_z,
            k_h: self.k_h, k_flap: self.k_flap, tau_m: self.tau_m,
            rotor_speed_min: self.rotor_speed_min,
            rotor_speed_max: self.rotor_speed_max,
            k_w: self.k_w,
        }
    }
}

fn hover_rpm(input: &QuadParamsInput) -> f64 {
    (input.mass * GRAVITY / (4.0 * input.k_eta)).sqrt()
}

#[wasm_bindgen]
pub struct WasmDrone {
    input: QuadParamsInput,
    params: QuadParams<f64>,
    state: State<f64>,
    n_sub: usize,
}

#[wasm_bindgen]
impl WasmDrone {
    /// `params`: a JS object (any subset of the camelCase fields; omitted ones
    /// default to the hummingbird). `n_sub`: fixed RK4 substeps per `step`.
    #[wasm_bindgen(constructor)]
    pub fn new(params: JsValue, n_sub: usize) -> Result<WasmDrone, JsValue> {
        let wp: WasmParams = if params.is_undefined() || params.is_null() {
            WasmParams::default()
        } else {
            serde_wasm_bindgen::from_value(params)?
        };
        let input = wp.to_input();
        let drone = WasmDrone {
            params: QuadParams::from_input(&input),
            state: hover_state(&input, 0.0, 0.0, 1.5),
            input,
            n_sub: n_sub.max(1),
        };
        Ok(drone)
    }

    /// Reset to a hover state at (x, y, z) with rotors at hover speed, level attitude.
    pub fn reset_hover(&mut self, x: f64, y: f64, z: f64) {
        self.state = hover_state(&self.input, x, y, z);
    }

    /// Advance one control step `dt` with four rotor forces [N].
    pub fn step_rotor_force(&mut self, forces: &[f64], dt: f64) {
        let f = [forces[0], forces[1], forces[2], forces[3]];
        let raw = RotorForce::cmd_rotor_speeds(&self.params, &self.state, &f);
        let speeds = clip_speeds(&self.params, raw);
        self.state = integrate(&self.params, &self.state, &speeds, dt, self.n_sub);
    }

    /// Advance one control step `dt` with collective thrust [N] + body rates [rad/s].
    pub fn step_ctbr(&mut self, thrust: f64, wx: f64, wy: f64, wz: f64, dt: f64) {
        let cmd = CtbrCmd { thrust, w_cmd: Vec3::new(wx, wy, wz) };
        let raw = Ctbr::cmd_rotor_speeds(&self.params, &self.state, &cmd);
        let speeds = clip_speeds(&self.params, raw);
        self.state = integrate(&self.params, &self.state, &speeds, dt, self.n_sub);
    }

    /// Current state, flat: [px,py,pz, vx,vy,vz, qi,qj,qk,qw, wx,wy,wz, r0,r1,r2,r3] (17).
    pub fn state(&self) -> Vec<f64> {
        let s = &self.state;
        let q = s.q.to_array();
        vec![
            s.x.x, s.x.y, s.x.z,
            s.v.x, s.v.y, s.v.z,
            q[0], q[1], q[2], q[3],
            s.w.x, s.w.y, s.w.z,
            s.rotor_speeds[0], s.rotor_speeds[1], s.rotor_speeds[2], s.rotor_speeds[3],
        ]
    }

    /// State in the SkyJEPA canonical 18-dim layout: [pos3, vel3, R(row-major 9), omega3].
    /// Feed this straight into the JEPA model's encoder.
    pub fn state_jepa(&self) -> Vec<f64> {
        let s = &self.state;
        let r = s.q.to_rotmat();
        vec![
            s.x.x, s.x.y, s.x.z,
            s.v.x, s.v.y, s.v.z,
            r.rows[0][0], r.rows[0][1], r.rows[0][2],
            r.rows[1][0], r.rows[1][1], r.rows[1][2],
            r.rows[2][0], r.rows[2][1], r.rows[2][2],
            s.w.x, s.w.y, s.w.z,
        ]
    }

    /// Overwrite the full state from a 17-float array (same layout as `state()`).
    pub fn set_state(&mut self, a: &[f64]) {
        self.state = State {
            x: Vec3::new(a[0], a[1], a[2]),
            v: Vec3::new(a[3], a[4], a[5]),
            q: Quat::new(a[6], a[7], a[8], a[9]).normalized(),
            w: Vec3::new(a[10], a[11], a[12]),
            wind: self.state.wind,
            rotor_speeds: [a[13], a[14], a[15], a[16]],
        };
    }

    /// Set the (constant) wind vector [m/s] — the "blow it around" knob.
    pub fn set_wind(&mut self, x: f64, y: f64, z: f64) {
        self.state.wind = Vec3::new(x, y, z);
    }

    /// Change mass live (the "double the weight / drop a payload" knob); rebuilds
    /// derived params (weight). Other params can be changed via `set_params`.
    pub fn set_mass(&mut self, mass: f64) {
        self.input.mass = mass;
        self.params = QuadParams::from_input(&self.input);
    }

    /// Replace dynamics params from a JS object (any subset; omitted -> hummingbird
    /// default, NOT the current value). Keeps the current kinematic state.
    pub fn set_params(&mut self, params: JsValue) -> Result<(), JsValue> {
        let wp: WasmParams = serde_wasm_bindgen::from_value(params)?;
        self.input = wp.to_input();
        self.params = QuadParams::from_input(&self.input);
        Ok(())
    }

    /// Hover rotor speed [rad/s] for the current params (handy for building actions).
    pub fn hover_rpm(&self) -> f64 {
        hover_rpm(&self.input)
    }

    /// Per-rotor hover force [N] (= mass*g/4) — a convenient nominal rotor-force action.
    pub fn hover_force(&self) -> f64 {
        self.input.mass * GRAVITY / 4.0
    }
}

fn hover_state(input: &QuadParamsInput, x: f64, y: f64, z: f64) -> State<f64> {
    State {
        x: Vec3::new(x, y, z),
        v: Vec3::zero(),
        q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(),
        wind: Vec3::zero(),
        rotor_speeds: [hover_rpm(input); NUM_ROTORS],
    }
}

fn state_to_jepa18(s: &State<f64>) -> [f64; 18] {
    let r = s.q.to_rotmat();
    [
        s.x.x, s.x.y, s.x.z,
        s.v.x, s.v.y, s.v.z,
        r.rows[0][0], r.rows[0][1], r.rows[0][2],
        r.rows[1][0], r.rows[1][1], r.rows[1][2],
        r.rows[2][0], r.rows[2][1], r.rows[2][2],
        s.w.x, s.w.y, s.w.z,
    ]
}

fn state_vec17(s: &State<f64>) -> Vec<f64> {
    let q = s.q.to_array();
    vec![
        s.x.x, s.x.y, s.x.z,
        s.v.x, s.v.y, s.v.z,
        q[0], q[1], q[2], q[3],
        s.w.x, s.w.y, s.w.z,
        s.rotor_speeds[0], s.rotor_speeds[1], s.rotor_speeds[2], s.rotor_speeds[3],
    ]
}

/// MUST stay identical to `terrainHeight` in web/forecast.js (that copy only
/// renders; THIS one is what the sim's clearance floor + lift boost act on).
/// Rolling hills + a diagonal ridge + fine detail; peaks ~3 m, so the path's
/// low passes (z ~4.4 m against floor = h + 2.5) force visible terrain-hugging
/// pops over the ridge — the interesting part.
fn forecast_terrain_height(x: f64, y: f64) -> f64 {
    let ridge = y - 0.35 * x - 3.5;
    -0.4
        + 1.3 * (0.10 * x + 0.4).sin() * (0.085 * y - 0.8).cos()
        + 0.7 * (0.23 * x).sin() * (0.19 * y + 1.1).sin()
        + 1.2 * (-(ridge * ridge) / 22.0).exp()
        + 0.25 * (0.55 * x + 0.3 * y).sin()
}

fn forecast_ref(t: f64) -> FlatRef {
    // An aerobatic-but-smooth showcase path: two harmonics per axis (banked
    // figure-eight with real altitude swings). Peak speed ~4.3 m/s == the p95
    // of the model's training references, so at speed 1.0 the ghost is
    // in-distribution while genuinely maneuvering; the demo's speed knob can
    // push it beyond (watch the forecast error grow). Closed-form derivatives
    // because the SE3 controller consumes the full flat reference.
    #[inline]
    fn axis(t: f64, a1: f64, w1: f64, p1: f64, a2: f64, w2: f64, p2: f64) -> (f64, f64, f64) {
        let (s1, c1) = (w1 * t + p1).sin_cos();
        let (s2, c2) = (w2 * t + p2).sin_cos();
        (a1 * s1 + a2 * s2,
         a1 * w1 * c1 + a2 * w2 * c2,
         -a1 * w1 * w1 * s1 - a2 * w2 * w2 * s2)
    }
    let (x, xd, xdd) = axis(t, 5.5, 0.28, 0.0, 1.8, 0.90, 1.3);
    let (y, yd, ydd) = axis(t, 4.5, 0.45, 0.7, 1.5, 0.75, 0.0);
    let (dz, zd, zdd) = axis(t, 1.8, 0.35, 0.0, 0.8, 0.60, 2.0);
    FlatRef {
        x: Vec3::new(x, y, 7.0 + dz),
        x_dot: Vec3::new(xd, yd, zd),
        x_ddot: Vec3::new(xdd, ydd, zdd),
        yaw: 0.9 * (0.18 * t).sin(),
        yaw_dot: 0.9 * 0.18 * (0.18 * t).cos(),
    }
}

fn enforce_forecast_clearance(mut state: State<f64>) -> State<f64> {
    let floor = forecast_terrain_height(state.x.x, state.x.y) + 2.5;
    if state.x.z < floor {
        state.x.z = floor;
        state.v.z = state.v.z.max(0.0);
    }
    state
}

#[cfg(feature = "jepa")]
#[derive(Clone, Copy)]
struct ForecastFrame {
    state: State<f64>,
    action: [f64; 4],
    t: f64,
}

#[cfg(feature = "jepa")]
const FORECAST_AHEAD_STEPS: usize = 400; // 20 s at 20 Hz.

#[cfg(feature = "jepa")]
#[wasm_bindgen]
pub struct WasmForecast {
    input: QuadParamsInput,
    reality: Multirotor<f64, RotorForce>,
    se3: Se3Control,
    model: SkyJepaLite,
    state: State<f64>,
    rng: Rng,
    t: f64,
    dt: f64,
    states: VecDeque<[f64; 18]>,
    actions: VecDeque<[f64; 4]>,
    future: VecDeque<ForecastFrame>,
    base_mass: f64,
    // reference time-warp (the demo's speed knob). Phase-preserving so a live
    // speed change bends the path's tempo without teleporting the target:
    // warped(t) = ref_phase + ref_speed * (t - ref_t0).
    ref_speed: f64,
    ref_phase: f64,
    ref_t0: f64,
}

#[cfg(feature = "jepa")]
#[wasm_bindgen]
impl WasmForecast {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: f64) -> WasmForecast {
        let seed = seed as u64;
        let mut rng = Rng::new(seed | 1);
        let input = sample_forecast_drone(&mut rng);
        let r0 = forecast_ref(0.0);
        let state = hover_state(&input, r0.x.x, r0.x.y, r0.x.z);
        let mut out = WasmForecast {
            reality: Multirotor::with_substeps(&input, 8),
            se3: Se3Control::new(&input),
            model: SkyJepaLite::from_blob(FORECAST_JEPA_BLOB),
            state,
            rng,
            t: 0.0,
            dt: 0.05,
            states: VecDeque::new(),
            actions: VecDeque::new(),
            future: VecDeque::new(),
            base_mass: input.mass,
            input,
            ref_speed: 1.0,
            ref_phase: 0.0,
            ref_t0: 0.0,
        };
        out.prime_history();
        out.fill_future(FORECAST_AHEAD_STEPS);
        out
    }

    fn warped(&self, t: f64) -> f64 {
        self.ref_phase + self.ref_speed * (t - self.ref_t0)
    }

    /// The demo's tempo knob (0.4 = lazy cruise, 1.0 = training-p95 pace,
    /// >1.2 = beyond the training envelope — the forecast visibly degrades).
    /// Phase-preserving, and the committed future is replanned immediately.
    pub fn set_speed(&mut self, k: f64) {
        self.ref_phase = self.warped(self.t);
        self.ref_t0 = self.t;
        self.ref_speed = k.clamp(0.25, 2.0);
        self.future.clear();
        self.fill_future(FORECAST_AHEAD_STEPS);
    }

    fn action_at(&self, state: &State<f64>, t: f64) -> [f64; 4] {
        let r = forecast_ref(self.warped(t));
        let raw = self.se3.update(state, &r);
        let f_max = self.input.k_eta * self.input.rotor_speed_max * self.input.rotor_speed_max;
        let terrain_floor = forecast_terrain_height(state.x.x, state.x.y) + 3.5;
        let low_err = (terrain_floor - state.x.z).max(0.0);
        let down_vel = if state.x.z < terrain_floor + 1.0 { (-state.v.z).max(0.0) } else { 0.0 };
        let lift_boost = self.input.mass * (4.5 * low_err + 2.8 * down_vel) / 4.0;
        core::array::from_fn(|i| (raw[i] + lift_boost).clamp(0.0, f_max))
    }

    fn reset_state(&mut self) {
        let r0 = forecast_ref(0.0);
        self.state = hover_state(&self.input, r0.x.x, r0.x.y, r0.x.z);
        self.t = 0.0;
        self.ref_phase = 0.0;
        self.ref_t0 = 0.0;
        self.future.clear();
        self.prime_history();
        self.fill_future(FORECAST_AHEAD_STEPS);
    }

    fn rebuild(&mut self) {
        self.reality = Multirotor::with_substeps(&self.input, 8);
        self.se3 = Se3Control::new(&self.input);
    }

    fn prime_history(&mut self) {
        self.states.clear();
        self.actions.clear();
        let a = self.action_at(&self.state, self.t);
        for _ in 0..self.model.h {
            self.states.push_back(state_to_jepa18(&self.state));
        }
        for _ in 0..self.model.h - 1 {
            self.actions.push_back(a);
        }
    }

    fn push_history(&mut self, state: State<f64>, action: [f64; 4]) {
        self.states.push_back(state_to_jepa18(&state));
        while self.states.len() > self.model.h {
            self.states.pop_front();
        }
        self.actions.push_back(action);
        while self.actions.len() > self.model.h - 1 {
            self.actions.pop_front();
        }
    }

    fn fill_future(&mut self, steps: usize) {
        let mut tail_state = self.future.back().map(|f| f.state).unwrap_or(self.state);
        let mut tail_t = self.future.back().map(|f| f.t).unwrap_or(self.t);
        while self.future.len() < steps {
            let action = self.action_at(&tail_state, tail_t);
            tail_state = self.reality.step(&tail_state, &action, self.dt);
            tail_state = enforce_forecast_clearance(tail_state);
            tail_t += self.dt;
            self.future.push_back(ForecastFrame { state: tail_state, action, t: tail_t });
        }
    }

    /// Advance one 20 Hz playback step through the committed simulated trajectory.
    pub fn step(&mut self) -> Vec<f64> {
        self.fill_future(FORECAST_AHEAD_STEPS);
        let frame = self.future.pop_front().expect("future buffer is primed");
        self.state = frame.state;
        self.t = frame.t;
        self.push_history(self.state, frame.action);
        self.fill_future(FORECAST_AHEAD_STEPS);
        state_vec17(&self.state)
    }

    pub fn state(&self) -> Vec<f64> {
        state_vec17(&self.state)
    }

    /// Forecast format: [committed-future xyz * N, JEPA xyz * N, naive-AR xyz * N].
    pub fn forecast(&mut self, horizon_steps: usize) -> Vec<f64> {
        let n = horizon_steps.clamp(1, self.model.t);
        self.fill_future(FORECAST_AHEAD_STEPS.max(self.model.t));
        let future_actions: Vec<[f64; 4]> = self
            .future
            .iter()
            .take(self.model.t)
            .map(|f| f.action)
            .collect();

        let hist: Vec<[f64; 18]> = self.states.iter().copied().collect();
        let mut aw: Vec<[f64; 4]> = self.actions.iter().copied().collect();
        aw.extend_from_slice(&future_actions);
        let last = *aw.last().unwrap_or(&[self.input.mass * GRAVITY / 4.0; 4]);
        while aw.len() < self.model.h + self.model.t {
            aw.push(last);
        }
        let pred = self.model.predict_batch(&[hist], &[aw])[0].clone();

        let cur = state_to_jepa18(&self.state);
        let prev = self
            .states
            .iter()
            .rev()
            .nth(1)
            .copied()
            .unwrap_or(cur);
        let vx = self.state.v;
        let ax = Vec3::new(
            (cur[3] - prev[3]) / self.dt,
            (cur[4] - prev[4]) / self.dt,
            (cur[5] - prev[5]) / self.dt,
        );
        let mut ar = Vec::with_capacity(n);
        for k in 1..=n {
            let h = k as f64 * self.dt;
            ar.push(self.state.x + vx.scale(h) + ax.scale(0.5 * h * h));
        }

        let mut out = Vec::with_capacity(n * 9);
        for f in self.future.iter().take(n) {
            out.extend_from_slice(&[f.state.x.x, f.state.x.y, f.state.x.z]);
        }
        for p in pred.into_iter().take(n) {
            out.extend_from_slice(&[p[0], p[1], p[2]]);
        }
        for p in &ar {
            out.extend_from_slice(&[p.x, p.y, p.z]);
        }
        out
    }

    pub fn set_wind(&mut self, x: f64, y: f64, z: f64) {
        self.state.wind = Vec3::new(x, y, z);
        self.future.clear();
        self.fill_future(FORECAST_AHEAD_STEPS);
    }

    pub fn gust(&mut self) {
        self.set_wind(4.0, -1.5, 0.0);
    }

    pub fn calm(&mut self) {
        self.set_wind(0.0, 0.0, 0.0);
    }

    pub fn reference(&self) -> Vec<f64> {
        let r = forecast_ref(self.warped(self.t));
        vec![r.x.x, r.x.y, r.x.z]
    }

    pub fn toggle_payload(&mut self) {
        let heavy = self.input.mass < self.base_mass * 1.2;
        self.input.mass = if heavy { self.base_mass * 1.45 } else { self.base_mass };
        self.rebuild();
        self.future.clear();
        self.fill_future(FORECAST_AHEAD_STEPS);
    }

    pub fn new_drone(&mut self) {
        let input = sample_forecast_drone(&mut self.rng);
        self.input = input;
        self.base_mass = input.mass;
        self.rebuild();
        self.reset_state();
    }

    pub fn drone_mass(&self) -> f64 {
        self.input.mass
    }

    pub fn drone_arm(&self) -> f64 {
        self.input.rotor_pos[0][0] * std::f64::consts::SQRT_2
    }

    /// Whether the payload toggle currently has the extra mass attached
    /// (drives the payload visual in the demo).
    pub fn payload_attached(&self) -> bool {
        self.input.mass > self.base_mass * 1.2
    }

    /// Current wind vector [m/s] (drives the wind-streak animation).
    pub fn wind(&self) -> Vec<f64> {
        let w = self.state.wind;
        vec![w.x, w.y, w.z]
    }

    pub fn time(&self) -> f64 {
        self.t
    }
}

// --------------------------------------------------------------------------- //
// WasmRacer: an autonomous MPPI gate racer (true dynamics) for the browser.
// Step 1 of the web demo — a self-flying drone you can perturb, no model needed.
// --------------------------------------------------------------------------- //
use crate::gates::{Course, Gate};
use crate::jepa::{JepaRollout, SkyJepaLite};
use crate::mppi::{Controller, MppiConfig, MppiController, RolloutModel, TrueDynamics};
use crate::rl::RlPolicy;
use crate::rotor_mppi::{RotorMppiConfig, RotorMppiController};
use crate::multirotor::Multirotor;
use crate::rng::Rng;

// The demos now fly the "bigquad" (drones/bigquad.json: 1.2 kg, 25 cm arms,
// TWR 3, slow motors) — the vehicle validated end-to-end by the train_recipe
// campaign (docs/EXPERIMENTS_fragility_campaign.md, "bigquad run").
/// The ROTOR-FORCE PPO racing policy, trained on the bigquad band — same raw
/// actuator space as the rotor JEPA drone (RL_ACTION_MODE=rotor in rl-env).
const RL_BLOB: &[u8] = include_bytes!("../assets/skyrl_rotor_bigquad.rlb");
/// ONE unified rotor-force JEPA for BOTH demos: warm-retrained from the
/// DAgger winner on the union mix (MPPI base + on-policy + SE3 actions).
/// Race: 12/12 wins, 0 crashes, 0 respawns, 5.0 gates. Follow (fresh SE3
/// test set): 0.21 m @1 s — vs 0.36 m for the race-only model.
const ROTOR_JEPA_BLOB: &[u8] = include_bytes!("../assets/bigquad_unified.jblob");
/// The Forecast Ghost uses the SAME unified model.
const FORECAST_JEPA_BLOB: &[u8] = include_bytes!("../assets/bigquad_unified.jblob");

/// 5 gates sampled fully independently in a 3-D "sky" volume: each is a free draw
/// of (x, y, z) and a random orientation — a real drone-racing arena, not a slalom.
fn random_course(rng: &mut Rng) -> Course {
    let n = 5;
    let gates = (0..n)
        .map(|_| {
            let x = 1.5 + rng.uniform() * 9.0; // [1.5, 10.5]
            let y = (rng.uniform() * 2.0 - 1.0) * 3.5; // [-3.5, 3.5]
            let z = 5.0 + rng.uniform() * 10.0; // [5, 15] m — high "sky" gates
            // random orientation: a roughly-uniform unit normal (Gaussian then normalize)
            let nrm = Vec3::new(rng.normal(), rng.normal(), rng.normal());
            Gate::new(Vec3::new(x, y, z), nrm, 0.85)
        })
        .collect();
    Course::new(gates, false) // non-looping: it can be "won"
}

// The MPPI rollout model needs fine substeps for the stiff k_w=16 rate loop,
// else every rollout diverges to NaN and the planner is rudderless.
const ROLLOUT_NSUB: usize = 8;

/// A domain-randomized drone — same fleet the JEPA/RL models train on. Each of the
/// 4 arms gets its OWN length (5-40 cm) so frames are genuinely asymmetric. Motor
/// strength (`k_eta`) is DERIVED from a sampled thrust-to-weight ratio so every
/// drone is guaranteed flyable (no heavy-with-weak-motors combos), and inertia
/// scales physically with mass·arm². (Per-motor power is not randomized — it's not
/// visible and is ~equivalent to clamping all rotors to the weakest.)
fn sample_drone(r: &mut Rng) -> (QuadParamsInput, [f64; NUM_ROTORS]) {
    let base = WasmParams::default().to_input();
    let lerp = |r: &mut Rng, a: f64, b: f64| a + (b - a) * r.uniform();
    // PAPER-WIDTH distribution (symmetric frames, ±~25% mass) centered on the
    // BIGQUAD — exactly the family the embedded JEPA/RL models trained on
    // (gen_dataset with MASS 0.9-1.5, ARM 0.2-0.3). Wider fleets are the regime
    // where a ~9.5K world model stops ranking plans honestly (see
    // docs/EXPERIMENTS_jepa_fleet.md); RL tolerates width, JEPA-MPPI does not.
    let arm = lerp(r, 0.20, 0.30); // symmetric arms
    let arms: [f64; 4] = [arm; 4];
    let d: [f64; 4] = core::array::from_fn(|i| arms[i] * std::f64::consts::FRAC_1_SQRT_2);
    let rotor_pos = [[d[0], d[0], 0.0], [d[1], -d[1], 0.0], [-d[2], -d[2], 0.0], [-d[3], d[3], 0.0]];
    let avg_arm = arms.iter().sum::<f64>() / 4.0;
    let mass = lerp(r, 0.9, 1.5);
    // k_eta from thrust-to-weight in [2,4] -> always enough thrust to fly/race
    let twr = lerp(r, 2.0, 4.0);
    let rpm2 = base.rotor_speed_max * base.rotor_speed_max;
    let k_eta = twr * mass * GRAVITY / (4.0 * rpm2);
    let k_m = k_eta * (base.k_m / base.k_eta) * lerp(r, 0.7, 1.3);
    let i_scale = (mass / 0.5) * (avg_arm / 0.17).powi(2) * lerp(r, 0.7, 1.3); // I ∝ m·r²
    let input = QuadParamsInput {
        mass,
        ixx: base.ixx * i_scale, iyy: base.iyy * i_scale, izz: base.izz * i_scale,
        ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos,
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: lerp(r, 0.02, 0.30), c_dy: lerp(r, 0.02, 0.30), c_dz: lerp(r, 0.05, 0.40),
        k_eta, k_m,
        k_d: base.k_d, k_z: base.k_z, k_h: base.k_h, k_flap: 0.0,
        tau_m: lerp(r, 0.01, 0.04),
        rotor_speed_min: 0.0, rotor_speed_max: base.rotor_speed_max,
        k_w: lerp(r, 8.0, 18.0),
    };
    (input, [1.0; NUM_ROTORS])
}

/// Forecast demo randomization: still changes the drone, but keeps it close to
/// the CTBR model's training regime. The race demo intentionally stresses JEPA;
/// this demo is meant to make the prediction skill legible.
fn sample_forecast_drone(r: &mut Rng) -> QuadParamsInput {
    let base = WasmParams::default().to_input();
    let lerp = |r: &mut Rng, a: f64, b: f64| a + (b - a) * r.uniform();
    // The unified model's TRAINING family around the bigquad: mass spans the
    // gen_dataset_rf band (1.2 x [0.7, 1.4]), thrust-to-weight 2.4-3.6 (so
    // k_eta co-varies with mass and every draw is flyable), drag and motor lag
    // across the trained ranges, and a visible +-12% arm spread. "new drone"
    // should FEEL different — a 0.85 kg nimble frame vs a 1.7 kg barge.
    let arm = lerp(r, 0.22, 0.28);
    let d = arm * std::f64::consts::FRAC_1_SQRT_2;
    let mass = lerp(r, 0.85, 1.70);
    let twr = lerp(r, 2.4, 3.6);
    let rpm2 = base.rotor_speed_max * base.rotor_speed_max;
    let k_eta = twr * mass * GRAVITY / (4.0 * rpm2);
    let i_scale = (mass / 0.5) * (arm / 0.17).powi(2) * lerp(r, 0.85, 1.15);
    QuadParamsInput {
        mass,
        ixx: base.ixx * i_scale,
        iyy: base.iyy * i_scale,
        izz: base.izz * i_scale,
        ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: base.rotor_directions,
        c_dx: lerp(r, 0.05, 0.25),
        c_dy: lerp(r, 0.05, 0.25),
        c_dz: lerp(r, 0.10, 0.35),
        k_eta,
        k_m: k_eta * (9.6e-8 / 3.92e-6) * lerp(r, 0.85, 1.15),
        k_d: base.k_d,
        k_z: base.k_z,
        k_h: base.k_h,
        k_flap: 0.0,
        tau_m: lerp(r, 0.015, 0.03),
        rotor_speed_min: base.rotor_speed_min,
        rotor_speed_max: base.rotor_speed_max,
        k_w: 16.0,
    }
}

/// Racing cost: drive DECISIVELY through gates. The single most important thing is
/// a STRONG velocity-toward-gate drive (`w_vel`) — a weak one makes MPPI's averaged
/// command wash out and the drone *meanders / looks undecided*. Critically `w_pos`
/// stays 0 here: a position spring cancels the forward drive and reintroduces the
/// dithering (it belongs in `hover_cfg`, not racing). We only lightly smooth the
/// motion: a touch of `w_vdamp` to cap top speed, slightly calmer rate noise. This
/// clears courses ~2× faster than an over-damped cost (4.5 s vs 9 s) at 8/8 wins.
fn race_cfg(mass: f64) -> MppiConfig {
    let mut cfg = MppiConfig::for_mass(mass);
    cfg.samples = 128; // browser-friendly
    cfg.w_vel = 3.0; // strong, decisive drive toward the gate (stock 2.0)
    cfg.w_pos = 0.0; // NO position spring while racing (it causes dithering)
    cfg.w_vdamp = 0.04; // light damping: just caps top speed
    cfg.w_effort = 0.01; // mild — do NOT over-penalise the tilts racing needs
    cfg.sigma_rate = 2.8; // slightly calmer rate exploration (was 4.0)
    cfg.rate_max = 10.0; // modestly capped aggression (was 12.0)
    cfg.beta = 0.90; // a bit more temporally-correlated noise → smoother plans
    cfg.w_floor = 300.0; // never plan into the ground (sim has no floor collision)
    cfg
}

/// Hover cost: HOLD a point. A QUADRATIC position spring (`w_pos·dist²`) + velocity
/// damping (`w_vdamp·|v|²`) = a textbook PD that settles to REST. We drop `w_vel`
/// (rewarding velocity toward a fixed point is negative damping → it orbits) and
/// `gate_reward` (no incentive to fly *through* the hold point and drift off).
/// Crucially we also CALM the MPPI exploration (small `sigma`, high `lambda`): at a
/// hover equilibrium the executed action is a softmax-mean of sampled actions, so
/// loud exploration injects residual motion — shrinking it drops the settled speed
/// from ~1.2 to ~0.26 m/s (verified in `examples/sky_verify`).
fn hover_cfg(mass: f64) -> MppiConfig {
    let mut cfg = MppiConfig::for_mass(mass);
    cfg.samples = 128;
    cfg.w_vel = 0.0;
    cfg.gate_reward = 0.0;
    cfg.w_pos = 3.0; // quadratic spring
    cfg.w_vdamp = 0.8; // firm damping → comes to rest
    cfg.w_alt = 0.0; // w_pos already covers full 3-D distance incl. altitude
    cfg.w_terminal = 0.0; // per-step spring handles it; no linear end-pull
    cfg.rate_max = 9.0;
    cfg.sigma_rate = 0.5; // calm rate exploration (was 4.0) → no jitter at rest
    cfg.sigma_thrust = 0.03 * mass * GRAVITY; // calm thrust exploration
    cfg.lambda = 1.0; // softer softmax → smoother averaged command
    cfg
}

/// Rotor-force JEPA racer. Learned-world-model cost tuning (horizon 12, gentle
/// speed, strong floor + latent trust region) but plans in 4-independent-rotor-force
/// space — the harder mode with no inner rate loop, so we keep the per-rotor noise
/// modest and lean hard on the trust region against model exploitation.
fn rotor_race_cfg(mass: f64) -> RotorMppiConfig {
    let mut cfg = RotorMppiConfig::for_mass(mass);
    let base = &mut cfg.base;
    base.horizon = 12;
    base.samples = 48;
    base.w_vel = 0.8;
    base.w_terminal = 2.5;
    base.w_alt = 3.5;
    base.w_vdamp = 0.18;
    base.v_max = 3.5;
    base.w_speed = 1.5;
    base.w_floor = 600.0;
    base.w_effort = 1e-4; // forces are O(N²) — keep effort weight tiny
    base.trust_lambda = 1000.0;
    base.beta = 0.85;
    base.lambda = 0.4;
    cfg.sigma_force = 0.22 * cfg.hover_force; // modest per-rotor exploration
    cfg
}
fn rotor_hover_cfg(mass: f64) -> RotorMppiConfig {
    let mut cfg = rotor_race_cfg(mass);
    let base = &mut cfg.base;
    base.w_vel = 0.0;
    base.gate_reward = 0.0;
    base.w_pos = 3.0; // quadratic position spring → settle to rest
    base.w_vdamp = 0.8;
    base.w_alt = 0.0;
    base.w_terminal = 0.0;
    base.lambda = 1.0;
    cfg.sigma_force = 0.06 * cfg.hover_force; // calm at rest → no jitter
    cfg
}

#[wasm_bindgen]
pub struct WasmRacer {
    input: QuadParamsInput,
    reality: Multirotor<f64, Ctbr>,
    dt: f64,
    rng: Rng,
    seed: u64, // the track seed (reproduces the whole course sequence)
    // true-dynamics drone (classical MPPI over the perfect sim)
    ctrl: MppiController<TrueDynamics>,
    course: Course,
    state: State<f64>,
    race: MppiConfig, // live-tunable racing cost (survives race↔hover swaps)
    wins: usize,
    done: bool, // finished the course (now holding at the finish)
    // RL drone (reactive PPO policy — no model, no planning), same course
    rl_ctrl: RlPolicy,
    rl_course: Course,
    rl_state: State<f64>,
    rl_wins: usize,
    rl_done: bool,
    rl_respawns: usize,
    // Rotor-force JEPA drone (MPPI over 4 independent rotor forces), same course.
    // Its own RotorForce reality (the others step CTBR); the JEPA/RL models don't
    // see the per-rotor gain disturbance, same as the other learned drones.
    reality_rf: Multirotor<f64, RotorForce>,
    rf_ctrl: RotorMppiController,
    rf_course: Course,
    rf_state: State<f64>,
    rf_race: RotorMppiConfig,
    rf_wins: usize,
    rf_done: bool,
    rf_respawns: usize,
    // shared race lifecycle
    race_gates: Vec<Gate>, // the 5 gates for RENDERING (a drone's own course gets
                           // replaced by a 1-gate hover course when it wins, so we
                           // can't render from that — gates would vanish on a win)
    idle_left: usize,   // counts up once BOTH are done → triggers a fresh course
    race_steps: usize,  // per-race step budget (so a stuck JEPA can't freeze it)
    randomize: bool,    // draw a fresh domain-randomized drone each race
    rotor_gain: [f64; NUM_ROTORS], // per-rotor thrust gain of the current drone
}

const FIRST_FINISH_ADVANCE_STEPS: usize = 200; // 10 s at 20 Hz
const MAX_RACE_STEPS: usize = 700; // ~35 s budget before forcing a new course
// Launch/respawn LOW (below the 5-15 m gate band) so the drone climbs UP into the
// gates. The JEPA model mispredicts steep descents, so starting in-band (forcing
// dives down to lower gates) caused frequent crashes; climbing from below is safe
// (0.3 vs 13.7 crashes/race in examples/jepa_fly).
const START_Z: f64 = 1.5;

/// Drive ONE drone a single control step against the true `reality`, returning the
/// executed action. Handles the race→hold transition (and feeds the model history
/// via `observe`, a no-op for the true-dynamics drone). Generic over the rollout
/// model so any CTBR MPPI drone shares identical flight logic.
fn drive_drone<M: RolloutModel>(
    ctrl: &mut MppiController<M>,
    state: &mut State<f64>,
    course: &mut Course,
    done: &mut bool,
    wins: &mut usize,
    reality: &Multirotor<f64, Ctbr>,
    dt: f64,
    hover: &MppiConfig,
) -> CtbrCmd<f64> {
    let a = ctrl.act(state, course);
    let prev = *state;
    *state = reality.step(state, &a, dt);
    ctrl.observe(&prev, &a); // record (state, action) for the learned model
    if !*done {
        course.advance(prev.x, state.x);
        if course.finished() {
            *done = true;
            *wins += 1;
            *course = hover_course(state.x);
            ctrl.cfg = *hover; // switch to the PD hold cost
        }
    }
    a
}

/// Drive the REACTIVE RL drone one step: obs → policy → action → integrate. No
/// planning, no history, no cost configs — just a forward pass. On finishing, it
/// holds at the finish (the policy flies toward the single hover gate).
fn drive_rl(
    ctrl: &mut RlPolicy,
    state: &mut State<f64>,
    course: &mut Course,
    done: &mut bool,
    wins: &mut usize,
    reality: &Multirotor<f64, Ctbr>,
    dt: f64,
) {
    let a = ctrl.act(state, course);
    let prev = *state;
    *state = reality.step(state, &a, dt);
    if !*done {
        course.advance(prev.x, state.x);
        if course.finished() {
            *done = true;
            *wins += 1;
            *course = hover_course(state.x);
        }
    }
}

/// The RL drone in ROTOR-FORCE mode: same reactive policy, raw per-rotor forces
/// (no inner rate loop) — the same actuator space as the rotor-force JEPA drone.
fn drive_rl_rotor(
    ctrl: &mut RlPolicy,
    state: &mut State<f64>,
    course: &mut Course,
    done: &mut bool,
    wins: &mut usize,
    reality: &Multirotor<f64, RotorForce>,
    dt: f64,
) {
    let a = ctrl.act_rotor(state, course);
    let prev = *state;
    *state = reality.step(state, &a, dt);
    if !*done {
        course.advance(prev.x, state.x);
        if course.finished() {
            *done = true;
            *wins += 1;
            *course = hover_course(state.x);
        }
    }
}

/// Drive the rotor-force JEPA drone one step against its RotorForce `reality`. Same
/// race→hold transition + history feed as `drive_drone`, but the action is four
/// rotor forces and the hover swap uses a [`RotorMppiConfig`].
fn drive_rotor(
    ctrl: &mut RotorMppiController,
    state: &mut State<f64>,
    course: &mut Course,
    done: &mut bool,
    wins: &mut usize,
    reality: &Multirotor<f64, RotorForce>,
    dt: f64,
    hover: &RotorMppiConfig,
) {
    let a = ctrl.act(state, course);
    let prev = *state;
    *state = reality.step(state, &a, dt);
    ctrl.observe(&prev, a); // record (state, rotor-forces) for the learned model
    if !*done {
        course.advance(prev.x, state.x);
        if course.finished() {
            *done = true;
            *wins += 1;
            *course = hover_course(state.x);
            ctrl.cfg = hover.clone(); // switch to the position-hold cost
        }
    }
}

/// A single gate at `p` so MPPI holds position (hover) there.
fn hover_course(p: Vec3<f64>) -> Course {
    let hp = Vec3::new(p.x, p.y, p.z.max(1.3)); // never hold at the ground
    Course::new(vec![Gate::new(hp, Vec3::new(1.0, 0.0, 0.0), 1.0)], false)
}

#[wasm_bindgen]
impl WasmRacer {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: f64) -> WasmRacer {
        let seed = seed as u64;
        let mut rng = Rng::new(seed | 1);
        let (input, rotor_gain) = sample_drone(&mut rng);
        let race = race_cfg(input.mass);
        let ctrl = MppiController::new(TrueDynamics::new(&input, ROLLOUT_NSUB), race, 42);
        let mut rl_ctrl = RlPolicy::from_blob(RL_BLOB, input.mass);
        rl_ctrl.set_f_max(input.k_eta * input.rotor_speed_max * input.rotor_speed_max);
        // rotor-force JEPA drone
        let rf_race = rotor_race_cfg(input.mass);
        let rf_model = SkyJepaLite::from_blob(ROTOR_JEPA_BLOB);
        let rf_ctrl = RotorMppiController::new(JepaRollout::new(rf_model), rf_race.clone(), 11);
        let course = random_course(&mut rng);
        let race_gates = course.gates.clone();
        let start = hover_state(&input, 0.0, 0.0, START_Z);
        WasmRacer {
            reality: Multirotor::with_substeps(&input, 8),
            dt: 0.05,
            rng,
            seed,
            ctrl,
            course: course.clone(),
            state: start,
            race,
            wins: 0,
            done: false,
            rl_ctrl,
            rl_course: course.clone(),
            rl_state: start,
            rl_wins: 0,
            rl_done: false,
            rl_respawns: 0,
            reality_rf: Multirotor::with_substeps(&input, 8),
            rf_ctrl,
            rf_course: course.clone(),
            rf_state: start,
            rf_race,
            rf_wins: 0,
            rf_done: false,
            rf_respawns: 0,
            race_gates,
            idle_left: 0,
            race_steps: 0,
            randomize: true,
            rotor_gain,
            input,
        }
    }

    /// Rebuild everything that depends on the drone params (reality + true-MPPI
    /// rollout model + thrust scales for all three controllers). Call after the
    /// drone (`self.input`) changes — mass slider or per-race randomization.
    fn apply_input(&mut self) {
        let m = self.input.mass;
        let fr = race_cfg(m);
        self.race.hover_thrust = fr.hover_thrust;
        self.race.thrust_min = fr.thrust_min;
        self.race.thrust_max = fr.thrust_max;
        self.rl_ctrl.set_mass(m);
        self.rl_ctrl.set_f_max(
            self.input.k_eta * self.input.rotor_speed_max * self.input.rotor_speed_max);
        // rotor-force planner: refresh its per-rotor hover/exploration force scale
        let rf = rotor_race_cfg(m);
        self.rf_race.hover_force = rf.hover_force;
        self.rf_race.f_max = rf.f_max;
        self.rf_race.sigma_force = rf.sigma_force;
        self.rf_ctrl.cfg.hover_force = rf.hover_force;
        self.rf_ctrl.cfg.f_max = rf.f_max;
        self.rf_ctrl.cfg.sigma_force = rf.sigma_force;
        self.reality = Multirotor::with_substeps(&self.input, 8);
        self.reality_rf = Multirotor::with_substeps(&self.input, 8);
        self.ctrl.model = TrueDynamics::new(&self.input, ROLLOUT_NSUB);
        // apply per-rotor "motor power" to reality AND the true-MPPI model (the
        // learned JEPA/RL models don't see it — it's a disturbance they must absorb).
        self.reality.set_rotor_gain(self.rotor_gain);
        self.reality_rf.set_rotor_gain(self.rotor_gain);
        self.ctrl.model.veh.set_rotor_gain(self.rotor_gain);
    }

    /// Toggle per-race domain randomization of the drone (shape + weight + motors).
    pub fn set_randomize(&mut self, on: bool) {
        self.randomize = on;
    }
    /// The 4 rotor (x,y,z) positions of the current drone — for rendering its frame.
    pub fn drone_rotor_pos(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(12);
        for rp in &self.input.rotor_pos {
            v.extend_from_slice(&[rp[0], rp[1], rp[2]]);
        }
        v
    }
    /// Current drone mass [kg] (for the HUD when randomization is on).
    pub fn drone_mass(&self) -> f64 {
        self.input.mass
    }
    /// Current frame arm length [m] (rotor distance from center) — the "shape".
    pub fn drone_arm(&self) -> f64 {
        self.input.rotor_pos[0][0] * std::f64::consts::SQRT_2
    }

    /// Live-tune a racing-cost weight by name (wired to the HUD sliders). Applies
    /// immediately if currently racing; always stored so it survives the next
    /// race↔hover swap. Unknown keys are ignored.
    pub fn set_race_param(&mut self, key: &str, val: f64) {
        let c = &mut self.race;
        match key {
            "w_vel" => c.w_vel = val,
            "w_pos" => c.w_pos = val,
            "w_vdamp" => c.w_vdamp = val,
            "w_effort" => c.w_effort = val,
            "w_alt" => c.w_alt = val,
            "w_terminal" => c.w_terminal = val,
            "gate_reward" => c.gate_reward = val,
            "sigma_rate" => c.sigma_rate = val,
            "sigma_thrust" => c.sigma_thrust = val * self.input.mass * GRAVITY, // as fraction of mg
            "rate_max" => c.rate_max = val,
            "beta" => c.beta = val.clamp(0.0, 0.98),
            "lambda" => c.lambda = val.max(1e-3),
            _ => {}
        }
        if !self.done {
            self.ctrl.cfg = self.race; // apply live while racing
        }
    }

    /// The track seed: pass it back to `new(seed)` (or `?seed=` in the URL) to
    /// replay the EXACT same sequence of courses. Returned as f64 for JS.
    pub fn seed(&self) -> f64 {
        self.seed as f64
    }

    /// A paste-ready debug snapshot (JSON): the seed plus the live situation —
    /// wins, current gate, mass, and the full 17-float state vector. Hand this to
    /// a developer to reproduce exactly what you were seeing.
    pub fn debug_snapshot(&self) -> String {
        let s = &self.state;
        let q = s.q.to_array();
        format!(
            "{{\"seed\":{},\"wins\":{},\"gate\":{},\"hovering\":{},\"mass\":{},\
             \"state\":[{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}]}}",
            self.seed, self.wins, self.course.next, self.done, self.input.mass,
            s.x.x, s.x.y, s.x.z, s.v.x, s.v.y, s.v.z,
            q[0], q[1], q[2], q[3], s.w.x, s.w.y, s.w.z,
        )
    }

    /// Generate a fresh shared course and reset all drones to the start hover.
    fn new_shared_race(&mut self) {
        if self.randomize {
            let (input, gain) = sample_drone(&mut self.rng);
            self.input = input;
            self.rotor_gain = gain;
            self.apply_input();
        }
        let course = random_course(&mut self.rng);
        self.race_gates = course.gates.clone();
        self.course = course.clone();
        self.rl_course = course.clone();
        self.rf_course = course;
        let start = hover_state(&self.input, 0.0, 0.0, START_Z);
        self.state = start;
        self.rl_state = start;
        self.rf_state = start;
        self.done = false;
        self.rl_done = false;
        self.rl_respawns = 0;
        self.rf_done = false;
        self.rf_respawns = 0;
        self.idle_left = 0;
        self.race_steps = 0;
        self.ctrl.cfg = self.race;
        self.ctrl.reset_nominal();
        self.rf_ctrl.cfg = self.rf_race.clone();
        self.rf_ctrl.model.reset(); // clear the learned model's stale history
        self.rf_ctrl.reset_nominal();
    }

    /// Restart the current race from hover (same course), resetting BOTH drones.
    pub fn reset(&mut self) {
        let start = hover_state(&self.input, 0.0, 0.0, START_Z);
        self.state = start;
        self.rl_state = start;
        self.rf_state = start;
        for c in [&mut self.course, &mut self.rl_course, &mut self.rf_course] {
            c.next = 0;
            c.laps_completed = 0;
        }
        // restore the race gates (a drone may have swapped its course to hover)
        self.course.gates = self.race_gates.clone();
        self.rl_course.gates = self.race_gates.clone();
        self.rf_course.gates = self.race_gates.clone();
        self.done = false;
        self.rl_done = false;
        self.rl_respawns = 0;
        self.rf_done = false;
        self.rf_respawns = 0;
        self.idle_left = 0;
        self.race_steps = 0;
        self.ctrl.cfg = self.race;
        self.ctrl.reset_nominal();
        self.rf_ctrl.cfg = self.rf_race.clone();
        self.rf_ctrl.model.reset();
        self.rf_ctrl.reset_nominal();
    }

    /// Advance one control step for all racers. Returns the hidden true drone's thrust.
    pub fn step(&mut self) -> f64 {
        let mut hov_t = hover_cfg(self.input.mass);
        let mut hov_rf = rotor_hover_cfg(self.input.mass);
        // carry the live (user-set) sample counts into the hover phase too
        hov_t.samples = self.race.samples;
        hov_rf.base.samples = self.rf_race.base.samples;
        let a = drive_drone(
            &mut self.ctrl, &mut self.state, &mut self.course, &mut self.done,
            &mut self.wins, &self.reality, self.dt, &hov_t,
        );
        // RL drone in rotor-force mode: apples-to-apples with the rotor JEPA
        // (same raw actuator space, no inner rate loop to lean on).
        drive_rl_rotor(
            &mut self.rl_ctrl, &mut self.rl_state, &mut self.rl_course,
            &mut self.rl_done, &mut self.rl_wins, &self.reality_rf, self.dt,
        );
        drive_rotor(
            &mut self.rf_ctrl, &mut self.rf_state, &mut self.rf_course,
            &mut self.rf_done, &mut self.rf_wins, &self.reality_rf, self.dt, &hov_rf,
        );
        // rotor-force is the hardest mode and most exploitable — same OOD respawn net.
        let fz = self.rf_state.x.z;
        if !self.rf_done && (!fz.is_finite() || fz < 0.15) {
            self.rf_respawns += 1;
            let p = self.rf_state.x;
            let sx = if p.x.is_finite() { p.x.clamp(0.0, 11.0) } else { 0.0 };
            let sy = if p.y.is_finite() { p.y.clamp(-4.0, 4.0) } else { 0.0 };
            self.rf_state = hover_state(&self.input, sx, sy, START_Z);
            self.rf_ctrl.model.reset();
            self.rf_ctrl.reset_nominal();
        }
        // RL safety net (it rarely crashes): respawn low if it hits the floor.
        let rz = self.rl_state.x.z;
        if !self.rl_done && (!rz.is_finite() || rz < 0.15) {
            self.rl_respawns += 1;
            let p = self.rl_state.x;
            let sx = if p.x.is_finite() { p.x.clamp(0.0, 11.0) } else { 0.0 };
            let sy = if p.y.is_finite() { p.y.clamp(-4.0, 4.0) } else { 0.0 };
            self.rl_state = hover_state(&self.input, sx, sy, START_Z);
        }
        self.race_steps += 1;
        // Public demo pacing is driven by the visible racers: RL and rotor-force
        // JEPA. Once any one of them finishes, keep the current track on screen
        // for 10 seconds, then advance automatically.
        let first_visible_done = self.rl_done || self.rf_done;
        if first_visible_done {
            self.idle_left += 1;
        }
        if (first_visible_done && self.idle_left > FIRST_FINISH_ADVANCE_STEPS)
            || self.race_steps > MAX_RACE_STEPS
        {
            self.new_shared_race();
        }
        a.thrust
    }

    /// Read a current racing-cost weight (to initialise the HUD sliders).
    pub fn race_param(&self, key: &str) -> f64 {
        let c = &self.race;
        match key {
            "w_vel" => c.w_vel,
            "w_pos" => c.w_pos,
            "w_vdamp" => c.w_vdamp,
            "w_effort" => c.w_effort,
            "w_alt" => c.w_alt,
            "w_terminal" => c.w_terminal,
            "gate_reward" => c.gate_reward,
            "sigma_rate" => c.sigma_rate,
            "sigma_thrust" => c.sigma_thrust / (self.input.mass * GRAVITY),
            "rate_max" => c.rate_max,
            "beta" => c.beta,
            "lambda" => c.lambda,
            _ => f64::NAN,
        }
    }

    pub fn wins(&self) -> usize {
        self.wins
    }
    /// true while the TRUE drone is holding at its finish (race won).
    pub fn hovering(&self) -> bool {
        self.done
    }

    fn state_vec(s: &State<f64>) -> Vec<f64> {
        let q = s.q.to_array();
        vec![
            s.x.x, s.x.y, s.x.z, s.v.x, s.v.y, s.v.z,
            q[0], q[1], q[2], q[3], s.w.x, s.w.y, s.w.z,
            s.rotor_speeds[0], s.rotor_speeds[1], s.rotor_speeds[2], s.rotor_speeds[3],
        ]
    }

    /// True drone state: [px,py,pz, vx,vy,vz, qi,qj,qk,qw, wx,wy,wz, r0..r3] (17).
    pub fn state(&self) -> Vec<f64> {
        Self::state_vec(&self.state)
    }

    /// Gate geometry for rendering: per gate [cx,cy,cz, nx,ny,nz, radius]. Uses the
    /// fixed race gates so they never vanish when a drone wins (its control-course
    /// gets swapped for a 1-gate hover course).
    pub fn gates(&self) -> Vec<f64> {
        let mut v = Vec::new();
        for g in &self.race_gates {
            v.extend_from_slice(&[g.center.x, g.center.y, g.center.z, g.normal.x, g.normal.y, g.normal.z, g.radius]);
        }
        v
    }
    /// Which gate to highlight: the true drone's target while it races, else the
    /// rotor-force JEPA drone's (so the highlight follows whoever is still racing).
    pub fn next_gate(&self) -> usize {
        let n = if self.done { self.rf_course.next } else { self.course.next };
        n.min(self.race_gates.len().saturating_sub(1))
    }
    pub fn gates_passed(&self) -> usize { self.course.gates_passed() }

    /// Number of MPPI candidate plans the TRUE drone evaluates per control step.
    pub fn true_samples(&self) -> usize { self.race.samples }
    /// Set it live (more = better planning, slower). Clamped to a sane range.
    pub fn set_true_samples(&mut self, n: usize) {
        let n = n.clamp(1, 512);
        self.race.samples = n;
        self.ctrl.cfg.samples = n;
    }
    /// RL drone state (17), gate, wins, status — the reactive PPO policy.
    pub fn rl_state(&self) -> Vec<f64> { Self::state_vec(&self.rl_state) }
    pub fn rl_next_gate(&self) -> usize { self.rl_course.next }
    pub fn rl_wins(&self) -> usize { self.rl_wins }
    pub fn rl_hovering(&self) -> bool { self.rl_done }
    pub fn rl_respawns(&self) -> usize { self.rl_respawns }

    /// Rotor-force JEPA drone state (17), gate, wins, status, respawns.
    pub fn rf_state(&self) -> Vec<f64> { Self::state_vec(&self.rf_state) }
    pub fn rf_next_gate(&self) -> usize { self.rf_course.next }
    pub fn rf_wins(&self) -> usize { self.rf_wins }
    pub fn rf_hovering(&self) -> bool { self.rf_done }
    pub fn rf_respawns(&self) -> usize { self.rf_respawns }
    /// Number of MPPI candidate plans the rotor-force drone evaluates per step.
    pub fn rf_samples(&self) -> usize { self.rf_race.base.samples }
    pub fn set_rf_samples(&mut self, n: usize) {
        let n = n.clamp(1, 512);
        self.rf_race.base.samples = n;
        self.rf_ctrl.cfg.base.samples = n;
    }

    pub fn set_wind(&mut self, x: f64, y: f64, z: f64) {
        let w = Vec3::new(x, y, z);
        self.state.wind = w;
        self.rl_state.wind = w;
        self.rf_state.wind = w;
    }
    /// Live mass change (the "break it" knob): rebuild reality + the true planner.
    /// The learned JEPA model keeps its trained `m_nominal` (so a mass change makes
    /// its predictions wrong — a feature: it shows the learned model degrade), but
    /// we do refresh the planners' thrust/force scales so nominal hover is sane.
    pub fn set_mass(&mut self, mass: f64) {
        self.input.mass = mass;
        self.apply_input(); // rebuild reality + planners + thrust scales for new mass
        if !self.done {
            self.ctrl.cfg = self.race;
        }
    }
}

#[cfg(all(test, feature = "wasm"))]
mod forecast_tests {
    use super::*;

    #[test]
    fn forecast_playback_stays_above_terrain() {
        let mut sim = WasmForecast::new(7.0);
        for _ in 0..600 {
            let state = sim.step();
            let floor = forecast_terrain_height(state[0], state[1]) + 2.45;
            assert!(
                state[2] >= floor,
                "drone below terrain clearance: z={} floor={} at t={}",
                state[2],
                floor,
                sim.time()
            );
            let forecast = sim.forecast(20);
            for p in forecast[..60].chunks_exact(3) {
                let floor = forecast_terrain_height(p[0], p[1]) + 2.45;
                assert!(p[2] >= floor, "future below terrain clearance: z={} floor={}", p[2], floor);
            }
        }
    }
}
