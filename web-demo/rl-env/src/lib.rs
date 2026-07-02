//! Vectorized gate-racing RL environment built on the rotor-rs dynamics, exposed
//! as a flat C ABI so PufferLib / CleanRL PPO can train on it at native speed (the
//! whole point: RL needs millions of env steps, and pure-Python RotorPy is too slow).
//!
//! N independent drones step in parallel each call. Action = CTBR in [-1,1]^4;
//! observation is translation-invariant (velocity, attitude, body-rates, and the
//! RELATIVE vectors to the next two gates — never absolute position). Reward =
//! progress toward the next gate + a gate-pass bonus − crash − a little effort.
//! Episodes auto-reset (the returned obs after `done=1` is already the new episode).

use std::os::raw::c_void;

use rotor_rs::rng::Rng;
use rotor_rs::{
    Ctbr, CtbrCmd, Gate, Multirotor, QuadParamsInput, Quat, RotorForce, State, Vec3, GRAVITY,
};

pub const OBS_DIM: usize = 21; // vel(3) + R(9) + omega(3) + rel_g1(3) + rel_g2(3)
pub const ACT_DIM: usize = 4; // ctbr: [thrust, wx, wy, wz]; rotor: 4 rotor forces — in [-1,1]

/// RL_ACTION_MODE=rotor -> actions are per-rotor forces around hover
/// (force_i = hover/4 * (1 + a_i)), no inner rate loop — the same raw actuator
/// space as the rotor-force JEPA drone. Default: CTBR through the rate loop.
fn rotor_mode() -> bool {
    std::env::var("RL_ACTION_MODE").map(|v| v == "rotor").unwrap_or(false)
}
fn envf(key: &str, d: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
const DT: f64 = 0.05;
const N_SUB: usize = 8; // stiff k_w rate loop needs fine substeps (else NaN)
const MAX_STEPS: usize = 500; // episode timeout (25 s)
const RATE_MAX: f64 = 10.0; // body-rate command scale [rad/s]

// reward weights
const ALIVE: f64 = 0.05; // per surviving step — learn to FLY before racing
const W_PROG: f64 = 1.0; // per metre closed toward the next gate
const GATE_BONUS: f64 = 10.0;
const FINISH_BONUS: f64 = 30.0;
const CRASH_PEN: f64 = 5.0;
const W_EFFORT: f64 = 0.001;

const HUM_IXX: f64 = 3.65e-3;
const HUM_IYY: f64 = 3.68e-3;
const HUM_IZZ: f64 = 7.03e-3;
const HUM_KETA: f64 = 5.57e-6;
const HUM_KM: f64 = 1.36e-7;

fn lerp(r: &mut Rng, a: f64, b: f64) -> f64 {
    a + (b - a) * r.uniform()
}

/// Domain-randomized drone — IDENTICAL ranges to the JEPA training data
/// (`gen_dataset.rs`): different frame size (arm), weight, inertia, drag, motor
/// coefficients, motor delay, and rate gain `k_w` on every episode. Each parallel
/// env races a different drone, so the policy must be robust to the whole fleet.
fn sample_drone(r: &mut Rng) -> (QuadParamsInput, [f64; 4]) {
    // distribution knobs (env-overridable, same convention as gen_dataset /
    // fleet_fly): MASS_LO/HI, ARM_LO/HI, SYM=1 for symmetric frames
    let (arm_lo, arm_hi) = (envf("ARM_LO", 0.05), envf("ARM_HI", 0.40));
    let arms: [f64; 4] = if envf("SYM", 0.0) != 0.0 {
        let a = lerp(r, arm_lo, arm_hi);
        [a; 4]
    } else {
        core::array::from_fn(|_| lerp(r, arm_lo, arm_hi))
    };
    let d: [f64; 4] = core::array::from_fn(|i| arms[i] * std::f64::consts::FRAC_1_SQRT_2);
    let rotor_pos = [[d[0], d[0], 0.0], [d[1], -d[1], 0.0], [-d[2], -d[2], 0.0], [-d[3], d[3], 0.0]];
    let avg_arm = arms.iter().sum::<f64>() / 4.0;
    let mass = lerp(r, envf("MASS_LO", 0.2), envf("MASS_HI", 2.0));
    let rpm_max = 1500.0;
    // k_eta from thrust-to-weight in [2,4] -> guaranteed flyable (no heavy+weak combos)
    let twr = lerp(r, 2.0, 4.0);
    let k_eta = twr * mass * GRAVITY / (4.0 * rpm_max * rpm_max);
    let k_m = k_eta * (HUM_KM / HUM_KETA) * lerp(r, 0.7, 1.3);
    let i_scale = (mass / 0.5) * (avg_arm / 0.17).powi(2) * lerp(r, 0.7, 1.3); // I ∝ m·r²
    let input = QuadParamsInput {
        mass,
        ixx: HUM_IXX * i_scale, iyy: HUM_IYY * i_scale, izz: HUM_IZZ * i_scale,
        ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos,
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: lerp(r, 0.02, 0.30), c_dy: lerp(r, 0.02, 0.30), c_dz: lerp(r, 0.05, 0.40),
        k_eta, k_m,
        k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: lerp(r, 0.01, 0.04),
        rotor_speed_min: 0.0, rotor_speed_max: rpm_max,
        k_w: lerp(r, 6.0, 18.0),
    };
    (input, [1.0; 4])
}

fn sky_course(r: &mut Rng) -> Vec<Gate> {
    (0..5)
        .map(|_| {
            let x = 1.5 + r.uniform() * 9.0;
            let y = (r.uniform() * 2.0 - 1.0) * 3.5;
            let z = 5.0 + r.uniform() * 10.0;
            Gate::new(Vec3::new(x, y, z), Vec3::new(r.normal(), r.normal(), r.normal()), 0.85)
        })
        .collect()
}

fn hover_state(p: &QuadParamsInput, z: f64) -> State<f64> {
    let hov = (p.mass * GRAVITY / (4.0 * p.k_eta)).sqrt();
    State {
        x: Vec3::new(0.0, 0.0, z), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [hov; rotor_rs::NUM_ROTORS],
    }
}

struct DroneEnv {
    input: QuadParamsInput,
    reality: Multirotor<f64, Ctbr>,
    reality_rf: Multirotor<f64, RotorForce>, // used when `rotor` (steppers are stateless)
    rotor: bool,
    f_max: f64, // per-rotor force ceiling k_eta * rpm_max^2
    state: State<f64>,
    gates: Vec<Gate>,
    next: usize,
    steps: usize,
    prev_dist: f64,
    hover_thrust: f64,
    rng: Rng,
}

impl DroneEnv {
    fn new(seed: u64) -> Self {
        let rng = Rng::new(seed | 1);
        let (input, _) = sample_drone(&mut Rng::new(seed)); // throwaway; reset() overwrites
        let mut e = DroneEnv {
            reality: Multirotor::with_substeps(&input, N_SUB),
            reality_rf: Multirotor::with_substeps(&input, N_SUB),
            rotor: rotor_mode(),
            f_max: input.k_eta * input.rotor_speed_max * input.rotor_speed_max,
            input,
            state: State { x: Vec3::zero(), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
                w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [0.0; rotor_rs::NUM_ROTORS] },
            gates: Vec::new(), next: 0, steps: 0, prev_dist: 0.0, hover_thrust: 0.0,
            rng,
        };
        e.reset();
        e
    }

    fn reset(&mut self) {
        let (input, gain) = sample_drone(&mut self.rng);
        self.input = input;
        self.reality = Multirotor::with_substeps(&self.input, N_SUB);
        self.reality.set_rotor_gain(gain); // per-motor power
        self.reality_rf = Multirotor::with_substeps(&self.input, N_SUB);
        self.reality_rf.set_rotor_gain(gain);
        self.f_max = self.input.k_eta * self.input.rotor_speed_max * self.input.rotor_speed_max;
        self.gates = sky_course(&mut self.rng);
        self.state = hover_state(&self.input, 1.5);
        // a little wind sometimes
        self.state.wind = Vec3::new(lerp(&mut self.rng, -2.0, 2.0), lerp(&mut self.rng, -2.0, 2.0),
                                    lerp(&mut self.rng, -1.0, 1.0));
        self.next = 0;
        self.steps = 0;
        self.hover_thrust = self.input.mass * GRAVITY;
        self.prev_dist = (self.gates[0].center - self.state.x).norm();
    }

    fn write_obs(&self, o: &mut [f32]) {
        let s = &self.state;
        let r = s.q.to_rotmat();
        let g1 = self.gates.get(self.next).map(|g| g.center).unwrap_or(s.x);
        let g2 = self.gates.get(self.next + 1).copied().map(|g| g.center)
            .unwrap_or(g1);
        let rel1 = g1 - s.x;
        let rel2 = g2 - s.x;
        let vs = 1.0 / 5.0;
        o[0] = (s.v.x * vs) as f32; o[1] = (s.v.y * vs) as f32; o[2] = (s.v.z * vs) as f32;
        for i in 0..3 { for j in 0..3 { o[3 + i * 3 + j] = r.rows[i][j] as f32; } }
        o[12] = (s.w.x * vs) as f32; o[13] = (s.w.y * vs) as f32; o[14] = (s.w.z * vs) as f32;
        o[15] = (rel1.x * vs) as f32; o[16] = (rel1.y * vs) as f32; o[17] = (rel1.z * vs) as f32;
        o[18] = (rel2.x * vs) as f32; o[19] = (rel2.y * vs) as f32; o[20] = (rel2.z * vs) as f32;
    }

    /// Apply one action (in [-1,1]^4), return (reward, done).
    /// ctbr: [thrust, wx, wy, wz]; rotor: 4 per-rotor forces around hover.
    fn step(&mut self, a: &[f32]) -> (f32, bool) {
        let prev = self.state.x;
        let mut effort = 0.0;
        if self.rotor {
            let per_hover = self.hover_thrust / 4.0;
            let forces: [f64; 4] = core::array::from_fn(|i| {
                (per_hover * (1.0 + a[i] as f64).clamp(0.0, 3.0)).min(self.f_max)
            });
            // penalize DIFFERENTIAL effort (attitude aggression), the analog of
            // the ctbr arm's |w_cmd|^2 penalty
            let mean = (a[0] + a[1] + a[2] + a[3]) as f64 / 4.0;
            effort = 25.0 * (0..4).map(|i| (a[i] as f64 - mean).powi(2)).sum::<f64>();
            self.state = self.reality_rf.step(&self.state, &forces, DT);
        } else {
            let thrust = self.hover_thrust * (1.0 + a[0] as f64).clamp(0.0, 3.0);
            let cmd = CtbrCmd {
                thrust,
                w_cmd: Vec3::new(a[1] as f64 * RATE_MAX, a[2] as f64 * RATE_MAX,
                                 a[3] as f64 * RATE_MAX),
            };
            effort = cmd.w_cmd.dot(cmd.w_cmd);
            self.state = self.reality.step(&self.state, &cmd, DT);
        }
        self.steps += 1;

        let s = &self.state;
        let bad = !s.x.z.is_finite() || s.x.z < 0.15 || s.v.norm() > 40.0;
        if bad {
            self.reset();
            return (-CRASH_PEN as f32, true);
        }

        let mut reward = ALIVE - W_EFFORT * effort;
        // gate progress (closing distance to the current target gate)
        if let Some(g) = self.gates.get(self.next) {
            let cur = (g.center - s.x).norm();
            reward += W_PROG * (self.prev_dist - cur);
            self.prev_dist = cur;
            if g.crossed(prev, s.x) {
                reward += GATE_BONUS;
                self.next += 1;
                if let Some(ng) = self.gates.get(self.next) {
                    self.prev_dist = (ng.center - s.x).norm();
                }
            }
        }
        let finished = self.next >= self.gates.len();
        let timeout = self.steps >= MAX_STEPS;
        if finished {
            reward += FINISH_BONUS;
        }
        let done = finished || timeout;
        if done {
            self.reset();
        }
        (reward as f32, done)
    }
}

pub struct VecEnv {
    drones: Vec<DroneEnv>,
}

// --------------------------------------------------------------------------- //
// C ABI for the Python/PufferLib binding. All buffers are flat f32, row-major.
// --------------------------------------------------------------------------- //

#[no_mangle]
pub extern "C" fn rlenv_obs_dim() -> usize { OBS_DIM }
#[no_mangle]
pub extern "C" fn rlenv_act_dim() -> usize { ACT_DIM }

/// Create `n` parallel envs. Returns an opaque handle (free with `rlenv_free`).
#[no_mangle]
pub extern "C" fn rlenv_create(n: usize, seed: u64) -> *mut c_void {
    let drones = (0..n).map(|i| DroneEnv::new(seed.wrapping_add(i as u64 * 2654435761))).collect();
    Box::into_raw(Box::new(VecEnv { drones })) as *mut c_void
}

/// Reset all envs; write the initial observations into `obs` (n * OBS_DIM f32).
#[no_mangle]
pub extern "C" fn rlenv_reset(handle: *mut c_void, obs: *mut f32) {
    let env = unsafe { &mut *(handle as *mut VecEnv) };
    let n = env.drones.len();
    let obs = unsafe { std::slice::from_raw_parts_mut(obs, n * OBS_DIM) };
    for (i, d) in env.drones.iter_mut().enumerate() {
        d.reset();
        d.write_obs(&mut obs[i * OBS_DIM..(i + 1) * OBS_DIM]);
    }
}

/// Step all envs with `actions` (n * ACT_DIM f32); write next `obs`, `rewards`
/// (n), `dones` (n, 0/1). Done envs auto-reset and `obs` holds the NEW episode.
#[no_mangle]
pub extern "C" fn rlenv_step(
    handle: *mut c_void,
    actions: *const f32,
    obs: *mut f32,
    rewards: *mut f32,
    dones: *mut f32,
) {
    let env = unsafe { &mut *(handle as *mut VecEnv) };
    let n = env.drones.len();
    let actions = unsafe { std::slice::from_raw_parts(actions, n * ACT_DIM) };
    let obs = unsafe { std::slice::from_raw_parts_mut(obs, n * OBS_DIM) };
    let rewards = unsafe { std::slice::from_raw_parts_mut(rewards, n) };
    let dones = unsafe { std::slice::from_raw_parts_mut(dones, n) };

    // Envs are independent → split into per-core chunks and step in parallel.
    let nthreads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let chunk = n.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        let iter = env.drones.chunks_mut(chunk)
            .zip(actions.chunks(chunk * ACT_DIM))
            .zip(obs.chunks_mut(chunk * OBS_DIM))
            .zip(rewards.chunks_mut(chunk))
            .zip(dones.chunks_mut(chunk));
        for ((((dch, ach), och), rch), doch) in iter {
            sc.spawn(move || {
                for (i, d) in dch.iter_mut().enumerate() {
                    let (r, done) = d.step(&ach[i * ACT_DIM..(i + 1) * ACT_DIM]);
                    rch[i] = r;
                    doch[i] = if done { 1.0 } else { 0.0 };
                    d.write_obs(&mut och[i * OBS_DIM..(i + 1) * OBS_DIM]);
                }
            });
        }
    });
}

#[no_mangle]
pub extern "C" fn rlenv_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut VecEnv)) };
    }
}
