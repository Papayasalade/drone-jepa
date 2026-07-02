//! How do the controllers fare across the WIDE randomized drone distribution
//! (mass 0.2-2kg, asymmetric arms 5-40cm, TWR-derived k_eta) — the same fleet the
//! demo's "randomize" toggle draws from? Each trial samples one wide drone and races
//! every controller on the SAME drone + course (sky gates, climb-from-low start).
//!   cargo run --release --features jepa --example fleet_fly
//!
//! Controllers: true-MPPI (perfect model = flyability baseline), RL old/v2 (reactive
//! PPO), JEPA-ctbr old/v2clean (MPPI over the learned world model).

use rotor_rs::jepa::{JepaRollout, SkyJepaLite};
use rotor_rs::rl::RlPolicy;
use rotor_rs::rng::Rng;
use rotor_rs::spline_mppi::{SplineMppiConfig, SplineMppiController};
use rotor_rs::{
    Controller, Course, Ctbr, CtbrCmd, Gate, MppiConfig, MppiController, Multirotor,
    QuadParamsInput, Quat, State, TrueDynamics, Vec3, GRAVITY,
};

/// Policy-guided MPC: each step, the RL policy proposes an action, MPPI re-centers its
/// candidate search on it (set_nominal_const) and the JEPA model RANKS the nearby plans.
/// Tests whether the wide world model is usable when it only has to rank IN-DISTRIBUTION
/// plans (near a known-good action) instead of arbitrary ones.
struct RlGuided {
    jepa: MppiController<JepaRollout>,
    rl: RlPolicy,
}
impl Controller for RlGuided {
    fn act(&mut self, state: &State<f64>, course: &Course) -> CtbrCmd<f64> {
        let a_rl = self.rl.act(state, course);
        self.jepa.set_nominal_const(a_rl);
        self.jepa.act(state, course)
    }
}

/// The geometric SE3 seed executed DIRECTLY (no world model) — the baseline the spline
/// planner refines, to see if the model adds value or just adds crashes.
struct Se3Only {
    inner: SplineMppiController,
}
impl Controller for Se3Only {
    fn act(&mut self, state: &State<f64>, course: &Course) -> CtbrCmd<f64> {
        self.inner.seed_only(state, course)
    }
}


const HUM_IXX: f64 = 3.65e-3; const HUM_IYY: f64 = 3.68e-3; const HUM_IZZ: f64 = 7.03e-3;
const HUM_KETA: f64 = 5.57e-6; const HUM_KM: f64 = 1.36e-7;

fn lerp(r: &mut Rng, a: f64, b: f64) -> f64 { a + (b - a) * r.uniform() }
fn envf(key: &str, d: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

// the demo's wide distribution (mirrors wasm::sample_drone / gen_dataset). Same env
// knobs as gen_dataset so a narrowed train distribution can be tested on a matching fleet.
fn sample_drone(r: &mut Rng) -> (QuadParamsInput, [f64; 4]) {
    let (arm_lo, arm_hi) = (envf("ARM_LO", 0.05), envf("ARM_HI", 0.40));
    let sym = envf("SYM", 0.0) != 0.0;
    let arms: [f64; 4] = if sym { let a = lerp(r, arm_lo, arm_hi); [a; 4] }
        else { core::array::from_fn(|_| lerp(r, arm_lo, arm_hi)) };
    let d: [f64; 4] = core::array::from_fn(|i| arms[i] * std::f64::consts::FRAC_1_SQRT_2);
    let rotor_pos = [[d[0], d[0], 0.0], [d[1], -d[1], 0.0], [-d[2], -d[2], 0.0], [-d[3], d[3], 0.0]];
    let avg_arm = arms.iter().sum::<f64>() / 4.0;
    let mass = lerp(r, envf("MASS_LO", 0.2), envf("MASS_HI", 2.0));
    let rpm_max = 1500.0;
    let twr = lerp(r, 2.0, 4.0);
    let k_eta = twr * mass * GRAVITY / (4.0 * rpm_max * rpm_max);
    let k_m = k_eta * (HUM_KM / HUM_KETA) * lerp(r, 0.7, 1.3);
    let i_scale = (mass / 0.5) * (avg_arm / 0.17).powi(2) * lerp(r, 0.7, 1.3);
    let input = QuadParamsInput {
        mass,
        ixx: HUM_IXX * i_scale, iyy: HUM_IYY * i_scale, izz: HUM_IZZ * i_scale,
        ixy: 0.0, iyz: 0.0, ixz: 0.0, rotor_pos, rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: lerp(r, 0.02, 0.30), c_dy: lerp(r, 0.02, 0.30), c_dz: lerp(r, 0.05, 0.40),
        k_eta, k_m, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: lerp(r, 0.01, 0.04), rotor_speed_min: 0.0, rotor_speed_max: rpm_max,
        k_w: lerp(r, 6.0, 18.0),
    };
    (input, [1.0; 4])
}

fn hover_state(p: &QuadParamsInput, z: f64) -> State<f64> {
    let hov = (p.mass * GRAVITY / (4.0 * p.k_eta)).sqrt();
    State { x: Vec3::new(0.0, 0.0, z), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [hov; 4] }
}
fn sky_course(r: &mut Rng) -> Course {
    let gates = (0..5).map(|_| {
        let x = 1.5 + r.uniform() * 9.0;
        let y = (r.uniform() * 2.0 - 1.0) * 3.5;
        let z = 5.0 + r.uniform() * 10.0;
        Gate::new(Vec3::new(x, y, z), Vec3::new(r.normal(), r.normal(), r.normal()), 0.85)
    }).collect();
    Course::new(gates, false)
}

// demo true-MPPI race cfg (mirrors wasm::race_cfg)
fn race_cfg(mass: f64) -> MppiConfig {
    let mut c = MppiConfig::for_mass(mass);
    c.samples = 128; c.w_vel = 3.0; c.w_pos = 0.0; c.w_vdamp = 0.04; c.w_effort = 0.01;
    c.sigma_rate = 2.8; c.rate_max = 10.0; c.beta = 0.90; c.w_floor = 300.0;
    c
}
// demo JEPA race cfg (mirrors wasm::jepa_race_cfg). "Tame the MPPI" knobs via env:
// HORIZON (shorter = less compounding), TRUST (latent trust-region weight), SIGMA
// (rate exploration), VMAX (speed cap).
fn jepa_race_cfg(mass: f64) -> MppiConfig {
    let mut c = MppiConfig::for_mass(mass);
    c.horizon = envf("HORIZON", 12.0) as usize;
    c.samples = 48; c.sigma_rate = envf("SIGMA", 1.6); c.sigma_thrust = 0.12 * mass * GRAVITY;
    c.w_vel = 0.8; c.w_terminal = 2.5; c.w_alt = 3.5; c.w_vdamp = 0.18; c.rate_max = 6.0;
    c.v_max = envf("VMAX", 3.5); c.w_speed = 1.5; c.w_floor = 600.0;
    c.trust_lambda = envf("TRUST", 1000.0); c.lambda = 0.4;
    c
}

// Race one controller on a fixed drone+course (with the demo's respawn-and-retry net).
// Returns (gates_passed, hard_crashed, respawns).
fn race<C: Controller>(
    ctrl: &mut C, reality: &Multirotor<f64, Ctbr>, input: &QuadParamsInput,
    mut course: Course,
) -> (usize, bool, usize) {
    let mut s = hover_state(input, 1.5);
    let (mut respawns, mut hard) = (0usize, false);
    for _ in 0..600 {
        let a = ctrl.act(&s, &course);
        let prev = s.x;
        s = reality.step(&s, &a, 0.05);
        course.advance(prev, s.x);
        if !s.x.z.is_finite() || s.x.z < 0.15 || s.x.z > 45.0 {
            respawns += 1;
            if respawns > 8 { hard = true; break; }
            s = hover_state(input, 1.5);
        }
        if course.finished() { break; }
    }
    (course.gates_passed(), hard, respawns)
}

fn main() {
    let root = env!("CARGO_MANIFEST_DIR");
    let rd = |n: &str| std::fs::read(format!("{root}/assets/{n}")).expect(n);
    let rl_v2 = rd("skyrl_ctbr_v2.rlb");
    // the JEPA model under test (override per experiment), default the clean baseline
    let jtest_name = std::env::var("JTEST").unwrap_or_else(|_| "skyjepa_ctbr_v2clean.jblob".into());
    let jtest = rd(&jtest_name);
    println!("JEPA under test: {jtest_name}");

    let names = ["true-MPPI", "RL-v2", "JEPA-test", "JEPA+RLprop", "SE3-seed-only", "JEPA+spline(SE3)"];
    let mut gates = [0usize; 6];
    let mut crashes = [0usize; 6];
    let mut resp = [0usize; 6];
    let trials = 16;

    for t in 0..trials {
        let mut rng = Rng::new((t as u64) * 2654435761 | 1);
        let (input, gain) = sample_drone(&mut rng);
        let course = sky_course(&mut rng);
        let mut reality: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&input, 8);
        reality.set_rotor_gain(gain);
        let m = input.mass;

        // build each controller fresh for this drone
        let mut t_mppi = { let mut md = TrueDynamics::new(&input, 8); md.veh.set_rotor_gain(gain);
            MppiController::new(md, race_cfg(m), 42) };
        let mut p_rlv = RlPolicy::from_blob(&rl_v2, m);
        let mut j_test = MppiController::new(JepaRollout::new(SkyJepaLite::from_blob(&jtest)), jepa_race_cfg(m), 7);
        let mut j_guided = RlGuided {
            jepa: MppiController::new(JepaRollout::new(SkyJepaLite::from_blob(&jtest)), jepa_race_cfg(m), 7),
            rl: RlPolicy::from_blob(&rl_v2, m),
        };
        let mut j_spline = SplineMppiController::new(
            JepaRollout::new(SkyJepaLite::from_blob(&jtest)), SplineMppiConfig::for_mass(m), &input, 7);
        let mut se3_only = Se3Only { inner: SplineMppiController::new(
            JepaRollout::new(SkyJepaLite::from_blob(&jtest)), SplineMppiConfig::for_mass(m), &input, 7) };

        let r0 = race(&mut t_mppi, &reality, &input, course.clone());
        let r1 = race(&mut p_rlv, &reality, &input, course.clone());
        let r2 = race(&mut j_test, &reality, &input, course.clone());
        let r3 = race(&mut j_guided, &reality, &input, course.clone());
        let r4 = race(&mut se3_only, &reality, &input, course.clone());
        let r5 = race(&mut j_spline, &reality, &input, course.clone());
        for (i, r) in [r0, r1, r2, r3, r4, r5].iter().enumerate() {
            gates[i] += r.0; if r.1 { crashes[i] += 1; } resp[i] += r.2;
        }
        println!("drone {t:2} m={m:.2}kg: true {}/5 RLv2 {}/5 Jtest {}/5 J+RL {}/5 SE3 {}/5 J+spl {}/5",
            r0.0, r1.0, r2.0, r3.0, r4.0, r5.0);
    }
    println!("\n== over {trials} wide-distribution drones (gates/race, hard-crashes, respawns) ==");
    for i in 0..6 {
        println!("  {:<16} {:.1} gates/race   {} crashed   {} respawns",
            names[i], gates[i] as f64 / trials as f64, crashes[i], resp[i]);
    }
}
