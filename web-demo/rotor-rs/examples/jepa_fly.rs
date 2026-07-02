//! Sanity + timing for the browser's JEPA drone: fly the LEARNED-model MPPI racer
//! (rotor-rs `jepa::SkyJepaLite` + `JepaRollout`) on a few sky courses, the SAME
//! way `WasmRacer` does. Reports gates passed, crashes, and per-step wall-clock so
//! we know it's real-time viable in the browser.
//!
//!   cargo run --release --features jepa --example jepa_fly

use std::time::Instant;

use rotor_rs::jepa::{JepaRollout, SkyJepaLite};
use rotor_rs::rng::Rng;
use rotor_rs::{
    Controller, Course, Ctbr, Gate, MppiConfig, MppiController, Multirotor, QuadParamsInput, Quat,
    State, TrueDynamics, Vec3, GRAVITY,
};

fn denv(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
/// Vehicle under test — hummingbird by default, overridable via DRONE_* env
/// vars (same convention as rotor_fly / gen_dataset_rf).
fn hummingbird() -> QuadParamsInput {
    let d = denv("DRONE_ARM", 0.17) * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: denv("DRONE_MASS", 0.5),
        ixx: denv("DRONE_IXX", 3.65e-3), iyy: denv("DRONE_IYY", 3.68e-3),
        izz: denv("DRONE_IZZ", 7.03e-3),
        ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: 0.5e-2, c_dy: 0.5e-2, c_dz: 1e-2,
        k_eta: denv("DRONE_K_ETA", 5.57e-6), k_m: denv("DRONE_K_M", 1.36e-7),
        k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: denv("DRONE_TAU_M", 0.005),
        rotor_speed_min: 0.0, rotor_speed_max: 1500.0,
        k_w: denv("DRONE_K_W", 16.0),
    }
}
fn hover_state(p: &QuadParamsInput, z: f64) -> State<f64> {
    let hov = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();
    State {
        x: Vec3::new(0.0, 0.0, z), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [hov; 4],
    }
}
const ROOF: f64 = 5.0;
fn sky_course(rng: &mut Rng) -> Course {
    let zlo = std::env::var("GATE_ZLO").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let zhi = std::env::var("GATE_ZHI").ok().and_then(|v| v.parse().ok()).unwrap_or(ROOF - 1.0);
    let gates = (0..5)
        .map(|_| {
            let x = 1.5 + rng.uniform() * 9.0;
            let y = (rng.uniform() * 2.0 - 1.0) * 3.5;
            let z = zlo + rng.uniform() * (zhi - zlo);
            Gate::new(Vec3::new(x, y, z), Vec3::new(rng.normal(), rng.normal(), rng.normal()), 0.85)
        })
        .collect();
    Course::new(gates, false)
}
fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn jepa_race_cfg(mass: f64) -> MppiConfig {
    let mut cfg = MppiConfig::for_mass(mass);
    cfg.horizon = env("J_HORIZON", 20.0) as usize; // shorter = trust the model less far ahead
    cfg.samples = env("J_SAMPLES", 48.0) as usize;
    cfg.sigma_rate = env("J_SIGR", 1.6);
    cfg.sigma_thrust = env("J_SIGT", 0.12) * mass * GRAVITY;
    cfg.w_vel = env("J_WVEL", 0.9);
    cfg.w_terminal = env("J_WTERM", 2.5);
    cfg.w_alt = env("J_WALT", 3.0);
    cfg.w_vdamp = env("J_WVDAMP", 0.18);
    cfg.rate_max = env("J_RMAX", 6.0);
    cfg.v_max = env("J_VMAX", 4.0);
    cfg.w_speed = env("J_WSPEED", 1.5);
    cfg.w_floor = env("J_WFLOOR", 250.0); // ground avoidance (sim has no floor)
    cfg.trust_lambda = env("J_TRUST", 0.0); // SIGReg latent trust region (anti-exploit)
    cfg.lambda = env("J_LAM", 0.4);
    cfg
}

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let blob_rel = std::env::var("JBLOB").unwrap_or_else(|_| "assets/skyjepa_ctbr_1x.jblob".into());
    let blob = std::fs::read(root.join(&blob_rel)).expect("jblob");
    println!("model: {blob_rel}");
    let input = hummingbird();
    let reality: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&input, 8);

    // timing: also build a true-dynamics planner to compare per-step cost
    let mut true_ctrl = MppiController::new(TrueDynamics::new(&input, 8), {
        let mut c = MppiConfig::for_mass(input.mass);
        c.samples = 128;
        c
    }, 42);

    let mut total_steps = 0u32;
    let mut jepa_time = 0.0f64;
    let mut true_time = 0.0f64;
    let n_trials = env("TRIALS", 10.0) as usize;
    let steps_per = 400; // 20 s per race
    let mut grand_crashes = 0usize;
    let mut grand_gates = 0usize;

    // Demo-faithful: each race runs a fixed duration; on a crash we RESPAWN (like
    // WasmRacer) and keep going, counting crashes. Reports crashes-per-race.
    for t in 0..n_trials {
        let mut rng = Rng::new((t as u64) * 2654435761 | 1);
        let mut course = sky_course(&mut rng);
        let model = SkyJepaLite::from_blob(&blob);
        let mut ctrl = MppiController::new(JepaRollout::new(model), jepa_race_cfg(input.mass), 7);
        let start_z = env("START_Z", 6.0);
        let wind = Vec3::new(env("WIND_X", 0.0), env("WIND_Y", 0.0), env("WIND_Z", 0.0));
        let mut s = hover_state(&input, start_z);
        s.wind = wind;
        let mut crashes = 0;
        for _ in 0..steps_per {
            let t0 = Instant::now();
            let a = ctrl.act(&s, &course);
            jepa_time += t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let _ = true_ctrl.act(&s, &course);
            true_time += t1.elapsed().as_secs_f64();
            total_steps += 1;

            let prev = s;
            s = reality.step(&s, &a, 0.05);
            ctrl.observe(&prev, &a);
            course.advance(prev.x, s.x);
            if !s.x.z.is_finite() || s.x.z < 0.15 || s.x.z > 40.0 {
                crashes += 1;
                let sx = if s.x.x.is_finite() { s.x.x.clamp(0.0, 11.0) } else { 0.0 };
                let sy = if s.x.y.is_finite() { s.x.y.clamp(-4.0, 4.0) } else { 0.0 };
                s = hover_state(&input, start_z);
                s.x = Vec3::new(sx, sy, start_z);
                s.wind = wind;
                ctrl.model.reset();
                ctrl.reset_nominal();
            }
            if course.finished() {
                break;
            }
        }
        grand_crashes += crashes;
        grand_gates += course.gates_passed();
        println!(
            "race {t}: gates {}/5  crashes {}  {}",
            course.gates_passed(), crashes,
            if course.finished() { "WON" } else { "(20s)" },
        );
    }
    println!(
        "\n== over {n_trials} races: {grand_gates} gates, {grand_crashes} crashes \
         ({:.1} gates/race, {:.2} crashes/race) ==",
        grand_gates as f64 / n_trials as f64, grand_crashes as f64 / n_trials as f64,
    );

    let jms = jepa_time / total_steps as f64 * 1e3;
    let tms = true_time / total_steps as f64 * 1e3;
    println!("\nper control-step plan time:  JEPA {jms:.2} ms   true {tms:.2} ms   (both run each step; budget 50 ms @ 20 Hz)");
}
