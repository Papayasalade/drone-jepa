//! Headless verification of the browser racer's race↔hover behaviour on
//! sky-domain courses (5 gates drawn independently in a 3-D volume). Mirrors
//! `wasm::WasmRacer`'s loop and cost-config swap, but deterministic so we can
//! assert: (a) the drive-through cost actually clears the gates, and (b) the PD
//! hover cost brings the drone to rest (no orbit/oscillation).
//!
//!   cargo run --release --example sky_verify

use racer::{
    Controller, Course, Ctbr, Gate, MppiConfig, MppiController, Multirotor, QuadParamsInput, Quat,
    State, TrueDynamics, Vec3,
};
use racer::rng::Rng;

fn hummingbird() -> QuadParamsInput {
    let d = 0.17 * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: 0.5, ixx: 3.65e-3, iyy: 3.68e-3, izz: 7.03e-3,
        ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: 0.5e-2, c_dy: 0.5e-2, c_dz: 1e-2,
        k_eta: 5.57e-6, k_m: 1.36e-7, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: 0.005, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w: 1.0,
    }
}

fn hover_state(p: &QuadParamsInput, x: f64, y: f64, z: f64) -> State<f64> {
    let hov = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();
    State {
        x: Vec3::new(x, y, z), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [hov; 4],
    }
}

const ROOF: f64 = 5.0;

// mirror wasm::random_course
fn sky_course(rng: &mut Rng) -> Course {
    let gates = (0..5)
        .map(|_| {
            let x = 1.5 + rng.uniform() * 9.0;
            let y = (rng.uniform() * 2.0 - 1.0) * 3.5;
            let z = 0.5 + rng.uniform() * (ROOF - 1.5);
            let nrm = Vec3::new(rng.normal(), rng.normal(), rng.normal());
            Gate::new(Vec3::new(x, y, z), nrm, 0.85)
        })
        .collect();
    Course::new(gates, false)
}

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn race_cfg(mass: f64) -> MppiConfig {
    let mut cfg = MppiConfig::for_mass(mass);
    cfg.samples = 128;
    cfg.w_vel = env("R_WVEL", 3.0);
    cfg.w_pos = env("R_WPOS", 0.0);
    cfg.w_vdamp = env("R_WVDAMP", 0.04);
    cfg.w_effort = env("R_WEFF", 0.01);
    cfg.sigma_rate = env("R_SIGR", 2.8);
    cfg.rate_max = env("R_RMAX", 10.0);
    cfg.beta = env("R_BETA", 0.90);
    cfg
}
fn hover_cfg(mass: f64) -> MppiConfig {
    let mut cfg = MppiConfig::for_mass(mass);
    cfg.samples = 128;
    cfg.w_vel = 0.0; cfg.gate_reward = 0.0; cfg.w_alt = 0.0; cfg.w_terminal = 0.0;
    cfg.w_pos = env("H_WPOS", 3.0);
    cfg.w_vdamp = env("H_WVDAMP", 0.8);
    cfg.rate_max = env("R_RMAX", 9.0);
    cfg.sigma_rate = env("H_SIGR", 0.5);
    cfg.sigma_thrust = env("H_SIGT", 0.03) * mass * 9.81;
    cfg.lambda = env("H_LAM", 1.0);
    cfg
}

fn main() {
    let mut input = hummingbird();
    input.k_w = 16.0;
    // SEED=N runs that ONE WasmRacer seed (matches the browser's debug snapshot);
    // otherwise sweep 8 deterministic trials.
    let one_seed = std::env::var("SEED").ok().and_then(|v| v.parse::<u64>().ok());
    let seeds: Vec<u64> = match one_seed {
        Some(s) => vec![s | 1],
        None => (0..8u64).map(|t| t * 2654435761 | 1).collect(),
    };
    let trials = seeds.len();
    let mut wins = 0;
    let mut hover_speeds = Vec::new();

    for (t, &seed) in seeds.iter().enumerate() {
        let mut rng = Rng::new(seed);
        let mut course = sky_course(&mut rng);
        let reality = Multirotor::<f64, Ctbr>::with_substeps(&input, 8);
        let mut ctrl = MppiController::new(TrueDynamics::new(&input, 8), race_cfg(input.mass), 42);
        let mut state = hover_state(&input, 0.0, 0.0, 1.5);
        let dt = 0.05;

        // race phase (cap 600 steps = 30 s); track stability via body-rate magnitude
        let mut won = false;
        let mut steps = 0;
        let mut wsum = 0.0;
        let mut wmax: f64 = 0.0;
        let mut spmax: f64 = 0.0;
        for _ in 0..600 {
            steps += 1;
            let a = ctrl.act(&state, &course);
            let prev = state.x;
            state = reality.step(&state, &a, dt);
            course.advance(prev, state.x);
            let wmag = state.w.norm();
            wsum += wmag;
            wmax = wmax.max(wmag);
            spmax = spmax.max(state.v.norm());
            if !state.x.x.is_finite() { break; } // crashed -> NaN
            if course.finished() { won = true; break; }
        }
        let wmean = wsum / steps as f64;
        if won {
            wins += 1;
            // hover phase: PD hold at finish point, measure terminal speed
            let hp = Vec3::new(state.x.x, state.x.y, state.x.z.max(1.3));
            let mut hcourse = Course::new(vec![Gate::new(hp, Vec3::new(1.0, 0.0, 0.0), 1.0)], false);
            ctrl.cfg = hover_cfg(input.mass);
            let mut last_speeds = Vec::new();
            for k in 0..80 {
                let a = ctrl.act(&state, &hcourse);
                state = reality.step(&state, &a, dt);
                let sp = state.v.norm();
                if k >= 60 { last_speeds.push(sp); } // settled tail (3..4 s)
            }
            let _ = &mut hcourse;
            let avg_tail: f64 = last_speeds.iter().sum::<f64>() / last_speeds.len() as f64;
            hover_speeds.push(avg_tail);
            let drift = (state.x - hp).norm();
            println!(
                "trial {t}: WON {steps}st ({:.1}s) · vmax {:.1} · ω mean {:.2}/max {:.1} rad/s · hover tail {:.3} m/s · drift {:.2} m",
                steps as f64 * dt, spmax, wmean, wmax, avg_tail, drift
            );
        } else {
            println!("trial {t}: did NOT finish in 30 s (gate {}/5)", course.next);
        }
    }

    let mean_hover = if hover_speeds.is_empty() { f64::NAN }
        else { hover_speeds.iter().sum::<f64>() / hover_speeds.len() as f64 };
    println!("\n== {wins}/{trials} courses won · mean hover tail-speed {:.3} m/s ==", mean_hover);
}
