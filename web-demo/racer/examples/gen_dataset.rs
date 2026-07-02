//! Generate closed-loop MPPI training data (the paper's "complementary action
//! distributions") for retraining SkyJEPA — using the fast Rust racer.
//!
//! Each trajectory: a domain-randomized drone (mass/inertia/motors/frame AND k_w,
//! so the model learns to infer dynamics from history) racing a random looping gate
//! course under MPPI, at a randomized speed cap (slow->fast), with some wind. Logs
//! state(18) + action(CTBR 4) at 20 Hz to a flat binary.
//!
//!   cargo run --release --example gen_dataset -- <n_traj> <steps> <out.bin> <seed>

use std::io::Write;

use racer::control::ControlLaw;
use racer::multirotor::clip_speeds;
use racer::rng::Rng;
use racer::{
    Controller, Course, Ctbr, Gate, MppiConfig, MppiController, Multirotor, QuadParams,
    QuadParamsInput, Quat, State, TrueDynamics, Vec3,
};

const HUM_IXX: f64 = 3.65e-3;
const HUM_IYY: f64 = 3.68e-3;
const HUM_IZZ: f64 = 7.03e-3;
const HUM_KETA: f64 = 5.57e-6;
const HUM_KM: f64 = 1.36e-7;

fn lerp(r: &mut Rng, a: f64, b: f64) -> f64 {
    a + (b - a) * r.uniform()
}

/// Domain-randomized drone: asymmetric arms (5-40 cm each), mass 0.2-2 kg, and
/// `k_eta` derived from a sampled thrust-to-weight (2-4) so every drone is flyable.
/// k_w varies so the encoder must infer the effective dynamics from history.
fn envf(key: &str, d: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn sample_drone(r: &mut Rng) -> (QuadParamsInput, [f64; 4]) {
    // distribution knobs (env-overridable for experiments): mass range, arm range,
    // and SYM=1 to make all four arms the same length (symmetric frame).
    let (arm_lo, arm_hi) = (envf("ARM_LO", 0.05), envf("ARM_HI", 0.40));
    let sym = envf("SYM", 0.0) != 0.0;
    let arms: [f64; 4] = if sym {
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
    let twr = lerp(r, 2.0, 4.0);
    let k_eta = twr * mass * 9.81 / (4.0 * rpm_max * rpm_max);
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

fn random_course(r: &mut Rng) -> Course {
    let n = 5 + (r.uniform() * 4.0) as usize; // 5..8 gates
    let gates = (0..n)
        .map(|_| {
            // wide altitude span so the model is in-distribution from near-ground
            // up to the demo's 5-15 m "sky" gates (absolute position is a model input).
            let c = Vec3::new(lerp(r, -6.0, 6.0), lerp(r, -6.0, 6.0), lerp(r, 0.8, 16.0));
            // normal roughly horizontal, random heading
            let ang = lerp(r, 0.0, std::f64::consts::TAU);
            let nrm = Vec3::new(ang.cos(), ang.sin(), lerp(r, -0.2, 0.2));
            Gate::new(c, nrm, lerp(r, 0.8, 1.2))
        })
        .collect();
    Course::new(gates, true) // loop so the drone keeps racing for the full window
}

fn hover_state(p: &QuadParamsInput, r: &mut Rng) -> State<f64> {
    let hov = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();
    State {
        x: Vec3::new(lerp(r, -1.0, 1.0), lerp(r, -1.0, 1.0), lerp(r, 1.0, 12.0)),
        v: Vec3::zero(),
        q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(),
        wind: Vec3::zero(),
        rotor_speeds: [hov; 4],
    }
}

fn state_to_18(s: &State<f64>) -> [f32; 18] {
    let r = s.q.to_rotmat();
    [
        s.x.x as f32, s.x.y as f32, s.x.z as f32,
        s.v.x as f32, s.v.y as f32, s.v.z as f32,
        r.rows[0][0] as f32, r.rows[0][1] as f32, r.rows[0][2] as f32,
        r.rows[1][0] as f32, r.rows[1][1] as f32, r.rows[1][2] as f32,
        r.rows[2][0] as f32, r.rows[2][1] as f32, r.rows[2][2] as f32,
        s.w.x as f32, s.w.y as f32, s.w.z as f32,
    ]
}

/// One trajectory; returns None if it diverged (caller retries).
// Per-trajectory dynamics params recorded for param-conditioning experiments.
const NP: usize = 10;
fn drone_params(p: &QuadParamsInput) -> [f32; NP] {
    let arm = |rp: &[f64; 3]| (rp[0] * rp[0] + rp[1] * rp[1]).sqrt();
    let avg_arm = p.rotor_pos.iter().map(arm).sum::<f64>() / 4.0;
    [
        p.mass as f32, p.ixx as f32, p.iyy as f32, p.izz as f32,
        p.k_eta as f32, p.k_m as f32, p.tau_m as f32, p.k_w as f32,
        p.c_dz as f32, avg_arm as f32,
    ]
}
fn one_traj(r: &mut Rng, steps: usize) -> Option<(Vec<f32>, Vec<f32>, [f32; NP])> {
    let (params, gain) = sample_drone(r);
    let dt = 0.05;
    let mut reality: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&params, 8);
    reality.set_rotor_gain(gain); // per-motor power
    let mut model = TrueDynamics::new(&params, 2); // cheaper rollout model (expert planner)
    model.veh.set_rotor_gain(gain);
    let mut cfg = MppiConfig::for_mass(params.mass);
    cfg.horizon = 20;
    cfg.samples = 128;
    cfg.v_max = lerp(r, 2.0, 10.0); // slow -> fast across trajectories
    cfg.w_speed = 8.0;
    let mut ctrl = MppiController::new(model, cfg, (r.uniform() * 1e9) as u64 + 1);

    let mut course = random_course(r);
    let wind = Vec3::new(lerp(r, -2.0, 2.0), lerp(r, -2.0, 2.0), lerp(r, -1.0, 1.0));
    let mut s = hover_state(&params, r);
    s.wind = wind;

    // for dual labeling: realized rotor forces from the CTBR command
    let qp = QuadParams::<f64>::from_input(&params);

    let mut states = Vec::with_capacity(steps * 18);
    let mut actions = Vec::with_capacity(steps * 8);
    for _ in 0..steps {
        let a = ctrl.act(&s, &course);
        // realized per-rotor forces: speeds from the CTBR command, clipped, f=k_eta*w^2
        let speeds = clip_speeds(&qp, Ctbr::cmd_rotor_speeds(&qp, &s, &a));
        let f: [f64; 4] = core::array::from_fn(|j| params.k_eta * speeds[j] * speeds[j]);
        states.extend_from_slice(&state_to_18(&s));
        actions.extend_from_slice(&[
            f[0] as f32, f[1] as f32, f[2] as f32, f[3] as f32,
            a.thrust as f32, a.w_cmd.x as f32, a.w_cmd.y as f32, a.w_cmd.z as f32,
        ]);
        let prev = s.x;
        s = reality.step(&s, &a, dt);
        s.wind = wind; // hold wind constant
        course.advance(prev, s.x);
        if !s.x.z.is_finite() || s.x.z.abs() > 50.0 || s.v.norm() > 40.0 {
            return None; // diverged
        }
    }
    Some((states, actions, drone_params(&params)))
}

/// Generate `chunk` valid trajectories on one thread with its own RNG.
fn gen_chunk(seed: u64, chunk: usize, steps: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Rng::new(seed);
    let mut st = Vec::with_capacity(chunk * steps * 18);
    let mut ac = Vec::with_capacity(chunk * steps * 8);
    let mut pr = Vec::with_capacity(chunk * NP);
    let mut kept = 0;
    while kept < chunk {
        if let Some((s, a, p)) = one_traj(&mut r, steps) {
            st.extend_from_slice(&s);
            ac.extend_from_slice(&a);
            pr.extend_from_slice(&p);
            kept += 1;
        }
    }
    (st, ac, pr)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_traj: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(800);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let out = args.get(3).cloned().unwrap_or_else(|| "artifacts/racing_ctbr.bin".into());
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);

    // parallelize across cores (trajectories are independent)
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let base = n_traj / n_threads;
    let rem = n_traj % n_threads;
    let chunks: Vec<usize> = (0..n_threads).map(|i| base + usize::from(i < rem)).collect();
    println!("collecting {n_traj} traj across {n_threads} threads...");
    let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = chunks
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| {
                let s = seed.wrapping_mul(1_000_003).wrapping_add(i as u64 + 1);
                sc.spawn(move || gen_chunk(s, c, steps))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut all_states = Vec::with_capacity(n_traj * steps * 18);
    let mut all_actions = Vec::with_capacity(n_traj * steps * 8);
    let mut all_params = Vec::with_capacity(n_traj * NP);
    for (st, ac, pr) in results {
        all_states.extend_from_slice(&st);
        all_actions.extend_from_slice(&ac);
        all_params.extend_from_slice(&pr);
    }

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    f.write_all(&(n_traj as u32).to_le_bytes()).unwrap();
    f.write_all(&(steps as u32).to_le_bytes()).unwrap();
    f.write_all(&8u32.to_le_bytes()).unwrap(); // n_act cols (4 rotor-force + 4 CTBR)
    for v in &all_states {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    for v in &all_actions {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    // sidecar: per-trajectory dynamics params (non-breaking — main .bin unchanged).
    // format: u32 n_traj, u32 NP, then n_traj*NP f32 (row-major).
    let pout = format!("{out}.params");
    let mut pf = std::io::BufWriter::new(std::fs::File::create(&pout).unwrap());
    pf.write_all(&(n_traj as u32).to_le_bytes()).unwrap();
    pf.write_all(&(NP as u32).to_le_bytes()).unwrap();
    for v in &all_params {
        pf.write_all(&v.to_le_bytes()).unwrap();
    }
    pf.flush().unwrap();
    println!("wrote {out}: {n_traj} traj x {steps} steps, 8 action cols (+{pout}: {NP} params)");
}
