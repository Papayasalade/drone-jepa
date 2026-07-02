//! DAgger-style data collection: fly the (possibly bad) rotor-force JEPA-MPPI
//! on the true hummingbird dynamics over sky-gate races — exactly like
//! `rotor_fly` — and DUMP the visited (state, executed-action) distribution,
//! including pre-crash tumbles. Records are 40-step segments chopped BACKWARD
//! from each crash/segment end, so the informative failure dynamics are kept
//! and no record straddles a respawn teleport.
//!
//!   ROTOR_BLOB=assets/<model>.jblob cargo run --release --features jepa \
//!       --example dagger_collect -- <out.bin> <n_records> [seed_base]
//!
//! Output .bin: u32 n_records, u32 steps(=40), u32 n_act(=4), then f32
//! states (n*40*18) and f32 actions (n*40*4) — same layout as gen_dataset_rf.

use std::io::Write;

use rotor_rs::jepa::{JepaRollout, SkyJepaLite};
use rotor_rs::rng::Rng;
use rotor_rs::rotor_mppi::{RotorMppiConfig, RotorMppiController};
use rotor_rs::{Course, Gate, Multirotor, QuadParamsInput, Quat, RotorForce, State, Vec3};

const STEPS: usize = 40;

fn denv(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
/// Vehicle — hummingbird by default, overridable via DRONE_* env vars
/// (same convention as rotor_fly / jepa_fly / gen_dataset_rf).
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
fn rotor_race_cfg(mass: f64) -> RotorMppiConfig {
    let mut cfg = RotorMppiConfig::for_mass(mass);
    let b = &mut cfg.base;
    b.horizon = 12; b.samples = 48;
    b.w_vel = 0.8; b.w_terminal = 2.5; b.w_alt = 3.5; b.w_vdamp = 0.18;
    b.v_max = 3.5; b.w_speed = 1.5; b.w_floor = 600.0; b.w_effort = 1e-4;
    b.trust_lambda = 1000.0; b.beta = 0.85; b.lambda = 0.4;
    cfg.sigma_force = 0.22 * cfg.hover_force;
    cfg
}

fn state_to_18(s: &State<f64>) -> [f32; 18] {
    let r = s.q.to_rotmat();
    [
        s.x.x as f32, s.x.y as f32, s.x.z as f32, s.v.x as f32, s.v.y as f32, s.v.z as f32,
        r.rows[0][0] as f32, r.rows[0][1] as f32, r.rows[0][2] as f32,
        r.rows[1][0] as f32, r.rows[1][1] as f32, r.rows[1][2] as f32,
        r.rows[2][0] as f32, r.rows[2][1] as f32, r.rows[2][2] as f32,
        s.w.x as f32, s.w.y as f32, s.w.z as f32,
    ]
}

/// Chop a segment into 40-step records aligned to the segment END.
fn flush_segment(seg: &mut Vec<([f32; 18], [f32; 4])>,
                 states: &mut Vec<f32>, actions: &mut Vec<f32>, n_rec: &mut usize) {
    let n = seg.len();
    let blocks = n / STEPS;
    let start = n - blocks * STEPS; // drop the OLDEST leftover, keep the end
    for b in 0..blocks {
        for k in 0..STEPS {
            let (s, a) = seg[start + b * STEPS + k];
            states.extend_from_slice(&s);
            actions.extend_from_slice(&a);
        }
        *n_rec += 1;
    }
    seg.clear();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).expect("out.bin").clone();
    let want: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let seed_base: u64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(0);
    let path = std::env::var("ROTOR_BLOB").expect("set ROTOR_BLOB");
    let blob = std::fs::read(&path).expect("rotor jblob");
    let input = hummingbird();
    let reality: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&input, 8);

    let (mut states, mut actions) = (Vec::new(), Vec::new());
    let mut n_rec = 0usize;
    let mut seg: Vec<([f32; 18], [f32; 4])> = Vec::new();
    let mut trial = 0u64;
    let mut respawn_total = 0usize;
    while n_rec < want {
        let mut rng = Rng::new((trial + seed_base) .wrapping_mul(2654435761) | 1);
        let mut course = sky_course(&mut rng);
        let model = SkyJepaLite::from_blob(&blob);
        let mut ctrl = RotorMppiController::new(JepaRollout::new(model), rotor_race_cfg(input.mass), 11 + trial);
        // spawn HIGH: the crashing model tumbles within ~1-2 s of any respawn, so a
        // low spawn never yields a 40-step contiguous segment. From 20 m the fall
        // alone lasts ~2 s, so the tumble dynamics fit inside a record.
        let mut s = hover_state(&input, 20.0);
        let mut local_respawn = 0;
        for _ in 0..600 {
            let a = ctrl.act(&s, &course);
            seg.push((state_to_18(&s), [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]));
            let prev = s;
            s = reality.step(&s, &a, 0.05);
            ctrl.observe(&prev, a);
            course.advance(prev.x, s.x);
            if !s.x.z.is_finite() || s.x.z < 0.15 {
                flush_segment(&mut seg, &mut states, &mut actions, &mut n_rec);
                local_respawn += 1;
                respawn_total += 1;
                if local_respawn > 30 { break; }
                let p = s.x;
                let sx = if p.x.is_finite() { p.x.clamp(0.0, 11.0) } else { 0.0 };
                let sy = if p.y.is_finite() { p.y.clamp(-4.0, 4.0) } else { 0.0 };
                s = hover_state(&input, 20.0);
                s.x.x = sx; s.x.y = sy;
                ctrl.model.reset();
                ctrl.reset_nominal();
                continue;
            }
            if course.finished() { break; }
        }
        flush_segment(&mut seg, &mut states, &mut actions, &mut n_rec);
        trial += 1;
        if trial % 20 == 0 {
            println!("trial {trial}: {n_rec}/{want} records ({respawn_total} respawns so far)");
        }
    }

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    f.write_all(&(n_rec as u32).to_le_bytes()).unwrap();
    f.write_all(&(STEPS as u32).to_le_bytes()).unwrap();
    f.write_all(&4u32.to_le_bytes()).unwrap();
    for v in &states { f.write_all(&v.to_le_bytes()).unwrap(); }
    for v in &actions { f.write_all(&v.to_le_bytes()).unwrap(); }
    f.flush().unwrap();
    println!("wrote {out}: {n_rec} records x {STEPS} steps (from {trial} trials, {respawn_total} respawns)");
}
