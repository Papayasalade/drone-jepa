//! Sanity: fly the rotor-force JEPA drone (4 independent rotor forces planned by
//! `RotorMppiController` over the rotor-force SkyJEPA) natively on sky gates, and
//! confirm it stays airborne + passes gates. This is the demo's 4th drone — it
//! validates the whole rotor-force path end-to-end without the browser.
//!   cargo run --release --features jepa --example rotor_fly
//!
//! Override the blob to test a freshly-trained checkpoint:
//!   ROTOR_BLOB=assets/bigquad_unified.jblob cargo run --release --features jepa --example rotor_fly

use racer::jepa::{JepaRollout, SkyJepaLite};
use racer::rng::Rng;
use racer::rotor_mppi::{RotorMppiConfig, RotorMppiController};
use racer::{Course, Gate, Multirotor, QuadParamsInput, Quat, RotorForce, State, Vec3, GRAVITY};

fn denv(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
/// Vehicle under test — hummingbird by default, overridable via DRONE_* env
/// vars (single source of truth: a drone-spec JSON consumed by train_recipe).
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

// Mirror of wasm::rotor_race_cfg (the demo's deployed tuning).
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

fn main() {
    let path = std::env::var("ROTOR_BLOB")
        .unwrap_or_else(|_| format!("{}/assets/skyjepa_rotor.jblob", env!("CARGO_MANIFEST_DIR")));
    let blob = std::fs::read(&path).expect("rotor jblob");
    let input = hummingbird();
    let reality: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&input, 8);
    let _ = GRAVITY;
    let (mut wins, mut crashes, mut total_gates, mut respawns) = (0, 0, 0usize, 0usize);
    let (mut inv_steps, mut tot_steps) = (0usize, 0usize); // R_zz<0 = past 90° (flipping)
    let trials = 12;
    println!("blob: {path}");
    for t in 0..trials {
        let mut rng = Rng::new((t as u64) * 2654435761 | 1);
        let mut course = sky_course(&mut rng);
        let model = SkyJepaLite::from_blob(&blob);
        let mut ctrl = RotorMppiController::new(JepaRollout::new(model), rotor_race_cfg(input.mass), 11);
        let mut s = hover_state(&input, 1.5);
        let mut won = false;
        let mut local_respawn = 0;
        for _ in 0..600 {
            let a = ctrl.act(&s, &course);
            let prev = s;
            s = reality.step(&s, &a, 0.05);
            ctrl.observe(&prev, a);
            tot_steps += 1;
            if s.q.to_rotmat().rows[2][2] < 0.0 { inv_steps += 1; } // past 90° = flipping
            course.advance(prev.x, s.x);
            if !s.x.z.is_finite() || s.x.z < 0.15 {
                // respawn-and-retry, same as the demo
                local_respawn += 1;
                if local_respawn > 8 { crashes += 1; break; }
                let p = s.x;
                let sx = if p.x.is_finite() { p.x.clamp(0.0, 11.0) } else { 0.0 };
                let sy = if p.y.is_finite() { p.y.clamp(-4.0, 4.0) } else { 0.0 };
                s = hover_state(&input, 1.5);
                s.x.x = sx; s.x.y = sy;
                ctrl.model.reset();
                ctrl.reset_nominal();
                continue;
            }
            if course.finished() { won = true; break; }
        }
        respawns += local_respawn;
        total_gates += course.gates_passed();
        if won { wins += 1; }
        println!("trial {t}: gates {}/5  respawns {local_respawn}  {}",
            course.gates_passed(), if won { "WON" } else { "(timeout/crash)" });
    }
    println!("\n== {wins}/{trials} WON, {crashes} hard-crashed, {respawns} respawns, {:.1} gates/race, {:.1}% steps flipped ==",
        total_gates as f64 / trials as f64, 100.0 * inv_steps as f64 / tot_steps.max(1) as f64);
}
