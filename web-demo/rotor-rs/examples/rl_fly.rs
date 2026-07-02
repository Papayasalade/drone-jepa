//! Sanity: fly the WASM `RlPolicy` (PPO policy) natively on sky gates and confirm
//! it races — i.e. the Rust obs/MLP/denorm match the trained Python policy.
//!   cargo run --release --features jepa --example rl_fly

use rotor_rs::rl::RlPolicy;
use rotor_rs::rng::Rng;
use rotor_rs::{
    Controller, Course, Ctbr, Gate, Multirotor, QuadParamsInput, Quat, RotorForce, State, Vec3,
};

fn denv(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
/// Hummingbird by default; DRONE_* env overrides (same convention as rotor_fly).
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
        tau_m: denv("DRONE_TAU_M", 0.005), rotor_speed_min: 0.0, rotor_speed_max: 1500.0,
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

fn main() {
    // RL_BLOB env overrides the policy; RL_ROTOR=1 reads actions as per-rotor
    // forces (validates the rotor-force policy + `act_rotor` denorm path).
    let path = std::env::var("RL_BLOB")
        .unwrap_or_else(|_| format!("{}/assets/skyrl_ctbr.rlb", env!("CARGO_MANIFEST_DIR")));
    let blob = std::fs::read(&path).expect("rlb");
    let rotor = std::env::var("RL_ROTOR").map(|v| v == "1").unwrap_or(false);
    let input = hummingbird();
    let reality: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&input, 8);
    let reality_rf: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&input, 8);
    let f_max = input.k_eta * input.rotor_speed_max * input.rotor_speed_max;
    println!("blob: {path}  rotor={rotor}  mass={}", input.mass);
    let (mut wins, mut crashes, mut total_gates) = (0, 0, 0);
    let trials = 12;
    for t in 0..trials {
        let mut rng = Rng::new((t as u64) * 2654435761 | 1);
        let mut course = sky_course(&mut rng);
        let mut policy = RlPolicy::from_blob(&blob, input.mass);
        policy.set_f_max(f_max);
        let mut s = hover_state(&input, 1.5);
        let mut won = false;
        for _ in 0..500 {
            let prev = s.x;
            if rotor {
                let f = policy.act_rotor(&s, &course);
                s = reality_rf.step(&s, &f, 0.05);
            } else {
                let a = policy.act(&s, &course);
                s = reality.step(&s, &a, 0.05);
            }
            course.advance(prev, s.x);
            if !s.x.z.is_finite() || s.x.z < 0.15 { crashes += 1; break; }
            if course.finished() { won = true; break; }
        }
        total_gates += course.gates_passed();
        if won { wins += 1; }
        println!("trial {t}: gates {}/5  {}", course.gates_passed(),
            if won { "WON" } else if !s.x.z.is_finite() || s.x.z < 0.15 { "crashed" } else { "(timeout)" });
    }
    println!("\n== {wins}/{trials} courses WON, {crashes} crashes, {:.1} gates/race ==",
        total_gates as f64 / trials as f64);
}
