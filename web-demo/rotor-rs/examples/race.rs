//! Autonomous gate racing: a drone flies a course of gates using CTBR-MPPI over
//! the true dynamics (the JEPA world model will drop in as the rollout model later).
//!
//!   cargo run --release --example race

use rotor_rs::{
    Controller, Course, Ctbr, Gate, MppiConfig, MppiController, Multirotor, QuadParamsInput, Quat,
    State, TrueDynamics, Vec3,
};

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
        x: Vec3::new(x, y, z),
        v: Vec3::zero(),
        q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(),
        wind: Vec3::zero(),
        rotor_speeds: [hov; 4],
    }
}

fn main() {
    let mut params = hummingbird();
    // Racing needs a FAST inner rate loop. The stock hummingbird k_w=1 gives a
    // ~1 s body-rate time constant — far too sluggish to pitch-and-go within the
    // MPPI horizon. Real racers run aggressive rate gains.
    params.k_w = 16.0;
    let dt = 0.05; // 20 Hz control

    // A short slalom course: gates marching out in +x, weaving in y, all faced +x.
    let gates = vec![
        Gate::new(Vec3::new(3.0, 0.0, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.8),
        Gate::new(Vec3::new(6.0, 1.5, 1.8), Vec3::new(1.0, 0.0, 0.0), 0.8),
        Gate::new(Vec3::new(9.0, -1.5, 1.2), Vec3::new(1.0, 0.0, 0.0), 0.8),
        Gate::new(Vec3::new(12.0, 0.0, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.8),
    ];
    let n_gates = gates.len();
    let mut course = Course::new(gates, false);

    // Reality (high-accuracy) and the MPPI rollout model (cheaper). Both true dynamics.
    let reality: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&params, 8);
    // The rollout model needs fine enough substeps to stay stable under the stiff
    // k_w rate loop + fast tau_m — too coarse and every rollout diverges to NaN.
    let model = TrueDynamics::new(&params, 8);
    let cfg = MppiConfig::for_mass(params.mass);
    let mut ctrl = MppiController::new(model, cfg, 42);

    let mut s = hover_state(&params, 0.0, 0.0, 1.5);
    let max_steps = 600; // 30 s budget

    println!("course: {n_gates} gates | MPPI: S={} T={} dt={dt}", cfg.samples, cfg.horizon);
    let mut t_done = None;
    for step in 0..max_steps {
        let a = ctrl.act(&s, &course);
        let prev = s.x;
        s = reality.step(&s, &a, dt);
        if course.advance(prev, s.x) {
            let g = course.gates_passed();
            println!(
                "  t={:5.2}s  passed gate {g}/{n_gates}  pos=[{:.2},{:.2},{:.2}]  |v|={:.2}",
                step as f64 * dt, s.x.x, s.x.y, s.x.z, s.v.norm()
            );
        }
        if !s.x.z.is_finite() || s.x.z < -1.0 {
            println!("  crashed at t={:.2}s", step as f64 * dt);
            break;
        }
        if course.finished() {
            t_done = Some((step + 1) as f64 * dt);
            break;
        }
    }

    match t_done {
        Some(t) => println!("\nFINISHED all {n_gates} gates in {t:.2}s  (avg {:.1} m/s)", 12.0 / t),
        None => println!("\npassed {}/{n_gates} gates in 30s", course.gates_passed()),
    }
}
