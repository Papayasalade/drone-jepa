//! Feasibility probe: can MPPI fly the gentle course SLOWLY on the hard k_w=1
//! plant (the regime the JEPA model is trained for) using the TRUE dynamics?
//! If yes, the same config should transfer to JEPA-MPPI.
//!
//!   cargo run --release --example race_gentle

use racer::{
    Controller, Course, Ctbr, Gate, MppiConfig, MppiController, Multirotor, QuadParamsInput, Quat,
    State, TrueDynamics, Vec3,
};

fn hummingbird(k_w: f64) -> QuadParamsInput {
    let d = 0.17 * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: 0.5, ixx: 3.65e-3, iyy: 3.68e-3, izz: 7.03e-3, ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: 0.5e-2, c_dy: 0.5e-2, c_dz: 1e-2,
        k_eta: 5.57e-6, k_m: 1.36e-7, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: 0.005, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w,
    }
}

fn hover_state(p: &QuadParamsInput, x: f64, y: f64, z: f64) -> State<f64> {
    let hov = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();
    State {
        x: Vec3::new(x, y, z), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [hov; 4],
    }
}

fn main() {
    let params = hummingbird(1.0); // model's training plant
    let dt = 0.05;

    let gates = vec![
        Gate::new(Vec3::new(2.0, 0.0, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(4.0, 0.8, 1.6), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(6.0, -0.8, 1.4), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(8.0, 0.0, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.9),
    ];
    let n_gates = gates.len();
    let mut course = Course::new(gates, false);

    let reality: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&params, 8);
    let model = TrueDynamics::new(&params, 8);

    let mut cfg = MppiConfig::for_mass(params.mass);
    cfg.horizon = 20;
    cfg.sigma_rate = 2.5;
    cfg.sigma_thrust = 0.15 * cfg.hover_thrust;
    cfg.w_vel = 1.0;
    cfg.w_terminal = 2.5;
    cfg.w_alt = 3.0;
    cfg.lambda = 0.4;
    cfg.v_max = 2.0; // creep through the gates at <= 2 m/s
    cfg.w_speed = 8.0;
    let mut ctrl = MppiController::new(model, cfg, 42);

    let mut s = hover_state(&params, 0.0, 0.0, 1.5);
    let max_steps = 800; // 40 s — slow plant
    let mut max_speed = 0.0_f64;
    for step in 0..max_steps {
        let a = ctrl.act(&s, &course);
        let prev = s.x;
        s = reality.step(&s, &a, dt);
        max_speed = max_speed.max(s.v.norm());
        if course.advance(prev, s.x) {
            println!(
                "  t={:5.2}s  gate {}/{n_gates}  pos=[{:.2},{:.2},{:.2}]  |v|={:.2}",
                step as f64 * dt, course.gates_passed(), s.x.x, s.x.y, s.x.z, s.v.norm()
            );
        }
        if !s.x.z.is_finite() || s.x.z < -1.0 {
            println!("  crashed at t={:.2}s", step as f64 * dt);
            break;
        }
        if course.finished() {
            println!("\nFINISHED {n_gates} gates in {:.2}s (max speed {max_speed:.1} m/s)", (step + 1) as f64 * dt);
            return;
        }
    }
    println!("\npassed {}/{n_gates} (k_w=1 true dynamics, max speed {max_speed:.1} m/s)", course.gates_passed());
}
