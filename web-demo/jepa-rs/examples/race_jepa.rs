//! Autonomous gate racing with the LEARNED world model in the loop (JEPA-MPPI):
//! MPPI plans by rolling out the Candle SkyJEPA model instead of the true sim.
//!
//!   cargo run --release --example race_jepa
//!
//! NOTE: the bundled checkpoint was trained on gentle SE3-controller data, so it
//! is out-of-distribution for aggressive racing — expect it to fly worse than the
//! true-dynamics racer (rotor-rs `race`). That gap is exactly what retraining on
//! racing-distribution data fixes.

use std::path::Path;

use jepa_rs::rollout::JepaRollout;
use jepa_rs::SkyJepa;
use rotor_rs::{
    Controller, Course, Ctbr, Gate, MppiConfig, MppiController, Multirotor, QuadParamsInput, Quat,
    State, Vec3,
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
        // k_w within the retrained model's DR range [6,18] (the encoder infers the
        // effective rate-loop from history). Racing outside this range is OOD.
        tau_m: 0.005, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w: 12.0,
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
    let params = hummingbird();
    let dt = 0.05;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = SkyJepa::load(
        root.join("weights/skyjepa_ctbr_1x.safetensors").to_str().unwrap(),
        root.join("weights/skyjepa_ctbr_1x.json").to_str().unwrap(),
    )
    .expect("load model (run scripts/export_jepa.py)");
    let t_model = model.config().horizon;
    println!("JEPA-MPPI: learned model in the loop (H={}, T={})", model.config().history, t_model);

    // Gentle, close course: k_w=1 is sluggish, so keep gates near and give time.
    let gates = vec![
        Gate::new(Vec3::new(2.0, 0.0, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(4.0, 0.8, 1.6), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(6.0, -0.8, 1.4), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(8.0, 0.0, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.9),
    ];
    let n_gates = gates.len();
    let mut course = Course::new(gates, false);

    let reality: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&params, 8);

    let mut cfg = MppiConfig::for_mass(params.mass);
    cfg.horizon = t_model; // MPPI horizon must match the model's prediction length
    cfg.samples = 96; // candle rollout is heavier than the true sim
    cfg.sigma_rate = 2.5;
    cfg.sigma_thrust = 0.15 * cfg.hover_thrust;
    cfg.w_vel = 1.0;
    cfg.w_terminal = 2.5;
    cfg.w_alt = 3.0;
    cfg.lambda = 0.4;
    // No speed fence: the model is retrained on closed-loop MPPI data (DR over
    // drones/k_w, slow+fast, wind) so it's in-distribution for racing.
    let mut ctrl = MppiController::new(JepaRollout::new(model), cfg, 42);

    let mut s = hover_state(&params, 0.0, 0.0, 1.5);
    let max_steps = 600; // 30 s budget — k_w=1 flight is slow

    for step in 0..max_steps {
        let a = ctrl.act(&s, &course);
        let prev_state = s; // full state we planned from (Copy)
        let prev = s.x;
        s = reality.step(&s, &a, dt);
        ctrl.observe(&prev_state, &a); // record (state, action) for the model's history
        if course.advance(prev, s.x) {
            println!(
                "  t={:5.2}s  passed gate {}/{n_gates}  pos=[{:.2},{:.2},{:.2}]  |v|={:.2}",
                step as f64 * dt, course.gates_passed(), s.x.x, s.x.y, s.x.z, s.v.norm()
            );
        }
        if !s.x.z.is_finite() || s.x.z < -1.0 {
            println!("  crashed at t={:.2}s", step as f64 * dt);
            break;
        }
        if course.finished() {
            println!("\nFINISHED all {n_gates} gates in {:.2}s", (step + 1) as f64 * dt);
            return;
        }
    }
    println!("\npassed {}/{n_gates} gates (JEPA model, OOD for racing)", course.gates_passed());
}
