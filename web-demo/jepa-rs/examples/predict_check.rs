//! Open-loop sanity: does the JEPA model predict the true dynamics for a gentle
//! maneuver at k_w=1 (in-distribution)? Compares JEPA rollout vs RotorPy rollout.
//!
//!   cargo run --release --example predict_check

use std::path::Path;

use jepa_rs::rollout::JepaRollout;
use jepa_rs::SkyJepa;
use rotor_rs::mppi::RolloutModel;
use rotor_rs::{Ctbr, CtbrCmd, Multirotor, QuadParamsInput, Quat, State, Vec3};

fn params() -> QuadParamsInput {
    let d = 0.17 * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: 0.5, ixx: 3.65e-3, iyy: 3.68e-3, izz: 7.03e-3, ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: 0.5e-2, c_dy: 0.5e-2, c_dz: 1e-2,
        k_eta: 5.57e-6, k_m: 1.36e-7, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: 0.005, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w: 1.0,
    }
}

fn hover(p: &QuadParamsInput) -> State<f64> {
    let h = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();
    State {
        x: Vec3::new(0.0, 0.0, 1.5), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [h; 4],
    }
}

fn run(label: &str, w_pitch: f64) {
    let p = params();
    let dt = 0.05;
    let hover_thrust = p.mass * 9.81;
    let t = 20usize;
    let seq: Vec<CtbrCmd<f64>> =
        (0..t).map(|_| CtbrCmd { thrust: hover_thrust, w_cmd: Vec3::new(0.0, w_pitch, 0.0) }).collect();

    // true dynamics
    let veh: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&p, 8);
    let mut s = hover(&p);
    let mut truth = Vec::new();
    for a in &seq {
        s = veh.step(&s, a, dt);
        truth.push(s.x);
    }

    // jepa model (seed history with hover)
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = SkyJepa::load(
        root.join("weights/skyjepa_ctbr_1x.safetensors").to_str().unwrap(),
        root.join("weights/skyjepa_ctbr_1x.json").to_str().unwrap(),
    ).unwrap();
    let mut jr = JepaRollout::new(model);
    let hov_a = CtbrCmd { thrust: hover_thrust, w_cmd: Vec3::zero() };
    for _ in 0..12 {
        jr.observe(&hover(&p), &hov_a);
    }
    let pred = jr.rollout_batch(&hover(&p), &[seq.clone()], dt).pop().unwrap();

    println!("\n== {label} (pitch rate {w_pitch}) ==");
    println!("  step   truth(x,y,z)            jepa(x,y,z)             |err|");
    for k in (0..t).step_by(4) {
        let tr = truth[k];
        let pr = pred[k];
        let e = ((tr.x - pr.x).powi(2) + (tr.y - pr.y).powi(2) + (tr.z - pr.z).powi(2)).sqrt();
        println!(
            "  {k:3}  [{:6.2},{:6.2},{:6.2}]   [{:6.2},{:6.2},{:6.2}]   {e:.3}",
            tr.x, tr.y, tr.z, pr.x, pr.y, pr.z
        );
    }
    let last_e = {
        let (tr, pr) = (truth[t - 1], pred[t - 1]);
        ((tr.x - pr.x).powi(2) + (tr.y - pr.y).powi(2) + (tr.z - pr.z).powi(2)).sqrt()
    };
    println!("  final |err| = {last_e:.3} m");
}

fn main() {
    run("hover", 0.0);
    run("gentle pitch", 0.5);
    run("stronger pitch", 1.5);
}
