//! Sanity: does the iLQR NMPC track a reference on the true dynamics?
//!   cargo run --release --example nmpc_track

use rotor_rs::{FourierRef, Multirotor, Nmpc, QuadParamsInput, Quat, RotorForce, State, Vec3};
use rotor_rs::rng::Rng;

fn params() -> QuadParamsInput {
    let d = 0.17 * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: 0.5, ixx: 3.65e-3, iyy: 3.68e-3, izz: 7.03e-3, ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: 0.5e-2, c_dy: 0.5e-2, c_dz: 1e-2,
        k_eta: 5.57e-6, k_m: 1.36e-7, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: 0.02, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w: 1.0,
    }
}

fn main() {
    let p = params();
    let dt = 0.05;
    let horizon = 15;
    let f_max = p.k_eta * p.rotor_speed_max * p.rotor_speed_max;
    let hov = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();

    let reality: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&p, 8);
    let mut nmpc = Nmpc::new(&p, dt, horizon);
    let mut r = Rng::new(7);
    let traj = FourierRef::sample(&mut r, Vec3::new(0.0, 0.0, 1.5));

    let mut s = State {
        x: Vec3::new(0.0, 0.0, 1.5), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [hov; 4],
    };

    let steps = 200;
    let mut err_sum = 0.0;
    let mut err_max = 0.0_f64;
    for k in 0..steps {
        let refs: Vec<_> = (0..horizon).map(|i| traj.at((k + i) as f64 * dt)).collect();
        let u = nmpc.update(&s, &refs);
        let f: [f64; 4] = core::array::from_fn(|j| u[j].clamp(0.0, f_max));
        s = reality.step(&s, &f, dt);
        let e = (s.x - traj.at(k as f64 * dt).x).norm();
        err_sum += e;
        err_max = err_max.max(e);
        if k % 40 == 0 {
            let rf = traj.at(k as f64 * dt).x;
            println!("  t={:4.2} pos=[{:5.2},{:5.2},{:5.2}] ref=[{:5.2},{:5.2},{:5.2}] err={:.3}",
                k as f64 * dt, s.x.x, s.x.y, s.x.z, rf.x, rf.y, rf.z, e);
        }
        if !s.x.z.is_finite() || s.x.z < -2.0 { println!("  diverged at t={:.2}", k as f64 * dt); break; }
    }
    println!("\nNMPC tracking: mean err={:.3} m  max err={:.3} m", err_sum / steps as f64, err_max);
}
