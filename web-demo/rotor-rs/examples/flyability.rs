//! How flyable is the wide drone distribution (mass 0.2-2kg, asymmetric arms
//! 5-40cm, per-motor power ±30%)? For each sampled drone we report thrust-to-weight
//! and whether a near-perfect controller (true-dynamics MPPI) can HOVER it.
//!   cargo run --release --example flyability

use rotor_rs::rng::Rng;
use rotor_rs::{
    Controller, Course, Ctbr, Gate, MppiConfig, MppiController, Multirotor, QuadParamsInput,
    Quat, State, TrueDynamics, Vec3, GRAVITY,
};

const HUM_IXX: f64 = 3.65e-3; const HUM_IYY: f64 = 3.68e-3; const HUM_IZZ: f64 = 7.03e-3;
const HUM_KETA: f64 = 5.57e-6; const HUM_KM: f64 = 1.36e-7;

fn lerp(r: &mut Rng, a: f64, b: f64) -> f64 { a + (b - a) * r.uniform() }

fn sample_drone(r: &mut Rng) -> (QuadParamsInput, [f64; 4]) {
    let arms: [f64; 4] = core::array::from_fn(|_| lerp(r, 0.05, 0.40));
    let d: [f64; 4] = core::array::from_fn(|i| arms[i] * std::f64::consts::FRAC_1_SQRT_2);
    let rotor_pos = [[d[0], d[0], 0.0], [d[1], -d[1], 0.0], [-d[2], -d[2], 0.0], [-d[3], d[3], 0.0]];
    let avg_arm = arms.iter().sum::<f64>() / 4.0;
    let mass = lerp(r, 0.2, 2.0);
    let rpm_max = 1500.0;
    let twr = lerp(r, 2.0, 4.0);
    let k_eta = twr * mass * GRAVITY / (4.0 * rpm_max * rpm_max);
    let k_m = k_eta * (HUM_KM / HUM_KETA) * lerp(r, 0.7, 1.3);
    let i_scale = (mass / 0.5) * (avg_arm / 0.17).powi(2) * lerp(r, 0.7, 1.3);
    let input = QuadParamsInput {
        mass,
        ixx: HUM_IXX * i_scale, iyy: HUM_IYY * i_scale, izz: HUM_IZZ * i_scale,
        ixy: 0.0, iyz: 0.0, ixz: 0.0, rotor_pos, rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: lerp(r, 0.02, 0.30), c_dy: lerp(r, 0.02, 0.30), c_dz: lerp(r, 0.05, 0.40),
        k_eta, k_m,
        k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: lerp(r, 0.01, 0.04), rotor_speed_min: 0.0, rotor_speed_max: rpm_max,
        k_w: lerp(r, 6.0, 18.0),
    };
    (input, [1.0; 4])
}

fn main() {
    let n = 600;
    let mut rng = Rng::new(20240626);
    let (mut flew, mut twr_lt12, mut twr_lt15) = (0, 0, 0);
    let mut min_twr_flew = f64::INFINITY;
    let mut max_twr_failed = 0.0f64;
    for _ in 0..n {
        let (input, gain) = sample_drone(&mut rng);
        // thrust-to-weight with the weakest motor's gain (conservative)
        let g_avg = gain.iter().sum::<f64>() / 4.0;
        let max_thrust = 4.0 * g_avg * input.k_eta * input.rotor_speed_max * input.rotor_speed_max;
        let twr = max_thrust / (input.mass * GRAVITY);
        if twr < 1.2 { twr_lt12 += 1; }
        if twr < 1.5 { twr_lt15 += 1; }

        // can the true-dynamics MPPI HOLD a hover for 4 s?
        let mut reality: Multirotor<f64, Ctbr> = Multirotor::with_substeps(&input, 8);
        reality.set_rotor_gain(gain);
        let mut model = TrueDynamics::new(&input, 8);
        model.veh.set_rotor_gain(gain);
        let mut cfg = MppiConfig::for_mass(input.mass);
        cfg.samples = 96; cfg.w_pos = 4.0; cfg.w_vdamp = 0.4; cfg.w_vel = 0.0;
        cfg.gate_reward = 0.0; cfg.w_floor = 300.0;
        let mut ctrl = MppiController::new(model, cfg, 1);
        let hov = (input.mass * GRAVITY / (4.0 * input.k_eta)).sqrt();
        let mut s = State { x: Vec3::new(0.0, 0.0, 3.0), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
            w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [hov; 4] };
        // hold at (0,0,3): a single gate there as the MPPI target
        let course = Course::new(vec![Gate::new(Vec3::new(0.0, 0.0, 3.0), Vec3::new(1.0, 0.0, 0.0), 1.0)], false);
        let mut ok = true;
        for _ in 0..80 {
            let a = ctrl.act(&s, &course);
            s = reality.step(&s, &a, 0.05);
            if !s.x.z.is_finite() || (s.x - Vec3::new(0.0, 0.0, 3.0)).norm() > 5.0 { ok = false; break; }
        }
        if ok { flew += 1; min_twr_flew = min_twr_flew.min(twr); }
        else { max_twr_failed = max_twr_failed.max(twr); }
    }
    println!("over {n} randomized drones:");
    println!("  HOVERABLE by true-MPPI: {} ({:.0}%)", flew, 100.0 * flew as f64 / n as f64);
    println!("  thrust-to-weight < 1.2: {} ({:.0}%)   < 1.5: {} ({:.0}%)",
        twr_lt12, 100.0 * twr_lt12 as f64 / n as f64, twr_lt15, 100.0 * twr_lt15 as f64 / n as f64);
    println!("  lowest TWR that flew: {:.2}   highest TWR that failed: {:.2}", min_twr_flew, max_twr_failed);
}
