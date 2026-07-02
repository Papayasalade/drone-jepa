//! Benchmark the ported `Multirotor.step`, scalar and SIMD-batched. Build in
//! release with native codegen so the branchless kernel auto-vectorizes:
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench
//!
//! Reports steps/s (scalar) and drone-steps/s (batched F64x<L>), comparable to
//! `scripts/bench_sim.py` (Python RotorPy).

use std::hint::black_box;
use std::time::Instant;

use rotor_rs::control::{ControlLaw, RotorForce};
use rotor_rs::multirotor::{clip_speeds, integrate};
use rotor_rs::simd::{pack_params, pack_rotor_forces, pack_state, F64x};
use rotor_rs::{Multirotor, QuadParams, QuadParamsInput, Quat, Scalar, State, Vec3};

fn hummingbird() -> QuadParamsInput {
    let d = 0.17 * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: 0.5,
        ixx: 3.65e-3, iyy: 3.68e-3, izz: 7.03e-3,
        ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: 0.5e-2, c_dy: 0.5e-2, c_dz: 1e-2,
        k_eta: 5.57e-6, k_m: 1.36e-7, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: 0.005, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w: 1.0,
    }
}

fn hover_state(p: &QuadParamsInput) -> State<f64> {
    let hov = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();
    State {
        x: Vec3::new(0.0, 0.0, 1.5),
        v: Vec3::zero(),
        q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(),
        wind: Vec3::zero(),
        rotor_speeds: [hov; 4],
    }
}

const DT: f64 = 0.005;
const HOVER_F: f64 = 0.5 * 9.81 / 4.0;

fn bench_scalar(n_sub: usize, n: usize) -> f64 {
    let veh: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&hummingbird(), n_sub);
    let cmd = black_box([HOVER_F; 4]);
    let mut s = black_box(hover_state(&hummingbird()));
    for _ in 0..1000 {
        s = veh.step(&s, &cmd, DT);
    }
    let t0 = Instant::now();
    for _ in 0..n {
        // loop-carried dependency on s prevents hoisting; inputs black_box'd above.
        s = veh.step(&s, &cmd, DT);
    }
    let el = t0.elapsed().as_secs_f64();
    black_box(s);
    let sps = n as f64 / el;
    println!(
        "[scalar f64 ] N_SUB={n_sub:<2} {sps:>14.0} steps/s   {:6.1} ns/step   {:.0}x real-time@200Hz",
        el / n as f64 * 1e9,
        sps / 200.0,
    );
    sps
}

fn bench_batch<const L: usize>(n_sub: usize, n: usize, scalar_sps: f64) {
    let input = hummingbird();
    let scal: [QuadParams<f64>; L] = core::array::from_fn(|_| QuadParams::from_input(&input));
    let states: [State<f64>; L] = core::array::from_fn(|_| hover_state(&input));
    let forces: [[f64; 4]; L] = core::array::from_fn(|_| [HOVER_F; 4]);

    // black_box the inputs once so nothing is const-folded; the loop-carried
    // dependency on `s` then prevents hoisting. Lanes are opaque to the compiler,
    // so it emits full-width SIMD; timing is data-independent (branchless kernel).
    let params = black_box(pack_params(&scal));
    let f = black_box(pack_rotor_forces(&forces));
    let dt = F64x::<L>::splat(DT);
    let mut s = black_box(pack_state(&states));

    for _ in 0..1000 {
        let raw = RotorForce::cmd_rotor_speeds(&params, &s, &f);
        let speeds = clip_speeds(&params, raw);
        s = integrate(&params, &s, &speeds, dt, n_sub);
    }
    let t0 = Instant::now();
    for _ in 0..n {
        let raw = RotorForce::cmd_rotor_speeds(&params, &s, &f);
        let speeds = clip_speeds(&params, raw);
        s = integrate(&params, &s, &speeds, dt, n_sub);
    }
    let el = t0.elapsed().as_secs_f64();
    black_box(s);
    let dsps = (n * L) as f64 / el;
    println!(
        "[F64x<{L:<2}> SIMD] N_SUB={n_sub:<2} {dsps:>14.0} drone-steps/s   {:6.2} ns/drone-step   {:.2}x vs scalar",
        el / (n * L) as f64 * 1e9,
        dsps / scalar_sps,
    );
}

fn main() {
    let n = 2_000_000;
    println!("# matched-accuracy integrator (N_SUB=8)");
    let s8 = bench_scalar(8, n);
    bench_batch::<2>(8, n, s8);
    bench_batch::<4>(8, n, s8);
    bench_batch::<8>(8, n, s8);
    bench_batch::<16>(8, n, s8);
    println!("\n# cheapest integrator (N_SUB=1)");
    let s1 = bench_scalar(1, n);
    bench_batch::<2>(1, n, s1);
    bench_batch::<4>(1, n, s1);
    bench_batch::<8>(1, n, s1);
}
