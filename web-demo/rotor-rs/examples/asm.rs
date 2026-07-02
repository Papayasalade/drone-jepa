//! Symbols to disassemble (objdump) to answer two questions:
//!   1. Is a single scalar step jump-free, straight-line code?
//!   2. Does the F64x batch path actually emit vector (NEON/AVX) instructions?
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo build --release --example asm
//!   objdump -d target/release/examples/asm | sed -n '/<step_scalar_once>:/,/ret/p'

use rotor_rs::control::{ControlLaw, RotorForce};
use rotor_rs::multirotor::{clip_speeds, integrate};
use rotor_rs::simd::F64x;
use rotor_rs::{QuadParams, Scalar, State};

#[no_mangle]
#[inline(never)]
pub extern "C" fn add_f64x4(a: &F64x<4>, b: &F64x<4>, out: &mut F64x<4>) {
    *out = *a + *b;
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn step_scalar_once(
    state: &State<f64>,
    params: &QuadParams<f64>,
    forces: &[f64; 4],
    out: &mut State<f64>,
) {
    let raw = RotorForce::cmd_rotor_speeds(params, state, forces);
    let speeds = clip_speeds(params, raw);
    *out = integrate(params, state, &speeds, 0.005, 1); // single RK4 substep
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn step_batch4_once(
    state: &State<F64x<4>>,
    params: &QuadParams<F64x<4>>,
    forces: &[F64x<4>; 4],
    out: &mut State<F64x<4>>,
) {
    let raw = RotorForce::cmd_rotor_speeds(params, state, forces);
    let speeds = clip_speeds(params, raw);
    *out = integrate(params, state, &speeds, F64x::splat(0.005), 1);
}

fn main() {
    // Reference the symbols so the linker keeps them, behind black_box so the
    // bodies aren't const-folded away.
    use std::hint::black_box;
    let a = F64x([1.0, 2.0, 3.0, 4.0]);
    let mut o = F64x([0.0; 4]);
    add_f64x4(black_box(&a), black_box(&a), &mut o);
    println!("{:?}", black_box(o).0[0]);

    let p = QuadParams::from_input(black_box(&rotor_rs::QuadParamsInput {
        mass: 0.5, ixx: 3.65e-3, iyy: 3.68e-3, izz: 7.03e-3, ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[0.12, 0.12, 0.0], [0.12, -0.12, 0.0], [-0.12, -0.12, 0.0], [-0.12, 0.12, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: 5e-3, c_dy: 5e-3, c_dz: 1e-2,
        k_eta: 5.57e-6, k_m: 1.36e-7, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: 0.005, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w: 1.0,
    }));
    let st = State {
        x: rotor_rs::Vec3::new(0.0, 0.0, 1.5),
        v: rotor_rs::Vec3::zero(),
        q: rotor_rs::Quat::new(0.0, 0.0, 0.0, 1.0),
        w: rotor_rs::Vec3::zero(),
        wind: rotor_rs::Vec3::zero(),
        rotor_speeds: [500.0; 4],
    };
    let mut so = st;
    step_scalar_once(black_box(&st), black_box(&p), black_box(&[1.2; 4]), &mut so);
    println!("{:?}", black_box(so).x.z);
}
