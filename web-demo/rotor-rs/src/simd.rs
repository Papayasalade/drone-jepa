//! `F64x<L>` — a width-`L` lane vector implementing [`Scalar`], so the *same*
//! generic dynamics kernel steps `L` drones at once. Pure `[f64; L]` arithmetic
//! with no data-dependent branches, so LLVM auto-vectorizes it to SIMD
//! (build with `RUSTFLAGS="-C target-cpu=native"` to get AVX). Zero dependencies.
//!
//! Because the kernel is branchless, all `L` lanes always execute the identical
//! instruction stream — timing is data-independent, and `integrate::<F64x<L>>`
//! advances `L` independent drones (different params per lane) in lockstep.

use crate::linalg::{Mat3, Mat4, Quat, Vec3};
use crate::params::{QuadParams, NUM_ROTORS};
use crate::scalar::Scalar;
use crate::state::State;
use core::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct F64x<const L: usize>(pub [f64; L]);

macro_rules! lanewise_bin {
    ($trait:ident, $method:ident) => {
        impl<const L: usize> $trait for F64x<L> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, o: Self) -> Self {
                F64x(core::array::from_fn(|i| self.0[i].$method(o.0[i])))
            }
        }
    };
}
lanewise_bin!(Add, add);
lanewise_bin!(Sub, sub);
lanewise_bin!(Mul, mul);
lanewise_bin!(Div, div);

impl<const L: usize> Neg for F64x<L> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        F64x(core::array::from_fn(|i| -self.0[i]))
    }
}

impl<const L: usize> Scalar for F64x<L> {
    const ZERO: Self = F64x([0.0; L]);
    const ONE: Self = F64x([1.0; L]);

    #[inline(always)]
    fn splat(x: f64) -> Self {
        F64x([x; L])
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        F64x(core::array::from_fn(|i| self.0[i].sqrt()))
    }
    #[inline(always)]
    fn abs(self) -> Self {
        F64x(core::array::from_fn(|i| self.0[i].abs()))
    }
    #[inline(always)]
    fn signum(self) -> Self {
        F64x(core::array::from_fn(|i| 1.0_f64.copysign(self.0[i])))
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        F64x(core::array::from_fn(|i| self.0[i].min(o.0[i])))
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        F64x(core::array::from_fn(|i| self.0[i].max(o.0[i])))
    }
}

// --------------------------------------------------------------------------- //
// Packing helpers: build a lane-packed QuadParams / State from L scalar ones,
// and read a single lane back out. Inverses etc. are computed per-lane in f64
// (via QuadParams::from_input) then transposed into lanes here, so the SIMD
// path reuses the exact verified scalar construction.
// --------------------------------------------------------------------------- //
#[inline]
fn lanes<const L: usize>(f: impl Fn(usize) -> f64) -> F64x<L> {
    F64x(core::array::from_fn(f))
}
#[inline]
fn pack_vec3<const L: usize>(g: impl Fn(usize) -> Vec3<f64>) -> Vec3<F64x<L>> {
    Vec3::new(lanes(|i| g(i).x), lanes(|i| g(i).y), lanes(|i| g(i).z))
}
#[inline]
fn pack_quat<const L: usize>(g: impl Fn(usize) -> Quat<f64>) -> Quat<F64x<L>> {
    Quat::new(lanes(|i| g(i).i), lanes(|i| g(i).j), lanes(|i| g(i).k), lanes(|i| g(i).w))
}
#[inline]
fn pack_mat3<const L: usize>(g: impl Fn(usize) -> Mat3<f64>) -> Mat3<F64x<L>> {
    Mat3::new(core::array::from_fn(|r| {
        core::array::from_fn(|c| lanes(|i| g(i).rows[r][c]))
    }))
}
#[inline]
fn pack_mat4<const L: usize>(g: impl Fn(usize) -> Mat4<f64>) -> Mat4<F64x<L>> {
    Mat4::new(core::array::from_fn(|r| {
        core::array::from_fn(|c| lanes(|i| g(i).rows[r][c]))
    }))
}

pub fn pack_params<const L: usize>(ps: &[QuadParams<f64>; L]) -> QuadParams<F64x<L>> {
    QuadParams {
        mass: lanes(|i| ps[i].mass),
        inertia: pack_mat3(|i| ps[i].inertia),
        inv_inertia: pack_mat3(|i| ps[i].inv_inertia),
        weight: pack_vec3(|i| ps[i].weight),
        drag_diag: pack_vec3(|i| ps[i].drag_diag),
        k_eta: lanes(|i| ps[i].k_eta),
        k_m: lanes(|i| ps[i].k_m),
        k_d: lanes(|i| ps[i].k_d),
        k_z: lanes(|i| ps[i].k_z),
        k_h: lanes(|i| ps[i].k_h),
        k_flap: lanes(|i| ps[i].k_flap),
        tau_m: lanes(|i| ps[i].tau_m),
        rotor_speed_min: lanes(|i| ps[i].rotor_speed_min),
        rotor_speed_max: lanes(|i| ps[i].rotor_speed_max),
        k_w: lanes(|i| ps[i].k_w),
        rotor_pos: core::array::from_fn(|r| pack_vec3(|i| ps[i].rotor_pos[r])),
        rotor_dir: core::array::from_fn(|r| lanes(|i| ps[i].rotor_dir[r])),
        rotor_gain: core::array::from_fn(|r| lanes(|i| ps[i].rotor_gain[r])),
        tm_to_f: pack_mat4(|i| ps[i].tm_to_f),
    }
}

pub fn pack_state<const L: usize>(s: &[State<f64>; L]) -> State<F64x<L>> {
    State {
        x: pack_vec3(|i| s[i].x),
        v: pack_vec3(|i| s[i].v),
        q: pack_quat(|i| s[i].q),
        w: pack_vec3(|i| s[i].w),
        wind: pack_vec3(|i| s[i].wind),
        rotor_speeds: core::array::from_fn(|r| lanes(|i| s[i].rotor_speeds[r])),
    }
}

/// Extract lane `l` of a packed state as a scalar `State<f64>`.
pub fn unpack_state_lane<const L: usize>(s: &State<F64x<L>>, l: usize) -> State<f64> {
    State {
        x: Vec3::new(s.x.x.0[l], s.x.y.0[l], s.x.z.0[l]),
        v: Vec3::new(s.v.x.0[l], s.v.y.0[l], s.v.z.0[l]),
        q: Quat::new(s.q.i.0[l], s.q.j.0[l], s.q.k.0[l], s.q.w.0[l]),
        w: Vec3::new(s.w.x.0[l], s.w.y.0[l], s.w.z.0[l]),
        wind: Vec3::new(s.wind.x.0[l], s.wind.y.0[l], s.wind.z.0[l]),
        rotor_speeds: core::array::from_fn(|r| s.rotor_speeds[r].0[l]),
    }
}

/// Pack `L` rotor-force commands into a lane-vector command.
pub fn pack_rotor_forces<const L: usize>(
    f: &[[f64; NUM_ROTORS]; L],
) -> [F64x<L>; NUM_ROTORS] {
    core::array::from_fn(|r| lanes(|i| f[i][r]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{ControlLaw, RotorForce};
    use crate::multirotor::{clip_speeds, integrate, Multirotor};
    use crate::params::QuadParamsInput;

    fn drone(mass: f64, k_eta: f64, tau_m: f64) -> QuadParamsInput {
        let d = 0.17 * std::f64::consts::FRAC_1_SQRT_2;
        QuadParamsInput {
            mass, ixx: 3.65e-3, iyy: 3.68e-3, izz: 7.03e-3,
            ixy: 0.0, iyz: 0.0, ixz: 0.0,
            rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
            rotor_directions: [1.0, -1.0, 1.0, -1.0],
            c_dx: 0.5e-2, c_dy: 0.5e-2, c_dz: 1e-2,
            k_eta, k_m: 1.36e-7, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
            tau_m, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w: 1.0,
        }
    }

    fn hover(input: &QuadParamsInput) -> State<f64> {
        let hov = (input.mass * 9.81 / (4.0 * input.k_eta)).sqrt();
        State {
            x: Vec3::new(0.0, 0.0, 1.5),
            v: Vec3::new(0.1, -0.2, 0.05),
            q: Quat::new(0.0, 0.0, 0.0, 1.0),
            w: Vec3::new(0.05, -0.03, 0.02),
            wind: Vec3::zero(),
            rotor_speeds: [hov; 4],
        }
    }

    #[test]
    fn batched_step_matches_per_lane_scalar() {
        // Four DIFFERENT drones; step the batch once and compare each lane to the
        // independent scalar step. Proves the SIMD Scalar impl is correct.
        const L: usize = 4;
        let inputs = [
            drone(0.5, 5.57e-6, 0.005),
            drone(0.8, 6.50e-6, 0.010),
            drone(0.4, 5.00e-6, 0.020),
            drone(1.1, 8.00e-6, 0.008),
        ];
        let scal_params: [QuadParams<f64>; L] =
            core::array::from_fn(|i| QuadParams::from_input(&inputs[i]));
        let states: [State<f64>; L] = core::array::from_fn(|i| hover(&inputs[i]));
        let forces: [[f64; 4]; L] =
            core::array::from_fn(|i| [inputs[i].mass * 9.81 / 4.0 * 1.05; 4]);

        // scalar references
        let n_sub = 8;
        let dt = 0.005;
        let mut scal_next = [states[0]; L];
        for i in 0..L {
            let veh: Multirotor<f64, RotorForce> =
                Multirotor::with_substeps(&inputs[i], n_sub);
            scal_next[i] = veh.step(&states[i], &forces[i], dt);
        }

        // batched
        let bp = pack_params(&scal_params);
        let bs = pack_state(&states);
        let bf = pack_rotor_forces(&forces);
        let raw = RotorForce::cmd_rotor_speeds(&bp, &bs, &bf);
        let speeds = clip_speeds(&bp, raw);
        let bn = integrate(&bp, &bs, &speeds, F64x::splat(dt), n_sub);

        for l in 0..L {
            let got = unpack_state_lane(&bn, l);
            let want = scal_next[l];
            let dp = ((got.x.x - want.x.x).powi(2)
                + (got.x.y - want.x.y).powi(2)
                + (got.x.z - want.x.z).powi(2))
            .sqrt();
            assert!(dp < 1e-12, "lane {l} pos mismatch: {dp}");
            for r in 0..4 {
                assert!((got.rotor_speeds[r] - want.rotor_speeds[r]).abs() < 1e-9);
            }
        }
    }
}
