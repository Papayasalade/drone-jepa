//! Vehicle physical parameters, ported from RotorPy `Multirotor.__init__`.
//!
//! `QuadParamsInput` is a plain `f64` struct (what you deserialize a fixture /
//! domain-randomized param set into). `QuadParams<S>` is the build-time-derived,
//! scalar-generic form the dynamics consume — inertia inverse, drag diagonal,
//! gravity weight, and the control-allocation matrix are all precomputed once.
//!
//! Fixed at 4 rotors (this project only flies quads), matching RotorPy's
//! quad param files.

use crate::linalg::{Mat3, Mat4, Vec3};
use crate::scalar::Scalar;

pub const NUM_ROTORS: usize = 4;
pub const GRAVITY: f64 = 9.81;

/// Plain `f64` parameter set (one domain). Mirrors RotorPy's `quad_params` dict,
/// keeping only the keys the single-vehicle dynamics use.
#[derive(Clone, Copy, Debug)]
pub struct QuadParamsInput {
    pub mass: f64,
    pub ixx: f64,
    pub iyy: f64,
    pub izz: f64,
    pub ixy: f64,
    pub iyz: f64,
    pub ixz: f64,
    /// rotor positions relative to CoM, in r1..r4 order.
    pub rotor_pos: [[f64; 3]; NUM_ROTORS],
    pub rotor_directions: [f64; NUM_ROTORS],
    pub c_dx: f64,
    pub c_dy: f64,
    pub c_dz: f64,
    pub k_eta: f64,
    pub k_m: f64,
    pub k_d: f64,
    pub k_z: f64,
    pub k_h: f64,
    pub k_flap: f64,
    pub tau_m: f64,
    pub rotor_speed_min: f64,
    pub rotor_speed_max: f64,
    pub k_w: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct QuadParams<S> {
    pub mass: S,
    pub inertia: Mat3<S>,
    pub inv_inertia: Mat3<S>,
    pub weight: Vec3<S>, // (0, 0, -m g) in world frame
    pub drag_diag: Vec3<S>,
    pub k_eta: S,
    pub k_m: S,
    pub k_d: S,
    pub k_z: S,
    pub k_h: S,
    pub k_flap: S,
    pub tau_m: S,
    pub rotor_speed_min: S,
    pub rotor_speed_max: S,
    pub k_w: S,
    pub rotor_pos: [Vec3<S>; NUM_ROTORS],
    pub rotor_dir: [S; NUM_ROTORS],
    /// Per-rotor thrust multiplier ("motor power"): the realized thrust is
    /// `rotor_gain[j] * k_eta * rpm²`. Default 1.0; the demo/RL-env randomize it so
    /// some motors push harder. The CTBR allocation doesn't know about it — the rate
    /// loop compensates, exactly like a real weak/strong motor.
    pub rotor_gain: [S; NUM_ROTORS],
    /// Maps [thrust, Mx, My, Mz] -> per-rotor forces (inverse of f_to_TM).
    pub tm_to_f: Mat4<S>,
}

impl<S: Scalar> QuadParams<S> {
    pub fn from_input(p: &QuadParamsInput) -> Self {
        let s = S::splat;

        let inertia = Mat3::new([
            [s(p.ixx), s(p.ixy), s(p.ixz)],
            [s(p.ixy), s(p.iyy), s(p.iyz)],
            [s(p.ixz), s(p.iyz), s(p.izz)],
        ]);
        let inv_inertia = inertia.inverse();
        let weight = Vec3::new(S::ZERO, S::ZERO, s(-p.mass * GRAVITY));

        let rotor_pos = core::array::from_fn(|r| {
            let v = p.rotor_pos[r];
            Vec3::new(s(v[0]), s(v[1]), s(v[2]))
        });
        let rotor_dir: [S; NUM_ROTORS] = core::array::from_fn(|r| s(p.rotor_directions[r]));

        // Control allocation f_to_TM (RotorPy multirotor.py L164-170):
        //   row0 = ones
        //   rows 1,2 = cross(rotor_pos, [0,0,1])[0:2] = (r_y, -r_x)
        //   row3 = (k_m/k_eta) * rotor_dir
        let kc = p.k_m / p.k_eta;
        let mut f_to_tm = [[S::ZERO; NUM_ROTORS]; 4];
        for r in 0..NUM_ROTORS {
            let rp = p.rotor_pos[r];
            f_to_tm[0][r] = S::ONE;
            f_to_tm[1][r] = s(rp[1]); // r_y
            f_to_tm[2][r] = s(-rp[0]); // -r_x
            f_to_tm[3][r] = s(kc * p.rotor_directions[r]);
        }
        let tm_to_f = Mat4::new(f_to_tm).inverse();

        Self {
            mass: s(p.mass),
            inertia,
            inv_inertia,
            weight,
            drag_diag: Vec3::new(s(p.c_dx), s(p.c_dy), s(p.c_dz)),
            k_eta: s(p.k_eta),
            k_m: s(p.k_m),
            k_d: s(p.k_d),
            k_z: s(p.k_z),
            k_h: s(p.k_h),
            k_flap: s(p.k_flap),
            tau_m: s(p.tau_m),
            rotor_speed_min: s(p.rotor_speed_min),
            rotor_speed_max: s(p.rotor_speed_max),
            k_w: s(p.k_w),
            rotor_pos,
            rotor_dir,
            rotor_gain: [S::ONE; NUM_ROTORS], // default: all motors equal
            tm_to_f,
        }
    }
}
