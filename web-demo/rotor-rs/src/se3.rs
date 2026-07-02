//! SE(3) geometric tracking controller, ported from RotorPy `SE3Control.update`
//! (Lee et al. 2010). Outputs four rotor forces [N] to track a flat-output
//! reference — the "smooth tracker" for generating rotor-force training data.
//!
//! Gains are the fixed RotorPy values (tuned for the ~0.5 kg hummingbird); per-drone
//! mass/inertia/allocation come from the params.

use crate::linalg::{Mat3, Mat4, Vec3};
use crate::params::{QuadParams, QuadParamsInput, GRAVITY, NUM_ROTORS};

/// Desired flat outputs the controller tracks.
#[derive(Clone, Copy, Debug)]
pub struct FlatRef {
    pub x: Vec3<f64>,
    pub x_dot: Vec3<f64>,
    pub x_ddot: Vec3<f64>,
    pub yaw: f64,
    pub yaw_dot: f64,
}

pub struct Se3Control {
    mass: f64,
    inertia: Mat3<f64>,
    tm_to_f: Mat4<f64>,
    kp_pos: Vec3<f64>,
    kd_pos: Vec3<f64>,
    kp_att: f64,
    kd_att: f64,
}

#[inline]
fn normalize(v: Vec3<f64>) -> Vec3<f64> {
    let n = v.norm();
    if n > 1e-12 {
        v.scale(1.0 / n)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

impl Se3Control {
    pub fn new(p: &QuadParamsInput) -> Self {
        let qp = QuadParams::<f64>::from_input(p);
        Se3Control {
            mass: p.mass,
            inertia: qp.inertia,
            tm_to_f: qp.tm_to_f,
            kp_pos: Vec3::new(6.5, 6.5, 15.0),
            kd_pos: Vec3::new(4.0, 4.0, 9.0),
            kp_att: 544.0,
            kd_att: 46.64,
        }
    }

    /// Returns four rotor forces [N] (unclamped; caller clamps to motor limits).
    pub fn update(&self, state: &crate::State<f64>, r: &FlatRef) -> [f64; NUM_ROTORS] {
        self.update_full(state, r).0
    }

    /// A reliable, policy-free CTBR SEED toward a flat reference: collective thrust +
    /// a BODY-RATE command (attitude error scaled by `kp_rate`, clamped to ±`rate_max`)
    /// — in the same bounded action space the CTBR JEPA was trained on (NOT the
    /// high-gain angular-accel `cmd_w` of `update_full`). Used to seed MPPI's search in
    /// a feasible, in-distribution region so the world model only has to RANK locally.
    pub fn ctbr_seed(
        &self, state: &crate::State<f64>, r: &FlatRef, kp_rate: f64, rate_max: f64,
    ) -> (f64, Vec3<f64>) {
        let rot = state.q.to_rotmat();
        let pos_err = state.x - r.x;
        let dpos_err = state.v - r.x_dot;
        let f_des = (r.x_ddot + Vec3::new(0.0, 0.0, GRAVITY)
            - pos_err.hadamard(self.kp_pos)
            - dpos_err.hadamard(self.kd_pos))
        .scale(self.mass);
        let b3 = Vec3::new(rot.rows[0][2], rot.rows[1][2], rot.rows[2][2]);
        let u1 = f_des.dot(b3).max(0.0); // collective thrust
        let b3_des = normalize(f_des);
        let c1 = Vec3::new(r.yaw.cos(), r.yaw.sin(), 0.0);
        let b2_des = normalize(b3_des.cross(c1));
        let b1_des = b2_des.cross(b3_des);
        let r_des = Mat3::new([
            [b1_des.x, b2_des.x, b3_des.x],
            [b1_des.y, b2_des.y, b3_des.y],
            [b1_des.z, b2_des.z, b3_des.z],
        ]);
        let s_err = {
            let a = r_des.transpose().mul(&rot);
            let b = rot.transpose().mul(&r_des);
            Mat3::new(core::array::from_fn(|i| {
                core::array::from_fn(|j| 0.5 * (a.rows[i][j] - b.rows[i][j]))
            }))
        };
        let att_err = Vec3::new(-s_err.rows[1][2], s_err.rows[0][2], -s_err.rows[0][1]);
        let w = att_err.scale(-kp_rate);
        let cl = |x: f64| x.clamp(-rate_max, rate_max);
        (u1, Vec3::new(cl(w.x), cl(w.y), cl(w.z)))
    }

    /// Full SE3 output: (rotor forces [N], collective thrust [N], cmd body-rate
    /// [the CTBR `cmd_w`]). Lets one flight be logged in BOTH action representations.
    pub fn update_full(
        &self,
        state: &crate::State<f64>,
        r: &FlatRef,
    ) -> ([f64; NUM_ROTORS], f64, Vec3<f64>) {
        let rot = state.q.to_rotmat(); // body->world

        // desired force vector
        let pos_err = state.x - r.x;
        let dpos_err = state.v - r.x_dot;
        let f_des = (r.x_ddot + Vec3::new(0.0, 0.0, GRAVITY)
            - pos_err.hadamard(self.kp_pos)
            - dpos_err.hadamard(self.kd_pos))
        .scale(self.mass);

        // thrust = F_des . b3 (body z in world = 3rd column of R)
        let b3 = Vec3::new(rot.rows[0][2], rot.rows[1][2], rot.rows[2][2]);
        let u1 = f_des.dot(b3);

        // desired attitude
        let b3_des = normalize(f_des);
        let c1 = Vec3::new(r.yaw.cos(), r.yaw.sin(), 0.0);
        let b2_des = normalize(b3_des.cross(c1));
        let b1_des = b2_des.cross(b3_des);
        // R_des columns = [b1, b2, b3]
        let r_des = Mat3::new([
            [b1_des.x, b2_des.x, b3_des.x],
            [b1_des.y, b2_des.y, b3_des.y],
            [b1_des.z, b2_des.z, b3_des.z],
        ]);

        // orientation error: 0.5 (R_des^T R - R^T R_des), vee
        let s_err = {
            let a = r_des.transpose().mul(&rot);
            let b = rot.transpose().mul(&r_des);
            // 0.5 (a - b)
            Mat3::new(core::array::from_fn(|i| {
                core::array::from_fn(|j| 0.5 * (a.rows[i][j] - b.rows[i][j]))
            }))
        };
        let att_err = Vec3::new(-s_err.rows[1][2], s_err.rows[0][2], -s_err.rows[0][1]);

        let w_des = Vec3::new(0.0, 0.0, r.yaw_dot);
        let w_err = state.w - w_des;

        // cmd_w = -kp att_err - kd w_err (the CTBR body-rate command)
        let cmd_w = att_err.scale(-self.kp_att) - w_err.scale(self.kd_att);
        // u2 = I @ cmd_w + w x (I w)
        let iw = self.inertia.matvec(state.w);
        let u2 = self.inertia.matvec(cmd_w) + state.w.cross(iw);

        // allocate to rotor forces; cmd_thrust = u1
        let tm = [u1, u2.x, u2.y, u2.z];
        (self.tm_to_f.matvec(tm), u1, cmd_w)
    }
}
