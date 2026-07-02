//! Vehicle state and its packed 20-vector form used by the integrator.
//!
//! Layout matches RotorPy `_pack_state`:
//!   s[0:3]=x, s[3:6]=v, s[6:10]=q[i,j,k,w], s[10:13]=w, s[13:16]=wind, s[16:20]=rotor_speeds

use crate::linalg::{Quat, Vec3};
use crate::params::NUM_ROTORS;
use crate::scalar::Scalar;

pub const STATE_LEN: usize = 16 + NUM_ROTORS; // 20

#[derive(Clone, Copy, Debug)]
pub struct State<S> {
    pub x: Vec3<S>,
    pub v: Vec3<S>,
    pub q: Quat<S>,
    pub w: Vec3<S>,
    pub wind: Vec3<S>,
    pub rotor_speeds: [S; NUM_ROTORS],
}

impl<S: Scalar> State<S> {
    pub fn pack(&self) -> [S; STATE_LEN] {
        let mut s = [S::ZERO; STATE_LEN];
        s[0] = self.x.x; s[1] = self.x.y; s[2] = self.x.z;
        s[3] = self.v.x; s[4] = self.v.y; s[5] = self.v.z;
        let q = self.q.to_array();
        s[6] = q[0]; s[7] = q[1]; s[8] = q[2]; s[9] = q[3];
        s[10] = self.w.x; s[11] = self.w.y; s[12] = self.w.z;
        s[13] = self.wind.x; s[14] = self.wind.y; s[15] = self.wind.z;
        for r in 0..NUM_ROTORS {
            s[16 + r] = self.rotor_speeds[r];
        }
        s
    }

    pub fn unpack(s: &[S; STATE_LEN]) -> Self {
        Self {
            x: Vec3::new(s[0], s[1], s[2]),
            v: Vec3::new(s[3], s[4], s[5]),
            q: Quat::new(s[6], s[7], s[8], s[9]),
            w: Vec3::new(s[10], s[11], s[12]),
            wind: Vec3::new(s[13], s[14], s[15]),
            rotor_speeds: core::array::from_fn(|r| s[16 + r]),
        }
    }
}
