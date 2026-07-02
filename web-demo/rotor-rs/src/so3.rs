//! SO(3) / rigid-body rotation helpers ported from RotorPy `multirotor.py`.
//!
//! `quat -> rotmat` and `hat_map` live in `linalg` (`Quat::to_rotmat`,
//! `Mat3::hat`); this module adds the quaternion kinematics `quat_dot`. These
//! are the pieces that will later be shared with the JEPA model's DKI decoder.

use crate::linalg::{Quat, Vec3};
use crate::scalar::Scalar;

/// Quaternion derivative for a body angular velocity `omega` (body frame),
/// matching RotorPy's `quat_dot` (Basile Graf form). Operates on the *raw*
/// (possibly non-unit) quaternion — RotorPy renormalizes only after the step.
///
/// q = [i, j, k, w]; returns d/dt [i, j, k, w].
#[inline]
pub fn quat_dot<S: Scalar>(q: Quat<S>, omega: Vec3<S>) -> Quat<S> {
    let (i, j, k, w) = (q.i, q.j, q.k, q.w);
    let (wx, wy, wz) = (omega.x, omega.y, omega.z);
    let half = S::splat(0.5);
    Quat::new(
        half * (w * wx - k * wy + j * wz),
        half * (k * wx + w * wy - i * wz),
        half * (-j * wx + i * wy + w * wz),
        half * (-i * wx - j * wy - k * wz),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quat_dot_spins_quaternion() {
        // Constant body rate about z, starting at identity: q_dot should be
        // 0.5 * [0,0,wz,0] (pure k-component growth, w-component zero).
        let q = Quat::new(0.0, 0.0, 0.0, 1.0);
        let omega = Vec3::new(0.0, 0.0, 2.0);
        let qd = quat_dot(q, omega);
        assert!((qd.i).abs() < 1e-15);
        assert!((qd.j).abs() < 1e-15);
        assert!((qd.k - 1.0).abs() < 1e-15); // 0.5 * 2.0
        assert!((qd.w).abs() < 1e-15);
    }

    #[test]
    fn quat_dot_preserves_unit_norm_rate() {
        // d/dt ||q||^2 = 2 q . q_dot must be 0 for any q, omega (unit-norm preserving).
        let q = Quat::new(0.1, -0.3, 0.55, 0.77);
        let omega = Vec3::new(1.3, -0.4, 2.1);
        let qd = quat_dot(q, omega);
        let dot = q.i * qd.i + q.j * qd.j + q.k * qd.k + q.w * qd.w;
        assert!(dot.abs() < 1e-15, "q.qd = {dot}");
    }
}
