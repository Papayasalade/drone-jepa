//! Hand-rolled, dependency-free linear algebra, generic over `S: Scalar`.
//!
//! Only the operations the quad dynamics actually need are implemented. Fixed
//! sizes (3-vectors, 3x3, 4x4) keep everything on the stack and branch-free.

use crate::scalar::Scalar;
use core::ops::{Add, Neg, Sub};

// --------------------------------------------------------------------------- //
// Vec3
// --------------------------------------------------------------------------- //
#[derive(Clone, Copy, Debug)]
pub struct Vec3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

impl<S: Scalar> Vec3<S> {
    #[inline]
    pub fn new(x: S, y: S, z: S) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub fn zero() -> Self {
        Self { x: S::ZERO, y: S::ZERO, z: S::ZERO }
    }
    #[inline]
    pub fn scale(self, s: S) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
    #[inline]
    pub fn dot(self, o: Self) -> S {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    #[inline]
    pub fn norm(self) -> S {
        self.dot(self).sqrt()
    }
    /// Component-wise (Hadamard) product.
    #[inline]
    pub fn hadamard(self, o: Self) -> Self {
        Self { x: self.x * o.x, y: self.y * o.y, z: self.z * o.z }
    }
    #[inline]
    pub fn to_array(self) -> [S; 3] {
        [self.x, self.y, self.z]
    }
}

impl<S: Scalar> Add for Vec3<S> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }
}
impl<S: Scalar> Sub for Vec3<S> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
}
impl<S: Scalar> Neg for Vec3<S> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

// --------------------------------------------------------------------------- //
// Mat3  (row-major)
// --------------------------------------------------------------------------- //
#[derive(Clone, Copy, Debug)]
pub struct Mat3<S> {
    /// rows[i] is the i-th row.
    pub rows: [[S; 3]; 3],
}

impl<S: Scalar> Mat3<S> {
    #[inline]
    pub fn new(rows: [[S; 3]; 3]) -> Self {
        Self { rows }
    }

    /// Skew-symmetric "hat" matrix of `s` so that `hat(s) * v == s.cross(v)`.
    #[inline]
    pub fn hat(s: Vec3<S>) -> Self {
        let z = S::ZERO;
        Self {
            rows: [
                [z, -s.z, s.y],
                [s.z, z, -s.x],
                [-s.y, s.x, z],
            ],
        }
    }

    #[inline]
    pub fn matvec(&self, v: Vec3<S>) -> Vec3<S> {
        let r = &self.rows;
        Vec3::new(
            r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
            r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
            r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
        )
    }

    /// Transpose-times-vector, i.e. `self^T * v` (used for world->body rotation).
    #[inline]
    pub fn tmatvec(&self, v: Vec3<S>) -> Vec3<S> {
        let r = &self.rows;
        Vec3::new(
            r[0][0] * v.x + r[1][0] * v.y + r[2][0] * v.z,
            r[0][1] * v.x + r[1][1] * v.y + r[2][1] * v.z,
            r[0][2] * v.x + r[1][2] * v.y + r[2][2] * v.z,
        )
    }

    pub fn transpose(&self) -> Self {
        let m = &self.rows;
        Self {
            rows: [
                [m[0][0], m[1][0], m[2][0]],
                [m[0][1], m[1][1], m[2][1]],
                [m[0][2], m[1][2], m[2][2]],
            ],
        }
    }

    /// Matrix product self * o.
    pub fn mul(&self, o: &Self) -> Self {
        let a = &self.rows;
        let b = &o.rows;
        let mut r = [[S::ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Self { rows: r }
    }

    /// General 3x3 inverse via cofactors / determinant (branchless arithmetic).
    pub fn inverse(&self) -> Self {
        let m = &self.rows;
        let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
        let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
        let c10 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
        let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
        let c12 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
        let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        let c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
        let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
        let inv_det = S::ONE / det;
        // adjugate (transpose of cofactor matrix) * 1/det
        Self {
            rows: [
                [c00 * inv_det, c10 * inv_det, c20 * inv_det],
                [c01 * inv_det, c11 * inv_det, c21 * inv_det],
                [c02 * inv_det, c12 * inv_det, c22 * inv_det],
            ],
        }
    }
}

// --------------------------------------------------------------------------- //
// Mat4  (row-major) — only needed for the cmd_ctbr control allocation
// --------------------------------------------------------------------------- //
#[derive(Clone, Copy, Debug)]
pub struct Mat4<S> {
    pub rows: [[S; 4]; 4],
}

impl<S: Scalar> Mat4<S> {
    #[inline]
    pub fn new(rows: [[S; 4]; 4]) -> Self {
        Self { rows }
    }

    #[inline]
    pub fn matvec(&self, v: [S; 4]) -> [S; 4] {
        let r = &self.rows;
        let mut out = [S::ZERO; 4];
        for i in 0..4 {
            out[i] = r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] + r[i][3] * v[3];
        }
        out
    }

    /// General 4x4 inverse via the adjugate (cofactor) method. Verbose but pure
    /// arithmetic — no pivoting, no branches.
    pub fn inverse(&self) -> Self {
        let m = &self.rows;
        // 2x2 minors of the bottom two rows (for the top cofactors) and top two
        // rows (for the bottom cofactors).
        let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
        let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
        let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
        let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
        let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
        let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

        let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        let invdet = S::ONE / det;

        let mut b = [[S::ZERO; 4]; 4];
        b[0][0] = (m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * invdet;
        b[0][1] = (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * invdet;
        b[0][2] = (m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * invdet;
        b[0][3] = (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * invdet;

        b[1][0] = (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * invdet;
        b[1][1] = (m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * invdet;
        b[1][2] = (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * invdet;
        b[1][3] = (m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * invdet;

        b[2][0] = (m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * invdet;
        b[2][1] = (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * invdet;
        b[2][2] = (m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * invdet;
        b[2][3] = (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * invdet;

        b[3][0] = (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * invdet;
        b[3][1] = (m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * invdet;
        b[3][2] = (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * invdet;
        b[3][3] = (m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * invdet;

        Self { rows: b }
    }
}

// --------------------------------------------------------------------------- //
// Quaternion  [i, j, k, w]  (scipy / RotorPy convention)
// --------------------------------------------------------------------------- //
#[derive(Clone, Copy, Debug)]
pub struct Quat<S> {
    pub i: S,
    pub j: S,
    pub k: S,
    pub w: S,
}

impl<S: Scalar> Quat<S> {
    #[inline]
    pub fn new(i: S, j: S, k: S, w: S) -> Self {
        Self { i, j, k, w }
    }
    #[inline]
    pub fn from_array(a: [S; 4]) -> Self {
        Self { i: a[0], j: a[1], k: a[2], w: a[3] }
    }
    #[inline]
    pub fn to_array(self) -> [S; 4] {
        [self.i, self.j, self.k, self.w]
    }
    #[inline]
    pub fn norm(self) -> S {
        (self.i * self.i + self.j * self.j + self.k * self.k + self.w * self.w).sqrt()
    }
    #[inline]
    pub fn normalized(self) -> Self {
        let inv = S::ONE / self.norm();
        Self { i: self.i * inv, j: self.j * inv, k: self.k * inv, w: self.w * inv }
    }

    /// Body->world rotation matrix, matching `scipy Rotation.from_quat([i,j,k,w]).as_matrix()`.
    /// scipy normalizes the quaternion first; we do the same so the result matches
    /// even when the quaternion has drifted off the unit sphere mid-integration.
    pub fn to_rotmat(self) -> Mat3<S> {
        let q = self.normalized();
        let (x, y, z, w) = (q.i, q.j, q.k, q.w);
        let two = S::splat(2.0);
        let one = S::ONE;
        Mat3::new([
            [
                one - two * (y * y + z * z),
                two * (x * y - z * w),
                two * (x * z + y * w),
            ],
            [
                two * (x * y + z * w),
                one - two * (x * x + z * z),
                two * (y * z - x * w),
            ],
            [
                two * (x * z - y * w),
                two * (y * z + x * w),
                one - two * (x * x + y * y),
            ],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn mat3_inverse_roundtrip() {
        let m = Mat3::new([[2.0, -1.0, 0.5], [0.3, 4.0, 1.0], [-0.2, 0.1, 3.0]]);
        let inv = m.inverse();
        // m * inv == I
        for c in 0..3 {
            let e = Vec3::new(
                if c == 0 { 1.0 } else { 0.0 },
                if c == 1 { 1.0 } else { 0.0 },
                if c == 2 { 1.0 } else { 0.0 },
            );
            let col = Vec3::new(inv.rows[0][c], inv.rows[1][c], inv.rows[2][c]);
            let got = m.matvec(col);
            assert!(approx(got.x, e.x, 1e-12) && approx(got.y, e.y, 1e-12) && approx(got.z, e.z, 1e-12));
        }
    }

    #[test]
    fn mat4_inverse_roundtrip() {
        let m = Mat4::new([
            [1.0, 1.0, 1.0, 1.0],
            [0.12, -0.12, -0.12, 0.12],
            [-0.12, -0.12, 0.12, 0.12],
            [0.024, -0.024, 0.024, -0.024],
        ]);
        let inv = m.inverse();
        for c in 0..4 {
            let mut e = [0.0; 4];
            e[c] = 1.0;
            let col = [inv.rows[0][c], inv.rows[1][c], inv.rows[2][c], inv.rows[3][c]];
            let got = m.matvec(col);
            for i in 0..4 {
                assert!(approx(got[i], e[i], 1e-9), "row {i} col {c}: {} != {}", got[i], e[i]);
            }
        }
    }

    #[test]
    fn quat_to_rotmat_matches_scipy() {
        // scipy: Rotation.from_quat([0.1, 0.2, 0.3, 0.9272...]).as_matrix() for a
        // 90-deg-ish rotation. Use a known unit quaternion: rotation of pi/2 about z
        // -> q = [0, 0, sin(pi/4), cos(pi/4)].
        let s = (std::f64::consts::FRAC_PI_4).sin();
        let q = Quat::new(0.0, 0.0, s, s);
        let r = q.to_rotmat();
        // Rz(90deg) = [[0,-1,0],[1,0,0],[0,0,1]]
        let expect = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx(r.rows[i][j], expect[i][j], 1e-12), "[{i}][{j}]");
            }
        }
    }

    #[test]
    fn quat_to_rotmat_normalizes() {
        // A non-unit quaternion must give an orthonormal matrix (scipy normalizes).
        let q = Quat::new(0.2, -0.4, 0.5, 0.8); // norm != 1
        let r = q.to_rotmat();
        // columns orthonormal: each column dot itself == 1
        for c in 0..3 {
            let col = Vec3::new(r.rows[0][c], r.rows[1][c], r.rows[2][c]);
            assert!(approx(col.dot(col), 1.0, 1e-12), "col {c} not unit");
        }
    }

    #[test]
    fn hat_is_cross() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-0.5, 0.7, 4.0);
        let h = Mat3::hat(a).matvec(b);
        let c = a.cross(b);
        assert!(approx(h.x, c.x, 1e-15) && approx(h.y, c.y, 1e-15) && approx(h.z, c.z, 1e-15));
    }
}
