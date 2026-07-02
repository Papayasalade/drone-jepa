//! NMPC via iLQR — an optimization-based tracking controller (an alternative to
//! SE3 for data generation, giving a "complementary action distribution" as the
//! paper notes). It optimizes a collective-thrust + body-moment sequence over a
//! short horizon to track a position reference, then maps the first command to
//! rotor forces via the allocation matrix.
//!
//! State x = [pos3, vel3, q4(i,j,k,w), omega3] (13). Control u = [T, Mx, My, Mz] (4).
//! Jacobians are finite-differenced (robust across domain-randomized drones).

use crate::linalg::{Mat3, Mat4, Quat, Vec3};
use crate::params::{QuadParams, QuadParamsInput, GRAVITY, NUM_ROTORS};
use crate::se3::FlatRef;

const NX: usize = 13;
const NU: usize = 4;

type X = [f64; NX];
type U = [f64; NU];

pub struct Nmpc {
    mass: f64,
    inertia: Mat3<f64>,
    inv_inertia: Mat3<f64>,
    tm_to_f: Mat4<f64>,
    dt: f64,
    horizon: usize,
    iters: usize,
    // cost weights (diagonal)
    q_pos: f64,
    q_vel: f64,
    q_att: f64,
    q_omega: f64,
    r_thrust: f64,
    r_moment: f64,
    u_prev: Vec<U>, // warm-start nominal control
}

#[inline]
fn quat_dot(q: &[f64; 4], w: &Vec3<f64>) -> [f64; 4] {
    let (i, j, k, ww) = (q[0], q[1], q[2], q[3]);
    [
        0.5 * (ww * w.x - k * w.y + j * w.z),
        0.5 * (k * w.x + ww * w.y - i * w.z),
        0.5 * (-j * w.x + i * w.y + ww * w.z),
        0.5 * (-i * w.x - j * w.y - k * w.z),
    ]
}

impl Nmpc {
    pub fn new(p: &QuadParamsInput, dt: f64, horizon: usize) -> Self {
        let qp = QuadParams::<f64>::from_input(p);
        Nmpc {
            mass: p.mass,
            inertia: qp.inertia,
            inv_inertia: qp.inv_inertia,
            tm_to_f: qp.tm_to_f,
            dt,
            horizon,
            iters: 6,
            q_pos: 12.0,
            q_vel: 2.0,
            q_att: 8.0,
            q_omega: 0.5,
            r_thrust: 0.002,
            r_moment: 2.0,
            u_prev: vec![[p.mass * GRAVITY, 0.0, 0.0, 0.0]; horizon],
        }
    }

    /// One rigid-body dynamics step (ctbm control).
    fn f(&self, x: &X, u: &U) -> X {
        let p = Vec3::new(x[0], x[1], x[2]);
        let v = Vec3::new(x[3], x[4], x[5]);
        let q = Quat::new(x[6], x[7], x[8], x[9]);
        let w = Vec3::new(x[10], x[11], x[12]);
        let r = q.to_rotmat();
        let thrust = u[0];
        let moment = Vec3::new(u[1], u[2], u[3]);

        let body_z = Vec3::new(r.rows[0][2], r.rows[1][2], r.rows[2][2]);
        let vdot = Vec3::new(0.0, 0.0, -GRAVITY) + body_z.scale(thrust / self.mass);
        let v_next = v + vdot.scale(self.dt);
        let p_next = p + v_next.scale(self.dt);

        let iw = self.inertia.matvec(w);
        let wdot = self.inv_inertia.matvec(moment - w.cross(iw));
        let w_next = w + wdot.scale(self.dt);

        let qd = quat_dot(&q.to_array(), &w);
        let mut qn = [x[6] + qd[0] * self.dt, x[7] + qd[1] * self.dt, x[8] + qd[2] * self.dt, x[9] + qd[3] * self.dt];
        let n = (qn[0] * qn[0] + qn[1] * qn[1] + qn[2] * qn[2] + qn[3] * qn[3]).sqrt();
        for c in &mut qn { *c /= n; }

        [
            p_next.x, p_next.y, p_next.z,
            v_next.x, v_next.y, v_next.z,
            qn[0], qn[1], qn[2], qn[3],
            w_next.x, w_next.y, w_next.z,
        ]
    }

    /// Stage cost gradient/Hessian are diagonal; build target state from the ref.
    fn x_ref(&self, r: &FlatRef) -> X {
        // upright attitude at the reference yaw
        let (cy, sy) = ((r.yaw * 0.5).cos(), (r.yaw * 0.5).sin());
        [
            r.x.x, r.x.y, r.x.z,
            r.x_dot.x, r.x_dot.y, r.x_dot.z,
            0.0, 0.0, sy, cy, // quaternion for yaw about z
            0.0, 0.0, r.yaw_dot,
        ]
    }

    #[inline]
    fn qdiag(&self) -> [f64; NX] {
        [
            self.q_pos, self.q_pos, self.q_pos,
            self.q_vel, self.q_vel, self.q_vel,
            self.q_att, self.q_att, self.q_att, self.q_att,
            self.q_omega, self.q_omega, self.q_omega,
        ]
    }
    #[inline]
    fn rdiag(&self) -> [f64; NU] {
        [self.r_thrust, self.r_moment, self.r_moment, self.r_moment]
    }

    fn stage_cost(&self, x: &X, u: &U, xr: &X, ur: &U) -> f64 {
        let qd = self.qdiag();
        let rd = self.rdiag();
        let mut c = 0.0;
        for i in 0..NX {
            let d = x[i] - xr[i];
            c += qd[i] * d * d;
        }
        for i in 0..NU {
            let d = u[i] - ur[i];
            c += rd[i] * d * d;
        }
        c
    }

    /// Finite-difference Jacobians A=df/dx (NX x NX), B=df/du (NX x NU).
    fn jacobians(&self, x: &X, u: &U) -> ([[f64; NX]; NX], [[f64; NU]; NX]) {
        let eps = 1e-6;
        let f0 = self.f(x, u);
        let mut a = [[0.0; NX]; NX];
        let mut b = [[0.0; NU]; NX];
        for j in 0..NX {
            let mut xp = *x;
            xp[j] += eps;
            let fp = self.f(&xp, u);
            for i in 0..NX {
                a[i][j] = (fp[i] - f0[i]) / eps;
            }
        }
        for j in 0..NU {
            let mut up = *u;
            up[j] += eps;
            let fp = self.f(x, &up);
            for i in 0..NX {
                b[i][j] = (fp[i] - f0[i]) / eps;
            }
        }
        (a, b)
    }

    /// iLQR: optimize the control sequence to track `refs`, returns first control.
    fn ilqr(&mut self, x0: &X, refs: &[FlatRef]) -> U {
        let h = self.horizon.min(refs.len());
        let qd = self.qdiag();
        let rd = self.rdiag();
        let ur = [self.mass * GRAVITY, 0.0, 0.0, 0.0];
        let xrs: Vec<X> = refs.iter().take(h).map(|r| self.x_ref(r)).collect();

        // nominal control (warm start) + rollout
        let mut us = self.u_prev.clone();
        us.resize(h, ur);
        let rollout = |us: &[U]| -> Vec<X> {
            let mut xs = vec![*x0];
            for t in 0..h {
                let nx = self.f(&xs[t], &us[t]);
                xs.push(nx);
            }
            xs
        };
        let cost = |xs: &[X], us: &[U]| -> f64 {
            let mut c = 0.0;
            for t in 0..h {
                c += self.stage_cost(&xs[t], &us[t], &xrs[t], &ur);
            }
            // terminal
            for i in 0..NX {
                let d = xs[h][i] - xrs[h - 1][i];
                c += 5.0 * qd[i] * d * d;
            }
            c
        };

        let mut xs = rollout(&us);
        let mut j_cur = cost(&xs, &us);
        let mut reg = 1e-3;

        for _ in 0..self.iters {
            // backward pass: quadratic value function V(x) = 0.5 x^T Vxx x + vx^T x
            let mut vx = [0.0; NX];
            let mut vxx = [[0.0; NX]; NX];
            for i in 0..NX {
                let d = xs[h][i] - xrs[h - 1][i];
                vx[i] = 2.0 * 5.0 * qd[i] * d;
                vxx[i][i] = 2.0 * 5.0 * qd[i];
            }
            let mut k_ff = vec![[0.0; NU]; h];
            let mut k_fb = vec![[[0.0; NX]; NU]; h];
            let mut ok = true;
            for t in (0..h).rev() {
                let (a, b) = self.jacobians(&xs[t], &us[t]);
                // l_x, l_u (gradients), l_xx, l_uu (diagonal Hessians)
                let mut lx = [0.0; NX];
                let mut lxx = [0.0; NX];
                for i in 0..NX {
                    lx[i] = 2.0 * qd[i] * (xs[t][i] - xrs[t][i]);
                    lxx[i] = 2.0 * qd[i];
                }
                let mut lu = [0.0; NU];
                let mut luu = [0.0; NU];
                for i in 0..NU {
                    lu[i] = 2.0 * rd[i] * (us[t][i] - ur[i]);
                    luu[i] = 2.0 * rd[i];
                }
                // Q-function terms.  Qx = lx + A^T vx ; Qu = lu + B^T vx
                let mut qx = lx;
                for i in 0..NX {
                    for k in 0..NX { qx[i] += a[k][i] * vx[k]; }
                }
                let mut qu = lu;
                for i in 0..NU {
                    for k in 0..NX { qu[i] += b[k][i] * vx[k]; }
                }
                // Qxx = lxx + A^T Vxx A ; Quu = luu + B^T Vxx B ; Qux = B^T Vxx A
                let mut qxx = [[0.0; NX]; NX];
                let mut quu = [[0.0; NU]; NU];
                let mut qux = [[0.0; NX]; NU];
                // precompute Vxx A and Vxx B
                let mut vxa = [[0.0; NX]; NX];
                let mut vxb = [[0.0; NU]; NX];
                for i in 0..NX {
                    for j in 0..NX {
                        let mut s = 0.0;
                        for k in 0..NX { s += vxx[i][k] * a[k][j]; }
                        vxa[i][j] = s;
                    }
                    for j in 0..NU {
                        let mut s = 0.0;
                        for k in 0..NX { s += vxx[i][k] * b[k][j]; }
                        vxb[i][j] = s;
                    }
                }
                for i in 0..NX {
                    for j in 0..NX {
                        let mut s = 0.0;
                        for k in 0..NX { s += a[k][i] * vxa[k][j]; }
                        qxx[i][j] = s;
                    }
                    qxx[i][i] += lxx[i];
                }
                for i in 0..NU {
                    for j in 0..NU {
                        let mut s = 0.0;
                        for k in 0..NX { s += b[k][i] * vxb[k][j]; }
                        quu[i][j] = s;
                    }
                    quu[i][i] += luu[i] + reg;
                    for j in 0..NX {
                        let mut s = 0.0;
                        for k in 0..NX { s += b[k][i] * vxa[k][j]; }
                        qux[i][j] = s;
                    }
                }
                // invert Quu (4x4)
                let quu_inv = match inv4(&quu) {
                    Some(m) => m,
                    None => { ok = false; break; }
                };
                // gains: k = -Quu^-1 Qu ; K = -Quu^-1 Qux
                let mut kff = [0.0; NU];
                let mut kfb = [[0.0; NX]; NU];
                for i in 0..NU {
                    for k in 0..NU { kff[i] -= quu_inv[i][k] * qu[k]; }
                    for j in 0..NX {
                        let mut s = 0.0;
                        for k in 0..NU { s += quu_inv[i][k] * qux[k][j]; }
                        kfb[i][j] = -s;
                    }
                }
                k_ff[t] = kff;
                k_fb[t] = kfb;
                // value update: Vx = Qx + K^T Quu k + K^T Qu + Qux^T k ; Vxx = Qxx + K^T Quu K + K^T Qux + Qux^T K
                let mut nvx = qx;
                for i in 0..NX {
                    for a2 in 0..NU {
                        nvx[i] += kfb[a2][i] * qu[a2];
                        for b2 in 0..NU { nvx[i] += kfb[a2][i] * quu[a2][b2] * kff[b2]; }
                        nvx[i] += qux[a2][i] * kff[a2];
                    }
                }
                let mut nvxx = qxx;
                for i in 0..NX {
                    for j in 0..NX {
                        for a2 in 0..NU {
                            nvxx[i][j] += kfb[a2][i] * qux[a2][j] + qux[a2][i] * kfb[a2][j];
                            for b2 in 0..NU { nvxx[i][j] += kfb[a2][i] * quu[a2][b2] * kfb[b2][j]; }
                        }
                    }
                }
                vx = nvx;
                vxx = nvxx;
            }
            if !ok {
                reg *= 10.0;
                if reg > 1e6 { break; }
                continue;
            }
            // forward pass with line search
            let mut improved = false;
            for &alpha in &[1.0, 0.5, 0.25, 0.1, 0.03] {
                let mut nus = us.clone();
                let mut nxs = vec![*x0];
                for t in 0..h {
                    let mut du = [0.0; NU];
                    for i in 0..NU {
                        du[i] = alpha * k_ff[t][i];
                        for j in 0..NX { du[i] += k_fb[t][i][j] * (nxs[t][j] - xs[t][j]); }
                        nus[t][i] = us[t][i] + du[i];
                    }
                    nus[t][0] = nus[t][0].max(0.0); // thrust >= 0
                    nxs.push(self.f(&nxs[t], &nus[t]));
                }
                let jn = cost(&nxs, &nus);
                if jn.is_finite() && jn < j_cur {
                    us = nus;
                    xs = nxs;
                    j_cur = jn;
                    improved = true;
                    reg = (reg * 0.7).max(1e-6);
                    break;
                }
            }
            if !improved {
                reg *= 10.0;
                if reg > 1e6 { break; }
            }
        }

        // warm-start next call (shift)
        let first = us[0];
        for t in 0..h - 1 { self.u_prev[t] = us[t + 1]; }
        if h >= 1 { self.u_prev[h - 1] = ur; }
        first
    }

    /// Track a window of references; returns four rotor forces [N] for the first step.
    pub fn update(&mut self, state: &crate::State<f64>, refs: &[FlatRef]) -> [f64; NUM_ROTORS] {
        let q = state.q.to_array();
        let x0: X = [
            state.x.x, state.x.y, state.x.z,
            state.v.x, state.v.y, state.v.z,
            q[0], q[1], q[2], q[3],
            state.w.x, state.w.y, state.w.z,
        ];
        let u = self.ilqr(&x0, refs);
        // ctbm -> rotor forces via allocation
        self.tm_to_f.matvec([u[0], u[1], u[2], u[3]])
    }
}

/// 4x4 inverse (Gauss-Jordan); None if singular.
fn inv4(m: &[[f64; NU]; NU]) -> Option<[[f64; NU]; NU]> {
    let mut a = *m;
    let mut inv = [[0.0; NU]; NU];
    for i in 0..NU { inv[i][i] = 1.0; }
    for col in 0..NU {
        // pivot
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in col + 1..NU {
            if a[r][col].abs() > best { best = a[r][col].abs(); piv = r; }
        }
        if best < 1e-12 { return None; }
        a.swap(col, piv);
        inv.swap(col, piv);
        let d = a[col][col];
        for k in 0..NU { a[col][k] /= d; inv[col][k] /= d; }
        for r in 0..NU {
            if r == col { continue; }
            let f = a[r][col];
            for k in 0..NU { a[r][k] -= f * a[col][k]; inv[r][k] -= f * inv[col][k]; }
        }
    }
    Some(inv)
}
