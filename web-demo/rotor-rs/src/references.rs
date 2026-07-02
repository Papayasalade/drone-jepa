//! Random smooth reference trajectories (sum-of-sines), ported from
//! drone_jepa/data_gen/references.py. Each axis is a sum of sinusoids so position
//! and its derivatives are available in closed form for the SE3 controller.
//! (The paper uses a GP with exp-sine-squared kernels; this is the flatness-
//! friendly stand-in already used by the Python pipeline.)

use crate::linalg::Vec3;
use crate::rng::Rng;
use crate::se3::FlatRef;

const N_MODES: usize = 3;

// --------------------------------------------------------------------------- //
// GP references: per-axis position sampled from a Gaussian Process with an
// exponential-sine-squared (periodic) kernel, then a natural cubic spline gives
// C2 position + analytic velocity/acceleration (the paper's recipe; we sample at
// knots and spline-resample, matching "resampled using cubic splines").
// --------------------------------------------------------------------------- //

/// Natural cubic spline through (t, y) knots; eval returns (value, 1st, 2nd deriv).
struct Spline {
    t: Vec<f64>,
    y: Vec<f64>,
    m: Vec<f64>, // second derivatives at knots (M_0 = M_{n-1} = 0)
}

impl Spline {
    fn natural(t: Vec<f64>, y: Vec<f64>) -> Self {
        let n = t.len();
        let mut m = vec![0.0; n];
        if n < 3 {
            return Self { t, y, m };
        }
        let h: Vec<f64> = (0..n - 1).map(|i| t[i + 1] - t[i]).collect();
        // tridiagonal system for interior M (Thomas algorithm)
        let mut lo = vec![0.0; n]; // sub-diagonal
        let mut di = vec![1.0; n]; // diagonal (1 at the natural boundaries)
        let mut up = vec![0.0; n]; // super-diagonal
        let mut rhs = vec![0.0; n];
        for i in 1..n - 1 {
            lo[i] = h[i - 1] / 6.0;
            di[i] = (h[i - 1] + h[i]) / 3.0;
            up[i] = h[i] / 6.0;
            rhs[i] = (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1];
        }
        // forward elimination
        for i in 1..n {
            let w = lo[i] / di[i - 1];
            di[i] -= w * up[i - 1];
            rhs[i] -= w * rhs[i - 1];
        }
        // back substitution
        m[n - 1] = rhs[n - 1] / di[n - 1];
        for i in (0..n - 1).rev() {
            m[i] = (rhs[i] - up[i] * m[i + 1]) / di[i];
        }
        Self { t, y, m }
    }

    fn eval(&self, x: f64) -> (f64, f64, f64) {
        let n = self.t.len();
        // locate interval (clamp to ends)
        let mut i = 0;
        while i + 2 < n && x > self.t[i + 1] {
            i += 1;
        }
        let h = self.t[i + 1] - self.t[i];
        let a = (self.t[i + 1] - x) / h;
        let b = (x - self.t[i]) / h;
        let (yi, yi1, mi, mi1) = (self.y[i], self.y[i + 1], self.m[i], self.m[i + 1]);
        let val = a * yi + b * yi1 + h * h / 6.0 * ((a * a * a - a) * mi + (b * b * b - b) * mi1);
        let d1 = (yi1 - yi) / h + h / 6.0 * (-(3.0 * a * a - 1.0) * mi + (3.0 * b * b - 1.0) * mi1);
        let d2 = a * mi + b * mi1;
        (val, d1, d2)
    }
}

pub struct GpRef {
    sx: Spline,
    sy: Spline,
    sz: Spline,
    yaw_a: f64,
    yaw_w: f64,
    yaw_phi: f64,
}

impl GpRef {
    pub fn sample(r: &mut Rng, center: Vec3<f64>, t_final: f64) -> Self {
        let n = 26usize;
        let knots: Vec<f64> = (0..n).map(|i| i as f64 * t_final / (n - 1) as f64).collect();
        let u = |r: &mut Rng, a: f64, b: f64| a + (b - a) * r.uniform();
        let period = u(r, 3.0, 8.0);
        let ls = u(r, 0.6, 1.4);
        let pi = std::f64::consts::PI;

        // kernel matrix + jitter, Cholesky once (shared across axes)
        let mut l = vec![vec![0.0f64; n]; n];
        let mut k = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let d = (knots[i] - knots[j]).abs();
                let s = (pi * d / period).sin();
                k[i][j] = (-2.0 * s * s / (ls * ls)).exp();
            }
            k[i][i] += 1e-9; // jitter for PD
        }
        // Cholesky: K = L L^T
        for i in 0..n {
            for j in 0..=i {
                let mut sum = k[i][j];
                for p in 0..j {
                    sum -= l[i][p] * l[j][p];
                }
                if i == j {
                    l[i][j] = sum.max(1e-12).sqrt();
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
        }

        let axis = |r: &mut Rng, c: f64, amp: f64| -> Spline {
            let z: Vec<f64> = (0..n).map(|_| r.normal()).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| {
                    let mut v = 0.0;
                    for p in 0..=i {
                        v += l[i][p] * z[p];
                    }
                    c + amp * v
                })
                .collect();
            Spline::natural(knots.clone(), y)
        };

        let amp = u(r, 0.5, 1.4);
        GpRef {
            sx: axis(r, center.x, amp),
            sy: axis(r, center.y, amp),
            sz: axis(r, center.z, amp * 0.4), // taper z
            yaw_a: u(r, -0.4, 0.4),
            yaw_w: u(r, 0.2, 0.8),
            yaw_phi: u(r, 0.0, std::f64::consts::TAU),
        }
    }

    pub fn at(&self, t: f64) -> FlatRef {
        let (x, dx, ddx) = self.sx.eval(t);
        let (y, dy, ddy) = self.sy.eval(t);
        let (z, dz, ddz) = self.sz.eval(t);
        let yaw = self.yaw_a * (self.yaw_w * t + self.yaw_phi).sin();
        let yaw_dot = self.yaw_a * self.yaw_w * (self.yaw_w * t + self.yaw_phi).cos();
        FlatRef {
            x: Vec3::new(x, y, z),
            x_dot: Vec3::new(dx, dy, dz),
            x_ddot: Vec3::new(ddx, ddy, ddz),
            yaw,
            yaw_dot,
        }
    }
}

pub struct FourierRef {
    center: Vec3<f64>,
    amp: [[f64; N_MODES]; 3],  // per axis
    freq: [[f64; N_MODES]; 3],
    phase: [[f64; N_MODES]; 3],
    yaw_a: f64,
    yaw_w: f64,
    yaw_phi: f64,
}

impl FourierRef {
    pub fn sample(r: &mut Rng, center: Vec3<f64>) -> Self {
        let mut amp = [[0.0; N_MODES]; 3];
        let mut freq = [[0.0; N_MODES]; 3];
        let mut phase = [[0.0; N_MODES]; 3];
        let u = |r: &mut Rng, a: f64, b: f64| a + (b - a) * r.uniform();
        for axis in 0..3 {
            for m in 0..N_MODES {
                amp[axis][m] = u(r, 0.3, 1.0);
                freq[axis][m] = u(r, 0.2, 1.0);
                phase[axis][m] = u(r, 0.0, std::f64::consts::TAU);
            }
        }
        // taper z so it doesn't dive into the ground
        for m in 0..N_MODES {
            amp[2][m] *= 0.4;
        }
        FourierRef {
            center,
            amp,
            freq,
            phase,
            yaw_a: u(r, -0.4, 0.4),
            yaw_w: u(r, 0.2, 0.8),
            yaw_phi: u(r, 0.0, std::f64::consts::TAU),
        }
    }

    pub fn at(&self, t: f64) -> FlatRef {
        let mut xv = [self.center.x, self.center.y, self.center.z];
        let mut dv = [0.0; 3];
        let mut ddv = [0.0; 3];
        for axis in 0..3 {
            for m in 0..N_MODES {
                let (a, w, p) = (self.amp[axis][m], self.freq[axis][m], self.phase[axis][m]);
                let s = (w * t + p).sin();
                let c = (w * t + p).cos();
                xv[axis] += a * s;
                dv[axis] += a * w * c;
                ddv[axis] += -a * w * w * s;
            }
        }
        let yaw = self.yaw_a * (self.yaw_w * t + self.yaw_phi).sin();
        let yaw_dot = self.yaw_a * self.yaw_w * (self.yaw_w * t + self.yaw_phi).cos();
        FlatRef {
            x: Vec3::new(xv[0], xv[1], xv[2]),
            x_dot: Vec3::new(dv[0], dv[1], dv[2]),
            x_ddot: Vec3::new(ddv[0], ddv[1], ddv[2]),
            yaw,
            yaw_dot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_interpolates_knots() {
        let t = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.5, 2.0, -1.0];
        let s = Spline::natural(t.clone(), y.clone());
        for i in 0..t.len() {
            let (v, _, _) = s.eval(t[i]);
            assert!((v - y[i]).abs() < 1e-9, "knot {i}: {v} != {}", y[i]);
        }
        // C1 continuity across an interior knot: derivative from both sides matches
        let (_, d_lo, _) = s.eval(2.0 - 1e-6);
        let (_, d_hi, _) = s.eval(2.0 + 1e-6);
        assert!((d_lo - d_hi).abs() < 1e-3, "derivative jump at knot");
    }

    #[test]
    fn gp_ref_is_finite_and_smooth() {
        let mut r = Rng::new(1);
        let g = GpRef::sample(&mut r, Vec3::new(0.0, 0.0, 1.5), 10.0);
        let f0 = g.at(0.0);
        let f5 = g.at(5.0);
        assert!(f0.x.x.is_finite() && f5.x_ddot.norm().is_finite());
        // finite-difference check: x_dot ~ d/dt x
        let dt = 1e-4;
        let a = g.at(3.0).x.x;
        let b = g.at(3.0 + dt).x.x;
        let fd = (b - a) / dt;
        assert!((fd - g.at(3.0).x_dot.x).abs() < 1e-2, "x_dot mismatch fd");
    }
}
