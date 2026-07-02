//! Quadrotor forward dynamics + fixed-step integrator, ported from RotorPy
//! `Multirotor` (single-vehicle path only). Generic over `S: Scalar` (the
//! batching lever) and `C: ControlLaw` (the abstraction, monomorphized).

use core::marker::PhantomData;

use crate::control::ControlLaw;
use crate::linalg::Vec3;
use crate::params::{QuadParams, QuadParamsInput, NUM_ROTORS};
use crate::scalar::Scalar;
use crate::state::{State, STATE_LEN};

/// Body-frame total force & moment for given rotor speeds and body-frame airspeed.
/// Ported from `compute_body_wrench` (aero always on; thrust + parasitic drag +
/// rotor drag H + translational lift + flapping + yaw moment).
#[inline(always)]
fn body_wrench<S: Scalar>(
    p: &QuadParams<S>,
    body_rates: Vec3<S>,
    rotor_speeds: &[S; NUM_ROTORS],
    body_airspeed: Vec3<S>,
) -> (Vec3<S>, Vec3<S>) {
    let mut ftot = Vec3::zero();
    // moment from rotor forces = sum_j r_j x (T+H), i.e. physical r x F.
    // (RotorPy writes `-einsum('ijk,ik->j', hat(geom), F)`, but contracting the
    // hat's ROW index is a transpose that cancels the leading minus -> +r x F.)
    let mut m_force_acc = Vec3::zero();
    let mut m_rest = Vec3::zero(); // yaw + flapping

    for j in 0..NUM_ROTORS {
        let om = rotor_speeds[j];
        let om2 = om * om;
        // local airspeed at rotor j: v_body + w x r_j
        let a = body_airspeed + body_rates.cross(p.rotor_pos[j]);

        // thrust (body z) + translational lift; per-rotor gain = "motor power"
        let tz = p.rotor_gain[j] * p.k_eta * om2 + p.k_h * (a.x * a.x + a.y * a.y);
        let t = Vec3::new(S::ZERO, S::ZERO, tz);
        // rotor drag (H force): -om * diag(k_d, k_d, k_z) @ a
        let h = Vec3::new(p.k_d * a.x, p.k_d * a.y, p.k_z * a.z).scale(-om);
        let tph = t + h;

        ftot = ftot + tph;
        m_force_acc = m_force_acc + p.rotor_pos[j].cross(tph);

        // yaw moment (body z) + pitching flapping moment
        let myaw = Vec3::new(S::ZERO, S::ZERO, p.rotor_dir[j] * p.k_m * om2);
        let mflap = Vec3::new(a.y, -a.x, S::ZERO).scale(-p.k_flap * om);
        m_rest = m_rest + myaw + mflap;
    }

    // parasitic drag at CoM: -||v_air|| * diag(c_D) @ v_air
    let speed = body_airspeed.norm();
    let drag = body_airspeed.hadamard(p.drag_diag).scale(-speed);
    ftot = ftot + drag;

    let mtot = m_force_acc + m_rest;
    (ftot, mtot)
}

/// Full 20-dim state derivative for fixed (already clipped) commanded rotor
/// speeds. Ported from `_s_dot_fn`.
#[inline(always)]
fn s_dot<S: Scalar>(
    p: &QuadParams<S>,
    s: &[S; STATE_LEN],
    cmd_rotor_speeds: &[S; NUM_ROTORS],
) -> [S; STATE_LEN] {
    let st = State::unpack(s);
    let r = st.q.to_rotmat(); // body->world (normalizes q like scipy)

    // body-frame airspeed = R^T (v - wind)
    let body_airspeed = r.tmatvec(st.v - st.wind);
    let (ftot_b, mtot_b) = body_wrench(p, st.w, &st.rotor_speeds, body_airspeed);

    // translational dynamics
    let ftot = r.matvec(ftot_b);
    let inv_mass = S::ONE / p.mass;
    let v_dot = (p.weight + ftot).scale(inv_mass);

    // rotational dynamics: w_dot = I^-1 (M - w x (I w))
    let iw = p.inertia.matvec(st.w);
    let gyro = st.w.cross(iw);
    let w_dot = p.inv_inertia.matvec(mtot_b - gyro);

    // kinematics
    let x_dot = st.v;
    let q_dot = crate::so3::quat_dot(st.q, st.w); // uses raw (non-unit) q

    // motor first-order lag (rotor_speeds is integrated state)
    let inv_tau = S::ONE / p.tau_m;
    let rotor_accel: [S; NUM_ROTORS] =
        core::array::from_fn(|j| (cmd_rotor_speeds[j] - st.rotor_speeds[j]) * inv_tau);

    let qd = q_dot.to_array();
    let mut out = [S::ZERO; STATE_LEN];
    out[0] = x_dot.x; out[1] = x_dot.y; out[2] = x_dot.z;
    out[3] = v_dot.x; out[4] = v_dot.y; out[5] = v_dot.z;
    out[6] = qd[0]; out[7] = qd[1]; out[8] = qd[2]; out[9] = qd[3];
    out[10] = w_dot.x; out[11] = w_dot.y; out[12] = w_dot.z;
    // wind_dot = 0 (s[13..16] stay zero)
    for j in 0..NUM_ROTORS {
        out[16 + j] = rotor_accel[j];
    }
    out
}

// ---- packed-vector arithmetic helpers (branchless, fixed length) ----------- //
#[inline(always)]
fn axpy<S: Scalar>(a: S, x: &[S; STATE_LEN], y: &[S; STATE_LEN]) -> [S; STATE_LEN] {
    core::array::from_fn(|i| a * x[i] + y[i])
}

/// One fixed-step RK4 step of size `h` for the autonomous ODE (control held constant).
#[inline(always)]
fn rk4<S: Scalar>(
    p: &QuadParams<S>,
    s: &[S; STATE_LEN],
    cmd: &[S; NUM_ROTORS],
    h: S,
) -> [S; STATE_LEN] {
    let half = h * S::splat(0.5);
    let k1 = s_dot(p, s, cmd);
    let s2 = axpy(half, &k1, s);
    let k2 = s_dot(p, &s2, cmd);
    let s3 = axpy(half, &k2, s);
    let k3 = s_dot(p, &s3, cmd);
    let s4 = axpy(h, &k3, s);
    let k4 = s_dot(p, &s4, cmd);
    let sixth = h * S::splat(1.0 / 6.0);
    let third = h * S::splat(1.0 / 3.0);
    core::array::from_fn(|i| s[i] + sixth * (k1[i] + k4[i]) + third * (k2[i] + k3[i]))
}

/// Clip commanded rotor speeds to the motor range (matches RotorPy `step`'s clip).
#[inline]
pub fn clip_speeds<S: Scalar>(
    params: &QuadParams<S>,
    raw: [S; NUM_ROTORS],
) -> [S; NUM_ROTORS] {
    core::array::from_fn(|j| raw[j].clamp(params.rotor_speed_min, params.rotor_speed_max))
}

/// Integrate one control step of duration `dt` holding the (already clipped)
/// commanded rotor speeds constant. Abstraction-independent, so callers that pick
/// the control law at runtime (e.g. the WASM wrapper) can share one `QuadParams`.
#[inline]
pub fn integrate<S: Scalar>(
    params: &QuadParams<S>,
    state: &State<S>,
    cmd_speeds: &[S; NUM_ROTORS],
    dt: S,
    n_sub: usize,
) -> State<S> {
    let n_sub = n_sub.max(1);
    let h = dt * S::splat(1.0 / n_sub as f64);
    let mut s = state.pack();
    for _ in 0..n_sub {
        s = rk4(params, &s, cmd_speeds, h);
    }
    let mut out = State::unpack(&s);
    // renormalize quaternion (after the full step, like RotorPy)
    out.q = out.q.normalized();
    // clip resulting rotor speeds to [min, max] (RotorPy step, motor noise = 0)
    out.rotor_speeds = clip_speeds(params, out.rotor_speeds);
    out
}

/// A quadrotor parameterized by scalar `S` and control abstraction `C`.
pub struct Multirotor<S, C> {
    pub params: QuadParams<S>,
    /// fixed substeps per `step` call (branchless: a compile-/run-time constant count).
    pub n_sub: usize,
    _marker: PhantomData<C>,
}

impl<S: Scalar, C: ControlLaw<S>> Multirotor<S, C> {
    pub fn new(input: &QuadParamsInput) -> Self {
        Self::with_substeps(input, 1)
    }

    pub fn with_substeps(input: &QuadParamsInput, n_sub: usize) -> Self {
        Self {
            params: QuadParams::from_input(input),
            n_sub: n_sub.max(1),
            _marker: PhantomData,
        }
    }

    /// Override the per-rotor thrust gain ("motor power"); 1.0 = nominal.
    pub fn set_rotor_gain(&mut self, gain: [S; NUM_ROTORS]) {
        self.params.rotor_gain = gain;
    }

    /// Commanded rotor speeds after clipping to [min, max] (the value the
    /// integrator actually uses), matching RotorPy `step`'s clip.
    #[inline]
    pub fn clipped_cmd_speeds(&self, state: &State<S>, cmd: &C::Command) -> [S; NUM_ROTORS] {
        clip_speeds(&self.params, C::cmd_rotor_speeds(&self.params, state, cmd))
    }

    /// Full 20-dim derivative at `state` for `cmd` (the exact-math gate target).
    pub fn state_derivative(&self, state: &State<S>, cmd: &C::Command) -> [S; STATE_LEN] {
        let speeds = self.clipped_cmd_speeds(state, cmd);
        s_dot(&self.params, &state.pack(), &speeds)
    }

    /// Integrate one control step of duration `dt` holding `cmd` constant.
    pub fn step(&self, state: &State<S>, cmd: &C::Command, dt: S) -> State<S> {
        let speeds = self.clipped_cmd_speeds(state, cmd);
        integrate(&self.params, state, &speeds, dt, self.n_sub)
    }
}
