//! Zero-dependency inference of the PPO-trained racing policy (the third drone).
//! Unlike the MPPI drones, this is REACTIVE: one tiny MLP forward per control step
//! maps the observation straight to a CTBR action — no model, no planning, no
//! rollouts. Trained with PufferLib on the vectorized rotor-rs env (see
//! `drone_jepa/rl/`); weights loaded from the `.rlb` blob (`export_rl_blob.py`).
//!
//! The observation MUST match the env's `write_obs` exactly:
//!   [ vel/5 (3), R row-major (9), omega/5 (3), rel_gate1/5 (3), rel_gate2/5 (3) ] = 21
//! Action in [-1,1]^4 → thrust = hover·(1+a0), body-rates = a[1..4]·RATE_MAX.

use crate::gates::Course;
use rotor_rs::linalg::Vec3;
use crate::mppi::Controller;
use rotor_rs::params::GRAVITY;
use rotor_rs::state::State;
use crate::CtbrCmd;

const RATE_MAX: f64 = 10.0; // must match the training env
const OBS_SCALE: f32 = 0.2; // 1/5, matches env write_obs

#[inline]
fn erf(x: f32) -> f32 {
    let z = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587
                                        + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    (if x >= 0.0 { 1.0 - ans } else { ans - 1.0 }) as f32
}
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// The reactive PPO policy: a 2-hidden-layer MLP (obs -> 85 -> 85 -> act).
pub struct RlPolicy {
    obs_dim: usize,
    hidden: usize,
    act_dim: usize,
    w0: Vec<f32>, b0: Vec<f32>, // obs -> hidden
    w2: Vec<f32>, b2: Vec<f32>, // hidden -> hidden
    wm: Vec<f32>, bm: Vec<f32>, // hidden -> act (mean)
    hover_thrust: f64,
    f_max: f64, // per-rotor ceiling for the rotor-force action mode
}

impl RlPolicy {
    /// Parse a `.rlb` blob (see `scripts/export_rl_blob.py`). `mass` sets the
    /// nominal hover thrust used to denormalize the action's thrust channel.
    pub fn from_blob(b: &[u8], mass: f64) -> Self {
        let u16r = |b: &[u8], p: &mut usize| { let v = u16::from_le_bytes([b[*p], b[*p+1]]); *p += 2; v as usize };
        let u32r = |b: &[u8], p: &mut usize| { let v = u32::from_le_bytes([b[*p], b[*p+1], b[*p+2], b[*p+3]]); *p += 4; v as usize };
        let u8r = |b: &[u8], p: &mut usize| { let v = b[*p]; *p += 1; v as usize };
        let f32r = |b: &[u8], p: &mut usize| { let mut a=[0u8;4]; a.copy_from_slice(&b[*p..*p+4]); *p+=4; f32::from_le_bytes(a) };
        let strr = |b: &[u8], p: &mut usize, n: usize| { let s = String::from_utf8(b[*p..*p+n].to_vec()).unwrap(); *p+=n; s };

        assert_eq!(&b[0..5], b"RLB1\n", "bad rlb magic");
        let mut p = 5usize;
        let obs_dim = u32r(b, &mut p);
        let hidden = u32r(b, &mut p);
        let act_dim = u32r(b, &mut p);
        let n = u32r(b, &mut p);
        use std::collections::HashMap;
        let mut t: HashMap<String, Vec<f32>> = HashMap::new();
        for _ in 0..n {
            let nn = u16r(b, &mut p);
            let name = strr(b, &mut p, nn);
            let ndim = u8r(b, &mut p);
            let mut count = 1usize;
            for _ in 0..ndim { count *= u32r(b, &mut p); }
            let mut data = Vec::with_capacity(count);
            for _ in 0..count { data.push(f32r(b, &mut p)); }
            t.insert(name, data);
        }
        RlPolicy {
            obs_dim, hidden, act_dim,
            w0: t.remove("encoder.0.weight").unwrap(), b0: t.remove("encoder.0.bias").unwrap(),
            w2: t.remove("encoder.2.weight").unwrap(), b2: t.remove("encoder.2.bias").unwrap(),
            wm: t.remove("decoder_mean.weight").unwrap(), bm: t.remove("decoder_mean.bias").unwrap(),
            hover_thrust: mass * GRAVITY,
            f_max: f64::INFINITY,
        }
    }

    /// Live mass change → refresh the thrust scale.
    pub fn set_mass(&mut self, mass: f64) {
        self.hover_thrust = mass * GRAVITY;
    }

    /// Per-rotor force ceiling (k_eta * rpm_max²) for the rotor-force mode.
    pub fn set_f_max(&mut self, f_max: f64) {
        self.f_max = f_max;
    }

    /// Rotor-force action mode: the SAME actor network, but actions are read as
    /// 4 per-rotor forces around hover (matches rl-env's RL_ACTION_MODE=rotor:
    /// f_i = hover/4 · (1 + a_i), clamped to [0, f_max]). No inner rate loop —
    /// the same raw actuator space as the rotor-force JEPA drone.
    pub fn act_rotor(&mut self, state: &State<f64>, course: &Course) -> [f64; 4] {
        let obs = self.build_obs(state, course);
        let a = self.forward(&obs);
        let per_hover = self.hover_thrust / 4.0;
        core::array::from_fn(|i| {
            (per_hover * (1.0 + a[i] as f64).clamp(0.0, 3.0)).min(self.f_max)
        })
    }

    fn build_obs(&self, s: &State<f64>, course: &Course) -> [f32; 21] {
        let r = s.q.to_rotmat();
        let p = s.x;
        let n = course.gates.len();
        let g1 = if course.next < n { course.gates[course.next].center } else { p };
        let g2 = if course.next + 1 < n { course.gates[course.next + 1].center } else { g1 };
        let rel1 = g1 - p;
        let rel2 = g2 - p;
        let mut o = [0f32; 21];
        o[0] = (s.v.x as f32) * OBS_SCALE; o[1] = (s.v.y as f32) * OBS_SCALE; o[2] = (s.v.z as f32) * OBS_SCALE;
        for i in 0..3 { for j in 0..3 { o[3 + i * 3 + j] = r.rows[i][j] as f32; } }
        o[12] = (s.w.x as f32) * OBS_SCALE; o[13] = (s.w.y as f32) * OBS_SCALE; o[14] = (s.w.z as f32) * OBS_SCALE;
        o[15] = (rel1.x as f32) * OBS_SCALE; o[16] = (rel1.y as f32) * OBS_SCALE; o[17] = (rel1.z as f32) * OBS_SCALE;
        o[18] = (rel2.x as f32) * OBS_SCALE; o[19] = (rel2.y as f32) * OBS_SCALE; o[20] = (rel2.z as f32) * OBS_SCALE;
        o
    }

    /// MLP forward → action mean (act_dim). obs -> 85 GELU -> 85 GELU -> act.
    fn forward(&self, obs: &[f32]) -> Vec<f32> {
        let layer = |inp: &[f32], w: &[f32], b: &[f32], out: usize, inn: usize, act: bool| -> Vec<f32> {
            (0..out).map(|i| {
                let mut acc = b[i];
                let row = &w[i * inn..i * inn + inn];
                for j in 0..inn { acc += row[j] * inp[j]; }
                if act { gelu(acc) } else { acc }
            }).collect()
        };
        let h1 = layer(obs, &self.w0, &self.b0, self.hidden, self.obs_dim, true);
        let h2 = layer(&h1, &self.w2, &self.b2, self.hidden, self.hidden, true);
        layer(&h2, &self.wm, &self.bm, self.act_dim, self.hidden, false)
    }
}

impl Controller for RlPolicy {
    fn act(&mut self, state: &State<f64>, course: &Course) -> CtbrCmd<f64> {
        let obs = self.build_obs(state, course);
        let a = self.forward(&obs);
        let thrust = self.hover_thrust * (1.0 + a[0] as f64).clamp(0.0, 3.0);
        CtbrCmd {
            thrust,
            w_cmd: Vec3::new(a[1] as f64 * RATE_MAX, a[2] as f64 * RATE_MAX, a[3] as f64 * RATE_MAX),
        }
    }
}
