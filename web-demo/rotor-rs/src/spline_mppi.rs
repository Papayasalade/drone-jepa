//! Spline / contouring-cost MPPI (MPCC-style) for the JEPA world model.
//!
//! Instead of asking the model to PLAN toward a gate (reward velocity-to-gate-center,
//! where its rank-fidelity collapses), we build a model-free reference SPLINE from the
//! drone to the next gate that arrives PERPENDICULAR to the gate disk (tangent = gate
//! normal), and cost each model rollout by how well it tracks that spline:
//!   - contouring error: perpendicular distance from the predicted path to the spline,
//!   - velocity term: reward speed ALONG the spline tangent, penalize across it,
//!   - progress: reward getting far along the spline (+ a gate-pass bonus).
//! The geometry does the planning; the model only has to TRACK — a far better-conditioned
//! question for an imperfect world model.

use crate::gates::{Course, Gate};
use crate::jepa::JepaRollout;
use crate::linalg::Vec3;
use crate::mppi::{Controller, MppiConfig, Z_FLOOR};
use crate::params::QuadParamsInput;
use crate::rng::Rng;
use crate::se3::{FlatRef, Se3Control};
use crate::state::State;
use crate::CtbrCmd;

const M: usize = 24; // points per spline
const K: usize = 4; // number of candidate reference splines (a fan of approaches)

#[derive(Clone)]
pub struct SplineMppiConfig {
    pub base: MppiConfig, // samples/horizon/dt/lambda/beta/sigma/bounds/trust_lambda/v_max
    pub w_contour: f64,   // perpendicular-to-spline distance penalty
    pub w_vperp: f64,     // velocity-across-spline penalty
    pub w_vprog: f64,     // reward for speed ALONG the spline
    pub w_progress: f64,  // reward for reaching far along the spline
    pub w_floor: f64,
    pub gate_reward: f64,
}

impl SplineMppiConfig {
    pub fn for_mass(mass: f64) -> Self {
        let mut base = MppiConfig::for_mass(mass);
        base.horizon = 12;
        base.samples = 48;
        base.sigma_rate = 1.6;
        base.sigma_thrust = 0.12 * mass * 9.81;
        base.rate_max = 6.0;
        base.beta = 0.85;
        base.lambda = 0.4;
        base.v_max = 4.0;
        base.w_speed = 1.5;
        base.trust_lambda = 1000.0;
        SplineMppiConfig {
            base,
            w_contour: 6.0,
            w_vperp: 0.4,
            w_vprog: 1.2,
            w_progress: 8.0,
            w_floor: 600.0,
            gate_reward: 60.0,
        }
    }
}

/// A FAN of K cubic-Hermite splines from p0 (tangent v0) to the gate (tangent =
/// oriented gate normal). They differ in approach "tension" (how sharply they commit to
/// the gate normal), so there's no single hard-coded path — the planner tracks whichever
/// fits best. Returns flattened K*M points + unit tangents + per-point progress fraction.
fn build_splines(
    p0: Vec3<f64>, v0: Vec3<f64>, g: &Gate,
) -> (Vec<Vec3<f64>>, Vec<Vec3<f64>>, Vec<f64>) {
    let to_gate = g.center - p0;
    let l = to_gate.norm().max(0.5);
    let m0 = if v0.norm() > 0.5 { v0.scale(l / v0.norm()) } else { to_gate };
    let mut n = g.normal;
    if n.dot(to_gate) < 0.0 {
        n = n.scale(-1.0);
    }
    let nhat = n.scale(1.0 / n.norm().max(1e-6));
    // K end-tangent tensions: gentle (early commit to normal) -> aggressive (straight in)
    let tensions = [0.5, 0.9, 1.4, 2.0];
    let (mut pts, mut tan, mut prog) = (Vec::with_capacity(K * M), Vec::with_capacity(K * M), Vec::with_capacity(K * M));
    for kk in 0..K {
        let m1 = nhat.scale(l * tensions[kk.min(tensions.len() - 1)]);
        for i in 0..M {
            let s = i as f64 / (M - 1) as f64;
            let (s2, s3) = (s * s, s * s * s);
            let p = p0.scale(2.0 * s3 - 3.0 * s2 + 1.0)
                + m0.scale(s3 - 2.0 * s2 + s)
                + g.center.scale(-2.0 * s3 + 3.0 * s2)
                + m1.scale(s3 - s2);
            let d = p0.scale(6.0 * s2 - 6.0 * s)
                + m0.scale(3.0 * s2 - 4.0 * s + 1.0)
                + g.center.scale(-6.0 * s2 + 6.0 * s)
                + m1.scale(3.0 * s2 - 2.0 * s);
            pts.push(p);
            tan.push(d.scale(1.0 / d.norm().max(1e-6)));
            prog.push(s);
        }
    }
    (pts, tan, prog)
}

pub struct SplineMppiController {
    pub model: JepaRollout,
    pub cfg: SplineMppiConfig,
    se3: Se3Control, // geometric (policy-free) seed generator
    nominal: Vec<CtbrCmd<f64>>,
    rng: Rng,
}

impl SplineMppiController {
    pub fn new(model: JepaRollout, cfg: SplineMppiConfig, params: &QuadParamsInput, seed: u64) -> Self {
        let nominal = vec![
            CtbrCmd { thrust: cfg.base.hover_thrust, w_cmd: Vec3::zero() };
            cfg.base.horizon
        ];
        SplineMppiController { model, cfg, se3: Se3Control::new(params), nominal, rng: Rng::new(seed) }
    }

    pub fn reset_nominal(&mut self) {
        let hover = CtbrCmd { thrust: self.cfg.base.hover_thrust, w_cmd: Vec3::zero() };
        for n in self.nominal.iter_mut() {
            *n = hover;
        }
    }
    pub fn set_nominal_const(&mut self, a: CtbrCmd<f64>) {
        for n in self.nominal.iter_mut() {
            *n = a;
        }
    }

    /// The raw geometric seed (no model, no sampling) — the proposal the MPPI refines.
    /// Lets us isolate whether the world model HELPS or HURTS over the seed alone.
    pub fn seed_only(&self, state: &State<f64>, course: &Course) -> CtbrCmd<f64> {
        let gate = course.gates.get(course.next).cloned().unwrap_or_else(|| {
            Gate::new(Vec3::new(state.x.x, state.x.y, state.x.z.max(1.3)), Vec3::new(1.0, 0.0, 0.0), 1.0)
        });
        let (pts, tan, _) = build_splines(state.x, state.v, &gate);
        let ci = M + (M / 4);
        let fref = FlatRef {
            x: pts[ci], x_dot: tan[ci].scale(self.cfg.base.v_max),
            x_ddot: Vec3::zero(), yaw: 0.0, yaw_dot: 0.0,
        };
        let (u1, w) = self.se3.ctbr_seed(state, &fref, 5.0, self.cfg.base.rate_max);
        CtbrCmd { thrust: u1, w_cmd: w }
    }
    pub fn observe(&mut self, state: &State<f64>, action: &CtbrCmd<f64>) {
        self.model.observe_raw(state, [action.thrust, action.w_cmd.x, action.w_cmd.y, action.w_cmd.z]);
    }

    fn clamp(&self, c: CtbrCmd<f64>) -> CtbrCmd<f64> {
        let b = &self.cfg.base;
        CtbrCmd {
            thrust: c.thrust.clamp(b.thrust_min, b.thrust_max),
            w_cmd: Vec3::new(
                c.w_cmd.x.clamp(-b.rate_max, b.rate_max),
                c.w_cmd.y.clamp(-b.rate_max, b.rate_max),
                c.w_cmd.z.clamp(-b.rate_max, b.rate_max),
            ),
        }
    }

}

impl Controller for SplineMppiController {
    fn act(&mut self, state: &State<f64>, course: &Course) -> CtbrCmd<f64> {
        let cfg = &self.cfg;
        let b = &cfg.base;
        let t = b.horizon;
        let s = b.samples;
        let dt = b.dt;
        let inv_dt = 1.0 / dt;

        // reference spline to the next gate (fall back to a hold point if finished)
        let gate = course.gates.get(course.next).cloned().unwrap_or_else(|| {
            Gate::new(Vec3::new(state.x.x, state.x.y, state.x.z.max(1.3)),
                      Vec3::new(1.0, 0.0, 0.0), 1.0)
        });
        let (pts, tan, prog) = build_splines(state.x, state.v, &gate);
        let np = pts.len();

        // SEED the search with a reliable, policy-free geometric proposal: a SE3
        // controller tracking a look-ahead carrot on the (medium-tension) spline gives
        // an in-distribution nominal, so the model only has to RANK locally around it.
        let rate_max = self.cfg.base.rate_max;
        let v_target = self.cfg.base.v_max;
        let ci = M + (M / 4); // ~25% along spline #1 (medium tension)
        let fref = FlatRef {
            x: pts[ci],
            x_dot: tan[ci].scale(v_target),
            x_ddot: Vec3::zero(),
            yaw: 0.0,
            yaw_dot: 0.0,
        };
        let (u1, w) = self.se3.ctbr_seed(state, &fref, 5.0, rate_max);
        let seed = CtbrCmd { thrust: u1, w_cmd: w };
        for n in self.nominal.iter_mut() {
            *n = seed;
        }

        // sample candidate CTBR plans (AR(1) around the warm-started nominal)
        let smear = (1.0 - b.beta * b.beta).sqrt();
        let mut seqs: Vec<Vec<CtbrCmd<f64>>> = Vec::with_capacity(s);
        for _ in 0..s {
            let mut e = [0.0f64; 4];
            let mut seq = Vec::with_capacity(t);
            for k in 0..t {
                for d in 0..4 {
                    e[d] = b.beta * e[d] + smear * self.rng.normal();
                }
                let base = self.nominal[k];
                seq.push(self.clamp(CtbrCmd {
                    thrust: base.thrust + b.sigma_thrust * e[0],
                    w_cmd: Vec3::new(base.w_cmd.x + b.sigma_rate * e[1],
                                     base.w_cmd.y + b.sigma_rate * e[2],
                                     base.w_cmd.z + b.sigma_rate * e[3]),
                }));
            }
            seqs.push(seq);
        }

        // model rollout over the candidate plans (raw [thrust,wx,wy,wz] actions)
        let raw: Vec<Vec<[f64; 4]>> = seqs.iter()
            .map(|sq| sq.iter().map(|a| [a.thrust, a.w_cmd.x, a.w_cmd.y, a.w_cmd.z]).collect())
            .collect();
        let (trajs, trust) = self.model.rollout_raw_trust(state, &raw);

        let mut costs = vec![0.0f64; s];
        for i in 0..s {
            let traj = &trajs[i];
            let mut cost = 0.0;
            let mut prev = state.x;
            let mut idx = course.next; // for gate-pass detection
            let mut max_prog = 0.0f64;
            for &p in traj.iter() {
                let v = (p - prev).scale(inv_dt);
                // nearest vertex over the WHOLE fan (auto-picks the best-fitting spline)
                let mut bj = 0usize;
                let mut bd = f64::INFINITY;
                for j in 0..np {
                    let d = (p - pts[j]).dot(p - pts[j]);
                    if d < bd { bd = d; bj = j; }
                }
                max_prog = max_prog.max(prog[bj]);
                let th = tan[bj];
                let vlong = v.dot(th);
                let vperp = v - th.scale(vlong);
                cost += cfg.w_contour * bd;                 // perpendicular distance²
                cost += cfg.w_vperp * vperp.dot(vperp);     // velocity across the spline
                cost -= cfg.w_vprog * vlong.max(0.0);       // reward forward speed
                let sp = v.norm();
                if sp > b.v_max { cost += b.w_speed * (sp - b.v_max) * (sp - b.v_max); }
                if p.z < Z_FLOOR { let d = Z_FLOOR - p.z; cost += cfg.w_floor * d * d; }
                if idx < course.gates.len() && course.gates[idx].crossed(prev, p) {
                    cost -= cfg.gate_reward;
                    idx += 1;
                }
                prev = p;
            }
            cost -= cfg.w_progress * max_prog;              // how far along the spline it gets
            cost += b.trust_lambda * trust[i];
            costs[i] = if cost.is_finite() { cost } else { 1e12 };
        }

        // softmax-weighted update of the nominal
        let min_c = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut wsum = 0.0;
        let mut w = vec![0.0f64; s];
        for i in 0..s {
            w[i] = (-(costs[i] - min_c) / b.lambda).exp();
            wsum += w[i];
        }
        let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
        for k in 0..t {
            let (mut th, mut wx, mut wy, mut wz) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..s {
                let ww = w[i] * inv;
                let c = seqs[i][k];
                th += ww * c.thrust; wx += ww * c.w_cmd.x; wy += ww * c.w_cmd.y; wz += ww * c.w_cmd.z;
            }
            self.nominal[k] = self.clamp(CtbrCmd { thrust: th, w_cmd: Vec3::new(wx, wy, wz) });
        }
        let action = self.nominal[0];
        for k in 0..t - 1 {
            self.nominal[k] = self.nominal[k + 1];
        }
        self.nominal[t - 1] = CtbrCmd { thrust: b.hover_thrust, w_cmd: Vec3::zero() };
        action
    }
}
