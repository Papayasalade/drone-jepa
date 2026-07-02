//! Gate-racing control: a CTBR-action MPPI planner, structured so the rollout
//! model and the controller are both swappable.
//!
//! - [`RolloutModel`]: predict a position trajectory for a CTBR action sequence.
//!   `TrueDynamics` (the rotor-rs sim) implements it now → classical MPPI; the
//!   Candle JEPA model will implement it later → JEPA-MPPI (same planner code).
//! - [`Controller`]: `state + course -> CTBR action`. `MppiController` now; an
//!   `RlPolicy` can implement it later.
//!
//! We plan in **CTBR** (collective thrust + body rates), not raw rotor forces:
//! RotorPy's inner P rate-loop tames attitude, which made CTBR-MPPI track ~3x
//! better than rotor-force MPPI — doubly important for aggressive gate racing.

use crate::control::{Ctbr, CtbrCmd};
use crate::gates::{Course, Gate};
use crate::linalg::Vec3;
use crate::multirotor::Multirotor;
use crate::params::QuadParamsInput;
use crate::rng::Rng;
use crate::state::State;

/// Predicts where the drone goes for a candidate CTBR action sequence.
pub trait RolloutModel {
    /// Roll out `actions` from `init`, writing the predicted position per step into
    /// `out` (cleared first; `out.len() == actions.len()` after).
    fn rollout(&self, init: &State<f64>, actions: &[CtbrCmd<f64>], dt: f64, out: &mut Vec<Vec3<f64>>);

    /// Roll out many candidate sequences at once. Default loops `rollout`; learned
    /// models (e.g. the JEPA net) override this to batch all candidates in one pass.
    fn rollout_batch(
        &self,
        init: &State<f64>,
        seqs: &[Vec<CtbrCmd<f64>>],
        dt: f64,
    ) -> Vec<Vec<Vec3<f64>>> {
        let mut tmp = Vec::new();
        seqs
            .iter()
            .map(|seq| {
                self.rollout(init, seq, dt, &mut tmp);
                core::mem::take(&mut tmp)
            })
            .collect()
    }

    /// Roll out candidates AND return a per-candidate trust penalty. For learned
    /// models this is the predicted-latent energy above the SIGReg manifold — large
    /// when a plan exploits the model. Default: zeros (memoryless models have no
    /// trust signal). The controller adds `trust_lambda · penalty` to each cost.
    fn rollout_batch_trust(
        &self,
        init: &State<f64>,
        seqs: &[Vec<CtbrCmd<f64>>],
        dt: f64,
    ) -> (Vec<Vec<Vec3<f64>>>, Vec<f64>) {
        let trajs = self.rollout_batch(init, seqs, dt);
        let z = vec![0.0; trajs.len()];
        (trajs, z)
    }

    /// Record a real (executed) step, for models that need state/action history
    /// (the JEPA encoder). No-op for memoryless models like `TrueDynamics`.
    fn observe(&mut self, _state: &State<f64>, _action: &CtbrCmd<f64>) {}
}

/// Anything that turns (state, course) into a CTBR command.
pub trait Controller {
    fn act(&mut self, state: &State<f64>, course: &Course) -> CtbrCmd<f64>;
}

/// Ground-truth rollout model: the rotor-rs CTBR dynamics.
pub struct TrueDynamics {
    pub veh: Multirotor<f64, Ctbr>,
}

impl TrueDynamics {
    pub fn new(params: &QuadParamsInput, n_sub: usize) -> Self {
        TrueDynamics { veh: Multirotor::with_substeps(params, n_sub) }
    }
}

impl RolloutModel for TrueDynamics {
    fn rollout(&self, init: &State<f64>, actions: &[CtbrCmd<f64>], dt: f64, out: &mut Vec<Vec3<f64>>) {
        out.clear();
        let mut s = *init;
        for a in actions {
            s = self.veh.step(&s, a, dt);
            out.push(s.x);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MppiConfig {
    pub samples: usize,
    pub horizon: usize,
    pub dt: f64,
    pub lambda: f64,        // softmax temperature
    pub hover_thrust: f64,  // nominal collective thrust [N] (= mass*g)
    pub thrust_min: f64,
    pub thrust_max: f64,
    pub sigma_thrust: f64,  // exploration std on thrust [N]
    pub sigma_rate: f64,    // exploration std on body rates [rad/s]
    pub rate_max: f64,      // clamp on |body rate| [rad/s]
    pub w_vel: f64,         // reward for speed TOWARD the next gate (drives flight)
    pub w_pos: f64,         // per-step distance-to-gate cost (stable attractor: race + settle)
    pub w_alt: f64,         // penalty on altitude error vs the gate
    pub w_terminal: f64,    // terminal distance-to-gate weight
    pub gate_reward: f64,   // reward for passing a gate during a rollout
    pub w_effort: f64,      // penalty on body-rate magnitude
    pub beta: f64,          // AR(1) noise smoothing in [0,1) (temporal correlation)
    pub v_max: f64,         // soft speed cap [m/s] (INFINITY = no cap)
    pub w_speed: f64,       // penalty weight for exceeding v_max (keeps flight slow)
    pub w_vdamp: f64,       // always-on velocity² penalty (caps racing speed + enables hover)
    pub w_floor: f64,       // penalty for a rollout dipping below Z_FLOOR (ground avoidance)
    pub trust_lambda: f64,  // SIGReg latent-trust-region weight (anti-model-exploitation)
}

/// Altitude below which the floor penalty kicks in [m]. The sim has no ground
/// collision, so without this nothing stops a planner from flying into the floor.
pub const Z_FLOOR: f64 = 0.4;

impl MppiConfig {
    /// Reasonable defaults for a ~0.5 kg quad racing 20 Hz.
    pub fn for_mass(mass: f64) -> Self {
        let hover = mass * 9.81;
        MppiConfig {
            samples: 256,
            horizon: 25,
            dt: 0.05,
            lambda: 0.5,
            hover_thrust: hover,
            thrust_min: 0.0,
            thrust_max: 4.0 * hover,
            sigma_thrust: 0.2 * hover,
            sigma_rate: 4.0,
            rate_max: 12.0,
            w_vel: 2.0,
            w_alt: 1.5,
            w_terminal: 1.0,
            gate_reward: 80.0,
            w_effort: 0.005,
            beta: 0.85,
            v_max: f64::INFINITY, // no speed cap by default
            w_speed: 0.0,
            w_vdamp: 0.0, // off by default (keep existing racers unchanged)
            w_pos: 0.0,
            w_floor: 0.0, // off by default
            trust_lambda: 0.0, // off by default (memoryless models have no latents)
        }
    }
}

pub struct MppiController<M: RolloutModel> {
    pub model: M,
    pub cfg: MppiConfig,
    nominal: Vec<CtbrCmd<f64>>,
    rng: Rng,
}

impl<M: RolloutModel> MppiController<M> {
    pub fn new(model: M, cfg: MppiConfig, seed: u64) -> Self {
        let nominal = vec![
            CtbrCmd { thrust: cfg.hover_thrust, w_cmd: Vec3::zero() };
            cfg.horizon
        ];
        MppiController { model, cfg, nominal, rng: Rng::new(seed) }
    }

    /// Forward an executed (state, action) to the rollout model's history
    /// (needed by learned models like JEPA; no-op for `TrueDynamics`).
    pub fn observe(&mut self, state: &State<f64>, action: &CtbrCmd<f64>) {
        self.model.observe(state, action);
    }

    /// Reset the warm-started nominal action sequence back to neutral hover. Call
    /// after a crash/respawn so the planner doesn't inherit the diving plan that
    /// caused the crash (otherwise it dives again → a crash loop).
    pub fn reset_nominal(&mut self) {
        let hover = CtbrCmd { thrust: self.cfg.hover_thrust, w_cmd: Vec3::zero() };
        for n in self.nominal.iter_mut() {
            *n = hover;
        }
    }

    /// Re-center the candidate search on a given action (policy-guided MPC): set the
    /// whole nominal sequence to `a`, so MPPI samples plans AROUND a known-good action
    /// (e.g. an RL policy's) rather than around hover — the model then only has to RANK
    /// among in-distribution plans, where its predictions are far more reliable.
    pub fn set_nominal_const(&mut self, a: CtbrCmd<f64>) {
        for n in self.nominal.iter_mut() {
            *n = a;
        }
    }

    #[inline]
    fn clamp_cmd(&self, c: CtbrCmd<f64>) -> CtbrCmd<f64> {
        let r = self.cfg.rate_max;
        CtbrCmd {
            thrust: c.thrust.clamp(self.cfg.thrust_min, self.cfg.thrust_max),
            w_cmd: Vec3::new(
                c.w_cmd.x.clamp(-r, r),
                c.w_cmd.y.clamp(-r, r),
                c.w_cmd.z.clamp(-r, r),
            ),
        }
    }
}

/// Cost of a predicted position trajectory wrt the course (lower is better).
/// `effort_sq[k]` is the action-effort magnitude² at step k (body-rate² for CTBR,
/// rotor-force deviation² for rotor-force) — kept action-agnostic so both the CTBR
/// and rotor-force planners share this exact cost.
pub(crate) fn rollout_cost(
    init_pos: Vec3<f64>,
    traj: &[Vec3<f64>],
    effort_sq: &[f64],
    course: &Course,
    cfg: &MppiConfig,
) -> f64 {
    let gates = &course.gates;
    let n = gates.len();
    let mut idx = course.next;
    let mut prev = init_pos;
    let mut cost = 0.0;
    let inv_dt = 1.0 / cfg.dt;

    let target = |i: usize| -> Option<&Gate> {
        if i < n {
            Some(&gates[i])
        } else if course.loop_course && n > 0 {
            Some(&gates[i % n])
        } else {
            None
        }
    };

    for (k, &p) in traj.iter().enumerate() {
        let vel = (p - prev).scale(inv_dt);
        // always-on velocity damping: with the velocity-toward-gate reward it caps
        // racing speed (~w_vel/2w_vdamp) AND, where there's no gate progress to make
        // (a hover point), it actively brings the drone to rest with the SAME cost.
        cost += cfg.w_vdamp * vel.dot(vel);
        // ground avoidance: the sim has no floor collision, so without this the
        // planner has no reason not to fly into (or through) the ground. A steep
        // quadratic below Z_FLOOR makes any low-altitude rollout very expensive.
        if p.z < Z_FLOOR {
            let d = Z_FLOOR - p.z;
            cost += cfg.w_floor * d * d;
        }
        if let Some(g) = target(idx) {
            // reward velocity TOWARD the gate (strong, always-achievable gradient)
            let to_gate = g.center - prev;
            let dist = to_gate.norm();
            // position attractor, QUADRATIC = a spring: its gradient (∝ dist)
            // vanishes AT the target, so with damping the drone settles to REST.
            // (A linear `dist` term has constant gradient → the planner forever
            // trades damping for closing distance → a limit cycle / orbit.)
            cost += cfg.w_pos * dist * dist;
            if dist > 1e-6 {
                let dir = to_gate.scale(1.0 / dist);
                cost -= cfg.w_vel * vel.dot(dir);
                // soft speed cap: keep flight slow (and, for a learned model, in-distribution)
                let sp = vel.norm();
                if sp > cfg.v_max {
                    cost += cfg.w_speed * (sp - cfg.v_max) * (sp - cfg.v_max);
                }
            }
            // hold the gate's altitude
            let dz = p.z - g.center.z;
            cost += cfg.w_alt * dz * dz;
            if g.crossed(prev, p) {
                cost -= cfg.gate_reward;
                idx += 1;
            }
        }
        cost += cfg.w_effort * effort_sq[k];
        prev = p;
    }
    // terminal: pull the rollout's end toward the (possibly advanced) target gate.
    if let (Some(end), Some(g)) = (traj.last(), target(idx)) {
        cost += cfg.w_terminal * (*end - g.center).norm();
    }
    cost
}

impl<M: RolloutModel> Controller for MppiController<M> {
    fn act(&mut self, state: &State<f64>, course: &Course) -> CtbrCmd<f64> {
        let cfg = self.cfg;
        let t = cfg.horizon;
        let s = cfg.samples;

        // sample S candidate CTBR sequences with AR(1) (temporally-correlated)
        // noise so rollouts include SUSTAINED tilt — white noise can't translate.
        let smear = (1.0 - cfg.beta * cfg.beta).sqrt();
        let mut seqs: Vec<Vec<CtbrCmd<f64>>> = Vec::with_capacity(s);
        for _ in 0..s {
            let mut e = [0.0f64; 4];
            let mut seq = Vec::with_capacity(t);
            for k in 0..t {
                for d in 0..4 {
                    e[d] = cfg.beta * e[d] + smear * self.rng.normal();
                }
                let base = self.nominal[k];
                seq.push(self.clamp_cmd(CtbrCmd {
                    thrust: base.thrust + cfg.sigma_thrust * e[0],
                    w_cmd: Vec3::new(
                        base.w_cmd.x + cfg.sigma_rate * e[1],
                        base.w_cmd.y + cfg.sigma_rate * e[2],
                        base.w_cmd.z + cfg.sigma_rate * e[3],
                    ),
                }));
            }
            seqs.push(seq);
        }

        // roll out all candidates (batched for learned models) and cost them.
        // `trust` is the per-candidate latent-trust penalty (0 for memoryless models).
        let (trajs, trust) = self.model.rollout_batch_trust(state, &seqs, cfg.dt);
        let mut costs = vec![0.0f64; s];
        for i in 0..s {
            let eff: Vec<f64> = seqs[i].iter().map(|a| a.w_cmd.dot(a.w_cmd)).collect();
            let mut c = rollout_cost(state.x, &trajs[i], &eff, course, &cfg);
            // SIGReg latent trust region: penalize plans that drive the predicted
            // latents off the training manifold (model exploitation).
            c += cfg.trust_lambda * trust[i];
            // Aggressive samples can blow up; a NaN/inf cost must NOT poison the
            // softmax, so map it to a large finite penalty.
            costs[i] = if c.is_finite() { c } else { 1e12 };
        }

        // softmax weights over -cost/lambda
        let min_c = costs.iter().cloned().fold(f64::INFINITY, f64::min);

        #[cfg(feature = "mppi_debug")]
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static N: AtomicUsize = AtomicUsize::new(0);
            let c = N.fetch_add(1, Ordering::Relaxed);
            if c < 3 {
                let argmin = costs.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
                let b = seqs[argmin][0]; // best sample's first action
                let nfin = costs.iter().filter(|c| c.is_finite() && **c < 1e11).count();
                eprintln!(
                    "[mppi#{c}] best_cost={min_c:.2} finite={nfin}/{s}  best_a0(th={:.2}, w=[{:+.2},{:+.2},{:+.2}])",
                    b.thrust, b.w_cmd.x, b.w_cmd.y, b.w_cmd.z
                );
            }
        }
        let mut wsum = 0.0;
        let mut weights = vec![0.0f64; s];
        for i in 0..s {
            let w = (-(costs[i] - min_c) / cfg.lambda).exp();
            weights[i] = w;
            wsum += w;
        }
        let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };

        // weighted update of the nominal sequence
        for k in 0..t {
            let mut th = 0.0;
            let mut wx = 0.0;
            let mut wy = 0.0;
            let mut wz = 0.0;
            for i in 0..s {
                let w = weights[i] * inv;
                let c = seqs[i][k];
                th += w * c.thrust;
                wx += w * c.w_cmd.x;
                wy += w * c.w_cmd.y;
                wz += w * c.w_cmd.z;
            }
            self.nominal[k] = self.clamp_cmd(CtbrCmd { thrust: th, w_cmd: Vec3::new(wx, wy, wz) });
        }

        let action = self.nominal[0];
        // warm-start shift: drop the executed step, repeat the last.
        for k in 0..t - 1 {
            self.nominal[k] = self.nominal[k + 1];
        }
        self.nominal[t - 1] = CtbrCmd { thrust: cfg.hover_thrust, w_cmd: Vec3::zero() };

        action
    }
}
