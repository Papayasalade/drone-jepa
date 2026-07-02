//! Rotor-force MPPI for the rotor-force JEPA drone (the 4th racer).
//!
//! Unlike [`crate::mppi::MppiController`] (which plans in CTBR — collective thrust +
//! body rates, leaning on an inner rate loop), this planner samples **four
//! independent rotor forces** per step and feeds them to a rotor-force JEPA world
//! model. That's the harder control mode (no inner loop tames attitude), and the
//! interesting comparison: it stresses whether the rotor-force model is accurate in
//! the independent-per-rotor directions its planner explores.
//!
//! It reuses the exact position cost ([`crate::mppi::rollout_cost`]) and the SIGReg
//! latent-trust region, so the only difference vs the CTBR racer is the action space.

use crate::gates::Course;
use crate::jepa::JepaRollout;
use crate::mppi::{rollout_cost, MppiConfig};
use rotor_rs::params::NUM_ROTORS;
use rotor_rs::rng::Rng;
use rotor_rs::state::State;

/// Config for the rotor-force planner. Embeds an [`MppiConfig`] for the shared cost
/// + planning scalars (samples / horizon / dt / lambda / beta / all cost weights),
/// and adds the rotor-force action-space parameters.
#[derive(Clone)]
pub struct RotorMppiConfig {
    pub base: MppiConfig,
    pub hover_force: f64, // per-rotor force at hover [N] (= mass*g/4)
    pub f_min: f64,       // per-rotor force lower clamp [N]
    pub f_max: f64,       // per-rotor force upper clamp [N]
    pub sigma_force: f64, // per-rotor exploration std [N]
    pub w_omega: f64,     // angular-rate damping: penalize predicted |ω|² (anti-shake)
    pub w_tilt: f64,      // anti-FLIP: penalize predicted tilt past ~53° (R_zz < 0.6)
}

impl RotorMppiConfig {
    pub fn for_mass(mass: f64) -> Self {
        let hover = mass * 9.81;
        let hover_force = hover / NUM_ROTORS as f64;
        let mut base = MppiConfig::for_mass(mass);
        base.beta = 0.85;
        RotorMppiConfig {
            base,
            hover_force,
            f_min: 0.0,
            f_max: 4.0 * hover_force, // TWR up to ~4 (matches CTBR's 4*hover total)
            sigma_force: 0.3 * hover_force,
            // Angular-rate damping (anti-shake). OFF by default: an IN-DISTRIBUTION
            // rotor model predicts the attitude dynamics well enough that MPPI flies
            // it cleanly (4.8 gates/race on the narrow fleet) and any damping just
            // stops it from tilting enough to race. The "violent shaking" was a
            // distribution mismatch (a wide/OOD model), fixed by retraining narrow —
            // not by damping. Kept env-tunable for an OOD model that needs propping up.
            w_omega: std::env::var("W_OMEGA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            // anti-flip: penalize predicted attitude past ~53° tilt. Unlike w_omega
            // (which penalizes RATE and kills the rates racing needs), this penalizes
            // the OUTCOME (being inverted), so moderate racing tilts stay free.
            w_tilt: std::env::var("W_TILT").ok().and_then(|v| v.parse().ok()).unwrap_or(12.0),
        }
    }
}

pub struct RotorMppiController {
    pub model: JepaRollout,
    pub cfg: RotorMppiConfig,
    nominal: Vec<[f64; NUM_ROTORS]>,
    rng: Rng,
}

impl RotorMppiController {
    pub fn new(model: JepaRollout, cfg: RotorMppiConfig, seed: u64) -> Self {
        let nominal = vec![[cfg.hover_force; NUM_ROTORS]; cfg.base.horizon];
        RotorMppiController { model, cfg, nominal, rng: Rng::new(seed) }
    }

    /// Reset the warm-started nominal sequence to per-rotor hover. Call on respawn so
    /// a stale diving plan isn't re-flown.
    pub fn reset_nominal(&mut self) {
        for n in self.nominal.iter_mut() {
            *n = [self.cfg.hover_force; NUM_ROTORS];
        }
    }

    pub fn observe(&mut self, state: &State<f64>, action: [f64; NUM_ROTORS]) {
        self.model.observe_raw(state, action);
    }

    fn clamp_force(&self, f: [f64; NUM_ROTORS]) -> [f64; NUM_ROTORS] {
        core::array::from_fn(|j| f[j].clamp(self.cfg.f_min, self.cfg.f_max))
    }

    pub fn act(&mut self, state: &State<f64>, course: &Course) -> [f64; NUM_ROTORS] {
        let cfg = &self.cfg;
        let base = &cfg.base;
        let t = base.horizon;
        let s = base.samples;

        // sample S candidate rotor-force sequences with AR(1) (temporally-correlated)
        // INDEPENDENT per-rotor noise — exactly the directions a rotor-force planner
        // explores that a CTBR controller's coordinated forces never visit.
        let smear = (1.0 - base.beta * base.beta).sqrt();
        let mut seqs: Vec<Vec<[f64; NUM_ROTORS]>> = Vec::with_capacity(s);
        for _ in 0..s {
            let mut e = [0.0f64; NUM_ROTORS];
            let mut seq = Vec::with_capacity(t);
            for k in 0..t {
                let base_k = self.nominal[k];
                let mut f = [0.0f64; NUM_ROTORS];
                for d in 0..NUM_ROTORS {
                    e[d] = base.beta * e[d] + smear * self.rng.normal();
                    f[d] = base_k[d] + cfg.sigma_force * e[d];
                }
                seq.push(self.clamp_force(f));
            }
            seqs.push(seq);
        }

        // roll out all candidates over the rotor-force JEPA + latent-trust energy +
        // predicted angular rate (for the anti-shake damping term)
        let (trajs, oms, uprs, trust) = self.model.rollout_raw_trust_om(state, &seqs);
        let mut costs = vec![0.0f64; s];
        for i in 0..s {
            // action effort = per-rotor deviation from hover (analog of body-rate²)
            let eff: Vec<f64> = seqs[i]
                .iter()
                .map(|f| {
                    f.iter()
                        .map(|&fj| {
                            let d = fj - cfg.hover_force;
                            d * d
                        })
                        .sum::<f64>()
                })
                .collect();
            let mut c = rollout_cost(state.x, &trajs[i], &eff, course, base);
            c += base.trust_lambda * trust[i];
            // angular-rate damping (off by default) + anti-FLIP attitude penalty:
            // penalize predicted tilt past ~53° (R_zz < 0.6) quadratically, so the
            // planner stops choosing plans that roll the drone over while still
            // allowing the moderate tilts needed to accelerate toward gates.
            c += cfg.w_omega * oms[i].iter().map(|o| o * o).sum::<f64>();
            c += cfg.w_tilt
                * uprs[i].iter().map(|&u| { let d = (0.6 - u).max(0.0); d * d }).sum::<f64>();
            costs[i] = if c.is_finite() { c } else { 1e12 };
        }

        // softmax weights over -cost/lambda
        let min_c = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut wsum = 0.0;
        let mut weights = vec![0.0f64; s];
        for i in 0..s {
            let w = (-(costs[i] - min_c) / base.lambda).exp();
            weights[i] = w;
            wsum += w;
        }
        let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };

        // weighted update of the nominal sequence
        for k in 0..t {
            let mut acc = [0.0f64; NUM_ROTORS];
            for i in 0..s {
                let w = weights[i] * inv;
                for d in 0..NUM_ROTORS {
                    acc[d] += w * seqs[i][k][d];
                }
            }
            self.nominal[k] = self.clamp_force(acc);
        }

        let action = self.nominal[0];
        // warm-start shift: drop the executed step, repeat hover at the tail.
        for k in 0..t - 1 {
            self.nominal[k] = self.nominal[k + 1];
        }
        self.nominal[t - 1] = [cfg.hover_force; NUM_ROTORS];

        action
    }
}
