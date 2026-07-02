//! Adapts the Candle SkyJepa model to rotor-rs's `RolloutModel` trait, so it
//! can drive the MPPI gate racer as the *learned* world model (JEPA-MPPI).
//!
//! The model needs a state+action *history* (the TCN encoders), which the
//! memoryless trait doesn't carry — so `JepaRollout` keeps a rolling buffer,
//! updated each real control step via `observe`.

use std::collections::VecDeque;

use rotor_rs::mppi::RolloutModel;
use rotor_rs::{CtbrCmd, State, Vec3};

use crate::{SkyJepa, AD, SD};

/// State (quaternion form) -> SkyJEPA's 18-dim [pos3, vel3, R(row-major 9), omega3].
fn state_to_18(s: &State<f64>) -> [f64; SD] {
    let r = s.q.to_rotmat(); // body->world, row-major rows
    [
        s.x.x, s.x.y, s.x.z,
        s.v.x, s.v.y, s.v.z,
        r.rows[0][0], r.rows[0][1], r.rows[0][2],
        r.rows[1][0], r.rows[1][1], r.rows[1][2],
        r.rows[2][0], r.rows[2][1], r.rows[2][2],
        s.w.x, s.w.y, s.w.z,
    ]
}

fn ctbr_to_4(a: &CtbrCmd<f64>) -> [f64; AD] {
    [a.thrust, a.w_cmd.x, a.w_cmd.y, a.w_cmd.z]
}

pub struct JepaRollout {
    model: SkyJepa,
    h: usize,
    t: usize,
    states: VecDeque<[f64; SD]>,  // most-recent observed states (<= H-1 kept)
    actions: VecDeque<[f64; AD]>, // most-recent observed actions (<= H-1 kept)
}

impl JepaRollout {
    pub fn new(model: SkyJepa) -> Self {
        let (h, t) = (model.config().history, model.config().horizon);
        JepaRollout { model, h, t, states: VecDeque::new(), actions: VecDeque::new() }
    }

    /// Build the H-length state history ending at `init`, left-padded if short.
    fn build_history(&self, init: &State<f64>) -> Vec<[f64; SD]> {
        let cur = state_to_18(init);
        let mut hist: Vec<[f64; SD]> = self.states.iter().copied().collect();
        hist.push(cur); // history ends at the current state
        while hist.len() < self.h {
            hist.insert(0, hist[0]); // left-pad by repeating the oldest
        }
        let start = hist.len() - self.h;
        hist[start..].to_vec()
    }

    /// Past-action prefix of length H-1, left-padded with hover-ish zeros if short.
    fn past_actions(&self) -> Vec<[f64; AD]> {
        let mut pa: Vec<[f64; AD]> = self.actions.iter().copied().collect();
        while pa.len() < self.h - 1 {
            pa.insert(0, *pa.first().unwrap_or(&[0.0; AD]));
        }
        let start = pa.len() - (self.h - 1);
        pa[start..].to_vec()
    }
}

impl RolloutModel for JepaRollout {
    fn rollout(&self, init: &State<f64>, actions: &[CtbrCmd<f64>], dt: f64, out: &mut Vec<Vec3<f64>>) {
        let trajs = self.rollout_batch(init, std::slice::from_ref(&actions.to_vec()), dt);
        *out = trajs.into_iter().next().unwrap_or_default();
    }

    fn rollout_batch(
        &self,
        init: &State<f64>,
        seqs: &[Vec<CtbrCmd<f64>>],
        _dt: f64,
    ) -> Vec<Vec<Vec3<f64>>> {
        let b = seqs.len();
        let hist = self.build_history(init); // (H,18), same for all candidates
        let past = self.past_actions(); // (H-1,4)

        // assemble per-candidate action windows: past (H-1) + candidate (T) + tail (1).
        let mut sh = Vec::with_capacity(b);
        let mut aw = Vec::with_capacity(b);
        for seq in seqs {
            sh.push(hist.clone());
            let mut w: Vec<[f64; AD]> = Vec::with_capacity(self.h + self.t);
            w.extend_from_slice(&past);
            for a in seq.iter().take(self.t) {
                w.push(ctbr_to_4(a));
            }
            let last = w.last().copied().unwrap_or([0.0; AD]);
            while w.len() < self.h + self.t {
                w.push(last);
            }
            aw.push(w);
        }

        match self.model.predict_batch(&sh, &aw) {
            Ok(preds) => preds
                .into_iter()
                .map(|traj| traj.into_iter().map(|x| Vec3::new(x[0], x[1], x[2])).collect())
                .collect(),
            // on failure, return empty trajectories -> MPPI penalizes them
            Err(_) => vec![Vec::new(); b],
        }
    }

    fn observe(&mut self, state: &State<f64>, action: &CtbrCmd<f64>) {
        self.states.push_back(state_to_18(state));
        while self.states.len() > self.h - 1 {
            self.states.pop_front();
        }
        self.actions.push_back(ctbr_to_4(action));
        while self.actions.len() > self.h - 1 {
            self.actions.pop_front();
        }
    }
}
