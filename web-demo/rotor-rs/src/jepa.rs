//! Zero-dependency inference of the SkyJEPA world model, so the *learned* model
//! can drive the MPPI gate racer in the browser (WASM) — no Candle, no PyTorch.
//!
//! The neural parts (two TCN encoders, a GRU predictor, a 3-layer MLP prober) are
//! hand-rolled in f32 to mirror `jepa-rs`'s Candle port op-for-op (validated in
//! `jepa-rs/tests/parity.rs`). The parameter-free DKI integrator (semi-implicit
//! Euler + SO(3) exp) runs in f64. Weights are loaded from the self-describing
//! `.jblob` produced by `scripts/export_jepa_blob.py`.
//!
//! State is 18-dim [pos3, vel3, R(row-major 9), omega3]; CTBR action is 4-dim
//! [thrust, wx, wy, wz].

use std::collections::HashMap;
use std::collections::VecDeque;

use crate::linalg::Vec3;
use crate::mppi::RolloutModel;
use crate::{CtbrCmd, State};

pub const SD: usize = 18;
pub const AD: usize = 4;

/// A flat tensor: row-major `data` with `shape`.
#[derive(Clone)]
struct Arr {
    shape: Vec<usize>,
    data: Vec<f32>,
}
impl Arr {
    fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }
}

// --------------------------------------------------------------------------- //
// math primitives
// --------------------------------------------------------------------------- //

/// erf via the Numerical-Recipes rational approximation (|err| < 1.2e-7).
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
    let erfc = ans; // = erfc(|x|)
    (if x >= 0.0 { 1.0 - erfc } else { erfc - 1.0 }) as f32
}

#[inline]
fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// --------------------------------------------------------------------------- //
// the model
// --------------------------------------------------------------------------- //

pub struct SkyJepaLite {
    w: HashMap<String, Arr>,
    pub action_mode_ctbr: bool,
    pub h: usize, // history
    pub t: usize, // horizon
    dt: f64,
    m: f64,
    g: f64,
    state_mean: [f64; SD],
    state_std: [f64; SD],
    action_mean: [f64; AD],
    action_std: [f64; AD],
    relative_pos: bool, // true = position relative to decision-time pos; false = zeroed
}

impl SkyJepaLite {
    /// Parse a `.jblob` (see `scripts/export_jepa_blob.py`). Panics on a malformed
    /// blob — it's an embedded build asset, so any failure is a build-time bug.
    pub fn from_blob(b: &[u8]) -> Self {
        let rd_u16 = |b: &[u8], p: &mut usize| {
            let v = u16::from_le_bytes([b[*p], b[*p + 1]]);
            *p += 2;
            v as usize
        };
        let rd_u32 = |b: &[u8], p: &mut usize| {
            let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
            *p += 4;
            v as usize
        };
        let rd_u8 = |b: &[u8], p: &mut usize| {
            let v = b[*p];
            *p += 1;
            v as usize
        };
        let rd_f64 = |b: &[u8], p: &mut usize| {
            let mut a = [0u8; 8];
            a.copy_from_slice(&b[*p..*p + 8]);
            *p += 8;
            f64::from_le_bytes(a)
        };
        let rd_f32 = |b: &[u8], p: &mut usize| {
            let mut a = [0u8; 4];
            a.copy_from_slice(&b[*p..*p + 4]);
            *p += 4;
            f32::from_le_bytes(a)
        };
        let rd_str = |b: &[u8], p: &mut usize, n: usize| {
            let s = String::from_utf8(b[*p..*p + n].to_vec()).unwrap();
            *p += n;
            s
        };

        assert_eq!(&b[0..7], b"JBLOB1\n", "bad jblob magic");
        let mut p = 7usize;
        let amn = rd_u16(b, &mut p);
        let action_mode = rd_str(b, &mut p, amn);
        let pmn = rd_u16(b, &mut p);
        let pos_mode = rd_str(b, &mut p, pmn);
        let h = rd_u32(b, &mut p);
        let t = rd_u32(b, &mut p);
        let dt = rd_f64(b, &mut p);
        let m = rd_f64(b, &mut p);
        let g = rd_f64(b, &mut p);
        let n_tensors = rd_u32(b, &mut p);

        let mut w = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let nn = rd_u16(b, &mut p);
            let name = rd_str(b, &mut p, nn);
            let ndim = rd_u8(b, &mut p);
            let mut shape = Vec::with_capacity(ndim);
            let mut count = 1usize;
            for _ in 0..ndim {
                let d = rd_u32(b, &mut p);
                shape.push(d);
                count *= d;
            }
            let mut data = Vec::with_capacity(count);
            for _ in 0..count {
                data.push(rd_f32(b, &mut p));
            }
            w.insert(name, Arr { shape, data });
        }

        let to64 = |a: &Arr| -> Vec<f64> { a.data.iter().map(|x| *x as f64).collect() };
        let v_sm = to64(&w["state_mean"]);
        let v_ss = to64(&w["state_std"]);
        let v_am = to64(&w["action_mean"]);
        let v_as = to64(&w["action_std"]);
        SkyJepaLite {
            action_mode_ctbr: action_mode == "ctbr",
            h,
            t,
            dt,
            m,
            g,
            state_mean: core::array::from_fn(|i| v_sm[i]),
            state_std: core::array::from_fn(|i| v_ss[i]),
            action_mean: core::array::from_fn(|i| v_am[i]),
            action_std: core::array::from_fn(|i| v_as[i]),
            relative_pos: pos_mode == "relative",
            w,
        }
    }

    #[inline]
    fn g(&self, k: &str) -> &Arr {
        &self.w[k]
    }

    // ---- neural ops (single-sample; batch handled by the caller) ---------- //

    /// y = x @ W^T + b, with W = (out, in), x = (in,).
    fn linear(&self, x: &[f32], prefix: &str) -> Vec<f32> {
        let w = self.g(&format!("{prefix}.weight"));
        let b = self.g(&format!("{prefix}.bias"));
        let (out, inn) = (w.dim(0), w.dim(1));
        // Loud failure over silent feature mis-wiring: a blob exported from a
        // prober-input ablation (--prober-inputs latent/latent_action) has a
        // narrower first layer and would otherwise silently consume the wrong
        // slice of the [latent | state | action] feature vector.
        assert_eq!(
            x.len(), inn,
            "{prefix}: input len {} != weight cols {} (prober-input variant blob?)",
            x.len(), inn
        );
        let mut y = vec![0.0f32; out];
        for o in 0..out {
            let mut acc = b.data[o];
            let row = &w.data[o * inn..o * inn + inn];
            for i in 0..inn {
                acc += row[i] * x[i];
            }
            y[o] = acc;
        }
        y
    }

    /// Causal dilated conv1d. inp = (Cin, L), weight = (Cout, Cin, K=3), left pad
    /// = (K-1)*dilation. Output = (Cout, L). Returns Vec indexed [cout*L + l].
    fn conv1d(&self, inp: &[Vec<f32>], wname: &str, bname: &str, dilation: usize) -> Vec<Vec<f32>> {
        let w = self.g(wname);
        let b = self.g(bname);
        let cout = w.dim(0);
        let cin = w.dim(1);
        let k = w.dim(2);
        let l = inp[0].len();
        let mut out = vec![vec![0.0f32; l]; cout];
        for co in 0..cout {
            for o in 0..l {
                let mut acc = b.data[co];
                for ci in 0..cin {
                    let wbase = (co * cin + ci) * k;
                    for kk in 0..k {
                        // padded index -> original index = o + kk*dil - (k-1)*dil
                        let idx = o as isize + (kk * dilation) as isize - ((k - 1) * dilation) as isize;
                        if idx >= 0 && (idx as usize) < l {
                            acc += w.data[wbase + kk] * inp[ci][idx as usize];
                        }
                    }
                }
                out[co][o] = acc;
            }
        }
        out
    }

    /// LayerNorm over the channel axis of an (C, L) buffer (eps 1e-5).
    fn layernorm_channels(&self, x: &mut Vec<Vec<f32>>, gname: &str, bname: &str) {
        let g = self.g(gname);
        let bn = self.g(bname);
        let c = x.len();
        let l = x[0].len();
        for t in 0..l {
            let mut mean = 0.0f32;
            for ch in 0..c {
                mean += x[ch][t];
            }
            mean /= c as f32;
            let mut var = 0.0f32;
            for ch in 0..c {
                let d = x[ch][t] - mean;
                var += d * d;
            }
            var /= c as f32;
            let denom = (var + 1e-5).sqrt();
            for ch in 0..c {
                x[ch][t] = (x[ch][t] - mean) / denom * g.data[ch] + bn.data[ch];
            }
        }
    }

    /// One TCN block: conv -> +bias -> gelu -> LN(channels) -> + residual.
    fn tcn_block(&self, x: &[Vec<f32>], root: &str, dilation: usize) -> Vec<Vec<f32>> {
        let mut y = self.conv1d(
            x,
            &format!("{root}.conv.conv.weight"),
            &format!("{root}.conv.conv.bias"),
            dilation,
        );
        for ch in y.iter_mut() {
            for v in ch.iter_mut() {
                *v = gelu_erf(*v);
            }
        }
        self.layernorm_channels(&mut y, &format!("{root}.norm.weight"), &format!("{root}.norm.bias"));
        // residual: 1x1 conv if present, else identity
        let res_key = format!("{root}.res.weight");
        if self.w.contains_key(&res_key) {
            let res = self.conv1d(x, &res_key, &format!("{root}.res.bias"), 1);
            for (yc, rc) in y.iter_mut().zip(res.iter()) {
                for (yv, rv) in yc.iter_mut().zip(rc.iter()) {
                    *yv += rv;
                }
            }
        } else {
            for (yc, xc) in y.iter_mut().zip(x.iter()) {
                for (yv, xv) in yc.iter_mut().zip(xc.iter()) {
                    *yv += xv;
                }
            }
        }
        y
    }

    /// TCN encoder. x_seq = (L, Cin) row-major -> (L, Cout).
    fn encoder(&self, x_seq: &[Vec<f32>], root: &str) -> Vec<Vec<f32>> {
        // to (Cin, L)
        let l = x_seq.len();
        let cin = x_seq[0].len();
        let mut h: Vec<Vec<f32>> = (0..cin).map(|c| (0..l).map(|t| x_seq[t][c]).collect()).collect();
        for (i, dil) in [1usize, 2, 4].into_iter().enumerate() {
            h = self.tcn_block(&h, &format!("{root}.tcn.blocks.{i}"), dil);
        }
        // back to (L, Cout)
        let cout = h.len();
        (0..l).map(|t| (0..cout).map(|c| h[c][t]).collect()).collect()
    }

    /// PyTorch GRUCell step. inp,h -> new h (len = hidden).
    fn gru_cell(&self, inp: &[f32], hprev: &[f32], prefix: &str) -> Vec<f32> {
        let wih = self.g(&format!("{prefix}.weight_ih"));
        let whh = self.g(&format!("{prefix}.weight_hh"));
        let bih = self.g(&format!("{prefix}.bias_ih"));
        let bhh = self.g(&format!("{prefix}.bias_hh"));
        let hid = hprev.len();
        let in_n = wih.dim(1);
        // gi = inp@wih^T + bih ; gh = h@whh^T + bhh  (each 3*hid)
        let mut gi = vec![0.0f32; 3 * hid];
        let mut gh = vec![0.0f32; 3 * hid];
        for o in 0..3 * hid {
            let mut a = bih.data[o];
            let row = &wih.data[o * in_n..o * in_n + in_n];
            for i in 0..in_n {
                a += row[i] * inp[i];
            }
            gi[o] = a;
            let mut b = bhh.data[o];
            let rowh = &whh.data[o * hid..o * hid + hid];
            for i in 0..hid {
                b += rowh[i] * hprev[i];
            }
            gh[o] = b;
        }
        let mut hnew = vec![0.0f32; hid];
        for i in 0..hid {
            let r = sigmoid(gi[i] + gh[i]);
            let z = sigmoid(gi[hid + i] + gh[hid + i]);
            let n = (gi[2 * hid + i] + r * gh[2 * hid + i]).tanh();
            hnew[i] = (1.0 - z) * n + z * hprev[i];
        }
        hnew
    }

    fn prober(&self, feat: &[f32]) -> Vec<f32> {
        let mut h = self.linear(feat, "prober.net.0");
        for v in h.iter_mut() {
            *v = gelu_erf(*v);
        }
        let mut h2 = self.linear(&h, "prober.net.2");
        for v in h2.iter_mut() {
            *v = gelu_erf(*v);
        }
        self.linear(&h2, "prober.net.4")
    }

    /// Normalize state with translation-invariant position handling. `ref_pos` is
    /// the decision-time position; in relative mode position becomes (x-ref)/std,
    /// in zero mode it's dropped. Must match drone_jepa/model/jepa.py `_norm_state`.
    fn norm_state(&self, x: &[f64; SD], ref_pos: &[f64; 3]) -> [f64; SD] {
        core::array::from_fn(|i| {
            if i < 3 {
                if self.relative_pos { (x[i] - ref_pos[i]) / self.state_std[i] } else { 0.0 }
            } else {
                (x[i] - self.state_mean[i]) / self.state_std[i]
            }
        })
    }
    fn norm_action(&self, a: &[f64; AD]) -> [f64; AD] {
        core::array::from_fn(|i| (a[i] - self.action_mean[i]) / self.action_std[i])
    }

    /// Batched forward. `state_hist`: B×(H,18), `action_window`: B×(H+T,4) f64.
    /// Returns predicted states B×(T,18) f64. Mirrors `jepa-rs` `predict_batch`.
    pub fn predict_batch(
        &self,
        state_hist: &[Vec<[f64; SD]>],
        action_window: &[Vec<[f64; AD]>],
    ) -> Vec<Vec<[f64; SD]>> {
        self.predict_batch_trust(state_hist, action_window).0
    }

    /// Like `predict_batch`, but also returns a per-candidate **latent trust energy**:
    /// `Σ_t max(0, mean_dim(s_pred[t]²) − margin)`. SIGReg trains latents to ~N(0,1)
    /// (per-dim energy ≈ 1), so a large value means the plan drove the predicted
    /// latents off the training manifold — i.e. it's exploiting the model. MPPI adds
    /// `trust_lambda · energy` to the cost to reject such plans.
    pub fn predict_batch_trust(
        &self,
        state_hist: &[Vec<[f64; SD]>],
        action_window: &[Vec<[f64; AD]>],
    ) -> (Vec<Vec<[f64; SD]>>, Vec<f64>) {
        // Energy above which a predicted latent counts as "off-manifold". SIGReg
        // targets N(0,1) (energy~1), but this checkpoint's latent_std≈0.7 → in-dist
        // energy≈0.3-0.5, low enough that margin 0 (penalize total latent energy =
        // "prefer the most confident plan") works best empirically. Env-overridable
        // for sweeps; on wasm `env::var` returns Err so the default (0) applies.
        let trust_margin: f64 = std::env::var("JEPA_TRUST_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let b = state_hist.len();
        let h = self.h;
        let t = self.t;
        let mut preds: Vec<Vec<[f64; SD]>> = vec![Vec::with_capacity(t); b];
        let mut trust: Vec<f64> = vec![0.0; b];

        for i in 0..b {
            // decision-time position: the reference for relative-position mode.
            let cur = state_hist[i][h - 1];
            let ref_pos = [cur[0], cur[1], cur[2]];
            // --- stage 1: encoders + predictor ---
            // normalized state history -> (H, 18)
            let sh: Vec<Vec<f32>> = state_hist[i]
                .iter()
                .map(|s| {
                    let n = self.norm_state(s, &ref_pos);
                    (0..SD).map(|j| n[j] as f32).collect()
                })
                .collect();
            let aw: Vec<Vec<f32>> = action_window[i]
                .iter()
                .map(|a| {
                    let n = self.norm_action(a);
                    (0..AD).map(|j| n[j] as f32).collect()
                })
                .collect();

            let s_enc = self.encoder(&sh, "state_encoder"); // (H,16)
            let z_enc = self.encoder(&aw, "action_encoder"); // (H+T,8)
            let s_now = s_enc[h - 1].clone(); // (16,)
            let latent = s_now.len();

            // predictor unroll
            let mut hgru = self.linear(&s_now, "predictor.h0");
            for v in hgru.iter_mut() {
                *v = v.tanh();
            }
            let mut s_lat = s_now.clone();
            let mut s_pred: Vec<Vec<f32>> = Vec::with_capacity(t); // (T,16)
            for k in 0..t {
                let zk = &z_enc[h - 1 + k]; // (8,)
                let mut inp = Vec::with_capacity(latent + zk.len());
                inp.extend_from_slice(&s_lat);
                inp.extend_from_slice(zk);
                hgru = self.gru_cell(&inp, &hgru, "predictor.cell");
                s_lat = self.linear(&hgru, "predictor.readout");
                s_pred.push(s_lat.clone());
            }

            // latent trust energy: Σ_t max(0, mean_dim(s_pred[t]²) − margin)
            let mut e = 0.0f64;
            for sp in &s_pred {
                let energy = sp.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / sp.len() as f64;
                e += (energy - trust_margin).max(0.0);
            }
            trust[i] = e;

            // --- stage 2: per-step prober + DKI ---
            let mut x = state_hist[i][h - 1];
            for k in 0..t {
                let s_k = if k == 0 { &s_now } else { &s_pred[k - 1] };
                let xn = self.norm_state(&x, &ref_pos);
                let a_k = action_window[i][h - 1 + k];
                let an = self.norm_action(&a_k);
                let mut feat = Vec::with_capacity(latent + SD + AD);
                feat.extend_from_slice(s_k);
                feat.extend(xn.iter().map(|v| *v as f32));
                feat.extend(an.iter().map(|v| *v as f32));
                let out = self.prober(&feat);
                let dv = [out[0] as f64, out[1] as f64, out[2] as f64];
                x = if self.action_mode_ctbr {
                    let om = [out[3] as f64, out[4] as f64, out[5] as f64];
                    dki::step_ctbr(&x, &a_k, &dv, &om, self.dt, self.m, self.g)
                } else {
                    let kmat: [f64; 12] = core::array::from_fn(|j| out[3 + j] as f64);
                    dki::step_rotor_force(&x, &a_k, &dv, &kmat, self.dt, self.m, self.g)
                };
                preds[i].push(x);
            }
        }
        (preds, trust)
    }
}

// --------------------------------------------------------------------------- //
// RolloutModel adapter (mirrors jepa-rs/src/rollout.rs) — feeds the MPPI racer.
// --------------------------------------------------------------------------- //

fn state_to_18(s: &State<f64>) -> [f64; SD] {
    let r = s.q.to_rotmat();
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
    model: SkyJepaLite,
    h: usize,
    t: usize,
    states: VecDeque<[f64; SD]>,
    actions: VecDeque<[f64; AD]>,
}

impl JepaRollout {
    pub fn new(model: SkyJepaLite) -> Self {
        let (h, t) = (model.h, model.t);
        JepaRollout { model, h, t, states: VecDeque::new(), actions: VecDeque::new() }
    }

    /// Clear the observed state/action history (call when starting a fresh race so
    /// stale history doesn't mislead the encoders).
    pub fn reset(&mut self) {
        self.states.clear();
        self.actions.clear();
    }

    fn build_history(&self, init: &State<f64>) -> Vec<[f64; SD]> {
        let cur = state_to_18(init);
        let mut hist: Vec<[f64; SD]> = self.states.iter().copied().collect();
        hist.push(cur);
        while hist.len() < self.h {
            hist.insert(0, hist[0]);
        }
        let start = hist.len() - self.h;
        hist[start..].to_vec()
    }

    fn past_actions(&self) -> Vec<[f64; AD]> {
        let mut pa: Vec<[f64; AD]> = self.actions.iter().copied().collect();
        while pa.len() < self.h - 1 {
            pa.insert(0, *pa.first().unwrap_or(&[0.0; AD]));
        }
        let start = pa.len() - (self.h - 1);
        pa[start..].to_vec()
    }

    /// Whether this model decodes rotor-force actions (vs CTBR). The rotor-force
    /// planner uses this to refuse a CTBR checkpoint.
    pub fn is_rotor_force(&self) -> bool {
        !self.model.action_mode_ctbr
    }

    /// Batched rollout over RAW `[f64; AD]` action sequences (rotor forces for a
    /// rotor-force checkpoint), returning predicted position trajectories + the
    /// per-candidate latent-trust energy. This is the action-type-agnostic core;
    /// the CTBR `RolloutModel` impl just converts then calls this.
    pub fn rollout_raw_trust(
        &self,
        init: &State<f64>,
        seqs: &[Vec<[f64; AD]>],
    ) -> (Vec<Vec<Vec3<f64>>>, Vec<f64>) {
        let b = seqs.len();
        let hist = self.build_history(init);
        let past = self.past_actions();
        let mut sh = Vec::with_capacity(b);
        let mut aw = Vec::with_capacity(b);
        for seq in seqs {
            sh.push(hist.clone());
            let mut w: Vec<[f64; AD]> = Vec::with_capacity(self.h + self.t);
            w.extend_from_slice(&past);
            for a in seq.iter().take(self.t) {
                w.push(*a);
            }
            let last = w.last().copied().unwrap_or([0.0; AD]);
            while w.len() < self.h + self.t {
                w.push(last);
            }
            aw.push(w);
        }
        let (preds, trust) = self.model.predict_batch_trust(&sh, &aw);
        let trajs = preds
            .into_iter()
            .zip(seqs.iter())
            .map(|(traj, seq)| {
                let n = seq.len().min(traj.len());
                traj.into_iter().take(n).map(|x| Vec3::new(x[0], x[1], x[2])).collect()
            })
            .collect();
        (trajs, trust)
    }

    /// Like `rollout_raw_trust` but ALSO returns the predicted per-step angular-rate
    /// magnitude |ω| (state dims 15..18). The rotor-force planner penalizes this to damp
    /// the attitude oscillation ("violent shaking") that a direct-rotor-force MPC has no
    /// inner rate loop to suppress.
    pub fn rollout_raw_trust_om(
        &self,
        init: &State<f64>,
        seqs: &[Vec<[f64; AD]>],
    ) -> (Vec<Vec<Vec3<f64>>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let b = seqs.len();
        let hist = self.build_history(init);
        let past = self.past_actions();
        let mut sh = Vec::with_capacity(b);
        let mut aw = Vec::with_capacity(b);
        for seq in seqs {
            sh.push(hist.clone());
            let mut w: Vec<[f64; AD]> = Vec::with_capacity(self.h + self.t);
            w.extend_from_slice(&past);
            for a in seq.iter().take(self.t) {
                w.push(*a);
            }
            let last = w.last().copied().unwrap_or([0.0; AD]);
            while w.len() < self.h + self.t {
                w.push(last);
            }
            aw.push(w);
        }
        let (preds, trust) = self.model.predict_batch_trust(&sh, &aw);
        let mut trajs = Vec::with_capacity(b);
        let mut oms = Vec::with_capacity(b);
        let mut uprs = Vec::with_capacity(b);
        for (traj, seq) in preds.into_iter().zip(seqs.iter()) {
            let n = seq.len().min(traj.len());
            let mut pos = Vec::with_capacity(n);
            let mut om = Vec::with_capacity(n);
            let mut upr = Vec::with_capacity(n);
            for x in traj.into_iter().take(n) {
                pos.push(Vec3::new(x[0], x[1], x[2]));
                om.push((x[15] * x[15] + x[16] * x[16] + x[17] * x[17]).sqrt());
                upr.push(x[14]); // R[2][2] = world-z of body-z: 1=level, 0=sideways, -1=inverted
            }
            trajs.push(pos);
            oms.push(om);
            uprs.push(upr);
        }
        (trajs, oms, uprs, trust)
    }

    /// Record an executed RAW action into the rolling history.
    pub fn observe_raw(&mut self, state: &State<f64>, action: [f64; AD]) {
        self.states.push_back(state_to_18(state));
        while self.states.len() > self.h - 1 {
            self.states.pop_front();
        }
        self.actions.push_back(action);
        while self.actions.len() > self.h - 1 {
            self.actions.pop_front();
        }
    }
}

impl RolloutModel for JepaRollout {
    fn rollout(&self, init: &State<f64>, actions: &[CtbrCmd<f64>], dt: f64, out: &mut Vec<Vec3<f64>>) {
        let seqs = [actions.to_vec()];
        let trajs = self.rollout_batch(init, &seqs, dt);
        *out = trajs.into_iter().next().unwrap_or_default();
    }

    fn rollout_batch(
        &self,
        init: &State<f64>,
        seqs: &[Vec<CtbrCmd<f64>>],
        dt: f64,
    ) -> Vec<Vec<Vec3<f64>>> {
        self.rollout_batch_trust(init, seqs, dt).0
    }

    fn rollout_batch_trust(
        &self,
        init: &State<f64>,
        seqs: &[Vec<CtbrCmd<f64>>],
        _dt: f64,
    ) -> (Vec<Vec<Vec3<f64>>>, Vec<f64>) {
        // The model always predicts T steps, but MPPI may plan a shorter horizon;
        // rollout_raw_trust truncates each trajectory to the candidate length.
        let raw: Vec<Vec<[f64; AD]>> =
            seqs.iter().map(|s| s.iter().map(ctbr_to_4).collect()).collect();
        self.rollout_raw_trust(init, &raw)
    }

    fn observe(&mut self, state: &State<f64>, action: &CtbrCmd<f64>) {
        self.observe_raw(state, ctbr_to_4(action));
    }
}

// Parameter-free DKI (ported from jepa-rs/src/dki.rs).
mod dki {
    type M3 = [[f64; 3]; 3];

    #[inline]
    fn matmul(a: &M3, b: &M3) -> M3 {
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        o
    }

    fn exp_so3(phi: [f64; 3]) -> M3 {
        let theta = (phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]).sqrt();
        let theta = theta.max(1e-8);
        let k = [phi[0] / theta, phi[1] / theta, phi[2] / theta];
        let kk: M3 = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
        let kk2 = matmul(&kk, &kk);
        let s = theta.sin();
        let c = 1.0 - theta.cos();
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let eye = if i == j { 1.0 } else { 0.0 };
                r[i][j] = eye + s * kk[i][j] + c * kk2[i][j];
            }
        }
        r
    }

    pub fn step_rotor_force(
        x: &[f64; 18], a: &[f64; 4], dvdot: &[f64; 3], k: &[f64; 12], dt: f64, m: f64, g: f64,
    ) -> [f64; 18] {
        let thrust = a[0] + a[1] + a[2] + a[3];
        let omega_dot = [
            k[0] * a[0] + k[1] * a[1] + k[2] * a[2] + k[3] * a[3],
            k[4] * a[0] + k[5] * a[1] + k[6] * a[2] + k[7] * a[3],
            k[8] * a[0] + k[9] * a[1] + k[10] * a[2] + k[11] * a[3],
        ];
        step_collective(x, thrust, dvdot, &omega_dot, dt, m, g)
    }

    pub fn step_ctbr(
        x: &[f64; 18], a: &[f64; 4], dvdot: &[f64; 3], omega_dot: &[f64; 3], dt: f64, m: f64, g: f64,
    ) -> [f64; 18] {
        step_collective(x, a[0], dvdot, omega_dot, dt, m, g)
    }

    fn step_collective(
        x: &[f64; 18], thrust: f64, dvdot: &[f64; 3], omega_dot: &[f64; 3], dt: f64, m: f64, g: f64,
    ) -> [f64; 18] {
        let p = [x[0], x[1], x[2]];
        let v = [x[3], x[4], x[5]];
        let r: M3 = [[x[6], x[7], x[8]], [x[9], x[10], x[11]], [x[12], x[13], x[14]]];
        let omega = [x[15], x[16], x[17]];
        let inv_mass = 1.0 / m;
        let body_z_world = [r[0][2], r[1][2], r[2][2]];
        let vdot = [
            thrust * inv_mass * body_z_world[0] + dvdot[0],
            thrust * inv_mass * body_z_world[1] + dvdot[1],
            -g + thrust * inv_mass * body_z_world[2] + dvdot[2],
        ];
        let v_next = [v[0] + vdot[0] * dt, v[1] + vdot[1] * dt, v[2] + vdot[2] * dt];
        let p_next = [p[0] + v_next[0] * dt, p[1] + v_next[1] * dt, p[2] + v_next[2] * dt];
        let omega_next = [
            omega[0] + omega_dot[0] * dt,
            omega[1] + omega_dot[1] * dt,
            omega[2] + omega_dot[2] * dt,
        ];
        let exp = exp_so3([omega[0] * dt, omega[1] * dt, omega[2] * dt]);
        let r_next = matmul(&r, &exp);
        [
            p_next[0], p_next[1], p_next[2],
            v_next[0], v_next[1], v_next[2],
            r_next[0][0], r_next[0][1], r_next[0][2],
            r_next[1][0], r_next[1][1], r_next[1][2],
            r_next[2][0], r_next[2][1], r_next[2][2],
            omega_next[0], omega_next[1], omega_next[2],
        ]
    }
}
