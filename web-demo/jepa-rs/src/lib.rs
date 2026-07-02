//! Candle (Rust) inference port of the SkyJEPA world model.
//!
//! The neural parts (TCN encoders, GRU predictor, MLP prober) run in Candle f32,
//! loaded from the PyTorch checkpoint via safetensors. The parameter-free DKI
//! integrator (semi-implicit Euler + SO(3) exp) runs in plain Rust f64 — it has
//! no weights, and batched SO(3) is far cleaner outside Candle. Validated against
//! PyTorch `predict()` goldens (see tests/golden.rs).
//!
//! Layout matches drone_jepa: state is 18-dim [pos3, vel3, R(row-major 9), omega3];
//! CTBR action is 4-dim [thrust, wx, wy, wz].

use std::collections::HashMap;

use candle_core::{Device, Result, Tensor, D};
use serde::Deserialize;

mod dki;
pub mod rollout;

pub const SD: usize = 18; // state dim
pub const AD: usize = 4; // action dim

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub action_mode: String,
    #[serde(default = "default_pos_mode")]
    pub pos_mode: String,
    pub history: usize,
    pub horizon: usize,
    pub width_mult: f64,
    pub prober_hidden: usize,
    pub dt: f64,
    pub m_nominal: f64,
    pub g: f64,
}

fn default_pos_mode() -> String {
    "zero".to_string()
}

/// A loaded SkyJEPA model ready for inference.
pub struct SkyJepa {
    w: HashMap<String, Tensor>,
    dev: Device,
    cfg: Config,
    // normalization as rust arrays (for the f64 DKI side)
    state_mean: [f64; SD],
    state_std: [f64; SD],
    action_mean: [f64; AD],
    action_std: [f64; AD],
}

impl SkyJepa {
    pub fn load(weights: &str, config_json: &str) -> Result<Self> {
        let dev = Device::Cpu;
        let w = candle_core::safetensors::load(weights, &dev)?;
        let cfg: Config = serde_json::from_str(
            &std::fs::read_to_string(config_json).map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let f = |k: &str| -> Result<Vec<f64>> {
            Ok(w[k].to_vec1::<f32>()?.into_iter().map(|x| x as f64).collect())
        };
        let arr18 = |v: Vec<f64>| -> [f64; SD] { core::array::from_fn(|i| v[i]) };
        let arr4 = |v: Vec<f64>| -> [f64; AD] { core::array::from_fn(|i| v[i]) };
        let state_mean = arr18(f("state_mean")?);
        let state_std = arr18(f("state_std")?);
        let action_mean = arr4(f("action_mean")?);
        let action_std = arr4(f("action_std")?);
        Ok(Self { w, dev, cfg, state_mean, state_std, action_mean, action_std })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    #[inline]
    fn t(&self, k: &str) -> &Tensor {
        &self.w[k]
    }

    // ---- candle helpers --------------------------------------------------- //
    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w = self.t(&format!("{prefix}.weight"));
        let b = self.t(&format!("{prefix}.bias"));
        x.matmul(&w.t()?)?.broadcast_add(b)
    }

    fn layernorm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let denom = (var + 1e-5)?.sqrt()?;
        let normed = xc.broadcast_div(&denom)?;
        normed.broadcast_mul(gamma)?.broadcast_add(beta)
    }

    /// One TCN block on x: (B, Cin, L) -> (B, Cout, L).
    fn tcn_block(&self, x: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
        let cw = self.t(&format!("{prefix}.conv.conv.weight"));
        let cb = self.t(&format!("{prefix}.conv.conv.bias"));
        let pad = 2 * dilation; // (kernel-1)*dilation, kernel=3
        let xp = x.pad_with_zeros(D::Minus1, pad, 0)?;
        let cout = cw.dim(0)?;
        let y = xp.conv1d(cw, 0, 1, dilation, 1)?;
        let y = y.broadcast_add(&cb.reshape((1, cout, 1))?)?;
        let y = y.gelu_erf()?;
        // LayerNorm over channels: (B,C,L) -> (B,L,C) -> LN -> back
        let yt = y.transpose(1, 2)?.contiguous()?;
        let g = self.t(&format!("{prefix}.norm.weight"));
        let bn = self.t(&format!("{prefix}.norm.bias"));
        let yt = self.layernorm(&yt, g, bn)?;
        let y = yt.transpose(1, 2)?.contiguous()?;
        // residual
        let res_key = format!("{prefix}.res.weight");
        let res = if self.w.contains_key(&res_key) {
            let rw = self.t(&res_key);
            let rb = self.t(&format!("{prefix}.res.bias"));
            let rcout = rw.dim(0)?;
            x.conv1d(rw, 0, 1, 1, 1)?.broadcast_add(&rb.reshape((1, rcout, 1))?)?
        } else {
            x.clone()
        };
        y + res
    }

    /// TCN encoder. x_seq: (B, L, Cin) -> (B, L, Cout).
    fn encoder(&self, x_seq: &Tensor, root: &str) -> Result<Tensor> {
        let mut h = x_seq.transpose(1, 2)?.contiguous()?; // (B, Cin, L)
        for (i, dil) in [1usize, 2, 4].into_iter().enumerate() {
            h = self.tcn_block(&h, &format!("{root}.tcn.blocks.{i}"), dil)?;
        }
        h.transpose(1, 2)?.contiguous() // (B, L, Cout)
    }

    fn sigmoid(x: &Tensor) -> Result<Tensor> {
        (x.neg()?.exp()? + 1.0)?.recip()
    }

    /// One PyTorch-style GRUCell step. inp,h: (B, *); returns new h: (B, hidden).
    fn gru_cell(&self, inp: &Tensor, h: &Tensor, prefix: &str) -> Result<Tensor> {
        let wih = self.t(&format!("{prefix}.weight_ih"));
        let whh = self.t(&format!("{prefix}.weight_hh"));
        let bih = self.t(&format!("{prefix}.bias_ih"));
        let bhh = self.t(&format!("{prefix}.bias_hh"));
        let gi = inp.matmul(&wih.t()?)?.broadcast_add(bih)?; // (B, 3H)
        let gh = h.matmul(&whh.t()?)?.broadcast_add(bhh)?;
        let hidden = h.dim(D::Minus1)?;
        let part = |t: &Tensor, i: usize| t.narrow(D::Minus1, i * hidden, hidden);
        let r = Self::sigmoid(&(part(&gi, 0)? + part(&gh, 0)?)?)?;
        let z = Self::sigmoid(&(part(&gi, 1)? + part(&gh, 1)?)?)?;
        let n = (part(&gi, 2)? + (&r * part(&gh, 2)?)?)?.tanh()?;
        let one_minus_z = (1.0 - &z)?;
        (one_minus_z * n)? + (z * h)?
    }

    /// Predictor unroll. s_now: (B,16); z_seq: (B,T,8). Returns s_pred (B,T,16).
    fn predictor(&self, s_now: &Tensor, z_seq: &Tensor) -> Result<Tensor> {
        let (_b, t, _) = z_seq.dims3()?;
        let mut h = self.linear(s_now, "predictor.h0")?.tanh()?;
        let mut s = s_now.clone();
        let mut out = Vec::with_capacity(t);
        for k in 0..t {
            let zk = z_seq.narrow(1, k, 1)?.squeeze(1)?; // (B,8)
            let inp = Tensor::cat(&[&s, &zk], D::Minus1)?; // (B,24)
            h = self.gru_cell(&inp, &h, "predictor.cell")?;
            s = self.linear(&h, "predictor.readout")?; // (B,16)
            out.push(s.clone());
        }
        Tensor::stack(&out, 1) // (B,T,16)
    }

    /// Prober. feat (B, latent+18+4) -> (dvdot (B,3), ang (B,3) ctbr).
    fn prober(&self, feat: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.linear(feat, "prober.net.0")?.gelu_erf()?;
        let h = self.linear(&h, "prober.net.2")?.gelu_erf()?;
        let out = self.linear(&h, "prober.net.4")?; // (B, 6) ctbr | (B, 15) rotor_force
        let dvdot = out.narrow(D::Minus1, 0, 3)?;
        let ang_dim = out.dim(D::Minus1)? - 3; // 3 (ctbr omega_dot) or 12 (rotor K)
        let ang = out.narrow(D::Minus1, 3, ang_dim)?;
        Ok((dvdot, ang))
    }

    fn norm_state_t(&self, x: &Tensor, ref_pos: &Tensor) -> Result<Tensor> {
        let n = x.broadcast_sub(self.t("state_mean"))?.broadcast_div(self.t("state_std"))?;
        // translation invariance on position (dims 0:3) — must match training
        // (drone_jepa) and the f64 `norm_state` below.
        let d = n.dims().len() - 1;
        let rest = n.narrow(d, 3, SD - 3)?; // velocity/R/omega, normalized
        let pos = if self.cfg.pos_mode == "relative" {
            let std3 = self.t("state_std").narrow(0, 0, 3)?;
            x.narrow(d, 0, 3)?.broadcast_sub(ref_pos)?.broadcast_div(&std3)?
        } else {
            x.narrow(d, 0, 3)?.zeros_like()? // zeroed
        };
        Tensor::cat(&[&pos, &rest], d)
    }
    fn norm_action_t(&self, a: &Tensor) -> Result<Tensor> {
        a.broadcast_sub(self.t("action_mean"))?.broadcast_div(self.t("action_std"))
    }

    /// Batched forward. state_hist: B×(H,18), action_window: B×(H+T,4) f64 row-major.
    /// Returns predicted states B×(T,18) f64.
    pub fn predict_batch(
        &self,
        state_hist: &[Vec<[f64; SD]>],
        action_window: &[Vec<[f64; AD]>],
    ) -> Result<Vec<Vec<[f64; SD]>>> {
        let b = state_hist.len();
        let h = self.cfg.history;
        let t = self.cfg.horizon;
        let dev = &self.dev;

        let sh_flat: Vec<f32> = state_hist.iter().flatten().flatten().map(|x| *x as f32).collect();
        let aw_len = action_window[0].len();
        let aw_flat: Vec<f32> =
            action_window.iter().flatten().flatten().map(|x| *x as f32).collect();
        let sh = Tensor::from_vec(sh_flat, (b, h, SD), dev)?;
        let aw = Tensor::from_vec(aw_flat, (b, aw_len, AD), dev)?;

        // decision-time position (B,1,3), the reference for relative-position mode
        let ref_t = sh.narrow(1, h - 1, 1)?.narrow(2, 0, 3)?.contiguous()?;

        // stage 1: encode + predictor (candle)
        let s_enc = self.encoder(&self.norm_state_t(&sh, &ref_t)?, "state_encoder")?; // (B,H,16)
        let z_enc = self.encoder(&self.norm_action_t(&aw)?, "action_encoder")?; // (B,H+T,8)
        let s_now = s_enc.narrow(1, h - 1, 1)?.squeeze(1)?; // (B,16)
        let z_seq = z_enc.narrow(1, h - 1, t)?.contiguous()?; // (B,T,8)
        let s_pred = self.predictor(&s_now, &z_seq)?; // (B,T,16)

        let latent = s_now.dim(D::Minus1)?;
        let s_now_v = s_now.to_vec2::<f32>()?;
        let s_pred_v = s_pred.to_vec3::<f32>()?;

        // stage 2: per-step prober (candle) + DKI (rust f64)
        let mut x: Vec<[f64; SD]> = (0..b).map(|i| state_hist[i][h - 1]).collect();
        let mut preds: Vec<Vec<[f64; SD]>> = vec![Vec::with_capacity(t); b];

        for k in 0..t {
            let mut feat = Vec::with_capacity(b * (latent + SD + AD));
            for i in 0..b {
                let s_k = if k == 0 { &s_now_v[i] } else { &s_pred_v[i][k - 1] };
                feat.extend(s_k.iter().copied());
                let c = state_hist[i][h - 1];
                let xn = self.norm_state(&x[i], &[c[0], c[1], c[2]]);
                feat.extend(xn.iter().map(|v| *v as f32));
                let a_k = action_window[i][h - 1 + k];
                let an = self.norm_action(&a_k);
                feat.extend(an.iter().map(|v| *v as f32));
            }
            let feat_t = Tensor::from_vec(feat, (b, latent + SD + AD), dev)?;
            let (dvdot, ang) = self.prober(&feat_t)?;
            let dvdot = dvdot.to_vec2::<f32>()?;
            let ang = ang.to_vec2::<f32>()?;
            let is_ctbr = self.cfg.action_mode == "ctbr";
            let (dt, m, g) = (self.cfg.dt, self.cfg.m_nominal, self.cfg.g);
            for i in 0..b {
                let a_k = action_window[i][h - 1 + k];
                let dv = [dvdot[i][0] as f64, dvdot[i][1] as f64, dvdot[i][2] as f64];
                x[i] = if is_ctbr {
                    let om = [ang[i][0] as f64, ang[i][1] as f64, ang[i][2] as f64];
                    dki::step_ctbr(&x[i], &a_k, &dv, &om, dt, m, g)
                } else {
                    let kmat: [f64; 12] = core::array::from_fn(|j| ang[i][j] as f64);
                    dki::step_rotor_force(&x[i], &a_k, &dv, &kmat, dt, m, g)
                };
                preds[i].push(x[i]);
            }
        }
        Ok(preds)
    }

    fn norm_state(&self, x: &[f64; SD], ref_pos: &[f64; 3]) -> [f64; SD] {
        // translation invariance on position (dims 0:3): relative or zeroed.
        let relative = self.cfg.pos_mode == "relative";
        core::array::from_fn(|i| {
            if i < 3 {
                if relative { (x[i] - ref_pos[i]) / self.state_std[i] } else { 0.0 }
            } else {
                (x[i] - self.state_mean[i]) / self.state_std[i]
            }
        })
    }
    fn norm_action(&self, a: &[f64; AD]) -> [f64; AD] {
        core::array::from_fn(|i| (a[i] - self.action_mean[i]) / self.action_std[i])
    }
}
