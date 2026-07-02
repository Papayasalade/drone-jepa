//! Parity test: the zero-dep `rotor-rs::jepa::SkyJepaLite` (the WASM inference)
//! must match the trusted Candle `jepa_rs::SkyJepa` op-for-op. Both load the SAME
//! checkpoint (Candle from .safetensors, Lite from the exported .jblob) and run
//! `predict_batch` on random histories; we assert the predicted trajectories agree.
//!
//! Run after exporting both: `python scripts/export_jepa.py` (goldens/safetensors)
//! and `python scripts/export_jepa_blob.py <stem>` (the .jblob).

use std::path::Path;

use jepa_rs::{SkyJepa, AD, SD};
use rotor_rs::jepa::SkyJepaLite;

/// Tiny deterministic LCG so the test is reproducible without rand.
struct Lcg(u64);
impl Lcg {
    fn next_f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0 // [-1,1)
    }
}

fn check_stem(stem: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candle = SkyJepa::load(
        root.join(format!("weights/{stem}.safetensors")).to_str().unwrap(),
        root.join(format!("weights/{stem}.json")).to_str().unwrap(),
    )
    .expect("load candle model");
    let blob = std::fs::read(root.join(format!("../rotor-rs/assets/{stem}.jblob")))
        .expect("read .jblob (run scripts/export_jepa_blob.py)");
    let lite = SkyJepaLite::from_blob(&blob);

    let h = candle.config().history;
    let t = candle.config().horizon;
    let b = 5usize;

    // random but plausible: position/vel small, rotation near identity, omega small
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let mut sh: Vec<Vec<[f64; SD]>> = Vec::new();
    let mut aw: Vec<Vec<[f64; AD]>> = Vec::new();
    for _ in 0..b {
        let mut hist = Vec::new();
        for _ in 0..h {
            let mut s = [0.0f64; SD];
            for k in 0..3 {
                s[k] = rng.next_f() * 2.0; // pos
                s[3 + k] = rng.next_f() * 1.0; // vel
                s[15 + k] = rng.next_f() * 0.5; // omega
            }
            // rotation ~ identity + small perturbation (rows 6..15)
            let r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            for k in 0..9 {
                s[6 + k] = r[k] + rng.next_f() * 0.05;
            }
            hist.push(s);
        }
        let mut win = Vec::new();
        for _ in 0..(h + t) {
            let mut a = [0.0f64; AD];
            a[0] = 4.9 + rng.next_f() * 1.0; // thrust near hover
            for k in 1..AD {
                a[k] = rng.next_f() * 2.0;
            }
            win.push(a);
        }
        sh.push(hist);
        aw.push(win);
    }

    let pc = candle.predict_batch(&sh, &aw).expect("candle predict");
    let pl = lite.predict_batch(&sh, &aw);

    let mut worst = 0.0f64;
    for i in 0..b {
        for k in 0..t {
            for d in 0..SD {
                worst = worst.max((pc[i][k][d] - pl[i][k][d]).abs());
            }
        }
    }
    println!("[{stem}] worst |candle - lite| = {worst:.3e}");
    assert!(worst < 2e-3, "{stem}: parity gap too large: {worst:.3e}");
}

#[test]
fn lite_matches_candle_ctbr() {
    check_stem("skyjepa_ctbr_1x");
}

#[test]
fn lite_matches_candle_rotor() {
    check_stem("skyjepa_rotor");
}
