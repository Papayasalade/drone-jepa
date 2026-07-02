//! Candle-vs-PyTorch differential test: the Rust SkyJepa forward must match the
//! PyTorch `predict()` goldens (web-demo/fixtures/jepa/goldens.json) within tol.
//! (Candle runs f32 like PyTorch; the DKI runs f64, so a small gap is expected.)

use std::path::Path;

use jepa_rs::{SkyJepa, AD, SD};
use serde::Deserialize;

#[derive(Deserialize)]
struct Goldens {
    config: Config,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Config {
    action_mode: String,
    stem: Option<String>,
}
#[derive(Deserialize)]
struct Case {
    state_hist: Vec<Vec<f64>>,    // (H, 18)
    action_window: Vec<Vec<f64>>, // (H+T, 4)
    pred: Vec<Vec<f64>>,          // (T, 18)
}

fn arr<const N: usize>(v: &[f64]) -> [f64; N] {
    core::array::from_fn(|i| v[i])
}

#[test]
fn candle_matches_pytorch() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let txt = std::fs::read_to_string(root.join("../fixtures/jepa/goldens.json"))
        .expect("goldens (run scripts/export_jepa.py)");
    let g: Goldens = serde_json::from_str(&txt).unwrap();
    // Stem must match the model used by export_jepa.py. Prefer JEPA_STEM for
    // explicit checks; otherwise infer the committed default from the fixture.
    let stem = std::env::var("JEPA_STEM").unwrap_or_else(|_| {
        g.config.stem.clone().unwrap_or_else(|| match g.config.action_mode.as_str() {
            "rotor_force" => "skyjepa_rotor".into(),
            "ctbr" => "skyjepa_ctbr_1x".into(),
            other => panic!("unknown JEPA action_mode in goldens: {other}"),
        })
    });
    let model = SkyJepa::load(
        root.join(format!("weights/{stem}.safetensors")).to_str().unwrap(),
        root.join(format!("weights/{stem}.json")).to_str().unwrap(),
    )
    .expect("load model (run scripts/export_jepa.py)");

    let mut worst = 0.0_f64;
    let mut worst_pos = 0.0_f64;
    for (ci, case) in g.cases.iter().enumerate() {
        let sh: Vec<[f64; SD]> = case.state_hist.iter().map(|r| arr(r)).collect();
        let aw: Vec<[f64; AD]> = case.action_window.iter().map(|r| arr(r)).collect();

        let pred = model.predict_batch(&[sh], &[aw]).unwrap();
        let pred = &pred[0]; // (T,18)

        for (k, (got, want)) in pred.iter().zip(case.pred.iter()).enumerate() {
            for d in 0..SD {
                let e = (got[d] - want[d]).abs();
                worst = worst.max(e);
                if d < 3 {
                    worst_pos = worst_pos.max(e);
                }
            }
            if k == case.pred.len() - 1 {
                println!(
                    "case {ci} final pos: rust=[{:.4},{:.4},{:.4}] py=[{:.4},{:.4},{:.4}]",
                    got[0], got[1], got[2], want[0], want[1], want[2]
                );
            }
        }
    }
    println!("worst abs err over all dims = {worst:.2e}  (position = {worst_pos:.2e})");
    assert!(worst < 5e-3, "Candle vs PyTorch mismatch: {worst:.3e}");
}
