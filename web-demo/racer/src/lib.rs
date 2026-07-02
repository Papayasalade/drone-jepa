//! `racer` — drone-jepa's application layer on top of the [`rotor_rs`]
//! simulator: the SkyJEPA world-model inference (`jepa`), sampling planners
//! (`mppi`, `rotor_mppi`, `spline_mppi`), gate-racing courses (`gates`), the
//! reactive RL policy runner (`rl`), and the browser demo bindings (`wasm`,
//! behind the `wasm` feature).
//!
//! Everything simulator-generic lives in the standalone
//! [rotor-rs](https://github.com/Papayasalade/rotor-rs) crate; this crate is
//! the project-specific part and is not published.

pub mod gates;
pub mod jepa;
pub mod mppi;
pub mod rl;
pub mod rotor_mppi;
pub mod spline_mppi;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export the whole simulator so downstream code and examples can keep the
// historical single-crate import style (`use racer::{Ctbr, Multirotor, ...}`).
pub use rotor_rs::*;

pub use gates::{Course, Gate};
pub use mppi::{Controller, MppiConfig, MppiController, RolloutModel, TrueDynamics};
