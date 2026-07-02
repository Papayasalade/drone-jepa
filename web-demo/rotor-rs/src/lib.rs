//! `rotor-rs` — a branchless, scalar-generic Rust port of RotorPy's
//! single-vehicle quadrotor dynamics (the ground-truth sim for the web demo).
//!
//! Write the physics once against [`scalar::Scalar`]; instantiate with `f64`
//! today and a SIMD/array lane type later to batch for free. Validated by
//! differential ("golden") testing against Python RotorPy — see `tests/golden.rs`.

pub mod control;
pub mod gates;
pub mod linalg;
pub mod mppi;
pub mod multirotor;
pub mod nmpc;
pub mod params;
pub mod references;
pub mod rng;
pub mod scalar;
pub mod se3;
pub mod simd;
pub mod so3;
pub mod state;

pub use control::{Ctbr, CtbrCmd, ControlLaw, RotorForce};
pub use gates::{Course, Gate};
pub use linalg::{Mat3, Mat4, Quat, Vec3};
pub use mppi::{Controller, MppiConfig, MppiController, RolloutModel, TrueDynamics};
pub use multirotor::{clip_speeds, integrate, Multirotor};
pub use nmpc::Nmpc;
pub use references::{FourierRef, GpRef};
pub use se3::{FlatRef, Se3Control};

#[cfg(feature = "jepa")]
pub mod jepa;

#[cfg(feature = "jepa")]
pub mod rotor_mppi;

#[cfg(feature = "jepa")]
pub mod spline_mppi;

#[cfg(feature = "jepa")]
pub mod rl;

#[cfg(feature = "wasm")]
pub mod wasm;
pub use params::{QuadParams, QuadParamsInput, GRAVITY, NUM_ROTORS};
pub use scalar::Scalar;
pub use state::{State, STATE_LEN};
