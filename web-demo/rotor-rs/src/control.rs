//! Control abstractions = how a controller's command maps to commanded rotor
//! speeds. Ported from RotorPy `get_cmd_motor_speeds` — only the two we use.
//!
//! Each abstraction is a zero-sized type implementing `ControlLaw`, so the choice
//! is a generic (`Multirotor<S, C>`) that monomorphizes away — the conversion is
//! inlined with no runtime branch, keeping the kernel batch-friendly.

use crate::linalg::Vec3;
use crate::params::{QuadParams, NUM_ROTORS};
use crate::scalar::Scalar;
use crate::state::State;

/// Force -> rotor speed: `sign(f/k_eta) * sqrt(|f/k_eta|)`. Branchless.
#[inline]
fn forces_to_speeds<S: Scalar>(forces: [S; NUM_ROTORS], k_eta: S) -> [S; NUM_ROTORS] {
    core::array::from_fn(|r| {
        let x = forces[r] / k_eta;
        x.signum() * x.abs().sqrt()
    })
}

pub trait ControlLaw<S: Scalar> {
    type Command;
    /// Commanded rotor speeds (pre clip to [min,max], which `step` applies).
    fn cmd_rotor_speeds(
        params: &QuadParams<S>,
        state: &State<S>,
        cmd: &Self::Command,
    ) -> [S; NUM_ROTORS];
}

/// `cmd_motor_thrusts`: the command is the four individual rotor forces [N].
pub struct RotorForce;

impl<S: Scalar> ControlLaw<S> for RotorForce {
    type Command = [S; NUM_ROTORS];
    #[inline]
    fn cmd_rotor_speeds(
        params: &QuadParams<S>,
        _state: &State<S>,
        cmd: &Self::Command,
    ) -> [S; NUM_ROTORS] {
        forces_to_speeds(*cmd, params.k_eta)
    }
}

/// `cmd_ctbr`: collective thrust [N] + desired body rates [rad/s]. An inner
/// P rate loop produces a moment, then the allocation matrix yields rotor forces.
#[derive(Clone, Copy, Debug)]
pub struct CtbrCmd<S> {
    pub thrust: S,
    pub w_cmd: Vec3<S>,
}

pub struct Ctbr;

impl<S: Scalar> ControlLaw<S> for Ctbr {
    type Command = CtbrCmd<S>;
    #[inline]
    fn cmd_rotor_speeds(
        params: &QuadParams<S>,
        state: &State<S>,
        cmd: &Self::Command,
    ) -> [S; NUM_ROTORS] {
        // Inner rate loop: wdot_cmd = -k_w (w - w_cmd); moment = I @ wdot_cmd.
        let w_err = state.w - cmd.w_cmd;
        let wdot_cmd = w_err.scale(-params.k_w);
        let moment = params.inertia.matvec(wdot_cmd);
        // Allocate [thrust, Mx, My, Mz] -> rotor forces -> rotor speeds.
        let tm = [cmd.thrust, moment.x, moment.y, moment.z];
        let forces = params.tm_to_f.matvec(tm);
        forces_to_speeds(forces, params.k_eta)
    }
}
