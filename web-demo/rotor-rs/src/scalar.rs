//! The `Scalar` trait: the "base arithmetic" abstraction the whole dynamics
//! kernel is generic over.
//!
//! Write the physics once against `S: Scalar`, instantiate with `f64` for the
//! single-vehicle sim today, and later implement `Scalar` for a SIMD / array
//! lane type so the *identical* monomorphized kernel steps a whole batch of
//! drones at once. Every method here is **branchless** — conditionals in the
//! dynamics become arithmetic (`clamp`, `copysign`, lane-wise select), so there
//! is no data-dependent control flow to break vectorization.

use core::ops::{Add, Div, Mul, Neg, Sub};

pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;

    /// Broadcast a constant to every lane.
    fn splat(x: f64) -> Self;

    fn sqrt(self) -> Self;
    fn abs(self) -> Self;

    /// `copysign(1, self)` — magnitude 1 carrying `self`'s sign. Branchless.
    /// Note: at `self == 0` this returns `+1`, but every use site multiplies by
    /// `sqrt(|self|) == 0` there, so the value is irrelevant (matches numpy's
    /// `sign(0) == 0` in effect).
    fn signum(self) -> Self;

    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;

    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        // max-then-min: branchless, and matches numpy.clip semantics for lo<=hi.
        self.max(lo).min(hi)
    }
}

impl Scalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;

    #[inline]
    fn splat(x: f64) -> f64 {
        x
    }
    #[inline]
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    #[inline]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline]
    fn signum(self) -> f64 {
        // copysign(1, self) is branchless (unlike f64::signum, which special-cases NaN/0).
        1.0_f64.copysign(self)
    }
    #[inline]
    fn min(self, other: f64) -> f64 {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: f64) -> f64 {
        f64::max(self, other)
    }
}
