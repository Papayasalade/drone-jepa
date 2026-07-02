//! Tiny zero-dependency PRNG for MPPI sampling (xorshift64* + Box–Muller).
//! Deterministic and seedable so racing runs are reproducible.

pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // avoid the all-zero state
        Rng { s: seed ^ 0x9E3779B97F4A7C15 | 1 }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        // top 53 bits -> f64 in [0,1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box–Muller.
    #[inline]
    pub fn normal(&mut self) -> f64 {
        // guard u1 away from 0 for the log
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}
