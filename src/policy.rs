// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Search policy: how much factoring a Mersenne number deserves before its
//! primality test. A factor ends the search for that exponent and saves
//! `tests_saved` primality tests of about p squarings each, so factoring
//! work pays when its chance of a factor times that saving exceeds its
//! own cost. The model has the shape GIMPS uses; the numbers here are this
//! program's own:
//!
//! * a factor of 2^p - 1 lies in [2^k, 2^(k + 1)) with probability about
//!   1 / k, for k above the trial factoring depth already done;
//! * such a factor q = 2 m p + 1 is found by P-1 with bounds (B1, B2) when
//!   m is B1 smooth apart from at most one prime up to B2, with Dickman's
//!   rho for the chance that a random m is;
//! * costs are counted in squarings modulo 2^p - 1: about 1.44 B1 for
//!   stage 1 and `stage2_cost` per prime between B1 and B2 for stage 2.
//!
//! Trial factoring a bit level costs one candidate test per surviving
//! candidate; its price in squarings (`tf_cost`) depends on the machine
//! and should come from a measurement.

use serde::Serialize;
use std::sync::OnceLock;

/// Steps per unit of u in the rho table.
const RHO_STEPS: usize = 256;
/// rho is tabulated up to this u; beyond it the chance is below 1e-29.
const RHO_MAX: f64 = 24.0;

/// Dickman's rho: the chance that a random integer x has no prime factor
/// above x^(1/u). Tabulated once by the trapezoid rule on
/// rho'(u) = -rho(u - 1) / u, and interpolated linearly.
pub fn rho(u: f64) -> f64 {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    if u <= 1.0 {
        return 1.0;
    }
    if u >= RHO_MAX {
        return 0.0;
    }
    let t = TABLE.get_or_init(|| {
        let h = 1.0 / RHO_STEPS as f64;
        let len = (RHO_MAX as usize) * RHO_STEPS + 1;
        let mut r = vec![1.0; len];
        for i in RHO_STEPS..len - 1 {
            let (u0, u1) = (i as f64 * h, (i + 1) as f64 * h);
            let f0 = r[i - RHO_STEPS] / u0;
            let f1 = r[i + 1 - RHO_STEPS] / u1;
            r[i + 1] = (r[i] - h / 2.0 * (f0 + f1)).max(0.0);
        }
        r
    });
    let x = u * RHO_STEPS as f64;
    let i = x as usize;
    let f = x - i as f64;
    t[i] * (1.0 - f) + t[i + 1] * f
}

/// The chance that a random integer near e^ln_x is B1 smooth apart from
/// at most one prime factor in (B1, B2].
pub fn smooth(ln_x: f64, b1: u64, b2: u64) -> f64 {
    if ln_x <= 0.0 {
        return 1.0;
    }
    let l1 = (b1 as f64).ln();
    let mut p = rho(ln_x / l1);
    if b2 > b1 {
        // Sum over primes t in (B1, B2] of rho((ln x - ln t) / ln B1) / t,
        // with primes of density 1 / ln t: in s = ln t, the integrand is
        // rho((ln x - s) / ln B1) / s.
        let l2 = (b2 as f64).ln();
        let steps = 64;
        let h = (l2 - l1) / steps as f64;
        let g = |s: f64| rho((ln_x - s) / l1) / s;
        let mut sum = (g(l1) + g(l2)) / 2.0;
        for i in 1..steps {
            sum += g(l1 + i as f64 * h);
        }
        p += sum * h;
    }
    p.min(1.0)
}

/// The chance that P-1 with bounds (b1, b2) finds a factor of 2^p - 1
/// that has none below 2^tf_bits.
pub fn pm1_probability(p: u32, tf_bits: f64, b1: u64, b2: u64) -> f64 {
    let ln2 = std::f64::consts::LN_2;
    let ln_2p = (2.0 * f64::from(p)).ln();
    let mut total = 0.0;
    let mut k = tf_bits.max(1.0).floor();
    loop {
        // m = (q - 1) / 2p for q near the middle of the bit level.
        let ln_m = (k + 0.5) * ln2 - ln_2p;
        let s = smooth(ln_m, b1, b2);
        if s < 1e-12 && ln_m > 0.0 {
            break;
        }
        total += s / k;
        k += 1.0;
    }
    total
}

/// Approximate count of primes in (a, b].
fn primes_between(a: u64, b: u64) -> f64 {
    let pi = |x: u64| {
        let x = x as f64;
        if x < 3.0 {
            0.0
        } else {
            x / (x.ln() - 1.0)
        }
    };
    (pi(b) - pi(a)).max(0.0)
}

/// P-1 bounds chosen by [`pm1_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Pm1Plan {
    pub b1: u64,
    pub b2: u64,
    /// Chance of a factor.
    pub probability: f64,
    /// Cost in squarings modulo 2^p - 1.
    pub cost: f64,
    /// probability * tests_saved * p - cost, in squarings.
    pub gain: f64,
}

/// The P-1 bounds that maximize the expected saving: B1 on a geometric
/// grid from 1,000, and B2 a multiple of B1 (B2 = B1 when `stage2_cost` is
/// None, as for a backend where stage 2 is not run). B2 stays below 2^31 so
/// the paired stage 2 applies.
pub fn pm1_plan(p: u32, tf_bits: f64, tests_saved: f64, stage2_cost: Option<f64>) -> Pm1Plan {
    let test = tests_saved * f64::from(p);
    let multiples: &[u64] = if stage2_cost.is_some() {
        &[1, 10, 20, 40, 80, 160]
    } else {
        &[1]
    };
    let mut best: Option<Pm1Plan> = None;
    let mut b1 = 1000.0f64;
    while b1 <= 1e9 && 1.44 * b1 <= test.max(1.0) {
        let b1u = b1 as u64;
        for &m in multiples {
            let b2 = b1u.saturating_mul(m);
            if b2 >= 1 << 31 {
                continue;
            }
            let cost = 1.44 * b1 + stage2_cost.unwrap_or(0.0) * primes_between(b1u, b2);
            let probability = pm1_probability(p, tf_bits, b1u, b2);
            let gain = probability * test - cost;
            if best.is_none_or(|b| gain > b.gain) {
                best = Some(Pm1Plan {
                    b1: b1u,
                    b2,
                    probability,
                    cost,
                    gain,
                });
            }
        }
        b1 *= 1.25;
    }
    best.unwrap_or(Pm1Plan {
        b1: 1000,
        b2: 1000,
        probability: pm1_probability(p, tf_bits, 1000, 1000),
        cost: 1440.0,
        gain: -1440.0,
    })
}

/// Fraction of k values whose candidate q = 2kp + 1 survives the class
/// split (960 of 4620) and the sieve by the primes 13 to 2^16 (Mertens:
/// e^-gamma / ln 2^16 over the product for the primes up to 11).
pub const TF_SURVIVORS: f64 = (960.0 / 4620.0) * 0.2436;

/// The bit level trial factoring should reach: every level k from
/// `from_bits` on whose chance of a factor (1 / k) times the tests saved
/// outweighs its candidate tests, each worth `tf_cost` squarings.
pub fn tf_target_bits(p: u32, from_bits: u32, tests_saved: f64, tf_cost: f64) -> u32 {
    let test = tests_saved * f64::from(p);
    let mut k = from_bits;
    while k < 120 {
        let candidates = 2f64.powi(k as i32) / (2.0 * f64::from(p)) * TF_SURVIVORS;
        if test / f64::from(k) <= candidates * tf_cost {
            break;
        }
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_matches_known_values() {
        // rho(2) = 1 - ln 2; rho(3), rho(4), rho(5) from the literature.
        for (u, want) in [
            (1.5, 1.0 - 1.5f64.ln()),
            (2.0, 1.0 - 2f64.ln()),
            (3.0, 0.048_608_3),
            (4.0, 0.004_910_9),
            (5.0, 0.000_354_7),
        ] {
            let got = rho(u);
            assert!(
                (got - want).abs() / want < 1e-3,
                "rho({u}) = {got}, want {want}"
            );
        }
        assert_eq!(rho(0.5), 1.0);
        assert_eq!(rho(30.0), 0.0);
    }

    #[test]
    fn plans_follow_the_saving() {
        let p = 100_000_000;
        // Deeper trial factoring leaves fewer factors to find.
        assert!(
            pm1_probability(p, 77.0, 1_000_000, 1_000_000)
                < pm1_probability(p, 70.0, 1_000_000, 1_000_000)
        );
        // Larger bounds find more.
        assert!(
            pm1_probability(p, 77.0, 100_000, 100_000)
                < pm1_probability(p, 77.0, 1_000_000, 30_000_000)
        );
        // A bigger saving buys at least as large a B1, and a stage 2 never
        // lowers the gain.
        let one = pm1_plan(p, 77.0, 1.0, None);
        let two = pm1_plan(p, 77.0, 2.0, None);
        assert!(two.b1 >= one.b1 && two.gain > one.gain, "{one:?} {two:?}");
        assert_eq!(two.b2, two.b1);
        let s2 = pm1_plan(p, 77.0, 2.0, Some(1.0));
        assert!(s2.gain >= two.gain && s2.b2 > s2.b1, "{s2:?}");
        // Wavefront P-1 finds a factor a few percent of the time.
        assert!((0.01..0.1).contains(&s2.probability), "{s2:?}");
        // Trial factoring stops where a level costs more than it saves,
        // and cheaper candidates go deeper.
        let a = tf_target_bits(p, 60, 2.0, 1e-3);
        let b = tf_target_bits(p, 60, 2.0, 1e-5);
        assert!(a > 60 && b > a, "{a} {b}");
    }
}
