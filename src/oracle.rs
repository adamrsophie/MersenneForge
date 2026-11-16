// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Independent CPU oracle and reference tests.
//!
//! Straightforward exact arithmetic on `num_bigint::BigUint` with generic `%`
//! reduction by N. It shares no reduction code with any backend: the CPU
//! backend reduces by Mersenne shift and add folding, the transform backends
//! by carry propagation. The PRP reference additionally uses
//! `BigUint::modpow`, whose odd-modulus path is Montgomery multiplication,
//! a third reduction method.
//!
//! The oracle is bounded only by time; use it for exponents up to a few
//! thousand in debug builds.

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::residue::{modulus, Residue};

/// a * b mod N, by generic division.
pub fn mul(a: &Residue, b: &Residue) -> Residue {
    assert_eq!(a.exponent(), b.exponent());
    let n = modulus(a.exponent());
    Residue::from_biguint(a.exponent(), &((a.to_biguint() * b.to_biguint()) % n))
}

/// x^(2^k) mod N by k plain squarings with `%`.
pub fn square_n(x: &Residue, k: u64) -> Residue {
    let n = modulus(x.exponent());
    let mut v = x.to_biguint();
    for _ in 0..k {
        v = (&v * &v) % &n;
    }
    Residue::from_biguint(x.exponent(), &v)
}

/// k Lucas-Lehmer steps s <- s^2 - 2 mod N.
pub fn ll_steps(s: &Residue, k: u64) -> Residue {
    let n = modulus(s.exponent());
    let two = BigUint::from(2u32);
    let mut v = s.to_biguint();
    for _ in 0..k {
        v = (&v * &v + &n - &two) % &n;
    }
    Residue::from_biguint(s.exponent(), &v)
}

/// Outcome of a reference test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// LL: M_p is prime. PRP: M_p is a base-3 probable prime.
    pub positive: bool,
    /// The full terminal residue: s_(p-2) for LL, 3^(2^p) for PRP.
    pub terminal: Residue,
}

/// Reference Lucas-Lehmer test for a prime exponent p.
///
/// p = 2 is handled separately: M_2 = 3 is prime, and the recurrence would
/// have zero steps from s = 4 = 1 mod 3. For odd p: s = 4, p - 2 steps,
/// M_p is prime iff the terminal residue is zero.
pub fn ll(p: u32) -> Reference {
    assert!(
        crate::exponent::is_prime_u32(p),
        "LL needs a prime exponent"
    );
    if p == 2 {
        return Reference {
            positive: true,
            terminal: Residue::zero(2),
        };
    }
    let s = ll_steps(&Residue::from_u64(p, 4), u64::from(p) - 2);
    Reference {
        positive: s.is_zero(),
        terminal: s,
    }
}

/// Reference base-3 PRP test, upstream semantics: x_0 = 3, p squarings,
/// probable prime iff x_p = 9 mod N (that is, 3^(N+1) = 3^2).
///
/// p = 2: N = 3 shares the base, so base 3 says nothing; M_2 = 3 is reported
/// positive because it is prime by inspection.
pub fn prp(p: u32) -> Reference {
    assert!(
        crate::exponent::is_prime_u32(p),
        "PRP needs a prime exponent"
    );
    if p == 2 {
        return Reference {
            positive: true,
            terminal: Residue::zero(2),
        };
    }
    let x = square_n(&Residue::from_u64(p, 3), u64::from(p));
    Reference {
        positive: x == Residue::from_u64(p, 9),
        terminal: x,
    }
}

/// The same PRP terminal residue by `modpow` (Montgomery reduction).
pub fn prp_terminal_modpow(p: u32) -> Residue {
    let n = modulus(p);
    let e = BigUint::one() << p;
    Residue::from_biguint(p, &BigUint::from(3u32).modpow(&e, &n))
}

/// Fermat test a^(N-1) = 1 mod N. Only for the regression that shows base 2
/// is useless for Mersenne numbers: 2^(N-1) = 1 mod N for every prime p,
/// because 2^p = 1 mod N and p divides N - 1 = 2^p - 2.
pub fn fermat(p: u32, base: u32) -> bool {
    let n = modulus(p);
    let r = BigUint::from(base).modpow(&(&n - 1u32), &n);
    r.is_one() && !n.is_zero()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base2_fermat_is_useless_for_mersenne_numbers() {
        // N = 2047 = 23 * 89 is composite but passes base 2.
        assert_eq!(modulus(11), BigUint::from(23u32 * 89));
        assert!(fermat(11, 2), "base 2 passes the composite 2047");
        assert!(!fermat(11, 3), "base 3 detects it");
        assert!(!prp(11).positive && !ll(11).positive);
    }
}
