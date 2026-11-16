// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Exponent domain validation.
//!
//! `is_prime_u32` is deterministic trial division by 2 and by every odd d with
//! d * d <= n. Every composite n has a prime factor at most sqrt(n), and for
//! n <= u32::MAX that bound is below 65536, so the routine is a complete
//! proof of primality for the whole `u32` range. It needs no probabilistic
//! bases and no unproven range claims; its worst case is about 32768
//! divisions, which is negligible next to any primality test of 2^p - 1.

/// Exact primality of a 32-bit integer by trial division (proven for all u32).
pub fn is_prime_u32(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let n = n as u64;
    let mut d = 3u64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExponentError {
    /// The LL and PRP criteria used here apply only to prime p.
    NotPrime(u32),
}

impl std::fmt::Display for ExponentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExponentError::NotPrime(p) => write!(
                f,
                "exponent {p} is not prime; 2^p - 1 is then trivially composite and the LL and PRP criteria here do not apply"
            ),
        }
    }
}

impl std::error::Error for ExponentError {}

/// Accept only prime exponents.
pub fn validate(p: u32) -> Result<u32, ExponentError> {
    if is_prime_u32(p) {
        Ok(p)
    } else {
        Err(ExponentError::NotPrime(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_and_boundary_values() {
        let primes: Vec<u32> = (0..60).filter(|&n| is_prime_u32(n)).collect();
        assert_eq!(
            primes,
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        );
        assert!(is_prime_u32(4_294_967_291)); // largest u32 prime
        assert!(!is_prime_u32(u32::MAX)); // 3 * 5 * 17 * 257 * 65537
        assert!(!is_prime_u32(65521 * 65521)); // square of the largest 16-bit prime
        assert!(!is_prime_u32(3_215_031_751)); // strong pseudoprime to bases 2, 3, 5, 7
        assert!(validate(1).is_err() && validate(9).is_err() && validate(127).is_ok());
    }

    #[test]
    fn agrees_with_sieve() {
        let n = 100_000usize;
        let mut sieve = vec![true; n];
        sieve[0] = false;
        sieve[1] = false;
        for i in 2..n {
            if sieve[i] {
                for j in (i * i..n).step_by(i) {
                    sieve[j] = false;
                }
            }
        }
        for (i, &s) in sieve.iter().enumerate() {
            assert_eq!(is_prime_u32(i as u32), s, "{i}");
        }
    }
}
