// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The canonical residue modulo N = 2^p - 1.
//!
//! Representation: exponent `p`, exactly `ceil(p / 32)` little-endian `u32`
//! limbs, value `0 <= x < N`. Bits at positions `>= p` are zero. Zero has a
//! unique representation: the all-ones pattern (the value N, which is
//! congruent to zero) is never stored.
//!
//! Byte order for files is little-endian throughout (limb 0 first, each limb
//! least significant byte first).

use num_bigint::BigUint;
use num_traits::Zero;

/// Why a limb vector is not a canonical residue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidueError {
    /// p is zero or 1, which defines no useful modulus.
    BadExponent(u32),
    /// Wrong number of limbs for p.
    Length { expected: usize, found: usize },
    /// A bit at position >= p is set.
    HighBits,
    /// The value is N itself (all ones); the canonical zero is all zeros.
    ValueIsModulus,
    /// The exponents of two operands differ.
    ExponentMismatch { expected: u32, found: u32 },
}

impl std::fmt::Display for ResidueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidueError::BadExponent(p) => write!(f, "exponent {p} defines no modulus"),
            ResidueError::Length { expected, found } => {
                write!(f, "residue has {found} limbs, expected {expected}")
            }
            ResidueError::HighBits => write!(f, "residue has bits set at or above bit p"),
            ResidueError::ValueIsModulus => {
                write!(f, "residue equals N; the canonical zero is all zeros")
            }
            ResidueError::ExponentMismatch { expected, found } => {
                write!(f, "residue exponent {found} does not match {expected}")
            }
        }
    }
}

impl std::error::Error for ResidueError {}

/// Number of 32-bit limbs of a residue modulo 2^p - 1.
pub fn limb_count(p: u32) -> usize {
    p.div_ceil(32) as usize
}

/// Number of bytes of a residue in the PRP proof file format: (p - 1) / 8 + 1.
pub fn proof_byte_len(p: u32) -> usize {
    ((p - 1) / 8 + 1) as usize
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Residue {
    p: u32,
    limbs: Vec<u32>,
}

impl std::fmt::Debug for Residue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Residue(p={}, res64={:016x})", self.p, self.res64())
    }
}

fn top_mask(p: u32) -> u32 {
    match p % 32 {
        0 => u32::MAX,
        r => (1u32 << r) - 1,
    }
}

fn check_shape(p: u32, limbs: &[u32]) -> Result<(), ResidueError> {
    if p < 2 {
        return Err(ResidueError::BadExponent(p));
    }
    let n = limb_count(p);
    if limbs.len() != n {
        return Err(ResidueError::Length {
            expected: n,
            found: limbs.len(),
        });
    }
    if limbs[n - 1] & !top_mask(p) != 0 {
        return Err(ResidueError::HighBits);
    }
    Ok(())
}

fn is_all_ones(p: u32, limbs: &[u32]) -> bool {
    let n = limbs.len();
    limbs[..n - 1].iter().all(|&w| w == u32::MAX) && limbs[n - 1] == top_mask(p)
}

impl Residue {
    pub fn exponent(&self) -> u32 {
        self.p
    }

    pub fn limbs(&self) -> &[u32] {
        &self.limbs
    }

    /// Strict: rejects wrong length, set bits above bit p-1, and the value N.
    pub fn from_limbs(p: u32, limbs: Vec<u32>) -> Result<Residue, ResidueError> {
        check_shape(p, &limbs)?;
        if is_all_ones(p, &limbs) {
            return Err(ResidueError::ValueIsModulus);
        }
        Ok(Residue { p, limbs })
    }

    /// Lenient boundary for backend exports: wrong length or bits above p-1
    /// are still errors, but the value N (all ones) is normalized to zero.
    pub fn from_limbs_normalizing(p: u32, limbs: Vec<u32>) -> Result<Residue, ResidueError> {
        check_shape(p, &limbs)?;
        if is_all_ones(p, &limbs) {
            return Ok(Residue::zero(p));
        }
        Ok(Residue { p, limbs })
    }

    pub fn zero(p: u32) -> Residue {
        assert!(p >= 2, "exponent {p} defines no modulus");
        Residue {
            p,
            limbs: vec![0; limb_count(p)],
        }
    }

    /// The residue of a small integer v, reduced modulo N.
    pub fn from_u64(p: u32, v: u64) -> Residue {
        Residue::from_biguint(p, &BigUint::from(v))
    }

    /// Reduce any nonnegative integer modulo N into canonical form.
    pub fn from_biguint(p: u32, v: &BigUint) -> Residue {
        assert!(p >= 2, "exponent {p} defines no modulus");
        let n = (BigUint::from(1u32) << p) - 1u32;
        let r = v % &n;
        let mut limbs = r.to_u32_digits();
        limbs.resize(limb_count(p), 0);
        Residue { p, limbs }
    }

    pub fn to_biguint(&self) -> BigUint {
        BigUint::from_slice(&self.limbs)
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&w| w == 0)
    }

    /// The low 64 bits of the canonical value.
    pub fn res64(&self) -> u64 {
        let lo = self.limbs[0] as u64;
        let hi = self.limbs.get(1).copied().unwrap_or(0) as u64;
        lo | (hi << 32)
    }

    /// The low 2048 bits as hex, most significant first (as upstream reports).
    pub fn res2048_hex(&self) -> String {
        let mut words: Vec<u32> = self.limbs.iter().take(64).copied().collect();
        words.resize(64, 0);
        words.iter().rev().map(|w| format!("{w:08x}")).collect()
    }

    /// Little-endian bytes, `4 * ceil(p / 32)` of them.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.limbs.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Strict inverse of `to_le_bytes`.
    pub fn from_le_bytes(p: u32, bytes: &[u8]) -> Result<Residue, ResidueError> {
        if p < 2 {
            return Err(ResidueError::BadExponent(p));
        }
        if bytes.len() != 4 * limb_count(p) {
            return Err(ResidueError::Length {
                expected: limb_count(p),
                found: bytes.len() / 4,
            });
        }
        let limbs = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        Residue::from_limbs(p, limbs)
    }

    /// The (p - 1) / 8 + 1 little-endian bytes used by the PRP proof format.
    pub fn to_proof_bytes(&self) -> Vec<u8> {
        let mut b = self.to_le_bytes();
        b.truncate(proof_byte_len(self.p));
        b
    }

    /// Parse proof-format bytes. Bits at or above p are rejected. The value N
    /// is normalized to zero (it is the same element modulo N).
    pub fn from_proof_bytes(p: u32, bytes: &[u8]) -> Result<Residue, ResidueError> {
        if p < 2 {
            return Err(ResidueError::BadExponent(p));
        }
        if bytes.len() != proof_byte_len(p) {
            return Err(ResidueError::Length {
                expected: proof_byte_len(p),
                found: bytes.len(),
            });
        }
        let mut padded = bytes.to_vec();
        padded.resize(4 * limb_count(p), 0);
        let limbs = padded
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        Residue::from_limbs_normalizing(p, limbs)
    }

    /// x - v mod N.
    pub fn minus_small(&self, v: u64) -> Residue {
        let n = modulus(self.p);
        let v = BigUint::from(v) % &n;
        let x = self.to_biguint();
        let r = if x >= v { x - v } else { x + &n - v };
        Residue::from_biguint(self.p, &r)
    }

    /// x + v mod N.
    pub fn add_small(&self, v: u64) -> Residue {
        Residue::from_biguint(self.p, &(self.to_biguint() + BigUint::from(v)))
    }

    /// x * v mod N for a small v.
    pub fn mul_small(&self, v: u64) -> Residue {
        Residue::from_biguint(self.p, &(self.to_biguint() * BigUint::from(v)))
    }

    /// x / 3 mod N. 3 does not divide N for odd p (2^p = 2 mod 3), so for odd
    /// p this is the unique y with 3y = x mod N: pick r in {0, 1, 2} with
    /// x + rN divisible by 3. For even p (only p = 2 is prime) 3 divides N and
    /// division is undefined, so this panics.
    pub fn div3(&self) -> Residue {
        assert!(
            self.p % 2 == 1,
            "division by 3 is undefined modulo 2^{} - 1",
            self.p
        );
        let n = modulus(self.p);
        let x = self.to_biguint();
        let three = BigUint::from(3u32);
        let mut t = x;
        while !(&t % &three).is_zero() {
            t += &n;
        }
        Residue::from_biguint(self.p, &(t / three))
    }

    /// Flip bit `bit % p`, normalizing the value N to zero. For tests.
    pub fn with_bit_flipped(&self, bit: u32) -> Residue {
        let b = bit % self.p;
        let mut limbs = self.limbs.clone();
        limbs[(b / 32) as usize] ^= 1 << (b % 32);
        Residue::from_limbs_normalizing(self.p, limbs).expect("bit below p keeps the shape")
    }
}

/// N = 2^p - 1 as a big integer.
pub fn modulus(p: u32) -> BigUint {
    (BigUint::from(1u32) << p) - 1u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_noncanonical() {
        assert_eq!(
            Residue::from_limbs(31, vec![0x7fff_ffff]),
            Err(ResidueError::ValueIsModulus)
        );
        assert_eq!(
            Residue::from_limbs(31, vec![0x8000_0000]),
            Err(ResidueError::HighBits)
        );
        assert_eq!(
            Residue::from_limbs(33, vec![0]),
            Err(ResidueError::Length {
                expected: 2,
                found: 1
            })
        );
        assert!(Residue::from_limbs_normalizing(33, vec![u32::MAX, 1])
            .unwrap()
            .is_zero());
        assert_eq!(
            Residue::from_limbs(1, vec![0]),
            Err(ResidueError::BadExponent(1))
        );
    }

    #[test]
    fn div3_inverts_mul3() {
        for p in [3u32, 5, 7, 13, 31, 61, 89, 127] {
            for v in [0u64, 1, 2, 9, 12345] {
                let x = Residue::from_u64(p, v);
                assert_eq!(x.mul_small(3).div3(), x, "p={p} v={v}");
            }
        }
        assert_eq!(
            Residue::from_u64(61, 9).div3().div3(),
            Residue::from_u64(61, 1)
        );
    }

    #[test]
    fn proof_bytes_round_trip() {
        for p in [7u32, 8 + 1, 31, 32 + 1, 61, 127, 521] {
            let x = Residue::from_biguint(p, &(modulus(p) - 5u32));
            let b = x.to_proof_bytes();
            assert_eq!(b.len(), proof_byte_len(p));
            assert_eq!(Residue::from_proof_bytes(p, &b).unwrap(), x);
        }
    }
}
