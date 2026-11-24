// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Factoring of 2^p - 1: trial factoring (TF) and P-1, with the worktodo
//! lines that request them (`Factor=` and `Pminus1=` / `Pfactor=`) and the
//! JSON result lines upstream tools write for them.
//!
//! Every factor either method reports is checked by
//! [`crate::worktodo::check_factor`] before it is returned, so an
//! arithmetic error can make a run miss a factor but never report a false
//! one. Neither method has an in-run error check (TF has none upstream
//! either; P-1 stage 1 here has no Gerbicz-style check).
//!
//! Limits (deliberate, see the `ponytail:` notes): TF runs on the CPU, with
//! a fast path for factors below 2^64 and a slow arbitrary-precision path
//! above it; P-1 runs both stages on any backend, but stage 2 takes one
//! multiply and one host round trip per prime (no prime pairing) and the
//! final gcd runs on the host, so it suits small and moderate exponents.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use serde::Serialize;

use crate::backend::Backend;
use crate::residue::{modulus, Residue};

/// Primes below `n`, by the sieve of Eratosthenes.
pub fn primes_below(n: u64) -> Vec<u64> {
    let n = n as usize;
    if n < 3 {
        return Vec::new();
    }
    let mut composite = vec![false; n];
    let mut out = Vec::new();
    for i in 2..n {
        if !composite[i] {
            out.push(i as u64);
            let mut j = i * i;
            while j < n {
                composite[j] = true;
                j += i;
            }
        }
    }
    out
}

/// 2^p mod q, for q < 2^64.
fn pow2_mod(p: u32, q: u64) -> u64 {
    let mut r: u128 = 1;
    let q = u128::from(q);
    for i in (0..32 - p.leading_zeros()).rev() {
        r = r * r % q;
        if p >> i & 1 == 1 {
            r = (r << 1) % q;
        }
    }
    r as u64
}

/// Outcome of trial factoring one range.
#[derive(Debug, Clone, Serialize)]
pub struct TfOutcome {
    pub exponent: u32,
    pub bit_lo: u32,
    pub bit_hi: u32,
    /// Factors found, checked, in increasing order.
    pub factors: Vec<String>,
    /// Every candidate in the range was tested.
    pub complete: bool,
    /// First k not yet tested (q = 2kp + 1), for resuming an incomplete run.
    pub next_k: u128,
}

/// Sieve segment length in k values.
const SEGMENT: usize = 1 << 16;
/// Sieve primes are below this bound.
const SIEVE_BOUND: u64 = 1 << 16;
/// mfaktc's class count, 4 * 3 * 5 * 7 * 11: k = c + 4620 i keeps q mod 8
/// and q mod 3, 5, 7, 11 fixed within a class, so only the 960 classes
/// whose q can divide 2^p - 1 are sieved at all.
pub const TF_CLASSES: u32 = 4620;
/// Primes the class split removes.
const CLASS_PRIMES: [u64; 4] = [3, 5, 7, 11];

/// Trial factor 2^p - 1 by every q = 2kp + 1 with 2^bit_lo <= q < 2^bit_hi,
/// starting at k = `from_k` when resuming. `stop` is polled once per segment;
/// `progress` receives the next untested k after each segment.
///
/// Candidates divisible by a prime below 2^16, or not +-1 mod 8, are skipped:
/// no prime divisor of 2^p - 1 has either property, and a composite divisor
/// is found through its prime factors.
pub fn trial_factor(
    p: u32,
    bit_lo: u32,
    bit_hi: u32,
    from_k: Option<u128>,
    stop: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(u128),
) -> Result<TfOutcome, String> {
    trial_factor_with(p, bit_lo, bit_hi, from_k, stop, progress, &mut |p, qs| {
        Ok(qs.iter().copied().filter(|&q| divides(p, q)).collect())
    })
}

/// Whether q divides 2^p - 1, on the host.
pub fn divides(p: u32, q: u128) -> bool {
    match u64::try_from(q) {
        Ok(q64) => pow2_mod(p, q64) == 1,
        // ponytail: arbitrary precision above 2^64 is slow; a two-limb
        // Montgomery multiply is the upgrade path.
        Err(_) => {
            let qb = BigUint::from(q);
            BigUint::from(2u32).modpow(&BigUint::from(p), &qb) == BigUint::one()
        }
    }
}

/// Batch tester for [`trial_factor_with`]: the candidates of one segment
/// in, the ones for which 2^p = 1 mod q out.
pub type TfTester<'a> = dyn FnMut(u32, &[u128]) -> Result<Vec<u128>, String> + 'a;

/// One sieve segment: k0 + i stride for i < len, and for each sieve prime
/// s the first i with s | 2 (k0 + i stride) p + 1 (every s-th i after it
/// also has it). The stride is 1, or [`TF_CLASSES`] for one class.
pub struct TfSegment<'a> {
    pub k0: u128,
    pub len: usize,
    pub stride: u32,
    /// (s, first index); the index may be at or past `len`.
    pub roots: &'a [(u32, u32)],
}

impl TfSegment<'_> {
    /// The k of index i.
    pub fn k(&self, i: usize) -> u128 {
        self.k0 + i as u128 * u128::from(self.stride)
    }

    /// One past the last k of the segment (k0 when it is empty).
    pub fn k_end(&self) -> u128 {
        if self.len == 0 {
            self.k0
        } else {
            self.k(self.len - 1) + 1
        }
    }

    /// The index of k, when k is in the segment.
    pub fn index(&self, k: u128) -> Option<usize> {
        let s = u128::from(self.stride);
        (k >= self.k0 && (k - self.k0) % s == 0)
            .then(|| (k - self.k0) / s)
            .filter(|&i| i < self.len as u128)
            .map(|i| i as usize)
    }

    /// The candidates q = 2kp + 1 of the segment that survive the sieve and
    /// are +-1 mod 8, in increasing order.
    pub fn candidates(&self, p: u32) -> Vec<u128> {
        let mut bad = vec![false; self.len];
        for &(s, first) in self.roots {
            let mut i = first as usize;
            while i < self.len {
                bad[i] = true;
                i += s as usize;
            }
        }
        let two_p = 2 * u128::from(p);
        bad.iter()
            .enumerate()
            .filter(|(_, b)| !**b)
            .map(|(i, _)| two_p * self.k(i) + 1)
            .filter(|q| matches!(q & 7, 1 | 7))
            .collect()
    }
}

/// Segment tester for [`trial_factor_segments`]: returns the q of the
/// segment's candidates for which 2^p = 1 mod q.
pub type TfSegmentTester<'a> = dyn FnMut(u32, &TfSegment<'_>) -> Result<Vec<u128>, String> + 'a;

/// [`trial_factor`] with the candidates of each sieved segment handed to
/// `test` in one batch (the GPU kernel, for example). Every q it returns is
/// checked again exactly before it is reported.
pub fn trial_factor_with(
    p: u32,
    bit_lo: u32,
    bit_hi: u32,
    from_k: Option<u128>,
    stop: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(u128),
    test: &mut TfTester<'_>,
) -> Result<TfOutcome, String> {
    trial_factor_segments(p, bit_lo, bit_hi, from_k, stop, progress, &mut |p, seg| {
        let cands = seg.candidates(p);
        let hits = test(p, &cands)?;
        if let Some(q) = hits.iter().find(|&&q| cands.binary_search(&q).is_err()) {
            return Err(format!("the tester returned {q}, which is not a candidate"));
        }
        Ok(hits)
    })
}

/// The segment driver under [`trial_factor_with`]: `test` gets each segment
/// with its sieve roots and may sieve and test it anywhere (the GPU, for
/// example). Every q it returns must be 2kp + 1 for a k in the segment and
/// is checked again exactly before it is reported.
pub fn trial_factor_segments(
    p: u32,
    bit_lo: u32,
    bit_hi: u32,
    from_k: Option<u128>,
    stop: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(u128),
    test: &mut TfSegmentTester<'_>,
) -> Result<TfOutcome, String> {
    crate::exponent::validate(p).map_err(|e| e.to_string())?;
    if !(bit_lo < bit_hi && bit_hi <= 120) {
        return Err(format!(
            "bad bit range {bit_lo}..{bit_hi} (need lo < hi <= 120)"
        ));
    }
    let two_p = 2 * u128::from(p);
    let k_of = |bits: u32| ((1u128 << bits) - 1).div_ceil(two_p);
    let (k_lo, k_end) = (k_of(bit_lo).max(1), k_of(bit_hi));
    let mut k0 = from_k.unwrap_or(k_lo).max(k_lo);
    // Sieving could strike a candidate equal to a sieve prime; such small
    // ranges are tested without the sieve.
    let sieve: Vec<(u64, u64)> = if two_p * k_lo + 1 > u128::from(SIEVE_BOUND) {
        primes_below(SIEVE_BOUND)
            .into_iter()
            .skip(1) // 2: every q is odd
            .filter(|&s| s != u64::from(p))
            .map(|s| {
                // q = 2kp + 1 = 0 mod s  <=>  k = -(2p)^-1 mod s.
                let inv =
                    mod_inverse((two_p % u128::from(s)) as u64, s).expect("s does not divide 2p");
                (s, (s - inv) % s)
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut factors = Vec::new();
    let mut roots = Vec::with_capacity(sieve.len());
    let mut run = |seg: &TfSegment<'_>, factors: &mut Vec<String>| -> Result<(), String> {
        let mut hits = test(p, seg)?;
        hits.sort_unstable();
        hits.dedup();
        for q in hits {
            if q % two_p != 1 || seg.index(q / two_p).is_none() {
                return Err(format!(
                    "the tester returned {q}, which is not in the segment"
                ));
            }
            let f = q.to_string();
            let c = crate::worktodo::check_factor(p, &f)?;
            if !c.accepted() {
                return Err(format!(
                    "{f} was reported as a factor but fails the factor check"
                ));
            }
            factors.push(f);
        }
        Ok(())
    };
    let stopped = |factors: Vec<String>, next_k| TfOutcome {
        exponent: p,
        bit_lo,
        bit_hi,
        factors,
        complete: false,
        next_k,
    };
    if sieve.is_empty() {
        while k0 < k_end {
            if stop() {
                return Ok(stopped(factors, k0));
            }
            let len = ((k_end - k0).min(SEGMENT as u128)) as usize;
            let seg = TfSegment {
                k0,
                len,
                stride: 1,
                roots: &[],
            };
            run(&seg, &mut factors)?;
            k0 += len as u128;
            progress(k0);
        }
    } else {
        // Blocks of TF_CLASSES * SEGMENT k values, each split into its
        // classes (as mfaktc does); progress moves a whole block at a time,
        // and a run stopped inside a block resumes at its start.
        let classes = u128::from(TF_CLASSES);
        // (s, root, 1 / TF_CLASSES mod s) for the primes the classes leave.
        let sieve: Vec<(u64, u64, u64)> = sieve
            .iter()
            .filter(|(s, _)| !CLASS_PRIMES.contains(s))
            .map(|&(s, root)| {
                let inv = mod_inverse(u64::from(TF_CLASSES) % s, s).expect("s > 11");
                (s, root, inv)
            })
            .collect();
        while k0 < k_end {
            let span = (k_end - k0).min(classes * SEGMENT as u128);
            for c in 0..classes.min(span) {
                if stop() {
                    return Ok(stopped(factors, k0));
                }
                let kc = k0 + c;
                let q = two_p * kc + 1;
                if !matches!(q & 7, 1 | 7) || CLASS_PRIMES.iter().any(|&s| q % u128::from(s) == 0) {
                    continue;
                }
                let len = (span - c).div_ceil(classes) as usize;
                roots.clear();
                roots.extend(sieve.iter().map(|&(s, root, inv)| {
                    let s128 = u128::from(s);
                    // kc + i C = root mod s  <=>  i = (root - kc) / C mod s.
                    let d = (u128::from(root) + s128 - kc % s128) % s128;
                    (s as u32, (d * u128::from(inv) % s128) as u32)
                }));
                let seg = TfSegment {
                    k0: kc,
                    len,
                    stride: TF_CLASSES,
                    roots: &roots,
                };
                run(&seg, &mut factors)?;
            }
            k0 += span;
            progress(k0);
        }
    }
    factors.sort_by_key(|f| f.parse::<u128>().unwrap_or(u128::MAX));
    Ok(TfOutcome {
        exponent: p,
        bit_lo,
        bit_hi,
        factors,
        complete: true,
        next_k: k_end,
    })
}

/// a^-1 mod m, for gcd(a, m) = 1.
fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let (mut t, mut new_t, mut r, mut new_r) = (0i128, 1i128, i128::from(m), i128::from(a));
    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }
    (r == 1).then(|| t.rem_euclid(i128::from(m)) as u64)
}

/// Outcome of a P-1 run.
#[derive(Debug, Clone, Serialize)]
pub struct Pm1Outcome {
    pub exponent: u32,
    pub b1: u64,
    pub b2: u64,
    /// A checked proper factor, when one was found (it may be composite).
    pub factor: Option<String>,
    /// The stage that found it (1 or 2).
    pub stage: Option<u8>,
    /// Both stages ran to their bounds.
    pub complete: bool,
    pub notes: Vec<String>,
}

/// E = 2p * prod over primes q <= b1 of q^floor(log_q b1).
pub fn stage1_exponent(p: u32, b1: u64) -> BigUint {
    let mut e = BigUint::from(2u64 * u64::from(p));
    for q in primes_below(b1 + 1) {
        let mut qk = q;
        while let Some(n) = qk.checked_mul(q).filter(|&n| n <= b1) {
            qk = n;
        }
        e *= qk;
    }
    e
}

/// A proper factor of N from gcd(x - 1, N), checked.
fn factor_from(
    p: u32,
    x_minus_1: &BigUint,
    n: &BigUint,
    notes: &mut Vec<String>,
) -> Option<String> {
    let g = x_minus_1.gcd(n);
    if g.is_one() || g.is_zero() {
        return None;
    }
    if &g == n {
        notes.push("every factor appeared at once (gcd = N); try a smaller bound".into());
        return None;
    }
    let f = g.to_string();
    match crate::worktodo::check_factor(p, &f) {
        Ok(c) if c.accepted() => Some(f),
        _ => {
            notes.push(format!(
                "gcd {f} failed the factor check; an arithmetic error is likely"
            ));
            None
        }
    }
}

/// P-1 with bounds b1 and b2 (b2 <= b1 skips stage 2). Stage 1 computes
/// x = 3^E mod N on `be`; stage 2 multiplies (x^q - 1) over primes
/// b1 < q <= b2, also on `be`. `stop` is polled about every 10,000 stage 1
/// bits and every 1,000 stage 2 primes; a stopped run reports complete =
/// false and must start over.
pub fn pm1(
    be: &mut dyn Backend,
    b1: u64,
    b2: u64,
    stop: &dyn Fn() -> bool,
) -> Result<Pm1Outcome, String> {
    pm1_resumable(be, b1, b2, None, stop, &mut |_, _| {})
}

/// dst = base^e on the backend (e >= 1), by left to right square and
/// multiply; `scratch` is clobbered.
fn pow_small(
    be: &mut dyn Backend,
    dst: crate::backend::Reg,
    base: crate::backend::Reg,
    e: u64,
    scratch: crate::backend::Reg,
) -> Result<(), String> {
    let berr = |e: crate::backend::BackendError| e.to_string();
    if e == 0 {
        return Err("exponent 0".into());
    }
    be.copy(scratch, base).map_err(berr)?;
    be.copy(dst, base).map_err(berr)?;
    for i in (0..63 - e.leading_zeros()).rev() {
        be.square(dst, 1).map_err(berr)?;
        if e >> i & 1 == 1 {
            be.mul(dst, scratch).map_err(berr)?;
        }
    }
    Ok(())
}

/// Giant step width of the paired stage 2: 2 * 3 * 5 * 7.
const PAIR_D: u64 = 210;

/// Stage 2 with prime pairing (Montgomery's baby step giant step with the
/// Brent-Suyama polynomial f(t) = t^2). Every prime q in `primes` is written
/// q = m D +- j with D = 210 and j < D / 2 coprime to D, and acc collects
/// x^((mD)^2) - x^(j^2) = x^(j^2) (x^((mD - j)(mD + j)) - 1), which is 0
/// modulo any factor where the order of x divides mD - j or mD + j. So one
/// term covers both primes of a pair, and no inverse is needed. Per term:
/// one copy, one subtraction, one multiplication; per giant step two
/// multiplications. The caller ensures every prime factor of D is at most
/// B1 and (B2 + D)^2 fits in a u64. Returns the product, or None when
/// stopped. ponytail: D is fixed at 210 (24 baby registers); a larger D
/// saves giant steps for large B2 at the cost of more resident registers.
fn stage2_paired(
    be: &mut dyn Backend,
    x: crate::backend::Reg,
    primes: &[u64],
    stop: &dyn Fn() -> bool,
) -> Result<Option<BigUint>, String> {
    let berr = |e: crate::backend::BackendError| e.to_string();
    let p = be.exponent();
    let d = PAIR_D;
    let (Some(&lo), Some(&hi)) = (primes.first(), primes.last()) else {
        return Ok(Some(BigUint::one()));
    };
    let is_prime = |q: u64| q >= lo && q <= hi && primes.binary_search(&q).is_ok();
    let js: Vec<u64> = (1..d / 2).filter(|j| j.gcd(&d) == 1).collect();
    let mut regs = Vec::new();
    let result = (|| -> Result<Option<BigUint>, String> {
        let mut new_reg = |be: &mut dyn Backend| -> Result<crate::backend::Reg, String> {
            let r = be.alloc().map_err(berr)?;
            regs.push(r);
            Ok(r)
        };
        let tmp = new_reg(be)?;
        // Baby steps: x^(j^2) for odd j, from x^((j+2)^2) = x^(j^2) x^(4j+4).
        let cur = new_reg(be)?;
        let inc = new_reg(be)?;
        let x8 = new_reg(be)?;
        be.copy(cur, x).map_err(berr)?;
        pow_small(be, x8, x, 8, tmp)?;
        be.copy(inc, x8).map_err(berr)?;
        let mut baby = std::collections::HashMap::new();
        let mut j = 1;
        while j <= *js.last().expect("D has units") {
            if js.contains(&j) {
                let r = new_reg(be)?;
                be.copy(r, cur).map_err(berr)?;
                baby.insert(j, r);
            }
            be.mul(cur, inc).map_err(berr)?;
            be.mul(inc, x8).map_err(berr)?;
            j += 2;
        }
        // Giant steps: G = x^((mD)^2), R = x^((2m + 1) D^2), S = x^(2 D^2).
        let m_lo = lo.saturating_sub(d / 2) / d;
        let m_hi = (hi + d / 2) / d;
        let (g, r, s) = (new_reg(be)?, new_reg(be)?, new_reg(be)?);
        if m_lo == 0 {
            be.import(g, &Residue::from_u64(p, 1)).map_err(berr)?;
        } else {
            pow_small(be, g, x, (m_lo * d) * (m_lo * d), tmp)?;
        }
        pow_small(be, r, x, (2 * m_lo + 1) * d * d, tmp)?;
        pow_small(be, s, x, 2 * d * d, tmp)?;
        let acc = new_reg(be)?;
        be.import(acc, &Residue::from_u64(p, 1)).map_err(berr)?;
        for m in m_lo..=m_hi {
            for &j in &js {
                let md = m * d;
                if is_prime(md + j) || (md > j && is_prime(md - j)) {
                    be.copy(tmp, g).map_err(berr)?;
                    be.sub(tmp, baby[&j]).map_err(berr)?;
                    be.mul(acc, tmp).map_err(berr)?;
                }
            }
            be.mul(g, r).map_err(berr)?;
            be.mul(r, s).map_err(berr)?;
            if m % 100 == 0 && stop() {
                return Ok(None);
            }
        }
        Ok(Some(be.export(acc).map_err(berr)?.to_biguint()))
    })();
    for r in regs {
        let _ = be.free(r);
    }
    result
}

/// Giant step width of the polynomial stage 2: 2 * 3 * 5 * 7 * 11, so 240
/// baby steps.
const POLY_D: u64 = 2310;

/// Largest exponent for the polynomial stage 2, whose coefficients all live
/// on the host (see [`crate::poly`]).
pub const POLY_MAX_P: u32 = 2_000_000;

/// The polynomial stage 2 was asked for (`MERSENNE_FORGE_PM1_POLY=1`).
fn poly_requested() -> bool {
    std::env::var("MERSENNE_FORGE_PM1_POLY").is_ok_and(|v| v == "1")
}

/// Stage 2 as a polynomial evaluation (the "FFT continuation"): with
/// b_j = x^(j^2) for the j below D / 2 coprime to D and g_m = x^((mD)^2),
/// the product over m of F(g_m), F(X) = prod_j (X - b_j), holds every term
/// g_m - b_j of [`stage2_paired`], so it covers every prime mD +- j up to
/// B2. The device computes the b_j and g_m with the same recurrences;
/// F is built once and evaluated at 240 giant points at a time on the host
/// with a remainder tree. It multiplies in the terms for composite mD +- j
/// too, which do no harm. The caller ensures every prime factor of D is at
/// most B1 and (B2 + D)^2 fits in a u64. Returns the product, or None when
/// stopped.
fn stage2_poly(
    be: &mut dyn Backend,
    x: crate::backend::Reg,
    primes: &[u64],
    stop: &dyn Fn() -> bool,
) -> Result<Option<BigUint>, String> {
    let berr = |e: crate::backend::BackendError| e.to_string();
    let p = be.exponent();
    let ring = crate::poly::Ring::new(p);
    let d = POLY_D;
    let (Some(&lo), Some(&hi)) = (primes.first(), primes.last()) else {
        return Ok(Some(BigUint::one()));
    };
    let js: Vec<u64> = (1..d / 2).filter(|j| j.gcd(&d) == 1).collect();
    let mut regs = Vec::new();
    let result = (|| -> Result<Option<BigUint>, String> {
        let mut new_reg = |be: &mut dyn Backend| -> Result<crate::backend::Reg, String> {
            let r = be.alloc().map_err(berr)?;
            regs.push(r);
            Ok(r)
        };
        let tmp = new_reg(be)?;
        let cur = new_reg(be)?;
        let inc = new_reg(be)?;
        let x8 = new_reg(be)?;
        be.copy(cur, x).map_err(berr)?;
        pow_small(be, x8, x, 8, tmp)?;
        be.copy(inc, x8).map_err(berr)?;
        let mut roots = Vec::with_capacity(js.len());
        let mut j = 1;
        while j <= *js.last().expect("D has units") {
            if js.binary_search(&j).is_ok() {
                roots.push(be.export(cur).map_err(berr)?.to_biguint());
            }
            be.mul(cur, inc).map_err(berr)?;
            be.mul(inc, x8).map_err(berr)?;
            j += 2;
        }
        let f = ring.from_roots(&roots);
        let m_lo = lo.saturating_sub(d / 2) / d;
        let m_hi = (hi + d / 2) / d;
        let (g, r, s) = (new_reg(be)?, new_reg(be)?, new_reg(be)?);
        if m_lo == 0 {
            be.import(g, &Residue::from_u64(p, 1)).map_err(berr)?;
        } else {
            pow_small(be, g, x, (m_lo * d) * (m_lo * d), tmp)?;
        }
        pow_small(be, r, x, (2 * m_lo + 1) * d * d, tmp)?;
        pow_small(be, s, x, 2 * d * d, tmp)?;
        let mut acc = BigUint::one();
        let mut batch = Vec::with_capacity(js.len());
        for m in m_lo..=m_hi {
            batch.push(be.export(g).map_err(berr)?.to_biguint());
            if batch.len() == js.len() || m == m_hi {
                for v in ring.eval(&f, &batch) {
                    acc = ring.mul_elt(&acc, &v);
                }
                batch.clear();
                if stop() {
                    return Ok(None);
                }
            }
            be.mul(g, r).map_err(berr)?;
            be.mul(r, s).map_err(berr)?;
        }
        Ok(Some(acc))
    })();
    for r in regs {
        let _ = be.free(r);
    }
    result
}

/// Stage 1 state: after bit `i` of E (counting down), x = 3^(E >> i).
pub type Stage1State = (u64, Residue);

/// [`pm1`] that can start from a saved stage 1 state and hands one to
/// `save` about once a minute and when stopped in stage 1.
pub fn pm1_resumable(
    be: &mut dyn Backend,
    b1: u64,
    b2: u64,
    resume: Option<Stage1State>,
    stop: &dyn Fn() -> bool,
    save: &mut dyn FnMut(u64, &Residue),
) -> Result<Pm1Outcome, String> {
    let p = be.exponent();
    if b1 < 2 {
        return Err("B1 must be at least 2".into());
    }
    let n = modulus(p);
    let mut out = Pm1Outcome {
        exponent: p,
        b1,
        b2,
        factor: None,
        stage: None,
        complete: false,
        notes: Vec::new(),
    };
    let e = stage1_exponent(p, b1);
    let berr = |e: crate::backend::BackendError| e.to_string();
    let x = be.alloc().map_err(berr)?;
    let base = be.alloc().map_err(berr)?;
    let three = Residue::from_u64(p, 3);
    let bits = e.bits();
    let start = match resume {
        Some((i, ref r)) if i < bits && r.exponent() == p => {
            be.import(x, r).map_err(berr)?;
            i
        }
        Some(_) => return Err("the saved P-1 state does not fit these bounds".into()),
        None => {
            be.import(x, &three).map_err(berr)?;
            bits - 1
        }
    };
    be.import(base, &three).map_err(berr)?;
    let mut pending = 0u64;
    let mut last_save = std::time::Instant::now();
    for i in (0..start).rev() {
        pending += 1;
        if e.bit(i) {
            be.square(x, pending).map_err(berr)?;
            pending = 0;
            be.mul(x, base).map_err(berr)?;
        }
        if i % 10_000 == 0 && i > 0 {
            let stopping = stop();
            if stopping || last_save.elapsed() >= std::time::Duration::from_secs(60) {
                be.square(x, pending).map_err(berr)?;
                pending = 0;
                save(i, &be.export(x).map_err(berr)?);
                last_save = std::time::Instant::now();
            }
            if stopping {
                be.free(base).map_err(berr)?;
                be.free(x).map_err(berr)?;
                out.notes
                    .push(format!("stopped in stage 1 with {i} bits left"));
                return Ok(out);
            }
        }
    }
    be.square(x, pending).map_err(berr)?;
    let x1 = be.export(x).map_err(berr)?;
    be.free(base).map_err(berr)?;
    be.free(x).map_err(berr)?;
    let xb = x1.to_biguint();
    let minus1 = |v: &BigUint| (v + &n - 1u32) % &n;
    if let Some(f) = factor_from(p, &minus1(&xb), &n, &mut out.notes) {
        out.factor = Some(f);
        out.stage = Some(1);
        out.complete = true;
        return Ok(out);
    }
    if b2 <= b1 {
        out.complete = true;
        return Ok(out);
    }
    let primes: Vec<u64> = primes_below(b2 + 1)
        .into_iter()
        .filter(|&q| q > b1)
        .collect();
    let Some(&q0) = primes.first() else {
        out.complete = true;
        return Ok(out);
    };
    // Polynomial stage 2 on request, where it fits (see `stage2_poly`).
    if poly_requested() {
        if b1 >= 11 && p <= POLY_MAX_P && (b2 + POLY_D).checked_mul(b2 + POLY_D).is_some() {
            let xr = be.alloc().map_err(berr)?;
            let res = be
                .import(xr, &x1)
                .map_err(berr)
                .and_then(|()| stage2_poly(be, xr, &primes, stop));
            let _ = be.free(xr);
            let Some(acc) = res? else {
                out.notes.push("stopped in stage 2".into());
                return Ok(out);
            };
            out.notes.push("polynomial stage 2".into());
            if let Some(f) = factor_from(p, &acc, &n, &mut out.notes) {
                out.factor = Some(f);
                out.stage = Some(2);
            }
            out.complete = true;
            return Ok(out);
        }
        out.notes.push(format!(
            "polynomial stage 2 needs B1 >= 11 and p <= {POLY_MAX_P}; paired stage 2 instead"
        ));
    }
    // Paired stage 2 when the bounds allow it (see `stage2_paired`).
    if b1 >= 7 && (b2 + PAIR_D).checked_mul(b2 + PAIR_D).is_some() {
        let xr = be.alloc().map_err(berr)?;
        let res = be
            .import(xr, &x1)
            .map_err(berr)
            .and_then(|()| stage2_paired(be, xr, &primes, stop));
        let _ = be.free(xr);
        let Some(acc) = res? else {
            out.notes.push("stopped in stage 2".into());
            return Ok(out);
        };
        if let Some(f) = factor_from(p, &acc, &n, &mut out.notes) {
            out.factor = Some(f);
            out.stage = Some(2);
        }
        out.complete = true;
        return Ok(out);
    }
    // Stage 2 for B1 < 7: y = x^q walks the primes by multiplying with
    // cached powers x^d for each prime gap d, and acc collects (y - 1), one
    // multiply and one host round trip per prime.
    let xr = be.alloc().map_err(berr)?;
    be.import(xr, &x1).map_err(berr)?;
    let y = be.alloc().map_err(berr)?;
    let acc = be.alloc().map_err(berr)?;
    let tmp = be.alloc().map_err(berr)?;
    let mut regs = vec![xr, y, acc, tmp];
    let result = (|| -> Result<Option<BigUint>, String> {
        let one_less = |r: &Residue| -> Result<Residue, String> {
            Ok(Residue::from_biguint(p, &minus1(&r.to_biguint())))
        };
        pow_small(be, y, xr, q0, tmp)?;
        let first = be.export(y).map_err(berr)?;
        be.import(acc, &one_less(&first)?).map_err(berr)?;
        let mut gaps: std::collections::HashMap<u64, crate::backend::Reg> = Default::default();
        let mut last = q0;
        for (j, &q) in primes.iter().enumerate().skip(1) {
            let d = q - last;
            let g = match gaps.get(&d) {
                Some(&g) => g,
                None => {
                    let g = be.alloc().map_err(berr)?;
                    regs.push(g);
                    pow_small(be, g, xr, d, tmp)?;
                    gaps.insert(d, g);
                    g
                }
            };
            be.mul(y, g).map_err(berr)?;
            let yv = be.export(y).map_err(berr)?;
            be.import(tmp, &one_less(&yv)?).map_err(berr)?;
            be.mul(acc, tmp).map_err(berr)?;
            last = q;
            if j % 1000 == 0 && stop() {
                return Ok(None);
            }
        }
        Ok(Some(be.export(acc).map_err(berr)?.to_biguint()))
    })();
    for r in regs {
        let _ = be.free(r);
    }
    let Some(acc) = result? else {
        out.notes.push("stopped in stage 2".into());
        return Ok(out);
    };
    if let Some(f) = factor_from(p, &acc, &n, &mut out.notes) {
        out.factor = Some(f);
        out.stage = Some(2);
    }
    out.complete = true;
    Ok(out)
}

/// Outcome of an ECM run.
#[derive(Debug, Clone, Serialize)]
pub struct EcmOutcome {
    pub exponent: u32,
    pub b1: u64,
    /// Stage 2 bound (B2 <= B1 skips stage 2).
    pub b2: u64,
    /// Curves run to the end (of stage 2 when it runs).
    pub curves: u32,
    /// Suyama parameter of the first curve; curve i uses sigma + i.
    pub sigma: u64,
    /// A checked proper factor, when one was found (it may be composite).
    pub factor: Option<String>,
    /// The sigma of the curve that found it.
    pub found_sigma: Option<u64>,
    pub notes: Vec<String>,
}

/// Arithmetic modulo N on the host.
struct ModN<'a>(&'a BigUint);

impl ModN<'_> {
    fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        (a + b) % self.0
    }
    fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint {
        (a + self.0 - b) % self.0
    }
    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        a * b % self.0
    }
}

/// Point X:Z on a Montgomery curve, with (A + 2) / 4.
type Point = (BigUint, BigUint);

fn xdbl(m: &ModN, p: &Point, a24: &BigUint) -> Point {
    let t1 = m.mul(&m.add(&p.0, &p.1), &m.add(&p.0, &p.1));
    let t2 = m.mul(&m.sub(&p.0, &p.1), &m.sub(&p.0, &p.1));
    let t3 = m.sub(&t1, &t2);
    (m.mul(&t1, &t2), m.mul(&t3, &m.add(&t2, &m.mul(a24, &t3))))
}

/// P + Q, given D = P - Q.
fn xadd(m: &ModN, p: &Point, q: &Point, d: &Point) -> Point {
    let u = m.mul(&m.sub(&p.0, &p.1), &m.add(&q.0, &q.1));
    let v = m.mul(&m.add(&p.0, &p.1), &m.sub(&q.0, &q.1));
    let s = m.add(&u, &v);
    let t = m.sub(&u, &v);
    (m.mul(&d.1, &m.mul(&s, &s)), m.mul(&d.0, &m.mul(&t, &t)))
}

/// k P by the Montgomery ladder.
fn ladder(m: &ModN, p: &Point, k: u64, a24: &BigUint) -> Point {
    if k == 1 {
        return p.clone();
    }
    let (mut r0, mut r1) = (p.clone(), xdbl(m, p, a24));
    for i in (0..63 - k.leading_zeros()).rev() {
        if k >> i & 1 == 1 {
            r0 = xadd(m, &r1, &r0, p);
            r1 = xdbl(m, &r1, a24);
        } else {
            r1 = xadd(m, &r1, &r0, p);
            r0 = xdbl(m, &r0, a24);
        }
    }
    r0
}

/// ECM stage 1 on `curves` Montgomery curves with Suyama's parametrization
/// (sigma, sigma + 1, ...), bound `b1`. Runs on the host with arbitrary
/// precision arithmetic: practical for small exponents only. ponytail: no
/// stage 2 and no Edwards curves; a device implementation with a stage 2 is
/// the upgrade path. `stop` is polled once per prime.
pub fn ecm(
    p: u32,
    b1: u64,
    b2: u64,
    curves: u32,
    sigma: u64,
    stop: &dyn Fn() -> bool,
) -> Result<EcmOutcome, String> {
    crate::exponent::validate(p).map_err(|e| e.to_string())?;
    if sigma < 6 || b1 < 2 {
        return Err("sigma must be at least 6 and B1 at least 2".into());
    }
    let n = modulus(p);
    let m = ModN(&n);
    let mut out = EcmOutcome {
        exponent: p,
        b1,
        b2,
        curves: 0,
        sigma,
        factor: None,
        found_sigma: None,
        notes: Vec::new(),
    };
    let primes = primes_below(b1.max(b2) + 1);
    let stage2: Vec<u64> = primes
        .iter()
        .copied()
        .filter(|&q| q > b1 && q <= b2)
        .collect();
    let primes: Vec<u64> = primes.into_iter().filter(|&q| q <= b1).collect();
    for c in 0..u64::from(curves) {
        let (a24, mut pt) = match suyama_curve(&m, &BigUint::from(sigma + c)) {
            Ok(v) => v,
            Err(den) => {
                if let Some(f) = factor_from(p, &den, &n, &mut out.notes) {
                    out.factor = Some(f);
                    out.found_sigma = Some(sigma + c);
                    return Ok(out);
                }
                continue;
            }
        };
        for &q in &primes {
            let mut qk = q;
            while let Some(x) = qk.checked_mul(q).filter(|&x| x <= b1) {
                qk = x;
            }
            pt = ladder(&m, &pt, qk, &a24);
            if stop() {
                out.notes
                    .push(format!("stopped in curve sigma = {}", sigma + c));
                return Ok(out);
            }
        }
        if let Some(f) = factor_from(p, &pt.1, &n, &mut out.notes) {
            out.curves += 1;
            out.factor = Some(f);
            out.found_sigma = Some(sigma + c);
            return Ok(out);
        }
        match ecm_stage2(&m, &pt, &a24, &stage2, stop) {
            None => {
                out.notes
                    .push(format!("stopped in stage 2 of curve sigma = {}", sigma + c));
                return Ok(out);
            }
            Some(acc) => {
                if let Some(f) = factor_from(p, &acc, &n, &mut out.notes) {
                    out.curves += 1;
                    out.factor = Some(f);
                    out.found_sigma = Some(sigma + c);
                    return Ok(out);
                }
            }
        }
        out.curves += 1;
    }
    Ok(out)
}

/// ECM stage 2, standard continuation, from the stage 1 point `pt`: walk
/// the odd multiples m Q with (m + 2) Q = xADD(m Q, 2 Q; difference
/// (m - 2) Q), one xADD per odd m, and multiply Z(q Q) into the product for
/// every prime q in `primes` (sorted, odd, above 2). A factor f of N
/// divides the product when the order of Q mod f is such a q. None when
/// stopped; the product of no terms is 1. ponytail: no baby step giant step
/// pairing; about 3 (B2 - B1) multiplications per curve.
fn ecm_stage2(
    m: &ModN,
    pt: &Point,
    a24: &BigUint,
    primes: &[u64],
    stop: &dyn Fn() -> bool,
) -> Option<BigUint> {
    let mut acc = BigUint::one();
    let Some(&first) = primes.first() else {
        return Some(acc);
    };
    let two = xdbl(m, pt, a24);
    let mut prev = ladder(m, pt, first - 2, a24);
    let mut cur = ladder(m, pt, first, a24);
    let mut mq = first;
    let mut next = primes.iter().peekable();
    while let Some(&&q) = next.peek() {
        if mq == q {
            acc = m.mul(&acc, &cur.1);
            next.next();
            if next.peek().is_none() {
                break;
            }
        }
        let after = xadd(m, &cur, &two, &prev);
        prev = std::mem::replace(&mut cur, after);
        mq += 2;
        if mq % 2_000_000 == 1 && stop() {
            return None;
        }
    }
    Some(acc)
}

/// (A + 2) / 4 of the Montgomery curve with Suyama's parametrization for
/// `sigma`, with the starting point (u^3 : v^3). Err(den) when the
/// denominator is not invertible modulo N (its gcd with N may be a factor).
fn suyama_curve(m: &ModN, sigma: &BigUint) -> Result<(BigUint, Point), BigUint> {
    let u = m.sub(&m.mul(sigma, sigma), &BigUint::from(5u32));
    let v = m.mul(&BigUint::from(4u32), sigma);
    let u3 = m.mul(&m.mul(&u, &u), &u);
    let vmu = m.sub(&v, &u);
    // a24 = (v - u)^3 (3u + v) / (16 u^3 v)
    let num = m.mul(
        &m.mul(&m.mul(&vmu, &vmu), &vmu),
        &m.add(&m.mul(&BigUint::from(3u32), &u), &v),
    );
    let den = m.mul(&m.mul(&BigUint::from(16u32), &u3), &v);
    match mod_inverse_big(&den, m.0) {
        Some(inv) => Ok((m.mul(&num, &inv), (u3, m.mul(&m.mul(&v, &v), &v)))),
        None => Err(den),
    }
}

/// One curve of a GMP-ECM save file (`-save`), as Prime95's
/// `read_resumefile_line` reads it: `METHOD=ECM; PARAM=0; SIGMA=..;
/// B1=..; N=..; X=..; [Z=..;] ...`. Only PARAM=0 (Suyama) curves are
/// supported here.
#[derive(Debug, Clone, PartialEq)]
pub struct GmpEcmCurve {
    pub sigma: u64,
    pub b1: u64,
    pub n: BigUint,
    /// The stage 1 point's x coordinate, normalized to Z = 1.
    pub x: BigUint,
}

/// Parse the curves of a GMP-ECM save file. Comment lines (`#`) and empty
/// lines are skipped; a line Prime95 would skip with a warning (unknown
/// tag, missing field) is an error here, so nothing is silently dropped.
pub fn parse_gmp_ecm_save(text: &str) -> Result<Vec<GmpEcmCurve>, String> {
    let num = |s: &str| -> Option<BigUint> {
        let s = s.trim();
        match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            Some(h) => BigUint::parse_bytes(h.as_bytes(), 16),
            None => BigUint::parse_bytes(s.as_bytes(), 10),
        }
    };
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let bad = |what: &str| format!("save file line {}: {what}", i + 1);
        let (mut method, mut param, mut sigma, mut b1, mut n, mut x, mut z) =
            (false, 0u64, None, None, None, None, None);
        for field in line.split(';').map(str::trim).filter(|f| !f.is_empty()) {
            let (tag, value) = field
                .split_once('=')
                .ok_or_else(|| bad("field without '='"))?;
            let value = value.trim();
            match tag.trim() {
                "METHOD" => method = value == "ECM",
                "PARAM" => param = value.parse().map_err(|_| bad("bad PARAM"))?,
                "SIGMA" => sigma = Some(value.parse::<u64>().map_err(|_| bad("bad SIGMA"))?),
                "B1" => {
                    let v: f64 = value.parse().map_err(|_| bad("bad B1"))?;
                    if !(v.is_finite() && v >= 2.0) {
                        return Err(bad("bad B1"));
                    }
                    b1 = Some(v as u64);
                }
                "N" => n = Some(num(value).ok_or_else(|| bad("bad N"))?),
                "X" => x = Some(num(value).ok_or_else(|| bad("bad X"))?),
                "Z" => z = Some(num(value).ok_or_else(|| bad("bad Z"))?),
                "Y" | "X0" | "Y0" | "A" | "ETYPE" | "CHECKSUM" | "COMMENT" | "PROGRAM" | "WHO"
                | "TIME" => {}
                t => return Err(bad(&format!("unknown tag {t}"))),
            }
        }
        if !method {
            return Err(bad("not an ECM line"));
        }
        if param != 0 {
            return Err(bad(&format!(
                "PARAM={param} is not supported (only 0, Suyama)"
            )));
        }
        let (Some(sigma), Some(b1), Some(n), Some(x)) = (sigma, b1, n, x) else {
            return Err(bad("SIGMA, B1, N, and X are required"));
        };
        if n <= BigUint::from(1u32) || sigma < 6 {
            return Err(bad("bad N or SIGMA"));
        }
        let x = match z {
            Some(z) => {
                let zi =
                    mod_inverse_big(&(z % &n), &n).ok_or_else(|| bad("Z is not invertible"))?;
                x * zi % &n
            }
            None => x % &n,
        };
        out.push(GmpEcmCurve { sigma, b1, n, x });
    }
    Ok(out)
}

/// ECM stage 2 from GMP-ECM stage 1 results (Prime95's `ECMSTAGE2=`):
/// curves `skip..skip + count` of `curves`, each to `b2` (0: 100 B1). Each
/// N must divide 2^p - 1; the arithmetic is modulo that N, and a factor is
/// checked against 2^p - 1. The outcome's b1 is the largest B1 of the
/// curves run, and its sigma the first curve's.
pub fn ecm_stage2_gmp(
    p: u32,
    curves: &[GmpEcmCurve],
    b2: u64,
    skip: usize,
    count: Option<usize>,
    stop: &dyn Fn() -> bool,
) -> Result<EcmOutcome, String> {
    crate::exponent::validate(p).map_err(|e| e.to_string())?;
    let mp = modulus(p);
    let chosen: Vec<&GmpEcmCurve> = curves
        .iter()
        .skip(skip)
        .take(count.unwrap_or(usize::MAX))
        .collect();
    let Some(first) = chosen.first() else {
        return Err("no curves to run in the save file".into());
    };
    let mut out = EcmOutcome {
        exponent: p,
        b1: chosen.iter().map(|c| c.b1).max().unwrap_or(0),
        b2: 0,
        curves: 0,
        sigma: first.sigma,
        factor: None,
        found_sigma: None,
        notes: Vec::new(),
    };
    for c in chosen {
        if !(&mp % &c.n).is_zero() {
            return Err(format!(
                "N of the curve with sigma {} does not divide 2^{p} - 1",
                c.sigma
            ));
        }
        let b2c = if b2 == 0 {
            c.b1.saturating_mul(100)
        } else {
            b2
        };
        out.b2 = out.b2.max(b2c);
        let m = ModN(&c.n);
        let (a24, _) = match suyama_curve(&m, &BigUint::from(c.sigma)) {
            Ok(v) => v,
            Err(den) => {
                if let Some(f) = factor_from(p, &den, &c.n, &mut out.notes) {
                    out.factor = Some(f);
                    out.found_sigma = Some(c.sigma);
                    return Ok(out);
                }
                continue;
            }
        };
        let primes: Vec<u64> = primes_below(b2c + 1)
            .into_iter()
            .filter(|&q| q > c.b1 && q > 2)
            .collect();
        let pt: Point = (c.x.clone(), BigUint::one());
        let Some(acc) = ecm_stage2(&m, &pt, &a24, &primes, stop) else {
            out.notes
                .push(format!("stopped in stage 2 of curve sigma = {}", c.sigma));
            return Ok(out);
        };
        out.curves += 1;
        if let Some(f) = factor_from(p, &acc, &c.n, &mut out.notes) {
            out.factor = Some(f);
            out.found_sigma = Some(c.sigma);
            return Ok(out);
        }
    }
    Ok(out)
}

/// The outcome of a P+1 run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Pp1Outcome {
    pub exponent: u32,
    pub b1: u64,
    pub b2: u64,
    /// The start value P0 = numerator / denominator, as Prime95 reports it.
    pub start: (u32, u32),
    /// A checked proper factor, when one was found (it may be composite).
    pub factor: Option<String>,
    /// The stage that found it (1 or 2).
    pub stage: Option<u8>,
    /// Both stages ran to their bounds.
    pub complete: bool,
    pub notes: Vec<String>,
}

/// The P+1 start value for Prime95's `nth_run` field: 2/7 for the first
/// run and 6/5 for the second, the two values Prime95 uses (each finds a
/// factor q with q - (D/q) smooth for a different discriminant D). Later
/// runs use (2n + 1)/(2n - 1). ponytail: Prime95 picks those at random;
/// any value works, and a fixed one makes a run repeatable.
pub fn pp1_start(nth_run: u32) -> (u32, u32) {
    match nth_run {
        0 | 1 => (2, 7),
        2 => (6, 5),
        n => (2 * n + 1, 2 * n - 1),
    }
}

/// V_k(P) of the Lucas sequence V_0 = 2, V_1 = P, V_(n+1) = P V_n - V_(n-1),
/// by the ladder on (V_m, V_(m+1)).
fn lucas_v(m: &ModN, pv: &BigUint, k: &BigUint) -> BigUint {
    let two = BigUint::from(2u32);
    if k.is_zero() {
        return two;
    }
    let (mut a, mut b) = (pv.clone(), m.sub(&m.mul(pv, pv), &two));
    for i in (0..k.bits() - 1).rev() {
        let ab = m.sub(&m.mul(&a, &b), pv);
        if k.bit(i) {
            a = ab;
            b = m.sub(&m.mul(&b, &b), &two);
        } else {
            b = ab;
            a = m.sub(&m.mul(&a, &a), &two);
        }
    }
    a
}

/// Williams' P+1 on 2^p - 1 on the host: stage 1 computes W = V_E(P0) with
/// E the stage 1 exponent for `b1` and tests gcd(W - 2, N); stage 2 tests
/// gcd(prod (V_q(W) - 2), N) over primes b1 < q <= b2. Practical for small
/// exponents only. ponytail: stage 2 runs one Lucas ladder per prime; a
/// prime-gap continuation (as ECM's) and a device implementation are the
/// upgrade path. `stop` is polled every few thousand primes.
pub fn pp1(
    p: u32,
    b1: u64,
    b2: u64,
    nth_run: u32,
    stop: &dyn Fn() -> bool,
) -> Result<Pp1Outcome, String> {
    crate::exponent::validate(p).map_err(|e| e.to_string())?;
    if b1 < 2 {
        return Err("B1 must be at least 2".into());
    }
    let n = modulus(p);
    let m = ModN(&n);
    let start = pp1_start(nth_run);
    let mut out = Pp1Outcome {
        exponent: p,
        b1,
        b2: b2.max(b1),
        start,
        factor: None,
        stage: None,
        complete: false,
        notes: Vec::new(),
    };
    let den = BigUint::from(start.1) % &n;
    let Some(inv) = mod_inverse_big(&den, &n) else {
        return Err(format!("{} shares a factor with N", start.1));
    };
    let p0 = m.mul(&BigUint::from(start.0), &inv);
    let w = lucas_v(&m, &p0, &stage1_exponent(p, b1));
    let two = BigUint::from(2u32);
    if let Some(f) = factor_from(p, &m.sub(&w, &two), &n, &mut out.notes) {
        out.factor = Some(f);
        out.stage = Some(1);
        out.complete = true;
        return Ok(out);
    }
    if b2 > b1 {
        let mut acc = BigUint::one();
        for (i, q) in primes_below(b2 + 1)
            .into_iter()
            .filter(|&q| q > b1)
            .enumerate()
        {
            if i % 4096 == 4095 && stop() {
                out.notes.push(format!("stopped in stage 2 before q = {q}"));
                return Ok(out);
            }
            acc = m.mul(&acc, &m.sub(&lucas_v(&m, &w, &BigUint::from(q)), &two));
        }
        if let Some(f) = factor_from(p, &acc, &n, &mut out.notes) {
            out.factor = Some(f);
            out.stage = Some(2);
        }
    }
    out.complete = true;
    Ok(out)
}

/// a^-1 mod n, when gcd(a, n) = 1, by the extended Euclidean algorithm.
fn mod_inverse_big(a: &BigUint, n: &BigUint) -> Option<BigUint> {
    use num_bigint::BigInt;
    let e = BigInt::from(a.clone()).extended_gcd(&BigInt::from(n.clone()));
    if !e.gcd.is_one() {
        return None;
    }
    let nn = BigInt::from(n.clone());
    ((e.x % &nn + &nn) % &nn).to_biguint()
}

/// What a factoring line asks for.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum FactorWork {
    /// `Factor=[AID,][M]p,bit_lo,bit_hi[,"known factors"]`
    Tf { bit_lo: u32, bit_hi: u32 },
    /// `Pminus1=[AID,]1,2,p,-1,B1,B2[,tf_bits][,B2_start][,"factors"]` or
    /// `Pfactor=[AID,]1,2,p,-1,tf_bits,tests_saved[,"factors"]` (also the
    /// old `Pfactor=[AID,]p,tf_bits,double_check`). Pfactor bounds are
    /// chosen here.
    Pm1 { b1: u64, b2: u64 },
    /// `Pplus1=[AID,]1,2,p,-1,B1,B2,nth_run[,tf_bits][,"factors"]`
    Pp1 { b1: u64, b2: u64, nth_run: u32 },
    /// `ECM2=` or `ECM=[AID,]1,2,p,-1,B1[,B2][,curves][,sigma][,"factors"]`
    Ecm {
        b1: u64,
        b2: u64,
        curves: u32,
        sigma: Option<u64>,
    },
    /// `ECMSTAGE2=[AID,]1,2,p,-1,file[,B2][,skip_curves][,num_curves][,"factors"]`:
    /// stage 2 of the curves in a GMP-ECM save file (a relative file name is
    /// next to the worktodo file). B2 0 means 100 B1 per curve.
    EcmStage2 {
        file: String,
        b2: u64,
        skip: u32,
        curves: Option<u32>,
    },
}

/// One factoring assignment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FactorLine {
    pub work: FactorWork,
    pub exponent: u32,
    pub aid: Option<String>,
    pub line: String,
    /// Known factors from the line. Factors found that are among them (or
    /// products of them) are not reported as new.
    pub known_factors: Vec<String>,
}

impl FactorLine {
    /// `f` with every known factor divided out, or None when nothing new
    /// is left (f was a product of known factors).
    pub fn new_part(&self, f: &str) -> Option<String> {
        let mut g = BigUint::parse_bytes(f.as_bytes(), 10)?;
        for k in &self.known_factors {
            let Some(k) = BigUint::parse_bytes(k.as_bytes(), 10) else {
                continue;
            };
            if k <= BigUint::one() {
                continue;
            }
            while (&g % &k).is_zero() {
                g /= &k;
            }
        }
        (!g.is_one()).then(|| g.to_string())
    }
}

/// Bounds for a `Pfactor=` line and for P-1 before a PRP test, from the
/// cost model of [`crate::policy`]: the B1 that maximizes the chance of a
/// factor (above 2^tf_bits) times the tests it saves, less its cost.
/// Stage 1 only (B2 = B1): stage 2 saves no progress, so `work` does not
/// run it.
pub fn pfactor_bounds(p: u32, tf_bits: f64, tests_saved: f64) -> (u64, u64) {
    let plan = crate::policy::pm1_plan(p, tf_bits, tests_saved, None);
    (plan.b1, plan.b1)
}

/// Largest exponent `work` runs P-1 for. Above it the final gcd on the
/// host (quadratic in p) and host stage 2 take too long, and the line
/// stays in place for another program.
pub const WORK_PM1_MAX_P: u32 = 10_000_000;

/// Largest exponent `work` runs P+1 or ECM for: both run entirely on the
/// host with arbitrary precision arithmetic.
pub const WORK_HOST_MAX_P: u32 = 1_000_000;

/// Largest exponent `work` takes a line of this kind for. Larger lines
/// stay in place for another program.
pub fn work_max_p(w: &FactorWork) -> u32 {
    match w {
        FactorWork::Tf { .. } => u32::MAX,
        FactorWork::Pm1 { .. } => WORK_PM1_MAX_P,
        FactorWork::Pp1 { .. } | FactorWork::Ecm { .. } | FactorWork::EcmStage2 { .. } => {
            WORK_HOST_MAX_P
        }
    }
}

/// Parse a `Factor=`, `Pminus1=`, `Pfactor=`, `Pplus1=`, `ECM=`, or
/// `ECM2=` line in Prime95's forms (see `FactorWork`). `Ok(None)` for any
/// other line (including comments), so callers try this parser first.
/// Only Mersenne numbers (1,2,p,-1) are accepted; known factors must each
/// divide 2^p - 1.
pub fn parse_factor_line(raw: &str) -> Result<Option<FactorLine>, String> {
    let line = raw.trim_end_matches(['\r', '\n']);
    let head = line.split('=').next().unwrap_or("").trim();
    if head.eq_ignore_ascii_case("ECMSTAGE2") {
        return parse_ecm_stage2_line(line).map(Some);
    }
    if head.eq_ignore_ascii_case("ECMSTAGE2N") {
        return Err(format!(
            "ECMSTAGE2N names no number; use ECMSTAGE2=1,2,p,-1,...: {line:?}"
        ));
    }
    let t = match crate::worktodo::tokenize(raw) {
        Ok(Some(t)) => t,
        Ok(None) => return Ok(None),
        // Not a factoring line unless it names a factoring work type.
        Err(e) => {
            let head = line.split(['=', ';']).next().unwrap_or("");
            let ours = ["Factor", "Pminus1", "Pfactor", "Pplus1", "ECM", "ECM2"]
                .iter()
                .any(|k| head.trim().eq_ignore_ascii_case(k));
            return if ours { Err(e) } else { Ok(None) };
        }
    };
    let kind = ["Factor", "Pminus1", "Pfactor", "Pplus1", "ECM", "ECM2"]
        .into_iter()
        .find(|k| t.is(k));
    let Some(kind) = kind else {
        return Ok(None);
    };
    let parts = &t.fields;
    let bad = || format!("malformed {kind} line: {line:?}");
    let int = |s: &str| -> Result<u64, String> {
        s.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| s.parse().ok())
            .flatten()
            .ok_or_else(bad)
    };
    let mersenne = parts.len() >= 4 && parts[..2] == ["1", "2"] && parts[3] == "-1";
    let (p, work) = match kind {
        "Factor" => {
            let [p, lo, hi] = parts[..] else {
                return Err(bad());
            };
            let p = p.strip_prefix('M').unwrap_or(p);
            let (lo, hi) = (int(lo)? as u32, int(hi)? as u32);
            (
                int(p)?,
                FactorWork::Tf {
                    bit_lo: lo,
                    bit_hi: hi,
                },
            )
        }
        "Pfactor" if !mersenne => {
            // The old form: p,tf_bits,double_check.
            let [p, tf, dc] = parts[..] else {
                return Err(bad());
            };
            let p = int(p)?;
            // A double check saves one test, a first test two.
            let saved = if int(dc)? == 0 { 2.0 } else { 1.0 };
            let (b1, b2) =
                pfactor_bounds(u32::try_from(p).map_err(|_| bad())?, int(tf)? as f64, saved);
            (p, FactorWork::Pm1 { b1, b2 })
        }
        _ if !mersenne => {
            return Err(format!(
                "only Mersenne numbers (1,2,p,-1) are supported: {line:?}"
            ))
        }
        "Pfactor" => {
            if parts.len() != 6 {
                return Err(bad());
            }
            let p = int(parts[2])?;
            let real = |s: &str| -> Result<f64, String> {
                s.parse::<f64>()
                    .ok()
                    .filter(|x| x.is_finite() && *x >= 0.0)
                    .ok_or_else(bad)
            };
            let (b1, b2) = pfactor_bounds(
                u32::try_from(p).map_err(|_| bad())?,
                real(parts[4])?,
                real(parts[5])?,
            );
            (p, FactorWork::Pm1 { b1, b2 })
        }
        "Pminus1" => {
            // Optional tf_bits and B2_start: stage 2 here always starts
            // at B1, which covers any later start.
            if !(6..=8).contains(&parts.len()) {
                return Err(bad());
            }
            let b1 = int(parts[4])?;
            let b2 = int(parts[5])?;
            (int(parts[2])?, FactorWork::Pm1 { b1, b2 })
        }
        "Pplus1" => {
            if !(7..=8).contains(&parts.len()) {
                return Err(bad());
            }
            let nth_run = u32::try_from(int(parts[6])?).map_err(|_| bad())?;
            if nth_run == 0 {
                return Err(bad());
            }
            (
                int(parts[2])?,
                FactorWork::Pp1 {
                    b1: int(parts[4])?,
                    b2: int(parts[5])?,
                    nth_run,
                },
            )
        }
        _ => {
            // ECM and ECM2.
            if !(5..=8).contains(&parts.len()) {
                return Err(bad());
            }
            let b1 = int(parts[4])?;
            let b2 = parts.get(5).map(|x| int(x)).transpose()?.unwrap_or(0);
            let curves = match parts.get(6) {
                Some(x) => u32::try_from(int(x)?).map_err(|_| bad())?,
                None => 100,
            };
            let sigma = parts.get(7).map(|x| int(x)).transpose()?;
            // ponytail: B2 = 100 B1 when the line gives none; Prime95
            // chooses its own B2 from a cost model.
            let b2 = if b2 == 0 { b1.saturating_mul(100) } else { b2 };
            (
                int(parts[2])?,
                FactorWork::Ecm {
                    b1,
                    b2,
                    curves,
                    sigma,
                },
            )
        }
    };
    let p = u32::try_from(p).map_err(|_| bad())?;
    crate::exponent::validate(p).map_err(|e| format!("{e}: {line:?}"))?;
    if let FactorWork::Tf { bit_lo, bit_hi } = work {
        if !(bit_lo < bit_hi && bit_hi <= 120) {
            return Err(format!("bad bit range: {line:?}"));
        }
    }
    for f in &t.known_factors {
        let c = crate::worktodo::check_factor(p, f).map_err(|e| format!("{e}: {line:?}"))?;
        if !c.accepted() {
            return Err(format!("{f} is not a proper factor of 2^{p} - 1: {line:?}"));
        }
    }
    Ok(Some(FactorLine {
        work,
        exponent: p,
        aid: t.aid,
        line: line.to_string(),
        known_factors: t.known_factors,
    }))
}

/// `ECMSTAGE2=` has its own reader because the file name may be quoted,
/// which the shared tokenizer would read as a known factor list.
fn parse_ecm_stage2_line(line: &str) -> Result<FactorLine, String> {
    let bad = || format!("malformed ECMSTAGE2 line: {line:?}");
    let rest = line.split_once('=').ok_or_else(bad)?.1.trim();
    // Split on commas outside double quotes, keeping the quotes.
    let mut fields = Vec::new();
    let (mut cur, mut quoted) = (String::new(), false);
    for ch in rest.chars() {
        match ch {
            '"' => {
                quoted = !quoted;
                cur.push(ch);
            }
            ',' if !quoted => fields.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    if quoted {
        return Err(bad());
    }
    fields.push(cur);
    let mut fields: Vec<String> = fields.into_iter().map(|f| f.trim().to_string()).collect();
    let aid = match fields.first() {
        Some(f) if f.len() == 32 && f.bytes().all(|b| b.is_ascii_hexdigit()) => {
            Some(fields.remove(0))
        }
        Some(f) if f.eq_ignore_ascii_case("N/A") || f.is_empty() => {
            fields.remove(0);
            None
        }
        _ => None,
    };
    if fields.len() < 5 || fields[0] != "1" || fields[1] != "2" || fields[3] != "-1" {
        return Err(bad());
    }
    let p: u32 = fields[2].parse().map_err(|_| bad())?;
    crate::exponent::validate(p).map_err(|e| format!("{e}: {line:?}"))?;
    let file = fields[4].trim_matches('"').to_string();
    if file.is_empty() {
        return Err(bad());
    }
    let mut known_factors = Vec::new();
    let mut nums = Vec::new();
    for f in &fields[5..] {
        if let Some(q) = f.strip_prefix('"') {
            let inner = q.strip_suffix('"').ok_or_else(bad)?;
            known_factors.extend(
                inner
                    .split(|c: char| !c.is_ascii_digit())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
            );
        } else if known_factors.is_empty() {
            nums.push(
                f.parse::<f64>()
                    .ok()
                    .filter(|x| x.is_finite() && *x >= 0.0)
                    .ok_or_else(bad)?,
            );
        } else {
            return Err(bad());
        }
    }
    if nums.len() > 3 {
        return Err(bad());
    }
    for f in &known_factors {
        let c = crate::worktodo::check_factor(p, f).map_err(|e| format!("{e}: {line:?}"))?;
        if !c.accepted() {
            return Err(format!("{f} is not a proper factor of 2^{p} - 1: {line:?}"));
        }
    }
    let as_u32 = |x: f64| u32::try_from(x as u64).map_err(|_| bad());
    Ok(FactorLine {
        work: FactorWork::EcmStage2 {
            file,
            b2: nums.first().map_or(0, |&x| x as u64),
            skip: nums.get(1).map_or(Ok(0), |&x| as_u32(x))?,
            curves: nums
                .get(2)
                .map(|&x| as_u32(x))
                .transpose()?
                .filter(|&c| c > 0),
        },
        exponent: p,
        aid,
        line: line.to_string(),
        known_factors,
    })
}

/// `line` (a PRP line asking for P-1) with its tests_saved field set to 0,
/// so the PRP test runs once P-1 has been done.
pub fn with_pm1_done(line: &str, p: u32) -> Option<String> {
    let (head, rest) = line.split_once('=')?;
    let mut f: Vec<String> = rest.split(',').map(str::to_string).collect();
    let pos = f.windows(4).position(|w| {
        w[0].trim() == "1"
            && w[1].trim() == "2"
            && w[2].trim() == p.to_string()
            && w[3].trim() == "-1"
    })?;
    *f.get_mut(pos + 5)? = "0".into();
    Some(format!("{head}={}", f.join(",")))
}

/// The JSON result line for a TF run. The keys follow mfaktc's JSON results
/// as recalled, NOT checked against mfaktc output or a PrimeNet server.
pub fn tf_result_line(a: &FactorLine, o: &TfOutcome, timestamp: &str) -> serde_json::Value {
    let mut v = serde_json::json!({
        "status": if o.factors.is_empty() { "NF" } else { "F" },
        "exponent": o.exponent,
        "worktype": "TF",
        "bitlo": o.bit_lo,
        "bithi": o.bit_hi,
        "rangecomplete": o.complete,
        "program": program(),
        "timestamp": timestamp,
    });
    if !o.factors.is_empty() {
        v["factors"] = serde_json::json!(o.factors);
    }
    if let Some(aid) = &a.aid {
        v["aid"] = aid.clone().into();
    }
    v
}

/// The JSON result line for a P-1 run, with the keys upstream PRPLL's P-1
/// wrote (`B1`, `B2` when stage 2 ran, `factors`). Upstream's
/// `fft-length` has no counterpart here and is left out.
pub fn pm1_result_line(a: &FactorLine, o: &Pm1Outcome, timestamp: &str) -> serde_json::Value {
    let mut v = serde_json::json!({
        "status": if o.factor.is_some() { "F" } else { "NF" },
        "exponent": o.exponent,
        "worktype": "PM1",
        "B1": o.b1,
        "program": program(),
        "timestamp": timestamp,
    });
    if o.b2 > o.b1 {
        v["B2"] = o.b2.into();
    }
    if let Some(f) = &o.factor {
        v["factors"] = serde_json::json!([f]);
    }
    if let Some(aid) = &a.aid {
        v["aid"] = aid.clone().into();
    }
    v
}

/// Known factors, as Prime95 adds them to a result line.
fn put_known(v: &mut serde_json::Value, a: &FactorLine) {
    if !a.known_factors.is_empty() {
        v["known-factors"] = serde_json::json!(a.known_factors);
    }
    if let Some(aid) = &a.aid {
        v["aid"] = aid.clone().into();
    }
}

/// The JSON result line for a P+1 run, with Prime95's keys (`worktype`
/// "P+1", `b1`, `b2` when stage 2 ran, `start` as "n/d"), the ones
/// AutoPrimeNet reads.
pub fn pp1_result_line(a: &FactorLine, o: &Pp1Outcome, timestamp: &str) -> serde_json::Value {
    let mut v = serde_json::json!({
        "status": if o.factor.is_some() { "F" } else { "NF" },
        "exponent": o.exponent,
        "worktype": "P+1",
        "b1": o.b1,
        "start": format!("{}/{}", o.start.0, o.start.1),
        "program": program(),
        "timestamp": timestamp,
    });
    if o.b2 > o.b1 {
        v["b2"] = o.b2.into();
    }
    if let Some(f) = &o.factor {
        v["factors"] = serde_json::json!([f]);
    }
    put_known(&mut v, a);
    v
}

/// The JSON result line for an ECM run, with Prime95's keys (`worktype`
/// "ECM", `b1`, `b2`, `curves`, and `sigma` of the curve that found a
/// factor), the ones AutoPrimeNet reads.
pub fn ecm_result_line(a: &FactorLine, o: &EcmOutcome, timestamp: &str) -> serde_json::Value {
    let mut v = serde_json::json!({
        "status": if o.factor.is_some() { "F" } else { "NF" },
        "exponent": o.exponent,
        "worktype": "ECM",
        "b1": o.b1,
        "b2": o.b2.max(o.b1),
        "curves": o.curves,
        "program": program(),
        "timestamp": timestamp,
    });
    if let Some(f) = &o.factor {
        v["factors"] = serde_json::json!([f]);
    }
    if let Some(s) = o.found_sigma {
        v["sigma"] = s.into();
    }
    put_known(&mut v, a);
    v
}

fn program() -> serde_json::Value {
    serde_json::json!({
        "name": "mersenne-forge",
        "version": env!("CARGO_PKG_VERSION"),
        "port": crate::worktodo::port_code(),
        "os": {"os": std::env::consts::OS, "architecture": std::env::consts::ARCH},
    })
}

/// Known small factors, as a cross-check that TF misses nothing it should
/// find: (p, q) with q | 2^p - 1. Also used by `selftest`.
pub const KNOWN: &[(u32, u64)] = &[
    (11, 23),
    (11, 89),
    (23, 47),
    (29, 233),
    (29, 1103),
    (29, 2089),
    (37, 223),
    (43, 431),
    (47, 2351),
    (47, 4513),
    (67, 193_707_721),
    (71, 228_479),
    (83, 167),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn tf(p: u32, lo: u32, hi: u32) -> Vec<String> {
        trial_factor(p, lo, hi, None, &|| false, &mut |_| {})
            .unwrap()
            .factors
    }

    #[test]
    fn trial_factoring_finds_known_factors() {
        for &(p, q) in KNOWN {
            let bits = 64 - q.leading_zeros();
            let f = tf(p, bits - 1, bits);
            assert!(f.contains(&q.to_string()), "M{p}: {q} not in {f:?}");
        }
        // M61 is prime: nothing below 2^30.
        assert!(tf(61, 1, 30).is_empty());
    }

    #[test]
    fn trial_factoring_uses_the_sieve_correctly() {
        // 2^67 - 1 = 193707721 * 761838257287; the first lies in [2^27, 2^28),
        // well above the sieve bound.
        assert_eq!(tf(67, 27, 28), vec!["193707721".to_string()]);
        assert!(tf(67, 26, 27).is_empty());
    }

    /// The class split finds exactly the divisors a plain scan of every k
    /// finds, less those with a prime factor below the sieve bound, and a
    /// resumed run picks up at a block start.
    #[test]
    fn class_split_matches_a_plain_scan() {
        for (p, lo, hi) in [
            (67u32, 17u32, 28u32),
            (101, 17, 27),
            (113, 18, 27),
            (11, 17, 26),
        ] {
            let two_p = 2 * u128::from(p);
            let small = primes_below(SIEVE_BOUND);
            let k_lo = ((1u128 << lo) - 1).div_ceil(two_p);
            let k_hi = ((1u128 << hi) - 1).div_ceil(two_p);
            let want: Vec<String> = (k_lo..k_hi)
                .map(|k| two_p * k + 1)
                .filter(|&q| divides(p, q))
                .filter(|&q| small.iter().all(|&s| q % u128::from(s) != 0))
                .map(|q| q.to_string())
                .collect();
            let got = trial_factor(p, lo, hi, None, &|| false, &mut |_| {}).unwrap();
            assert!(got.complete);
            assert_eq!(got.factors, want, "p = {p}");
        }
        // The scan is not vacuous: 2^67 - 1's first factor is in range.
        assert_eq!(
            trial_factor(67, 17, 28, None, &|| false, &mut |_| {})
                .unwrap()
                .factors,
            ["193707721"]
        );
    }

    #[test]
    fn a_stopped_range_resumes() {
        let o = trial_factor(67, 27, 28, None, &|| true, &mut |_| {}).unwrap();
        assert!(!o.complete);
        let o = trial_factor(67, 27, 28, Some(o.next_k), &|| false, &mut |_| {}).unwrap();
        assert!(o.complete && o.factors == ["193707721"]);
    }

    #[test]
    fn pm1_finds_smooth_factors() {
        // 2^67 - 1: 193707721 - 1 = 2^3 * 3^3 * 5 * 67 * 2677, so B1 = 2677
        // finds it in stage 1, and B1 = 1000, B2 = 3000 in stage 2. The
        // cofactor 761838257287 - 1 = 2 * 3^2 * 29 * 67 * 2551 * 8539 stays
        // out of reach of both, so the gcd is the factor and not N.
        let mut be = crate::backend::cpu::CpuBackend::new(67).unwrap();
        let o = pm1(&mut be, 2677, 0, &|| false).unwrap();
        assert_eq!((o.factor.as_deref(), o.stage), (Some("193707721"), Some(1)));
        let o = pm1(&mut be, 1000, 3000, &|| false).unwrap();
        assert_eq!((o.factor.as_deref(), o.stage), (Some("193707721"), Some(2)));
        let o = pm1(&mut be, 100, 200, &|| false).unwrap();
        assert!(o.factor.is_none() && o.complete);
        // The paired stage 2 covers B2 exactly (2677 = 13 * 210 - 53) and
        // stops short of it below.
        let o = pm1(&mut be, 1000, 2677, &|| false).unwrap();
        assert_eq!(o.factor.as_deref(), Some("193707721"));
        let o = pm1(&mut be, 1000, 2676, &|| false).unwrap();
        assert!(o.factor.is_none() && o.complete);
        // A backend without device subtraction (the default `sub`) agrees.
        let mut dwt = crate::backend::dwt::DwtBackend::new(67).unwrap();
        let o = pm1(&mut dwt, 1000, 3000, &|| false).unwrap();
        assert_eq!((o.factor.as_deref(), o.stage), (Some("193707721"), Some(2)));
    }

    /// The polynomial stage 2 finds what the paired one finds, and its
    /// product holds every paired term (so a factor of one divides the
    /// other's gcd with N).
    #[test]
    fn polynomial_stage_2_matches_the_paired_one() {
        use crate::backend::Backend;
        let p = 67;
        let n = modulus(p);
        let mut be = crate::backend::cpu::CpuBackend::new(p).unwrap();
        // x = 3^E for B1 = 1000, as stage 1 leaves it.
        let e = stage1_exponent(p, 1000);
        let x1 = Residue::from_biguint(p, &BigUint::from(3u32).modpow(&e, &n));
        for b2 in [2676u64, 2677, 3000, 20_000] {
            let primes: Vec<u64> = primes_below(b2 + 1)
                .into_iter()
                .filter(|&q| q > 1000)
                .collect();
            let xr = be.alloc().unwrap();
            be.import(xr, &x1).unwrap();
            let poly = stage2_poly(&mut be, xr, &primes, &|| false)
                .unwrap()
                .unwrap();
            let paired = stage2_paired(&mut be, xr, &primes, &|| false)
                .unwrap()
                .unwrap();
            be.free(xr).unwrap();
            let (gp, gq) = (poly.gcd(&n), paired.gcd(&n));
            assert!((&gp % &gq).is_zero(), "B2 = {b2}: {gp} {gq}");
            // 193707721 needs the prime 2677. The giant steps of D = 2310
            // run to 3464 = 2310 + 1154 whatever B2, so even B2 = 2676,
            // where the paired gcd is 1, finds it.
            assert_eq!(gp.to_string(), "193707721", "B2 = {b2}");
        }
    }

    #[test]
    fn the_default_sub_matches_the_cpu_backend() {
        use crate::backend::Backend;
        let p = 127;
        let n = modulus(p);
        let mut cpu = crate::backend::cpu::CpuBackend::new(p).unwrap();
        let mut dwt = crate::backend::dwt::DwtBackend::new(p).unwrap();
        for (a, b) in [(5u64, 3u64), (3, 5), (0, 1), (7, 7)] {
            let want = (BigUint::from(a) + &n - BigUint::from(b)) % &n;
            for be in [&mut cpu as &mut dyn Backend, &mut dwt] {
                let (ra, rb) = (be.alloc().unwrap(), be.alloc().unwrap());
                be.import(ra, &Residue::from_u64(p, a)).unwrap();
                be.import(rb, &Residue::from_u64(p, b)).unwrap();
                be.sub(ra, rb).unwrap();
                assert_eq!(be.export(ra).unwrap().to_biguint(), want, "{a} - {b}");
                assert!(be.sub(ra, ra).is_err());
                be.free(ra).unwrap();
                be.free(rb).unwrap();
            }
        }
    }

    /// A GMP-ECM style save line built from this module's own stage 1 (on
    /// N = 2^67 - 1, X and Z written in hex) gives the same stage 2 result
    /// as a full run of that curve.
    #[test]
    fn ecm_stage2_from_a_save_file_matches_a_full_run() {
        let p = 67;
        let n = modulus(p);
        let m = ModN(&n);
        let (b1, b2) = (1000u64, 100_000u64);
        let primes: Vec<u64> = primes_below(b1 + 1);
        let mut agreed = 0;
        for sigma in 6..80u64 {
            let full = ecm(p, b1, b2, 1, sigma, &|| false).unwrap();
            let only1 = ecm(p, b1, 0, 1, sigma, &|| false).unwrap();
            let (a24, mut pt) = suyama_curve(&m, &BigUint::from(sigma)).unwrap();
            for &q in &primes {
                let mut qk = q;
                while let Some(x) = qk.checked_mul(q).filter(|&x| x <= b1) {
                    qk = x;
                }
                pt = ladder(&m, &pt, qk, &a24);
            }
            if only1.factor.is_some() {
                continue;
            }
            let line = format!(
                "# saved\nMETHOD=ECM; PARAM=0; SIGMA={sigma}; B1={b1}; N={n}; X=0x{:x}; Z=0x{:x}; CHECKSUM=1; PROGRAM=GMP-ECM 7.0.4;\n",
                pt.0, pt.1
            );
            let curves = parse_gmp_ecm_save(&line).unwrap();
            assert_eq!(curves.len(), 1);
            let o = ecm_stage2_gmp(p, &curves, b2, 0, None, &|| false).unwrap();
            assert_eq!(o.factor, full.factor, "sigma {sigma}");
            agreed += usize::from(o.factor.is_some());
        }
        assert!(agreed > 0, "no curve found a factor in stage 2");
        assert!(parse_gmp_ecm_save("METHOD=ECM; PARAM=1; SIGMA=7; B1=10; N=31; X=3;").is_err());
        assert!(parse_gmp_ecm_save("METHOD=ECM; SIGMA=7; B1=10; N=31;").is_err());
        let l = parse_factor_line(
            "ECMSTAGE2=0123456789ABCDEF0123456789ABCDEF,1,2,67,-1,\"save, 1.txt\",50000,2,3,\"193707721\"",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            l.work,
            FactorWork::EcmStage2 {
                file: "save, 1.txt".into(),
                b2: 50000,
                skip: 2,
                curves: Some(3),
            }
        );
        assert_eq!(l.known_factors, ["193707721"]);
        assert!(l.aid.is_some());
        let l = parse_factor_line("ECMSTAGE2=1,2,67,-1,s.txt")
            .unwrap()
            .unwrap();
        assert!(matches!(
            l.work,
            FactorWork::EcmStage2 {
                b2: 0,
                skip: 0,
                curves: None,
                ..
            }
        ));
        assert!(parse_factor_line("ECMSTAGE2N=s.txt").is_err());
        assert!(parse_factor_line("ECMSTAGE2=1,2,67,-1").is_err());
        assert!(parse_factor_line("ECMSTAGE2=1,2,67,-1,s.txt,\"7\"").is_err());
    }

    #[test]
    fn ecm_finds_a_small_factor_and_checks_it() {
        // 2^67 - 1 = 193707721 * 761838257287: about a quarter of the curves
        // at B1 = 2000 find the 28-bit factor (the 40-bit one is rarer).
        let o = ecm(67, 2000, 0, 40, 6, &|| false).unwrap();
        assert_eq!(o.b2, 0);
        assert!(
            matches!(o.factor.as_deref(), Some("193707721" | "761838257287")),
            "{o:?}"
        );
        // M61 is prime: nothing to find, and nothing false is reported.
        let o = ecm(61, 500, 5000, 3, 6, &|| false).unwrap();
        assert!(o.factor.is_none() && o.curves == 3);
    }

    #[test]
    fn ecm_stage_2_finds_at_least_what_stage_1_finds() {
        // Stage 1 is the same with or without stage 2, so stage 2 can only
        // find a factor on the same curve or an earlier one.
        let s1 = ecm(67, 300, 0, 60, 6, &|| false).unwrap();
        let s2 = ecm(67, 300, 30_000, 60, 6, &|| false).unwrap();
        if let Some(at) = s1.found_sigma {
            assert!(s2.found_sigma.is_some_and(|b| b <= at), "{s1:?} {s2:?}");
        }
        if let Some(f) = s2.factor.as_deref() {
            assert!(matches!(f, "193707721" | "761838257287"), "{s2:?}");
        }
    }

    #[test]
    fn montgomery_ladder_matches_repeated_addition() {
        let n = modulus(61);
        let m = ModN(&n);
        let a24 = BigUint::from(12345u32);
        let p: Point = (BigUint::from(7u32), BigUint::one());
        // 5P by the ladder against 2(2P) + P with explicit differences.
        let p2 = xdbl(&m, &p, &a24);
        let p3 = xadd(&m, &p2, &p, &p);
        let p5 = xadd(&m, &p3, &p2, &p);
        let l = ladder(&m, &p, 5, &a24);
        // Compare X/Z projectively: X1 Z2 = X2 Z1.
        assert_eq!(m.mul(&l.0, &p5.1), m.mul(&p5.0, &l.1));
    }

    #[test]
    fn pp1_finds_a_factor_with_smooth_q_plus_1() {
        // M29 = 233 * 1103 * 2089. With B1 = 13 (E includes 2^3, 3^2, 13,
        // and 2 * 29), both 233 - 1 = 2^3 * 29 and 233 + 1 = 2 * 3^2 * 13
        // divide E, so 233 is found whatever the sign of (D/233); 1103 is
        // smooth on neither side, so the gcd is never N.
        let mut found = false;
        for run in 1..=2 {
            let o = pp1(29, 13, 13, run, &|| false).unwrap();
            if let Some(f) = &o.factor {
                assert!(crate::worktodo::check_factor(29, f).unwrap().accepted());
                found = true;
            }
            assert!(o.complete);
        }
        assert!(found);
        let o = pp1(67, 100, 100, 1, &|| false).unwrap();
        assert_eq!(o.start, (2, 7));
        assert_eq!(pp1_start(2), (6, 5));
        // V_n(P) mod a large modulus matches the recurrence.
        let n = modulus(61);
        let m = ModN(&n);
        let pv = BigUint::from(5u32);
        let (mut v0, mut v1) = (BigUint::from(2u32), pv.clone());
        for k in 1..40u32 {
            assert_eq!(lucas_v(&m, &pv, &BigUint::from(k)), v1, "k = {k}");
            let v2 = m.sub(&m.mul(&pv, &v1), &v0);
            v0 = v1;
            v1 = v2;
        }
    }

    #[test]
    fn factoring_lines_parse() {
        let a = parse_factor_line("Factor=0123456789ABCDEF0123456789ABCDEF,67,20,40")
            .unwrap()
            .unwrap();
        assert_eq!(
            a.work,
            FactorWork::Tf {
                bit_lo: 20,
                bit_hi: 40
            }
        );
        assert!(a.aid.is_some());
        let a = parse_factor_line("Pminus1=1,2,67,-1,1000,3000")
            .unwrap()
            .unwrap();
        assert_eq!(a.work, FactorWork::Pm1 { b1: 1000, b2: 3000 });
        assert!(parse_factor_line("Pfactor=1,2,67,-1,70,1")
            .unwrap()
            .is_some());
        assert_eq!(parse_factor_line("PRP=1,2,67,-1").unwrap(), None);
        assert!(parse_factor_line("Factor=67,40,20").is_err());
        assert!(parse_factor_line("Factor=68,20,40").is_err());
        // M67 = 193707721 * 761838257287.
        let a = parse_factor_line("Pminus1=1,2,67,-1,1000,3000,\"193707721\"")
            .unwrap()
            .unwrap();
        assert_eq!(a.known_factors, ["193707721"]);
        assert_eq!(a.new_part("193707721"), None);
        assert_eq!(
            a.new_part(&(193_707_721u128 * 761_838_257_287).to_string()),
            Some("761838257287".into())
        );
        assert!(parse_factor_line("Pminus1=1,2,67,-1,1000,3000,\"7\"").is_err());
        assert!(parse_factor_line("Pminus1=1,2,67,1,1000,3000").is_err());
        assert_eq!(
            parse_factor_line("B1=50000,B2=5000000;Pplus1=N/A,1,2,67,-1,1000,3000,2")
                .unwrap()
                .unwrap()
                .work,
            FactorWork::Pp1 {
                b1: 1000,
                b2: 3000,
                nth_run: 2
            }
        );
        assert_eq!(
            parse_factor_line("ECM2=1,2,67,-1,1000")
                .unwrap()
                .unwrap()
                .work,
            FactorWork::Ecm {
                b1: 1000,
                b2: 100_000,
                curves: 100,
                sigma: None
            }
        );
        assert_eq!(
            parse_factor_line("factor=M67,20,40")
                .unwrap()
                .unwrap()
                .exponent,
            67
        );
        assert!(parse_factor_line("Pfactor=67,60,1").unwrap().is_some());
        assert_eq!(parse_factor_line("// mfaktc comment").unwrap(), None);
        assert_eq!(
            with_pm1_done(
                "PRP=ABCDEF0123456789ABCDEF0123456789,1,2,1279,-1,70,1.3",
                1279
            )
            .unwrap(),
            "PRP=ABCDEF0123456789ABCDEF0123456789,1,2,1279,-1,70,0"
        );
    }
}
