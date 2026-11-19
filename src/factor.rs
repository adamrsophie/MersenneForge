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
//! above it; P-1 stage 1 runs on any backend, stage 2 runs on the host with
//! arbitrary-precision arithmetic and is practical only for small exponents.

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
    let sieve: Vec<u64> = if two_p * k_lo + 1 > u128::from(SIEVE_BOUND) {
        primes_below(SIEVE_BOUND)
            .into_iter()
            .skip(1) // 2: every q is odd
            .filter(|&s| s != u64::from(p))
            .collect()
    } else {
        Vec::new()
    };
    let mut factors = Vec::new();
    let mut bad = vec![false; SEGMENT];
    while k0 < k_end {
        if stop() {
            return Ok(TfOutcome {
                exponent: p,
                bit_lo,
                bit_hi,
                factors,
                complete: false,
                next_k: k0,
            });
        }
        let len = ((k_end - k0).min(SEGMENT as u128)) as usize;
        bad[..len].fill(false);
        for &s in &sieve {
            // q = 2kp + 1 = 0 mod s  <=>  k = -(2p)^-1 mod s.
            let s128 = u128::from(s);
            let inv = mod_inverse((two_p % s128) as u64, s).expect("s does not divide 2p");
            let root = (s - inv) % s;
            let first = (u128::from(root) + s128 - k0 % s128) % s128;
            let mut i = first as usize;
            while i < len {
                bad[i] = true;
                i += s as usize;
            }
        }
        for (i, _) in bad[..len].iter().enumerate().filter(|(_, b)| !**b) {
            let q = two_p * (k0 + i as u128) + 1;
            if !matches!(q & 7, 1 | 7) {
                continue;
            }
            let divides = match u64::try_from(q) {
                Ok(q64) => pow2_mod(p, q64) == 1,
                // ponytail: arbitrary precision above 2^64 is slow; a two-limb
                // Montgomery multiply (or a GPU kernel) is the upgrade path.
                Err(_) => {
                    let qb = BigUint::from(q);
                    BigUint::from(2u32).modpow(&BigUint::from(p), &qb) == BigUint::one()
                }
            };
            if divides {
                let f = q.to_string();
                let c = crate::worktodo::check_factor(p, &f)?;
                if !c.accepted() {
                    return Err(format!("internal error: {f} failed the factor check"));
                }
                factors.push(f);
            }
        }
        k0 += len as u128;
        progress(k0);
    }
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
/// b1 < q <= b2 on the host. `stop` is polled about every 10,000 stage 1
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
    // ponytail: host stage 2 with one modular multiply per prime; a device
    // stage 2 needs a subtract operation on the backend and a prime-pairing
    // plan, and is the upgrade path for large exponents.
    let primes: Vec<u64> = primes_below(b2 + 1)
        .into_iter()
        .filter(|&q| q > b1)
        .collect();
    let Some(&q0) = primes.first() else {
        out.complete = true;
        return Ok(out);
    };
    let mut gaps: std::collections::HashMap<u64, BigUint> = Default::default();
    let mut y = xb.modpow(&BigUint::from(q0), &n);
    let mut acc = minus1(&y);
    let mut last = q0;
    for (j, &q) in primes.iter().enumerate().skip(1) {
        let d = q - last;
        let xd = gaps
            .entry(d)
            .or_insert_with(|| xb.modpow(&BigUint::from(d), &n));
        y = &y * &*xd % &n;
        acc = acc * minus1(&y) % &n;
        last = q;
        if j % 1000 == 0 && stop() {
            out.notes.push(format!("stopped in stage 2 at q = {q}"));
            return Ok(out);
        }
    }
    if let Some(f) = factor_from(p, &acc, &n, &mut out.notes) {
        out.factor = Some(f);
        out.stage = Some(2);
    }
    out.complete = true;
    Ok(out)
}

/// What a factoring line asks for.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum FactorWork {
    /// `Factor=[AID,]p,bit_lo,bit_hi`
    Tf { bit_lo: u32, bit_hi: u32 },
    /// `Pminus1=[AID,]1,2,p,-1,B1,B2` or `Pfactor=[AID,]1,2,p,-1,tf_bits,tests_saved`
    /// (Pfactor bounds are chosen here from the line's fields).
    Pm1 { b1: u64, b2: u64 },
}

/// One factoring assignment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FactorLine {
    pub work: FactorWork,
    pub exponent: u32,
    pub aid: Option<String>,
    pub line: String,
}

/// Bounds for a `Pfactor=` line and for P-1 before a PRP test: stage 1
/// only (B2 = B1), since stage 2 runs on the host. ponytail: a fixed rule,
/// B1 = p / 10 between 1,000 and 1,000,000; upstream's cost model
/// (probability of a factor against tests saved) is the upgrade path.
pub fn pfactor_bounds(p: u32) -> (u64, u64) {
    let b1 = (u64::from(p) / 10).clamp(1000, 1_000_000);
    (b1, b1)
}

/// Largest exponent `work` runs P-1 for. Above it the final gcd on the
/// host (quadratic in p) and host stage 2 take too long, and the line
/// stays in place for another program.
pub const WORK_PM1_MAX_P: u32 = 10_000_000;

/// Parse a `Factor=`, `Pminus1=`, or `Pfactor=` line. `Ok(None)` for any
/// other line (including comments), so callers try this parser first.
pub fn parse_factor_line(raw: &str) -> Result<Option<FactorLine>, String> {
    let line = raw.trim_end_matches(['\r', '\n']);
    let Some((kind, rest)) = line.split_once('=') else {
        return Ok(None);
    };
    if !matches!(kind, "Factor" | "Pminus1" | "Pfactor") {
        return Ok(None);
    }
    if line.len() > crate::worktodo::MAX_LINE || rest.contains('=') {
        return Err(format!("malformed line: {line:?}"));
    }
    let mut parts: Vec<&str> = rest.split(',').map(str::trim).collect();
    if parts.first().is_some_and(|f| f.is_empty() || *f == "N/A") {
        parts.remove(0);
    }
    let mut aid = None;
    if parts
        .first()
        .is_some_and(|f| f.len() == 32 && f.chars().all(|c| c.is_ascii_hexdigit()))
    {
        aid = Some(parts.remove(0).to_string());
    }
    let bad = || format!("malformed {kind} line: {line:?}");
    let int = |s: &str| -> Result<u64, String> {
        s.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| s.parse().ok())
            .flatten()
            .ok_or_else(bad)
    };
    let (p, work) = match kind {
        "Factor" => {
            let [p, lo, hi] = parts[..] else {
                return Err(bad());
            };
            let (lo, hi) = (int(lo)? as u32, int(hi)? as u32);
            (
                int(p)?,
                FactorWork::Tf {
                    bit_lo: lo,
                    bit_hi: hi,
                },
            )
        }
        _ => {
            if parts.len() < 6 || parts[..2] != ["1", "2"] || parts[3] != "-1" {
                return Err(bad());
            }
            let p = int(parts[2])?;
            if kind == "Pminus1" {
                if parts.len() > 6 {
                    return Err(format!(
                        "known factors or extra fields are not supported: {line:?}"
                    ));
                }
                let b1 = int(parts[4])?;
                let b2 = int(parts[5])?;
                (p, FactorWork::Pm1 { b1, b2 })
            } else {
                let (b1, b2) = pfactor_bounds(u32::try_from(p).map_err(|_| bad())?);
                (p, FactorWork::Pm1 { b1, b2 })
            }
        }
    };
    let p = u32::try_from(p).map_err(|_| bad())?;
    crate::exponent::validate(p).map_err(|e| format!("{e}: {line:?}"))?;
    if let FactorWork::Tf { bit_lo, bit_hi } = work {
        if !(bit_lo < bit_hi && bit_hi <= 120) {
            return Err(format!("bad bit range: {line:?}"));
        }
    }
    Ok(Some(FactorLine {
        work,
        exponent: p,
        aid,
        line: line.to_string(),
    }))
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

fn program() -> serde_json::Value {
    serde_json::json!({
        "name": "mersenne-forge",
        "version": env!("CARGO_PKG_VERSION"),
        "port": crate::worktodo::port_code(),
        "os": {"os": std::env::consts::OS, "architecture": std::env::consts::ARCH},
    })
}

/// Known small factors, as a cross-check that TF misses nothing it should
/// find: (p, q) with q | 2^p - 1.
#[cfg(test)]
const KNOWN: &[(u32, u64)] = &[
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
        assert!(parse_factor_line("Pminus1=1,2,67,-1,1000,3000,\"193707721\"").is_err());
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
