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
        let cands: Vec<u128> = bad[..len]
            .iter()
            .enumerate()
            .filter(|(_, b)| !**b)
            .map(|(i, _)| two_p * (k0 + i as u128) + 1)
            .filter(|q| matches!(q & 7, 1 | 7))
            .collect();
        for q in test(p, &cands)? {
            if !cands.contains(&q) {
                return Err(format!("the tester returned {q}, which is not a candidate"));
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
    // Stage 2 on the backend: y = x^q walks the primes by multiplying with
    // cached powers x^d for each prime gap d, and acc collects (y - 1). The
    // backend has no subtraction, so y goes to the host once per prime for
    // the O(p) "- 1" and comes back; every multiplication stays on the
    // device. ponytail: no prime pairing or baby step giant step; one device
    // multiply plus one round trip per prime, upgrade when B2 grows large.
    let primes: Vec<u64> = primes_below(b2 + 1)
        .into_iter()
        .filter(|&q| q > b1)
        .collect();
    let Some(&q0) = primes.first() else {
        out.complete = true;
        return Ok(out);
    };
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
        let s = BigUint::from(sigma + c);
        let u = m.sub(&m.mul(&s, &s), &BigUint::from(5u32));
        let v = m.mul(&BigUint::from(4u32), &s);
        let u3 = m.mul(&m.mul(&u, &u), &u);
        let vmu = m.sub(&v, &u);
        // a24 = (v - u)^3 (3u + v) / (16 u^3 v)
        let num = m.mul(
            &m.mul(&m.mul(&vmu, &vmu), &vmu),
            &m.add(&m.mul(&BigUint::from(3u32), &u), &v),
        );
        let den = m.mul(&m.mul(&BigUint::from(16u32), &u3), &v);
        let g = den.gcd(&n);
        if !g.is_one() {
            if let Some(f) = factor_from(p, &den, &n, &mut out.notes) {
                out.factor = Some(f);
                out.found_sigma = Some(sigma + c);
                return Ok(out);
            }
            continue;
        }
        let inv = mod_inverse_big(&den, &n).expect("gcd is 1");
        let a24 = m.mul(&num, &inv);
        let mut pt: Point = (u3, m.mul(&m.mul(&v, &v), &v));
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
        // Stage 2, standard continuation: walk the odd multiples m Q with
        // (m + 2) Q = xADD(m Q, 2 Q; difference (m - 2) Q), one xADD per odd
        // m, and multiply Z(q Q) into acc for every prime q in (B1, B2]. A
        // factor f of N divides acc when the order of Q mod f is such a q.
        // ponytail: no baby step giant step pairing; about 3 (B2 - B1)
        // multiplications per curve.
        if let Some(&first) = stage2.first() {
            let two = xdbl(&m, &pt, &a24);
            let mut prev = ladder(&m, &pt, first - 2, &a24);
            let mut cur = ladder(&m, &pt, first, &a24);
            let mut mq = first;
            let mut acc = BigUint::one();
            let mut next = stage2.iter().peekable();
            while let Some(&&q) = next.peek() {
                if mq == q {
                    acc = m.mul(&acc, &cur.1);
                    next.next();
                    if next.peek().is_none() {
                        break;
                    }
                }
                let after = xadd(&m, &cur, &two, &prev);
                prev = std::mem::replace(&mut cur, after);
                mq += 2;
                if mq % 2_000_000 == 1 && stop() {
                    out.notes
                        .push(format!("stopped in stage 2 of curve sigma = {}", sigma + c));
                    return Ok(out);
                }
            }
            if let Some(f) = factor_from(p, &acc, &n, &mut out.notes) {
                out.curves += 1;
                out.factor = Some(f);
                out.found_sigma = Some(sigma + c);
                return Ok(out);
            }
        }
        out.curves += 1;
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

/// Bounds for a `Pfactor=` line and for P-1 before a PRP test: stage 1
/// only (B2 = B1): stage 2 pays a host round trip per prime and saves no
/// progress, so `work` does not run it. ponytail: a fixed rule,
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

/// Largest exponent `work` runs P+1 or ECM for: both run entirely on the
/// host with arbitrary precision arithmetic.
pub const WORK_HOST_MAX_P: u32 = 1_000_000;

/// Largest exponent `work` takes a line of this kind for. Larger lines
/// stay in place for another program.
pub fn work_max_p(w: &FactorWork) -> u32 {
    match w {
        FactorWork::Tf { .. } => u32::MAX,
        FactorWork::Pm1 { .. } => WORK_PM1_MAX_P,
        FactorWork::Pp1 { .. } | FactorWork::Ecm { .. } => WORK_HOST_MAX_P,
    }
}

/// Parse a `Factor=`, `Pminus1=`, `Pfactor=`, `Pplus1=`, `ECM=`, or
/// `ECM2=` line in Prime95's forms (see `FactorWork`). `Ok(None)` for any
/// other line (including comments), so callers try this parser first.
/// Only Mersenne numbers (1,2,p,-1) are accepted; known factors must each
/// divide 2^p - 1.
pub fn parse_factor_line(raw: &str) -> Result<Option<FactorLine>, String> {
    let line = raw.trim_end_matches(['\r', '\n']);
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
            let [p, _, _] = parts[..] else {
                return Err(bad());
            };
            let p = int(p)?;
            let (b1, b2) = pfactor_bounds(u32::try_from(p).map_err(|_| bad())?);
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
            let (b1, b2) = pfactor_bounds(u32::try_from(p).map_err(|_| bad())?);
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
