// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The wide field: exact NTT arithmetic modulo 2^p - 1 over GF(q^2) with
//! q = 2^61 - 1, for exponents past the end of the GF((2^31 - 1)^2) field
//! (40,894,464). The kernels are in `kernels61.wgsl`; this module holds the
//! host model they are compared against, the plan, and the tables.
//!
//! The construction is the one in [`super::model`] with q = 2^61 - 1:
//! q = 3 mod 4, the group of GF(q^2) has order 2^62 (2^60 - 1), twiddles
//! of order n <= 2^61 have norm 1 (so the conjugate inverts them), and 2
//! has order 61 in GF(q), so the n-th root of 2 is 2^(n^-1 mod 61) and
//! every weight is a rotation of 61 bits.
//!
//! # Capacity
//!
//! With the same word bound W_k = 2^(w_k - 1) + 1 and the same Cauchy and
//! Schwarz argument, |z_k| <= 2 S, and a plan is accepted when
//! 2 S <= 2^59, half of the exact limit (q - 1) / 2. That allows words of
//! about 17 bits at n = 2^23 (p up to about 150 million) instead of the
//! 2 bits the 31-bit field allows there.
//!
//! # Splitting the coefficients
//!
//! A coefficient z_k does not fit the i32 words of the carry passes, so the
//! unweight pass splits it into D balanced digits: digit j has the width of
//! word k + j and is added (atomically on the GPU) into that word, the
//! carries entering the next digit exactly. After m digits the remainder is
//! at most |z| / 2^(m w_min) + 1 in magnitude, so with
//! D = ceil(bits(2 S) / w_min) + 1 the last term is at most 2. Every word
//! then holds at most D (2^(w_max - 1)) + 2, and the ordinary carry passes
//! (the count from [`super::model::carry_passes`] for that bound) bring it
//! to the word bound. Word k receives digits from coefficients k, k - 1,
//! ..., k - D + 1, which is the same value modulo 2^p - 1 as carrying z
//! along the words.

use super::model::{carry_passes, Plan, N_MIN, W_MIN};

/// q = 2^61 - 1.
pub const Q: u64 = (1 << 61) - 1;
/// Accepted bound on |z_k|: half of (q - 1) / 2.
pub const Z_LIMIT: u128 = 1 << 59;
/// Largest wide transform length. Its transform buffers are 16 bytes per
/// element (256 MiB at 2^24); the backend checks the adapter limits.
pub const N_MAX: u32 = 1 << 24;
/// Widest word a wide plan may use.
const W_MAX: u32 = 30;

pub fn add(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= Q {
        s - Q
    } else {
        s
    }
}

pub fn sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + Q - b
    }
}

pub fn mul(a: u64, b: u64) -> u64 {
    let p = u128::from(a) * u128::from(b);
    let s = (p as u64 & Q) + (p >> 61) as u64;
    let s = (s & Q) + (s >> 61);
    if s >= Q {
        s - Q
    } else {
        s
    }
}

/// v 2^s mod q for v < q and s < 61.
pub fn rot(v: u64, s: u32) -> u64 {
    if s == 0 {
        return v;
    }
    ((v << s) | (v >> (61 - s))) & Q
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G2 {
    pub re: u64,
    pub im: u64,
}

impl G2 {
    pub const ONE: G2 = G2 { re: 1, im: 0 };
    pub fn cadd(self, o: G2) -> G2 {
        G2 {
            re: add(self.re, o.re),
            im: add(self.im, o.im),
        }
    }
    pub fn csub(self, o: G2) -> G2 {
        G2 {
            re: sub(self.re, o.re),
            im: sub(self.im, o.im),
        }
    }
    pub fn cmul(self, o: G2) -> G2 {
        G2 {
            re: sub(mul(self.re, o.re), mul(self.im, o.im)),
            im: add(mul(self.re, o.im), mul(self.im, o.re)),
        }
    }
    pub fn sqr(self) -> G2 {
        let t = mul(self.re, self.im);
        G2 {
            re: sub(mul(self.re, self.re), mul(self.im, self.im)),
            im: add(t, t),
        }
    }
    pub fn conj(self) -> G2 {
        G2 {
            re: self.re,
            im: sub(0, self.im),
        }
    }
    pub fn pow(self, mut e: u64) -> G2 {
        let (mut acc, mut b) = (G2::ONE, self);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.cmul(b);
            }
            b = b.sqr();
            e >>= 1;
        }
        acc
    }
}

/// i32 word to GF(q).
pub fn to_field(x: i32) -> u64 {
    if x >= 0 {
        x as u64
    } else {
        Q - u64::from(x.unsigned_abs())
    }
}

/// Canonical v in [0, q) to the balanced integer.
pub fn balanced(v: u64) -> i64 {
    if v > Q / 2 {
        v as i64 - Q as i64
    } else {
        v as i64
    }
}

/// A primitive 2^62-th root of unity: the first (k + i), k >= 1, whose
/// (2^60 - 1)-th power has exact order 2^62.
pub fn root_2_62() -> G2 {
    let minus_one = G2 { re: Q - 1, im: 0 };
    (1..)
        .map(|k| G2 { re: k, im: 1 }.pow((1u64 << 60) - 1))
        .find(|h| h.pow(1u64 << 61) == minus_one)
        .expect("a primitive 2^62-th root exists")
}

/// Twiddle table w_n^j for j < n / 2.
pub fn twiddles(n: u32) -> Vec<G2> {
    let w = root_2_62().pow((1u64 << 62) / u64::from(n));
    let mut t = Vec::with_capacity((n / 2) as usize);
    let mut x = G2::ONE;
    for _ in 0..n / 2 {
        t.push(x);
        x = x.cmul(w);
    }
    t
}

/// The proven coefficient bound 2 S for (p, n) in the wide field.
pub fn capacity_bound(p: u32, n: u32) -> Option<u128> {
    let w_min = p / n;
    if !(W_MIN..W_MAX).contains(&w_min) {
        return None;
    }
    let n_max = u128::from(p % n);
    let wb = |w: u32| ((1u128 << (w - 1)) + 1).pow(2);
    let s = n_max * wb(w_min + 1) + (u128::from(n) - n_max) * wb(w_min);
    Some(2 * s)
}

/// Smallest wide transform length for p.
pub fn choose_n(p: u32) -> Option<u32> {
    let mut n = N_MIN;
    while n <= N_MAX {
        if capacity_bound(p, n).is_some_and(|z| z <= Z_LIMIT) {
            return Some(n);
        }
        n *= 2;
    }
    None
}

/// Largest p with a wide plan at length n.
pub fn edge(n: u32) -> Option<u32> {
    let ok = |p: u32| capacity_bound(p, n).is_some_and(|z| z <= Z_LIMIT);
    let (mut a, mut b) = (
        W_MIN * n,
        (u64::from(W_MAX) * u64::from(n)).min(u64::from(u32::MAX)) as u32,
    );
    if !ok(a) {
        return None;
    }
    while a < b {
        let mid = a + (b - a).div_ceil(2);
        if ok(mid) {
            a = mid;
        } else {
            b = mid - 1;
        }
    }
    Some(a)
}

/// Largest exponent with a wide plan.
pub fn top() -> u32 {
    let mut hi = 0;
    let mut n = N_MIN;
    while n <= N_MAX {
        hi = hi.max(edge(n).unwrap_or(0));
        n *= 2;
    }
    hi
}

/// Digit count D for a coefficient bound and the narrowest word.
pub fn digits(z_bound: u128, w_min: u32) -> u32 {
    let bits = 128 - z_bound.leading_zeros();
    bits.div_ceil(w_min) + 1
}

/// a^e mod m for small values.
fn pow_mod(a: u64, mut e: u64, m: u64) -> u64 {
    let (mut acc, mut b) = (1 % m, a % m);
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    acc
}

impl Plan {
    /// The wide field plan for p, if p has one.
    pub fn new_wide(p: u32) -> Option<Plan> {
        let n = choose_n(p)?;
        let z = capacity_bound(p, n)?;
        let log_n = n.trailing_zeros();
        let (p64, n64) = (u64::from(p), u64::from(n));
        let pos = |k: u64| (p64 * k).div_ceil(n64);
        // n^-1 mod 61 (Fermat).
        let kinv = pow_mod(n64 % 61, 59, 61);
        let (w_min, w_max) = (p / n, p.div_ceil(n));
        let d = digits(z, w_min);
        let mut plan = Plan {
            p,
            n,
            log_n,
            bitpos: Vec::with_capacity(n as usize),
            width: Vec::with_capacity(n as usize),
            wshift: Vec::with_capacity(n as usize),
            ushift: Vec::with_capacity(n as usize),
            w_min,
            w_max,
            z_bound: u64::try_from(z).ok()?,
            passes: 0,
            wide: true,
            digits: d,
        };
        for k in 0..n64 {
            let b = pos(k);
            let e = n64 * b - p64 * k;
            let s = ((kinv * e) % 61) as u32;
            plan.bitpos.push(b as u32);
            plan.width.push((pos(k + 1) - b) as u32);
            plan.wshift.push(s);
            plan.ushift.push((122 - s - log_n % 61) % 61);
        }
        // Every word after the digit split, and every carry pass value,
        // must fit i32.
        let first = u64::from(d) * (1u64 << (w_max - 1)) + 2;
        if first + (1u64 << w_max) >= 1 << 31 {
            return None;
        }
        plan.passes = carry_passes(first, w_min, w_max);
        Some(plan)
    }
}

// ---------------------------------------------------------------------------
// The stages of one operation, as the wide kernels compute them.

pub fn weight(plan: &Plan, x: &[i32]) -> Vec<G2> {
    x.iter()
        .zip(&plan.wshift)
        .map(|(&v, &s)| G2 {
            re: rot(to_field(v), s),
            im: 0,
        })
        .collect()
}

pub fn forward(plan: &Plan, t: &mut [G2], tw: &[G2]) {
    let n = plan.n as usize;
    let mut m = n / 2;
    while m >= 1 {
        let stride = n / (2 * m);
        for u in 0..n / 2 {
            let (blk, j) = (u / m, u % m);
            let i0 = blk * 2 * m + j;
            let (a, b) = (t[i0], t[i0 + m]);
            t[i0] = a.cadd(b);
            t[i0 + m] = a.csub(b).cmul(tw[j * stride]);
        }
        m /= 2;
    }
}

pub fn inverse(plan: &Plan, t: &mut [G2], tw: &[G2]) {
    let n = plan.n as usize;
    let mut m = 1;
    while m < n {
        let stride = n / (2 * m);
        for u in 0..n / 2 {
            let (blk, j) = (u / m, u % m);
            let i0 = blk * 2 * m + j;
            let a = t[i0];
            let b = t[i0 + m].cmul(tw[j * stride].conj());
            t[i0] = a.cadd(b);
            t[i0 + m] = a.csub(b);
        }
        m *= 2;
    }
}

/// Unweight, scale, and split every coefficient into digits added into the
/// following words; `sub` comes off word 0. Also reports a nonzero
/// imaginary part.
pub fn unweight(plan: &Plan, t: &[G2], sub: i32) -> (Vec<i32>, bool) {
    let n = plan.n as usize;
    let d = plan.digits as usize;
    let mut out = vec![0i32; n];
    let mut bad = false;
    for (i, (v, &s)) in t.iter().zip(&plan.ushift).enumerate() {
        bad |= v.im != 0;
        let mut z = balanced(rot(v.re, s));
        for j in 0..d - 1 {
            let idx = (i + j) % n;
            let w = plan.width[idx];
            let half = 1i64 << (w - 1);
            let lo = (z + half).rem_euclid(1 << w) - half;
            out[idx] += lo as i32;
            z = (z - lo) >> w;
        }
        out[(i + d - 1) % n] += z as i32;
    }
    out[0] -= sub;
    (out, bad)
}

fn finish(plan: &Plan, t: &mut [G2], tw: &[G2], sub: i32) -> Vec<i32> {
    inverse(plan, t, tw);
    let (mut z, bad) = unweight(plan, t, sub);
    assert!(!bad, "nonzero imaginary part in the model");
    for _ in 0..plan.passes {
        z = plan.carry_pass(&z);
    }
    z
}

/// x^2 - sub.
pub fn square(plan: &Plan, x: &[i32], tw: &[G2], sub: i32) -> Vec<i32> {
    let mut t = weight(plan, x);
    forward(plan, &mut t, tw);
    for v in t.iter_mut() {
        *v = v.sqr();
    }
    finish(plan, &mut t, tw, sub)
}

/// x y.
pub fn mul_words(plan: &Plan, x: &[i32], y: &[i32], tw: &[G2]) -> Vec<i32> {
    let mut t = weight(plan, x);
    let mut u = weight(plan, y);
    forward(plan, &mut t, tw);
    forward(plan, &mut u, tw);
    for (a, b) in t.iter_mut().zip(&u) {
        *a = a.cmul(*b);
    }
    finish(plan, &mut t, tw, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residue::Residue;
    use num_bigint::BigUint;

    #[test]
    fn field_and_roots() {
        let a = Q - 5;
        assert_eq!(mul(a, a), 25);
        assert_eq!(mul(1 << 60, 2), 1);
        assert_eq!(rot(1, 60), 1 << 60);
        assert_eq!(rot(1 << 60, 1), 1);
        let h = root_2_62();
        assert_eq!(h.pow(1u64 << 62), G2::ONE);
        assert_ne!(h.pow(1u64 << 61), G2::ONE);
        // Norm 1 twiddles: the conjugate is the inverse.
        let w = twiddles(1 << 10)[3];
        assert_eq!(w.cmul(w.conj()), G2::ONE);
        // n^-1 mod 61 makes 2^(kinv n) = 2.
        for log_n in [2u32, 10, 22, 24] {
            let n = 1u64 << log_n;
            let kinv = pow_mod(n % 61, 59, 61);
            assert_eq!(kinv * n % 61, 1);
        }
    }

    /// The model squares and multiplies exactly, against big integers,
    /// for small and mid size exponents with the wide field forced.
    #[test]
    fn model_matches_bigint() {
        for p in [127u32, 1279, 4423, 86243, 1_398_269] {
            let plan = Plan::new_wide(p).unwrap();
            assert!(plan.wide && plan.digits >= 2);
            let tw = twiddles(plan.n);
            let n = (BigUint::from(1u32) << p) - 1u32;
            let x = BigUint::from(3u32).modpow(&BigUint::from(1_000_003u32), &n);
            let y = BigUint::from(7u32).modpow(&BigUint::from(999_983u32), &n);
            let (rx, ry) = (Residue::from_biguint(p, &x), Residue::from_biguint(p, &y));
            let (wx, wy) = (plan.words_from_residue(&rx), plan.words_from_residue(&ry));
            let sq = square(&plan, &wx, &tw, 2);
            let want = (&x * &x + &n - 2u32) % &n;
            assert_eq!(
                plan.residue_from_words(&sq).unwrap().to_biguint(),
                want,
                "M{p}"
            );
            let pr = mul_words(&plan, &wx, &wy, &tw);
            assert_eq!(
                plan.residue_from_words(&pr).unwrap().to_biguint(),
                &x * &y % &n,
                "M{p}"
            );
        }
    }

    /// Wide plans cover past the narrow field's end, contiguously.
    #[test]
    fn wide_range() {
        let narrow = super::super::model::supported_range().1;
        assert!(top() > 140_000_000, "{}", top());
        let mut n = N_MIN;
        let mut prev: Option<u32> = None;
        while n <= N_MAX {
            if let Some(e) = edge(n) {
                if let Some(pe) = prev {
                    assert_eq!(choose_n(pe + 1), Some(n), "gap after {pe}");
                }
                prev = Some(e);
            }
            n *= 2;
        }
        let p = narrow + 1;
        let plan = Plan::new_wide(p).unwrap();
        assert!(plan.n <= 1 << 22, "n = {}", plan.n);
        assert!(plan.passes <= 8);
        // Every length's largest exponent has a plan, so the i32 guard in
        // new_wide never cuts the range.
        let mut n = N_MIN;
        while n <= N_MAX {
            if let Some(e) = edge(n) {
                assert!(Plan::new_wide(e).is_some(), "p={e}");
            }
            n *= 2;
        }
    }
}
