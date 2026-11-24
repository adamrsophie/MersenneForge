// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Host model of the GPU arithmetic, plan selection, and the exact host
//! conversions between canonical residues and balanced words.
//!
//! Every function in the "shader mirror" part below is a line by line copy of
//! the WGSL in `kernels.wgsl` (same u32 limb products, same rotations, same
//! carry rule), so the model can be validated against exact bigint arithmetic
//! and the GPU can then be compared against the model stage by stage. The
//! plan, the twiddle table and the per-word table come from here and are used
//! by the real backend.
//!
//! # Transform
//!
//! Field GF(q^2) with q = 2^31 - 1 and i^2 = -1 (q = 3 mod 4, so x^2 + 1 is
//! irreducible). The multiplicative group has order q^2 - 1 = 2^32 (2^30 - 1),
//! so a primitive 2^32-th root of unity h exists and w_n = h^(2^32 / n) is a
//! primitive n-th root for every power of two n. For n <= 2^31 the norm of
//! w_n is 1, so w_n^-1 is the conjugate of w_n; one twiddle table serves both
//! directions.
//!
//! # Word layout and weights
//!
//! Word k sits at bit b_k = ceil(p k / n) and has w_k = b_{k+1} - b_k bits,
//! which is floor(p / n) or ceil(p / n). Its weight is r^(e_k) with
//! e_k = n b_k - p k in [0, n) and r an n-th root of 2. In GF(q), 2 has order
//! 31, so r = 2^(n^-1 mod 31) works and every weight is a power of two: the
//! weighting is a rotation of a 31-bit value. For a product term x_i y_j with
//! i + j = k (mod n), the weights combine to exactly 2^c with
//! c = b_i + b_j - b_k (minus p on wraparound), and c is 0 or 1. So the
//! unweighted, n^-1 scaled inverse transform returns exactly the integers
//! z_k = sum x_i y_j 2^c, reduced mod q. They are recovered exactly as long
//! as |z_k| <= (q - 1) / 2.
//!
//! # Capacity bound
//!
//! Between operations every word satisfies |x_k| <= W_k = 2^(w_k - 1) + 1
//! (see the carry section for the +1). By Cauchy and Schwarz, and since
//! i -> k - i is a permutation and 2^c <= 2,
//! |z_k| <= 2 sum_i |x_i| |y_(k-i)| <= 2 S, with S = sum_k W_k^2.
//! A plan is accepted only if 2 S <= Z_LIMIT = 2^29, which is half of the
//! exact recovery limit 2^30 - 1 (one bit of explicit margin).
//!
//! # Carry
//!
//! A carry pass maps every word in parallel to
//! x'_k = lo(v_k) + hi(v_(k-1)), with hi(v) = floor((v + 2^(w-1)) / 2^w) and
//! lo(v) = v - hi(v) 2^w in [-2^(w-1), 2^(w-1)). The carry out of word n - 1
//! enters word 0 because 2^p = 1 mod N. A pass preserves the value mod N.
//! If every carry has magnitude at most H_t in pass t, then
//! |x_k| <= 2^(w_k - 1) + H_t afterwards, and the next carries satisfy
//! H_(t+1) <= 1 + floor(H_t / 2^w_min). The first pass sees |v| <= 2 S + 2
//! (the 2 is the Lucas-Lehmer subtraction). [`carry_passes`] iterates this
//! recurrence and stops at the first pass with H_t <= 1, which gives exactly
//! the invariant |x_k| <= 2^(w_k - 1) + 1 assumed by the capacity bound.

use crate::backend::BackendError;
use crate::residue::Residue;

/// q = 2^31 - 1.
pub const Q: u32 = 0x7fff_ffff;
/// Accepted bound on |z_k| (2^29): half of the exact limit (q - 1) / 2.
pub const Z_LIMIT: u64 = 1 << 29;
/// Smallest transform length.
pub const N_MIN: u32 = 4;
/// Largest transform length. Its largest buffer (n elements of 8 bytes,
/// 128 MiB) just fits the WebGPU default storage binding limit of 128 MiB;
/// the backend also checks the real adapter limits. Above 2^24 no plan has
/// words of at least `W_MIN` bits, so a longer transform needs another field.
pub const N_MAX: u32 = 1 << 24;
/// Smallest word width a plan may use (keeps the carry pass count bounded).
pub const W_MIN: u32 = 2;

// ---------------------------------------------------------------------------
// Shader mirror: GF(q) and GF(q^2) arithmetic, identical to kernels.wgsl.

pub fn add(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= Q {
        s - Q
    } else {
        s
    }
}

pub fn sub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + Q - b
    }
}

/// a b mod q for a, b < q, using only u32 products of 16-bit limbs.
pub fn mul(a: u32, b: u32) -> u32 {
    let (a0, a1) = (a & 0xffff, a >> 16);
    let (b0, b1) = (b & 0xffff, b >> 16);
    let lo = a0 * b0;
    let mid = a1 * b0 + a0 * b1; // < 2^32 because a1, b1 < 2^15
    let hi = a1 * b1;
    let l = lo.wrapping_add(mid << 16);
    let c = u32::from(l < lo);
    let h = hi + (mid >> 16) + c; // the 62-bit product is h 2^32 + l
    let s = ((h << 1) | (l >> 31)) + (l & Q); // product >> 31 plus low 31 bits
    let s = (s & Q) + (s >> 31);
    if s >= Q {
        s - Q
    } else {
        s
    }
}

/// v 2^s mod q for v < q and s < 31: a 31-bit rotation.
pub fn rot(v: u32, s: u32) -> u32 {
    ((v << s) | (v >> (31 - s))) & Q
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F2 {
    pub re: u32,
    pub im: u32,
}

impl F2 {
    pub const ONE: F2 = F2 { re: 1, im: 0 };
    pub fn cadd(self, o: F2) -> F2 {
        F2 {
            re: add(self.re, o.re),
            im: add(self.im, o.im),
        }
    }
    pub fn csub(self, o: F2) -> F2 {
        F2 {
            re: sub(self.re, o.re),
            im: sub(self.im, o.im),
        }
    }
    pub fn cmul(self, o: F2) -> F2 {
        F2 {
            re: sub(mul(self.re, o.re), mul(self.im, o.im)),
            im: add(mul(self.re, o.im), mul(self.im, o.re)),
        }
    }
    pub fn sqr(self) -> F2 {
        let t = mul(self.re, self.im);
        F2 {
            re: sub(mul(self.re, self.re), mul(self.im, self.im)),
            im: add(t, t),
        }
    }
    pub fn conj(self) -> F2 {
        F2 {
            re: self.re,
            im: sub(0, self.im),
        }
    }
    pub fn pow(self, mut e: u64) -> F2 {
        let (mut acc, mut b) = (F2::ONE, self);
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

/// i32 word to GF(q), for |x| < q.
pub fn to_field(x: i32) -> u32 {
    if x >= 0 {
        x as u32
    } else {
        (x + Q as i32) as u32
    }
}

/// Canonical v in [0, q) to the balanced integer in [-(q-1)/2, (q-1)/2].
pub fn balanced(v: u32) -> i32 {
    if v > Q / 2 {
        v as i32 - Q as i32
    } else {
        v as i32
    }
}

pub fn carry_hi(v: i32, w: u32) -> i32 {
    (v + (1i32 << (w - 1))) >> w
}

// ---------------------------------------------------------------------------
// Roots of unity and plans.

/// A primitive 2^32-th root of unity in GF(q^2): the first (k + i), k >= 1,
/// whose (2^30 - 1)-th power has exact order 2^32.
pub fn root_2_32() -> F2 {
    let minus_one = F2 { re: Q - 1, im: 0 };
    (1..)
        .map(|k| F2 { re: k, im: 1 }.pow((1u64 << 30) - 1))
        .find(|h| h.pow(1u64 << 31) == minus_one)
        .expect("a primitive 2^32-th root exists")
}

/// Twiddle table w_n^j for j < n / 2.
pub fn twiddles(n: u32) -> Vec<F2> {
    let w = root_2_32().pow((1u64 << 32) / u64::from(n));
    let mut t = Vec::with_capacity((n / 2) as usize);
    let mut x = F2::ONE;
    for _ in 0..n / 2 {
        t.push(x);
        x = x.cmul(w);
    }
    t
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub p: u32,
    pub n: u32,
    pub log_n: u32,
    /// Bit position b_k of each word.
    pub bitpos: Vec<u32>,
    /// Width w_k of each word.
    pub width: Vec<u32>,
    /// Weight rotation of each word.
    pub wshift: Vec<u32>,
    /// Unweight and n^-1 rotation of each word.
    pub ushift: Vec<u32>,
    pub w_min: u32,
    pub w_max: u32,
    /// 2 S, the proven bound on every convolution coefficient.
    pub z_bound: u64,
    /// Carry passes per operation.
    pub passes: u32,
    /// The GF((2^61 - 1)^2) field of [`super::wide`] instead of
    /// GF((2^31 - 1)^2).
    pub wide: bool,
    /// Wide plans: balanced digits each unweighted coefficient is split
    /// into (see [`super::wide::digits`]); 0 otherwise.
    pub digits: u32,
}

/// The proven coefficient bound 2 S for (p, n), or None if the layout is not
/// allowed (word width below W_MIN).
pub fn capacity_bound(p: u32, n: u32) -> Option<u64> {
    let w_min = p / n;
    if !(W_MIN..16).contains(&w_min) {
        return None;
    }
    let n_max = u64::from(p % n); // words of width w_min + 1
    let wb = |w: u32| ((1u64 << (w - 1)) + 1).pow(2);
    let s = n_max * wb(w_min + 1) + (u64::from(n) - n_max) * wb(w_min);
    Some(2 * s)
}

/// Number of carry passes that brings every carry to magnitude <= 1, from the
/// recurrence in the module documentation.
pub fn carry_passes(z_bound: u64, w_min: u32, w_max: u32) -> u32 {
    let m = z_bound + 2;
    let mut h = [w_min, w_max]
        .iter()
        .map(|&w| (m + (1u64 << (w - 1))) >> w)
        .max()
        .unwrap();
    let mut passes = 1;
    while h > 1 {
        h = 1 + (h >> w_min);
        passes += 1;
    }
    passes
}

/// Smallest admissible transform length for p.
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

/// Largest p whose plan is accepted at length n (the bound is increasing in
/// p for fixed n), or None if no p is accepted at n.
pub fn edge(n: u32) -> Option<u32> {
    let ok = |p: u32| capacity_bound(p, n).is_some_and(|z| z <= Z_LIMIT);
    let (mut a, mut b) = (W_MIN * n, 16 * n);
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

/// Smallest and largest exponent with a plan. The range has two gaps,
/// 15,503,946 to 16,777,215 and 26,214,401 to 33,554,431, which the wide
/// field of [`super::wide`] covers; a unit test checks every exponent.
pub fn supported_range() -> (u32, u32) {
    let mut hi = 0;
    let mut n = N_MIN;
    while n <= N_MAX {
        hi = hi.max(edge(n).unwrap_or(0));
        n *= 2;
    }
    (W_MIN * N_MIN, hi)
}

impl Plan {
    pub fn new(p: u32) -> Option<Plan> {
        let n = choose_n(p)?;
        let z_bound = capacity_bound(p, n)?;
        let log_n = n.trailing_zeros();
        let (p64, n64) = (u64::from(p), u64::from(n));
        let pos = |k: u64| (p64 * k).div_ceil(n64);
        // n^-1 mod 31: 2 has order 5 mod 31.
        let kinv = (1u64 << ((5 - log_n % 5) % 5)) % 31;
        let mut plan = Plan {
            p,
            n,
            log_n,
            bitpos: Vec::with_capacity(n as usize),
            width: Vec::with_capacity(n as usize),
            wshift: Vec::with_capacity(n as usize),
            ushift: Vec::with_capacity(n as usize),
            w_min: p / n,
            w_max: p.div_ceil(n),
            z_bound,
            passes: 0,
            wide: false,
            digits: 0,
        };
        for k in 0..n64 {
            let b = pos(k);
            let e = n64 * b - p64 * k;
            let s = ((kinv * e) % 31) as u32;
            plan.bitpos.push(b as u32);
            plan.width.push((pos(k + 1) - b) as u32);
            plan.wshift.push(s);
            plan.ushift.push((62 - s - log_n % 31) % 31);
        }
        plan.passes = carry_passes(z_bound, plan.w_min, plan.w_max);
        Some(plan)
    }

    /// Per-word table for the shader: width, weight rotation, unweight
    /// rotation, and the width of the previous word (cyclically).
    pub fn word_table(&self) -> Vec<u32> {
        let n = self.n as usize;
        (0..n)
            .map(|k| {
                let prev = self.width[(k + n - 1) % n];
                self.width[k] | (self.wshift[k] << 8) | (self.ushift[k] << 16) | (prev << 24)
            })
            .collect()
    }

    /// Largest magnitude a stored word may have.
    pub fn word_limit(&self, k: usize) -> i32 {
        (1i32 << (self.width[k] - 1)) + 1
    }

    // -----------------------------------------------------------------------
    // Shader mirror: the stages of one operation.

    // ponytail: real input, a 2x ceiling; see the weight kernel.
    pub fn weight(&self, x: &[i32]) -> Vec<F2> {
        x.iter()
            .zip(&self.wshift)
            .map(|(&v, &s)| F2 {
                re: rot(to_field(v), s),
                im: 0,
            })
            .collect()
    }

    /// Decimation in frequency: natural order in, bit-reversed order out.
    pub fn forward(&self, t: &mut [F2], tw: &[F2]) {
        let n = self.n as usize;
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

    /// Decimation in time with conjugate twiddles: bit-reversed order in,
    /// natural order out, not scaled.
    pub fn inverse(&self, t: &mut [F2], tw: &[F2]) {
        let n = self.n as usize;
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

    /// Unweight, scale by n^-1, recover balanced integers, subtract `sub`
    /// from word 0. Returns the words and whether any imaginary part was
    /// nonzero (which exact arithmetic never produces).
    pub fn unweight(&self, t: &[F2], sub: i32) -> (Vec<i32>, bool) {
        let mut bad = false;
        let mut z: Vec<i32> = t
            .iter()
            .zip(&self.ushift)
            .map(|(v, &s)| {
                bad |= v.im != 0;
                balanced(rot(v.re, s))
            })
            .collect();
        z[0] -= sub;
        (z, bad)
    }

    pub fn carry_pass(&self, v: &[i32]) -> Vec<i32> {
        let n = v.len();
        (0..n)
            .map(|k| {
                let (w, wp) = (self.width[k], self.width[(k + n - 1) % n]);
                let hi = carry_hi(v[k], w);
                let lo = v[k] - hi * (1i32 << w);
                lo + carry_hi(v[(k + n - 1) % n], wp)
            })
            .collect()
    }

    fn finish(&self, t: &mut [F2], tw: &[F2], sub: i32) -> Vec<i32> {
        self.inverse(t, tw);
        let (mut z, bad) = self.unweight(t, sub);
        assert!(!bad, "nonzero imaginary part in the model");
        for _ in 0..self.passes {
            z = self.carry_pass(&z);
        }
        z
    }

    /// x^2 - sub, as the GPU computes it.
    pub fn square(&self, x: &[i32], tw: &[F2], sub: i32) -> Vec<i32> {
        let mut t = self.weight(x);
        self.forward(&mut t, tw);
        for v in t.iter_mut() {
            *v = v.sqr();
        }
        self.finish(&mut t, tw, sub)
    }

    /// x y, as the GPU computes it.
    pub fn mul(&self, x: &[i32], y: &[i32], tw: &[F2]) -> Vec<i32> {
        let mut t = self.weight(x);
        let mut u = self.weight(y);
        self.forward(&mut t, tw);
        self.forward(&mut u, tw);
        for (a, b) in t.iter_mut().zip(&u) {
            *a = a.cmul(*b);
        }
        self.finish(&mut t, tw, 0)
    }

    // -----------------------------------------------------------------------
    // Exact host conversions used by the backend's import and export.

    /// Canonical residue to balanced words, |x_k| <= 2^(w_k - 1) + 1.
    pub fn words_from_residue(&self, x: &Residue) -> Vec<i32> {
        let limbs = x.limbs();
        let bit = |pos: u32, w: u32| -> i64 {
            let (i, o) = ((pos / 32) as usize, pos % 32);
            let lo = u64::from(limbs[i]);
            let hi = limbs.get(i + 1).map_or(0, |&l| u64::from(l));
            (((hi << 32 | lo) >> o) & ((1u64 << w) - 1)) as i64
        };
        let mut carry = 0;
        let mut out: Vec<i32> = (0..self.n as usize)
            .map(|k| {
                let w = self.width[k];
                let mut v = bit(self.bitpos[k], w) + carry;
                carry = 0;
                if v >= 1 << (w - 1) {
                    v -= 1 << w;
                    carry = 1;
                }
                v as i32
            })
            .collect();
        out[0] += carry as i32;
        out
    }

    /// Balanced words to the canonical residue, with full carry normalization.
    /// A word outside its allowed range is a `Fault`: the GPU never produces
    /// one when it works correctly.
    pub fn residue_from_words(&self, words: &[i32]) -> Result<Residue, BackendError> {
        if words.len() != self.n as usize {
            return Err(BackendError::Fault(
                "word buffer has the wrong length".into(),
            ));
        }
        if let Some(k) =
            (0..words.len()).find(|&k| words[k].unsigned_abs() > self.word_limit(k) as u32)
        {
            return Err(BackendError::Fault(format!(
                "word {k} = {} is outside its range +-{}",
                words[k],
                self.word_limit(k)
            )));
        }
        let mut d: Vec<i64> = words.iter().map(|&v| i64::from(v)).collect();
        // Each round leaves digits in [0, 2^w) and a carry out of the top that
        // re-enters at word 0 (2^p = 1 mod N). Three rounds always suffice;
        // the fourth is a guard.
        let mut settled = false;
        for _ in 0..4 {
            let mut carry = 0i64;
            for (k, v) in d.iter_mut().enumerate() {
                let w = self.width[k];
                let t = *v + carry;
                *v = t.rem_euclid(1 << w);
                carry = (t - *v) >> w;
            }
            if carry == 0 {
                settled = true;
                break;
            }
            d[0] += carry;
        }
        if !settled {
            return Err(BackendError::Fault(
                "carry normalization did not settle".into(),
            ));
        }
        let mut limbs = vec![0u32; self.p.div_ceil(32) as usize];
        for (k, &v) in d.iter().enumerate() {
            let pos = self.bitpos[k];
            let (i, o) = ((pos / 32) as usize, pos % 32);
            let v = (v as u64) << o;
            limbs[i] |= v as u32;
            if let Some(l) = limbs.get_mut(i + 1) {
                *l |= (v >> 32) as u32;
            }
        }
        Residue::from_limbs_normalizing(self.p, limbs)
            .map_err(|e| BackendError::Fault(format!("export produced a bad residue: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn residue(&mut self, p: u32) -> Residue {
            let n = BigUint::from_bytes_le(
                &(0..p.div_ceil(8) + 8)
                    .map(|_| self.next() as u8)
                    .collect::<Vec<_>>(),
            );
            let m = (BigUint::from(1u8) << p) - 1u8;
            Residue::from_biguint(p, &(n % m))
        }
    }

    fn modulus(p: u32) -> BigUint {
        (BigUint::from(1u8) << p) - 1u8
    }

    #[test]
    fn mulmod_matches_u64_on_edges_and_random() {
        let edges = [
            0,
            1,
            2,
            3,
            0x7fff,
            0x8000,
            0xffff,
            0x1_0000,
            0x1_0001,
            0x3fff_ffff,
            0x4000_0000,
            0x4000_0001,
            Q - 2,
            Q - 1,
        ];
        let reference = |a: u32, b: u32| ((u64::from(a) * u64::from(b)) % u64::from(Q)) as u32;
        for &a in &edges {
            for &b in &edges {
                assert_eq!(mul(a, b), reference(a, b), "{a} * {b}");
            }
        }
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..1_000_000 {
            let (a, b) = (
                (r.next() % u64::from(Q)) as u32,
                (r.next() % u64::from(Q)) as u32,
            );
            assert_eq!(mul(a, b), reference(a, b));
        }
        for s in 0..31 {
            for &a in &edges {
                assert_eq!(u64::from(rot(a, s)), (u64::from(a) << s) % u64::from(Q));
            }
        }
    }

    #[test]
    fn root_has_exact_order_and_inverse_is_conjugate() {
        let h = root_2_32();
        assert_eq!(h.pow(1 << 32), F2::ONE);
        assert_ne!(h.pow(1 << 31), F2::ONE);
        for log_n in [2, 10, 22, 31] {
            let w = h.pow((1u64 << 32) >> log_n);
            assert_eq!(w.cmul(w.conj()), F2::ONE);
        }
    }

    #[test]
    fn transform_round_trip_and_cyclic_convolution() {
        let plan = Plan::new(40).unwrap(); // any plan; only n matters here
        let n = plan.n as usize;
        let tw = twiddles(plan.n);
        let mut r = Rng(7);
        let x: Vec<F2> = (0..n)
            .map(|_| F2 {
                re: (r.next() % 1000) as u32,
                im: 0,
            })
            .collect();
        let y: Vec<F2> = (0..n)
            .map(|_| F2 {
                re: (r.next() % 1000) as u32,
                im: 0,
            })
            .collect();
        let (mut a, mut b) = (x.clone(), y.clone());
        plan.forward(&mut a, &tw);
        plan.forward(&mut b, &tw);
        for (u, v) in a.iter_mut().zip(&b) {
            *u = u.cmul(*v);
        }
        plan.inverse(&mut a, &tw);
        let ninv = rot(1, (31 - plan.log_n) % 31);
        for k in 0..n {
            let want: u64 = (0..n)
                .map(|i| u64::from(x[i].re) * u64::from(y[(k + n - i) % n].re))
                .sum();
            assert_eq!(mul(a[k].re, ninv), want as u32);
            assert_eq!(a[k].im, 0);
        }
    }

    fn check_square(p: u32, x: &Residue) {
        let plan = Plan::new(p).unwrap();
        let tw = twiddles(plan.n);
        let words = plan.words_from_residue(x);
        assert_eq!(
            plan.residue_from_words(&words).unwrap(),
            *x,
            "round trip p={p}"
        );
        let got = plan
            .residue_from_words(&plan.square(&words, &tw, 0))
            .unwrap();
        let want = (x.to_biguint() * x.to_biguint()) % modulus(p);
        assert_eq!(got.to_biguint(), want, "square p={p} n={}", plan.n);
    }

    #[test]
    fn model_squaring_matches_bigint() {
        let mut r = Rng(12345);
        for p in [
            11, 31, 61, 89, 127, 521, 607, 1279, 2203, 4253, 9941, 23209, 44497,
        ] {
            let m = modulus(p);
            for v in [
                BigUint::from(0u8),
                BigUint::from(1u8),
                BigUint::from(2u8),
                &m - 1u8,
                &m - 2u8,
            ] {
                check_square(p, &Residue::from_biguint(p, &v));
            }
            for _ in 0..3 {
                check_square(p, &r.residue(p));
            }
        }
    }

    #[test]
    fn model_ll_and_mul_match_bigint() {
        let mut r = Rng(99);
        for p in [127, 521, 1279, 4423] {
            let plan = Plan::new(p).unwrap();
            let tw = twiddles(plan.n);
            let m = modulus(p);
            let (x, y) = (r.residue(p), r.residue(p));
            let got = plan
                .residue_from_words(&plan.mul(
                    &plan.words_from_residue(&x),
                    &plan.words_from_residue(&y),
                    &tw,
                ))
                .unwrap();
            assert_eq!(got.to_biguint(), x.to_biguint() * y.to_biguint() % &m);
            let mut w = plan.words_from_residue(&Residue::from_u64(p, 4));
            let mut s = BigUint::from(4u8);
            for _ in 0..50 {
                w = plan.square(&w, &tw, 2);
                s = (&s * &s + &m - 2u8) % &m;
            }
            assert_eq!(plan.residue_from_words(&w).unwrap().to_biguint(), s);
        }
        // Complete LL runs: M521 and M607 are prime, M1277 is composite.
        for (p, prime) in [(521, true), (607, true), (1277, false)] {
            let plan = Plan::new(p).unwrap();
            let tw = twiddles(plan.n);
            let mut w = plan.words_from_residue(&Residue::from_u64(p, 4));
            for _ in 0..p - 2 {
                w = plan.square(&w, &tw, 2);
            }
            assert_eq!(
                plan.residue_from_words(&w).unwrap().is_zero(),
                prime,
                "p={p}"
            );
        }
    }

    /// The edge of the bound for a length: the largest p using that n, with
    /// every word at its extreme allowed magnitude and the same sign. The
    /// exact integer convolution must be recovered, and stay within Z_LIMIT.
    #[test]
    fn capacity_bound_holds_at_the_edge() {
        for log_n in [2u32, 4, 6, 10, 14] {
            let n = 1u32 << log_n;
            let p = edge(n).unwrap();
            let plan = Plan::new(p).unwrap();
            assert_eq!(plan.n, n, "p={p}");
            for sign in [1, -1] {
                let x: Vec<i32> = (0..n as usize).map(|k| sign * plan.word_limit(k)).collect();
                let exact = exact_square(&plan, &x);
                assert_eq!(model_square_coeffs(&plan, &x), exact, "p={p} n={n}");
                let m = exact.iter().map(|e| e.unsigned_abs()).max().unwrap();
                assert!(
                    m <= plan.z_bound && plan.z_bound <= Z_LIMIT,
                    "p={p} max={m}"
                );
            }
        }
    }

    /// Exact weighted cyclic self-convolution in i64.
    fn exact_square(plan: &Plan, x: &[i32]) -> Vec<i64> {
        let n = plan.n as usize;
        (0..n)
            .map(|k| {
                (0..n)
                    .map(|i| {
                        let j = (k + n - i) % n;
                        let wrap = if i + j >= n { i64::from(plan.p) } else { 0 };
                        let c = i64::from(plan.bitpos[i]) + i64::from(plan.bitpos[j])
                            - i64::from(plan.bitpos[k])
                            - wrap;
                        assert!(c == 0 || c == 1);
                        (i64::from(x[i]) * i64::from(x[j])) << c
                    })
                    .sum()
            })
            .collect()
    }

    /// The model's recovered coefficients before carrying.
    fn model_square_coeffs(plan: &Plan, x: &[i32]) -> Vec<i64> {
        let tw = twiddles(plan.n);
        let mut t = plan.weight(x);
        plan.forward(&mut t, &tw);
        for v in t.iter_mut() {
            *v = v.sqr();
        }
        plan.inverse(&mut t, &tw);
        plan.unweight(&t, 0).0.into_iter().map(i64::from).collect()
    }

    /// Non-vacuity: words just large enough that the exact coefficients pass
    /// (q - 1) / 2 are recovered wrongly, so the edge test above can fail.
    #[test]
    fn exceeding_the_field_capacity_gives_wrong_coefficients() {
        let plan = Plan::new(1279).unwrap();
        let a = ((1u64 << 31) / u64::from(plan.n)).isqrt() as i32 + 1;
        let x = vec![a; plan.n as usize];
        let exact = exact_square(&plan, &x);
        assert!(exact.iter().any(|e| e.unsigned_abs() > u64::from(Q / 2)));
        assert_ne!(model_square_coeffs(&plan, &x), exact);
    }

    #[test]
    fn carry_passes_reach_the_invariant_from_worst_inputs() {
        for p in [11, 521, 9941, 44497, 1_000_003, 15_000_017] {
            let plan = Plan::new(p).unwrap();
            let z = plan.z_bound as i32;
            for pattern in 0..4 {
                let v: Vec<i32> = (0..plan.n as usize)
                    .map(|k| match pattern {
                        0 => z,
                        1 => -z - 2,
                        2 => {
                            if k % 2 == 0 {
                                z
                            } else {
                                -z - 2
                            }
                        }
                        _ => {
                            if k % 3 == 0 {
                                -z - 2
                            } else {
                                z
                            }
                        }
                    })
                    .collect();
                let mut x = v;
                for _ in 0..plan.passes {
                    x = plan.carry_pass(&x);
                }
                for (k, &w) in x.iter().enumerate() {
                    assert!(
                        w.abs() <= plan.word_limit(k),
                        "p={p} pattern={pattern} k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn export_rejects_out_of_range_words_including_the_extremes() {
        let plan = Plan::new(1279).unwrap();
        let lim = plan.word_limit(1);
        for (v, ok) in [
            (lim, true),
            (-lim, true),
            (lim + 1, false),
            (-lim - 1, false),
            (i32::MIN, false),
            (i32::MAX, false),
        ] {
            let mut w = vec![0i32; plan.n as usize];
            w[1] = v;
            let r = plan.residue_from_words(&w);
            assert_eq!(r.is_ok(), ok, "{v}");
            if !ok {
                assert!(matches!(r, Err(BackendError::Fault(_))), "{v}");
            }
        }
    }

    #[test]
    fn supported_range_is_contiguous() {
        let (lo, hi) = supported_range();
        assert!(lo >= 8 && hi > 40_000_000, "{lo} {hi}");
        assert_eq!(choose_n(hi), Some(N_MAX));
        assert!(choose_n(lo - 1).is_none());
        assert!(choose_n(hi + 1).is_none());
        // The 31-bit field has gaps below its top where the narrowest words
        // of one length hold more than the length allows and the next
        // length needs words of 2 bits (15,503,946 to 16,777,215 and
        // 26,214,401 to 33,554,431); the wide field covers them. Every p in
        // the range has a wide length where the narrow one is missing, and
        // the full wide plan (which builds O(n) tables) is built at each
        // gap's two ends.
        let mut ends = Vec::new();
        let mut inside = false;
        for p in lo..=hi {
            let gap = choose_n(p).is_none();
            if gap {
                assert!(super::super::wide::choose_n(p).is_some(), "p={p}");
            }
            if gap != inside {
                ends.push(if gap { p } else { p - 1 });
                inside = gap;
            }
        }
        assert!(!inside && ends.len() >= 4, "{ends:?}");
        for &p in &ends {
            assert!(Plan::new_wide(p).is_some(), "p={p}");
        }
        assert!(choose_n(30_000_000).is_none() && Plan::new_wide(30_000_000).is_some());
    }
}
