// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Exact CPU model of the CUDA arithmetic, and everything the host needs that
//! does not touch a device: the plan (transform length and carry parameters
//! per exponent), the twiddle table, the launch sequence, and the exact
//! conversions between canonical residues and balanced words.
//!
//! Every function named `k_*` mirrors one kernel of `kernels.cu` statement
//! for statement, on host vectors of 64-bit elements. The launch sequence
//! produced by [`Plan::square_launches`] and [`Plan::mul_launches`] is the one
//! the CUDA host submits, and [`execute`] runs it on the model kernels.
//!
//! Arithmetic: q = 2^61 - 1, field GF(q^2) = GF(q)[i] / (i^2 + 1) (valid since
//! q = 3 mod 4). The norm-1 subgroup has order q + 1 = 2^61, so every
//! power-of-two root of unity used here lies in it and its inverse is its
//! conjugate. 2 has order 61 modulo q, so the n-th root of 2 is 2^k with
//! k = n^-1 mod 61 and every weight is a power of two.
//!
//! Capacity bound (proof in docs/CUDA.md): with input words bounded by W, the
//! exact convolution coefficient is z_k = sum x_i y_j 2^d with d in {0, 1}, so
//! |z_k| <= 2 n W^2. A plan is accepted when 2 n W^2 <= 2^59, one bit below
//! the hard limit (q - 1) / 2 < 2^60 for reading z_k back as a balanced integer.

use crate::residue::{limb_count, Residue};

/// The field prime q = 2^61 - 1.
pub const Q: u64 = (1 << 61) - 1;
/// (q - 1) / 2: the largest magnitude a balanced coefficient can take.
pub const HALF_Q: u64 = Q >> 1;
/// Smallest and largest transform lengths, as log2.
pub const MIN_LOG_N: u32 = 2;
pub const MAX_LOG_N: u32 = 24;
/// Every accepted plan satisfies 2 n W^2 <= CAPACITY_LIMIT.
pub const CAPACITY_LIMIT: u128 = 1 << 59;
/// Smallest word width. The LL subtraction at word 0 needs w >= 3 for the
/// carry slack to stay floor(pieces / 2).
pub const MIN_WIDTH: u32 = 3;

// ---------------------------------------------------------------- GF(q) ----

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

/// Mirrors `mf_mulmod`. Input a, b < 2^61; output in [0, q).
pub fn mul(a: u64, b: u64) -> u64 {
    let lo = a.wrapping_mul(b);
    let hi = ((u128::from(a) * u128::from(b)) >> 64) as u64; // __umul64hi
    let r = (lo & Q) + ((hi << 3) | (lo >> 61));
    let r = (r & Q) + (r >> 61);
    if r >= Q {
        r - Q
    } else {
        r
    }
}

/// x * 2^s mod q, s in [0, 61). Mirrors `mf_rotl`.
pub fn rotl(x: u64, s: u64) -> u64 {
    ((x << s) & Q) | (x >> (61 - s))
}

/// An element a + b i of GF(q^2).
pub type Fq2 = (u64, u64);

pub fn mul2(x: Fq2, y: Fq2) -> Fq2 {
    (
        sub(mul(x.0, y.0), mul(x.1, y.1)),
        add(mul(x.0, y.1), mul(x.1, y.0)),
    )
}

pub fn pow2(mut x: Fq2, mut e: u128) -> Fq2 {
    let mut r = (1, 0);
    while e > 0 {
        if e & 1 == 1 {
            r = mul2(r, x);
        }
        x = mul2(x, x);
        e >>= 1;
    }
    r
}

/// A generator of the subgroup of order 2^61 of GF(q^2)*: h = g^(q - 1) for
/// the first g = a + i whose image has order exactly 2^61 (h^(2^60) != 1).
pub fn root_2_61() -> Fq2 {
    for a in 1u64.. {
        let h = pow2((a, 1), u128::from(Q - 1));
        if pow2(h, 1 << 60) != (1, 0) {
            return h;
        }
    }
    unreachable!("the subgroup of order 2^61 is cyclic and has generators")
}

// ----------------------------------------------------------------- plan ----

/// Transform length and carry parameters for one exponent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub p: u32,
    pub log_n: u32,
    pub n: u64,
    /// n^-1 mod 61: the n-th root of 2 modulo q is 2^kinv.
    pub kinv: u64,
    /// floor(p / n) and ceil(p / n).
    pub w_min: u32,
    pub w_max: u32,
    /// Parts each coefficient is split into by carry pass A.
    pub pieces: u64,
    /// floor(pieces / 2): how far a stored word may exceed its nominal
    /// balanced range 2^(w - 1).
    pub slack: u64,
    /// W = 2^(w_max - 1) + slack, the bound on every stored word.
    pub word_bound: u64,
    /// 2 n W^2, the bound on every convolution coefficient.
    pub coeff_bound: u128,
}

/// The plan for exponent p at transform length 2^log_n, if the capacity bound
/// holds with its margin and every word has width at least `MIN_WIDTH`.
pub fn plan_for_len(p: u32, log_n: u32) -> Option<Plan> {
    if !(MIN_LOG_N..=MAX_LOG_N).contains(&log_n) {
        return None;
    }
    let n = 1u64 << log_n;
    let w_min = (u64::from(p) / n) as u32;
    let w_max = u64::from(p).div_ceil(n) as u32;
    if w_min < MIN_WIDTH || w_max > 30 {
        return None;
    }
    // The smallest m whose (m - 1) balanced splits shrink the largest possible
    // coefficient to a remainder of magnitude <= 1: Z <= 2^((m - 1) w_min),
    // where Z = 2 n W(m)^2 and W(m) = 2^(w_max - 1) + floor(m / 2).
    let bound = |m: u64| {
        let w = (1u64 << (w_max - 1)) + m / 2;
        (w, 2 * u128::from(n) * u128::from(w) * u128::from(w))
    };
    let mut m = 2u64;
    loop {
        let (_, z) = bound(m);
        let shift = (m - 1) * u64::from(w_min);
        if shift >= 127 || z <= 1u128 << shift {
            break;
        }
        m += 1;
    }
    let (word_bound, coeff_bound) = bound(m);
    if coeff_bound > CAPACITY_LIMIT {
        return None;
    }
    let mut kinv = 1u64;
    while (kinv * n) % 61 != 1 {
        kinv += 1;
    }
    Some(Plan {
        p,
        log_n,
        n,
        kinv,
        w_min,
        w_max,
        pieces: m,
        slack: m / 2,
        word_bound,
        coeff_bound,
    })
}

/// The plan with the smallest transform length that holds p.
pub fn plan(p: u32) -> Option<Plan> {
    (MIN_LOG_N..=MAX_LOG_N).find_map(|l| plan_for_len(p, l))
}

/// The largest exponent with a plan at length 2^log_n. Validity depends on p
/// only through (floor(p / n), ceil(p / n)), so one representative per class
/// decides it: p = w n has (w, w), and every p in (w n, w n + n) has (w, w + 1).
pub fn max_exponent_for_len(log_n: u32) -> Option<u32> {
    let n = 1u32 << log_n;
    let mut best = None;
    for w in MIN_WIDTH..=30 {
        if plan_for_len(w * n, log_n).is_some() {
            best = Some(w * n);
        }
        if plan_for_len(w * n + 1, log_n).is_some() {
            best = Some(w * n + n - 1);
        }
    }
    best
}

/// Inclusive exponent range with a plan: from the smallest p whose words at
/// n = 2^MIN_LOG_N have width MIN_WIDTH to the largest p at n = 2^MAX_LOG_N.
/// The range has no gaps (checked by a unit test).
pub fn supported_exponents() -> (u32, u32) {
    let lo = MIN_WIDTH << MIN_LOG_N;
    let hi = max_exponent_for_len(MAX_LOG_N).expect("the largest length has a plan");
    (lo, hi)
}

// ------------------------------------------------- word layout helpers ----

/// Mirrors `mf_offset`: bit offset ceil(p i / n) of word i.
pub fn offset(p: u64, log_n: u64, i: u64) -> u64 {
    (p * i + (1 << log_n) - 1) >> log_n
}

/// Mirrors `mf_width`.
pub fn width(p: u64, log_n: u64, i: u64) -> u64 {
    offset(p, log_n, i + 1) - offset(p, log_n, i)
}

/// Mirrors `mf_weight_shift`.
pub fn weight_shift(n: u64, p: u64, kinv: u64, i: u64) -> u64 {
    let e = (n - ((p * i) & (n - 1))) & (n - 1);
    (kinv * e) % 61
}

/// Mirrors `mf_low`.
pub fn low(v: i64, w: u64) -> i64 {
    let half = 1i64 << (w - 1);
    let mask = (1i64 << w) - 1;
    ((v + half) & mask) - half
}

/// Mirrors `mf_carry`.
pub fn carry(v: i64, w: u64) -> i64 {
    (v - low(v, w)) >> w
}

// -------------------------------------------------------------- kernels ----
// Each loops over the thread index exactly as the CUDA kernel's grid does.

pub fn k_weight(t: &mut [u64], x: &[u64], n: u64, p: u64, kinv: u64) {
    for i in 0..n {
        let v = x[i as usize] as i64;
        let m = if v < 0 {
            (v + Q as i64) as u64
        } else {
            v as u64
        };
        t[2 * i as usize] = rotl(m, weight_shift(n, p, kinv, i));
        t[2 * i as usize + 1] = 0;
    }
}

pub fn k_fwd_stage(t: &mut [u64], tw: &[u64], n: u64, h: u64, stride: u64) {
    for k in 0..n / 2 {
        let j = k & (h - 1);
        let i0 = (2 * (k - j) + j) as usize;
        let i1 = i0 + h as usize;
        let ti = (2 * stride * j) as usize;
        let (ur, ui, vr, vi) = (t[2 * i0], t[2 * i0 + 1], t[2 * i1], t[2 * i1 + 1]);
        let (wr, wi) = (tw[ti], tw[ti + 1]);
        t[2 * i0] = add(ur, vr);
        t[2 * i0 + 1] = add(ui, vi);
        let (dr, di) = (sub(ur, vr), sub(ui, vi));
        t[2 * i1] = sub(mul(dr, wr), mul(di, wi));
        t[2 * i1 + 1] = add(mul(dr, wi), mul(di, wr));
    }
}

pub fn k_inv_stage(t: &mut [u64], tw: &[u64], n: u64, h: u64, stride: u64) {
    for k in 0..n / 2 {
        let j = k & (h - 1);
        let i0 = (2 * (k - j) + j) as usize;
        let i1 = i0 + h as usize;
        let ti = (2 * stride * j) as usize;
        let (wr, wi) = (tw[ti], tw[ti + 1]);
        let (xr, xi) = (t[2 * i1], t[2 * i1 + 1]);
        let vr = add(mul(xr, wr), mul(xi, wi));
        let vi = sub(mul(xi, wr), mul(xr, wi));
        let (ur, ui) = (t[2 * i0], t[2 * i0 + 1]);
        t[2 * i0] = add(ur, vr);
        t[2 * i0 + 1] = add(ui, vi);
        t[2 * i1] = sub(ur, vr);
        t[2 * i1 + 1] = sub(ui, vi);
    }
}

pub fn k_sqr(t: &mut [u64], n: u64) {
    for i in 0..n as usize {
        let (a, b) = (t[2 * i], t[2 * i + 1]);
        t[2 * i] = sub(mul(a, a), mul(b, b));
        t[2 * i + 1] = mul(add(a, a), b);
    }
}

pub fn k_mul(t: &mut [u64], t2: &[u64], n: u64) {
    for i in 0..n as usize {
        let (a, b, c, d) = (t[2 * i], t[2 * i + 1], t2[2 * i], t2[2 * i + 1]);
        t[2 * i] = sub(mul(a, c), mul(b, d));
        t[2 * i + 1] = add(mul(a, d), mul(b, c));
    }
}

pub fn k_unweight(z: &mut [u64], t: &[u64], n: u64, p: u64, log_n: u64, kinv: u64) {
    for i in 0..n {
        let s = weight_shift(n, p, kinv, i);
        let c = rotl(t[2 * i as usize], (122 - s - log_n) % 61);
        let v = if c > HALF_Q {
            c as i64 - Q as i64
        } else {
            c as i64
        };
        z[i as usize] = v as u64;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn k_carry_a(a: &mut [u64], z: &[u64], n: u64, p: u64, log_n: u64, pieces: u64, add0: u64) {
    for k in 0..n {
        let mut acc = if k == 0 { add0 as i64 } else { 0 };
        for j in 0..pieces {
            let s = k.wrapping_sub(j) & (n - 1);
            let mut v = z[s as usize] as i64;
            for t in 0..j {
                v = carry(v, width(p, log_n, (s + t) & (n - 1)));
            }
            acc += if j + 1 < pieces {
                low(v, width(p, log_n, k))
            } else {
                v
            };
        }
        a[k as usize] = acc as u64;
    }
}

pub fn k_carry_b(x: &mut [u64], a: &[u64], n: u64, p: u64, log_n: u64) {
    for k in 0..n {
        let prev = k.wrapping_sub(1) & (n - 1);
        let d = low(a[k as usize] as i64, width(p, log_n, k));
        let c = carry(a[prev as usize] as i64, width(p, log_n, prev));
        x[k as usize] = (d + c) as u64;
    }
}

// ------------------------------------------------------ launch sequence ----

/// The kernels of `kernels.cu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Weight,
    FwdStage,
    InvStage,
    Sqr,
    Mul,
    Unweight,
    CarryA,
    CarryB,
}

/// Kind of a kernel parameter: a pointer to 64-bit elements or a 64-bit scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    Ptr,
    Scalar,
}

impl Kernel {
    pub const ALL: [Kernel; 8] = [
        Kernel::Weight,
        Kernel::FwdStage,
        Kernel::InvStage,
        Kernel::Sqr,
        Kernel::Mul,
        Kernel::Unweight,
        Kernel::CarryA,
        Kernel::CarryB,
    ];

    /// The `extern "C"` name in `kernels.cu`.
    pub fn name(self) -> &'static str {
        match self {
            Kernel::Weight => "mf_weight",
            Kernel::FwdStage => "mf_fwd_stage",
            Kernel::InvStage => "mf_inv_stage",
            Kernel::Sqr => "mf_sqr",
            Kernel::Mul => "mf_mul",
            Kernel::Unweight => "mf_unweight",
            Kernel::CarryA => "mf_carry_a",
            Kernel::CarryB => "mf_carry_b",
        }
    }

    /// Parameter list in declaration order. The first parameter is always
    /// the pointer the kernel writes; other pointers are read only.
    pub fn params(self) -> &'static [Param] {
        use Param::{Ptr, Scalar};
        match self {
            Kernel::Weight => &[Ptr, Ptr, Scalar, Scalar, Scalar],
            Kernel::FwdStage | Kernel::InvStage => &[Ptr, Ptr, Scalar, Scalar, Scalar],
            Kernel::Sqr => &[Ptr, Scalar],
            Kernel::Mul => &[Ptr, Ptr, Scalar],
            Kernel::Unweight => &[Ptr, Ptr, Scalar, Scalar, Scalar, Scalar],
            Kernel::CarryA => &[Ptr, Ptr, Scalar, Scalar, Scalar, Scalar, Scalar],
            Kernel::CarryB => &[Ptr, Ptr, Scalar, Scalar, Scalar],
        }
    }
}

/// Device buffers named by a launch. `X` is the residue being updated and
/// `Src` the second operand of a multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buf {
    X,
    Src,
    T,
    T2,
    Tw,
    Z,
    A,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arg {
    Buf(Buf),
    U64(u64),
}

/// One kernel launch: `threads` is the number of threads that do work; the
/// host rounds the grid up and the kernel guards its index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub kernel: Kernel,
    pub threads: u64,
    pub args: Vec<Arg>,
}

impl Plan {
    /// Element count (64-bit elements) of each device buffer.
    pub fn buf_len(&self, b: Buf) -> usize {
        let n = self.n as usize;
        match b {
            Buf::X | Buf::Src | Buf::Tw | Buf::Z | Buf::A => n,
            Buf::T | Buf::T2 => 2 * n,
        }
    }

    /// tw[2 j], tw[2 j + 1] = w_n^j for j < n / 2, w_n of order exactly n.
    pub fn twiddles(&self) -> Vec<u64> {
        let w = pow2(root_2_61(), 1u128 << (61 - self.log_n));
        let mut tw = Vec::with_capacity(self.n as usize);
        let mut cur = (1, 0);
        for _ in 0..self.n / 2 {
            tw.push(cur.0);
            tw.push(cur.1);
            cur = mul2(cur, w);
        }
        tw
    }

    fn forward(&self, buf: Buf, out: &mut Vec<Launch>) {
        let (n, p) = (self.n, u64::from(self.p));
        let src = if buf == Buf::T { Buf::X } else { Buf::Src };
        out.push(Launch {
            kernel: Kernel::Weight,
            threads: n,
            args: vec![
                Arg::Buf(buf),
                Arg::Buf(src),
                Arg::U64(n),
                Arg::U64(p),
                Arg::U64(self.kinv),
            ],
        });
        let mut h = n / 2;
        while h >= 1 {
            out.push(Launch {
                kernel: Kernel::FwdStage,
                threads: n / 2,
                args: vec![
                    Arg::Buf(buf),
                    Arg::Buf(Buf::Tw),
                    Arg::U64(n),
                    Arg::U64(h),
                    Arg::U64(n / (2 * h)),
                ],
            });
            h /= 2;
        }
    }

    fn back(&self, add0: i64, out: &mut Vec<Launch>) {
        let (n, p, l) = (self.n, u64::from(self.p), u64::from(self.log_n));
        let mut h = 1;
        while h < n {
            out.push(Launch {
                kernel: Kernel::InvStage,
                threads: n / 2,
                args: vec![
                    Arg::Buf(Buf::T),
                    Arg::Buf(Buf::Tw),
                    Arg::U64(n),
                    Arg::U64(h),
                    Arg::U64(n / (2 * h)),
                ],
            });
            h *= 2;
        }
        out.push(Launch {
            kernel: Kernel::Unweight,
            threads: n,
            args: vec![
                Arg::Buf(Buf::Z),
                Arg::Buf(Buf::T),
                Arg::U64(n),
                Arg::U64(p),
                Arg::U64(l),
                Arg::U64(self.kinv),
            ],
        });
        out.push(Launch {
            kernel: Kernel::CarryA,
            threads: n,
            args: vec![
                Arg::Buf(Buf::A),
                Arg::Buf(Buf::Z),
                Arg::U64(n),
                Arg::U64(p),
                Arg::U64(l),
                Arg::U64(self.pieces),
                Arg::U64(add0 as u64),
            ],
        });
        out.push(Launch {
            kernel: Kernel::CarryB,
            threads: n,
            args: vec![
                Arg::Buf(Buf::X),
                Arg::Buf(Buf::A),
                Arg::U64(n),
                Arg::U64(p),
                Arg::U64(l),
            ],
        });
    }

    /// X <- X^2 + add0 mod N (add0 = 0 for a squaring, -2 for an LL step).
    pub fn square_launches(&self, add0: i64) -> Vec<Launch> {
        let mut v = Vec::new();
        self.forward(Buf::T, &mut v);
        v.push(Launch {
            kernel: Kernel::Sqr,
            threads: self.n,
            args: vec![Arg::Buf(Buf::T), Arg::U64(self.n)],
        });
        self.back(add0, &mut v);
        v
    }

    /// X <- X * Src mod N.
    pub fn mul_launches(&self) -> Vec<Launch> {
        let mut v = Vec::new();
        self.forward(Buf::T, &mut v);
        self.forward(Buf::T2, &mut v);
        v.push(Launch {
            kernel: Kernel::Mul,
            threads: self.n,
            args: vec![Arg::Buf(Buf::T), Arg::Buf(Buf::T2), Arg::U64(self.n)],
        });
        self.back(0, &mut v);
        v
    }

    // ------------------------------------------- exact host conversions ----

    /// Canonical residue to balanced words (as two's complement u64 bits).
    /// Every word has |x_k| <= 2^(w_k - 1), except word 0, which may reach
    /// 2^(w_0 - 1) + 1 from the wrapped final carry; both are within W.
    pub fn to_words(&self, x: &Residue) -> Vec<u64> {
        let (p, l) = (u64::from(self.p), u64::from(self.log_n));
        let limbs = x.limbs();
        let bit = |b: u64| (limbs[(b / 32) as usize] >> (b % 32)) & 1;
        let mut words = vec![0u64; self.n as usize];
        let mut c = 0i64;
        for k in 0..self.n {
            let (b, w) = (offset(p, l, k), width(p, l, k));
            let u = (0..w).fold(0i64, |acc, t| acc | (i64::from(bit(b + t)) << t));
            let v = u + c;
            let d = low(v, w);
            c = (v - d) >> w;
            words[k as usize] = d as u64;
        }
        words[0] = (words[0] as i64 + c) as u64;
        words
    }

    /// Balanced words to the canonical residue, exactly. Rejects words
    /// outside the proven bound |x_k| <= 2^(w_k - 1) + slack, which no correct
    /// computation produces.
    pub fn from_words(&self, words: &[u64]) -> Result<Residue, String> {
        let (p, l, n) = (u64::from(self.p), u64::from(self.log_n), self.n);
        if words.len() != n as usize {
            return Err(format!("{} words, expected {n}", words.len()));
        }
        let mut d: Vec<i64> = words.iter().map(|&w| w as i64).collect();
        for (k, &v) in d.iter().enumerate() {
            let lim = (1i64 << (width(p, l, k as u64) - 1)) + self.slack as i64;
            // unsigned_abs: i64::MIN has no positive counterpart.
            if v.unsigned_abs() > lim as u64 {
                return Err(format!("word {k} = {v} exceeds the bound {lim}"));
            }
        }
        // Sequential nonnegative carry with the wrap 2^p = 1 mod N. The
        // first round leaves a small carry c; adding it at word 0 either
        // stays in range or wraps once more with carry +1 or -1, which the
        // third round absorbs without a further wrap.
        let mut rounds = 0;
        loop {
            let mut c = 0i64;
            for (k, v) in d.iter_mut().enumerate() {
                let w = width(p, l, k as u64);
                let s = *v + c;
                *v = s & ((1i64 << w) - 1);
                c = s >> w;
            }
            if c == 0 {
                break;
            }
            d[0] += c;
            rounds += 1;
            if rounds > 4 {
                return Err("carry did not settle".into());
            }
        }
        let mut limbs = Vec::with_capacity(limb_count(self.p));
        let (mut acc, mut nbits) = (0u64, 0u64);
        for (k, &v) in d.iter().enumerate() {
            acc |= (v as u64) << nbits;
            nbits += width(p, l, k as u64);
            while nbits >= 32 {
                limbs.push(acc as u32);
                acc >>= 32;
                nbits -= 32;
            }
        }
        if nbits > 0 {
            limbs.push(acc as u32);
        }
        Residue::from_limbs_normalizing(self.p, limbs).map_err(|e| e.to_string())
    }
}

// ------------------------------------------------------------- executor ----

/// Host copies of every device buffer, for the model and for emulation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Buffers {
    pub x: Vec<u64>,
    pub src: Vec<u64>,
    pub t: Vec<u64>,
    pub t2: Vec<u64>,
    pub tw: Vec<u64>,
    pub z: Vec<u64>,
    pub a: Vec<u64>,
}

impl Buffers {
    pub fn new(plan: &Plan) -> Buffers {
        let z = |b| vec![0u64; plan.buf_len(b)];
        Buffers {
            x: z(Buf::X),
            src: z(Buf::Src),
            t: z(Buf::T),
            t2: z(Buf::T2),
            tw: plan.twiddles(),
            z: z(Buf::Z),
            a: z(Buf::A),
        }
    }

    pub fn get_mut(&mut self, b: Buf) -> &mut Vec<u64> {
        match b {
            Buf::X => &mut self.x,
            Buf::Src => &mut self.src,
            Buf::T => &mut self.t,
            Buf::T2 => &mut self.t2,
            Buf::Tw => &mut self.tw,
            Buf::Z => &mut self.z,
            Buf::A => &mut self.a,
        }
    }
}

/// Run one launch on the model kernels.
pub fn execute(l: &Launch, bufs: &mut Buffers) {
    let ptrs: Vec<Buf> = l
        .args
        .iter()
        .filter_map(|a| if let Arg::Buf(b) = a { Some(*b) } else { None })
        .collect();
    let s: Vec<u64> = l
        .args
        .iter()
        .filter_map(|a| if let Arg::U64(v) = a { Some(*v) } else { None })
        .collect();
    let mut out = std::mem::take(bufs.get_mut(ptrs[0]));
    let inp = ptrs
        .get(1)
        .map(|&b| std::mem::take(bufs.get_mut(b)))
        .unwrap_or_default();
    match l.kernel {
        Kernel::Weight => k_weight(&mut out, &inp, s[0], s[1], s[2]),
        Kernel::FwdStage => k_fwd_stage(&mut out, &inp, s[0], s[1], s[2]),
        Kernel::InvStage => k_inv_stage(&mut out, &inp, s[0], s[1], s[2]),
        Kernel::Sqr => k_sqr(&mut out, s[0]),
        Kernel::Mul => k_mul(&mut out, &inp, s[0]),
        Kernel::Unweight => k_unweight(&mut out, &inp, s[0], s[1], s[2], s[3]),
        Kernel::CarryA => k_carry_a(&mut out, &inp, s[0], s[1], s[2], s[3], s[4]),
        Kernel::CarryB => k_carry_b(&mut out, &inp, s[0], s[1], s[2]),
    }
    if let Some(&b) = ptrs.get(1) {
        *bufs.get_mut(b) = inp;
    }
    *bufs.get_mut(ptrs[0]) = out;
}

pub fn execute_all(ls: &[Launch], bufs: &mut Buffers) {
    for l in ls {
        execute(l, bufs);
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::residue::modulus;
    use num_bigint::{BigInt, BigUint};
    use num_traits::{One, Zero};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn random(p: u32, rng: &mut Rng) -> BigUint {
        let limbs: Vec<u32> = (0..limb_count(p)).map(|_| rng.next() as u32).collect();
        BigUint::from_slice(&limbs) % modulus(p)
    }

    fn edges(p: u32) -> Vec<BigUint> {
        let n = modulus(p);
        let mut v = vec![
            BigUint::zero(),
            BigUint::one(),
            BigUint::from(2u32),
            &n - 1u32,
            &n - 2u32,
        ];
        v.push(n.clone() >> 1u32); // 0111...1
        v.push(BigUint::one() << (p - 1)); // 1000...0
        v.push(BigUint::from(0x5555_5555u32) % &n);
        let mut alt = BigUint::zero();
        for b in (0..p).step_by(2) {
            alt.set_bit(u64::from(b), true);
        }
        v.push(alt % &n);
        v
    }

    /// A model engine: the launch sequence on the model kernels.
    struct M {
        plan: Plan,
        b: Buffers,
    }
    impl M {
        fn new(plan: Plan) -> M {
            let b = Buffers::new(&plan);
            M { plan, b }
        }
        fn load(&mut self, x: &BigUint) {
            self.b.x = self.plan.to_words(&Residue::from_biguint(self.plan.p, x));
        }
        fn read(&self) -> BigUint {
            self.plan.from_words(&self.b.x).unwrap().to_biguint()
        }
        fn square(&mut self, add0: i64) {
            execute_all(&self.plan.square_launches(add0), &mut self.b);
        }
        fn mul(&mut self, y: &BigUint) {
            self.b.src = self.plan.to_words(&Residue::from_biguint(self.plan.p, y));
            execute_all(&self.plan.mul_launches(), &mut self.b);
        }
    }

    fn sub2(x: &BigUint, n: &BigUint) -> BigUint {
        (x + n - 2u32) % n
    }

    #[test]
    fn field_mul_matches_u128() {
        let mut rng = Rng(7);
        let mut vals = vec![0, 1, 2, Q - 1, Q - 2, 1 << 60, (1 << 60) - 1, HALF_Q];
        vals.extend((0..200).map(|_| rng.next() % Q));
        for &a in &vals {
            for &b in &vals {
                let want = ((u128::from(a) * u128::from(b)) % u128::from(Q)) as u64;
                assert_eq!(mul(a, b), want, "{a} * {b}");
            }
            for s in 0..61 {
                assert_eq!(rotl(a, s), mul(a, 1 << s), "rotl {a} {s}");
            }
            assert_eq!(add(a, Q - 1 - a.min(Q - 1)), Q - 1);
            assert_eq!(sub(a, a), 0);
        }
    }

    #[test]
    fn roots_and_weights() {
        let h = root_2_61();
        assert_eq!(pow2(h, 1 << 61), (1, 0));
        assert_eq!(pow2(h, 1 << 60), (Q - 1, 0), "h^(2^60) must be -1");
        // Norm 1: the conjugate is the inverse.
        assert_eq!(mul2(h, (h.0, sub(0, h.1))), (1, 0));
        for log_n in MIN_LOG_N..=MAX_LOG_N {
            let n = 1u64 << log_n;
            let w = pow2(h, 1u128 << (61 - log_n));
            assert_eq!(pow2(w, u128::from(n)), (1, 0));
            assert_eq!(pow2(w, u128::from(n / 2)), (Q - 1, 0));
            let pl = plan_for_len(10 * n as u32, log_n).unwrap();
            // (2^kinv)^n = 2 mod q.
            assert_eq!(pow2((1u64 << pl.kinv, 0), u128::from(n)), (2, 0));
        }
    }

    #[test]
    fn transform_is_cyclic_convolution() {
        let mut rng = Rng(11);
        for log_n in [2u32, 3, 5, 8] {
            let plan = plan_for_len(10 << log_n, log_n).unwrap();
            let n = plan.n as usize;
            let mut b = Buffers::new(&plan);
            let xs: Vec<u64> = (0..2 * n).map(|_| rng.next() % Q).collect();
            let ys: Vec<u64> = (0..2 * n).map(|_| rng.next() % Q).collect();
            b.t = xs.clone();
            b.t2 = ys.clone();
            let mut seq = Vec::new();
            plan.forward(Buf::T, &mut seq);
            plan.forward(Buf::T2, &mut seq);
            for l in seq.iter().filter(|l| l.kernel == Kernel::FwdStage) {
                execute(l, &mut b);
            }
            execute(
                &Launch {
                    kernel: Kernel::Mul,
                    threads: plan.n,
                    args: vec![Arg::Buf(Buf::T), Arg::Buf(Buf::T2), Arg::U64(plan.n)],
                },
                &mut b,
            );
            let mut back = Vec::new();
            plan.back(0, &mut back);
            for l in back.iter().filter(|l| l.kernel == Kernel::InvStage) {
                execute(l, &mut b);
            }
            for k in 0..n {
                let mut acc = (0, 0);
                for i in 0..n {
                    let j = (k + n - i) % n;
                    let prod = mul2((xs[2 * i], xs[2 * i + 1]), (ys[2 * j], ys[2 * j + 1]));
                    acc = (add(acc.0, prod.0), add(acc.1, prod.1));
                }
                let nn = plan.n % Q;
                assert_eq!(
                    (b.t[2 * k], b.t[2 * k + 1]),
                    (mul(acc.0, nn), mul(acc.1, nn)),
                    "n={n} k={k}"
                );
            }
        }
    }

    #[test]
    fn supported_range_is_contiguous() {
        let (lo, hi) = supported_exponents();
        assert_eq!(lo, 12);
        assert!(plan(lo - 1).is_none());
        assert!(plan(hi).is_some());
        assert!(plan(hi + 1).is_none());
        for log_n in MIN_LOG_N..=MAX_LOG_N {
            let n = 1u32 << log_n;
            let top = max_exponent_for_len(log_n).unwrap();
            // Every class from the smallest width to the top is valid.
            for p in [3 * n, 3 * n + 1, top - 1, top] {
                assert_eq!(
                    plan_for_len(p, log_n).map(|pl| pl.log_n),
                    Some(log_n),
                    "p={p}"
                );
            }
            for w in MIN_WIDTH..top / n {
                assert!(
                    plan_for_len(w * n, log_n).is_some()
                        && plan_for_len(w * n + 1, log_n).is_some()
                );
            }
            if log_n < MAX_LOG_N {
                // The next length picks up exactly where this one stops.
                assert_eq!(plan(top + 1).unwrap().log_n, log_n + 1, "after {top}");
                assert!(3 * 2 * n <= top + 1);
            }
        }
        // Brute force over the low range.
        for p in lo..200_000 {
            let pl = plan(p).unwrap_or_else(|| panic!("no plan for {p}"));
            assert!(pl.coeff_bound <= CAPACITY_LIMIT && pl.w_min >= MIN_WIDTH);
            assert!(pl.coeff_bound < u128::from(HALF_Q));
        }
    }

    #[test]
    fn launches_match_kernel_signatures() {
        for p in [12u32, 13, 127, 1279, 86243] {
            let pl = plan(p).unwrap();
            for l in pl
                .square_launches(-2)
                .iter()
                .chain(pl.mul_launches().iter())
            {
                let kinds: Vec<Param> = l
                    .args
                    .iter()
                    .map(|a| {
                        if matches!(a, Arg::Buf(_)) {
                            Param::Ptr
                        } else {
                            Param::Scalar
                        }
                    })
                    .collect();
                assert_eq!(kinds, l.kernel.params(), "{:?}", l.kernel);
                assert!(l.threads == pl.n || l.threads == pl.n / 2);
            }
        }
    }

    #[test]
    fn import_export_round_trip() {
        let mut rng = Rng(3);
        for p in [12u32, 13, 31, 61, 89, 127, 521, 607, 1279, 4423, 9941] {
            let pl = plan(p).unwrap();
            let mut vals = edges(p);
            vals.extend((0..20).map(|_| random(p, &mut rng)));
            for x in vals {
                let r = Residue::from_biguint(p, &x);
                let w = pl.to_words(&r);
                assert_eq!(pl.from_words(&w).unwrap(), r, "p={p}");
            }
            // Words whose value is exactly N (the words of N - 1, plus one at
            // word 0) carry out to the all-ones pattern, which must come back
            // as the unique zero.
            let mut ws = pl.to_words(&Residue::from_biguint(p, &(modulus(p) - 1u32)));
            ws[0] = (ws[0] as i64 + 1) as u64;
            assert!(pl.from_words(&ws).unwrap().is_zero());
            let mut neg = vec![0u64; pl.n as usize];
            neg[0] = (-1i64) as u64;
            assert_eq!(pl.from_words(&neg).unwrap().to_biguint(), modulus(p) - 1u32);
        }
    }

    #[test]
    fn squarings_and_ll_steps_match_bigint() {
        let mut rng = Rng(5);
        for p in [
            12u32, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 4423, 9941, 23209, 86243,
        ] {
            let pl = plan(p).unwrap();
            let n = modulus(p);
            let mut vals = edges(p);
            vals.extend((0..3).map(|_| random(p, &mut rng)));
            for x in vals {
                let mut m = M::new(pl.clone());
                m.load(&x);
                let mut want = x.clone();
                for _ in 0..4 {
                    m.square(0);
                    want = (&want * &want) % &n;
                    assert_eq!(m.read(), want, "square p={p}");
                }
                for _ in 0..4 {
                    m.square(-2);
                    want = sub2(&(&want * &want), &n);
                    assert_eq!(m.read(), want, "ll p={p}");
                }
            }
        }
    }

    #[test]
    fn mul_matches_bigint() {
        let mut rng = Rng(9);
        for p in [12u32, 13, 31, 127, 607, 4423, 44497] {
            let pl = plan(p).unwrap();
            let n = modulus(p);
            let mut vals = edges(p);
            vals.extend((0..3).map(|_| random(p, &mut rng)));
            for x in &vals {
                for y in vals.iter().take(5).chain([random(p, &mut rng)].iter()) {
                    let mut m = M::new(pl.clone());
                    m.load(x);
                    m.mul(y);
                    assert_eq!(m.read(), (x * y) % &n, "mul p={p}");
                    // The result is a valid input to further arithmetic.
                    m.mul(y);
                    assert_eq!(m.read(), (x * y * y) % &n, "mul twice p={p}");
                }
            }
        }
    }

    /// Exact signed value of balanced words.
    fn value(pl: &Plan, words: &[u64]) -> BigInt {
        let (p, l) = (u64::from(pl.p), u64::from(pl.log_n));
        words
            .iter()
            .enumerate()
            .fold(BigInt::zero(), |acc, (k, &w)| {
                acc + (BigInt::from(w as i64) << offset(p, l, k as u64))
            })
    }

    fn mod_n(v: &BigInt, p: u32) -> BigUint {
        let n = BigInt::from(modulus(p));
        (((v % &n) + &n) % &n).to_biguint().unwrap()
    }

    #[test]
    fn worst_case_words_at_capacity_edge() {
        for log_n in MIN_LOG_N..=10 {
            let p = max_exponent_for_len(log_n).unwrap();
            let pl = plan_for_len(p, log_n).unwrap();
            assert_eq!(plan(p).unwrap().log_n, log_n);
            let n = pl.n as usize;
            let wb = pl.word_bound as i64;
            let patterns: Vec<Vec<i64>> = vec![
                vec![wb; n],
                vec![-wb; n],
                (0..n).map(|k| if k % 2 == 0 { wb } else { -wb }).collect(),
                (0..n).map(|k| if k < n / 2 { wb } else { -wb }).collect(),
            ];
            let (pp, l) = (u64::from(p), u64::from(log_n));
            for words in patterns {
                let raw: Vec<u64> = words.iter().map(|&v| v as u64).collect();
                let mut b = Buffers::new(&pl);
                b.x = raw.clone();
                let seq = pl.square_launches(-2);
                let cut = seq.iter().position(|l| l.kernel == Kernel::CarryA).unwrap();
                execute_all(&seq[..cut], &mut b);
                // Schoolbook: z_k = sum x_i x_j 2^d, d = off_i + off_j - off_k (- p on wrap).
                let mut zmax = 0i128;
                for k in 0..n {
                    let mut z = 0i128;
                    for i in 0..n {
                        let j = (k + n - i) % n;
                        let mut d = offset(pp, l, i as u64) + offset(pp, l, j as u64);
                        d -= offset(pp, l, k as u64) + if i + j >= n { pp } else { 0 };
                        assert!(d <= 1);
                        z += (i128::from(words[i]) * i128::from(words[j])) << d;
                    }
                    assert_eq!(b.z[k] as i64 as i128, z, "log_n={log_n} k={k}");
                    zmax = zmax.max(z.abs());
                }
                assert!(zmax as u128 <= pl.coeff_bound && pl.coeff_bound <= CAPACITY_LIMIT);
                execute_all(&seq[cut..], &mut b);
                let v = value(&pl, &raw);
                let want = mod_n(&(&v * &v - 2), p);
                assert_eq!(
                    pl.from_words(&b.x).unwrap().to_biguint(),
                    want,
                    "log_n={log_n}"
                );
            }
        }
    }

    #[test]
    fn random_squarings_at_capacity_edge() {
        let mut rng = Rng(13);
        for log_n in MIN_LOG_N..=12 {
            let p = max_exponent_for_len(log_n).unwrap();
            let pl = plan_for_len(p, log_n).unwrap();
            let n = modulus(p);
            let mut m = M::new(pl);
            let mut want = random(p, &mut rng);
            m.load(&want);
            for _ in 0..6 {
                m.square(-2);
                want = sub2(&(&want * &want), &n);
            }
            assert_eq!(m.read(), want, "p={p}");
        }
    }

    #[test]
    fn large_lengths_at_capacity_edge() {
        // The largest lengths this test can afford: one squaring, one LL step
        // and one multiplication at the capacity edge of n = 2^16 and 2^18.
        let mut rng = Rng(19);
        for log_n in [16u32, 18] {
            let p = max_exponent_for_len(log_n).unwrap();
            let pl = plan_for_len(p, log_n).unwrap();
            let n = modulus(p);
            let mut m = M::new(pl);
            let (x, y) = (random(p, &mut rng), random(p, &mut rng));
            m.load(&x);
            m.square(0);
            m.square(-2);
            m.mul(&y);
            let s = (&x * &x) % &n;
            let want = (sub2(&(&s * &s), &n) * &y) % &n;
            assert_eq!(m.read(), want, "p={p}");
        }
    }

    #[test]
    fn carry_slack_bound_holds() {
        // Brute force the carry passes on extreme and random coefficients for
        // small widths: the output stays within 2^(w - 1) + slack and the value
        // is preserved modulo N, with and without the LL subtraction.
        let mut rng = Rng(17);
        for (p, log_n) in [
            (12u32, 2u32),
            (13, 2),
            (15, 2),
            (25, 3),
            (31, 3),
            (40, 3),
            (70, 4),
            (150, 4),
        ] {
            let pl = plan_for_len(p, log_n).unwrap();
            let n = pl.n as usize;
            let zb = pl.coeff_bound as i64;
            let (pp, l) = (u64::from(p), u64::from(log_n));
            for trial in 0..400 {
                let z: Vec<i64> = (0..n)
                    .map(|_| match trial % 4 {
                        0 => zb,
                        1 => -zb,
                        2 => {
                            if rng.next() & 1 == 0 {
                                zb
                            } else {
                                -zb
                            }
                        }
                        _ => (rng.next() % (2 * zb as u64 + 1)) as i64 - zb,
                    })
                    .collect();
                for add0 in [0i64, -2] {
                    let mut b = Buffers::new(&pl);
                    b.z = z.iter().map(|&v| v as u64).collect();
                    let seq = pl.square_launches(add0);
                    let cut = seq.iter().position(|l| l.kernel == Kernel::CarryA).unwrap();
                    execute_all(&seq[cut..], &mut b);
                    for k in 0..n {
                        let lim = (1i64 << (width(pp, l, k as u64) - 1)) + pl.slack as i64;
                        assert!(
                            (b.x[k] as i64).abs() <= lim,
                            "p={p} k={k} x={}",
                            b.x[k] as i64
                        );
                    }
                    let zv = value(&pl, &b.z) + add0;
                    assert_eq!(mod_n(&value(&pl, &b.x), p), mod_n(&zv, p), "p={p}");
                }
            }
        }
    }

    #[test]
    fn pieces_may_exceed_length_for_tiny_exponents() {
        let pl = plan(13).unwrap();
        assert_eq!(pl.n, 4);
        assert!(pl.pieces > pl.n, "{pl:?}");
        let pl = plan(1279).unwrap();
        assert!(pl.pieces <= pl.n);
    }

    fn ll_is_zero(p: u32) -> bool {
        let pl = plan(p).unwrap();
        let mut m = M::new(pl);
        m.load(&BigUint::from(4u32));
        for _ in 0..p - 2 {
            m.square(-2);
        }
        let r = m.read();
        // Cross-check the final residue against bigint for the small cases.
        if p < 3000 {
            let n = modulus(p);
            let mut s = BigUint::from(4u32);
            for _ in 0..p - 2 {
                s = sub2(&(&s * &s), &n);
            }
            assert_eq!(r, s, "p={p}");
        }
        r.is_zero()
    }

    #[test]
    fn complete_ll_runs() {
        for p in [
            13u32, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423,
            9689, 9941,
        ] {
            assert!(ll_is_zero(p), "M{p} is prime");
        }
        for p in [
            23u32, 29, 37, 41, 43, 47, 53, 59, 67, 71, 73, 79, 83, 97, 101, 103, 109, 113, 131,
            509, 1277, 4421,
        ] {
            assert!(!ll_is_zero(p), "M{p} is composite");
        }
    }

    #[test]
    fn export_rejects_out_of_bound_words() {
        let pl = plan(1279).unwrap();
        for k in [0u64, 3, pl.n - 1] {
            let lim = (1i64 << (width(1279, u64::from(pl.log_n), k) - 1)) + pl.slack as i64;
            let mut w = vec![0u64; pl.n as usize];
            w[k as usize] = (lim + 1) as u64;
            assert!(pl.from_words(&w).is_err());
            w[k as usize] = (-lim - 1) as u64;
            assert!(pl.from_words(&w).is_err());
            w[k as usize] = (-lim) as u64;
            assert!(pl.from_words(&w).is_ok());
        }
        assert!(pl.from_words(&[0u64; 3]).is_err());
        for v in [i64::MIN, i64::MAX] {
            let mut w = vec![0u64; pl.n as usize];
            w[1] = v as u64;
            assert!(pl.from_words(&w).is_err(), "{v}");
        }
    }
}
