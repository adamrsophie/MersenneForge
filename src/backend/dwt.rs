// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Pure-Rust floating-point backend: the irrational-base discrete weighted
//! transform (IBDWT) of Crandall and Fagin (1994), in f64.
//!
//! # Representations
//!
//! Three representations appear in this module and are kept distinct:
//!
//! 1. Canonical [`Residue`]: little-endian `u32` limbs, `0 <= x < N`,
//!    `N = 2^p - 1`. The only form that crosses the [`Backend`] trait.
//! 2. `BalancedWords`: the resident form. For a transform length `n`, word
//!    `i` starts at bit `b_i = ceil(p*i/n)` and has width
//!    `w_i = b_{i+1} - b_i`, which is `floor(p/n)` or `ceil(p/n)`. The value
//!    is `sum(d_i * 2^b_i) mod N` with balanced digits
//!    `-2^(w_i - 1) <= d_i < 2^(w_i - 1)`. Equal integers modulo N can have
//!    different balanced forms, so equality goes through the canonical form.
//! 3. Weighted transform domain: complex `f64` vectors holding the forward FFT
//!    of `a_i * d_i`, with weights `a_i = 2^(b_i - p*i/n)` in `[1, 2)`.
//!    Scratch only; never stored between calls.
//!
//! Because `b_n = p` and `2^p = 1 mod N`, the weighted cyclic convolution of
//! two weighted digit vectors, divided by the weights, is the product modulo N
//! in the same variable base. Rounding each output to the nearest integer and
//! propagating carries (the carry out of the top word wraps into word 0)
//! gives balanced words again.
//!
//! # Transform length and the bits-per-word bound
//!
//! With balanced digits `|d_i| <= 2^(w_max - 1)` and weights `a_i < 2`, every
//! weighted input has magnitude below `2^w_max`, so every exact convolution
//! output has magnitude at most `n * 2^(2 * w_max)`. A length `n` is accepted
//! for `p` only when
//!
//! ```text
//! 2 * w_max + log2(n) <= 53 - SAFETY_BITS,   w_max = ceil(p / n)
//! ```
//!
//! that is, the worst-case exact output stays `SAFETY_BITS` below the 53-bit
//! f64 mantissa, leaving those bits for the transform's rounding error. The
//! limit on `w_max` shrinks by half a bit for every doubling of `n`. This is a
//! worst-case magnitude bound, not a proof about floating-point rounding: the
//! measured roundoff (see `tests/dwt_backend.rs`) is the evidence that the
//! margin is enough, and the roundoff check below is the run-time guard.
//!
//! Candidate lengths are `2^k`, `3 * 2^k`, `5 * 2^k`, and `7 * 2^k`, from 1 to
//! [`MAX_LENGTH`], and also never more than `p` (each word must hold at least
//! one bit). The smallest candidate satisfying the bound is used. The
//! supported range is `2 <= p <= MAX_LENGTH * max_word_bits(MAX_LENGTH)`;
//! `p = 1` is rejected because `N = 1` has no nonzero residue.
//!
//! # Roundoff check
//!
//! After every inverse transform, each output's distance from its nearest
//! integer is measured. The largest distance of a call is compared against
//! [`ROUNDOFF_LIMIT`]; a larger value, a non-finite value, or an output beyond
//! [`MAX_OUTPUT_MAGNITUDE`] is reported as [`BackendError::Fault`]. The operation then leaves the
//! target residue exactly as it was before the call (a batch is undone as a
//! whole), so no silently wrong value remains resident. The running maximum is
//! reported by `diagnostics()`.

use std::collections::HashMap;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::cpu::Corruption;
use super::{
    new_handle, Backend, BackendError, BackendIdentity, BackendResult, MemoryEstimate, Reg,
};
use crate::residue::Residue;

/// Bits of the f64 mantissa kept free for rounding error by the length bound.
pub const SAFETY_BITS: f64 = 6.0;

/// Largest transform length. At this length one plan holds a few hundred MiB
/// of weights, twiddles, and scratch, and every resident residue 16 MiB.
pub const MAX_LENGTH: usize = 1 << 22;

/// Largest accepted distance of a transform output from its nearest integer.
///
/// Any distance below 0.5 still rounds to some integer, but the observed
/// distance only estimates the true error: when the largest observed value
/// approaches 0.5, other outputs may already have crossed it and rounded to
/// the wrong integer unseen. 0.40 is the conventional limit used by FFT
/// Mersenne engines. With the length bound above, measured maxima are orders
/// of magnitude lower, so reaching 0.40 means the plan or the data is wrong.
pub const ROUNDOFF_LIMIT: f64 = 0.40;

/// Widest word the resident `i32` storage and the carry code accept.
const STORAGE_WORD_BITS: u32 = 30;

/// Largest accepted output magnitude, `2^(53 - SAFETY_BITS)`: the worst-case
/// exact output of every plan `new` selects stays at or below it. The cap is
/// what makes the distance check meaningful. Every f64 from `2^52` up is an
/// integer and between `2^51` and `2^52` the distance is 0 or 0.5, so a
/// wrongly rounded output there would pass the distance check unseen; below
/// the cap at least five fractional bits remain. (A sweep of forced
/// too-short transforms with only a `2^53` cap accepted wrong products.)
pub const MAX_OUTPUT_MAGNITUDE: f64 = (1u64 << 47) as f64;
const _: () = assert!(
    SAFETY_BITS == 6.0,
    "MAX_OUTPUT_MAGNITUDE assumes SAFETY_BITS = 6"
);

type C64 = Complex<f64>;

/// The resident representation: one balanced digit per transform word.
#[derive(Clone, Debug)]
struct BalancedWords(Vec<i32>);

/// Largest word width the length bound allows for transform length `n`.
pub fn max_word_bits(n: usize) -> u32 {
    let w = ((53.0 - SAFETY_BITS - (n as f64).log2()) / 2.0).floor();
    (w as u32).min(STORAGE_WORD_BITS)
}

/// Candidate transform lengths in increasing order.
fn candidate_lengths() -> Vec<usize> {
    let mut v = Vec::new();
    for m in [1usize, 3, 5, 7] {
        let mut n = m;
        while n <= MAX_LENGTH {
            v.push(n);
            n *= 2;
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Smallest candidate length satisfying the bound for `p`, if any.
fn choose_length(p: u32) -> Option<usize> {
    candidate_lengths()
        .into_iter()
        .find(|&n| n as u64 <= p as u64 && (p as u64).div_ceil(n as u64) <= max_word_bits(n) as u64)
}

/// Inclusive exponent range this backend accepts.
pub fn supported_exponents() -> (u32, u32) {
    let max = MAX_LENGTH as u64 * max_word_bits(MAX_LENGTH) as u64;
    (2, max.min(u32::MAX as u64) as u32)
}

fn plan_name(n: usize) -> String {
    format!("f64-dwt n={n}")
}

/// Host memory for exponent p at the length `new` would choose. A register
/// is `n` balanced `i32` digits. The fixed figure counts the plan tables
/// (word offsets, widths, weights, inverse weights) and the transform
/// scratch (two complex buffers, the FFT scratch, and the carry buffer). The
/// FFT library's own twiddle tables (about one complex value per element for
/// each direction) are not counted.
pub fn memory_estimate(p: u32) -> BackendResult<MemoryEstimate> {
    let n = length_for(p)?;
    let mut planner = FftPlanner::<f64>::new();
    let fft = fft_scratch_len(&*planner.plan_fft_forward(n), &*planner.plan_fft_inverse(n));
    Ok(MemoryEstimate {
        plan: plan_name(n),
        bytes_per_register: 4 * n as u64,
        fixed_bytes: fixed_bytes(n, fft),
        on_device: false,
    })
}

fn fft_scratch_len(fwd: &dyn Fft<f64>, inv: &dyn Fft<f64>) -> usize {
    fwd.get_inplace_scratch_len()
        .max(inv.get_inplace_scratch_len())
}

/// Bytes of the plan tables and scratch built by `DwtBackend::build`.
fn fixed_bytes(n: usize, fft_scratch: usize) -> u64 {
    let (n, f) = (n as u64, fft_scratch as u64);
    let c = std::mem::size_of::<C64>() as u64;
    let tables = 4 * (n + 1) + 4 * n + 8 * n + 8 * n;
    let scratch = 2 * c * n + c * f + 8 * n;
    tables + scratch
}

fn length_for(p: u32) -> BackendResult<usize> {
    let (lo, hi) = supported_exponents();
    if p < lo || p > hi {
        return Err(BackendError::Unsupported(format!(
            "exponent {p} is outside the dwt backend range {lo}..={hi}"
        )));
    }
    choose_length(p)
        .ok_or_else(|| BackendError::Unsupported(format!("no transform length for exponent {p}")))
}

/// Fixed data of one (p, n) plan.
struct Plan {
    p: u32,
    n: usize,
    /// b_i for i in 0..=n; b_n = p.
    offsets: Vec<u32>,
    widths: Vec<u32>,
    weight: Vec<f64>,
    inv_weight: Vec<f64>,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
    /// Opt-in packed real transform (see [`Packed`]).
    packed: Option<Packed>,
    /// Opt-in threaded transforms (forward, inverse) of length n, or of the
    /// packed length when packing is on.
    par: Option<(ParFft, ParFft)>,
}

/// The real input packed two values per complex value: a length n/2 complex
/// transform of z_j = a_{2j} + i a_{2j+1}, untangled into the length n
/// spectrum X_0..X_{n/2} (the rest is its conjugate mirror), about half the
/// work of the complex transform of real data. Opt-in through the
/// `MERSENNE_FORGE_DWT_PACKED=1` environment variable; its plan name differs,
/// so checkpoints never cross between the two forms.
///
/// Untangle: with E = FFT(a_even), O = FFT(a_odd), w = exp(-2 pi i / n),
/// Z = E + iO, so E_k = (Z_k + conj Z_{m-k}) / 2, O_k = (Z_k - conj Z_{m-k}) / 2i,
/// and X_k = E_k + w^k O_k, X_{k+m} = E_k - w^k O_k. The inverse runs the
/// same identities backwards: E_k = (Y_k + conj Y_{m-k}) / 2,
/// O_k = (Y_k - conj Y_{m-k}) / (2 w^k), and IFFT(E + iO) / m gives the even
/// and odd outputs as real and imaginary parts.
struct Packed {
    m: usize,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
    /// w^k for k = 0..=m.
    tw: Vec<C64>,
}

impl Packed {
    fn new(n: usize, planner: &mut FftPlanner<f64>) -> Packed {
        let m = n / 2;
        let tw = (0..=m)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                C64::new(a.cos(), a.sin())
            })
            .collect();
        Packed {
            m,
            fwd: planner.plan_fft_forward(m),
            inv: planner.plan_fft_inverse(m),
            tw,
        }
    }

    /// X_k from Z_k and conj Z_{m-k}.
    fn untangle(zk: C64, zc: C64, w: C64) -> C64 {
        (zk + zc) * 0.5 + w * ((zk - zc) * C64::new(0.0, -0.5))
    }

    /// E_k + i O_k from Y_k and conj Y_{m-k}.
    fn retangle(yk: C64, yc: C64, w: C64) -> C64 {
        (yk + yc) * 0.5 + C64::new(0.0, 1.0) * ((yk - yc) * 0.5 * w.conj())
    }
}

/// A four step FFT of length n = n1 * n2 spread over threads (opt-in with
/// `MERSENNE_FORGE_DWT_THREADS=N`). With j = j1 + n1 j2 and k = n2 k1 + k2,
///
/// ```text
/// X[n2 k1 + k2] = sum_j1 W_n^(j1 k2) W_n1^(j1 k1) sum_j2 x[j1 + n1 j2] W_n2^(j2 k2)
/// ```
///
/// so: n1 transforms of length n2 (one per j1, on a gathered row), a twiddle
/// by W_n^(j1 k2), a transpose, n2 transforms of length n1, and a transpose
/// back. Each step splits its rows over the threads. Unnormalized in both
/// directions, as rustfft is.
struct ParFft {
    n1: usize,
    n2: usize,
    f1: Arc<dyn Fft<f64>>,
    f2: Arc<dyn Fft<f64>>,
    /// W_n^(j1 k2) at [j1 * n2 + k2], with the transform's sign.
    tw: Vec<C64>,
    threads: usize,
    tmp: std::sync::Mutex<Vec<C64>>,
}

impl ParFft {
    /// None when n has no useful split (both factors at least 16).
    fn new(n: usize, inverse: bool, threads: usize) -> Option<ParFft> {
        let n1 = (16..=n.isqrt()).rev().find(|&d| n.is_multiple_of(d))?;
        let n2 = n / n1;
        let mut planner = FftPlanner::<f64>::new();
        let (f1, f2) = if inverse {
            (planner.plan_fft_inverse(n1), planner.plan_fft_inverse(n2))
        } else {
            (planner.plan_fft_forward(n1), planner.plan_fft_forward(n2))
        };
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut tw = Vec::with_capacity(n);
        for j1 in 0..n1 {
            for k2 in 0..n2 {
                // Reduce the exponent first so the angle stays accurate.
                let e = (j1 * k2) % n;
                let a = sign * 2.0 * std::f64::consts::PI * e as f64 / n as f64;
                tw.push(C64::new(a.cos(), a.sin()));
            }
        }
        Some(ParFft {
            n1,
            n2,
            f1,
            f2,
            tw,
            threads: threads.max(1),
            tmp: std::sync::Mutex::new(vec![C64::new(0.0, 0.0); n]),
        })
    }

    /// Rows per thread for `rows` rows.
    fn chunk(&self, rows: usize) -> usize {
        rows.div_ceil(self.threads).max(1)
    }

    fn process(&self, buf: &mut [C64]) {
        let (n1, n2) = (self.n1, self.n2);
        let mut guard = self.tmp.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = &mut guard[..];
        // 1. Rows j1 of tmp: gather x[j1 + n1 j2], transform (length n2),
        //    twiddle.
        let rows = self.chunk(n1);
        std::thread::scope(|s| {
            let src = &*buf;
            for (c, block) in tmp.chunks_mut(rows * n2).enumerate() {
                s.spawn(move || {
                    let mut scratch = vec![C64::new(0.0, 0.0); self.f2.get_inplace_scratch_len()];
                    for (r, row) in block.chunks_mut(n2).enumerate() {
                        let j1 = c * rows + r;
                        for (j2, v) in row.iter_mut().enumerate() {
                            *v = src[j1 + n1 * j2];
                        }
                        self.f2.process_with_scratch(row, &mut scratch);
                        for (k2, v) in row.iter_mut().enumerate() {
                            *v *= self.tw[j1 * n2 + k2];
                        }
                    }
                });
            }
        });
        // 2. Rows k2 of buf: gather tmp[j1 n2 + k2], transform (length n1).
        let rows = self.chunk(n2);
        std::thread::scope(|s| {
            let src = &*tmp;
            for (c, block) in buf.chunks_mut(rows * n1).enumerate() {
                s.spawn(move || {
                    let mut scratch = vec![C64::new(0.0, 0.0); self.f1.get_inplace_scratch_len()];
                    for (r, row) in block.chunks_mut(n1).enumerate() {
                        let k2 = c * rows + r;
                        for (j1, v) in row.iter_mut().enumerate() {
                            *v = src[j1 * n2 + k2];
                        }
                        self.f1.process_with_scratch(row, &mut scratch);
                    }
                });
            }
        });
        // 3. X[n2 k1 + k2] = buf[k2 n1 + k1]: transpose into tmp, copy back.
        let rows = self.chunk(n1);
        std::thread::scope(|s| {
            let src = &*buf;
            for (c, block) in tmp.chunks_mut(rows * n2).enumerate() {
                s.spawn(move || {
                    for (r, row) in block.chunks_mut(n2).enumerate() {
                        let k1 = c * rows + r;
                        for (k2, v) in row.iter_mut().enumerate() {
                            *v = src[k2 * n1 + k1];
                        }
                    }
                });
            }
        });
        buf.copy_from_slice(tmp);
    }
}

/// Run one transform: the threaded four step form when present, otherwise
/// the single threaded plan.
fn run_fft(f: &Arc<dyn Fft<f64>>, par: Option<&ParFft>, buf: &mut [C64], scratch: &mut [C64]) {
    match par {
        Some(p) => p.process(buf),
        None => f.process_with_scratch(buf, scratch),
    }
}

/// Threads asked for with MERSENNE_FORGE_DWT_THREADS (2 or more), if any.
fn threads_requested() -> Option<usize> {
    std::env::var("MERSENNE_FORGE_DWT_THREADS")
        .ok()?
        .parse()
        .ok()
        .filter(|&t: &usize| t >= 2)
}

/// Whether the packed transform was asked for.
fn packed_requested() -> bool {
    std::env::var("MERSENNE_FORGE_DWT_PACKED").is_ok_and(|v| v == "1")
}

impl Plan {
    fn new(p: u32, n: usize) -> Plan {
        let (pu, nu) = (p as u64, n as u64);
        let offsets: Vec<u32> = (0..=nu).map(|i| (pu * i).div_ceil(nu) as u32).collect();
        let widths = offsets.windows(2).map(|w| w[1] - w[0]).collect();
        // b_i - p*i/n = (b_i*n - p*i) / n, with an exact integer numerator in [0, n).
        let frac = |i: usize| (offsets[i] as u64 * nu - pu * i as u64) as f64 / n as f64;
        let weight = (0..n).map(|i| frac(i).exp2()).collect();
        let inv_weight = (0..n).map(|i| (-frac(i)).exp2()).collect();
        let mut planner = FftPlanner::<f64>::new();
        Plan {
            p,
            n,
            offsets,
            widths,
            weight,
            inv_weight,
            fwd: planner.plan_fft_forward(n),
            inv: planner.plan_fft_inverse(n),
            packed: None,
            par: None,
        }
    }

    /// The same plan with the packed real transform, when n allows it.
    fn with_packing(mut self) -> Plan {
        if self.n >= 4 && self.n.is_multiple_of(2) {
            self.packed = Some(Packed::new(self.n, &mut FftPlanner::<f64>::new()));
        }
        self
    }

    /// Packed weighted transform of balanced words: X_0..=X_m into buf.
    fn forward_packed(&self, pk: &Packed, x: &BalancedWords, buf: &mut [C64], scratch: &mut [C64]) {
        let m = pk.m;
        for (j, c) in buf[..m].iter_mut().enumerate() {
            *c = C64::new(
                x.0[2 * j] as f64 * self.weight[2 * j],
                x.0[2 * j + 1] as f64 * self.weight[2 * j + 1],
            );
        }
        run_fft(
            &pk.fwd,
            self.par.as_ref().map(|p| &p.0),
            &mut buf[..m],
            scratch,
        );
        let z0 = buf[0];
        buf[m] = C64::new(z0.re - z0.im, 0.0);
        buf[0] = C64::new(z0.re + z0.im, 0.0);
        for k in 1..=m / 2 {
            let (zk, zmk) = (buf[k], buf[m - k]);
            buf[k] = Packed::untangle(zk, zmk.conj(), pk.tw[k]);
            buf[m - k] = Packed::untangle(zmk, zk.conj(), pk.tw[m - k]);
        }
    }

    /// Packed counterpart of [`Plan::finish`] for a spectrum Y_0..=Y_m in buf.
    #[allow(clippy::too_many_arguments)]
    fn finish_packed(
        &self,
        pk: &Packed,
        buf: &mut [C64],
        scratch: &mut [C64],
        vals: &mut [i64],
        addend: i64,
        max_seen: &mut f64,
        out: &mut BalancedWords,
    ) -> BackendResult<()> {
        let m = pk.m;
        let (y0, ym) = (buf[0], buf[m]);
        buf[0] = Packed::retangle(y0, ym.conj(), pk.tw[0]);
        for k in 1..=m / 2 {
            let (yk, ymk) = (buf[k], buf[m - k]);
            buf[k] = Packed::retangle(yk, ymk.conj(), pk.tw[k]);
            buf[m - k] = Packed::retangle(ymk, yk.conj(), pk.tw[m - k]);
        }
        run_fft(
            &pk.inv,
            self.par.as_ref().map(|p| &p.1),
            &mut buf[..m],
            scratch,
        );
        let scale = 1.0 / m as f64;
        let mut worst = 0.0f64;
        for i in 0..self.n {
            let c = buf[i / 2];
            let raw = if i % 2 == 0 { c.re } else { c.im };
            let x = raw * scale * self.inv_weight[i];
            let r = x.round();
            let err = (x - r).abs();
            if !(err <= ROUNDOFF_LIMIT && r.abs() <= MAX_OUTPUT_MAGNITUDE) {
                *max_seen = max_seen.max(if err.is_nan() { f64::INFINITY } else { err });
                return Err(BackendError::Fault(format!(
                    "roundoff check failed (packed transform): word {i} is {x} (distance {err:.4} from an integer, limit {ROUNDOFF_LIMIT}); \
                     transform length {} is too short for p = {} or the data is corrupt",
                    self.n, self.p
                )));
            }
            worst = worst.max(err);
            vals[i] = r as i64;
        }
        *max_seen = max_seen.max(worst);
        vals[0] += addend;
        self.carry(vals, true)?;
        for (o, &v) in out.0.iter_mut().zip(vals.iter()) {
            *o = v as i32;
        }
        Ok(())
    }

    /// Weighted transform of balanced words into `buf`.
    // ponytail: complex FFT of real input, about 2x the work of a real or
    // right-angle transform; upgrade by packing two reals per complex value
    // (or the right-angle weighted transform) once speed matters.
    fn forward(&self, x: &BalancedWords, buf: &mut [C64], scratch: &mut [C64]) {
        for (i, c) in buf.iter_mut().enumerate() {
            *c = C64::new(x.0[i] as f64 * self.weight[i], 0.0);
        }
        run_fft(&self.fwd, self.par.as_ref().map(|p| &p.0), buf, scratch);
    }

    /// Inverse transform, unweight, round, check roundoff, add `addend` to
    /// word 0, and carry into balanced words, written to `out` (reused, no
    /// allocation). Records the largest distance from an integer in
    /// `max_seen`. On `Err`, `out` is unchanged.
    fn finish(
        &self,
        buf: &mut [C64],
        scratch: &mut [C64],
        vals: &mut [i64],
        addend: i64,
        max_seen: &mut f64,
        out: &mut BalancedWords,
    ) -> BackendResult<()> {
        run_fft(&self.inv, self.par.as_ref().map(|p| &p.1), buf, scratch);
        let scale = 1.0 / self.n as f64;
        let mut worst = 0.0f64;
        for i in 0..self.n {
            let x = buf[i].re * scale * self.inv_weight[i];
            let r = x.round();
            let err = (x - r).abs();
            // NaN fails both comparisons, so it is reported, never rounded.
            if !(err <= ROUNDOFF_LIMIT && r.abs() <= MAX_OUTPUT_MAGNITUDE) {
                *max_seen = max_seen.max(if err.is_nan() { f64::INFINITY } else { err });
                return Err(BackendError::Fault(format!(
                    "roundoff check failed: word {i} is {x} (distance {err:.4} from an integer, limit {ROUNDOFF_LIMIT}, magnitude cap 2^47); \
                     transform length {} is too short for p = {} or the data is corrupt",
                    self.n, self.p
                )));
            }
            worst = worst.max(err);
            vals[i] = r as i64;
        }
        *max_seen = max_seen.max(worst);
        vals[0] += addend;
        self.carry(vals, true)?;
        for (o, &v) in out.0.iter_mut().zip(vals.iter()) {
            *o = v as i32;
        }
        Ok(())
    }

    /// Carry propagation in the variable base. `balanced` selects digits in
    /// `[-2^(w-1), 2^(w-1))`, otherwise `[0, 2^w)`. The carry out of the top
    /// word is 2^p = 1 times the carry, so it re-enters at word 0.
    fn carry(&self, vals: &mut [i64], balanced: bool) -> BackendResult<()> {
        let split = |v: i64, w: u32| -> (i64, i64) {
            let d = if balanced {
                let h = 1i64 << (w - 1);
                (v + h).rem_euclid(1i64 << w) - h
            } else {
                v.rem_euclid(1i64 << w)
            };
            (d, (v - d) >> w)
        };
        let mut carry = 0i64;
        for (v, &w) in vals.iter_mut().zip(&self.widths) {
            (*v, carry) = split(*v + carry, w);
        }
        // After one pass the wrapped carry is a few units; it settles within a
        // pass or two. More passes would mean a logic error, reported as such.
        for _ in 0..4 {
            if carry == 0 {
                return Ok(());
            }
            for (v, &w) in vals.iter_mut().zip(&self.widths) {
                if carry == 0 {
                    break;
                }
                (*v, carry) = split(*v + carry, w);
            }
        }
        if carry == 0 {
            Ok(())
        } else {
            Err(BackendError::Fault(format!(
                "carry did not settle (residual carry {carry})"
            )))
        }
    }

    /// Canonical residue to balanced words (exact).
    fn canonical_to_balanced(&self, x: &Residue) -> BackendResult<BalancedWords> {
        let limbs = x.limbs();
        let mut vals = vec![0i64; self.n];
        for (i, v) in vals.iter_mut().enumerate() {
            let (b, w) = (self.offsets[i] as usize, self.widths[i]);
            let lo = limbs[b / 32] as u64;
            let hi = limbs.get(b / 32 + 1).copied().unwrap_or(0) as u64;
            *v = (((hi << 32 | lo) >> (b % 32)) & ((1u64 << w) - 1)) as i64;
        }
        self.carry(&mut vals, true)?;
        Ok(BalancedWords(vals.iter().map(|&v| v as i32).collect()))
    }

    /// Balanced words to the canonical residue (exact; unique zero).
    fn balanced_to_canonical(&self, x: &BalancedWords) -> BackendResult<Residue> {
        let mut vals: Vec<i64> = x.0.iter().map(|&d| d as i64).collect();
        self.carry(&mut vals, false)?;
        let mut limbs = vec![0u32; (self.p as usize).div_ceil(32)];
        for (i, &u) in vals.iter().enumerate() {
            let b = self.offsets[i] as usize;
            let shifted = (u as u64) << (b % 32);
            limbs[b / 32] |= shifted as u32;
            if shifted >> 32 != 0 {
                limbs[b / 32 + 1] |= (shifted >> 32) as u32;
            }
        }
        // Unsigned digits give 0 <= value <= N; N (all ones) becomes zero here.
        Residue::from_limbs_normalizing(self.p, limbs)
            .map_err(|_| BackendError::Fault("export produced a noncanonical residue".into()))
    }
}

/// Reusable transform scratch, so batches do not allocate per iteration.
struct Scratch {
    a: Vec<C64>,
    b: Vec<C64>,
    fft: Vec<C64>,
    vals: Vec<i64>,
}

/// IBDWT backend for one exponent. Single-threaded; owns all its residues.
pub struct DwtBackend {
    plan: Plan,
    scratch: Scratch,
    regs: HashMap<u32, BalancedWords>,
    max_roundoff: f64,
}

impl DwtBackend {
    /// Picks the smallest supported transform length for p. Errors with
    /// BackendError::Unsupported outside supported_exponents().
    pub fn new(p: u32) -> BackendResult<DwtBackend> {
        let mut be = Self::build(p, length_for(p)?);
        if packed_requested() {
            be.use_packed();
        }
        if let Some(t) = threads_requested() {
            be.use_threads(t);
        }
        Ok(be)
    }

    /// Run the transforms as four step FFTs on `threads` threads (when the
    /// length splits into two factors of at least 16). Call after
    /// `use_packed`, which replaces the plan. Used by `new` under
    /// `MERSENNE_FORGE_DWT_THREADS=N`, and by tests directly.
    #[doc(hidden)]
    pub fn use_threads(&mut self, threads: usize) {
        let len = self.plan.packed.as_ref().map_or(self.plan.n, |pk| pk.m);
        self.plan.par = ParFft::new(len, false, threads)
            .and_then(|f| ParFft::new(len, true, threads).map(|i| (f, i)));
    }

    /// Threads the transforms use (1 when not threaded).
    pub fn threads(&self) -> usize {
        self.plan.par.as_ref().map_or(1, |p| p.0.threads)
    }

    /// Switch to the packed real transform (when the length is even and at
    /// least 4). Used by `new` under `MERSENNE_FORGE_DWT_PACKED=1`, and by
    /// tests directly.
    #[doc(hidden)]
    pub fn use_packed(&mut self) {
        let (p, n) = (self.plan.p, self.plan.n);
        let plan = std::mem::replace(&mut self.plan, Plan::new(p, n));
        self.plan = plan.with_packing();
        if let Some(pk) = &self.plan.packed {
            let need = pk
                .fwd
                .get_inplace_scratch_len()
                .max(pk.inv.get_inplace_scratch_len());
            if self.scratch.fft.len() < need {
                self.scratch.fft.resize(need, C64::new(0.0, 0.0));
            }
        }
    }

    /// Whether the packed real transform is in use.
    pub fn is_packed(&self) -> bool {
        self.plan.packed.is_some()
    }

    /// Test-only: use transform length `n` regardless of the bits-per-word
    /// bound, to show that the roundoff check catches a too-short transform.
    /// Still requires `1 <= n <= p` and words of at most 30 bits.
    #[doc(hidden)]
    pub fn with_transform_length(p: u32, n: usize) -> BackendResult<DwtBackend> {
        if p < 2
            || n == 0
            || n as u64 > p as u64
            || (p as u64).div_ceil(n as u64) > STORAGE_WORD_BITS as u64
        {
            return Err(BackendError::Unsupported(format!(
                "transform length {n} is not usable for exponent {p}"
            )));
        }
        Ok(Self::build(p, n))
    }

    fn build(p: u32, n: usize) -> DwtBackend {
        let plan = Plan::new(p, n);
        let fft_len = fft_scratch_len(&*plan.fwd, &*plan.inv);
        let zero = C64::new(0.0, 0.0);
        DwtBackend {
            scratch: Scratch {
                a: vec![zero; n],
                b: vec![zero; n],
                fft: vec![zero; fft_len],
                vals: vec![0; n],
            },
            plan,
            regs: HashMap::new(),
            max_roundoff: 0.0,
        }
    }

    pub fn transform_length(&self) -> usize {
        self.plan.n
    }

    /// Test hook, the same as `CpuBackend::corrupt`: corrupt a resident
    /// residue in place (through its canonical form, so the result is exact).
    pub fn corrupt(&mut self, r: Reg, c: Corruption) -> BackendResult<()> {
        let p = self.plan.p;
        let mut limbs = self.export(r)?.limbs().to_vec();
        match c {
            Corruption::Zero => limbs.fill(0),
            Corruption::FlipBit(b) => {
                let b = b % p;
                limbs[(b / 32) as usize] ^= 1 << (b % 32);
            }
        }
        let x = Residue::from_limbs_normalizing(p, limbs)
            .map_err(|_| BackendError::Fault("corrupt produced a noncanonical residue".into()))?;
        self.import(r, &x)
    }

    fn take(&mut self, r: Reg) -> BackendResult<BalancedWords> {
        self.regs
            .remove(&r.0)
            .ok_or(BackendError::InvalidHandle(r.0))
    }

    fn get(&self, r: Reg) -> BackendResult<&BalancedWords> {
        self.regs.get(&r.0).ok_or(BackendError::InvalidHandle(r.0))
    }

    /// `count` squarings of `x`, subtracting `minus` after each. On error the
    /// caller restores the original.
    fn square_loop(&mut self, x: &mut BalancedWords, count: u64, minus: i64) -> BackendResult<()> {
        let Scratch { a, fft, vals, .. } = &mut self.scratch;
        if let Some(pk) = &self.plan.packed {
            for _ in 0..count {
                self.plan.forward_packed(pk, x, a, fft);
                for c in a[..=pk.m].iter_mut() {
                    *c = *c * *c;
                }
                self.plan
                    .finish_packed(pk, a, fft, vals, -minus, &mut self.max_roundoff, x)?;
            }
            return Ok(());
        }
        for _ in 0..count {
            self.plan.forward(x, a, fft);
            for c in a.iter_mut() {
                *c = *c * *c;
            }
            self.plan
                .finish(a, fft, vals, -minus, &mut self.max_roundoff, x)?;
        }
        Ok(())
    }

    /// Shared body of `square` and `ll_step`: the batch commits only if every
    /// step passes the roundoff check.
    fn batch(&mut self, r: Reg, count: u64, minus: i64) -> BackendResult<()> {
        let original = self.take(r)?;
        let mut x = original.clone();
        let result = self.square_loop(&mut x, count, minus);
        self.regs
            .insert(r.0, if result.is_ok() { x } else { original });
        result
    }
}

impl Backend for DwtBackend {
    fn exponent(&self) -> u32 {
        self.plan.p
    }

    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            kind: "dwt".into(),
            build: crate::build_identity().into(),
            plan: {
                let mut name = plan_name(self.plan.n);
                if self.plan.packed.is_some() {
                    name.push_str(" packed");
                }
                if let Some((f, _)) = &self.plan.par {
                    name.push_str(&format!(" threads={}", f.threads));
                }
                name
            },
            device: String::new(),
        }
    }

    fn alloc(&mut self) -> BackendResult<Reg> {
        // Handles come from a process-wide counter and are never reused, so a
        // freed or foreign handle stays invalid.
        let r = new_handle()?;
        self.regs.insert(r.0, BalancedWords(vec![0; self.plan.n]));
        Ok(r)
    }

    fn free(&mut self, r: Reg) -> BackendResult<()> {
        self.take(r).map(drop)
    }

    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        self.get(r)?;
        if x.exponent() != self.plan.p {
            return Err(BackendError::Unsupported(format!(
                "residue exponent {} does not match backend exponent {}",
                x.exponent(),
                self.plan.p
            )));
        }
        let words = self.plan.canonical_to_balanced(x)?;
        self.regs.insert(r.0, words);
        Ok(())
    }

    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        self.plan.balanced_to_canonical(self.get(r)?)
    }

    fn copy(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        let x = self.get(src)?.clone();
        self.get(dst)?;
        self.regs.insert(dst.0, x);
        Ok(())
    }

    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.batch(r, n, 0)
    }

    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.batch(r, n, 2)
    }

    fn mul(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        if dst == src {
            self.get(dst)?;
            return Err(BackendError::Unsupported(
                "mul needs distinct handles; use square".into(),
            ));
        }
        let Scratch { a, b, fft, vals } = &mut self.scratch;
        let (x, y) = (
            self.regs
                .get(&dst.0)
                .ok_or(BackendError::InvalidHandle(dst.0))?,
            self.regs
                .get(&src.0)
                .ok_or(BackendError::InvalidHandle(src.0))?,
        );
        if let Some(pk) = &self.plan.packed {
            self.plan.forward_packed(pk, x, a, fft);
            self.plan.forward_packed(pk, y, b, fft);
            for (u, v) in a[..=pk.m].iter_mut().zip(b[..=pk.m].iter()) {
                *u *= *v;
            }
            let out = self
                .regs
                .get_mut(&dst.0)
                .ok_or(BackendError::InvalidHandle(dst.0))?;
            return self
                .plan
                .finish_packed(pk, a, fft, vals, 0, &mut self.max_roundoff, out);
        }
        self.plan.forward(x, a, fft);
        self.plan.forward(y, b, fft);
        for (u, v) in a.iter_mut().zip(b.iter()) {
            *u *= *v;
        }
        let out = self
            .regs
            .get_mut(&dst.0)
            .ok_or(BackendError::InvalidHandle(dst.0))?;
        self.plan
            .finish(a, fft, vals, 0, &mut self.max_roundoff, out)
    }

    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        Ok(self.export(a)?.limbs() == self.export(b)?.limbs())
    }

    fn sync(&mut self) -> BackendResult<()> {
        Ok(())
    }

    fn diagnostics(&mut self) -> BackendResult<Vec<(String, String)>> {
        Ok(vec![
            ("max_roundoff".into(), format!("{:.6}", self.max_roundoff)),
            ("roundoff_limit".into(), format!("{ROUNDOFF_LIMIT}")),
            ("transform_length".into(), self.plan.n.to_string()),
            (
                "bits_per_word".into(),
                format!("{:.4}", self.plan.p as f64 / self.plan.n as f64),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed real transform gives the same residues as the CPU
    /// reference, for squaring, LL steps, and products, at even lengths of
    /// each radix family.
    #[test]
    fn threaded_transform_matches_the_reference() {
        use crate::backend::cpu::CpuBackend;
        for (p, packed) in [
            (21701u32, false),
            (86243, false),
            (86243, true),
            (216091, true),
        ] {
            let mut d = DwtBackend::new(p).unwrap();
            if packed {
                d.use_packed();
            }
            d.use_threads(3);
            if d.threads() != 3 {
                continue;
            }
            assert!(d.identity().plan.contains(" threads=3"));
            let mut c = CpuBackend::new(p).unwrap();
            let seed = Residue::from_u64(p, 3);
            let (dx, dy, cx, cy) = (
                d.alloc().unwrap(),
                d.alloc().unwrap(),
                c.alloc().unwrap(),
                c.alloc().unwrap(),
            );
            for (be, x, y) in [
                (&mut d as &mut dyn Backend, dx, dy),
                (&mut c as &mut dyn Backend, cx, cy),
            ] {
                be.import(x, &seed).unwrap();
                be.square(x, 20).unwrap();
                be.copy(y, x).unwrap();
                be.mul(x, y).unwrap();
            }
            assert_eq!(d.export(dx).unwrap(), c.export(cx).unwrap(), "p = {p}");
        }
    }

    #[test]
    fn packed_transform_matches_the_reference() {
        use crate::backend::cpu::CpuBackend;
        for p in [127u32, 521, 1279, 4423, 9941, 21701] {
            let mut d = DwtBackend::new(p).unwrap();
            d.use_packed();
            if !d.is_packed() {
                continue;
            }
            assert!(d.identity().plan.ends_with(" packed"));
            let mut c = CpuBackend::new(p).unwrap();
            let seed = Residue::from_u64(p, 3);
            let (dx, dy, cx, cy) = (
                d.alloc().unwrap(),
                d.alloc().unwrap(),
                c.alloc().unwrap(),
                c.alloc().unwrap(),
            );
            for (be, x, y) in [
                (&mut d as &mut dyn Backend, dx, dy),
                (&mut c as &mut dyn Backend, cx, cy),
            ] {
                be.import(x, &seed).unwrap();
                be.square(x, 50).unwrap();
                be.copy(y, x).unwrap();
                be.ll_step(y, 7).unwrap();
                be.mul(x, y).unwrap();
            }
            assert_eq!(d.export(dx).unwrap(), c.export(cx).unwrap(), "p = {p}");
            assert_eq!(d.export(dy).unwrap(), c.export(cy).unwrap(), "p = {p}");
        }
    }

    #[test]
    fn memory_estimate_matches_the_built_backend() {
        let (lo, hi) = supported_exponents();
        for p in [lo - 1, hi + 1] {
            assert!(matches!(
                memory_estimate(p),
                Err(BackendError::Unsupported(_))
            ));
        }
        let mut odd_parts = std::collections::BTreeSet::new();
        for p in [
            2u32, 31, 127, 521, 1279, 4423, 9689, 21701, 44497, 86243, 132049,
        ] {
            let m = memory_estimate(p).unwrap();
            let mut be = DwtBackend::new(p).unwrap();
            let n = be.transform_length();
            odd_parts.insert(n >> n.trailing_zeros());
            assert_eq!(m.plan, be.identity().plan, "p={p}");
            assert!(!m.on_device);
            let r = be.alloc().unwrap();
            assert_eq!(
                m.bytes_per_register,
                4 * be.regs[&r.0].0.len() as u64,
                "p={p}"
            );
            let (sc, pl) = (&be.scratch, &be.plan);
            let c = std::mem::size_of::<C64>();
            let real = c * (sc.a.len() + sc.b.len() + sc.fft.len())
                + 8 * sc.vals.len()
                + 4 * (pl.offsets.len() + pl.widths.len())
                + 8 * (pl.weight.len() + pl.inv_weight.len());
            assert_eq!(m.fixed_bytes, real as u64, "p={p}");
        }
        assert_eq!(odd_parts.into_iter().collect::<Vec<_>>(), [1, 3, 5, 7]);
    }
}
