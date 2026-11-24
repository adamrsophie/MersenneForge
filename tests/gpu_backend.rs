// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! GPU backend tests on real hardware. Every expected value is computed here
//! with plain `num-bigint` arithmetic modulo 2^p - 1. These tests fail (they
//! never skip) when no hardware adapter exists.

#![cfg(feature = "gpu")]

use std::time::Instant;

use mersenne_forge::backend::gpu::{self, GpuBackend, GpuOptions};
use mersenne_forge::backend::{Backend, BackendError, Reg};
use mersenne_forge::residue::Residue;
use num_bigint::BigUint;

fn gpu(p: u32) -> GpuBackend {
    GpuBackend::new(p, &GpuOptions::default())
        .unwrap_or_else(|e| panic!("GPU backend for p={p} must be available: {e}"))
}

fn modulus(p: u32) -> BigUint {
    (BigUint::from(1u8) << p) - 1u8
}

fn res(p: u32, v: &BigUint) -> Residue {
    Residue::from_biguint(p, v)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn big(&mut self, p: u32) -> BigUint {
        let bytes: Vec<u8> = (0..p.div_ceil(8) + 8).map(|_| self.next() as u8).collect();
        BigUint::from_bytes_le(&bytes) % modulus(p)
    }
}

/// Transform length from the documented plan string "ntt-gf-m31sq n=<len>"
/// (or "ntt-gf-m61sq n=<len>" for the wide field).
fn plan_len(g: &GpuBackend) -> u32 {
    let id = g.identity().plan;
    id.strip_prefix("ntt-gf-m31sq n=")
        .or_else(|| id.strip_prefix("ntt-gf-m61sq n="))
        .unwrap()
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

/// Largest exponent of the 31-bit field (length 2^24).
fn narrow_top() -> u32 {
    gpu::model::supported_range().1
}

/// Values whose balanced words (layout b_k = ceil(p k / n), documented in
/// docs/GPU.md) all have the largest magnitude 2^(w_k - 1): either every
/// digit is 2^(w-1) - 1 plus an incoming carry (all words negative), or every
/// digit is 2^(w-1) - 1 (all words positive).
fn extreme_values(p: u32, n: u32) -> Vec<BigUint> {
    let pos = |k: u64| (u64::from(p) * k).div_ceil(u64::from(n));
    let mut neg = vec![0u32; p.div_ceil(32) as usize + 1];
    let mut pos_max = neg.clone();
    let put = |limbs: &mut [u32], at: u64, d: u64| {
        let (i, o) = ((at / 32) as usize, at % 32);
        let v = d << o;
        limbs[i] |= v as u32;
        limbs[i + 1] |= (v >> 32) as u32;
    };
    for k in 0..u64::from(n) {
        let (b, w) = (pos(k), pos(k + 1) - pos(k));
        let half = 1u64 << (w - 1);
        put(&mut neg, b, if k == 0 { half } else { half - 1 });
        put(&mut pos_max, b, half - 1);
    }
    vec![BigUint::from_slice(&neg), BigUint::from_slice(&pos_max)]
}

fn special_values(p: u32) -> Vec<BigUint> {
    let m = modulus(p);
    let one = BigUint::from(1u8);
    let mut v = vec![
        BigUint::from(0u8),
        one.clone(),
        BigUint::from(2u8),
        &m - 1u8,
        &m - 2u8,
        (&one << (p - 1)) - 1u8, // long carry chain of ones
        &one << (p - 1),         // top bit only
        (&one << (p / 2)) - 1u8, // ones to the middle
        (&m >> 1) ^ (&m >> 2),   // alternating bits
    ];
    // All-ones limbs below the top limb.
    let below = (p - 1) / 32 * 32;
    if below > 0 {
        v.push((&one << below) - 1u8);
    }
    v
}

#[test]
fn probe_lists_an_adapter_and_hashes_the_kernels() {
    let pr = gpu::probe();
    assert!(!pr.devices.is_empty(), "no adapter enumerated");
    assert_eq!(pr.kernel_source_hash.len(), 64);
    assert!(pr.transform.starts_with("NTT GF(M31^2)"));
    let g = gpu(521);
    let id = g.identity();
    assert_eq!(id.kind, "gpu");
    assert_eq!(id.plan, "ntt-gf-m31sq n=64");
    assert!(id.build.contains(&pr.kernel_source_hash));
    assert!(id.build.contains(env!("CARGO_PKG_VERSION")));
    assert!(!id.device.is_empty());
    eprintln!("adapter: {}", id.device);
}

/// Exponents spanning many lengths, including word-unaligned ones, the
/// largest exponent of several lengths (worst case for the bound), lengths
/// 2^16 and 2^18, the largest exponent of length 2^22, and the largest
/// supported exponent (length 2^24).
fn exponents() -> Vec<u32> {
    let mut ps = vec![8, 11, 31, 61, 89, 127, 521, 1279, 4423, 9941, 44497];
    for log_n in [2, 6, 10, 14, 16, 22] {
        ps.push(gpu::model::edge(1 << log_n).unwrap());
    }
    ps.push(1_000_003);
    ps.push(narrow_top());
    ps
}

#[test]
fn import_export_round_trips() {
    let mut rng = Rng(1);
    for p in exponents() {
        let mut g = gpu(p);
        let r = g.alloc().unwrap();
        let mut vals = special_values(p);
        vals.extend(extreme_values(p, plan_len(&g)));
        vals.extend((0..3).map(|_| rng.big(p)));
        for v in vals {
            let x = res(p, &v);
            g.import(r, &x).unwrap();
            assert_eq!(g.export(r).unwrap(), x, "p={p}");
        }
    }
}

#[test]
fn single_squarings_match_bigint() {
    let mut rng = Rng(2);
    for p in exponents() {
        let m = modulus(p);
        let mut g = gpu(p);
        let r = g.alloc().unwrap();
        let mut vals = special_values(p);
        vals.extend(extreme_values(p, plan_len(&g)));
        vals.push(rng.big(p));
        for v in vals {
            g.import(r, &res(p, &v)).unwrap();
            g.square(r, 1).unwrap();
            let want = res(p, &(&v * &v % &m));
            assert_eq!(g.export(r).unwrap(), want, "p={p} n={}", plan_len(&g));
        }
    }
}

#[test]
fn batched_squarings_in_one_call_and_split() {
    let mut rng = Rng(3);
    for p in [127, 4423, 44497] {
        let m = modulus(p);
        let mut g = gpu(p);
        let (a, b) = (g.alloc().unwrap(), g.alloc().unwrap());
        for k in [1u64, 2, 7, 100, 1000] {
            let v = rng.big(p);
            let mut want = v.clone();
            for _ in 0..k {
                want = &want * &want % &m;
            }
            g.import(a, &res(p, &v)).unwrap();
            g.import(b, &res(p, &v)).unwrap();
            g.square(a, k).unwrap();
            let mut done = 0;
            for part in [k / 3, 1, k] {
                let step = part.min(k - done);
                g.square(b, step).unwrap();
                done += step;
            }
            g.square(b, 0).unwrap();
            assert_eq!(g.export(a).unwrap(), res(p, &want), "p={p} k={k}");
            assert_eq!(g.export(b).unwrap(), res(p, &want), "split p={p} k={k}");
        }
    }
}

#[test]
fn repeated_squaring_from_extreme_words() {
    // The largest supported exponent (n = 2^24, 15 carry passes) gets a
    // shorter batch because each bigint squaring there is slow.
    for (p, k) in [
        (gpu::model::edge(1 << 10).unwrap(), 20),
        (gpu::model::edge(1 << 16).unwrap(), 20),
        (gpu::model::edge(1 << 22).unwrap(), 5),
        (narrow_top(), 3),
    ] {
        let m = modulus(p);
        let mut g = gpu(p);
        let r = g.alloc().unwrap();
        for v in extreme_values(p, plan_len(&g)) {
            let mut want = v.clone();
            for _ in 0..k {
                want = &want * &want % &m;
            }
            g.import(r, &res(p, &v)).unwrap();
            g.square(r, k).unwrap();
            assert_eq!(g.export(r).unwrap(), res(p, &want), "p={p}");
        }
    }
}

#[test]
fn mul_copy_equal_and_ll_step_match_bigint() {
    let mut rng = Rng(4);
    for p in [
        89,
        1279,
        9941,
        gpu::model::edge(1 << 14).unwrap(),
        gpu::model::edge(1 << 16).unwrap(),
    ] {
        let m = modulus(p);
        let mut g = gpu(p);
        let (a, b, c) = (g.alloc().unwrap(), g.alloc().unwrap(), g.alloc().unwrap());
        let mut vals = vec![rng.big(p), rng.big(p), &m - 1u8];
        vals.extend(extreme_values(p, plan_len(&g)));
        for (x, y) in vals.iter().zip(vals.iter().rev()) {
            g.import(a, &res(p, x)).unwrap();
            g.import(b, &res(p, y)).unwrap();
            g.mul(a, b).unwrap();
            assert_eq!(g.export(a).unwrap(), res(p, &(x * y % &m)), "mul p={p}");
            assert_eq!(g.export(b).unwrap(), res(p, y), "mul source unchanged");
        }
        let x = rng.big(p);
        g.import(a, &res(p, &x)).unwrap();
        g.copy(c, a).unwrap();
        g.copy(c, c).unwrap();
        assert!(g.equal(a, c).unwrap());
        g.square(c, 1).unwrap();
        assert!(!g.equal(a, c).unwrap());
        assert_eq!(g.export(c).unwrap(), res(p, &(&x * &x % &m)));

        let mut s = x.clone();
        for _ in 0..37 {
            s = (&s * &s + &m - 2u8) % &m;
        }
        g.ll_step(a, 37).unwrap();
        assert_eq!(g.export(a).unwrap(), res(p, &s), "ll p={p}");
        // Zero and one: 0^2 - 2 = N - 2, 1^2 - 2 = N - 1.
        g.import(a, &Residue::zero(p)).unwrap();
        g.ll_step(a, 1).unwrap();
        assert_eq!(g.export(a).unwrap(), res(p, &(&m - 2u8)));
        g.import(a, &Residue::from_u64(p, 1)).unwrap();
        g.ll_step(a, 1).unwrap();
        assert_eq!(g.export(a).unwrap(), res(p, &(&m - 1u8)));
        g.sync().unwrap();
    }
}

fn bigint_ll(p: u32) -> BigUint {
    let m = modulus(p);
    let mut s = BigUint::from(4u8);
    for _ in 0..p - 2 {
        s = (&s * &s + &m - 2u8) % &m;
    }
    s
}

fn gpu_ll(p: u32) -> Residue {
    let mut g = gpu(p);
    let r = g.alloc().unwrap();
    g.import(r, &Residue::from_u64(p, 4)).unwrap();
    g.ll_step(r, u64::from(p - 2)).unwrap();
    g.export(r).unwrap()
}

#[test]
fn complete_ll_runs_for_mersenne_primes_end_at_zero() {
    for p in [521, 607, 1279, 2203, 2281, 4423, 9689] {
        let t = Instant::now();
        assert!(gpu_ll(p).is_zero(), "M{p} is prime");
        eprintln!("LL p={p}: {:?}", t.elapsed());
    }
}

#[test]
fn complete_ll_runs_for_composites_match_bigint() {
    for p in [1277, 4421] {
        let got = gpu_ll(p);
        assert!(!got.is_zero());
        assert_eq!(got, res(p, &bigint_ll(p)), "p={p}");
    }
}

/// Run once with: cargo test --features gpu --test gpu_backend -- --ignored
#[test]
#[ignore]
fn complete_ll_run_p110503() {
    let t = Instant::now();
    assert!(gpu_ll(110503).is_zero(), "M110503 is prime");
    eprintln!("LL p=110503: {:?}", t.elapsed());
}

#[test]
fn error_paths() {
    let (lo, hi) = gpu::supported_exponents();
    for p in [2, 3, lo - 1, hi + 1] {
        match GpuBackend::new(p, &GpuOptions::default()) {
            Err(BackendError::Unsupported(_)) => {}
            Err(e) => panic!("p={p}: wrong error {e}"),
            Ok(_) => panic!("p={p} must be unsupported"),
        }
    }
    assert!(matches!(
        GpuBackend::new(
            521,
            &GpuOptions {
                device: Some(9999),
                ..GpuOptions::default()
            }
        ),
        Err(BackendError::Unsupported(_))
    ));

    let mut g = gpu(521);
    let mut other = gpu(521);
    let r = g.alloc().unwrap();
    let foreign = other.alloc().unwrap();
    g.free(r).unwrap();
    for bad in [r, foreign, Reg(0), Reg(u32::MAX)] {
        let inv = Err(BackendError::InvalidHandle(bad.0));
        assert_eq!(g.export(bad), inv);
        assert_eq!(g.square(bad, 1), inv.clone().map(|_: Residue| ()));
        assert_eq!(g.free(bad), inv.clone().map(|_: Residue| ()));
    }
    let a = g.alloc().unwrap();
    assert_eq!(g.copy(a, r), Err(BackendError::InvalidHandle(r.0)));
    assert_eq!(g.mul(r, a), Err(BackendError::InvalidHandle(r.0)));
    assert!(matches!(
        g.import(a, &Residue::from_u64(523, 5)),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(g.mul(a, a), Err(BackendError::Unsupported(_))));
    // The backend still works after rejected calls.
    g.import(a, &Residue::from_u64(521, 7)).unwrap();
    g.square(a, 1).unwrap();
    assert_eq!(g.export(a).unwrap(), Residue::from_u64(521, 49));
    g.sync().unwrap();
}

fn wide_gpu(p: u32) -> GpuBackend {
    GpuBackend::new(
        p,
        &GpuOptions {
            wide: true,
            ..GpuOptions::default()
        },
    )
    .expect("wide GPU backend")
}

/// The GF((2^61 - 1)^2) kernels: forced on small exponents, and chosen on
/// their own just past the end of the 31-bit field.
#[test]
fn wide_field_matches_bigint() {
    let mut rng = Rng(61);
    assert!(gpu::supported_exponents().1 > 140_000_000);
    let past = narrow_top() + 1;
    for (p, k) in [
        (89u32, 20u64),
        (1279, 20),
        (4423, 10),
        (86243, 5),
        (past, 2),
    ] {
        let m = modulus(p);
        let mut g = if p == past { gpu(p) } else { wide_gpu(p) };
        assert!(g.identity().plan.starts_with("ntt-gf-m61sq"), "p={p}");
        let n = plan_len(&g);
        let (a, b) = (g.alloc().unwrap(), g.alloc().unwrap());
        let mut vals = special_values(p);
        vals.extend(extreme_values(p, n));
        vals.push(rng.big(p));
        for v in &vals {
            let x = res(p, v);
            g.import(a, &x).unwrap();
            assert_eq!(g.export(a).unwrap(), x, "round trip p={p}");
        }
        for v in extreme_values(p, n).into_iter().chain([rng.big(p)]) {
            let mut want = v.clone();
            for _ in 0..k {
                want = &want * &want % &m;
            }
            g.import(a, &res(p, &v)).unwrap();
            g.square(a, k).unwrap();
            assert_eq!(g.export(a).unwrap(), res(p, &want), "square p={p} n={n}");
        }
        let (x, y) = (rng.big(p), rng.big(p));
        g.import(a, &res(p, &x)).unwrap();
        g.import(b, &res(p, &y)).unwrap();
        g.mul(a, b).unwrap();
        assert_eq!(g.export(a).unwrap(), res(p, &(&x * &y % &m)), "mul p={p}");
    }
}
