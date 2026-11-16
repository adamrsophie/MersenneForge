// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! CUDA backend on real hardware: run on a host with an NVIDIA GPU, the CUDA
//! driver, and NVRTC. Without them every test fails with the reason; nothing
//! skips. Every comparison is of complete canonical residues against exact
//! `num-bigint` arithmetic computed here, independently of the backend.
//!
//! Device: ordinal 0, or the value of MERSENNE_FORGE_CUDA_DEVICE.
//! Tests serialize on one lock so that one engine owns the device at a time.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, MutexGuard};

use mersenne_forge::backend::cuda::{model, probe, supported_exponents, CudaBackend, CudaOptions};
use mersenne_forge::backend::{Backend, BackendError, Reg};
use mersenne_forge::residue::{modulus, Residue};
use num_bigint::BigUint;

static DEVICE_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    DEVICE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn ordinal() -> u32 {
    std::env::var("MERSENNE_FORGE_CUDA_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Fail with the reason when no usable CUDA device exists.
fn require_device() {
    let pr = probe();
    assert!(
        pr.driver_loaded,
        "no CUDA driver: this suite needs an NVIDIA GPU with libcuda and NVRTC. Probe notes: {:?}",
        pr.notes
    );
    assert!(
        pr.devices.iter().any(|d| d.id == ordinal()),
        "no CUDA device {}: probe found {:?}; notes {:?}",
        ordinal(),
        pr.devices,
        pr.notes
    );
}

fn open(p: u32) -> CudaBackend {
    require_device();
    CudaBackend::new(
        p,
        &CudaOptions {
            device: Some(ordinal()),
        },
    )
    .unwrap_or_else(|e| panic!("CUDA backend for p={p} failed to open: {e}"))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

fn random(p: u32, rng: &mut Rng) -> BigUint {
    let limbs: Vec<u32> = (0..(p as usize).div_ceil(32)).map(|_| rng.next()).collect();
    BigUint::from_slice(&limbs) % modulus(p)
}

fn edges(p: u32, rng: &mut Rng) -> Vec<BigUint> {
    let n = modulus(p);
    let mut alt = BigUint::from(0u32);
    for b in (0..p).step_by(2) {
        alt.set_bit(u64::from(b), true);
    }
    // Random values first, so that a shortened list still has them.
    vec![
        random(p, rng),
        &n - 1u32,
        BigUint::from(0u32),
        BigUint::from(1u32),
        BigUint::from(2u32),
        &n - 2u32,
        &n >> 1u32,
        BigUint::from(1u32) << (p - 1),
        alt % &n,
        random(p, rng),
    ]
}

/// The value whose balanced words in the CUDA layout all have magnitude
/// 2^(w_k - 1) - 1, the largest digit `import` produces without a carry.
fn maximal_words(p: u32) -> BigUint {
    let pl = model::plan(p).unwrap();
    let (p64, l) = (u64::from(p), u64::from(pl.log_n));
    let mut v = BigUint::from(0u32);
    for k in 0..pl.n {
        // Digit 2^(w - 1) - 1: the low w - 1 bits of word k set.
        let b = model::offset(p64, l, k);
        for t in 0..model::width(p64, l, k) - 1 {
            v.set_bit(b + t, true);
        }
    }
    v % modulus(p)
}

fn load(be: &mut CudaBackend, x: &BigUint) -> Reg {
    let r = be.alloc().unwrap();
    be.import(r, &Residue::from_biguint(be.exponent(), x))
        .unwrap();
    r
}

fn read(be: &mut CudaBackend, r: Reg) -> BigUint {
    be.export(r).unwrap().to_biguint()
}

/// Exponents across the range, including the capacity edge of several lengths.
/// How many values to test at exponent p: `num-bigint` division, the
/// independent oracle, costs about a second per reduction at p = 2e7.
fn budget(p: u32, full: usize) -> usize {
    if p > 1_000_000 {
        full.min(2)
    } else {
        full
    }
}

fn exponents() -> Vec<u32> {
    let mut v = vec![12u32, 13, 127, 1279, 9941, 86243, 1_257_787];
    for log_n in [3u32, 10, 16, 20] {
        v.push(model::max_exponent_for_len(log_n).unwrap());
    }
    v
}

#[test]
fn probe_lists_the_device() {
    let _g = lock();
    require_device();
    let pr = probe();
    assert_eq!(pr.kernel_source_hash.len(), 64);
}

#[test]
fn import_export_round_trip() {
    let _g = lock();
    let mut rng = Rng(1);
    for p in exponents() {
        let mut be = open(p);
        for x in edges(p, &mut rng) {
            let r = load(&mut be, &x);
            assert_eq!(read(&mut be, r), x, "p={p}");
            be.free(r).unwrap();
        }
    }
}

#[test]
fn single_and_batched_squarings() {
    let _g = lock();
    let mut rng = Rng(2);
    for p in exponents() {
        let n = modulus(p);
        let mut be = open(p);
        for x in edges(p, &mut rng).into_iter().take(budget(p, 10)) {
            let r = load(&mut be, &x);
            be.square(r, 0).unwrap();
            assert_eq!(read(&mut be, r), x, "n = 0 is a no-op, p={p}");
            let mut want = x.clone();
            be.square(r, 1).unwrap();
            want = (&want * &want) % &n;
            assert_eq!(read(&mut be, r), want, "one squaring p={p}");
            be.square(r, 9).unwrap();
            for _ in 0..9 {
                want = (&want * &want) % &n;
            }
            assert_eq!(read(&mut be, r), want, "nine batched squarings p={p}");
            be.free(r).unwrap();
        }
    }
}

#[test]
fn ll_steps_match_bigint() {
    let _g = lock();
    let mut rng = Rng(3);
    for p in exponents() {
        let n = modulus(p);
        let mut be = open(p);
        for x in edges(p, &mut rng).into_iter().take(budget(p, 4)) {
            let r = load(&mut be, &x);
            be.ll_step(r, 7).unwrap();
            let mut want = x.clone();
            for _ in 0..7 {
                want = (&want * &want + &n - 2u32) % &n;
            }
            assert_eq!(read(&mut be, r), want, "p={p}");
        }
    }
}

#[test]
fn multiplication_copy_and_equal() {
    let _g = lock();
    let mut rng = Rng(4);
    for p in exponents() {
        let n = modulus(p);
        let mut be = open(p);
        let vals = edges(p, &mut rng);
        // Dense second operands (a random value distinct from the first x,
        // and one whose words all have maximal magnitude) are chosen before
        // the budget cut, so large exponents still multiply dense by dense.
        // Small constants, including N - 1 = -1, are a separate case.
        let dense = [vals[9].clone(), maximal_words(p)];
        let small = [BigUint::from(1u32), BigUint::from(2u32), &n - 1u32];
        let mut pairs: Vec<(&BigUint, &BigUint)> = vec![(&dense[1], &dense[1])];
        for x in vals.iter().take(budget(p, 5)) {
            pairs.extend(dense.iter().map(|y| (x, y)));
        }
        pairs.extend(small.iter().map(|y| (&vals[0], y)));
        for (x, y) in pairs {
            let a = load(&mut be, x);
            let b = load(&mut be, y);
            be.mul(a, b).unwrap();
            assert_eq!(read(&mut be, a), (x * y) % &n, "mul p={p}");
            assert_eq!(read(&mut be, b), *y, "mul leaves src unchanged p={p}");
            be.copy(b, a).unwrap();
            assert!(be.equal(a, b).unwrap());
            be.copy(a, a).unwrap();
            assert_eq!(read(&mut be, a), (x * y) % &n);
            be.square(b, 1).unwrap();
            let prod = (x * y) % &n;
            assert_eq!(be.equal(a, b).unwrap(), (&prod * &prod) % &n == prod);
            be.free(a).unwrap();
            be.free(b).unwrap();
        }
        be.sync().unwrap();
    }
}

fn ll_oracle(p: u32) -> BigUint {
    let n = modulus(p);
    let mut s = BigUint::from(4u32);
    for _ in 0..p - 2 {
        s = (&s * &s + &n - 2u32) % &n;
    }
    s
}

#[test]
fn complete_ll_runs() {
    let _g = lock();
    let primes = [
        13u32, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689,
        9941, 11213, 19937, 21701, 23209, 44497,
    ];
    let composites = [23u32, 29, 101, 131, 523, 1277, 4421, 9949, 23203];
    for &p in primes.iter().chain(composites.iter()) {
        let mut be = open(p);
        let r = load(&mut be, &BigUint::from(4u32));
        be.ll_step(r, u64::from(p - 2)).unwrap();
        be.sync().unwrap();
        let got = read(&mut be, r);
        if primes.contains(&p) {
            assert_eq!(got, BigUint::from(0u32), "M{p} is prime");
        } else if p < 5000 {
            assert_eq!(got, ll_oracle(p), "M{p} final residue");
        } else {
            assert_ne!(got, BigUint::from(0u32), "M{p} is composite");
        }
    }
}

#[test]
fn error_paths() {
    let _g = lock();
    let (lo, hi) = supported_exponents();
    require_device();
    for p in [lo - 1, hi + 1] {
        assert!(matches!(
            CudaBackend::new(p, &CudaOptions { device: None }),
            Err(BackendError::Unsupported(_))
        ));
    }
    assert!(matches!(
        CudaBackend::new(127, &CudaOptions { device: Some(9999) }),
        Err(BackendError::Unsupported(_))
    ));

    let mut be = open(127);
    let id = be.identity();
    assert_eq!(id.kind, "cuda");
    assert_eq!(
        id.plan,
        format!("ntt-gf-m61sq n={}", model::plan(127).unwrap().n)
    );
    assert!(
        id.device.contains(&format!("(CUDA {}, cc ", ordinal())),
        "{}",
        id.device
    );
    assert!(
        id.build.contains(&probe().kernel_source_hash)
            && id.build.contains("--gpu-architecture=compute_")
    );

    let a = load(&mut be, &BigUint::from(5u32));
    let b = be.alloc().unwrap();
    assert!(matches!(be.mul(a, a), Err(BackendError::Unsupported(_))));
    assert!(matches!(
        be.import(a, &Residue::from_u64(61, 3)),
        Err(BackendError::Unsupported(_))
    ));
    be.free(b).unwrap();
    // A handle of another live instance on the same device is foreign here.
    let mut other = open(127);
    let foreign = other.alloc().unwrap();
    let bad = [b, foreign, Reg(0), Reg(u32::MAX)];
    for r in bad {
        assert_eq!(be.free(r), Err(BackendError::InvalidHandle(r.0)));
        assert_eq!(be.export(r).err(), Some(BackendError::InvalidHandle(r.0)));
        assert_eq!(
            be.import(r, &Residue::from_u64(127, 1)),
            Err(BackendError::InvalidHandle(r.0))
        );
        assert_eq!(be.square(r, 1), Err(BackendError::InvalidHandle(r.0)));
        assert_eq!(be.ll_step(r, 1), Err(BackendError::InvalidHandle(r.0)));
        assert_eq!(be.mul(a, r), Err(BackendError::InvalidHandle(r.0)));
        assert_eq!(be.mul(r, a), Err(BackendError::InvalidHandle(r.0)));
        assert_eq!(be.copy(a, r), Err(BackendError::InvalidHandle(r.0)));
        assert_eq!(be.copy(r, a), Err(BackendError::InvalidHandle(r.0)));
        assert_eq!(be.equal(a, r).err(), Some(BackendError::InvalidHandle(r.0)));
    }
    // The valid register is unaffected by the failed calls.
    assert_eq!(read(&mut be, a), BigUint::from(5u32));
    be.sync().unwrap();
}
