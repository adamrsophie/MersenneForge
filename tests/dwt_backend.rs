// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The f64 IBDWT backend against exact `num-bigint` arithmetic computed here,
//! independently of the backend. Every comparison is of complete canonical
//! residues.

use mersenne_forge::backend::cpu::Corruption;
use mersenne_forge::backend::dwt::{max_word_bits, supported_exponents, DwtBackend, MAX_LENGTH};
use mersenne_forge::backend::{Backend, BackendError, Reg};
use mersenne_forge::residue::Residue;
use num_bigint::BigUint;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

fn modulus(p: u32) -> BigUint {
    (BigUint::from(1u32) << p) - 1u32
}

/// x mod N by folding: 2^p = 1 mod N. Checked against `%` in
/// `bound_edge_plans_match_bigint`; used where `%` on million-bit values is slow.
fn fold(x: BigUint, p: u32) -> BigUint {
    let n = modulus(p);
    let mut x = x;
    while x > n {
        x = (&x & &n) + (&x >> p);
    }
    if x == n {
        BigUint::from(0u32)
    } else {
        x
    }
}

fn to_big(x: &Residue) -> BigUint {
    BigUint::from_slice(x.limbs())
}

fn to_residue(p: u32, x: &BigUint) -> Residue {
    let mut limbs = (x % modulus(p)).to_u32_digits();
    limbs.resize((p as usize).div_ceil(32), 0);
    Residue::from_limbs(p, limbs).expect("canonical value")
}

fn random(p: u32, rng: &mut Rng) -> BigUint {
    let limbs: Vec<u32> = (0..(p as usize).div_ceil(32)).map(|_| rng.next()).collect();
    BigUint::from_slice(&limbs) % modulus(p)
}

/// Values at the edges: 0, 1, 2, N-1, N-2, powers of two around limb and
/// word boundaries, long carry chains, and random values.
fn edge_values(p: u32, rng: &mut Rng) -> Vec<BigUint> {
    let n = modulus(p);
    let one = BigUint::from(1u32);
    let mut v = vec![
        BigUint::from(0u32),
        one.clone(),
        BigUint::from(2u32),
        &n - 1u32,
        &n - 2u32,
    ];
    for k in [1u32, 31, 32, 33, 63, 64, p / 2, p - 1] {
        if k < p {
            let x = &one << k;
            v.push(&x % &n);
            v.push((&x - 1u32) % &n);
            v.push((&x + 1u32) % &n);
        }
    }
    // Alternating runs of ones and zeros: long carry and borrow chains.
    let mut alt = BigUint::from(0u32);
    for i in (0..p).step_by(2) {
        alt |= &one << i;
    }
    v.push(&alt % &n);
    v.push((&n - &alt) % &n);
    for _ in 0..4 {
        v.push(random(p, rng));
    }
    v
}

const EXPONENTS: &[u32] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 32, 33, 61, 64, 65, 89, 107, 127, 521, 607, 1279, 2203,
    2281, 4423, 9941,
];

fn load(be: &mut DwtBackend, x: &BigUint) -> Reg {
    let r = be.alloc().unwrap();
    be.import(r, &to_residue(be.exponent(), x)).unwrap();
    r
}

fn read(be: &mut DwtBackend, r: Reg) -> BigUint {
    let x = be.export(r).unwrap();
    assert_eq!(x.exponent(), be.exponent());
    assert_eq!(x.limbs().len(), (be.exponent() as usize).div_ceil(32));
    to_big(&x)
}

#[test]
fn import_export_round_trip() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for &p in EXPONENTS {
        let mut be = DwtBackend::new(p).unwrap();
        for x in edge_values(p, &mut rng) {
            let r = load(&mut be, &x);
            assert_eq!(read(&mut be, r), x, "p = {p}");
            be.free(r).unwrap();
        }
    }
}

#[test]
fn squarings_match_bigint() {
    let mut rng = Rng(0x0bad_cafe_f00d_0001);
    for &p in EXPONENTS {
        let n = modulus(p);
        let mut be = DwtBackend::new(p).unwrap();
        for x in edge_values(p, &mut rng) {
            for count in [1u64, 3, 17] {
                let r = load(&mut be, &x);
                be.square(r, count).unwrap();
                let expect = x.modpow(&(BigUint::from(1u32) << count), &n);
                assert_eq!(read(&mut be, r), expect, "p = {p}, count = {count}");
                be.free(r).unwrap();
            }
        }
    }
}

#[test]
fn square_zero_count_is_noop() {
    let mut be = DwtBackend::new(127).unwrap();
    let x = BigUint::from(123_456_789u32);
    let r = load(&mut be, &x);
    be.square(r, 0).unwrap();
    be.ll_step(r, 0).unwrap();
    assert_eq!(read(&mut be, r), x);
}

#[test]
fn multiplication_matches_bigint() {
    let mut rng = Rng(0x5eed_0000_0000_0002);
    for &p in EXPONENTS {
        let n = modulus(p);
        let mut be = DwtBackend::new(p).unwrap();
        let vals = edge_values(p, &mut rng);
        for (i, x) in vals.iter().enumerate() {
            let y = &vals[(i * 7 + 3) % vals.len()];
            let (a, b) = (load(&mut be, x), load(&mut be, y));
            be.mul(a, b).unwrap();
            assert_eq!(read(&mut be, a), (x * y) % &n, "p = {p}");
            assert_eq!(read(&mut be, b), *y, "mul must not change src");
            be.free(a).unwrap();
            be.free(b).unwrap();
        }
    }
}

fn ll_oracle(s: &BigUint, n: &BigUint) -> BigUint {
    (s * s + n - 2u32) % n
}

#[test]
fn ll_steps_match_bigint() {
    let mut rng = Rng(0x5eed_0000_0000_0003);
    for &p in EXPONENTS {
        let n = modulus(p);
        let mut be = DwtBackend::new(p).unwrap();
        for x in edge_values(p, &mut rng) {
            let r = load(&mut be, &x);
            be.ll_step(r, 5).unwrap();
            let mut s = x.clone();
            for _ in 0..5 {
                s = ll_oracle(&s, &n);
            }
            assert_eq!(read(&mut be, r), s, "p = {p}");
            be.free(r).unwrap();
        }
    }
}

/// Lengths other than the default, including arbitrary (non-smooth) ones and
/// n = p, where every word is one bit wide.
#[test]
fn explicit_transform_lengths_match_bigint() {
    let mut rng = Rng(0x5eed_0000_0000_0004);
    let p = 1279;
    let n = modulus(p);
    for len in [64usize, 77, 80, 96, 100, 128, 255, 640, 1279] {
        let mut be = DwtBackend::with_transform_length(p, len).unwrap();
        assert_eq!(be.transform_length(), len);
        for x in edge_values(p, &mut rng) {
            let y = random(p, &mut rng);
            let (a, b) = (load(&mut be, &x), load(&mut be, &y));
            assert_eq!(read(&mut be, a), x, "round trip, n = {len}");
            be.square(a, 4).unwrap();
            let sq = x.modpow(&BigUint::from(16u32), &n);
            assert_eq!(read(&mut be, a), sq, "square, n = {len}");
            be.mul(a, b).unwrap();
            assert_eq!(read(&mut be, a), (&sq * &y) % &n, "mul, n = {len}");
            be.ll_step(a, 2).unwrap();
            let s = ll_oracle(&ll_oracle(&((&sq * &y) % &n), &n), &n);
            assert_eq!(read(&mut be, a), s, "ll, n = {len}");
        }
    }
}

/// Complete Lucas-Lehmer: s_0 = 4, p - 2 steps; zero iff M_p is prime.
fn ll_residue(p: u32) -> BigUint {
    let mut be = DwtBackend::new(p).unwrap();
    let r = load(&mut be, &BigUint::from(4u32));
    be.ll_step(r, (p - 2) as u64).unwrap();
    read(&mut be, r)
}

#[test]
fn ll_known_mersenne_primes() {
    for p in [
        3u32, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423,
        9689,
    ] {
        assert_eq!(ll_residue(p), BigUint::from(0u32), "M{p} is prime");
    }
}

#[test]
fn ll_composites_match_bigint_residue() {
    for p in [
        11u32, 23, 29, 37, 41, 43, 47, 53, 59, 67, 71, 73, 79, 83, 97, 101, 103, 109, 113, 131,
        1277, 2207, 4421,
    ] {
        let n = modulus(p);
        let mut s = BigUint::from(4u32);
        for _ in 0..p - 2 {
            s = ll_oracle(&s, &n);
        }
        assert_ne!(s, BigUint::from(0u32));
        assert_eq!(ll_residue(p), s, "M{p} final LL residue");
    }
}

#[test]
#[ignore = "long: run with cargo test --release -- --ignored"]
fn ll_large_mersenne_primes() {
    for p in [44497u32, 86243] {
        assert_eq!(ll_residue(p), BigUint::from(0u32), "M{p} is prime");
    }
}

/// A transform far too short for p: the roundoff check must report a fault
/// and leave the residue as it was before the call.
#[test]
fn too_short_transform_is_a_fault_not_a_wrong_residue() {
    let mut rng = Rng(0x5eed_0000_0000_0005);
    for (p, len) in [(1279u32, 48usize), (4423, 160), (9941, 400)] {
        let mut be = DwtBackend::with_transform_length(p, len).unwrap();
        let x = random(p, &mut rng);
        let y = random(p, &mut rng);
        let (a, b) = (load(&mut be, &x), load(&mut be, &y));
        for result in [be.square(a, 10), be.ll_step(a, 10), be.mul(a, b)] {
            match result {
                Err(BackendError::Fault(m)) => assert!(m.contains("roundoff"), "{m}"),
                other => panic!("p = {p}, n = {len}: expected a roundoff fault, got {other:?}"),
            }
            assert_eq!(read(&mut be, a), x, "state must be unchanged after a fault");
        }
    }
}

/// Transforms forced beyond the bits-per-word bound: every result is either a
/// fault or exactly right. This exercises the band where outputs approach
/// 2^53 and the distance from an integer alone would not reveal an error.
#[test]
fn forced_short_transforms_fault_or_agree() {
    let mut rng = Rng(0x5eed_0000_0000_0009);
    let (mut faults, mut exact) = (0u32, 0u32);
    for len in [1usize, 2, 3, 5, 6, 12, 48] {
        for w in max_word_bits(len) + 1..=30 {
            let p = len as u32 * w - len as u32 / 2;
            let Ok(mut be) = DwtBackend::with_transform_length(p, len) else {
                continue;
            };
            let n = modulus(p);
            for _ in 0..20 {
                let (x, y) = (random(p, &mut rng), random(p, &mut rng));
                let (a, b) = (load(&mut be, &x), load(&mut be, &y));
                match be.square(a, 1) {
                    Err(BackendError::Fault(_)) => {
                        faults += 1;
                        assert_eq!(read(&mut be, a), x);
                    }
                    Ok(()) => {
                        exact += 1;
                        assert_eq!(read(&mut be, a), (&x * &x) % &n, "p = {p}, n = {len}");
                    }
                    Err(e) => panic!("unexpected {e:?}"),
                }
                be.import(a, &to_residue(p, &x)).unwrap();
                match be.mul(a, b) {
                    Err(BackendError::Fault(_)) => faults += 1,
                    Ok(()) => {
                        exact += 1;
                        assert_eq!(read(&mut be, a), (&x * &y) % &n, "mul p = {p}, n = {len}");
                    }
                    Err(e) => panic!("unexpected {e:?}"),
                }
                be.free(a).unwrap();
                be.free(b).unwrap();
            }
        }
    }
    assert!(faults > 0 && exact > 0, "faults {faults}, exact {exact}");
}

/// Exponents at the bits-per-word bound, p = n * max_word_bits(n), for
/// lengths of every candidate shape up to 2^16 words.
#[test]
fn bound_edge_plans_match_bigint() {
    let mut rng = Rng(0x5eed_0000_0000_000a);
    for p in [89u32, 1279, 9941] {
        let (x, y) = (random(p, &mut rng), random(p, &mut rng));
        assert_eq!(fold(&x * &y, p), (&x * &y) % modulus(p), "fold oracle");
    }
    for len in [
        3usize, 5, 7, 12, 40, 56, 384, 1024, 5120, 7168, 24576, 65536,
    ] {
        let p = len as u32 * max_word_bits(len);
        let n = modulus(p);
        let mut be = DwtBackend::with_transform_length(p, len).unwrap();
        let (x, y) = (random(p, &mut rng), random(p, &mut rng));
        let (a, b) = (load(&mut be, &x), load(&mut be, &y));
        be.square(a, 3).unwrap();
        let mut sq = x.clone();
        for _ in 0..3 {
            sq = fold(&sq * &sq, p);
        }
        assert_eq!(read(&mut be, a), sq, "square, n = {len}");
        be.mul(a, b).unwrap();
        assert_eq!(read(&mut be, a), fold(&sq * &y, p), "mul, n = {len}");
        let top = load(&mut be, &(&n - 1u32));
        be.ll_step(top, 1).unwrap();
        assert_eq!(
            read(&mut be, top),
            (&n - 1u32),
            "(N-1)^2 - 2 = N - 1, n = {len}"
        );
    }
}

/// The largest supported exponent on the largest transform length.
#[test]
#[ignore = "long and memory heavy: run with cargo test --release -- --ignored"]
fn top_of_range_matches_bigint() {
    let mut rng = Rng(0x5eed_0000_0000_000b);
    let p = supported_exponents().1;
    let mut be = DwtBackend::new(p).unwrap();
    assert_eq!(be.transform_length(), MAX_LENGTH);
    let (x, y) = (random(p, &mut rng), random(p, &mut rng));
    let (a, b) = (load(&mut be, &x), load(&mut be, &y));
    be.square(a, 1).unwrap();
    let sq = fold(&x * &x, p);
    assert_eq!(read(&mut be, a), sq);
    be.mul(a, b).unwrap();
    assert_eq!(read(&mut be, a), fold(&sq * &y, p));
}

/// Default plans keep the measured roundoff far below the limit.
#[test]
fn default_plans_have_low_roundoff() {
    let mut rng = Rng(0x5eed_0000_0000_0006);
    for p in [127u32, 1279, 9941, 23209, 110503] {
        let mut be = DwtBackend::new(p).unwrap();
        let r = load(&mut be, &random(p, &mut rng));
        be.square(r, 20).unwrap();
        let d = be.diagnostics().unwrap();
        let max: f64 = d
            .iter()
            .find(|(k, _)| k == "max_roundoff")
            .unwrap()
            .1
            .parse()
            .unwrap();
        assert!(max < 0.05, "p = {p}: max roundoff {max}");
    }
}

#[test]
fn equal_compares_values() {
    let mut rng = Rng(0x5eed_0000_0000_0007);
    let p = 607;
    let mut be = DwtBackend::new(p).unwrap();
    let x = random(p, &mut rng);
    let a = load(&mut be, &x);
    // b = x * 1 through the transform: same value, possibly other balanced words.
    let b = load(&mut be, &x);
    let one = load(&mut be, &BigUint::from(1u32));
    be.mul(b, one).unwrap();
    assert!(be.equal(a, b).unwrap());
    be.square(b, 1).unwrap();
    assert_eq!(be.equal(a, b).unwrap(), x == (&x * &x) % modulus(p));
    // (N - 1)^2 = 1: a value reached through the transform versus an imported one.
    let c = load(&mut be, &(modulus(p) - 1u32));
    be.square(c, 1).unwrap();
    assert!(be.equal(c, one).unwrap());
}

#[test]
fn copy_and_fault_injection() {
    let mut rng = Rng(0x5eed_0000_0000_0008);
    let p = 521;
    let mut be = DwtBackend::new(p).unwrap();
    let x = random(p, &mut rng);
    let a = load(&mut be, &x);
    let b = be.alloc().unwrap();
    be.copy(b, a).unwrap();
    be.copy(b, b).unwrap();
    assert_eq!(read(&mut be, b), x);
    be.corrupt(b, Corruption::FlipBit(p + 37)).unwrap();
    assert!(!be.equal(a, b).unwrap());
    assert_eq!(read(&mut be, b), &x ^ (BigUint::from(1u32) << 37u32));
    assert_eq!(read(&mut be, a), x, "copy is independent storage");
    be.corrupt(b, Corruption::Zero).unwrap();
    assert_eq!(read(&mut be, b), BigUint::from(0u32));
}

#[test]
fn handle_misuse_and_range_errors() {
    let mut be = DwtBackend::new(127).unwrap();
    let a = be.alloc().unwrap();
    let b = be.alloc().unwrap();
    assert_ne!(a, b);
    be.free(b).unwrap();
    assert_eq!(be.free(b), Err(BackendError::InvalidHandle(b.0)));
    assert_eq!(be.export(b).err(), Some(BackendError::InvalidHandle(b.0)));
    assert_eq!(be.square(b, 1), Err(BackendError::InvalidHandle(b.0)));
    assert_eq!(be.copy(a, b), Err(BackendError::InvalidHandle(b.0)));
    assert_eq!(be.copy(b, a), Err(BackendError::InvalidHandle(b.0)));
    assert_eq!(be.mul(a, b), Err(BackendError::InvalidHandle(b.0)));
    assert_eq!(
        be.equal(a, Reg(9999)),
        Err(BackendError::InvalidHandle(9999))
    );
    // A freed handle is never reissued.
    let c = be.alloc().unwrap();
    assert_ne!(c, b);
    assert!(matches!(be.mul(a, a), Err(BackendError::Unsupported(_))));
    assert!(matches!(
        be.import(a, &Residue::zero(89)),
        Err(BackendError::Unsupported(_))
    ));
    assert_eq!(
        be.corrupt(b, Corruption::Zero),
        Err(BackendError::InvalidHandle(b.0))
    );

    let (lo, hi) = supported_exponents();
    assert_eq!(lo, 2);
    assert!(matches!(
        DwtBackend::new(lo - 1),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        DwtBackend::new(hi + 1),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        DwtBackend::new(0),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        DwtBackend::with_transform_length(100, 101),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        DwtBackend::with_transform_length(127, 4),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn every_small_exponent_has_a_plan_within_the_bound() {
    for p in 2..=3000u32 {
        let mut be = DwtBackend::new(p).unwrap();
        let d = be.diagnostics().unwrap();
        let get = |k: &str| d.iter().find(|(key, _)| key == k).unwrap().1.clone();
        let n: u32 = get("transform_length").parse().unwrap();
        assert!(n >= 1 && n <= p);
        assert_eq!(be.identity().plan, format!("f64-dwt n={n}"));
        assert_eq!(be.identity().kind, "dwt");
        assert_eq!(be.identity().device, "");
    }
}
