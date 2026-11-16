// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Residue encoding boundaries and differential tests of the backends
//! against the independent oracle.

use num_bigint::BigUint;
use proptest::prelude::*;

use mersenne_forge::backend::cpu::{Corruption, CpuBackend};
use mersenne_forge::backend::dwt::DwtBackend;
use mersenne_forge::backend::{Backend, BackendError, Reg};
use mersenne_forge::oracle;
use mersenne_forge::residue::{limb_count, modulus, Residue, ResidueError};

/// Word-aligned and non-word-aligned prime exponents.
const EXPONENTS: [u32; 10] = [2, 3, 5, 31, 61, 89, 127, 521, 607, 1279];

fn boundary_values(p: u32) -> Vec<Residue> {
    let n = modulus(p);
    let one = BigUint::from(1u32);
    let mut v = vec![
        BigUint::from(0u32),
        one.clone(),
        BigUint::from(2u32),
        &n - 1u32,
        &n - 2u32,
        &n >> 1,
        (&n >> 1) + 1u32,
    ];
    // Values around every limb boundary, and all-ones words below the top.
    for b in (32..p).step_by(32) {
        v.push((&one << b) - 1u32);
        v.push(&one << b);
        v.push((&one << b) + 1u32);
    }
    v.into_iter()
        .map(|x| Residue::from_biguint(p, &x))
        .collect()
}

#[test]
fn canonical_encoding_boundaries() {
    for p in EXPONENTS.iter().copied().chain([32 + 1, 64 - 3, 4423]) {
        let n = limb_count(p);
        for x in boundary_values(p) {
            assert_eq!(Residue::from_limbs(p, x.limbs().to_vec()).unwrap(), x);
            assert_eq!(Residue::from_le_bytes(p, &x.to_le_bytes()).unwrap(), x);
            assert!(x.to_biguint() < modulus(p));
        }
        let mut all_ones = vec![u32::MAX; n];
        if p % 32 != 0 {
            all_ones[n - 1] = (1 << (p % 32)) - 1;
        }
        assert_eq!(
            Residue::from_limbs(p, all_ones.clone()),
            Err(ResidueError::ValueIsModulus),
            "p={p}"
        );
        assert!(Residue::from_limbs_normalizing(p, all_ones)
            .unwrap()
            .is_zero());
        if p % 32 != 0 {
            let mut high = vec![0u32; n];
            high[n - 1] = 1 << (p % 32);
            assert_eq!(
                Residue::from_limbs(p, high.clone()),
                Err(ResidueError::HighBits)
            );
            assert_eq!(
                Residue::from_limbs_normalizing(p, high),
                Err(ResidueError::HighBits)
            );
        }
        assert!(matches!(
            Residue::from_limbs(p, vec![0; n + 1]),
            Err(ResidueError::Length { .. })
        ));
        // N reduces to the unique zero, N + 1 to one.
        assert!(Residue::from_biguint(p, &modulus(p)).is_zero());
        assert_eq!(
            Residue::from_biguint(p, &(modulus(p) + 1u32)),
            Residue::from_u64(p, 1)
        );
    }
}

fn check_backend<B: Backend>(be: &mut B, x: &Residue, y: &Residue) {
    let (a, b) = (be.alloc().unwrap(), be.alloc().unwrap());
    be.import(a, x).unwrap();
    be.import(b, y).unwrap();
    be.mul(a, b).unwrap();
    assert_eq!(be.export(a).unwrap(), oracle::mul(x, y));
    be.import(a, x).unwrap();
    be.square(a, 3).unwrap();
    assert_eq!(be.export(a).unwrap(), oracle::square_n(x, 3));
    be.import(a, x).unwrap();
    be.ll_step(a, 2).unwrap();
    assert_eq!(be.export(a).unwrap(), oracle::ll_steps(x, 2));
    be.import(a, x).unwrap();
    be.import(b, y).unwrap();
    assert_eq!(be.equal(a, b).unwrap(), x == y);
    be.copy(b, a).unwrap();
    assert!(be.equal(a, b).unwrap());
    be.free(a).unwrap();
    be.free(b).unwrap();
}

#[test]
fn backends_match_oracle_on_boundaries() {
    for p in EXPONENTS {
        let vals = boundary_values(p);
        let mut cpu = CpuBackend::new(p).unwrap();
        let mut dwt = DwtBackend::new(p).unwrap();
        for x in &vals {
            for y in vals.iter().take(8) {
                check_backend(&mut cpu, x, y);
                check_backend(&mut dwt, x, y);
            }
        }
    }
}

#[test]
fn backend_handle_errors() {
    let mut be = CpuBackend::new(31).unwrap();
    let a = be.alloc().unwrap();
    assert_eq!(
        be.mul(a, a),
        Err(BackendError::Unsupported(
            "mul needs distinct handles".into()
        ))
    );
    be.free(a).unwrap();
    assert_eq!(be.export(a), Err(BackendError::InvalidHandle(a.0)));
    for bad in [Reg(0), Reg(u32::MAX)] {
        assert_eq!(be.square(bad, 1), Err(BackendError::InvalidHandle(bad.0)));
    }
    let b = be.alloc().unwrap();
    assert!(be.import(b, &Residue::from_u64(61, 1)).is_err());
}

/// A handle issued by one instance is `InvalidHandle` on another instance
/// of the same kind, and neither instance's residue changes.
fn handles_stay_with_their_instance<B: Backend>(make: impl Fn() -> B) {
    let p = make().exponent();
    let (mut one, mut two) = (make(), make());
    let a = one.alloc().unwrap();
    let b = two.alloc().unwrap();
    one.import(a, &Residue::from_u64(p, 5)).unwrap();
    two.import(b, &Residue::from_u64(p, 7)).unwrap();
    let inv = BackendError::InvalidHandle(a.0);
    assert_eq!(two.export(a).err(), Some(inv.clone()));
    assert_eq!(two.square(a, 1), Err(inv.clone()));
    assert_eq!(two.import(a, &Residue::from_u64(p, 9)), Err(inv.clone()));
    assert_eq!(two.copy(b, a), Err(inv.clone()));
    assert_eq!(two.mul(b, a), Err(inv.clone()));
    assert_eq!(two.free(a), Err(inv));
    assert_eq!(one.free(b), Err(BackendError::InvalidHandle(b.0)));
    assert_eq!(one.export(a).unwrap(), Residue::from_u64(p, 5));
    assert_eq!(two.export(b).unwrap(), Residue::from_u64(p, 7));
}

#[test]
fn cpu_handles_stay_with_their_instance() {
    handles_stay_with_their_instance(|| CpuBackend::new(127).unwrap());
}

#[test]
fn dwt_handles_stay_with_their_instance() {
    handles_stay_with_their_instance(|| DwtBackend::new(127).unwrap());
}

#[test]
fn cpu_corruption_hook() {
    let p = 127;
    let mut be = CpuBackend::new(p).unwrap();
    let a = be.alloc().unwrap();
    let x = Residue::from_u64(p, 12345);
    be.import(a, &x).unwrap();
    be.corrupt(a, Corruption::FlipBit(3)).unwrap();
    assert_eq!(be.export(a).unwrap(), Residue::from_u64(p, 12345 ^ 8));
    be.corrupt(a, Corruption::Zero).unwrap();
    assert!(be.export(a).unwrap().is_zero());
}

fn residue_strategy(p: u32) -> impl Strategy<Value = Residue> {
    proptest::collection::vec(any::<u32>(), limb_count(p))
        .prop_map(move |limbs| Residue::from_biguint(p, &BigUint::from_slice(&limbs)))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_residues_cpu_and_dwt_match_oracle(
        idx in 0usize..EXPONENTS.len(),
        seed_x in proptest::collection::vec(any::<u32>(), 40),
        seed_y in proptest::collection::vec(any::<u32>(), 40),
    ) {
        let p = EXPONENTS[idx];
        let x = Residue::from_biguint(p, &BigUint::from_slice(&seed_x));
        let y = Residue::from_biguint(p, &BigUint::from_slice(&seed_y));
        check_backend(&mut CpuBackend::new(p).unwrap(), &x, &y);
        check_backend(&mut DwtBackend::new(p).unwrap(), &x, &y);
    }

    #[test]
    fn random_residues_at_1279(x in residue_strategy(1279), y in residue_strategy(1279)) {
        check_backend(&mut CpuBackend::new(1279).unwrap(), &x, &y);
        check_backend(&mut DwtBackend::new(1279).unwrap(), &x, &y);
    }

    #[test]
    fn byte_round_trips(x in residue_strategy(607)) {
        prop_assert_eq!(Residue::from_le_bytes(607, &x.to_le_bytes()).unwrap(), x.clone());
        prop_assert_eq!(Residue::from_proof_bytes(607, &x.to_proof_bytes()).unwrap(), x);
    }
}
