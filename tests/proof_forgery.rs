// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Chosen proofs for small moduli. For a small M_p the version 2 proof
//! equation accepts many wrong terminal values, so `verify` recomputes the
//! terminal residue below `DIRECT_RECOMPUTE_BELOW`; `verify_equation` keeps
//! the equation alone, and these tests document what it accepts.

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::proof::{self, Proof, ProofError, DIRECT_RECOMPUTE_BELOW};
use mersenne_forge::prp::{checked_power, CheckedPowerConfig};
use mersenne_forge::residue::Residue;
use mersenne_forge::run::{Control, ProtocolError};

/// Two little-endian bytes: the proof residue size for M11 and M13.
fn le2(v: u16) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn proof(p: u32, b: u16, middles: &[u16]) -> Proof {
    Proof {
        p,
        b: le2(b),
        middles: middles.iter().map(|&m| le2(m)).collect(),
    }
}

fn full(pr: &Proof) -> Result<proof::Verified, ProtocolError> {
    proof::verify(
        &mut CpuBackend::new(pr.p).unwrap(),
        pr,
        &mut Control::default(),
    )
}

fn equation(pr: &Proof) -> Result<proof::Verified, ProtocolError> {
    proof::verify_equation(
        &mut CpuBackend::new(pr.p).unwrap(),
        pr,
        &mut Control::default(),
    )
}

fn rejected(r: Result<proof::Verified, ProtocolError>) -> bool {
    matches!(r, Err(ProtocolError::Proof(ProofError::Rejected)))
}

#[test]
fn small_exponents_are_recomputed() {
    const { assert!(DIRECT_RECOMPUTE_BELOW == 1 << 17) };
    // 3^(2^13) mod 8191 = 9 (M13 is prime); 3^(2^11) mod 2047 = 929.
    let honest = Proof::parse(
        &[
            b"PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=3\nNUMBER=M13\n".as_slice(),
            &le2(9),
            &le2(7434),
            &le2(6019),
            &le2(3085),
        ]
        .concat(),
        Some(13),
    )
    .expect("a power whose generation schedule collides still parses");
    let v = full(&honest).unwrap();
    assert!(v.probable_prime && v.terminal_recomputed);
    assert!(!equation(&honest).unwrap().terminal_recomputed);
}

#[test]
fn forged_terminal_values_pass_the_equation_but_not_verify() {
    // M13 is prime, terminal 9: the claim 8182 (= -9) satisfies the equation.
    // M11: a unit middle (9) and a zero divisor middle (1513 = 17 * 89) both
    // make the wrong terminal 9 satisfy the equation, with nonzero folds.
    for pr in [
        proof(13, 8182, &[7434]),
        proof(11, 9, &[9]),
        proof(11, 9, &[1513]),
    ] {
        let eq = equation(&pr).expect("the equation alone accepts this forgery");
        assert!(!eq.terminal_recomputed);
        assert!(rejected(full(&pr)), "{pr:?}");
    }
}

#[test]
fn zero_folds_and_zero_middles_are_rejected() {
    // 23 * 89 = 2047: the two middles drive both folded values to zero.
    let pr = proof(11, 9, &[23, 89]);
    assert!(rejected(equation(&pr)));
    assert!(rejected(full(&pr)));
    // A zero middle in a directly constructed proof is refused before any
    // arithmetic, as the parser refuses it.
    let direct = Proof {
        p: 11,
        b: vec![9, 0],
        middles: vec![vec![0, 0]],
    };
    for r in [equation(&direct), full(&direct)] {
        assert!(
            matches!(
                r,
                Err(ProtocolError::Proof(ProofError::ZeroResidue { index: 1 }))
            ),
            "{r:?}"
        );
    }
    let bad_power = Proof {
        p: 11,
        b: vec![9, 0],
        middles: vec![],
    };
    assert!(matches!(
        full(&bad_power),
        Err(ProtocolError::Proof(ProofError::InvalidPower { .. }))
    ));
}

/// Every (B, middle) pair for M11 at power 1. Under `verify` only the true
/// terminal 929 is ever accepted; the equation alone accepts a middle for
/// most wrong terminal values.
#[test]
fn exhaustive_m11_power_1() {
    const P: u32 = 11;
    const N: u16 = 2047;
    const TRUE_TERMINAL: u16 = 929;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (accepted, equation_b): (Vec<(u16, u16)>, Vec<u16>) = std::thread::scope(|sc| {
        let jobs: Vec<_> = (0..threads)
            .map(|t| {
                sc.spawn(move || {
                    let mut be = CpuBackend::new(P).unwrap();
                    let mut found = Vec::new();
                    let mut eq_b = Vec::new();
                    for b in (0..N).filter(|b| usize::from(*b) % threads == t) {
                        for m in 1..N {
                            let pr = proof(P, b, &[m]);
                            if proof::verify_equation(&mut be, &pr, &mut Control::default()).is_ok()
                            {
                                eq_b.push(b);
                                break;
                            }
                        }
                        for m in 1..N {
                            let pr = proof(P, b, &[m]);
                            match proof::verify(&mut be, &pr, &mut Control::default()) {
                                Ok(_) => found.push((b, m)),
                                Err(ProtocolError::Proof(ProofError::Rejected)) => {}
                                Err(e) => panic!("B {b}, middle {m}: {e}"),
                            }
                        }
                    }
                    (found, eq_b)
                })
            })
            .collect();
        let mut all = (Vec::new(), Vec::new());
        for j in jobs {
            let (f, e) = j.join().unwrap();
            all.0.extend(f);
            all.1.extend(e);
        }
        all
    });
    let wrong = equation_b.iter().filter(|&&b| b != TRUE_TERMINAL).count();
    println!(
        "wrong terminal values with an accepted middle: {wrong} of {}",
        N - 1
    );
    // 1992 of the 2046 wrong values at the time of writing.
    assert!(2 * wrong > usize::from(N - 1), "{wrong}");
    assert!(!accepted.is_empty(), "the honest proof is accepted");
    assert!(
        accepted.iter().all(|&(b, _)| b == TRUE_TERMINAL),
        "{accepted:?}"
    );
    assert!(equation(&proof(P, 9, &[9])).is_ok());
}

#[test]
fn checked_power_validates_its_input() {
    let p = 11;
    let mut be = CpuBackend::new(p).unwrap();
    let cfg = CheckedPowerConfig::for_length(20);
    let three = Residue::from_u64(p, 3);
    let zero = Residue::zero(p);
    assert!(matches!(
        checked_power(&mut be, &zero, 20, &cfg, &mut Control::default()),
        Err(ProtocolError::Config(_))
    ));
    let same = checked_power(&mut be, &three, 0, &cfg, &mut Control::default()).unwrap();
    assert_eq!(same.residue, three);
    for bad in [
        CheckedPowerConfig { block: 0, ..cfg },
        CheckedPowerConfig {
            max_batch: 0,
            ..cfg
        },
        CheckedPowerConfig {
            check_step: 0,
            ..cfg
        },
    ] {
        assert!(matches!(
            checked_power(&mut be, &three, 20, &bad, &mut Control::default()),
            Err(ProtocolError::Config(_))
        ));
    }
    let x = checked_power(&mut be, &three, 20, &cfg, &mut Control::default()).unwrap();
    assert_eq!(x.residue, mersenne_forge::oracle::square_n(&three, 20));
    assert!(matches!(
        checked_power(&mut be, &three, u64::MAX, &cfg, &mut Control::default()),
        Err(ProtocolError::Config(_))
    ));
    let wide = CheckedPowerConfig {
        max_batch: u64::MAX,
        ..cfg
    };
    let x = checked_power(&mut be, &three, 20, &wide, &mut Control::default()).unwrap();
    assert_eq!(x.residue, mersenne_forge::oracle::square_n(&three, 20));
}
