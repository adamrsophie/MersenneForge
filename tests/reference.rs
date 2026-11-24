// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Reference vectors: known Mersenne prime exponents and composite Mersenne
//! numbers with prime exponents, through the oracle and through the checked
//! engines on the CPU and DWT backends.

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::backend::dwt::DwtBackend;
use mersenne_forge::backend::Backend;
use mersenne_forge::ll::{run_ll, JacobiCadence, LlConfig};
use mersenne_forge::oracle;
use mersenne_forge::prp::{run_prp, PrpConfig};
use mersenne_forge::run::{Control, Verdict};

const PRIMES: [u32; 15] = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279];
const COMPOSITES: [u32; 5] = [11, 23, 29, 37, 41];

fn engine_verdicts<B: Backend>(mut be: B, p: u32) -> (Verdict, Verdict) {
    let ll = LlConfig {
        log_step: 100,
        jacobi: JacobiCadence::EveryLog,
        checkpoint: None,
        max_batch: 1000,
    };
    let block = if p < 64 { 2 } else { 16 };
    let prp = PrpConfig {
        block_size: block,
        check_step: Some(4 * u64::from(block)),
        ..PrpConfig::default()
    };
    let l = run_ll(
        &mut be,
        p,
        &ll,
        serde_json::json!({}),
        None,
        &mut Control::default(),
    )
    .unwrap();
    let r = run_prp(
        &mut be,
        p,
        &prp,
        serde_json::json!({}),
        None,
        &mut Control::default(),
    )
    .unwrap();
    (l.result.unwrap().verdict, r.result.unwrap().verdict)
}

#[test]
fn oracle_reference_vectors() {
    for p in PRIMES {
        assert!(oracle::ll(p).positive, "LL M{p}");
        assert!(oracle::prp(p).positive, "PRP M{p}");
    }
    for p in COMPOSITES {
        assert!(!oracle::ll(p).positive, "LL M{p}");
        assert!(!oracle::prp(p).positive, "PRP M{p}");
    }
    for p in [3u32, 61, 127, 1279, 1277] {
        assert_eq!(
            oracle::prp(p).terminal,
            oracle::prp_terminal_modpow(p),
            "M{p}"
        );
    }
}

#[test]
fn engines_match_reference_vectors() {
    for p in PRIMES.iter().chain(COMPOSITES.iter()).copied() {
        let want = if oracle::ll(p).positive {
            (Verdict::Prime, Verdict::ProbablePrime)
        } else {
            (Verdict::Composite, Verdict::Composite)
        };
        let want = if p == 2 {
            (Verdict::Prime, Verdict::Prime)
        } else {
            want
        };
        assert_eq!(
            engine_verdicts(CpuBackend::new(p).unwrap(), p),
            want,
            "cpu M{p}"
        );
        assert_eq!(
            engine_verdicts(DwtBackend::new(p).unwrap(), p),
            want,
            "dwt M{p}"
        );
    }
}

#[test]
fn larger_mersenne_primes() {
    for p in [2203u32, 2281, 3217, 4253, 4423] {
        let want = (Verdict::Prime, Verdict::ProbablePrime);
        assert_eq!(
            engine_verdicts(DwtBackend::new(p).unwrap(), p),
            want,
            "dwt M{p}"
        );
    }
    assert_eq!(
        engine_verdicts(CpuBackend::new(4423).unwrap(), 4423),
        (Verdict::Prime, Verdict::ProbablePrime)
    );
}

#[test]
#[ignore = "slow in debug builds; run with --release -- --ignored"]
fn large_mersenne_primes_slow() {
    for p in [9689u32, 9941, 11213, 19937] {
        let want = (Verdict::Prime, Verdict::ProbablePrime);
        assert_eq!(
            engine_verdicts(DwtBackend::new(p).unwrap(), p),
            want,
            "dwt M{p}"
        );
    }
    assert_eq!(
        engine_verdicts(CpuBackend::new(9689).unwrap(), 9689),
        (Verdict::Prime, Verdict::ProbablePrime)
    );
}

/// Residue types 2 and 4: the chain stops at y = 3^(2^(p-1)), which is
/// N - 3 for every Mersenne prime N with p odd, and the final value matches
/// the host's power.
#[test]
fn prp_before_last_matches_the_host_and_flags_primes() {
    use num_bigint::BigUint;
    for p in PRIMES.iter().chain(COMPOSITES.iter()).copied() {
        if p < 3 {
            continue;
        }
        let cfg = PrpConfig {
            block_size: if p < 64 { 2 } else { 16 },
            check_step: Some(4 * if p < 64 { 2 } else { 16 }),
            before_last: true,
            ..PrpConfig::default()
        };
        let mut be = CpuBackend::new(p).unwrap();
        let r = run_prp(
            &mut be,
            p,
            &cfg,
            serde_json::json!({}),
            None,
            &mut Control::default(),
        )
        .unwrap()
        .result
        .unwrap();
        let n = (BigUint::from(1u32) << p) - 1u32;
        let y = BigUint::from(3u32).modpow(&(BigUint::from(1u32) << (p - 1)), &n);
        assert_eq!(r.final_residue.unwrap().to_biguint(), y, "M{p}");
        let prime = oracle::ll(p).positive;
        assert_eq!(
            r.verdict,
            if prime {
                Verdict::ProbablePrime
            } else {
                Verdict::Composite
            },
            "M{p}"
        );
        if prime {
            assert_eq!(y, &n - 3u32, "M{p}");
        }
    }
    // No proof exists for the shorter chain.
    let cfg = PrpConfig {
        before_last: true,
        proof_dir: Some(std::env::temp_dir()),
        ..PrpConfig::default()
    };
    let mut be = CpuBackend::new(127).unwrap();
    assert!(run_prp(
        &mut be,
        127,
        &cfg,
        serde_json::json!({}),
        None,
        &mut Control::default()
    )
    .is_err());
}
