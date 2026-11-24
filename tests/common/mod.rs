// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Backend-generic protocol tests. Each function takes a backend factory, so
//! the same assertions run on `cpu`, `dwt`, `gpu`, and `cuda`. The
//! `protocol_suite!` macro instantiates them as `#[test]` functions.

#![allow(dead_code)]

use std::path::Path;

use mersenne_forge::backend::Backend;
use mersenne_forge::checkpoint::Checkpoint;
use mersenne_forge::ll::{run_ll, JacobiCadence, LlConfig};
use mersenne_forge::oracle;
use mersenne_forge::proof::{self, Proof, ProofError};
use mersenne_forge::prp::{run_prp, PrpConfig};
use mersenne_forge::residue::Residue;
use mersenne_forge::run::{
    Control, FaultKind, FaultPlan, FaultTarget, ProcessState, ProofStage, ProtocolError, RunReport,
    Verdict,
};

/// M1279 is prime; M1277 is composite.
pub const PRIME_P: u32 = 1279;
pub const COMPOSITE_P: u32 = 1277;

pub fn prp_cfg(dir: &Path, proof_power: Option<u32>) -> PrpConfig {
    PrpConfig {
        block_size: 16,
        check_step: Some(64),
        proof_power,
        proof_dir: proof_power.map(|_| dir.join("proof")),
        verify_proof: true,
        checkpoint: Some(dir.join("ckpt").join("prp.ckpt")),
        max_batch: 1000,
        before_last: false,
        shift: 0,
    }
}

pub fn ll_cfg(dir: &Path) -> LlConfig {
    LlConfig {
        log_step: 50,
        jacobi: JacobiCadence::EveryLog,
        checkpoint: Some(dir.join("ckpt").join("ll.ckpt")),
        max_batch: 1000,
    }
}

/// The PRP report's type 1 res64 as the oracle computes it.
pub fn oracle_prp_res64(p: u32) -> String {
    let x = oracle::prp(p).terminal;
    format!("{:016x}", x.div3().div3().res64())
}

pub fn prp_run<B: Backend>(
    be: &mut B,
    p: u32,
    cfg: &PrpConfig,
    ctl: &mut Control,
) -> Result<RunReport, ProtocolError> {
    run_prp(be, p, cfg, serde_json::json!({}), None, ctl)
}

pub fn prp_resume<B: Backend>(
    be: &mut B,
    p: u32,
    cfg: &PrpConfig,
    ctl: &mut Control,
) -> Result<RunReport, ProtocolError> {
    let (ck, _) =
        Checkpoint::load_latest(cfg.checkpoint.as_ref().unwrap()).expect("checkpoint loads");
    run_prp(be, p, cfg, serde_json::json!({}), Some(&ck), ctl)
}

fn assert_prp_correct(r: &RunReport, p: u32) {
    assert_eq!(r.process, ProcessState::Completed);
    let res = r.result.as_ref().expect("completed run has a result");
    let want = if oracle::prp(p).positive {
        Verdict::ProbablePrime
    } else {
        Verdict::Composite
    };
    assert_eq!(res.verdict, want, "p={p}");
    assert_eq!(res.res64, oracle_prp_res64(p), "p={p}");
    let full = oracle::prp(p).terminal.div3().div3();
    assert_eq!(res.res2048, full.res2048_hex(), "p={p}");
}

fn proof_path(r: &RunReport) -> std::path::PathBuf {
    let pe = r.evidence.proof.as_ref().expect("proof generated");
    assert_eq!(pe.self_verification, "accepted");
    pe.path.clone()
}

/// Complete PRP runs with proofs for a prime and a composite exponent.
pub fn prp_prime_and_composite_with_proofs<B: Backend>(mk: impl Fn(u32) -> B) {
    for p in [PRIME_P, COMPOSITE_P] {
        let dir = tempfile::tempdir().unwrap();
        let mut be = mk(p);
        let r = prp_run(
            &mut be,
            p,
            &prp_cfg(dir.path(), Some(5)),
            &mut Control::default(),
        )
        .unwrap();
        assert_prp_correct(&r, p);
        assert!(r.evidence.gerbicz_passed >= 20, "{:?}", r.evidence);
        assert_eq!(r.evidence.gerbicz_failed, 0);
        let proof = Proof::read(&proof_path(&r), Some(p)).unwrap();
        assert_eq!(proof.power(), 5);
        let v = proof::verify(&mut be, &proof, &mut Control::default()).unwrap();
        assert_eq!(v.probable_prime, oracle::prp(p).positive);
        assert!(
            v.terminal_recomputed,
            "p = {p} is below the recompute cutoff"
        );
        assert_eq!(r.evidence.proof_stage, ProofStage::Verified);
        let bytes = std::fs::read(proof_path(&r)).unwrap();
        let resumed = prp_resume(
            &mut be,
            p,
            &prp_cfg(dir.path(), Some(5)),
            &mut Control::default(),
        )
        .unwrap();
        assert_prp_correct(&resumed, p);
        assert_eq!(resumed.evidence.gerbicz_passed, 0);
        assert_eq!(resumed.timings.iterations, 0.0);
        assert_eq!(std::fs::read(proof_path(&resumed)).unwrap(), bytes);
    }
}

pub fn ll_prime_and_composite<B: Backend>(mk: impl Fn(u32) -> B) {
    for p in [PRIME_P, COMPOSITE_P] {
        let dir = tempfile::tempdir().unwrap();
        let r = run_ll(
            &mut mk(p),
            p,
            &ll_cfg(dir.path()),
            serde_json::json!({}),
            None,
            &mut Control::default(),
        )
        .unwrap();
        let reference = oracle::ll(p);
        let res = r.result.unwrap();
        assert_eq!(res.verdict == Verdict::Prime, reference.positive);
        assert_eq!(res.res2048, reference.terminal.res2048_hex());
        assert!(r.evidence.jacobi_passed >= 20 && r.evidence.jacobi_failed == 0);
    }
}

/// A corrupted data residue and a corrupted check value are both detected,
/// rolled back, and the final result is still correct.
pub fn prp_faults_detected_and_recovered<B: Backend>(mk: impl Fn(u32) -> B) {
    let p = PRIME_P;
    let cases = [
        FaultPlan::once(100, FaultTarget::Data, FaultKind::FlipBit(7)),
        FaultPlan::once(100, FaultTarget::Check, FaultKind::FlipBit(300)),
        FaultPlan::once(130, FaultTarget::Check, FaultKind::Zero),
        FaultPlan::once(1279, FaultTarget::Data, FaultKind::Triple),
        FaultPlan::once(200, FaultTarget::Data, FaultKind::BackendFault),
    ];
    for fault in cases {
        let dir = tempfile::tempdir().unwrap();
        let mut ctl = Control {
            faults: vec![fault.clone()],
            ..Control::default()
        };
        let r = prp_run(&mut mk(p), p, &prp_cfg(dir.path(), Some(4)), &mut ctl).unwrap();
        assert!(ctl.faults[0].fired, "{fault:?}");
        assert_prp_correct(&r, p);
        let detected = r.evidence.gerbicz_failed + r.evidence.backend_faults;
        assert_eq!(detected, 1, "{fault:?}: {:?}", r.evidence);
        assert!(r.evidence.rollbacks >= 1);
        proof_path(&r);
    }
}

/// The same wrong residue twice stops; three different failures in a row stop.
pub fn prp_stop_conditions<B: Backend>(mk: impl Fn(u32) -> B) {
    let p = PRIME_P;
    let dir = tempfile::tempdir().unwrap();
    let mut ctl = Control {
        faults: vec![FaultPlan::every_pass(
            100,
            FaultTarget::Data,
            FaultKind::FlipBit(3),
        )],
        ..Control::default()
    };
    let e = prp_run(&mut mk(p), p, &prp_cfg(dir.path(), None), &mut ctl).unwrap_err();
    assert!(matches!(e, ProtocolError::ConsistentError { .. }), "{e}");

    let dir = tempfile::tempdir().unwrap();
    let faults = (0..3)
        .map(|i| FaultPlan::once(65, FaultTarget::Data, FaultKind::FlipBit(i)))
        .collect();
    let mut ctl = Control {
        faults,
        ..Control::default()
    };
    let e = prp_run(&mut mk(p), p, &prp_cfg(dir.path(), None), &mut ctl).unwrap_err();
    assert!(matches!(e, ProtocolError::TooManyErrors { .. }), "{e}");

    // Two failures then success: recovered, not stopped.
    let dir = tempfile::tempdir().unwrap();
    let faults = (0..2)
        .map(|i| FaultPlan::once(100, FaultTarget::Data, FaultKind::FlipBit(i)))
        .collect();
    let mut ctl = Control {
        faults,
        ..Control::default()
    };
    let r = prp_run(&mut mk(p), p, &prp_cfg(dir.path(), None), &mut ctl).unwrap();
    assert_prp_correct(&r, p);
    assert_eq!(r.evidence.gerbicz_failed, 2);
}

/// Simulated abrupt termination at several boundaries, then resume: same
/// final residue and a byte-identical, accepted proof.
pub fn prp_interrupt_and_resume<B: Backend>(mk: impl Fn(u32) -> B) {
    let p = PRIME_P;
    let base_dir = tempfile::tempdir().unwrap();
    let base = prp_run(
        &mut mk(p),
        p,
        &prp_cfg(base_dir.path(), Some(5)),
        &mut Control::default(),
    )
    .unwrap();
    let base_proof = std::fs::read(proof_path(&base)).unwrap();
    // Points for power 5 include 640, 960, 1280 - 1 = 1279; checks every 64.
    let sched = proof::ProofSchedule::new(p, 5).unwrap();
    assert!(sched.contains(640) && sched.contains(600) && sched.contains(680));
    for halt in [77u64, 128, 129, 639, 640, 641, 1279, 1280] {
        let dir = tempfile::tempdir().unwrap();
        let cfg = prp_cfg(dir.path(), Some(5));
        let mut ctl = Control {
            halt_at: Some(halt),
            ..Control::default()
        };
        let r = prp_run(&mut mk(p), p, &cfg, &mut ctl).unwrap();
        assert_eq!(r.process, ProcessState::Halted { k: halt });
        let r = prp_resume(&mut mk(p), p, &cfg, &mut Control::default()).unwrap();
        assert_prp_correct(&r, p);
        assert_eq!(r.result, base.result, "halt at {halt}");
        assert_eq!(
            std::fs::read(proof_path(&r)).unwrap(),
            base_proof,
            "halt at {halt}"
        );
    }
    // Cooperative stop (budget): stops at a check boundary and resumes.
    let dir = tempfile::tempdir().unwrap();
    let cfg = prp_cfg(dir.path(), Some(5));
    let mut ctl = Control {
        deadline: Some(std::time::Instant::now()),
        ..Control::default()
    };
    let r = prp_run(&mut mk(p), p, &cfg, &mut ctl).unwrap();
    assert!(
        matches!(r.process, ProcessState::Incomplete { k: 16, .. }),
        "{:?}",
        r.process
    );
    assert!(r.result.is_none());
    let r = prp_resume(&mut mk(p), p, &cfg, &mut Control::default()).unwrap();
    assert_eq!(r.result, base.result);
    assert_eq!(std::fs::read(proof_path(&r)).unwrap(), base_proof);
}

/// A proof residue that fails to become durable stops the run; the
/// checkpoint on disk is before it, and resuming produces a valid proof.
pub fn prp_proof_residue_write_failure<B: Backend>(mk: impl Fn(u32) -> B) {
    let p = PRIME_P;
    let dir = tempfile::tempdir().unwrap();
    let cfg = prp_cfg(dir.path(), Some(5));
    let missing = 680u64;
    let mut ctl = Control {
        fail_proof_write_at: Some(missing),
        ..Control::default()
    };
    let e = prp_run(&mut mk(p), p, &cfg, &mut ctl).unwrap_err();
    assert!(
        matches!(e, ProtocolError::ProofResidueWrite { k: 680, .. }),
        "{e}"
    );
    let (ck, _) = Checkpoint::load_latest(cfg.checkpoint.as_ref().unwrap()).unwrap();
    assert!(
        ck.meta.k < missing,
        "checkpoint at {} advanced past the missing residue",
        ck.meta.k
    );
    assert_eq!(ck.meta.k, 640);
    let r = prp_resume(&mut mk(p), p, &cfg, &mut Control::default()).unwrap();
    assert_prp_correct(&r, p);
    proof_path(&r);

    // Deleting a durable residue below the checkpoint makes resume refuse.
    let dir = tempfile::tempdir().unwrap();
    let cfg = prp_cfg(dir.path(), Some(5));
    let mut ctl = Control {
        halt_at: Some(1000),
        ..Control::default()
    };
    prp_run(&mut mk(p), p, &cfg, &mut ctl).unwrap();
    let rdir = proof::ProofStore::residue_dir(cfg.proof_dir.as_ref().unwrap(), p);
    std::fs::remove_file(rdir.join("640")).unwrap();
    let e = prp_resume(&mut mk(p), p, &cfg, &mut Control::default()).unwrap_err();
    assert!(
        matches!(e, ProtocolError::ProofResidueMissing { k: 640 }),
        "{e}"
    );
}

/// The LL Jacobi check detects a symbol-flipping fault applied at a check
/// iteration (s - 2 -> 3(s - 2) always flips the symbol there; later residues
/// carry a fresh symbol, so detection after further steps is only about half).
pub fn ll_fault_detected_and_recovered<B: Backend>(mk: impl Fn(u32) -> B) {
    let p = PRIME_P;
    let dir = tempfile::tempdir().unwrap();
    let mut ctl = Control {
        faults: vec![FaultPlan::once(150, FaultTarget::Data, FaultKind::Triple)],
        ..Control::default()
    };
    let r = run_ll(
        &mut mk(p),
        p,
        &ll_cfg(dir.path()),
        serde_json::json!({}),
        None,
        &mut ctl,
    )
    .unwrap();
    assert_eq!(r.evidence.jacobi_failed, 1);
    assert_eq!(r.evidence.rollbacks, 1);
    assert_eq!(r.result.unwrap().verdict, Verdict::Prime);

    let dir = tempfile::tempdir().unwrap();
    let mut ctl = Control {
        faults: vec![FaultPlan::every_pass(
            150,
            FaultTarget::Data,
            FaultKind::Triple,
        )],
        ..Control::default()
    };
    let e = run_ll(
        &mut mk(p),
        p,
        &ll_cfg(dir.path()),
        serde_json::json!({}),
        None,
        &mut ctl,
    )
    .unwrap_err();
    assert!(
        matches!(e, ProtocolError::ConsistentError { k: 150 }),
        "{e}"
    );

    // A zeroed residue is caught (s = N - 2 then s = 2 paths).
    let dir = tempfile::tempdir().unwrap();
    let mut ctl = Control {
        faults: vec![FaultPlan::once(120, FaultTarget::Data, FaultKind::Zero)],
        ..Control::default()
    };
    let r = run_ll(
        &mut mk(p),
        p,
        &ll_cfg(dir.path()),
        serde_json::json!({}),
        None,
        &mut ctl,
    )
    .unwrap();
    assert_eq!(r.evidence.jacobi_failed, 1);
    assert_eq!(r.result.unwrap().verdict, Verdict::Prime);
}

/// LL interrupted at several points and resumed reaches the same residue.
pub fn ll_interrupt_and_resume<B: Backend>(mk: impl Fn(u32) -> B) {
    let p = COMPOSITE_P;
    let base_dir = tempfile::tempdir().unwrap();
    let base = run_ll(
        &mut mk(p),
        p,
        &ll_cfg(base_dir.path()),
        serde_json::json!({}),
        None,
        &mut Control::default(),
    )
    .unwrap();
    for halt in [51u64, 100, 101, 1274, 1275] {
        let dir = tempfile::tempdir().unwrap();
        let cfg = ll_cfg(dir.path());
        let mut ctl = Control {
            halt_at: Some(halt),
            ..Control::default()
        };
        let r = run_ll(&mut mk(p), p, &cfg, serde_json::json!({}), None, &mut ctl).unwrap();
        assert_eq!(r.process, ProcessState::Halted { k: halt });
        let (ck, _) = Checkpoint::load_latest(cfg.checkpoint.as_ref().unwrap()).unwrap();
        let r = run_ll(
            &mut mk(p),
            p,
            &cfg,
            serde_json::json!({}),
            Some(&ck),
            &mut Control::default(),
        )
        .unwrap();
        assert_eq!(r.result, base.result, "halt at {halt}");
    }
}

/// Altered, truncated, wrong-exponent, and incompatible proofs are rejected
/// with distinct errors.
pub fn proof_rejections<B: Backend>(mk: impl Fn(u32) -> B) {
    let p = PRIME_P;
    let dir = tempfile::tempdir().unwrap();
    let mut be = mk(p);
    let r = prp_run(
        &mut be,
        p,
        &prp_cfg(dir.path(), Some(4)),
        &mut Control::default(),
    )
    .unwrap();
    let path = proof_path(&r);
    let good = std::fs::read(&path).unwrap();
    let parse = |b: &[u8], e: Option<u32>| Proof::parse(b, e);
    let proof = parse(&good, Some(p)).unwrap();
    assert!(proof::verify(&mut be, &proof, &mut Control::default()).is_ok());

    let v = proof::verify_equation(&mut be, &proof, &mut Control::default()).unwrap();
    assert!(!v.terminal_recomputed);

    // Altered B or middle: parses; the full verification (which recomputes
    // the terminal residue for this small p) and the proof equation alone
    // both reject.
    let hlen = good.len() - 5 * mersenne_forge::residue::proof_byte_len(p);
    for at in [hlen + 3, hlen + 170, good.len() - 2] {
        let mut bad = good.clone();
        bad[at] ^= 0x10;
        let pr = parse(&bad, Some(p)).unwrap();
        let e = proof::verify(&mut be, &pr, &mut Control::default()).unwrap_err();
        assert!(
            matches!(e, ProtocolError::Proof(ProofError::Rejected)),
            "byte {at}: {e}"
        );
        let e = proof::verify_equation(&mut be, &pr, &mut Control::default()).unwrap_err();
        assert!(
            matches!(e, ProtocolError::Proof(ProofError::Rejected)),
            "equation, byte {at}: {e}"
        );
    }
    assert!(matches!(
        parse(&good[..good.len() - 1], Some(p)),
        Err(ProofError::Truncated { .. })
    ));
    let mut longer = good.clone();
    longer.push(0);
    assert!(matches!(
        parse(&longer, Some(p)),
        Err(ProofError::TrailingData { .. })
    ));
    assert!(matches!(
        parse(&good, Some(1277)),
        Err(ProofError::WrongExponent {
            expected: 1277,
            found: 1279
        })
    ));
    let v1 = String::from_utf8_lossy(&good)
        .replacen("VERSION=2", "VERSION=3", 1)
        .into_bytes();
    assert_eq!(parse(&v1, None), Err(ProofError::BadHeader));
    // Bits above p in B (1279 = 159 * 8 + 7, so the top byte has one free bit).
    let mut high = good.clone();
    high[hlen + mersenne_forge::residue::proof_byte_len(p) - 1] |= 0x80;
    assert!(matches!(
        parse(&high, None),
        Err(ProofError::NoncanonicalResidue { index: 0 })
    ));
}

/// Final residues and proof files from two backends are identical.
pub fn cross_backend_identical<B1: Backend, B2: Backend>(
    mk1: impl Fn(u32) -> B1,
    mk2: impl Fn(u32) -> B2,
) {
    for p in [PRIME_P, COMPOSITE_P] {
        let d1 = tempfile::tempdir().unwrap();
        let d2 = tempfile::tempdir().unwrap();
        let r1 = prp_run(
            &mut mk1(p),
            p,
            &prp_cfg(d1.path(), Some(5)),
            &mut Control::default(),
        )
        .unwrap();
        let r2 = prp_run(
            &mut mk2(p),
            p,
            &prp_cfg(d2.path(), Some(5)),
            &mut Control::default(),
        )
        .unwrap();
        assert_eq!(r1.result, r2.result);
        assert_eq!(
            std::fs::read(proof_path(&r1)).unwrap(),
            std::fs::read(proof_path(&r2)).unwrap()
        );
        let l1 = run_ll(
            &mut mk1(p),
            p,
            &ll_cfg(d1.path()),
            serde_json::json!({}),
            None,
            &mut Control::default(),
        );
        let l2 = run_ll(
            &mut mk2(p),
            p,
            &ll_cfg(d2.path()),
            serde_json::json!({}),
            None,
            &mut Control::default(),
        );
        assert_eq!(l1.unwrap().result, l2.unwrap().result);
    }
}

/// Raw arithmetic agreement with the oracle on boundary residues.
pub fn arithmetic_matches_oracle<B: Backend>(mk: impl Fn(u32) -> B, p: u32) {
    let n = mersenne_forge::residue::modulus(p);
    let vals = [
        Residue::zero(p),
        Residue::from_u64(p, 1),
        Residue::from_u64(p, 3),
        Residue::from_biguint(p, &(&n - 1u32)),
        Residue::from_biguint(p, &(&n - 2u32)),
        Residue::from_biguint(p, &(num_bigint::BigUint::from(1u32) << (p / 2))),
    ];
    let mut be = mk(p);
    let (a, b) = (be.alloc().unwrap(), be.alloc().unwrap());
    for x in &vals {
        for y in &vals {
            be.import(a, x).unwrap();
            be.import(b, y).unwrap();
            be.mul(a, b).unwrap();
            assert_eq!(be.export(a).unwrap(), oracle::mul(x, y));
        }
        be.import(a, x).unwrap();
        be.square(a, 5).unwrap();
        assert_eq!(be.export(a).unwrap(), oracle::square_n(x, 5));
        be.import(a, x).unwrap();
        be.ll_step(a, 3).unwrap();
        assert_eq!(be.export(a).unwrap(), oracle::ll_steps(x, 3));
    }
}

#[macro_export]
macro_rules! protocol_suite {
    ($mk:expr) => {
        #[test]
        fn arithmetic_matches_oracle() {
            common::arithmetic_matches_oracle($mk, common::PRIME_P);
        }
        #[test]
        fn prp_prime_and_composite_with_proofs() {
            common::prp_prime_and_composite_with_proofs($mk);
        }
        #[test]
        fn ll_prime_and_composite() {
            common::ll_prime_and_composite($mk);
        }
        #[test]
        fn prp_faults_detected_and_recovered() {
            common::prp_faults_detected_and_recovered($mk);
        }
        #[test]
        fn prp_stop_conditions() {
            common::prp_stop_conditions($mk);
        }
        #[test]
        fn prp_interrupt_and_resume() {
            common::prp_interrupt_and_resume($mk);
        }
        #[test]
        fn prp_proof_residue_write_failure() {
            common::prp_proof_residue_write_failure($mk);
        }
        #[test]
        fn ll_fault_detected_and_recovered() {
            common::ll_fault_detected_and_recovered($mk);
        }
        #[test]
        fn ll_interrupt_and_resume() {
            common::ll_interrupt_and_resume($mk);
        }
        #[test]
        fn proof_rejections() {
            common::proof_rejections($mk);
        }
    };
}
