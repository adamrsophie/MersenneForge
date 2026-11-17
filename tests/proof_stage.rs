// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The proof stage of a PRP run has its own status. The PRP result is kept
//! whatever happens to the proof.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::backend::{Backend, BackendIdentity, BackendResult, Reg};
use mersenne_forge::checkpoint::Checkpoint;
use mersenne_forge::proof::{ProofSchedule, ProofStore};
use mersenne_forge::residue::Residue;
use mersenne_forge::run::{Control, ProcessState, ProofStage, RunReport, Verdict};

use common::{prp_cfg, prp_resume, prp_run, PRIME_P};

fn cpu() -> CpuBackend {
    CpuBackend::new(PRIME_P).unwrap()
}

/// The PRP result is present and correct, whatever the proof stage.
fn assert_result_kept(r: &RunReport) {
    assert_eq!(r.process, ProcessState::Completed);
    let res = r.result.as_ref().unwrap();
    assert_eq!(res.verdict, Verdict::ProbablePrime);
    assert_eq!(res.res64, common::oracle_prp_res64(PRIME_P));
}

#[test]
fn not_requested_generated_and_verified() {
    let dir = tempfile::tempdir().unwrap();
    let r = prp_run(
        &mut cpu(),
        PRIME_P,
        &prp_cfg(dir.path(), None),
        &mut Control::default(),
    )
    .unwrap();
    assert_result_kept(&r);
    assert_eq!(r.evidence.proof_stage, ProofStage::NotRequested);

    let dir = tempfile::tempdir().unwrap();
    let mut cfg = prp_cfg(dir.path(), Some(4));
    cfg.verify_proof = false;
    let r = prp_run(&mut cpu(), PRIME_P, &cfg, &mut Control::default()).unwrap();
    assert_result_kept(&r);
    assert_eq!(r.evidence.proof_stage, ProofStage::Generated);
    assert!(r.evidence.proof.unwrap().path.exists());

    let dir = tempfile::tempdir().unwrap();
    let r = prp_run(
        &mut cpu(),
        PRIME_P,
        &prp_cfg(dir.path(), Some(4)),
        &mut Control::default(),
    )
    .unwrap();
    assert_result_kept(&r);
    assert_eq!(r.evidence.proof_stage, ProofStage::Verified);
}

#[test]
fn a_damaged_stored_residue_gives_a_rejected_proof() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = prp_cfg(dir.path(), Some(5));
    let mut ctl = Control {
        halt_at: Some(1000),
        ..Control::default()
    };
    prp_run(&mut cpu(), PRIME_P, &cfg, &mut ctl).unwrap();
    // The checkpoint is at 960, so the residue at 640 is not recomputed; a
    // different, intact value there makes the generated proof wrong.
    let store = ProofStore {
        schedule: ProofSchedule::new(PRIME_P, 5).unwrap(),
        dir: ProofStore::residue_dir(cfg.proof_dir.as_ref().unwrap(), PRIME_P),
    };
    let x = store.load(640).unwrap();
    store.save(640, &x.add_small(1)).unwrap();
    let r = prp_resume(&mut cpu(), PRIME_P, &cfg, &mut Control::default()).unwrap();
    assert_result_kept(&r);
    assert_eq!(r.evidence.proof_stage, ProofStage::Rejected);
    assert_eq!(r.evidence.proof.unwrap().self_verification, "rejected");
}

#[test]
fn an_unwritable_proof_file_is_a_failed_stage() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = prp_cfg(dir.path(), Some(4));
    // A directory where the proof file goes makes the final rename fail.
    std::fs::create_dir_all(cfg.proof_dir.as_ref().unwrap().join("1279-4.proof")).unwrap();
    let r = prp_run(&mut cpu(), PRIME_P, &cfg, &mut Control::default()).unwrap();
    assert_result_kept(&r);
    assert_eq!(r.evidence.proof_stage, ProofStage::Failed);
    assert!(r.evidence.proof.is_none());
}

static STOP: AtomicBool = AtomicBool::new(false);

/// Requests a stop when the PRP engine frees its registers, which happens
/// after the final check and before the proof stage.
struct StopAfterEngine(CpuBackend);

impl Backend for StopAfterEngine {
    fn exponent(&self) -> u32 {
        self.0.exponent()
    }
    fn identity(&self) -> BackendIdentity {
        self.0.identity()
    }
    fn alloc(&mut self) -> BackendResult<Reg> {
        self.0.alloc()
    }
    fn free(&mut self, r: Reg) -> BackendResult<()> {
        STOP.store(true, Ordering::SeqCst);
        self.0.free(r)
    }
    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        self.0.import(r, x)
    }
    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        self.0.export(r)
    }
    fn copy(&mut self, d: Reg, s: Reg) -> BackendResult<()> {
        self.0.copy(d, s)
    }
    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.0.square(r, n)
    }
    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.0.ll_step(r, n)
    }
    fn mul(&mut self, d: Reg, s: Reg) -> BackendResult<()> {
        self.0.mul(d, s)
    }
    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        self.0.equal(a, b)
    }
    fn sync(&mut self) -> BackendResult<()> {
        self.0.sync()
    }
}

#[test]
fn a_stop_during_the_proof_stage_interrupts_it() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = prp_cfg(dir.path(), Some(5));
    let mut ctl = Control {
        stop: Some(&STOP),
        ..Control::default()
    };
    let r = prp_run(&mut StopAfterEngine(cpu()), PRIME_P, &cfg, &mut ctl).unwrap();
    assert_result_kept(&r);
    assert_eq!(r.evidence.proof_stage, ProofStage::Interrupted);
    assert!(r.evidence.proof.is_none());
    let ck = Checkpoint::load_file(cfg.checkpoint.as_ref().unwrap()).unwrap();
    assert_eq!(ck.meta.state, "completed");
    assert_eq!(ck.meta.k, 1280);
    assert_eq!(
        ck.residues[2],
        mersenne_forge::oracle::prp(PRIME_P).terminal
    );

    let resumed = prp_resume(&mut cpu(), PRIME_P, &cfg, &mut Control::default()).unwrap();
    assert_result_kept(&resumed);
    assert_eq!(resumed.evidence.proof_stage, ProofStage::Verified);
    assert_eq!(resumed.evidence.gerbicz_passed, 0);
    assert_eq!(resumed.timings.iterations, 0.0);
}
