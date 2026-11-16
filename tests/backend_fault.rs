// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! A backend that reports `BackendError::Fault` late (from `equal`, `export`,
//! or `ll_step`), as a transform backend does when its roundoff check trips.
//! The state machines must roll back and finish correctly, and a fault that
//! never clears must end in an execution error, never in "composite". Every
//! register is freed on the error paths too.
//!
//! A second wrapper corrupts only the host copy of the terminal residue,
//! leaving the resident value intact: the round trip after that export must
//! catch it.

mod common;

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::backend::{Backend, BackendError, BackendIdentity, BackendResult, Reg};
use mersenne_forge::ll::run_ll;
use mersenne_forge::oracle;
use mersenne_forge::proof::{self, Proof};
use mersenne_forge::residue::Residue;
use mersenne_forge::run::{Control, ProtocolError, Verdict};

struct Flaky {
    inner: CpuBackend,
    calls: u64,
    fail_at: Vec<u64>,
    fail_always_after: Option<u64>,
    /// Registers allocated and not yet freed.
    live: i64,
}

impl Flaky {
    fn tick(&mut self) -> BackendResult<()> {
        self.calls += 1;
        if self.fail_at.contains(&self.calls)
            || self.fail_always_after.is_some_and(|n| self.calls > n)
        {
            return Err(BackendError::Fault("simulated roundoff check".into()));
        }
        Ok(())
    }
}

impl Backend for Flaky {
    fn exponent(&self) -> u32 {
        self.inner.exponent()
    }
    fn identity(&self) -> BackendIdentity {
        self.inner.identity()
    }
    fn alloc(&mut self) -> BackendResult<Reg> {
        let r = self.inner.alloc()?;
        self.live += 1;
        Ok(r)
    }
    fn free(&mut self, r: Reg) -> BackendResult<()> {
        self.inner.free(r)?;
        self.live -= 1;
        Ok(())
    }
    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        self.inner.import(r, x)
    }
    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        self.tick()?;
        self.inner.export(r)
    }
    fn copy(&mut self, d: Reg, s: Reg) -> BackendResult<()> {
        self.inner.copy(d, s)
    }
    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.inner.square(r, n)
    }
    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.tick()?;
        self.inner.ll_step(r, n)
    }
    fn mul(&mut self, d: Reg, s: Reg) -> BackendResult<()> {
        self.inner.mul(d, s)
    }
    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        self.tick()?;
        self.inner.equal(a, b)
    }
    fn sync(&mut self) -> BackendResult<()> {
        self.inner.sync()
    }
}

fn flaky(p: u32, fail_at: Vec<u64>, always: Option<u64>) -> Flaky {
    Flaky {
        inner: CpuBackend::new(p).unwrap(),
        calls: 0,
        fail_at,
        fail_always_after: always,
        live: 0,
    }
}

#[test]
fn late_faults_roll_back_and_finish() {
    let p = common::PRIME_P;
    let dir = tempfile::tempdir().unwrap();
    let mut be = flaky(p, vec![1, 25, 40], None);
    let r = common::prp_run(
        &mut be,
        p,
        &common::prp_cfg(dir.path(), Some(4)),
        &mut Control::default(),
    )
    .unwrap();
    assert_eq!(r.evidence.backend_faults, 3);
    assert_eq!(r.result.unwrap().res64, common::oracle_prp_res64(p));

    let dir = tempfile::tempdir().unwrap();
    let mut be = flaky(p, vec![5, 9], None);
    let r = run_ll(
        &mut be,
        p,
        &common::ll_cfg(dir.path()),
        serde_json::json!({}),
        None,
        &mut Control::default(),
    )
    .unwrap();
    assert_eq!(r.evidence.backend_faults, 2);
    assert_eq!(r.result.unwrap().verdict, Verdict::Prime);
}

#[test]
fn persistent_faults_are_an_execution_error() {
    let p = common::PRIME_P;
    let dir = tempfile::tempdir().unwrap();
    let mut be = flaky(p, vec![], Some(20));
    let e = common::prp_run(
        &mut be,
        p,
        &common::prp_cfg(dir.path(), None),
        &mut Control::default(),
    )
    .unwrap_err();
    assert!(matches!(e, ProtocolError::TooManyErrors { .. }), "{e}");
    assert_eq!(be.live, 0, "PRP registers leaked");
    let mut be = flaky(p, vec![], Some(20));
    let e = run_ll(
        &mut be,
        p,
        &common::ll_cfg(dir.path()),
        serde_json::json!({}),
        None,
        &mut Control::default(),
    )
    .unwrap_err();
    assert!(matches!(e, ProtocolError::TooManyErrors { .. }), "{e}");
    assert_eq!(be.live, 0, "LL registers leaked");
}

#[test]
fn proof_errors_free_their_registers() {
    let p = common::PRIME_P;
    let dir = tempfile::tempdir().unwrap();
    let r = common::prp_run(
        &mut CpuBackend::new(p).unwrap(),
        p,
        &common::prp_cfg(dir.path(), Some(3)),
        &mut Control::default(),
    )
    .unwrap();
    let pr = Proof::read(&r.evidence.proof.unwrap().path, Some(p)).unwrap();
    for verify in [proof::verify::<Flaky>, proof::verify_equation::<Flaky>] {
        let mut be = flaky(p, vec![], Some(0));
        // Every export fails: an execution error, never an acceptance.
        assert!(verify(&mut be, &pr, &mut Control::default()).is_err());
        assert_eq!(be.live, 0, "verification registers leaked");
    }
}

/// Corrupts the first export whose value is `target`; the resident value
/// stays correct.
struct CorruptExport {
    inner: CpuBackend,
    target: Residue,
    fired: bool,
}

impl Backend for CorruptExport {
    fn exponent(&self) -> u32 {
        self.inner.exponent()
    }
    fn identity(&self) -> BackendIdentity {
        self.inner.identity()
    }
    fn alloc(&mut self) -> BackendResult<Reg> {
        self.inner.alloc()
    }
    fn free(&mut self, r: Reg) -> BackendResult<()> {
        self.inner.free(r)
    }
    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        self.inner.import(r, x)
    }
    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        let v = self.inner.export(r)?;
        if !self.fired && v == self.target {
            self.fired = true;
            return Ok(v.with_bit_flipped(0));
        }
        Ok(v)
    }
    fn copy(&mut self, d: Reg, s: Reg) -> BackendResult<()> {
        self.inner.copy(d, s)
    }
    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.inner.square(r, n)
    }
    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.inner.ll_step(r, n)
    }
    fn mul(&mut self, d: Reg, s: Reg) -> BackendResult<()> {
        self.inner.mul(d, s)
    }
    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        self.inner.equal(a, b)
    }
    fn sync(&mut self) -> BackendResult<()> {
        self.inner.sync()
    }
}

fn corrupt_export(p: u32, target: Residue) -> CorruptExport {
    CorruptExport {
        inner: CpuBackend::new(p).unwrap(),
        target,
        fired: false,
    }
}

#[test]
fn corrupted_terminal_export_is_caught() {
    let p = common::PRIME_P;
    // PRP: x_p = 9. Its export is the only export of that value in this run.
    let dir = tempfile::tempdir().unwrap();
    let mut be = corrupt_export(p, oracle::prp(p).terminal);
    let r = common::prp_run(
        &mut be,
        p,
        &common::prp_cfg(dir.path(), None),
        &mut Control::default(),
    )
    .unwrap();
    assert!(be.fired);
    assert_eq!(r.evidence.backend_faults, 1, "{:?}", r.evidence);
    assert!(r.evidence.rollbacks >= 1);
    let res = r.result.unwrap();
    assert_eq!(res.verdict, Verdict::ProbablePrime);
    assert_eq!(res.res64, common::oracle_prp_res64(p));

    // LL: s_(p-2) = 0. The corrupted copy 1 passes the Jacobi check
    // ((1 - 2 | N) = -1), so only the round trip can catch it.
    let dir = tempfile::tempdir().unwrap();
    let mut be = corrupt_export(p, oracle::ll(p).terminal);
    let r = run_ll(
        &mut be,
        p,
        &common::ll_cfg(dir.path()),
        serde_json::json!({}),
        None,
        &mut Control::default(),
    )
    .unwrap();
    assert!(be.fired);
    assert_eq!(r.evidence.backend_faults, 1, "{:?}", r.evidence);
    assert_eq!(r.result.unwrap().verdict, Verdict::Prime);
}
