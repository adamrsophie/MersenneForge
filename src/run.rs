// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Types shared by the LL and PRP state machines: run control, fault plans
//! for tests, errors, statistics, and the three separately recorded things a
//! run produces: process state, mathematical result, verification evidence.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::backend::{Backend, BackendError, BackendIdentity, Reg};
use crate::checkpoint::CheckpointError;
use crate::proof::ProofError;

/// Where an injected test fault lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultTarget {
    /// The working residue (PRP x_k or LL s_k).
    Data,
    /// The PRP Gerbicz check value C.
    Check,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultKind {
    /// Flip one bit of the residue.
    FlipBit(u32),
    /// Replace the residue by zero.
    Zero,
    /// PRP: x -> 3x. LL: s -> 3(s - 2) + 2, which flips the Jacobi symbol of
    /// s - 2 because (3 | N) = -1.
    Triple,
    /// Behave as if the backend reported `BackendError::Fault` here.
    BackendFault,
}

/// A fault injected right after iteration `k` completes, before any check or
/// save at `k`. Applied through `export` and `import`, so it works on every
/// backend. Fires once unless `every_pass`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultPlan {
    pub k: u64,
    pub target: FaultTarget,
    pub kind: FaultKind,
    pub every_pass: bool,
    pub fired: bool,
}

impl FaultPlan {
    pub fn once(k: u64, target: FaultTarget, kind: FaultKind) -> FaultPlan {
        FaultPlan {
            k,
            target,
            kind,
            every_pass: false,
            fired: false,
        }
    }
    pub fn every_pass(k: u64, target: FaultTarget, kind: FaultKind) -> FaultPlan {
        FaultPlan {
            k,
            target,
            kind,
            every_pass: true,
            fired: false,
        }
    }
}

/// How a run may be stopped, plus test hooks.
#[derive(Default)]
pub struct Control<'a> {
    /// Cooperative stop request (Ctrl-C). Honored at the next valid boundary.
    pub stop: Option<&'a AtomicBool>,
    /// Budget: stop at the next valid boundary after this instant.
    pub deadline: Option<Instant>,
    /// Test hook: simulate abrupt termination right after iteration k.
    /// Nothing is written at or after k.
    pub halt_at: Option<u64>,
    /// Test hook: faults to inject.
    pub faults: Vec<FaultPlan>,
    /// Test hook: the proof residue at this k fails to become durable.
    pub fail_proof_write_at: Option<u64>,
}

impl Control<'_> {
    pub fn stop_requested(&self) -> bool {
        self.stop.is_some_and(|s| s.load(Ordering::SeqCst))
            || self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// The smallest pending fault iteration above k.
    pub(crate) fn next_fault_after(&self, k: u64) -> Option<u64> {
        self.faults
            .iter()
            .filter(|f| f.k > k && (!f.fired || f.every_pass))
            .map(|f| f.k)
            .min()
    }

    /// Take the faults to apply at k (marks one-shot faults fired; the first
    /// unfired one-shot plan at k fires, so several plans at one k fire on
    /// successive passes).
    pub(crate) fn faults_at(&mut self, k: u64) -> Vec<(FaultTarget, FaultKind)> {
        let mut out = Vec::new();
        let mut one_shot_taken = false;
        for f in self.faults.iter_mut().filter(|f| f.k == k) {
            if f.every_pass {
                out.push((f.target, f.kind));
            } else if !f.fired && !one_shot_taken {
                f.fired = true;
                one_shot_taken = true;
                out.push((f.target, f.kind));
            }
        }
        out
    }
}

#[derive(Debug)]
pub enum ProtocolError {
    /// A backend failure that is not a recoverable fault.
    Backend(BackendError),
    /// Three failed checks in a row.
    TooManyErrors {
        k: u64,
        errors: u32,
    },
    /// The same wrong residue was produced twice.
    ConsistentError {
        k: u64,
    },
    /// A proof residue could not be made durable. No checkpoint was written
    /// past it; resume from the last checkpoint recomputes it.
    ProofResidueWrite {
        k: u64,
        message: String,
    },
    /// A checkpoint claims progress past a proof residue that is missing.
    ProofResidueMissing {
        k: u64,
    },
    /// A run that returns no checkpoint (proof verification) was stopped.
    Interrupted {
        k: u64,
    },
    /// A loaded checkpoint failed its on-load mathematical check.
    CheckpointRejected {
        k: u64,
    },
    Checkpoint(CheckpointError),
    Proof(ProofError),
    Config(String),
    Io(std::io::Error),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::Backend(e) => write!(f, "backend error: {e}"),
            ProtocolError::TooManyErrors { k, errors } => {
                write!(
                    f,
                    "stopped at iteration {k}: 3 failed checks in a row ({errors} errors in total)"
                )
            }
            ProtocolError::ConsistentError { k } => {
                write!(
                    f,
                    "stopped at iteration {k}: the same wrong residue was produced twice"
                )
            }
            ProtocolError::ProofResidueWrite { k, message } => {
                write!(
                    f,
                    "proof residue at iteration {k} could not be written: {message}"
                )
            }
            ProtocolError::ProofResidueMissing { k } => {
                write!(f, "proof residue at iteration {k} is missing or corrupt")
            }
            ProtocolError::Interrupted { k } => {
                write!(f, "stopped at iteration {k} before completion")
            }
            ProtocolError::CheckpointRejected { k } => {
                write!(f, "checkpoint at iteration {k} failed its on-load check")
            }
            ProtocolError::Checkpoint(e) => write!(f, "checkpoint: {e}"),
            ProtocolError::Proof(e) => write!(f, "proof: {e}"),
            ProtocolError::Config(m) => write!(f, "invalid configuration: {m}"),
            ProtocolError::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<BackendError> for ProtocolError {
    fn from(e: BackendError) -> Self {
        ProtocolError::Backend(e)
    }
}
impl From<CheckpointError> for ProtocolError {
    fn from(e: CheckpointError) -> Self {
        ProtocolError::Checkpoint(e)
    }
}
impl From<ProofError> for ProtocolError {
    fn from(e: ProofError) -> Self {
        ProtocolError::Proof(e)
    }
}
impl From<std::io::Error> for ProtocolError {
    fn from(e: std::io::Error) -> Self {
        ProtocolError::Io(e)
    }
}

/// Wall time split by activity, in seconds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    pub init: f64,
    pub iterations: f64,
    pub checks: f64,
    pub checkpoint_io: f64,
    /// PRP: export, durable write, and read back of proof points.
    #[serde(default)]
    pub proof_points: f64,
    pub proof_generation: f64,
    pub proof_verification: f64,
}

pub(crate) struct Stopwatch(Instant);
impl Stopwatch {
    pub fn start() -> Stopwatch {
        Stopwatch(Instant::now())
    }
    pub fn secs(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
}

/// Process state of a run: what happened to the computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProcessState {
    /// Every iteration ran and the final residue passed its check.
    Completed,
    /// Stopped at a valid boundary (budget or stop request). Not a result.
    Incomplete {
        k: u64,
        reason: String,
        checkpoint: Option<PathBuf>,
    },
    /// Test hook only: simulated abrupt termination.
    Halted { k: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// LL terminal residue zero: M_p is prime, given correct arithmetic.
    Prime,
    /// Base-3 PRP: 3^(2^p) = 9 mod N. Probable prime, not a proof.
    ProbablePrime,
    Composite,
}

/// The mathematical result, present only for a completed run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MathResult {
    pub test: String,
    pub exponent: u32,
    pub verdict: Verdict,
    /// LL: low 64 bits of s_(p-2). PRP: low 64 bits of the "type 1" residue
    /// x_p / 9 mod N, as upstream reports.
    pub res64: String,
    /// Low 2048 bits of the same residue, hex, most significant first.
    pub res2048: String,
    /// Trivial case decided without the recurrence (p = 2).
    pub special_case: Option<String>,
}

/// Evidence supporting the result, recorded separately from it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub backend: Option<BackendIdentity>,
    /// PRP: Gerbicz checks that passed and failed (all failures were rolled back).
    pub gerbicz_passed: u64,
    pub gerbicz_failed: u64,
    /// LL: Jacobi checks.
    pub jacobi_passed: u64,
    pub jacobi_failed: u64,
    pub jacobi_inconclusive: u64,
    /// Detected backend faults (`BackendError::Fault`), all rolled back.
    pub backend_faults: u64,
    /// Rollbacks to the last verified state.
    pub rollbacks: u64,
    pub proof: Option<ProofEvidence>,
    /// PRP: what happened to the requested proof, separately from the result.
    pub proof_stage: ProofStage,
    /// PRP: the Gerbicz check step derived from the configuration at the
    /// start (before any halving after an error). Informational output; not
    /// part of the configuration a resume compares.
    pub check_step: Option<u64>,
    pub notes: Vec<String>,
}

/// Status of the proof stage of a PRP run. The PRP result is kept whatever
/// happens here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofStage {
    /// No proof directory was given (or the test is LL).
    #[default]
    NotRequested,
    /// The proof was written; self verification was not requested.
    Generated,
    /// The proof was written and its self verification accepted it.
    Verified,
    /// The proof was written and its self verification rejected it.
    Rejected,
    /// Generation, writing, or verification failed with an error.
    Failed,
    /// Generation or verification was stopped by the budget or a stop request.
    Interrupted,
}

impl ProofStage {
    pub fn name(self) -> &'static str {
        match self {
            ProofStage::NotRequested => "not requested",
            ProofStage::Generated => "generated",
            ProofStage::Verified => "verified",
            ProofStage::Rejected => "rejected",
            ProofStage::Failed => "failed",
            ProofStage::Interrupted => "interrupted",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofEvidence {
    pub path: PathBuf,
    pub power: u32,
    /// "accepted", "rejected", or "not run".
    pub self_verification: String,
    pub hashes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunReport {
    pub process: ProcessState,
    pub result: Option<MathResult>,
    pub evidence: Evidence,
    pub timings: Timings,
}

/// Allocate `n` registers, run `f`, then free every register that was
/// allocated, on every path (a failed partial allocation included). An error
/// from `f` takes precedence over an error from `free`.
pub(crate) fn with_regs<B: Backend, T>(
    be: &mut B,
    n: usize,
    f: impl FnOnce(&mut B, &[Reg]) -> Result<T, ProtocolError>,
) -> Result<T, ProtocolError> {
    let mut regs = Vec::with_capacity(n);
    let mut alloc = Ok(());
    for _ in 0..n {
        match be.alloc() {
            Ok(r) => regs.push(r),
            Err(e) => {
                alloc = Err(ProtocolError::Backend(e));
                break;
            }
        }
    }
    let out = alloc.and_then(|()| f(be, &regs));
    let mut freed = Ok(());
    for r in regs {
        if let Err(e) = be.free(r) {
            freed = freed.and(Err(ProtocolError::Backend(e)));
        }
    }
    let v = out?;
    freed?;
    Ok(v)
}

pub(crate) fn round_up(k: u64, m: u64) -> u64 {
    k.div_ceil(m) * m
}
