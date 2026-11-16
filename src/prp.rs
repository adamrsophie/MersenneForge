// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Base-3 PRP with the Gerbicz check, written once over [`Backend`].
//!
//! Semantics follow PRPLL (`Gpu::isPrimePRP`, `Gpu::doCheck`):
//!
//! - x_0 = A (A = 3 for PRP), x_(k+1) = x_k^2. Probable prime iff x_p = 9.
//! - Check value C, block size B: before squaring at every k that is a
//!   multiple of B, C <- C * x_k (C starts at 1, so C = x_0 after k = 0).
//! - Check at a multiple k of B, with C_old the value before the update at k:
//!   C_old^(2^B) * A == C_old * x_k. The check itself performs the update
//!   C <- C_old * x_k, so the update at the start of the next block is
//!   skipped after a passed check.
//! - Checks run at multiples of the check step, at start + 2B (after every
//!   load), at a stop request, and at roundUp(p, B): the run continues past p
//!   so the final residue is covered by a check.
//! - A failed check (or a detected backend fault) rolls back to the last
//!   verified state. Three failures in a row, or the same wrong residue twice,
//!   stop the run with an error. After any error the check step is halved.
//!
//! A verified state is (k, C_old, x_k). Both are stored, so loading needs no
//! rebuild; the loader runs the check equation on the pair, which is a
//! mathematical verification of the loaded state and leaves C updated.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::backend::{Backend, BackendError, Reg};
use crate::checkpoint::{Checkpoint, CheckpointError, CheckpointMeta, FORMAT_VERSION};
use crate::proof::{self, ProofSchedule, ProofStore};
use crate::residue::Residue;
use crate::run::{
    round_up, Control, Evidence, FaultKind, FaultTarget, MathResult, ProcessState, ProofEvidence,
    ProtocolError, RunReport, Stopwatch, Timings, Verdict,
};

pub const PROTOCOL: &str = "prp-base3-gerbicz/1";

/// Block sizes upstream knows, with their base check steps.
pub fn upstream_check_step(block: u32) -> Option<u64> {
    match block {
        200 => Some(40_000),
        400 => Some(160_000),
        500 => Some(200_000),
        1000 => Some(1_000_000),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrpConfig {
    /// Gerbicz block size B. Upstream compatible set: 200, 400, 500, 1000.
    pub block_size: u32,
    /// Explicit check step (multiple of B). None: upstream rule, scaled down
    /// for small exponents.
    pub check_step: Option<u64>,
    /// Proof power; None with a proof directory means the upstream default.
    pub proof_power: Option<u32>,
    pub proof_dir: Option<PathBuf>,
    /// Verify the generated proof before reporting.
    pub verify_proof: bool,
    pub checkpoint: Option<PathBuf>,
    /// Largest number of squarings per backend call.
    pub max_batch: u64,
}

impl Default for PrpConfig {
    fn default() -> Self {
        PrpConfig {
            block_size: 1000,
            check_step: None,
            proof_power: None,
            proof_dir: None,
            verify_proof: true,
            checkpoint: None,
            max_batch: 1000,
        }
    }
}

/// (k, C_old, x_k): C_old excludes x_k.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedState {
    pub k: u64,
    pub c_old: Residue,
    pub x: Residue,
}

/// How the check step is chosen.
#[derive(Debug, Clone, Copy)]
struct StepRule {
    base: u64,
    /// Scale down as upstream when base > n / 8.
    scale_for: Option<u64>,
}

impl StepRule {
    fn effective(&self, block: u64, n_errors: u32) -> u64 {
        let mut s = if n_errors > 0 {
            self.base / 2
        } else {
            self.base
        };
        s = (s / block * block).max(block);
        if let Some(n) = self.scale_for {
            if s > n / 8 {
                let log = if 20_000 % block == 0 { 20_000 } else { block };
                s = log.max(s.min(n / 8) / log * log);
            }
        }
        s
    }
}

struct Spec<'h> {
    block: u64,
    kend: u64,
    rule: StepRule,
    max_batch: u64,
    proof: Option<&'h ProofStore>,
}

#[derive(Default)]
struct Counters {
    passed: u64,
    failed: u64,
    faults: u64,
    rollbacks: u64,
    n_errors: u32,
    t_iter: f64,
    t_check: f64,
    t_ckpt: f64,
}

enum End {
    Done { final_x: Residue },
    Stopped { state: VerifiedState },
    Halted { k: u64 },
}

enum StepErr {
    Fault(String),
    Fatal(ProtocolError),
}

impl From<BackendError> for StepErr {
    fn from(e: BackendError) -> Self {
        match e {
            BackendError::Fault(m) => StepErr::Fault(m),
            other => StepErr::Fatal(ProtocolError::Backend(other)),
        }
    }
}

enum Step {
    Continue,
    Halted,
    Checked {
        ok: bool,
        state: VerifiedState,
        stop: bool,
    },
}

struct Engine<'b, B: Backend> {
    be: &'b mut B,
    x: Reg,
    c: Reg,
    a: Reg,
    t: Reg,
    k: u64,
    skip_update: bool,
    early: u64,
    step: u64,
    final_x: Option<Residue>,
}

fn apply_fault<B: Backend>(be: &mut B, r: Reg, kind: FaultKind, ll: bool) -> Result<(), StepErr> {
    let v = be.export(r)?;
    let w = match kind {
        FaultKind::FlipBit(b) => v.with_bit_flipped(b),
        FaultKind::Zero => Residue::zero(v.exponent()),
        FaultKind::Triple if ll => v.minus_small(2).mul_small(3).add_small(2),
        FaultKind::Triple => v.mul_small(3),
        FaultKind::BackendFault => return Err(StepErr::Fault("injected fault".into())),
    };
    be.import(r, &w)?;
    Ok(())
}

/// LL faults share the injection code.
pub(crate) fn inject<B: Backend>(
    be: &mut B,
    r: Reg,
    kind: FaultKind,
    ll: bool,
) -> Result<(), BackendError> {
    match apply_fault(be, r, kind, ll) {
        Ok(()) => Ok(()),
        Err(StepErr::Fault(m)) => Err(BackendError::Fault(m)),
        Err(StepErr::Fatal(ProtocolError::Backend(e))) => Err(e),
        Err(StepErr::Fatal(e)) => Err(BackendError::Fault(e.to_string())),
    }
}

impl<'b, B: Backend> Engine<'b, B> {
    fn square_batched(&mut self, r: Reg, n: u64, max_batch: u64) -> Result<(), BackendError> {
        let mut left = n;
        while left > 0 {
            let m = left.min(max_batch);
            self.be.square(r, m)?;
            left -= m;
        }
        Ok(())
    }

    /// The Gerbicz check at the current k. Leaves C = C_old * x_k.
    fn check(&mut self, spec: &Spec) -> Result<(bool, Residue), BackendError> {
        let c_old = self.be.export(self.c)?;
        self.be.copy(self.t, self.c)?;
        self.square_batched(self.t, spec.block, spec.max_batch)?;
        self.be.mul(self.t, self.a)?;
        self.be.mul(self.c, self.x)?;
        let eq = self.be.equal(self.t, self.c)?;
        Ok((eq && !c_old.is_zero(), c_old))
    }

    /// Load a verified state and run the check equation on it.
    fn load(&mut self, st: &VerifiedState, spec: &Spec) -> Result<bool, BackendError> {
        self.be.import(self.c, &st.c_old)?;
        self.be.import(self.x, &st.x)?;
        let (ok, _) = self.check(spec)?;
        self.k = st.k;
        self.skip_update = true;
        self.early = st.k + 2 * spec.block;
        self.final_x = None;
        Ok(ok && !st.x.is_zero())
    }

    fn advance(
        &mut self,
        spec: &Spec,
        ctl: &mut Control,
        cnt: &mut Counters,
    ) -> Result<Step, StepErr> {
        let b = spec.block;
        let k = self.k;
        if k % b == 0 {
            if self.skip_update {
                self.skip_update = false;
            } else {
                self.be.mul(self.c, self.x)?;
            }
        }
        let mut next = round_up(k + 1, b).min(k + spec.max_batch);
        if k < spec.kend {
            next = next.min(spec.kend);
        }
        if let Some(pt) = spec.proof.and_then(|s| s.schedule.next_after(k)) {
            next = next.min(pt);
        }
        if let Some(h) = ctl.halt_at.filter(|&h| h > k) {
            next = next.min(h);
        }
        if let Some(f) = ctl.next_fault_after(k) {
            next = next.min(f);
        }
        let sw = Stopwatch::start();
        self.be.square(self.x, next - k)?;
        // Synchronize so queued work is timed here, once per segment (at
        // most max_batch squarings), never per iteration.
        self.be.sync()?;
        cnt.t_iter += sw.secs();
        self.k = next;
        let k = next;

        for (target, kind) in ctl.faults_at(k) {
            let r = if target == FaultTarget::Data {
                self.x
            } else {
                self.c
            };
            apply_fault(self.be, r, kind, false)?;
        }
        if let Some(store) = spec.proof.filter(|s| s.schedule.contains(k)) {
            let v = self.be.export(self.x)?;
            if v.is_zero() {
                return Err(StepErr::Fault(format!("zero residue at proof point {k}")));
            }
            if ctl.fail_proof_write_at == Some(k) {
                return Err(StepErr::Fatal(ProtocolError::ProofResidueWrite {
                    k,
                    message: "simulated write failure".into(),
                }));
            }
            store.save(k, &v).map_err(|e| {
                StepErr::Fatal(ProtocolError::ProofResidueWrite {
                    k,
                    message: e.to_string(),
                })
            })?;
        }
        if k == spec.kend {
            self.final_x = Some(self.be.export(self.x)?);
        }
        if ctl.halt_at == Some(k) {
            return Ok(Step::Halted);
        }
        if k % b != 0 {
            return Ok(Step::Continue);
        }
        let stop = ctl.stop_requested();
        let kendend = round_up(spec.kend, b);
        if !(stop || k % self.step == 0 || k >= kendend || k == self.early) {
            return Ok(Step::Continue);
        }
        let sw = Stopwatch::start();
        let (ok, c_old) = self.check(spec)?;
        // The check performed C <- C_old * x_k: skip the next block update.
        self.skip_update = true;
        let x = self.be.export(self.x)?;
        cnt.t_check += sw.secs();
        Ok(Step::Checked {
            ok: ok && !x.is_zero(),
            state: VerifiedState { k, c_old, x },
            stop,
        })
    }
}

type OnVerified<'f> = &'f mut dyn FnMut(&VerifiedState, u32) -> Result<(), ProtocolError>;

/// The checked squaring run with x_0 = A, up to roundUp(n, B).
#[allow(clippy::too_many_arguments)]
fn run_engine<B: Backend>(
    be: &mut B,
    a: &Residue,
    from: VerifiedState,
    n_errors: u32,
    spec: &Spec,
    ctl: &mut Control,
    on_verified: Option<OnVerified>,
    cnt: &mut Counters,
) -> Result<End, ProtocolError> {
    let regs = [be.alloc()?, be.alloc()?, be.alloc()?, be.alloc()?];
    be.import(regs[2], a)?;
    let mut eng = Engine {
        be,
        x: regs[0],
        c: regs[1],
        a: regs[2],
        t: regs[3],
        k: from.k,
        skip_update: true,
        early: 0,
        step: spec.rule.effective(spec.block, n_errors),
        final_x: None,
    };
    let out = drive(&mut eng, from, n_errors, spec, ctl, on_verified, cnt);
    for r in regs {
        eng.be.free(r)?;
    }
    out
}

fn drive<B: Backend>(
    eng: &mut Engine<B>,
    from: VerifiedState,
    n_errors: u32,
    spec: &Spec,
    ctl: &mut Control,
    mut on_verified: Option<OnVerified>,
    cnt: &mut Counters,
) -> Result<End, ProtocolError> {
    cnt.n_errors = n_errors;
    let mut verified = from;
    // Load the starting state. A failed on-load check is retried once (it may
    // be a transient arithmetic error) before the state is rejected; a
    // detected backend fault is retried up to three times.
    let (mut check_failures, mut faults) = (0, 0);
    loop {
        match eng.load(&verified, spec) {
            Ok(true) => break,
            Ok(false) => {
                check_failures += 1;
                if check_failures >= 2 {
                    return Err(ProtocolError::CheckpointRejected { k: verified.k });
                }
            }
            Err(BackendError::Fault(_)) => {
                cnt.faults += 1;
                faults += 1;
                if faults >= 3 {
                    return Err(ProtocolError::TooManyErrors {
                        k: verified.k,
                        errors: faults,
                    });
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
    let mut n_seq = 0u32;
    let mut last_failed: Option<Residue> = None;

    loop {
        let failure: Option<Residue>;
        let stop;
        match eng.advance(spec, ctl, cnt) {
            Ok(Step::Continue) => continue,
            Ok(Step::Halted) => return Ok(End::Halted { k: eng.k }),
            Ok(Step::Checked {
                ok: true,
                state,
                stop,
            }) => {
                n_seq = 0;
                cnt.passed += 1;
                verified = state;
                if verified.k >= round_up(spec.kend, spec.block) {
                    let final_x = eng
                        .final_x
                        .clone()
                        .expect("the final residue was recorded at k = n");
                    return Ok(End::Done { final_x });
                }
                if let Some(f) = on_verified.as_mut() {
                    let sw = Stopwatch::start();
                    f(&verified, cnt.n_errors)?;
                    cnt.t_ckpt += sw.secs();
                }
                if stop {
                    return Ok(End::Stopped { state: verified });
                }
                continue;
            }
            Ok(Step::Checked {
                ok: false,
                state,
                stop: s,
            }) => {
                cnt.failed += 1;
                failure = Some(state.x);
                stop = s;
            }
            Err(StepErr::Fault(_)) => {
                cnt.faults += 1;
                failure = None;
                stop = ctl.stop_requested();
            }
            Err(StepErr::Fatal(e)) => return Err(e),
        }
        // Failure accounting, then roll back.
        let mut fail =
            |res: Option<Residue>, k: u64, cnt: &mut Counters| -> Result<(), ProtocolError> {
                cnt.n_errors += 1;
                n_seq += 1;
                if n_seq > 2 {
                    return Err(ProtocolError::TooManyErrors {
                        k,
                        errors: cnt.n_errors,
                    });
                }
                if res.is_some() && res == last_failed {
                    return Err(ProtocolError::ConsistentError { k });
                }
                if res.is_some() {
                    last_failed = res;
                }
                Ok(())
            };
        fail(failure, eng.k, cnt)?;
        if stop {
            return Ok(End::Stopped { state: verified });
        }
        loop {
            cnt.rollbacks += 1;
            match eng.load(&verified, spec) {
                Ok(true) => break,
                Ok(false) | Err(BackendError::Fault(_)) => fail(None, verified.k, cnt)?,
                Err(e) => return Err(e.into()),
            }
        }
        eng.step = spec.rule.effective(spec.block, cnt.n_errors);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CheckedPowerConfig {
    pub block: u32,
    pub check_step: u64,
    pub max_batch: u64,
}

impl CheckedPowerConfig {
    /// About ten checks per run, blocks of about sqrt(n / 10).
    pub fn for_length(n: u64) -> CheckedPowerConfig {
        let block = ((n as f64 / 10.0).sqrt().round() as u64).clamp(2, 1000);
        let check_step = round_up((n / 10).max(1), block);
        CheckedPowerConfig {
            block: block as u32,
            check_step,
            max_batch: 1000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckedPower {
    pub residue: Residue,
    pub passed: u64,
    pub failed: u64,
}

/// a^(2^n) with every squaring covered by the generic Gerbicz check
/// (start value a instead of 3). Used by proof verification.
pub fn checked_power<B: Backend>(
    be: &mut B,
    a: &Residue,
    n: u64,
    cfg: &CheckedPowerConfig,
    ctl: &mut Control,
) -> Result<CheckedPower, ProtocolError> {
    if a.is_zero() {
        return Ok(CheckedPower {
            residue: a.clone(),
            passed: 0,
            failed: 0,
        });
    }
    let p = a.exponent();
    let spec = Spec {
        block: u64::from(cfg.block),
        kend: n,
        rule: StepRule {
            base: cfg.check_step,
            scale_for: None,
        },
        max_batch: cfg.max_batch,
        proof: None,
    };
    let from = VerifiedState {
        k: 0,
        c_old: Residue::from_u64(p, 1),
        x: a.clone(),
    };
    let mut cnt = Counters::default();
    match run_engine(be, a, from, 0, &spec, ctl, None, &mut cnt)? {
        End::Done { final_x, .. } => Ok(CheckedPower {
            residue: final_x,
            passed: cnt.passed,
            failed: cnt.failed,
        }),
        End::Stopped { state } => Err(ProtocolError::Interrupted { k: state.k }),
        End::Halted { k } => Err(ProtocolError::Interrupted { k }),
    }
}

fn validate(p: u32, cfg: &PrpConfig) -> Result<Option<ProofSchedule>, ProtocolError> {
    crate::exponent::validate(p).map_err(|e| ProtocolError::Config(e.to_string()))?;
    if cfg.block_size < 2 {
        return Err(ProtocolError::Config(
            "block size must be at least 2".into(),
        ));
    }
    if cfg.max_batch == 0 {
        return Err(ProtocolError::Config("max batch must be positive".into()));
    }
    match cfg.check_step {
        Some(s) if s == 0 || s % u64::from(cfg.block_size) != 0 => {
            return Err(ProtocolError::Config(format!(
                "check step {s} is not a positive multiple of the block size"
            )));
        }
        None if upstream_check_step(cfg.block_size).is_none() => {
            return Err(ProtocolError::Config(format!(
                "block size {} is outside the upstream set (200, 400, 500, 1000); give an explicit check step",
                cfg.block_size
            )));
        }
        _ => {}
    }
    match (&cfg.proof_dir, cfg.proof_power) {
        (None, Some(_)) => Err(ProtocolError::Config(
            "a proof power needs a proof directory".into(),
        )),
        (None, None) => Ok(None),
        (Some(_), power) if p > 2 => Ok(Some(ProofSchedule::new(
            p,
            power.unwrap_or_else(|| proof::default_power(p)),
        )?)),
        (Some(_), _) => Err(ProtocolError::Config("no proof exists for p = 2".into())),
    }
}

/// Effective configuration after defaults (the proof power is filled in).
pub fn effective_config(p: u32, cfg: &PrpConfig) -> Result<PrpConfig, ProtocolError> {
    let sched = validate(p, cfg)?;
    let mut eff = cfg.clone();
    eff.proof_power = sched.map(|s| s.power());
    Ok(eff)
}

fn final_result(p: u32, x: &Residue) -> MathResult {
    let pp = *x == Residue::from_u64(p, 9);
    let type1 = x.div3().div3();
    MathResult {
        test: "prp".into(),
        exponent: p,
        verdict: if pp {
            Verdict::ProbablePrime
        } else {
            Verdict::Composite
        },
        res64: format!("{:016x}", type1.res64()),
        res2048: type1.res2048_hex(),
        special_case: None,
    }
}

/// Run (or resume) a base-3 PRP test of M_p.
pub fn run_prp<B: Backend>(
    be: &mut B,
    p: u32,
    cfg: &PrpConfig,
    requested: serde_json::Value,
    resume: Option<&Checkpoint>,
    ctl: &mut Control,
) -> Result<RunReport, ProtocolError> {
    let init = Stopwatch::start();
    let mut timings = Timings::default();
    let mut evidence = Evidence {
        backend: Some(be.identity()),
        ..Evidence::default()
    };
    if be.exponent() != p {
        return Err(ProtocolError::Config(format!(
            "backend is for M{}, run is for M{p}",
            be.exponent()
        )));
    }
    let schedule = validate(p, cfg)?;
    if p == 2 {
        evidence
            .notes
            .push("M2 = 3 is prime by inspection; base 3 is not coprime to N".into());
        return Ok(RunReport {
            process: ProcessState::Completed,
            result: Some(MathResult {
                test: "prp".into(),
                exponent: 2,
                verdict: Verdict::Prime,
                res64: format!("{:016x}", 0),
                res2048: Residue::zero(2).res2048_hex(),
                special_case: Some("p = 2".into()),
            }),
            evidence,
            timings,
        });
    }
    let store = match (&schedule, &cfg.proof_dir) {
        (Some(s), Some(dir)) => Some(ProofStore::open(dir, s.clone())?),
        _ => None,
    };
    let eff = effective_config(p, cfg)?;
    let config_json = serde_json::to_value(&eff).expect("config serializes");

    let (from, n_errors, elapsed_before) = match resume {
        None => (
            VerifiedState {
                k: 0,
                c_old: Residue::from_u64(p, 1),
                x: Residue::from_u64(p, 3),
            },
            0,
            0.0,
        ),
        Some(ck) => {
            let m = &ck.meta;
            if m.protocol != PROTOCOL
                || m.exponent != p
                || ck.residues.len() != 2
                || m.k >= u64::from(p)
                || m.k % u64::from(cfg.block_size) != 0
            {
                return Err(CheckpointError::Mismatch(format!(
                    "not a {PROTOCOL} checkpoint for M{p}"
                ))
                .into());
            }
            if m.backend != be.identity() {
                return Err(CheckpointError::Mismatch(format!(
                    "written by backend {:?}, resuming on {:?}",
                    m.backend,
                    be.identity()
                ))
                .into());
            }
            if without_checkpoint(&m.config) != without_checkpoint(&config_json) {
                return Err(CheckpointError::Mismatch("run configuration differs".into()).into());
            }
            if let Some(s) = &store {
                s.check_through(m.k)?;
            }
            let st = VerifiedState {
                k: m.k,
                c_old: ck.residues[0].clone(),
                x: ck.residues[1].clone(),
            };
            (st, m.n_errors, m.elapsed_secs)
        }
    };

    let spec = Spec {
        block: u64::from(cfg.block_size),
        kend: u64::from(p),
        rule: match cfg.check_step {
            Some(s) => StepRule {
                base: s,
                scale_for: None,
            },
            None => StepRule {
                base: upstream_check_step(cfg.block_size).expect("validated"),
                scale_for: Some(u64::from(p)),
            },
        },
        max_batch: cfg.max_batch,
        proof: store.as_ref(),
    };
    let start = Stopwatch::start();
    let identity = be.identity();
    let ckpt_path = cfg.checkpoint.clone();
    let mut save = |st: &VerifiedState, n_errors: u32| -> Result<(), ProtocolError> {
        let Some(path) = &ckpt_path else {
            return Ok(());
        };
        let ck = Checkpoint {
            meta: CheckpointMeta {
                format_version: FORMAT_VERSION,
                protocol: PROTOCOL.into(),
                exponent: p,
                k: st.k,
                state: "verified".into(),
                verified_k: st.k,
                n_errors,
                proof_durable_through: spec_proof_through(store.as_ref(), st.k),
                backend: identity.clone(),
                config: config_json.clone(),
                requested: requested.clone(),
                elapsed_secs: elapsed_before + start.secs(),
                crate_version: env!("CARGO_PKG_VERSION").into(),
            },
            residues: vec![st.c_old.clone(), st.x.clone()],
        };
        ck.save(path)?;
        Ok(())
    };
    timings.init = init.secs();
    let mut cnt = Counters::default();
    let a = Residue::from_u64(p, 3);
    let end = run_engine(
        be,
        &a,
        from,
        n_errors,
        &spec,
        ctl,
        Some(&mut save),
        &mut cnt,
    )?;
    timings.iterations = cnt.t_iter;
    timings.checks = cnt.t_check;
    timings.checkpoint_io = cnt.t_ckpt;
    evidence.gerbicz_passed = cnt.passed;
    evidence.gerbicz_failed = cnt.failed;
    evidence.backend_faults = cnt.faults;
    evidence.rollbacks = cnt.rollbacks;

    let final_x = match end {
        End::Halted { k } => {
            return Ok(RunReport {
                process: ProcessState::Halted { k },
                result: None,
                evidence,
                timings,
            })
        }
        End::Stopped { state } => {
            return Ok(RunReport {
                process: ProcessState::Incomplete {
                    k: state.k,
                    reason: "stop requested or budget exhausted".into(),
                    checkpoint: cfg.checkpoint.clone(),
                },
                result: None,
                evidence,
                timings,
            })
        }
        End::Done { final_x, .. } => final_x,
    };
    let result = final_result(p, &final_x);
    if let Some(store) = &store {
        let sw = Stopwatch::start();
        let generated = proof::generate(be, store).and_then(|pr| {
            let path = pr.write(cfg.proof_dir.as_deref().expect("store implies dir"))?;
            Ok((pr, path))
        });
        timings.proof_generation = sw.secs();
        match generated {
            Err(e) => evidence.notes.push(format!("proof generation failed: {e}")),
            Ok((pr, path)) => {
                let hashes = pr.hashes().iter().map(|h| format!("{h:016x}")).collect();
                let self_verification = if cfg.verify_proof {
                    let sw = Stopwatch::start();
                    let v = proof::verify(be, &pr, ctl);
                    timings.proof_verification = sw.secs();
                    match v {
                        Ok(v) if v.probable_prime == (result.verdict == Verdict::ProbablePrime) => {
                            "accepted".into()
                        }
                        Ok(_) => "rejected".into(),
                        Err(ProtocolError::Proof(_)) => "rejected".into(),
                        Err(e) => format!("not completed: {e}"),
                    }
                } else {
                    "not run".into()
                };
                evidence.proof = Some(ProofEvidence {
                    path,
                    power: pr.power(),
                    self_verification,
                    hashes,
                });
            }
        }
    }
    evidence.notes.push(
        "a positive local result is not a discovery claim; compare with known results".into(),
    );
    Ok(RunReport {
        process: ProcessState::Completed,
        result: Some(result),
        evidence,
        timings,
    })
}

fn spec_proof_through(store: Option<&ProofStore>, k: u64) -> Option<u64> {
    store.map(|s| {
        s.schedule
            .points()
            .iter()
            .copied()
            .filter(|&pt| pt <= k)
            .max()
            .unwrap_or(0)
    })
}

/// The configuration without the checkpoint path, which may move on resume.
pub(crate) fn without_checkpoint(v: &serde_json::Value) -> serde_json::Value {
    let mut v = v.clone();
    if let Some(o) = v.as_object_mut() {
        o.remove("checkpoint");
    }
    v
}
