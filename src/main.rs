// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The `mersenne-forge` command line.
//!
//! Exit codes: 0 success (completed run, accepted proof, doctor, bench);
//! 2 usage error; 3 incomplete (budget or stop request, also during the proof
//! stage); 4 unsupported (backend not built, exponent or option out of
//! range); 5 execution failure (repeated check failures, backend error, proof
//! generation or write failure); 6 checkpoint, proof residue, or file error;
//! 7 proof rejected (also a rejected self verification); 1 any other error.
//! With `--json`, a failure is printed on stdout as a JSON object with
//! `kind`, `message`, and `exit_code`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;

use mersenne_forge::bench;
use mersenne_forge::checkpoint::{create_dir_all_durable, prev_path, Checkpoint};
use mersenne_forge::ll::{self, JacobiCadence, LlConfig};
use mersenne_forge::proof::{self, Proof, ProofError};
use mersenne_forge::prp::{self, PrpConfig};
use mersenne_forge::run::{Control, ProcessState, ProofStage, ProtocolError, RunReport};
use mersenne_forge::select::{self, BackendKind};

static STOP: AtomicBool = AtomicBool::new(false);

const EXIT_INCOMPLETE: u8 = 3;
const EXIT_UNSUPPORTED: u8 = 4;
const EXIT_EXECUTION: u8 = 5;
const EXIT_STATE: u8 = 6;
const EXIT_REJECTED: u8 = 7;

#[derive(Parser)]
#[command(
    name = "mersenne-forge",
    version,
    about = "Lucas-Lehmer and base-3 PRP tests of Mersenne numbers"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Test {
    Ll,
    Prp,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    Cpu,
    Dwt,
    Gpu,
    Cuda,
}

impl BackendArg {
    fn kind(self) -> BackendKind {
        match self {
            BackendArg::Cpu => BackendKind::Cpu,
            BackendArg::Dwt => BackendKind::Dwt,
            BackendArg::Gpu => BackendKind::Gpu,
            BackendArg::Cuda => BackendKind::Cuda,
        }
    }
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum JacobiArg {
    Auto,
    EveryLog,
}

#[derive(Subcommand)]
enum Cmd {
    /// Report backends, build identity, supported ranges, and (with
    /// --exponent) per backend memory figures.
    Doctor {
        /// Exponent for the memory figures.
        #[arg(long)]
        exponent: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// Start a test.
    Run {
        #[arg(long, value_enum)]
        test: Test,
        #[arg(long)]
        exponent: u32,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: BackendArg,
        /// Adapter index (gpu) or CUDA ordinal (cuda).
        #[arg(long)]
        device: Option<u32>,
        /// Run budget, for example 90s, 10m, 2h, 1h30m. Required for gpu and cuda.
        #[arg(long)]
        budget: Option<String>,
        /// Directory for proof residues and the proof file (PRP).
        #[arg(long)]
        proof_dir: Option<PathBuf>,
        /// Proof power (PRP); default: upstream rule, lowered to a valid value for p.
        #[arg(long)]
        proof_power: Option<u32>,
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        /// Gerbicz block size (PRP). Upstream set: 200, 400, 500, 1000.
        #[arg(long, default_value_t = 1000)]
        block_size: u32,
        /// Explicit Gerbicz check step, a multiple of the block size (PRP).
        #[arg(long)]
        check_step: Option<u64>,
        /// Log and checkpoint step (LL).
        #[arg(long, default_value_t = 20_000)]
        log_step: u64,
        /// Jacobi check cadence (LL).
        #[arg(long, value_enum, default_value = "auto")]
        jacobi: JacobiArg,
        /// Skip verifying the generated proof.
        #[arg(long)]
        no_proof_verify: bool,
        #[arg(long)]
        json: bool,
    },
    /// Resume from a checkpoint. The checkpoint carries the configuration.
    Resume {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        budget: Option<String>,
        #[arg(long)]
        device: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// Verify a PRP proof file.
    Verify {
        #[arg(long)]
        proof: PathBuf,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: BackendArg,
        /// Reject the proof unless it is for this exponent.
        #[arg(long)]
        exponent: Option<u32>,
        #[arg(long)]
        device: Option<u32>,
        #[arg(long)]
        budget: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Run a benchmark plan and write raw observations (.json or .csv).
    Bench {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

struct Failure {
    code: u8,
    message: String,
}

fn fail(code: u8, message: impl Into<String>) -> Failure {
    Failure {
        code,
        message: message.into(),
    }
}

fn protocol_failure(e: ProtocolError) -> Failure {
    let code = match &e {
        ProtocolError::Config(_) => EXIT_UNSUPPORTED,
        ProtocolError::Backend(mersenne_forge::backend::BackendError::Unsupported(_)) => {
            EXIT_UNSUPPORTED
        }
        ProtocolError::Backend(_)
        | ProtocolError::TooManyErrors { .. }
        | ProtocolError::ConsistentError { .. } => EXIT_EXECUTION,
        ProtocolError::Proof(ProofError::Rejected) => EXIT_REJECTED,
        ProtocolError::Proof(ProofError::InvalidPower { .. }) => EXIT_UNSUPPORTED,
        ProtocolError::Interrupted { .. } => EXIT_INCOMPLETE,
        _ => EXIT_STATE,
    };
    fail(code, e.to_string())
}

/// Parse 90, 90s, 10m, 2h, 1h30m, 1d.
fn parse_duration(s: &str) -> Option<Duration> {
    let mut total = 0f64;
    let mut num = String::new();
    let mut any = false;
    for c in s.trim().chars() {
        if c.is_ascii_digit() || c == '.' {
            num.push(c);
            continue;
        }
        let v: f64 = num.parse().ok()?;
        num.clear();
        total += v * match c {
            's' => 1.0,
            'm' => 60.0,
            'h' => 3600.0,
            'd' => 86400.0,
            _ => return None,
        };
        any = true;
    }
    if !num.is_empty() {
        total += num.parse::<f64>().ok()?;
        any = true;
    }
    if !(any && total > 0.0 && total.is_finite()) {
        return None;
    }
    Duration::try_from_secs_f64(total).ok()
}

fn deadline(budget: &Option<String>, kind: BackendKind) -> Result<Option<Instant>, Failure> {
    match budget {
        None if kind.is_device() => Err(fail(
            EXIT_UNSUPPORTED,
            format!("the {} backend requires a bounded --budget", kind.name()),
        )),
        None => Ok(None),
        Some(b) => {
            let invalid = || fail(2, format!("invalid --budget {b:?}"));
            let d = parse_duration(b).ok_or_else(invalid)?;
            Ok(Some(Instant::now().checked_add(d).ok_or_else(invalid)?))
        }
    }
}

/// Hold an advisory lock on `<checkpoint>.lock` for as long as the returned
/// file lives, or refuse when another process holds it. The lock file is
/// never deleted.
fn lock_checkpoint(ckpt: &Path) -> Result<std::fs::File, Failure> {
    let mut name = ckpt.as_os_str().to_owned();
    name.push(".lock");
    let lock = PathBuf::from(name);
    let state = |e: std::io::Error| fail(EXIT_STATE, format!("{}: {e}", lock.display()));
    if let Some(dir) = lock.parent().filter(|d| !d.as_os_str().is_empty()) {
        create_dir_all_durable(dir).map_err(state)?;
    }
    let f = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock)
        .map_err(state)?;
    match f.try_lock() {
        Ok(()) => Ok(f),
        Err(std::fs::TryLockError::WouldBlock) => Err(fail(
            EXIT_STATE,
            format!(
                "{} is held by another run or resume of this checkpoint",
                lock.display()
            ),
        )),
        Err(std::fs::TryLockError::Error(e)) => Err(state(e)),
    }
}

fn open(
    kind: BackendKind,
    p: u32,
    device: Option<u32>,
) -> Result<Box<dyn mersenne_forge::backend::Backend>, Failure> {
    select::open(kind, p, device).map_err(|e| fail(EXIT_UNSUPPORTED, e.to_string()))
}

fn print_report(report: &RunReport, out: &serde_json::Value, as_json: bool) {
    if as_json {
        println!("{}", serde_json::to_string_pretty(out).expect("serializes"));
        return;
    }
    match (&report.process, &report.result) {
        (ProcessState::Completed, Some(r)) => {
            let what = match r.verdict {
                mersenne_forge::run::Verdict::Prime => "prime (LL criterion satisfied)",
                mersenne_forge::run::Verdict::ProbablePrime => "probable prime (base 3)",
                mersenne_forge::run::Verdict::Composite => "composite",
            };
            println!(
                "M{} {}: {what}, res64 {}",
                r.exponent,
                r.test.to_uppercase(),
                r.res64
            );
        }
        (
            ProcessState::Incomplete {
                k,
                reason,
                checkpoint,
            },
            _,
        ) => {
            println!("incomplete at iteration {k}: {reason}");
            match checkpoint {
                Some(c) => println!(
                    "resume with: mersenne-forge resume --checkpoint {}",
                    c.display()
                ),
                None => println!("no checkpoint was written; progress is lost"),
            }
        }
        (s, _) => println!("{s:?}"),
    }
    let e = &report.evidence;
    if e.gerbicz_passed + e.gerbicz_failed > 0 {
        println!(
            "gerbicz checks: {} passed, {} failed and rolled back",
            e.gerbicz_passed, e.gerbicz_failed
        );
    }
    if e.jacobi_passed + e.jacobi_failed + e.jacobi_inconclusive > 0 {
        println!(
            "jacobi checks: {} passed, {} failed and rolled back, {} inconclusive",
            e.jacobi_passed, e.jacobi_failed, e.jacobi_inconclusive
        );
    }
    if let Some(step) = e.check_step {
        println!("gerbicz check step: {step}");
    }
    if e.backend_faults > 0 {
        println!(
            "backend faults detected and rolled back: {}",
            e.backend_faults
        );
    }
    if let Some(pr) = &e.proof {
        println!(
            "proof: {} (power {}, self verification {})",
            pr.path.display(),
            pr.power,
            pr.self_verification
        );
    }
    if e.proof_stage != ProofStage::NotRequested {
        println!("proof stage: {}", e.proof_stage.name());
    }
    for n in &e.notes {
        println!("note: {n}");
    }
}

fn exit_for(report: &RunReport) -> u8 {
    match (&report.process, report.evidence.proof_stage) {
        (ProcessState::Completed, ProofStage::Failed) => EXIT_EXECUTION,
        (ProcessState::Completed, ProofStage::Rejected) => EXIT_REJECTED,
        (ProcessState::Completed, ProofStage::Interrupted) => EXIT_INCOMPLETE,
        (ProcessState::Completed, _) => 0,
        _ => EXIT_INCOMPLETE,
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    test: Test,
    p: u32,
    backend: BackendArg,
    device: Option<u32>,
    budget: Option<String>,
    proof_dir: Option<PathBuf>,
    proof_power: Option<u32>,
    checkpoint_dir: Option<PathBuf>,
    block_size: u32,
    check_step: Option<u64>,
    log_step: u64,
    jacobi: JacobiArg,
    no_proof_verify: bool,
    as_json: bool,
) -> Result<u8, Failure> {
    let kind = backend.kind();
    let requested = json!({
        "test": match test { Test::Ll => "ll", Test::Prp => "prp" },
        "exponent": p, "backend": kind.name(), "device": device, "budget": budget,
        "proof_dir": proof_dir, "proof_power": proof_power, "checkpoint_dir": checkpoint_dir,
        "block_size": block_size, "check_step": check_step, "log_step": log_step,
        "jacobi": if jacobi == JacobiArg::Auto { "auto" } else { "every_log" },
        "verify_proof": !no_proof_verify,
    });
    let deadline = deadline(&budget, kind)?;
    mersenne_forge::exponent::validate(p).map_err(|e| fail(EXIT_UNSUPPORTED, e.to_string()))?;
    let ckpt = checkpoint_dir.as_ref().map(|d| {
        d.join(match test {
            Test::Prp => format!("M{p}-prp.ckpt"),
            Test::Ll => format!("M{p}-ll.ckpt"),
        })
    });
    // The lock comes first, so two runs cannot both pass the check below.
    let _lock = ckpt.as_deref().map(lock_checkpoint).transpose()?;
    if let Some(c) = &ckpt {
        if c.exists() || prev_path(c).exists() {
            return Err(fail(
                EXIT_STATE,
                format!(
                    "a checkpoint for this run already exists at {}: continue it with `mersenne-forge resume --checkpoint {}`, or delete {} and {} to start over",
                    c.display(),
                    c.display(),
                    c.display(),
                    prev_path(c).display()
                ),
            ));
        }
    }
    let mut be = open(kind, p, device)?;
    let mut ctl = Control {
        stop: Some(&STOP),
        deadline,
        ..Control::default()
    };
    let (report, effective) = match test {
        Test::Prp => {
            let cfg = PrpConfig {
                block_size,
                check_step,
                proof_power,
                proof_dir,
                verify_proof: !no_proof_verify,
                checkpoint: ckpt.clone(),
                max_batch: 1000,
            };
            let eff = prp::effective_config(p, &cfg).map_err(protocol_failure)?;
            let r = prp::run_prp(&mut be, p, &cfg, requested.clone(), None, &mut ctl)
                .map_err(protocol_failure)?;
            (r, serde_json::to_value(eff).expect("serializes"))
        }
        Test::Ll => {
            if proof_dir.is_some() || proof_power.is_some() {
                return Err(fail(
                    EXIT_UNSUPPORTED,
                    "LL has no proof; --proof-dir applies to PRP",
                ));
            }
            let cfg = LlConfig {
                log_step,
                jacobi: if jacobi == JacobiArg::Auto {
                    JacobiCadence::Auto
                } else {
                    JacobiCadence::EveryLog
                },
                checkpoint: ckpt.clone(),
                max_batch: 1000,
            };
            let r = ll::run_ll(&mut be, p, &cfg, requested.clone(), None, &mut ctl)
                .map_err(protocol_failure)?;
            (r, serde_json::to_value(&cfg).expect("serializes"))
        }
    };
    let out = json!({ "command": "run", "requested": requested, "effective": effective, "report": report });
    print_report(&report, &out, as_json);
    Ok(exit_for(&report))
}

/// One resume attempt from `ck`, saving to `path`.
fn resume_from(
    be: &mut Box<dyn mersenne_forge::backend::Backend>,
    ck: &Checkpoint,
    path: &Path,
    ctl: &mut Control,
) -> Result<Result<RunReport, ProtocolError>, Failure> {
    let m = &ck.meta;
    let bad = |e: serde_json::Error| fail(EXIT_STATE, format!("checkpoint configuration: {e}"));
    if m.protocol == prp::PROTOCOL {
        let mut cfg: PrpConfig = serde_json::from_value(m.config.clone()).map_err(bad)?;
        cfg.checkpoint = Some(path.to_path_buf());
        Ok(prp::run_prp(
            be,
            m.exponent,
            &cfg,
            m.requested.clone(),
            Some(ck),
            ctl,
        ))
    } else if m.protocol == ll::PROTOCOL {
        let mut cfg: LlConfig = serde_json::from_value(m.config.clone()).map_err(bad)?;
        cfg.checkpoint = Some(path.to_path_buf());
        Ok(ll::run_ll(
            be,
            m.exponent,
            &cfg,
            m.requested.clone(),
            Some(ck),
            ctl,
        ))
    } else {
        Err(fail(
            EXIT_UNSUPPORTED,
            format!("unknown protocol {:?}", m.protocol),
        ))
    }
}

fn cmd_resume(
    path: &Path,
    budget: Option<String>,
    device: Option<u32>,
    as_json: bool,
) -> Result<u8, Failure> {
    let _lock = lock_checkpoint(path)?;
    let (mut ck, mut used) =
        Checkpoint::load_latest(path).map_err(|e| fail(EXIT_STATE, e.to_string()))?;
    let kind = BackendKind::parse(&ck.meta.backend.kind).ok_or_else(|| {
        fail(
            EXIT_UNSUPPORTED,
            format!("unknown backend {:?} in checkpoint", ck.meta.backend.kind),
        )
    })?;
    let deadline = deadline(&budget, kind)?;
    let mut be = open(kind, ck.meta.exponent, device)?;
    let mut ctl = Control {
        stop: Some(&STOP),
        deadline,
        ..Control::default()
    };
    let mut fallback_note = None;
    let mut result = resume_from(&mut be, &ck, path, &mut ctl)?;
    // A primary that loads but fails the resume validation (its on-load check
    // or a missing proof residue) falls back to `.prev`. An identity or
    // configuration mismatch does not: `.prev` belongs to the same run.
    if used == path
        && matches!(
            result,
            Err(ProtocolError::CheckpointRejected { .. }
                | ProtocolError::ProofResidueMissing { .. })
        )
    {
        let prev = prev_path(path);
        if let Ok(pck) = Checkpoint::load_file(&prev) {
            // Move the rejected primary aside, so the next save cannot rotate
            // it over the good `.prev`; it is kept for inspection.
            let mut aside = path.as_os_str().to_owned();
            aside.push(".rejected");
            std::fs::rename(path, &aside)
                .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", path.display())))?;
            let why = result.err().map(|e| e.to_string()).unwrap_or_default();
            fallback_note = Some(format!(
                "{} was not usable ({why}); it was moved to {} and the run resumed from {}",
                path.display(),
                PathBuf::from(&aside).display(),
                prev.display()
            ));
            ck = pck;
            used = prev;
            result = resume_from(&mut be, &ck, path, &mut ctl)?;
        }
    }
    let mut report = result.map_err(protocol_failure)?;
    if let Some(n) = fallback_note {
        report.evidence.notes.insert(0, n);
    }
    let requested =
        json!({ "checkpoint": path, "loaded_from": used, "budget": budget, "device": device });
    let out = json!({ "command": "resume", "requested": requested, "loaded_from": used, "resumed_at": ck.meta.k, "effective": ck.meta.config, "report": report });
    if !as_json {
        println!("resumed from {} at iteration {}", used.display(), ck.meta.k);
    }
    print_report(&report, &out, as_json);
    Ok(exit_for(&report))
}

fn cmd_verify(
    path: &Path,
    backend: BackendArg,
    exponent: Option<u32>,
    device: Option<u32>,
    budget: Option<String>,
    as_json: bool,
) -> Result<u8, Failure> {
    let kind = backend.kind();
    let deadline = deadline(&budget, kind)?;
    let emit = |status: &str, detail: serde_json::Value, code: u8| {
        if as_json {
            let out = json!({ "command": "verify", "proof": path, "backend": kind.name(), "status": status, "detail": detail, "exit_code": code });
            println!(
                "{}",
                serde_json::to_string_pretty(&out).expect("serializes")
            );
        } else {
            println!("proof {}: {status}", path.display());
            if let Some(m) = detail.get("message").and_then(|m| m.as_str()) {
                println!("{m}");
            }
        }
        code
    };
    let proof = match Proof::read(path, exponent) {
        Ok(pr) => pr,
        Err(e) => {
            let code = match e {
                ProofError::Io(_) => EXIT_STATE,
                _ => EXIT_REJECTED,
            };
            return Ok(emit(
                "rejected",
                json!({ "error": format!("{e:?}"), "message": e.to_string() }),
                code,
            ));
        }
    };
    let mut be = open(kind, proof.p, device)?;
    let mut ctl = Control {
        stop: Some(&STOP),
        deadline,
        ..Control::default()
    };
    let start = Instant::now();
    match proof::verify(&mut be, &proof, &mut ctl) {
        Ok(v) => {
            let claim = if v.probable_prime {
                "probable prime"
            } else {
                "composite"
            };
            Ok(emit(
                "accepted",
                json!({
                    "exponent": proof.p, "power": proof.power(), "claim": claim,
                    "terminal_recomputed": v.terminal_recomputed,
                    "squarings": v.squarings, "gerbicz_passed": v.gerbicz_passed, "gerbicz_failed": v.gerbicz_failed,
                    "secs": start.elapsed().as_secs_f64(),
                    "message": verify_message(proof.p, claim, v.terminal_recomputed),
                }),
                0,
            ))
        }
        Err(ProtocolError::Proof(e)) => Ok(emit(
            "rejected",
            json!({ "error": format!("{e:?}"), "message": e.to_string() }),
            EXIT_REJECTED,
        )),
        Err(e) => Err(protocol_failure(e)),
    }
}

/// What an accepted proof establishes, for the `verify` output.
fn verify_message(p: u32, claim: &str, recomputed: bool) -> String {
    if recomputed {
        format!(
            "M{p} {claim}: the terminal residue was recomputed directly and matches the proof; this verifies the PRP computation, not deterministic primality"
        )
    } else {
        format!(
            "M{p} {claim}: the file satisfies the version 2 proof equation (the relation the upstream verifier checks). \
             That is sound against computation errors and against a prover who skipped the work, \
             not against a prover who knows a factor of M{p} or elements of small order modulo M{p}; \
             the challenge is 64 bits. A probable prime claim from a proof needs an independent test"
        )
    }
}

fn cmd_bench(plan: &Path, output: &Path) -> Result<u8, Failure> {
    let text = std::fs::read_to_string(plan)
        .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", plan.display())))?;
    let plan: bench::Plan =
        serde_json::from_str(&text).map_err(|e| fail(2, format!("invalid plan: {e}")))?;
    plan.validate()
        .map_err(|e| fail(2, format!("invalid plan: {e}")))?;
    let out = bench::run_plan(&plan, output, Some(&STOP), |o| {
        eprintln!(
            "{:>4} {:<24} repeat {:<3}{} {:<10} {:.3}s",
            o.sequence,
            o.entry,
            o.repeat,
            if o.warmup { " (warmup)" } else { "" },
            o.status,
            o.wall_secs
        )
    })
    .map_err(|e| fail(EXIT_STATE, e))?;
    let failed = out
        .observations
        .iter()
        .filter(|o| o.status != "completed")
        .count();
    println!(
        "{} observations written to {} ({failed} not completed)",
        out.observations.len(),
        output.display()
    );
    Ok(if STOP.load(Ordering::SeqCst) {
        EXIT_INCOMPLETE
    } else {
        0
    })
}

/// Memory figures of one backend for exponent p. Canonical residues (what
/// crosses the backend trait and goes into files) are shown apart from the
/// backend's resident registers and fixed scratch.
fn memory_entry(kind: BackendKind, p: u32) -> serde_json::Value {
    match kind.memory_estimate(p) {
        None => json!(null),
        Some(Err(e)) => json!({ "error": e.to_string() }),
        Some(Ok(m)) => {
            let power = proof::default_power(p);
            let regs = |n: u64| m.fixed_bytes + n * m.bytes_per_register;
            json!({
                "plan": m.plan,
                "on_device": m.on_device,
                "canonical_residue_bytes": 4 * mersenne_forge::residue::limb_count(p) as u64,
                "bytes_per_register": m.bytes_per_register,
                "fixed_bytes": m.fixed_bytes,
                "registers": { "prp": 4, "ll": 2, "proof_verification": 4, "proof_generation": power + 2 },
                "default_proof_power": power,
                "peak_bytes": { "prp": regs(4), "ll": regs(2), "proof_generation": regs(u64::from(power) + 2) },
            })
        }
    }
}

fn cmd_doctor(exponent: Option<u32>, as_json: bool) -> Result<u8, Failure> {
    let mut backends = Vec::new();
    for kind in [
        BackendKind::Cpu,
        BackendKind::Dwt,
        BackendKind::Gpu,
        BackendKind::Cuda,
    ] {
        let mut entry = json!({
            "kind": kind.name(),
            "built": kind.built(),
            "supported_exponents": kind.supported_exponents(),
        });
        if let Some(p) = exponent {
            entry["memory"] = memory_entry(kind, p);
        }
        match kind {
            BackendKind::Cpu => entry["arithmetic"] = json!("exact bigint, Mersenne folding"),
            BackendKind::Dwt => {
                entry["arithmetic"] = json!("f64 irrational-base DWT, roundoff checked")
            }
            BackendKind::Gpu => {
                entry["arithmetic"] = json!("exact integer NTT on the GPU via wgpu");
                #[cfg(feature = "gpu")]
                {
                    entry["probe"] = serde_json::to_value(mersenne_forge::backend::gpu::probe())
                        .expect("serializes");
                }
                #[cfg(not(feature = "gpu"))]
                {
                    entry["note"] = json!("not built; enable with --features gpu");
                }
            }
            BackendKind::Cuda => {
                entry["arithmetic"] = json!("CUDA kernels via cudarc");
                #[cfg(feature = "cuda")]
                {
                    entry["probe"] = serde_json::to_value(mersenne_forge::backend::cuda::probe())
                        .expect("serializes");
                }
                #[cfg(not(feature = "cuda"))]
                {
                    entry["note"] = json!("not built; enable with --features cuda");
                }
            }
        }
        backends.push(entry);
    }
    let out = json!({
        "name": "mersenne-forge",
        "version": env!("CARGO_PKG_VERSION"),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "backends": backends,
        "exponent": exponent,
        "requirements": {
            "memory": "per backend with --exponent: resident registers (PRP 4, LL 2, proof generation power + 2) plus fixed scratch",
            "proof_disk_bytes": "2^power * ((p - 1) / 8 + 1) for the proof residues, plus the proof file",
            "checkpoint_bytes": "about 2 * p / 8 plus metadata",
        },
        "notes": [
            "listing a device is not validation of its arithmetic",
            "the proof file format follows PRPLL version 2; acceptance by an upstream verifier has not been tested",
        ],
    });
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&out).expect("serializes")
        );
    } else {
        println!(
            "mersenne-forge {} ({})",
            env!("CARGO_PKG_VERSION"),
            out["profile"].as_str().unwrap_or("")
        );
        for b in out["backends"].as_array().into_iter().flatten() {
            let range = match b["supported_exponents"].as_array() {
                Some(r) => format!("exponents {}..={}", r[0], r[1]),
                None => "not built".into(),
            };
            println!(
                "  {:<5} {:<45} {range}",
                b["kind"].as_str().unwrap_or(""),
                b["arithmetic"].as_str().unwrap_or("")
            );
            let m = &b["memory"];
            if let Some(e) = m["error"].as_str() {
                println!("        memory: {e}");
            } else if m.is_object() {
                println!(
                    "        memory for M{}: plan {}, canonical residue {} B, register {} B{}, fixed {} B; peak PRP {} B, LL {} B, proof generation (power {}) {} B",
                    exponent.unwrap_or(0),
                    m["plan"].as_str().unwrap_or(""),
                    m["canonical_residue_bytes"],
                    m["bytes_per_register"],
                    if m["on_device"] == true { " on device" } else { "" },
                    m["fixed_bytes"],
                    m["peak_bytes"]["prp"],
                    m["peak_bytes"]["ll"],
                    m["default_proof_power"],
                    m["peak_bytes"]["proof_generation"]
                );
            }
            if let Some(devs) = b["probe"]["devices"].as_array() {
                for d in devs {
                    println!(
                        "        device {}: {}",
                        d["id"],
                        d["name"].as_str().unwrap_or("")
                    );
                }
            }
        }
        println!("note: listing a device is not validation of its arithmetic");
    }
    Ok(0)
}

/// The `kind` of a failure in JSON output, from its exit code.
fn failure_kind(code: u8) -> &'static str {
    match code {
        2 => "usage",
        EXIT_INCOMPLETE => "incomplete",
        EXIT_UNSUPPORTED => "unsupported",
        EXIT_EXECUTION => "execution",
        EXIT_STATE => "state",
        EXIT_REJECTED => "rejected",
        _ => "error",
    }
}

fn emit_failure(f: &Failure, as_json: bool) {
    if as_json {
        let out =
            json!({ "kind": failure_kind(f.code), "message": f.message, "exit_code": f.code });
        println!(
            "{}",
            serde_json::to_string_pretty(&out).expect("serializes")
        );
    } else {
        eprintln!("error: {}", f.message);
    }
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            let json = std::env::args().any(|a| a == "--json");
            if !json || !e.use_stderr() {
                e.exit();
            }
            emit_failure(&fail(2, e.to_string().trim_end()), true);
            return ExitCode::from(2);
        }
    };
    let as_json = match &cli.cmd {
        Cmd::Doctor { json, .. }
        | Cmd::Run { json, .. }
        | Cmd::Resume { json, .. }
        | Cmd::Verify { json, .. } => *json,
        Cmd::Bench { .. } => false,
    };
    let _ = ctrlc::set_handler(|| {
        if STOP.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        eprintln!(
            "stop requested; finishing at the next valid boundary (press Ctrl-C again to abort)"
        );
    });
    let r = match cli.cmd {
        Cmd::Doctor { exponent, json } => cmd_doctor(exponent, json),
        Cmd::Run {
            test,
            exponent,
            backend,
            device,
            budget,
            proof_dir,
            proof_power,
            checkpoint_dir,
            block_size,
            check_step,
            log_step,
            jacobi,
            no_proof_verify,
            json,
        } => cmd_run(
            test,
            exponent,
            backend,
            device,
            budget,
            proof_dir,
            proof_power,
            checkpoint_dir,
            block_size,
            check_step,
            log_step,
            jacobi,
            no_proof_verify,
            json,
        ),
        Cmd::Resume {
            checkpoint,
            budget,
            device,
            json,
        } => cmd_resume(&checkpoint, budget, device, json),
        Cmd::Verify {
            proof,
            backend,
            exponent,
            device,
            budget,
            json,
        } => cmd_verify(&proof, backend, exponent, device, budget, json),
        Cmd::Bench { plan, output } => cmd_bench(&plan, &output),
    };
    match r {
        Ok(code) => ExitCode::from(code),
        Err(f) => {
            emit_failure(&f, as_json);
            ExitCode::from(f.code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_duration;
    use std::time::Duration;

    #[test]
    fn durations() {
        assert_eq!(parse_duration("90"), Some(Duration::from_secs(90)));
        assert_eq!(parse_duration("90s"), Some(Duration::from_secs(90)));
        assert_eq!(parse_duration("1h30m"), Some(Duration::from_secs(5400)));
        assert_eq!(parse_duration("2d"), Some(Duration::from_secs(172_800)));
        assert_eq!(parse_duration("0s"), None);
        assert_eq!(parse_duration("5x"), None);
        assert_eq!(parse_duration(""), None);
    }
}
