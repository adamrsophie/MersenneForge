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
enum V5Action {
    /// Register (or update) this computer, creating its GUID when needed.
    Register {
        /// PrimeNet user ID; none registers anonymously.
        #[arg(long)]
        user: Option<String>,
        #[arg(long)]
        computer: Option<String>,
        #[arg(long, default_value = "unknown")]
        cpu: String,
        #[arg(long, default_value_t = 3000)]
        mhz: u32,
        #[arg(long, default_value_t = 8192)]
        memory_mib: u64,
        #[arg(long, default_value_t = 1024)]
        l2_kib: u32,
        #[arg(long, default_value_t = 24)]
        hours: u32,
        /// Work preference (PRP, PRP_DC, LL_DC, ... or a number).
        #[arg(long)]
        worktype: Option<String>,
    },
    /// Get assignments and append them to the worktodo file.
    Fetch {
        #[arg(long, default_value = "worktodo.txt")]
        worktodo: PathBuf,
        #[arg(long, default_value_t = 1)]
        count: u32,
        #[arg(long, default_value_t = 0)]
        worker: u32,
        /// CERT work only.
        #[arg(long)]
        cert: bool,
    },
    /// Send the result lines not sent yet.
    Submit {
        #[arg(long, default_value = "results.txt")]
        results: PathBuf,
    },
    /// Report progress on one assignment.
    Progress {
        #[arg(long)]
        aid: String,
        /// Percent done.
        #[arg(long)]
        percent: f64,
        /// Hours to completion.
        #[arg(long)]
        eta_hours: f64,
        #[arg(long, default_value_t = 0)]
        worker: u32,
    },
    /// Give one assignment back.
    Unreserve {
        #[arg(long)]
        aid: String,
    },
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
        /// Relocated proof directory, including the saved M<p>.residues directory.
        #[arg(long)]
        proof_dir: Option<PathBuf>,
        #[arg(long)]
        budget: Option<String>,
        #[arg(long)]
        device: Option<u32>,
        /// Backend for a portable checkpoint (from `checkpoint-export`);
        /// other checkpoints resume on the backend that wrote them.
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
        #[arg(long)]
        json: bool,
    },
    /// Write a portable copy of a checkpoint: its last verified state, not
    /// tied to a backend, build, or device. Resume it with `resume
    /// --backend`, or place it at a `work` checkpoint path.
    CheckpointExport {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
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
    /// Process assignments from a worktodo file, one after another: run or
    /// resume each, append its result line to the results file, then remove
    /// the assignment. Lines this program cannot run are left in place.
    Work {
        #[arg(long)]
        worktodo: PathBuf,
        #[arg(long)]
        results: PathBuf,
        /// Checkpoints, one per assignment; an interrupted assignment resumes.
        #[arg(long)]
        checkpoint_dir: PathBuf,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: BackendArg,
        #[arg(long)]
        device: Option<u32>,
        /// Budget for the whole session (required for gpu and cuda).
        #[arg(long)]
        budget: Option<String>,
        /// Proof directory for PRP assignments (the proof files are kept).
        #[arg(long)]
        proof_dir: Option<PathBuf>,
        #[arg(long)]
        proof_power: Option<u32>,
        /// JSON health state, rewritten atomically as the session moves on:
        /// pid, state, current line, completed and skipped counts, update
        /// time, and the exit code when the session ends.
        #[arg(long)]
        status_file: Option<PathBuf>,
    },
    /// Run one `work` process per entry of a JSON list of workers (one
    /// device and worktodo file each), restart a worker that crashes, and
    /// stop them all on Ctrl-C.
    Supervise {
        #[arg(long)]
        config: PathBuf,
        /// Restarts allowed per worker before it is given up on.
        #[arg(long, default_value_t = 5)]
        max_restarts: u32,
    },
    /// Print queue, checkpoint, and worker state; with --watch, refresh
    /// every N seconds until Ctrl-C.
    Status {
        #[arg(long, default_value = "worktodo.txt")]
        worktodo: PathBuf,
        #[arg(long, default_value = "results.txt")]
        results: PathBuf,
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        /// Worker status files to show (from `work --status-file`).
        #[arg(long)]
        status_file: Vec<PathBuf>,
        #[arg(long)]
        watch: Option<u64>,
        /// Serve the status as a web page (refreshed every 5 s) and as text
        /// at /status.txt on this loopback address, for example
        /// 127.0.0.1:8080, until Ctrl-C. Read only.
        #[arg(long)]
        serve: Option<std::net::SocketAddr>,
    },
    /// Print one power reading in watts from a built-in sensor, for use as a
    /// benchmark telemetry command: rapl (Linux CPU package), nvidia
    /// (nvidia-smi, per device), or powermetrics (macOS, needs root).
    Power {
        #[arg(long)]
        source: String,
        #[arg(long, default_value_t = 0)]
        device: u32,
    },
    /// Exchange with PrimeNet through its manual web forms: send unsent
    /// result lines, top up the worktodo file, and upload proof files.
    /// Untested against the server. The password is read from a file.
    Primenet {
        #[arg(long)]
        user: String,
        #[arg(long)]
        password_file: PathBuf,
        #[arg(long, default_value = "worktodo.txt")]
        worktodo: PathBuf,
        #[arg(long, default_value = "results.txt")]
        results: PathBuf,
        /// Upload the `.proof` files found here.
        #[arg(long)]
        proof_dir: Option<PathBuf>,
        /// Work type to fetch (PRP, PRP_DC, LL_DC, ... or a number); none
        /// fetches nothing.
        #[arg(long)]
        worktype: Option<String>,
        /// Runnable lines to keep in the worktodo file.
        #[arg(long, default_value_t = 2)]
        ahead: usize,
        /// Holds the session cookies and the sent and refused logs.
        #[arg(long, default_value = "primenet")]
        state_dir: PathBuf,
    },
    /// The PrimeNet v5 API, the protocol Prime95 and AutoPrimeNet speak:
    /// register this computer, fetch assignments, send results, report
    /// progress, unreserve. NOT TESTED against the PrimeNet server.
    PrimenetV5 {
        /// Holds the computer GUID and the log of results sent.
        #[arg(long, default_value = "primenet")]
        state_dir: PathBuf,
        #[command(subcommand)]
        action: V5Action,
    },
    /// Trial factor 2^p - 1 between 2^from and 2^to on the CPU. Exit 0 with
    /// or without a factor; factors are checked before they are printed.
    Tf {
        #[arg(long)]
        exponent: u32,
        #[arg(long)]
        from: u32,
        #[arg(long)]
        to: u32,
        /// Test candidates below 2^95 on the GPU (needs the gpu feature);
        /// the sieve and every hit's final check stay on the host.
        #[arg(long)]
        gpu: bool,
        #[arg(long)]
        device: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// P-1 factoring of 2^p - 1 with bounds B1 and B2, both stages on the
    /// chosen backend (stage 2 also makes one host round trip per prime).
    Pm1 {
        #[arg(long)]
        exponent: u32,
        #[arg(long)]
        b1: u64,
        #[arg(long, default_value_t = 0)]
        b2: u64,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: BackendArg,
        #[arg(long)]
        device: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// ECM stages 1 and 2 on 2^p - 1 (host arithmetic, small exponents
    /// only). Stage 2 runs when --b2 is above --b1.
    Ecm {
        #[arg(long)]
        exponent: u32,
        #[arg(long)]
        b1: u64,
        #[arg(long, default_value_t = 0)]
        b2: u64,
        #[arg(long, default_value_t = 10)]
        curves: u32,
        #[arg(long, default_value_t = 6)]
        sigma: u64,
        #[arg(long)]
        json: bool,
    },
    /// Time the default transform of a backend against its opt-in variants
    /// (dwt: packed real transform; gpu: fused radix 4 stages), check that
    /// every variant gives the same residue, and print the fastest with the
    /// environment setting that selects it.
    Tune {
        #[arg(long)]
        exponent: u32,
        #[arg(long, value_enum, default_value = "dwt")]
        backend: BackendArg,
        #[arg(long)]
        device: Option<u32>,
        #[arg(long, default_value_t = 200)]
        iters: u64,
    },
    /// Williams' P+1 on 2^p - 1 (host arithmetic, small exponents only).
    /// --run picks the start value as Prime95's nth_run does (1: 2/7,
    /// 2: 6/5).
    Pp1 {
        #[arg(long)]
        exponent: u32,
        #[arg(long)]
        b1: u64,
        #[arg(long, default_value_t = 0)]
        b2: u64,
        #[arg(long, default_value_t = 1)]
        run: u32,
        #[arg(long)]
        json: bool,
    },
    /// Test a backend and the factoring code against known answers: LL and
    /// PRP of small exponents, Gerbicz checked squaring chains, trial
    /// factoring of known factors, and P+1. Exit 0 when every case passes.
    Selftest {
        #[arg(long, value_enum, default_value = "dwt")]
        backend: BackendArg,
        #[arg(long)]
        device: Option<u32>,
        /// Add larger exponents (minutes on a CPU).
        #[arg(long)]
        full: bool,
        #[arg(long)]
        json: bool,
    },
    /// Check that a reported factor divides 2^p - 1. Exit 0 when it does.
    CheckFactor {
        #[arg(long)]
        exponent: u32,
        #[arg(long)]
        factor: String,
        #[arg(long)]
        json: bool,
    },
    /// Write a recorded benchmark JSON output again as .json or .csv,
    /// without running anything.
    BenchRender {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Paired comparison of two entries of a recorded benchmark (by entry
    /// index), with a bootstrap interval. Prints JSON.
    BenchAnalyze {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        baseline: usize,
        #[arg(long)]
        candidate: usize,
        /// wall_secs, iterations, or energy_joules.
        #[arg(long, default_value = "wall_secs")]
        metric: String,
        #[arg(long, default_value_t = 1)]
        seed: u64,
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
    select::open(kind, p, device).map_err(|e| {
        let code = match e {
            mersenne_forge::backend::BackendError::Unsupported(_) => EXIT_UNSUPPORTED,
            _ => EXIT_EXECUTION,
        };
        fail(code, e.to_string())
    })
}

fn print_report(report: &RunReport, out: &serde_json::Value, as_json: bool) {
    if as_json {
        println!("{}", serde_json::to_string_pretty(out).expect("serializes"));
        return;
    }
    match (&report.process, &report.result) {
        (ProcessState::Completed, Some(r)) => {
            let what = match r.verdict {
                mersenne_forge::run::Verdict::Prime if r.special_case.is_some() => {
                    "prime (decided directly; p = 2)"
                }
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
                before_last: false,
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
    proof_dir: Option<&Path>,
    ctl: &mut Control,
) -> Result<Result<RunReport, ProtocolError>, Failure> {
    let m = &ck.meta;
    let bad = |e: serde_json::Error| fail(EXIT_STATE, format!("checkpoint configuration: {e}"));
    if m.protocol == prp::PROTOCOL {
        let mut cfg: PrpConfig = serde_json::from_value(m.config.clone()).map_err(bad)?;
        cfg.checkpoint = Some(path.to_path_buf());
        if let Some(dir) = proof_dir {
            if cfg.proof_dir.is_none() {
                return Err(fail(
                    2,
                    "--proof-dir requires a checkpoint with a proof schedule",
                ));
            }
            let residues = proof::ProofStore::residue_dir(dir, m.exponent);
            if !residues.is_dir() {
                return Err(fail(
                    EXIT_STATE,
                    format!("saved proof residues not found: {}", residues.display()),
                ));
            }
            cfg.proof_dir = Some(dir.to_path_buf());
        }
        Ok(prp::run_prp(
            be,
            m.exponent,
            &cfg,
            m.requested.clone(),
            Some(ck),
            ctl,
        ))
    } else if m.protocol == ll::PROTOCOL {
        if proof_dir.is_some() {
            return Err(fail(2, "--proof-dir is only supported for PRP checkpoints"));
        }
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
    backend: Option<BackendKind>,
    proof_dir: Option<PathBuf>,
    as_json: bool,
) -> Result<u8, Failure> {
    let _lock = lock_checkpoint(path)?;
    let (first, _) = Checkpoint::load_latest(path).map_err(|e| fail(EXIT_STATE, e.to_string()))?;
    let portable = first.meta.backend.kind == mersenne_forge::checkpoint::PORTABLE;
    let kind = match (portable, backend) {
        (true, Some(k)) => Ok(k),
        (true, None) => Err(fail(2, "a portable checkpoint needs --backend")),
        (false, Some(_)) => Err(fail(
            2,
            "--backend applies only to a portable checkpoint (see checkpoint-export)",
        )),
        (false, None) => BackendKind::parse(&first.meta.backend.kind).ok_or_else(|| {
            fail(
                EXIT_UNSUPPORTED,
                format!(
                    "unknown backend {:?} in checkpoint",
                    first.meta.backend.kind
                ),
            )
        }),
    }?;
    let deadline = deadline(&budget, kind)?;
    let mut be = open(kind, first.meta.exponent, device)?;
    let mut ctl = Control {
        stop: Some(&STOP),
        deadline,
        ..Control::default()
    };
    let (result, ck, used, fallback_note) =
        resume_with_fallback(&mut be, path, proof_dir.as_deref(), &mut ctl)?;
    let mut report = result.map_err(protocol_failure)?;
    if let Some(n) = fallback_note {
        report.evidence.notes.insert(0, n);
    }
    let requested = json!({ "checkpoint": path, "loaded_from": used, "budget": budget, "device": device, "proof_dir": proof_dir });
    let mut effective = ck.meta.config.clone();
    effective["checkpoint"] = json!(path);
    if let Some(dir) = proof_dir {
        effective["proof_dir"] = json!(dir);
    }
    let out = json!({ "command": "resume", "requested": requested, "loaded_from": used, "resumed_at": ck.meta.k, "effective": effective, "report": report });
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
            // A file that cannot be read was not checked: that is a file
            // error, not a rejection.
            let (status, code) = match e {
                ProofError::Io(_) => ("error", EXIT_STATE),
                _ => ("rejected", EXIT_REJECTED),
            };
            return Ok(emit(
                status,
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
            let claim = if !v.terminal_recomputed && proof.sign_decides_claim() {
                "undetermined"
            } else if v.probable_prime {
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
    if claim == "undetermined" {
        format!(
            "M{p}: the file satisfies the version 2 proof equation, but that equation also holds with the terminal residue negated, \
             and here the sign decides between probable prime and composite. Recompute the terminal residue to decide"
        )
    } else if recomputed {
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

struct WorkArgs {
    worktodo: PathBuf,
    results: PathBuf,
    checkpoint_dir: PathBuf,
    kind: BackendKind,
    device: Option<u32>,
    budget: Option<String>,
    proof_dir: Option<PathBuf>,
    proof_power: Option<u32>,
    status_file: Option<PathBuf>,
}

impl WorkArgs {
    /// Rewrite the status file, when there is one. Failures are reported and
    /// never stop the work.
    fn status(&self, state: &str, current: Option<&str>, extra: serde_json::Value) {
        let Some(path) = &self.status_file else {
            return;
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut v = json!({
            "pid": std::process::id(),
            "state": state,
            "current": current,
            "worktodo": self.worktodo,
            "backend": self.kind.name(),
            "device": self.device,
            "updated": mersenne_forge::worktodo::utc_timestamp(now),
        });
        if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        let text = serde_json::to_string_pretty(&v).expect("serializable");
        if let Err(e) = mersenne_forge::checkpoint::write_atomic(path, text.as_bytes(), false) {
            eprintln!("status file {}: {e}", path.display());
        }
    }
}

/// An exclusive lock in the convention AutoPrimeNet uses for the files it
/// shares with a worker: `<file>.lck`, created exclusively and removed when
/// done. Waits up to 60 s for another holder. (Interoperation with
/// AutoPrimeNet itself has not been tested.)
struct LckGuard(PathBuf);

impl Drop for LckGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn lck(path: &Path) -> Result<LckGuard, Failure> {
    let mut name = path.as_os_str().to_owned();
    name.push(".lck");
    let l = PathBuf::from(name);
    for _ in 0..600 {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&l)
        {
            Ok(_) => return Ok(LckGuard(l)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                std::thread::sleep(Duration::from_millis(100))
            }
            Err(e) => return Err(fail(EXIT_STATE, format!("{}: {e}", l.display()))),
        }
    }
    Err(fail(
        EXIT_STATE,
        format!(
            "{} has been held for 60 s; if no other program is using it, delete it",
            l.display()
        ),
    ))
}

/// The completion journal next to the results file: one assignment line per
/// line, written after its result and before the assignment is removed.
fn journal_path(results: &Path) -> PathBuf {
    let mut n = results.as_os_str().to_owned();
    n.push(".done");
    PathBuf::from(n)
}

/// Append `line` plus a newline to `path` and make it durable: a fragment a
/// crash left at the end is moved to `<path>.torn` first, so every line
/// stays whole, then the file and its directory entry are synced.
fn append_durable(path: &Path, line: &str) -> Result<(), Failure> {
    use std::io::Write;
    let state = |e: std::io::Error| fail(EXIT_STATE, format!("{}: {e}", path.display()));
    if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        create_dir_all_durable(d).map_err(state)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .map_err(state)?;
    let bytes = std::fs::read(path).map_err(state)?;
    if bytes.last().is_some_and(|&c| c != b'\n') {
        let cut = bytes.iter().rposition(|&c| c == b'\n').map_or(0, |i| i + 1);
        let mut torn = path.as_os_str().to_owned();
        torn.push(".torn");
        let mut t = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(PathBuf::from(&torn))
            .map_err(state)?;
        t.write_all(&bytes[cut..]).map_err(state)?;
        t.write_all(b"\n").map_err(state)?;
        t.sync_all().map_err(state)?;
        f.set_len(cut as u64).map_err(state)?;
    }
    f.write_all(format!("{line}\n").as_bytes()).map_err(state)?;
    f.sync_all().map_err(state)?;
    mersenne_forge::checkpoint::sync_dir(path).map_err(state)
}

/// True when this assignment is already complete: its line is in the
/// completion journal, or (for an assignment with an ID) the results file
/// holds a final result with the same ID, exponent, and work type. A
/// matching result without a journal entry is the window between the two
/// writes of a crashed session. Without an ID, only the journal counts, so
/// a deliberate second test of the same exponent still runs.
fn already_reported(results: &Path, a: &mersenne_forge::worktodo::Assignment) -> bool {
    let journal = std::fs::read_to_string(journal_path(results)).unwrap_or_default();
    if journal.lines().any(|l| l == a.line) {
        return true;
    }
    let Some(aid) = &a.aid else {
        return false;
    };
    std::fs::read_to_string(results)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .any(|v| {
            v["aid"] == aid.as_str()
                && v["exponent"] == a.exponent
                && v["worktype"] == a.worktype()
                && matches!(v["status"].as_str(), Some("P" | "C"))
        })
}

/// Remove the first line equal to `line` from the worktodo file, atomically,
/// under its `.lck` lock so a program adding lines does not lose them.
fn remove_line(worktodo: &Path, line: &str) -> Result<(), Failure> {
    replace_line(worktodo, line, None)
}

/// Replace the first line equal to `line` by `with`, or remove it when
/// `with` is None, atomically and under the worktodo `.lck` lock.
fn replace_line(worktodo: &Path, line: &str, with: Option<&str>) -> Result<(), Failure> {
    let state = |e: std::io::Error| fail(EXIT_STATE, format!("{}: {e}", worktodo.display()));
    let _l = lck(worktodo)?;
    let text = std::fs::read_to_string(worktodo).map_err(state)?;
    let mut done = false;
    let kept: String = text
        .split_inclusive('\n')
        .filter_map(|l| {
            let hit = !done && l.trim_end_matches(['\r', '\n']) == line;
            done |= hit;
            match (hit, with) {
                (false, _) => Some(l.to_string()),
                (true, Some(r)) => Some(format!("{r}\n")),
                (true, None) => None,
            }
        })
        .collect();
    mersenne_forge::checkpoint::write_atomic(worktodo, kept.as_bytes(), false).map_err(state)
}

/// Record a finished assignment: result line, then journal, then removal,
/// each durable before the next, under the results file's lock so two
/// queues sharing one results file cannot both report the same work.
fn commit_result(
    w: &WorkArgs,
    a: &mersenne_forge::worktodo::Assignment,
    line: Option<&str>,
) -> Result<(), Failure> {
    {
        let _l = lck(&w.results)?;
        if let Some(line) = line {
            if !already_reported(&w.results, a) {
                append_durable(&w.results, line)?;
            }
        } else {
            // Recovery: the result is already in the file; make it durable
            // before the assignment goes.
            if let Ok(f) = std::fs::File::open(&w.results) {
                let _ = f.sync_all();
            }
            let _ = mersenne_forge::checkpoint::sync_dir(&w.results);
        }
        let journal = journal_path(&w.results);
        let logged = std::fs::read_to_string(&journal)
            .unwrap_or_default()
            .lines()
            .any(|l| l == a.line);
        if !logged {
            append_durable(&journal, &a.line)?;
        }
    }
    remove_line(&w.worktodo, &a.line)
}

/// Checkpoint file of one assignment: the ID when there is one, otherwise a
/// hash of the line, so two assignments for one exponent never share state.
fn assignment_checkpoint(dir: &Path, a: &mersenne_forge::worktodo::Assignment) -> PathBuf {
    use mersenne_forge::worktodo::WorkKind;
    let key = line_key(a.aid.as_deref(), &a.line);
    let test = match a.kind {
        WorkKind::Prp => "prp",
        WorkKind::Ll => "ll",
    };
    dir.join(format!("M{}-{test}-{key}.ckpt", a.exponent))
}

/// The assignment ID in lowercase, or the first 8 bytes of SHA3 of the line.
fn line_key(aid: Option<&str>, line: &str) -> String {
    use sha3::Digest;
    match aid {
        Some(aid) => aid.to_ascii_lowercase(),
        None => sha3::Sha3_256::digest(line.as_bytes())[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    }
}

/// Record a finished line by its text alone: the result line (once), the
/// journal entry, then the line's removal, or its replacement by
/// `replacement` (a PRP line whose P-1 has been done).
///
/// `entry` is the journal entry: the line itself when the line is done, or
/// a marker (`pm1:<line>`) when only a step of it is, so a crash between
/// journal and rewrite never makes the whole assignment look reported.
fn commit_line(
    w: &WorkArgs,
    line: &str,
    entry: &str,
    result: &str,
    replacement: Option<&str>,
) -> Result<(), Failure> {
    {
        let _l = lck(&w.results)?;
        let journal = journal_path(&w.results);
        let logged = std::fs::read_to_string(&journal)
            .unwrap_or_default()
            .lines()
            .any(|l| l == entry);
        if !logged {
            append_durable(&w.results, result)?;
            append_durable(&journal, entry)?;
        }
    }
    replace_line(&w.worktodo, line, replacement)
}

/// Run P-1 for a `work` line, saving stage 1 progress next to the
/// checkpoints so a budgeted session continues where the last one stopped.
fn work_pm1(
    w: &WorkArgs,
    p: u32,
    key: &str,
    b1: u64,
    b2: u64,
    deadline: Option<Instant>,
) -> Result<mersenne_forge::factor::Pm1Outcome, Failure> {
    use mersenne_forge::checkpoint::{CheckpointMeta, FORMAT_VERSION};
    const PROTO: &str = "pm1-stage1/1";
    if p > mersenne_forge::factor::WORK_PM1_MAX_P {
        return Err(fail(
            EXIT_UNSUPPORTED,
            format!(
                "P-1 of exponents above {} is left to another program",
                mersenne_forge::factor::WORK_PM1_MAX_P
            ),
        ));
    }
    let path = w.checkpoint_dir.join(format!("M{p}-pm1-{key}.ckpt"));
    create_dir_all_durable(&w.checkpoint_dir)
        .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", w.checkpoint_dir.display())))?;
    let resume = Checkpoint::load_latest(&path).ok().and_then(|(ck, _)| {
        (ck.meta.protocol == PROTO
            && ck.meta.exponent == p
            && ck.meta.config == json!({"b1": b1})
            && ck.residues.len() == 1)
            .then(|| (ck.meta.k, ck.residues[0].clone()))
    });
    let mut be = open(w.kind, p, w.device)?;
    let identity = be.identity();
    let stop = || STOP.load(Ordering::SeqCst) || deadline.is_some_and(|d| Instant::now() >= d);
    let mut save_err = None;
    let mut save = |i: u64, x: &mersenne_forge::residue::Residue| {
        let ck = Checkpoint {
            meta: CheckpointMeta {
                format_version: FORMAT_VERSION,
                protocol: PROTO.into(),
                exponent: p,
                k: i,
                state: "provisional".into(),
                verified_k: 0,
                n_errors: 0,
                proof_durable_through: None,
                backend: identity.clone(),
                config: json!({"b1": b1}),
                requested: json!({}),
                elapsed_secs: 0.0,
                crate_version: env!("CARGO_PKG_VERSION").into(),
            },
            residues: vec![x.clone()],
        };
        if let Err(e) = ck.save(&path) {
            save_err.get_or_insert(e.to_string());
        }
    };
    let o = mersenne_forge::factor::pm1_resumable(be.as_mut(), b1, b2, resume, &stop, &mut save)
        .map_err(|e| fail(EXIT_EXECUTION, e))?;
    if let Some(e) = save_err {
        eprintln!("M{p}: P-1 progress not saved: {e}");
    }
    for n in &o.notes {
        eprintln!("M{p}: {n}");
    }
    if o.complete {
        for f in [prev_path(&path), path] {
            let _ = std::fs::remove_file(f);
        }
    }
    Ok(o)
}

/// Run one CERT line: square the start value from `M<p>.cert` (next to
/// the worktodo file) and report the hash. Ok(true): reported, the line
/// and the start value file removed (gpuowl removes it too); Ok(false):
/// stopped, the line stays.
fn run_cert_line(
    w: &WorkArgs,
    c: &mersenne_forge::cert::CertLine,
    deadline: Option<Instant>,
) -> Result<bool, Failure> {
    use mersenne_forge::cert;
    let p = c.exponent;
    // A result already made durable (a crash after the commit) is not
    // computed and reported again.
    let done_before = std::fs::read_to_string(journal_path(&w.results))
        .unwrap_or_default()
        .lines()
        .any(|l| l == c.line);
    if done_before {
        replace_line(&w.worktodo, &c.line, None)?;
        return Ok(true);
    }
    let dir = w
        .worktodo
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let path = cert::start_path(dir, p);
    let start = cert::read_start(&path, p).map_err(|e| fail(EXIT_STATE, e))?;
    let mut be = open(w.kind, p, w.device)?;
    let mut ctl = Control {
        stop: Some(&STOP),
        deadline,
        ..Control::default()
    };
    let (x, ev) =
        prp::gerbicz_squarings(&mut be, &start, c.squarings, &mut ctl).map_err(protocol_failure)?;
    let Some(x) = x else {
        return Ok(false);
    };
    let h = cert::hash(&x);
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let line = cert::result_line(
        c,
        &h,
        ev.errors_total,
        be.identity().kind.as_str(),
        &mersenne_forge::worktodo::utc_timestamp(secs),
    );
    commit_line(w, &c.line, &c.line, &line.to_string(), None)?;
    let _ = std::fs::remove_file(&path);
    println!("M{p}: CERT of {} squarings, hash {h}", c.squarings);
    Ok(true)
}

/// Run one factoring line. Ok(true): reported and removed; Ok(false):
/// stopped by the budget or a stop request, the line stays.
fn run_factor_line(
    w: &WorkArgs,
    f: &mersenne_forge::factor::FactorLine,
    deadline: Option<Instant>,
) -> Result<bool, Failure> {
    use mersenne_forge::factor::{self, FactorWork};
    let p = f.exponent;
    let stop = || STOP.load(Ordering::SeqCst) || deadline.is_some_and(|d| Instant::now() >= d);
    let now = || {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        mersenne_forge::worktodo::utc_timestamp(secs)
    };
    let done_before = std::fs::read_to_string(journal_path(&w.results))
        .unwrap_or_default()
        .lines()
        .any(|l| l == f.line);
    if done_before {
        replace_line(&w.worktodo, &f.line, None)?;
        return Ok(true);
    }
    match f.work {
        FactorWork::Tf { bit_lo, bit_hi } => {
            // Progress: the next untested k, as decimal text.
            let state = w.checkpoint_dir.join(format!(
                "M{p}-tf-{}.state",
                line_key(f.aid.as_deref(), &f.line)
            ));
            create_dir_all_durable(&w.checkpoint_dir)
                .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", w.checkpoint_dir.display())))?;
            let from = std::fs::read_to_string(&state)
                .ok()
                .and_then(|t| t.trim().parse::<u128>().ok());
            let mut last_save = Instant::now();
            let mut save_err = None;
            let mut o = factor::trial_factor(p, bit_lo, bit_hi, from, &stop, &mut |k| {
                if last_save.elapsed() >= Duration::from_secs(30) {
                    last_save = Instant::now();
                    if let Err(e) = mersenne_forge::checkpoint::write_atomic(
                        &state,
                        k.to_string().as_bytes(),
                        false,
                    ) {
                        save_err.get_or_insert(e.to_string());
                    }
                }
            })
            .map_err(|e| fail(EXIT_EXECUTION, e))?;
            if let Some(e) = save_err {
                eprintln!("M{p}: TF progress not saved: {e}");
            }
            if !o.complete {
                mersenne_forge::checkpoint::write_atomic(
                    &state,
                    o.next_k.to_string().as_bytes(),
                    false,
                )
                .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", state.display())))?;
                return Ok(false);
            }
            // ponytail: a factor found early in the range does not end it, so
            // the whole range is reported as searched (rangecomplete).
            // Factors the line already named are not new.
            o.factors = o.factors.iter().filter_map(|x| f.new_part(x)).collect();
            let line = factor::tf_result_line(f, &o, &now());
            commit_line(w, &f.line, &f.line, &line.to_string(), None)?;
            let _ = std::fs::remove_file(&state);
            println!("M{p}: TF {bit_lo}-{bit_hi} factors {:?}", o.factors);
            Ok(true)
        }
        FactorWork::Pm1 { b1, b2 } => {
            let key = line_key(f.aid.as_deref(), &f.line);
            let mut o = work_pm1(w, p, &key, b1, b2, deadline)?;
            if !o.complete {
                return Ok(false);
            }
            o.factor = o.factor.and_then(|x| f.new_part(&x));
            let line = factor::pm1_result_line(f, &o, &now());
            commit_line(w, &f.line, &f.line, &line.to_string(), None)?;
            println!("M{p}: P-1 B1={b1} B2={b2} factor {:?}", o.factor);
            Ok(true)
        }
        // P+1 and ECM run on the host in one go with no saved progress:
        // a stop loses the run, and the line stays.
        FactorWork::Pp1 { b1, b2, nth_run } => {
            let mut o =
                factor::pp1(p, b1, b2, nth_run, &stop).map_err(|e| fail(EXIT_EXECUTION, e))?;
            if !o.complete {
                return Ok(false);
            }
            o.factor = o.factor.and_then(|x| f.new_part(&x));
            let line = factor::pp1_result_line(f, &o, &now());
            commit_line(w, &f.line, &f.line, &line.to_string(), None)?;
            println!("M{p}: P+1 B1={b1} B2={b2} factor {:?}", o.factor);
            Ok(true)
        }
        FactorWork::Ecm {
            b1,
            b2,
            curves,
            sigma,
        } => {
            // ponytail: without a sigma on the line, one from the clock;
            // Prime95 also picks a random one.
            let sigma = sigma.unwrap_or_else(|| {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.subsec_nanos());
                6 + u64::from(t)
            });
            let mut o = factor::ecm(p, b1, b2, curves, sigma, &stop)
                .map_err(|e| fail(EXIT_EXECUTION, e))?;
            if o.factor.is_none() && o.curves < curves {
                return Ok(false);
            }
            o.factor = o.factor.and_then(|x| f.new_part(&x));
            let line = factor::ecm_result_line(f, &o, &now());
            commit_line(w, &f.line, &f.line, &line.to_string(), None)?;
            println!(
                "M{p}: ECM B1={b1} B2={b2} {} curves, factor {:?}",
                o.curves, o.factor
            );
            Ok(true)
        }
        FactorWork::EcmStage2 {
            ref file,
            b2,
            skip,
            curves,
        } => {
            let dir = w
                .worktodo
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            let path = dir.join(file);
            let text = std::fs::read_to_string(&path)
                .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", path.display())))?;
            let saved = factor::parse_gmp_ecm_save(&text)
                .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", path.display())))?;
            let want = curves.map(|c| c as usize);
            let mut o = factor::ecm_stage2_gmp(p, &saved, b2, skip as usize, want, &stop)
                .map_err(|e| fail(EXIT_EXECUTION, e))?;
            let available = saved.len().saturating_sub(skip as usize);
            let target = want.map_or(available, |c| c.min(available));
            if o.factor.is_none() && (o.curves as usize) < target {
                return Ok(false);
            }
            o.factor = o.factor.and_then(|x| f.new_part(&x));
            let line = factor::ecm_result_line(f, &o, &now());
            commit_line(w, &f.line, &f.line, &line.to_string(), None)?;
            println!(
                "M{p}: ECM stage 2 from {} ({} curves), factor {:?}",
                path.display(),
                o.curves,
                o.factor
            );
            Ok(true)
        }
    }
}

/// Delete the proof points of a finished run while holding the store's
/// lock, removing only point files; anything else in the directory stays.
fn remove_proof_points(residues: &Path) -> Result<(), String> {
    let lock_path = residues.join(".lock");
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&lock_path)
        .map_err(|e| format!("{}: {e}", lock_path.display()))?;
    match f.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            return Err(format!("{} is in use by another run", residues.display()))
        }
        Err(std::fs::TryLockError::Error(e)) => return Err(e.to_string()),
    }
    let entries =
        std::fs::read_dir(residues).map_err(|e| format!("{}: {e}", residues.display()))?;
    let mut others = 0usize;
    for e in entries.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.parse::<u64>().is_ok() {
            std::fs::remove_file(e.path()).map_err(|x| format!("{}: {x}", e.path().display()))?;
        } else if name != ".lock" {
            others += 1;
        }
    }
    if others == 0 {
        let _ = std::fs::remove_file(&lock_path);
        let _ = std::fs::remove_dir(residues);
    }
    drop(f);
    Ok(())
}

/// Resume from a checkpoint, falling back to `.prev` when the primary loads
/// but fails its resume validation (its on-load check or a missing proof
/// residue). The rejected primary is moved aside. Returns the run, the
/// checkpoint used, its file, and a note when the fallback was taken.
type Resumed = (
    Result<RunReport, ProtocolError>,
    Checkpoint,
    PathBuf,
    Option<String>,
);

fn resume_with_fallback(
    be: &mut Box<dyn mersenne_forge::backend::Backend>,
    path: &Path,
    proof_dir: Option<&Path>,
    ctl: &mut Control,
) -> Result<Resumed, Failure> {
    let (mut ck, mut used) =
        Checkpoint::load_latest(path).map_err(|e| fail(EXIT_STATE, e.to_string()))?;
    let mut note = None;
    let mut result = resume_from(be, &ck, path, proof_dir, ctl)?;
    if used == path
        && matches!(
            result,
            Err(ProtocolError::CheckpointRejected { .. }
                | ProtocolError::ProofResidueMissing { .. })
        )
    {
        let prev = prev_path(path);
        if let Ok(pck) = Checkpoint::load_file(&prev) {
            let mut aside = path.as_os_str().to_owned();
            aside.push(".rejected");
            std::fs::rename(path, &aside)
                .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", path.display())))?;
            let why = result.err().map(|e| e.to_string()).unwrap_or_default();
            note = Some(format!(
                "{} was not usable ({why}); it was moved to {} and the run resumed from {}",
                path.display(),
                PathBuf::from(&aside).display(),
                prev.display()
            ));
            ck = pck;
            used = prev;
            result = resume_from(be, &ck, path, proof_dir, ctl)?;
            // The previous state failed too: put the newer file back, so a
            // failed attempt never sidelines the newest valid checkpoint.
            if matches!(
                result,
                Err(ProtocolError::CheckpointRejected { .. }
                    | ProtocolError::ProofResidueMissing { .. })
            ) && !path.exists()
            {
                std::fs::rename(&aside, path)
                    .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", path.display())))?;
                note = None;
            }
        }
    }
    Ok((result, ck, used, note))
}

/// Refuse argument combinations that would lose work: a proof power without
/// a proof directory, and a results file that is the worktodo file or lies
/// inside the checkpoint or proof directory (cleanup would delete it).
fn check_work_paths(w: &WorkArgs) -> Result<(), Failure> {
    if w.proof_power.is_some() && w.proof_dir.is_none() {
        return Err(fail(2, "--proof-power needs --proof-dir"));
    }
    let abs = |p: &Path| std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let (results, todo) = (abs(&w.results), abs(&w.worktodo));
    if results == todo {
        return Err(fail(2, "--results and --worktodo name the same file"));
    }
    for (what, dir) in [
        ("--checkpoint-dir", Some(&w.checkpoint_dir)),
        ("--proof-dir", w.proof_dir.as_ref()),
    ] {
        if let Some(d) = dir {
            let d = abs(d);
            if results.starts_with(&d) || todo.starts_with(&d) {
                return Err(fail(
                    2,
                    format!("the results and worktodo files must lie outside {what}"),
                ));
            }
        }
    }
    Ok(())
}

fn cmd_work(w: &WorkArgs) -> Result<u8, Failure> {
    use mersenne_forge::worktodo::{self as wt, WorkKind};
    check_work_paths(w)?;
    let deadline = deadline(&w.budget, w.kind)?;
    // One worker per worktodo file.
    let _lock = lock_checkpoint(&w.worktodo)?;
    let mut done = 0usize;
    // Assignments this session could not start, and messages already shown.
    let mut skipped = std::collections::HashSet::new();
    let mut warned = std::collections::HashSet::new();
    let leave = |a: &wt::Assignment, why: &str, skipped: &mut std::collections::HashSet<String>| {
        eprintln!("left in place: M{}: {why}", a.exponent);
        skipped.insert(a.line.clone());
    };
    loop {
        if STOP.load(Ordering::SeqCst) || deadline.is_some_and(|d| Instant::now() >= d) {
            println!("stopped after {done} assignments (stop requested or budget used)");
            return Ok(EXIT_INCOMPLETE);
        }
        let text = std::fs::read_to_string(&w.worktodo)
            .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", w.worktodo.display())))?;
        // CERT lines first, as gpuowl does: they are short and the
        // server waits on them.
        let next_cert =
            text.lines()
                .find_map(|raw| match mersenne_forge::cert::parse_cert_line(raw) {
                    Ok(Some(c)) if !skipped.contains(&c.line) => Some(c),
                    Err(e) => {
                        if warned.insert(e.clone()) {
                            eprintln!("left in place: {e}");
                        }
                        None
                    }
                    _ => None,
                });
        if let Some(c) = next_cert {
            w.status(
                "running",
                Some(&c.line),
                json!({"done": done, "skipped": skipped.len()}),
            );
            match run_cert_line(w, &c, deadline) {
                Ok(true) => done += 1,
                Ok(false) => {
                    println!("M{}: CERT not finished; the line stays", c.exponent);
                    return Ok(EXIT_INCOMPLETE);
                }
                Err(e) => {
                    eprintln!("left in place: M{}: {}", c.exponent, e.message);
                    skipped.insert(c.line.clone());
                }
            }
            continue;
        }
        let mut next = None;
        let mut next_factor = None;
        for raw in text.lines() {
            if mersenne_forge::cert::parse_cert_line(raw).is_ok_and(|c| c.is_some()) {
                continue;
            }
            match mersenne_forge::factor::parse_factor_line(raw) {
                Ok(Some(f)) if f.exponent > mersenne_forge::factor::work_max_p(&f.work) => {
                    let e = format!(
                        "{} of M{} is left to another program (above {})",
                        f.line.split('=').next().unwrap_or("factoring"),
                        f.exponent,
                        mersenne_forge::factor::work_max_p(&f.work)
                    );
                    if warned.insert(e.clone()) {
                        eprintln!("left in place: {e}");
                    }
                    continue;
                }
                Ok(Some(f)) if !skipped.contains(&f.line) => {
                    next_factor = Some(f);
                    break;
                }
                Ok(Some(_)) => continue,
                Err(e) => {
                    if warned.insert(e.clone()) {
                        eprintln!("left in place: {e}");
                    }
                    continue;
                }
                Ok(None) => {}
            }
            match wt::parse_line(raw) {
                Ok(Some(a)) if !skipped.contains(&a.line) => {
                    next = Some(a);
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    if warned.insert(e.clone()) {
                        eprintln!("left in place: {e}");
                    }
                }
            }
        }
        if let Some(f) = next_factor {
            w.status(
                "running",
                Some(&f.line),
                json!({"done": done, "skipped": skipped.len()}),
            );
            match run_factor_line(w, &f, deadline) {
                Ok(true) => done += 1,
                Ok(false) => {
                    println!("M{}: factoring not finished; the line stays", f.exponent);
                    return Ok(EXIT_INCOMPLETE);
                }
                Err(e) => {
                    eprintln!("left in place: M{}: {}", f.exponent, e.message);
                    skipped.insert(f.line.clone());
                }
            }
            continue;
        }
        let Some(a) = next else {
            println!(
                "no runnable assignment left; {done} completed, {} could not start",
                skipped.len()
            );
            return Ok(if skipped.is_empty() { 0 } else { EXIT_STATE });
        };
        if already_reported(&w.results, &a) {
            eprintln!(
                "M{}: already in {}; removing the assignment",
                a.exponent,
                w.results.display()
            );
            commit_result(w, &a, None)?;
            // Its checkpoints must not be adopted by a later line.
            let ck = assignment_checkpoint(&w.checkpoint_dir, &a);
            for f in [prev_path(&ck), ck] {
                let _ = std::fs::remove_file(f);
            }
            continue;
        }
        if a.wants_pm1() {
            // P-1 first, as upstream does; a factor ends the assignment,
            // otherwise the line stays with tests_saved = 0 for the test.
            use mersenne_forge::factor::{self, FactorLine, FactorWork};
            let (b1, b2) = factor::pfactor_bounds(a.exponent);
            let Some(next_line) = factor::with_pm1_done(&a.line, a.exponent) else {
                leave(&a, "cannot rewrite the line after P-1", &mut skipped);
                continue;
            };
            let fl = FactorLine {
                work: FactorWork::Pm1 { b1, b2 },
                exponent: a.exponent,
                aid: a.aid.clone(),
                line: a.line.clone(),
                known_factors: a.known_factors.clone(),
            };
            // A P-1 step recorded before a crash: only the rewrite is left.
            let entry = format!("pm1:{}", a.line);
            let stepped = std::fs::read_to_string(journal_path(&w.results))
                .unwrap_or_default()
                .lines()
                .any(|l| l == entry);
            if stepped {
                replace_line(&w.worktodo, &a.line, Some(&next_line))?;
                continue;
            }
            let key = line_key(a.aid.as_deref(), &a.line);
            match work_pm1(w, a.exponent, &key, b1, b2, deadline) {
                Ok(o) if !o.complete => {
                    println!("M{}: P-1 not finished; the line stays", a.exponent);
                    return Ok(EXIT_INCOMPLETE);
                }
                Ok(mut o) => {
                    // A factor the line already names is not new: the
                    // cofactor test still has to run.
                    o.factor = o.factor.and_then(|x| fl.new_part(&x));
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs());
                    let line = factor::pm1_result_line(&fl, &o, &wt::utc_timestamp(now));
                    // A factor ends the assignment; otherwise only the P-1
                    // step is done and the line comes back for the test.
                    let (entry, replacement) = match o.factor {
                        Some(_) => (a.line.clone(), None),
                        None => (entry, Some(next_line.as_str())),
                    };
                    commit_line(w, &a.line, &entry, &line.to_string(), replacement)?;
                    println!("M{}: P-1 B1={b1} B2={b2} factor {:?}", a.exponent, o.factor);
                }
                Err(e) => leave(&a, &e.message, &mut skipped),
            }
            continue;
        }
        let p = a.exponent;
        w.status(
            "running",
            Some(&a.line),
            json!({"done": done, "skipped": skipped.len()}),
        );
        // Residue types 2 and 4 stop one squaring early; no proof exists
        // for that chain, as in Prime95.
        let before_last = a.kind == WorkKind::Prp && matches!(a.residue_type, 2 | 4);
        let proof_dir = match a.kind {
            WorkKind::Prp if !before_last => w.proof_dir.clone(),
            _ => None,
        };
        let ckpt = assignment_checkpoint(&w.checkpoint_dir, &a);
        let prp_cfg = PrpConfig {
            block_size: 1000,
            check_step: None,
            proof_power: proof_dir.as_ref().and(w.proof_power),
            proof_dir: proof_dir.clone(),
            verify_proof: true,
            checkpoint: Some(ckpt.clone()),
            max_batch: 1000,
            before_last,
        };
        let mut ctl = Control {
            stop: Some(&STOP),
            deadline,
            ..Control::default()
        };
        // Anything that keeps this assignment from starting leaves it in
        // place and moves on to the next one.
        let started = (|| -> Result<_, Failure> {
            let lock = lock_checkpoint(&ckpt)?;
            let resumed = ckpt.exists() || prev_path(&ckpt).exists();
            if resumed {
                let (ck, _) =
                    Checkpoint::load_latest(&ckpt).map_err(|e| fail(EXIT_STATE, e.to_string()))?;
                // Compared as parsed assignments, so a checkpoint written
                // before a field was added (it takes its default) still
                // matches its line.
                let stored = ck
                    .meta
                    .requested
                    .get("assignment")
                    .and_then(|v| serde_json::from_value::<wt::Assignment>(v.clone()).ok());
                if stored.as_ref() != Some(&a) {
                    return Err(fail(
                        EXIT_STATE,
                        format!(
                            "{} belongs to another assignment; finish it with `resume` or delete it",
                            ckpt.display()
                        ),
                    ));
                }
                // Resume on the backend the run started on; a portable
                // checkpoint runs on the backend this session uses.
                let kind = if ck.meta.backend.kind == mersenne_forge::checkpoint::PORTABLE {
                    Some(w.kind)
                } else {
                    BackendKind::parse(&ck.meta.backend.kind)
                };
                let kind = kind.ok_or_else(|| {
                    fail(
                        EXIT_UNSUPPORTED,
                        format!("unknown backend {:?} in checkpoint", ck.meta.backend.kind),
                    )
                })?;
                let be = open(kind, p, w.device)?;
                return Ok((lock, true, be));
            }
            if let (Some(dir), WorkKind::Prp) = (&proof_dir, a.kind) {
                let eff = prp::effective_config(p, &prp_cfg).map_err(protocol_failure)?;
                let power = eff.proof_power.unwrap_or(0);
                create_dir_all_durable(dir)
                    .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", dir.display())))?;
                // An earlier proof of this exponent and power may not have
                // been uploaded yet; never overwrite it.
                let existing = dir.join(format!("{p}-{power}.proof"));
                if existing.exists() {
                    return Err(fail(
                        EXIT_STATE,
                        format!(
                            "{} exists; upload or move it before this assignment runs",
                            existing.display()
                        ),
                    ));
                }
                let need = wt::proof_disk_need(p, power);
                match wt::free_bytes(dir) {
                    Some(free) if free < need => {
                        return Err(fail(
                            EXIT_STATE,
                            format!(
                                "the proof needs about {need} bytes in {}, {free} are free",
                                dir.display()
                            ),
                        ))
                    }
                    Some(_) => {}
                    None => eprintln!(
                        "M{p}: free disk space unknown; the proof needs about {need} bytes"
                    ),
                }
            }
            let be = open(w.kind, p, w.device)?;
            Ok((lock, false, be))
        })();
        let (ck_lock, resumed, mut be) = match started {
            Ok(x) => x,
            Err(f) => {
                leave(&a, &f.message, &mut skipped);
                continue;
            }
        };
        let (result, note) = if resumed {
            match resume_with_fallback(&mut be, &ckpt, None, &mut ctl) {
                // A resumed run that cannot continue (another build, a
                // configuration mismatch) is this assignment's problem only.
                Ok((Err(e), ..)) if !matches!(e, ProtocolError::Interrupted { .. }) => {
                    leave(&a, &e.to_string(), &mut skipped);
                    continue;
                }
                Ok((r, _, _, note)) => (r, note),
                Err(f) => {
                    leave(&a, &f.message, &mut skipped);
                    continue;
                }
            }
        } else {
            let r = match a.kind {
                WorkKind::Prp => prp::run_prp(
                    &mut be,
                    p,
                    &prp_cfg,
                    json!({"assignment": a}),
                    None,
                    &mut ctl,
                ),
                WorkKind::Ll => ll::run_ll(
                    &mut be,
                    p,
                    &LlConfig {
                        log_step: 20_000,
                        jacobi: JacobiCadence::Auto,
                        checkpoint: Some(ckpt.clone()),
                        max_batch: 1000,
                    },
                    json!({"assignment": a}),
                    None,
                    &mut ctl,
                ),
            };
            (r, None)
        };
        if let Some(n) = note {
            eprintln!("M{p}: {n}");
        }
        // A protocol error (mismatched or unreadable state, a missing proof
        // point, a failed check) belongs to this assignment only.
        let report = match result {
            Ok(r) => r,
            Err(e) => {
                leave(&a, &e.to_string(), &mut skipped);
                continue;
            }
        };
        let r = match (&report.process, report.result.as_ref()) {
            (ProcessState::Completed, Some(r))
                if report.evidence.proof_stage != ProofStage::Interrupted =>
            {
                r
            }
            (ProcessState::Incomplete { reason, .. }, _) => {
                println!("M{p}: not completed ({reason}); the assignment stays, and the next session resumes it");
                return Ok(EXIT_INCOMPLETE);
            }
            _ => {
                println!(
                    "M{p}: not completed (stopped during the proof stage); the assignment stays"
                );
                return Ok(EXIT_INCOMPLETE);
            }
        };
        // A proof that failed or was rejected is not reported; the result is,
        // as upstream does.
        let published = matches!(
            report.evidence.proof_stage,
            ProofStage::Verified | ProofStage::Generated
        );
        if !published && report.evidence.proof_stage != ProofStage::NotRequested {
            eprintln!(
                "M{p}: proof stage {}; reporting the result without a proof",
                report.evidence.proof_stage.name()
            );
        }
        let proof_info = report
            .evidence
            .proof
            .as_ref()
            .filter(|_| published)
            .map(|e| -> Result<wt::ProofInfo, Failure> {
                Ok(wt::ProofInfo {
                    power: e.power,
                    md5: wt::file_md5(&e.path)
                        .map_err(|x| fail(EXIT_STATE, format!("{}: {x}", e.path.display())))?,
                })
            })
            .transpose()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let line = wt::result_line(
            &a,
            r,
            &report,
            proof_info.as_ref(),
            be.identity().kind.as_str(),
            &wt::utc_timestamp(now),
        );
        commit_result(w, &a, Some(&line.to_string()))?;
        // The result is durable: the checkpoints and, after a verified proof,
        // the proof points are no longer needed. The proof file is kept. The
        // points are found from the proof the run actually wrote.
        for f in [ckpt.clone(), prev_path(&ckpt)] {
            let _ = std::fs::remove_file(f);
        }
        if report.evidence.proof_stage == ProofStage::Verified {
            if let Some(dir) = report
                .evidence
                .proof
                .as_ref()
                .and_then(|e| e.path.parent().map(Path::to_path_buf))
            {
                if let Err(e) = remove_proof_points(&proof::ProofStore::residue_dir(&dir, p)) {
                    eprintln!("M{p}: proof points kept: {e}");
                }
            }
        }
        drop(ck_lock);
        done += 1;
        println!("M{p}: {:?} res64 {}", r.verdict, r.res64);
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_primenet(
    user: &str,
    password_file: &Path,
    worktodo: &Path,
    results: &Path,
    proof_dir: Option<PathBuf>,
    worktype: Option<&str>,
    ahead: usize,
    state_dir: &Path,
) -> Result<u8, Failure> {
    use mersenne_forge::primenet as pn;
    let code = worktype
        .map(|w| pn::worktype_code(w).ok_or_else(|| fail(2, format!("unknown work type {w:?}"))))
        .transpose()?;
    let password = std::fs::read_to_string(password_file)
        .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", password_file.display())))?;
    let session = pn::Session::login(state_dir, user, password.trim_end_matches(['\r', '\n']))
        .map_err(|e| fail(EXIT_EXECUTION, e))?;
    let x = pn::Exchange {
        results: results.to_path_buf(),
        sent: state_dir.join("sent.txt"),
        refused: state_dir.join("refused.txt"),
        proof_dir,
    };
    let mut report = pn::exchange(&session, user, &x).map_err(|e| fail(EXIT_EXECUTION, e))?;
    // The worktodo file is shared with `work`: fetch over the network
    // first, then append each line under its lock.
    if let Some(code) = code {
        for _ in pn::runnable_lines(worktodo)..ahead {
            let line = session
                .fetch_assignment(code)
                .map_err(|e| fail(EXIT_EXECUTION, e))?;
            let _g = lck(worktodo)?;
            pn::append_line(worktodo, &line).map_err(|e| fail(EXIT_STATE, e))?;
            report.fetched.push(line);
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serializable")
    );
    Ok(if report.refused + report.upload_failed.len() > 0 {
        EXIT_EXECUTION
    } else {
        0
    })
}

#[cfg(feature = "gpu")]
fn tf_on_gpu(
    p: u32,
    from: u32,
    to: u32,
    device: Option<u32>,
    stop: &dyn Fn() -> bool,
) -> Result<mersenne_forge::factor::TfOutcome, Failure> {
    use mersenne_forge::backend::gpu::tf::{GpuTf, Q_LIMIT};
    let tf = GpuTf::new(device).map_err(|e| fail(EXIT_UNSUPPORTED, e))?;
    mersenne_forge::factor::trial_factor_with(p, from, to, None, stop, &mut |_| {}, &mut |p, qs| {
        let (small, big): (Vec<u128>, Vec<u128>) = qs.iter().partition(|&&q| q < Q_LIMIT);
        let mut hits: Vec<u128> = match tf.test(p, &small)? {
            Some(h) => h,
            // More hits than slots: the host tests this batch.
            None => small
                .iter()
                .copied()
                .filter(|&q| mersenne_forge::factor::divides(p, q))
                .collect(),
        };
        hits.extend(
            big.into_iter()
                .filter(|&q| mersenne_forge::factor::divides(p, q)),
        );
        Ok(hits)
    })
    .map_err(|e| fail(EXIT_EXECUTION, e))
}

#[cfg(not(feature = "gpu"))]
fn tf_on_gpu(
    _: u32,
    _: u32,
    _: u32,
    _: Option<u32>,
    _: &dyn Fn() -> bool,
) -> Result<mersenne_forge::factor::TfOutcome, Failure> {
    Err(fail(
        EXIT_UNSUPPORTED,
        "this build has no GPU support (build with --features gpu)",
    ))
}

fn cmd_primenet_v5(dir: &Path, action: V5Action) -> Result<u8, Failure> {
    use mersenne_forge::primenet as pn;
    use mersenne_forge::primenet_v5 as v5;
    let state = |e: String| fail(EXIT_STATE, e);
    let net = |e: String| fail(EXIT_EXECUTION, e);
    create_dir_all_durable(dir).map_err(|e| state(format!("{}: {e}", dir.display())))?;
    // A GUID file holds one line; it is created once and kept.
    let id = |name: &str| -> Result<String, Failure> {
        let path = dir.join(name);
        if let Ok(t) = std::fs::read_to_string(&path) {
            return Ok(t.trim().to_string());
        }
        let g = v5::new_guid().map_err(state)?;
        pn::append_line(&path, &g).map_err(state)?;
        Ok(g)
    };
    let guid = id("guid")?;
    match action {
        V5Action::Register {
            user,
            computer,
            cpu,
            mhz,
            memory_mib,
            l2_kib,
            hours,
            worktype,
        } => {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get()) as u32;
            let m = v5::Machine {
                guid: guid.clone(),
                hardware_guid: id("hardware-guid")?,
                cpu_brand: cpu,
                cpu_features: String::new(),
                l1_kib: 32,
                l2_kib,
                cores,
                threads_per_core: 1,
                memory_mib,
                mhz,
                hours_per_day: hours,
                user,
                computer_name: computer,
            };
            let r = v5::send(&v5::register_params(&m)).map_err(net)?;
            println!(
                "registered {guid} as user {:?}, computer {:?}",
                r.get("u").map_or("", String::as_str),
                r.get("cn").map_or("", String::as_str)
            );
            if let Some(w) = worktype {
                let code = pn::worktype_code(&w)
                    .ok_or_else(|| fail(2, format!("unknown work type {w:?}")))?;
                v5::send(&v5::options_params(&guid, 0, code)).map_err(net)?;
                println!("work preference {code}");
            }
            Ok(0)
        }
        V5Action::Fetch {
            worktodo,
            count,
            worker,
            cert,
        } => {
            for _ in 0..count {
                let r = v5::send(&v5::assignment_params(&guid, worker, cert)).map_err(net)?;
                let line = v5::assignment_line(&r).map_err(net)?;
                // A CERT line needs its start value next to the worktodo file.
                if let Some(c) = mersenne_forge::cert::parse_cert_line(&line).map_err(net)? {
                    let wdir = worktodo
                        .parent()
                        .filter(|d| !d.as_os_str().is_empty())
                        .unwrap_or(Path::new("."));
                    let aid = c.aid.clone().unwrap_or_default();
                    v5::download_cert(&aid, &mersenne_forge::cert::start_path(wdir, c.exponent))
                        .map_err(net)?;
                }
                pn::append_line(&worktodo, &line).map_err(state)?;
                println!("{line}");
            }
            Ok(0)
        }
        V5Action::Submit { results } => {
            let log = dir.join("v5-sent.txt");
            let sent = std::fs::read_to_string(&log).unwrap_or_default();
            let text = std::fs::read_to_string(&results)
                .map_err(|e| state(format!("{}: {e}", results.display())))?;
            let mut failed = 0;
            for line in text.lines().filter(|l| l.trim_start().starts_with('{')) {
                if sent.lines().any(|s| s == line) {
                    continue;
                }
                let params = match v5::result_params(&guid, line) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("not sent: {e}: {line}");
                        continue;
                    }
                };
                match v5::send(&params) {
                    Ok(r) => {
                        pn::append_line(&log, line).map_err(state)?;
                        println!(
                            "sent: {}",
                            r.get("pnErrorDetail").map_or("", String::as_str)
                        );
                    }
                    Err(e) => {
                        failed += 1;
                        eprintln!("refused or failed: {e}: {line}");
                    }
                }
            }
            Ok(if failed == 0 { 0 } else { EXIT_EXECUTION })
        }
        V5Action::Progress {
            aid,
            percent,
            eta_hours,
            worker,
        } => {
            let eta = (eta_hours.max(0.0) * 3600.0) as u64;
            v5::send(&v5::progress_params(
                &guid,
                &aid,
                worker,
                percent / 100.0,
                eta,
                86_400,
            ))
            .map_err(net)?;
            println!("progress sent for {aid}");
            Ok(0)
        }
        V5Action::Unreserve { aid } => {
            v5::send(&v5::unreserve_params(&guid, &aid)).map_err(net)?;
            println!("unreserved {aid}");
            Ok(0)
        }
    }
}

fn cmd_tune(p: u32, kind: BackendKind, device: Option<u32>, iters: u64) -> Result<u8, Failure> {
    use mersenne_forge::backend::Backend;
    let unsupported =
        |e: mersenne_forge::backend::BackendError| fail(EXIT_UNSUPPORTED, e.to_string());
    let mut variants: Vec<(String, Box<dyn Backend>)> =
        vec![(String::new(), open(kind, p, device)?)];
    match kind {
        BackendKind::Dwt => {
            use mersenne_forge::backend::dwt::DwtBackend;
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let mut b = DwtBackend::new(p).map_err(unsupported)?;
            b.use_packed();
            if b.is_packed() {
                variants.push(("MERSENNE_FORGE_DWT_PACKED=1".into(), Box::new(b)));
            }
            if threads >= 2 {
                for packed in [false, true] {
                    let mut b = DwtBackend::new(p).map_err(unsupported)?;
                    if packed {
                        b.use_packed();
                    }
                    b.use_threads(threads);
                    if b.threads() > 1 && (!packed || b.is_packed()) {
                        let setting = format!(
                            "{}MERSENNE_FORGE_DWT_THREADS={threads}",
                            if packed {
                                "MERSENNE_FORGE_DWT_PACKED=1 "
                            } else {
                                ""
                            }
                        );
                        variants.push((setting.to_string(), Box::new(b)));
                    }
                }
            }
        }
        #[cfg(feature = "gpu")]
        BackendKind::Gpu => {
            use mersenne_forge::backend::gpu::{GpuBackend, GpuOptions};
            for (radix4, packed, setting) in [
                (true, false, "MERSENNE_FORGE_GPU_RADIX4=1"),
                (false, true, "MERSENNE_FORGE_GPU_PACKED=1"),
                (
                    true,
                    true,
                    "MERSENNE_FORGE_GPU_PACKED=1 MERSENNE_FORGE_GPU_RADIX4=1",
                ),
            ] {
                let b = GpuBackend::new(
                    p,
                    &GpuOptions {
                        device,
                        radix4,
                        packed,
                    },
                )
                .map_err(unsupported)?;
                variants.push((setting.to_string(), Box::new(b)));
            }
        }
        _ => {}
    }
    let x0 = mersenne_forge::residue::Residue::from_u64(p, 3);
    let mut rows = Vec::new();
    let mut reference = None;
    for (setting, be) in variants.iter_mut() {
        let r = be.alloc().map_err(unsupported)?;
        be.import(r, &x0).map_err(unsupported)?;
        be.square(r, iters.min(10)).map_err(unsupported)?;
        be.sync().map_err(unsupported)?;
        let t = Instant::now();
        be.square(r, iters).map_err(unsupported)?;
        be.sync().map_err(unsupported)?;
        let secs = t.elapsed().as_secs_f64();
        let out = be.export(r).map_err(unsupported)?;
        be.free(r).map_err(unsupported)?;
        let agrees = *reference.get_or_insert_with(|| out.clone()) == out;
        rows.push(json!({
            "plan": be.identity().plan,
            "setting": if setting.is_empty() { "(default)" } else { setting.as_str() },
            "ms_per_iteration": 1e3 * secs / iters.max(1) as f64,
            "agrees_with_default": agrees,
        }));
    }
    let best = rows
        .iter()
        .filter(|r| r["agrees_with_default"] == true)
        .min_by(|a, b| {
            a["ms_per_iteration"]
                .as_f64()
                .partial_cmp(&b["ms_per_iteration"].as_f64())
                .expect("finite times")
        })
        .cloned();
    let disagree = rows.iter().any(|r| r["agrees_with_default"] == false);
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"exponent": p, "iterations": iters, "variants": rows, "fastest": best})
        )
        .expect("serializable")
    );
    Ok(if disagree { EXIT_REJECTED } else { 0 })
}

/// One worker of `supervise`: the `work` arguments for one device.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerSpec {
    worktodo: PathBuf,
    results: PathBuf,
    checkpoint_dir: PathBuf,
    #[serde(default = "cpu_name")]
    backend: String,
    device: Option<u32>,
    budget: Option<String>,
    proof_dir: Option<PathBuf>,
    proof_power: Option<u32>,
    status_file: Option<PathBuf>,
}

fn cpu_name() -> String {
    "cpu".into()
}

fn cmd_supervise(config: &Path, max_restarts: u32) -> Result<u8, Failure> {
    use std::process::{Child, Command};
    let text = std::fs::read_to_string(config)
        .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", config.display())))?;
    let specs: Vec<WorkerSpec> =
        serde_json::from_str(&text).map_err(|e| fail(2, format!("{}: {e}", config.display())))?;
    if specs.is_empty() {
        return Err(fail(2, "no workers in the configuration"));
    }
    // Two workers must never share a worktodo file or a device.
    let mut seen = std::collections::HashSet::new();
    for s in &specs {
        let abs = std::path::absolute(&s.worktodo).unwrap_or_else(|_| s.worktodo.clone());
        if !seen.insert(abs) {
            return Err(fail(
                2,
                format!("{} is used by two workers", s.worktodo.display()),
            ));
        }
        if BackendKind::parse(&s.backend).is_none() {
            return Err(fail(2, format!("unknown backend {:?}", s.backend)));
        }
    }
    let devices: Vec<_> = specs
        .iter()
        .filter(|s| s.backend != "cpu" && s.backend != "dwt")
        .map(|s| (s.backend.clone(), s.device.unwrap_or(0)))
        .collect();
    if devices
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != devices.len()
    {
        return Err(fail(2, "two workers use the same device"));
    }
    let exe = std::env::current_exe().map_err(|e| fail(EXIT_STATE, e.to_string()))?;
    let spawn = |s: &WorkerSpec| -> std::io::Result<Child> {
        let mut c = Command::new(&exe);
        c.arg("work")
            .arg("--worktodo")
            .arg(&s.worktodo)
            .arg("--results")
            .arg(&s.results)
            .arg("--checkpoint-dir")
            .arg(&s.checkpoint_dir)
            .arg("--backend")
            .arg(&s.backend);
        if let Some(d) = s.device {
            c.arg("--device").arg(d.to_string());
        }
        if let Some(b) = &s.budget {
            c.arg("--budget").arg(b);
        }
        if let Some(d) = &s.proof_dir {
            c.arg("--proof-dir").arg(d);
        }
        if let Some(p) = s.proof_power {
            c.arg("--proof-power").arg(p.to_string());
        }
        if let Some(f) = &s.status_file {
            c.arg("--status-file").arg(f);
        }
        // Ctrl-C reaches the children through the terminal's process group;
        // each finishes at its next valid boundary.
        c.spawn()
    };
    let mut kids: Vec<(Option<Child>, u32)> = Vec::new();
    for s in &specs {
        let c =
            spawn(s).map_err(|e| fail(EXIT_EXECUTION, format!("cannot start a worker: {e}")))?;
        kids.push((Some(c), 0));
    }
    let mut worst = 0u8;
    loop {
        let mut running = 0;
        for (i, (slot, restarts)) in kids.iter_mut().enumerate() {
            let Some(child) = slot.as_mut() else { continue };
            match child.try_wait() {
                Ok(None) => running += 1,
                Ok(Some(st)) => {
                    let code = st.code().map_or(EXIT_EXECUTION, |c| c as u8);
                    *slot = None;
                    // 0: queue empty; 2/3/4/6: usage, stop or budget, unsupported,
                    // or state a person must look at. Only crashes restart.
                    let crashed = st.code().is_none() || code == EXIT_EXECUTION;
                    if crashed && !STOP.load(Ordering::SeqCst) && *restarts < max_restarts {
                        *restarts += 1;
                        eprintln!(
                            "worker {i} ({}) ended with {st}; restart {} of {max_restarts}",
                            specs[i].worktodo.display(),
                            restarts
                        );
                        std::thread::sleep(Duration::from_secs(5 * u64::from(*restarts)));
                        match spawn(&specs[i]) {
                            Ok(c) => {
                                *slot = Some(c);
                                running += 1;
                            }
                            Err(e) => eprintln!("worker {i} cannot restart: {e}"),
                        }
                    } else {
                        eprintln!(
                            "worker {i} ({}) ended with {st}",
                            specs[i].worktodo.display()
                        );
                        worst = worst.max(code);
                    }
                }
                Err(e) => {
                    eprintln!("worker {i}: {e}");
                    *slot = None;
                    worst = worst.max(EXIT_EXECUTION);
                }
            }
        }
        if running == 0 {
            return Ok(worst);
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// The state of one queue as text: runnable and refused lines,
/// checkpoints, result counts, and worker status files.
fn status_text(
    worktodo: &Path,
    results: &Path,
    ckpt_dir: Option<&Path>,
    status: &[PathBuf],
) -> String {
    use mersenne_forge::{factor, worktodo as wt};
    use std::fmt::Write as _;
    let mut out = String::new();
    let text = std::fs::read_to_string(worktodo).unwrap_or_default();
    let (mut runnable, mut refused) = (Vec::new(), 0usize);
    for l in text.lines() {
        match (factor::parse_factor_line(l), wt::parse_line(l)) {
            (Ok(Some(f)), _) => runnable.push(f.line),
            (Ok(None), Ok(Some(a))) => runnable.push(a.line),
            (Ok(None), Ok(None)) => {}
            _ => refused += 1,
        }
    }
    let _ = writeln!(
        out,
        "{}: {} runnable, {} left for other programs",
        worktodo.display(),
        runnable.len(),
        refused
    );
    for l in runnable.iter().take(5) {
        let _ = writeln!(out, "  {l}");
    }
    let lines = std::fs::read_to_string(results).unwrap_or_default();
    let mut by_status = std::collections::BTreeMap::<String, usize>::new();
    for v in lines
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
    {
        let key = format!(
            "{} {}",
            v["worktype"].as_str().unwrap_or("?"),
            v["status"].as_str().unwrap_or("?")
        );
        *by_status.entry(key).or_default() += 1;
    }
    let _ = writeln!(out, "{}: {by_status:?}", results.display());
    if let Some(d) = ckpt_dir {
        for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
            let path = e.path();
            if path.extension().is_some_and(|x| x == "ckpt") {
                let _ = match Checkpoint::load_file(&path) {
                    Ok(ck) => {
                        let done = (ck.meta.k as f64 / f64::from(ck.meta.exponent.max(1))).min(1.0);
                        let filled = (done * 30.0).round() as usize;
                        writeln!(
                            out,
                            "  {}: {} k = {} of {}\n    [{}{}] {:.1}%, {:.0} s",
                            path.display(),
                            ck.meta.protocol,
                            ck.meta.k,
                            ck.meta.exponent,
                            "#".repeat(filled),
                            ".".repeat(30 - filled),
                            100.0 * done,
                            ck.meta.elapsed_secs
                        )
                    }
                    Err(e) => writeln!(out, "  {}: {e}", path.display()),
                };
            }
        }
    }
    for s in status {
        let _ = match std::fs::read_to_string(s) {
            Ok(t) => match serde_json::from_str::<serde_json::Value>(&t) {
                Ok(v) => writeln!(
                    out,
                    "{}: {} on {} {} at {}: {}{}",
                    s.display(),
                    v["state"].as_str().unwrap_or("?"),
                    v["backend"].as_str().unwrap_or("?"),
                    v["device"],
                    v["updated"].as_str().unwrap_or("?"),
                    v["current"].as_str().unwrap_or(""),
                    v["exit_code"]
                        .as_i64()
                        .map(|c| format!(" (exit {c})"))
                        .unwrap_or_default()
                ),
                Err(_) => writeln!(out, "{}:\n{t}", s.display()),
            },
            Err(e) => writeln!(out, "{}: {e}", s.display()),
        };
    }
    out
}

/// Serve `text()` over HTTP on a loopback address until Ctrl-C:
/// `/<token>/` is an HTML page that reloads every 5 s, `/<token>/status.txt`
/// the plain text, where the token is random per run and printed at start
/// (other local users and web pages cannot guess the URL). A request whose
/// Host header is not this address or localhost is refused, which stops DNS
/// rebinding. Any other path is 404; only GET and HEAD are answered.
/// Requests are handled one at a time. ponytail: a single threaded loop with no keep-alive; a
/// slow client delays the next one by at most the 5 s read timeout.
fn serve_status(addr: std::net::SocketAddr, text: impl Fn() -> String) -> Result<u8, Failure> {
    use std::io::{Read, Write};
    // The page shows local paths and queue contents: never on a public
    // interface.
    if !addr.ip().is_loopback() {
        return Err(fail(
            2,
            format!("--serve {addr}: only loopback addresses are allowed"),
        ));
    }
    let listener = std::net::TcpListener::bind(addr)
        .map_err(|e| fail(EXIT_STATE, format!("--serve {addr}: {e}")))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| fail(EXIT_STATE, e.to_string()))?;
    let addr = listener.local_addr().unwrap_or(addr);
    let token = {
        use std::hash::{BuildHasher, Hasher};
        let mut t = String::new();
        for i in 0..2u64 {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(i);
            h.write_u128(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos()),
            );
            h.write_u32(std::process::id());
            t.push_str(&format!("{:016x}", h.finish()));
        }
        t
    };
    let hosts = [addr.to_string(), format!("localhost:{}", addr.port())];
    let (page_path, text_path) = (format!("/{token}/"), format!("/{token}/status.txt"));
    eprintln!("serving status at http://{addr}{page_path} (Ctrl-C to stop)");
    while !STOP.load(Ordering::SeqCst) {
        let mut stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            Err(e) => {
                eprintln!("--serve: {e}");
                continue;
            }
        };
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buf = [0u8; 4096];
        let mut req = Vec::new();
        while !req.windows(4).any(|w| w == b"\r\n\r\n") && req.len() < buf.len() {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => req.extend_from_slice(&buf[..n]),
            }
        }
        let first = String::from_utf8_lossy(&req);
        let mut words = first.lines().next().unwrap_or("").split(' ');
        let (method, path) = (words.next().unwrap_or(""), words.next().unwrap_or(""));
        let host_ok = first.lines().skip(1).any(|l| {
            l.split_once(':').is_some_and(|(k, v)| {
                k.trim().eq_ignore_ascii_case("host") && hosts.iter().any(|h| h == v.trim())
            })
        });
        let (code, kind, body) = match (method, path) {
            _ if !host_ok => (
                "421 Misdirected Request",
                "text/plain; charset=utf-8",
                "unexpected Host\n".into(),
            ),
            ("GET" | "HEAD", p) if p == page_path => {
                ("200 OK", "text/html; charset=utf-8", status_page(&text()))
            }
            ("GET" | "HEAD", p) if p == text_path => {
                ("200 OK", "text/plain; charset=utf-8", text())
            }
            ("GET" | "HEAD", _) => (
                "404 Not Found",
                "text/plain; charset=utf-8",
                "not found\n".into(),
            ),
            _ => (
                "405 Method Not Allowed",
                "text/plain; charset=utf-8",
                "only GET and HEAD\n".into(),
            ),
        };
        let head = format!(
            "HTTP/1.1 {code}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(head.as_bytes());
        if method != "HEAD" {
            let _ = stream.write_all(body.as_bytes());
        }
    }
    Ok(0)
}

/// The status text as a small self refreshing HTML page, every character
/// that HTML treats specially escaped.
fn status_page(text: &str) -> String {
    let mut esc = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => esc.push_str("&amp;"),
            '<' => esc.push_str("&lt;"),
            '>' => esc.push_str("&gt;"),
            '"' => esc.push_str("&quot;"),
            _ => esc.push(c),
        }
    }
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <meta http-equiv=\"refresh\" content=\"5\"><title>mersenne-forge status</title>\
         <style>body{{font:14px/1.4 ui-monospace,monospace;margin:16px;background:#fff;color:#111}}\
         @media (prefers-color-scheme:dark){{body{{background:#111;color:#eee}}}}\
         pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head>\
         <body><h1>mersenne-forge status</h1><pre>{esc}</pre></body></html>\n"
    )
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
        "build": mersenne_forge::build_identity(),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "backends": backends,
        "exponent": exponent,
        "requirements": {
            "memory": "per backend with --exponent: resident registers (PRP 4, LL 2, proof generation power + 2) plus fixed scratch",
            "proof_disk_bytes": "2^power * ((p - 1) / 8 + 1) for the proof residues, plus the proof file",
            "checkpoint_bytes": "up to 3 * 4 * ceil(p / 32) plus metadata per file; rotation keeps a previous file",
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
        println!("build: {}", mersenne_forge::build_identity());
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
    // Paths are written into JSON reports, which hold only UTF-8; refuse
    // other arguments up front instead of failing halfway through a run.
    let json = std::env::args_os().any(|a| a == "--json");
    if let Some(bad) = std::env::args_os().find(|a| a.to_str().is_none()) {
        emit_failure(
            &fail(
                2,
                format!("arguments must be valid UTF-8: {}", bad.to_string_lossy()),
            ),
            json,
        );
        return ExitCode::from(2);
    }
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
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
        Cmd::CheckFactor { json, .. }
        | Cmd::Tf { json, .. }
        | Cmd::Pm1 { json, .. }
        | Cmd::Ecm { json, .. }
        | Cmd::Pp1 { json, .. }
        | Cmd::Selftest { json, .. } => *json,
        Cmd::Bench { .. }
        | Cmd::BenchRender { .. }
        | Cmd::BenchAnalyze { .. }
        | Cmd::Work { .. }
        | Cmd::Supervise { .. }
        | Cmd::Tune { .. }
        | Cmd::Status { .. }
        | Cmd::Power { .. }
        | Cmd::CheckpointExport { .. }
        | Cmd::Primenet { .. }
        | Cmd::PrimenetV5 { .. } => false,
    };
    let _ = ctrlc::set_handler(|| {
        if STOP.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        eprintln!(
            "stop requested; finishing at the next valid boundary (press Ctrl-C again to abort)"
        );
    });
    // A closure so that `?` in an arm ends the command, not `main`.
    let r = (move || -> Result<u8, Failure> {
        match cli.cmd {
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
                backend,
                proof_dir,
                json,
            } => cmd_resume(
                &checkpoint,
                budget,
                device,
                backend.map(BackendArg::kind),
                proof_dir,
                json,
            ),
            Cmd::Tf {
                exponent,
                from,
                to,
                gpu,
                device,
                json,
            } => {
                let stop = || STOP.load(Ordering::SeqCst);
                let o = if gpu {
                    tf_on_gpu(exponent, from, to, device, &stop)?
                } else {
                    mersenne_forge::factor::trial_factor(
                        exponent,
                        from,
                        to,
                        None,
                        &stop,
                        &mut |_| {},
                    )
                    .map_err(|e| fail(2, e))?
                };
                if json {
                    println!("{}", serde_json::to_string(&o).expect("serializable"));
                } else {
                    println!(
                        "M{exponent} TF {from}-{to}: {} factors {:?}",
                        if o.complete { "complete," } else { "stopped," },
                        o.factors
                    );
                }
                Ok(if o.complete { 0 } else { EXIT_INCOMPLETE })
            }
            Cmd::Pm1 {
                exponent,
                b1,
                b2,
                backend,
                device,
                json,
            } => {
                let mut be = open(backend.kind(), exponent, device)?;
                let stop = || STOP.load(Ordering::SeqCst);
                let o = mersenne_forge::factor::pm1(be.as_mut(), b1, b2, &stop)
                    .map_err(|e| fail(EXIT_EXECUTION, e))?;
                if json {
                    println!("{}", serde_json::to_string(&o).expect("serializable"));
                } else {
                    for n in &o.notes {
                        println!("note: {n}");
                    }
                    println!("M{exponent} P-1 B1={b1} B2={b2}: factor {:?}", o.factor);
                }
                Ok(if o.complete { 0 } else { EXIT_INCOMPLETE })
            }
            Cmd::Ecm {
                exponent,
                b1,
                b2,
                curves,
                sigma,
                json,
            } => {
                let stop = || STOP.load(Ordering::SeqCst);
                let o = mersenne_forge::factor::ecm(exponent, b1, b2, curves, sigma, &stop)
                    .map_err(|e| fail(2, e))?;
                if json {
                    println!("{}", serde_json::to_string(&o).expect("serializable"));
                } else {
                    for n in &o.notes {
                        println!("note: {n}");
                    }
                    println!(
                        "M{exponent} ECM B1={b1}: {} curves, factor {:?} (sigma {:?})",
                        o.curves, o.factor, o.found_sigma
                    );
                }
                Ok(0)
            }
            Cmd::Pp1 {
                exponent,
                b1,
                b2,
                run,
                json,
            } => {
                let stop = || STOP.load(Ordering::SeqCst);
                let o = mersenne_forge::factor::pp1(exponent, b1, b2, run, &stop)
                    .map_err(|e| fail(2, e))?;
                if json {
                    println!("{}", serde_json::to_string(&o).expect("serializable"));
                } else {
                    for n in &o.notes {
                        println!("note: {n}");
                    }
                    println!(
                        "M{exponent} P+1 B1={b1} B2={} start {}/{}: factor {:?}",
                        o.b2, o.start.0, o.start.1, o.factor
                    );
                }
                Ok(0)
            }
            Cmd::Selftest {
                backend,
                device,
                full,
                json,
            } => {
                let kind = backend.kind();
                let mut open_one =
                    |p: u32| select::open(kind, p, device).map_err(|e| e.to_string());
                let stop = || STOP.load(Ordering::SeqCst);
                let cases = mersenne_forge::selftest::run(&mut open_one, full, &stop);
                let failed = cases.iter().filter(|c| c.ok == Some(false)).count();
                if json {
                    println!("{}", serde_json::to_string(&cases).expect("serializable"));
                } else {
                    for c in &cases {
                        let mark = match c.ok {
                            Some(true) => "pass",
                            Some(false) => "FAIL",
                            None => "skip",
                        };
                        println!(
                            "{mark} {}{}",
                            c.name,
                            if c.detail.is_empty() {
                                String::new()
                            } else {
                                format!(": {}", c.detail)
                            }
                        );
                    }
                    println!("{} cases, {failed} failed", cases.len());
                }
                Ok(if failed == 0 { 0 } else { EXIT_EXECUTION })
            }
            Cmd::CheckpointExport { input, output } => {
                let ck = Checkpoint::load_file(&input)
                    .and_then(Checkpoint::portable)
                    .map_err(|e| fail(EXIT_STATE, e.to_string()))?;
                if output.exists() {
                    return Err(fail(EXIT_STATE, format!("{} exists", output.display())));
                }
                ck.save(&output)
                    .map_err(|e| fail(EXIT_STATE, e.to_string()))?;
                println!(
                    "M{} {} at k = {} written to {}",
                    ck.meta.exponent,
                    ck.meta.protocol,
                    ck.meta.k,
                    output.display()
                );
                Ok(0)
            }
            Cmd::Verify {
                proof,
                backend,
                exponent,
                device,
                budget,
                json,
            } => cmd_verify(&proof, backend, exponent, device, budget, json),
            Cmd::Bench { plan, output } => cmd_bench(&plan, &output),
            Cmd::Work {
                worktodo,
                results,
                checkpoint_dir,
                backend,
                device,
                budget,
                proof_dir,
                proof_power,
                status_file,
            } => {
                let w = WorkArgs {
                    worktodo,
                    results,
                    checkpoint_dir,
                    kind: backend.kind(),
                    device,
                    budget,
                    proof_dir,
                    proof_power,
                    status_file,
                };
                let r = cmd_work(&w);
                let code = match &r {
                    Ok(c) => *c,
                    Err(f) => f.code,
                };
                w.status("exited", None, json!({"exit_code": code}));
                r
            }
            Cmd::Supervise {
                config,
                max_restarts,
            } => cmd_supervise(&config, max_restarts),
            Cmd::Tune {
                exponent,
                backend,
                device,
                iters,
            } => cmd_tune(exponent, backend.kind(), device, iters),
            Cmd::Status {
                worktodo,
                results,
                checkpoint_dir,
                status_file,
                watch,
                serve,
            } => {
                if let Some(addr) = serve {
                    return serve_status(addr, || {
                        status_text(&worktodo, &results, checkpoint_dir.as_deref(), &status_file)
                    });
                }
                // --watch: a full screen view on the terminal's alternate screen,
                // redrawn every N seconds until Ctrl-C, then the screen is
                // restored. Without it, one plain report.
                let tty = watch.is_some() && std::io::IsTerminal::is_terminal(&std::io::stdout());
                if tty {
                    print!("\x1b[?1049h\x1b[?25l");
                }
                loop {
                    if tty {
                        print!("\x1b[H\x1b[2J");
                        println!(
                            "mersenne-forge status, every {} s (Ctrl-C to quit)\n",
                            watch.unwrap_or(1).max(1)
                        );
                    }
                    print!(
                        "{}",
                        status_text(&worktodo, &results, checkpoint_dir.as_deref(), &status_file)
                    );
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                    match watch {
                        Some(n) if !STOP.load(Ordering::SeqCst) => {
                            let until = std::time::Instant::now() + Duration::from_secs(n.max(1));
                            while std::time::Instant::now() < until && !STOP.load(Ordering::SeqCst)
                            {
                                std::thread::sleep(Duration::from_millis(100));
                            }
                        }
                        _ => break,
                    }
                    if STOP.load(Ordering::SeqCst) {
                        break;
                    }
                }
                if tty {
                    print!("\x1b[?25h\x1b[?1049l");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
                Ok(0)
            }
            Cmd::PrimenetV5 { state_dir, action } => cmd_primenet_v5(&state_dir, action),
            Cmd::Primenet {
                user,
                password_file,
                worktodo,
                results,
                proof_dir,
                worktype,
                ahead,
                state_dir,
            } => cmd_primenet(
                &user,
                &password_file,
                &worktodo,
                &results,
                proof_dir,
                worktype.as_deref(),
                ahead,
                &state_dir,
            ),
            Cmd::Power { source, device } => {
                mersenne_forge::evidence::sample_power(&source, device)
                    .map(|w| {
                        println!("{w:.3} W");
                        0
                    })
                    .map_err(|e| fail(EXIT_UNSUPPORTED, e))
            }
            Cmd::CheckFactor {
                exponent,
                factor,
                json,
            } => mersenne_forge::worktodo::check_factor(exponent, &factor)
                .map_err(|e| fail(2, e))
                .map(|c| {
                    if json {
                        println!("{}", serde_json::to_string(&c).expect("json"));
                    } else {
                        println!(
                            "{} {} M{}{}",
                            c.factor,
                            if c.divides {
                                "divides"
                            } else {
                                "does not divide"
                            },
                            c.exponent,
                            if c.divides && !c.proper {
                                " but is not a proper factor (it is 1 or M_p itself)"
                            } else {
                                ""
                            }
                        );
                    }
                    if c.accepted() {
                        0
                    } else {
                        EXIT_REJECTED
                    }
                }),
            Cmd::BenchRender { input, output } => bench::render(&input, &output)
                .map(|out| {
                    println!(
                        "{} observations written to {}",
                        out.observations.len(),
                        output.display()
                    );
                    0
                })
                .map_err(|e| fail(EXIT_STATE, e)),
            Cmd::BenchAnalyze {
                input,
                baseline,
                candidate,
                metric,
                seed,
            } => {
                let text = std::fs::read_to_string(&input)
                    .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", input.display())));
                text.and_then(|t| {
                    serde_json::from_str::<bench::BenchOutput>(&t)
                        .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", input.display())))
                })
                .and_then(|out| {
                    mersenne_forge::evidence::analyze(&out, baseline, candidate, &metric, seed)
                        .map_err(|e| fail(2, e))
                })
                .map(|v| {
                    println!("{}", serde_json::to_string_pretty(&v).expect("json"));
                    0
                })
            }
        }
    })();
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
