// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The `mersenne-forge` command line.
//!
//! Exit codes: 0 success (completed run, accepted proof, doctor, bench);
//! 2 usage error; 3 incomplete (budget or stop request); 4 unsupported
//! (backend not built, exponent or option out of range); 5 execution failure
//! (repeated check failures, backend error); 6 checkpoint, proof residue, or
//! file error; 7 proof rejected; 1 any other error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;

use mersenne_forge::bench;
use mersenne_forge::checkpoint::Checkpoint;
use mersenne_forge::ll::{self, JacobiCadence, LlConfig};
use mersenne_forge::proof::{self, Proof, ProofError};
use mersenne_forge::prp::{self, PrpConfig};
use mersenne_forge::run::{Control, ProcessState, ProtocolError, RunReport};
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
    /// Report backends, build identity, and supported ranges.
    Doctor {
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
    (any && total > 0.0 && total.is_finite()).then(|| Duration::from_secs_f64(total))
}

fn deadline(budget: &Option<String>, kind: BackendKind) -> Result<Option<Instant>, Failure> {
    match budget {
        None if kind.is_device() => Err(fail(
            EXIT_UNSUPPORTED,
            format!("the {} backend requires a bounded --budget", kind.name()),
        )),
        None => Ok(None),
        Some(b) => {
            let d = parse_duration(b).ok_or_else(|| fail(2, format!("invalid --budget {b:?}")))?;
            Ok(Some(Instant::now() + d))
        }
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
                None => println!("no checkpoint directory was given; progress is lost"),
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
    for n in &e.notes {
        println!("note: {n}");
    }
}

fn exit_for(report: &RunReport) -> u8 {
    match report.process {
        ProcessState::Completed => 0,
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
                checkpoint: checkpoint_dir.map(|d| d.join(format!("M{p}-prp.ckpt"))),
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
                checkpoint: checkpoint_dir.map(|d| d.join(format!("M{p}-ll.ckpt"))),
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

fn cmd_resume(
    path: &Path,
    budget: Option<String>,
    device: Option<u32>,
    as_json: bool,
) -> Result<u8, Failure> {
    let (ck, used) = Checkpoint::load_latest(path).map_err(|e| fail(EXIT_STATE, e.to_string()))?;
    let m = &ck.meta;
    let kind = BackendKind::parse(&m.backend.kind).ok_or_else(|| {
        fail(
            EXIT_UNSUPPORTED,
            format!("unknown backend {:?} in checkpoint", m.backend.kind),
        )
    })?;
    let deadline = deadline(&budget, kind)?;
    let p = m.exponent;
    let mut be = open(kind, p, device)?;
    let mut ctl = Control {
        stop: Some(&STOP),
        deadline,
        ..Control::default()
    };
    let requested =
        json!({ "checkpoint": path, "loaded_from": used, "budget": budget, "device": device });
    let bad = |e: serde_json::Error| fail(EXIT_STATE, format!("checkpoint configuration: {e}"));
    let report = if m.protocol == prp::PROTOCOL {
        let mut cfg: PrpConfig = serde_json::from_value(m.config.clone()).map_err(bad)?;
        cfg.checkpoint = Some(path.to_path_buf());
        prp::run_prp(&mut be, p, &cfg, m.requested.clone(), Some(&ck), &mut ctl)
    } else if m.protocol == ll::PROTOCOL {
        let mut cfg: LlConfig = serde_json::from_value(m.config.clone()).map_err(bad)?;
        cfg.checkpoint = Some(path.to_path_buf());
        ll::run_ll(&mut be, p, &cfg, m.requested.clone(), Some(&ck), &mut ctl)
    } else {
        return Err(fail(
            EXIT_UNSUPPORTED,
            format!("unknown protocol {:?}", m.protocol),
        ));
    }
    .map_err(protocol_failure)?;
    let out = json!({ "command": "resume", "requested": requested, "resumed_at": m.k, "effective": m.config, "report": report });
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
                    "squarings": v.squarings, "gerbicz_passed": v.gerbicz_passed, "gerbicz_failed": v.gerbicz_failed,
                    "secs": start.elapsed().as_secs_f64(),
                    "message": format!("M{} {claim}: the proof verifies the PRP computation, not deterministic primality", proof.p),
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

fn cmd_bench(plan: &Path, output: &Path) -> Result<u8, Failure> {
    let text = std::fs::read_to_string(plan)
        .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", plan.display())))?;
    let plan: bench::Plan =
        serde_json::from_str(&text).map_err(|e| fail(2, format!("invalid plan: {e}")))?;
    let work = output.with_extension("work");
    let out = bench::run_plan(&plan, &work, |o| {
        eprintln!(
            "{:>4} {:<24} repeat {:<3}{} {:<10} {:.3}s",
            o.sequence,
            o.entry,
            o.repeat,
            if o.warmup { " (warmup)" } else { "" },
            o.status,
            o.wall_secs
        )
    });
    if let Some(dir) = output.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| fail(EXIT_STATE, e.to_string()))?;
    }
    let body = if output.extension().is_some_and(|e| e == "csv") {
        bench::to_csv(&out)
    } else {
        serde_json::to_string_pretty(&out).expect("serializes")
    };
    std::fs::write(output, body)
        .map_err(|e| fail(EXIT_STATE, format!("{}: {e}", output.display())))?;
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
    Ok(0)
}

fn cmd_doctor(as_json: bool) -> Result<u8, Failure> {
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
        "requirements": {
            "residue_bytes": "about p / 8 per resident residue; a run keeps 4 to 6",
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let _ = ctrlc::set_handler(|| {
        if STOP.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        eprintln!(
            "stop requested; finishing at the next valid boundary (press Ctrl-C again to abort)"
        );
    });
    let r = match cli.cmd {
        Cmd::Doctor { json } => cmd_doctor(json),
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
            eprintln!("error: {}", f.message);
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
