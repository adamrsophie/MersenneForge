// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Benchmark harness. A plan lists engine workloads (run in process on a
//! named backend) and external commands (timed as subprocesses, with argv
//! given verbatim so no flag of another program is invented here). Every run
//! is recorded as a raw observation, failures included, and the output file
//! is rewritten after every observation. No energy value is produced unless
//! measured; this harness has no power telemetry, so energy is always
//! recorded as unavailable.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::backend::BackendIdentity;
use crate::ll::{run_ll, JacobiCadence, LlConfig};
use crate::prp::{run_prp, PrpConfig};
use crate::run::{Control, ProcessState, ProofStage, Timings};
use crate::select::{self, BackendKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    /// All repeats of an entry, then the next entry.
    Sequential,
    /// Each repeat runs every entry in declared order (ABAB).
    Paired,
    /// Each repeat runs every entry in an order shuffled from `seed`.
    Randomized,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineOptions {
    pub block_size: Option<u32>,
    pub check_step: Option<u64>,
    pub proof_power: Option<u32>,
    pub verify_proof: Option<bool>,
    pub log_step: Option<u64>,
    pub jacobi_every_log: Option<bool>,
    pub device: Option<u32>,
    pub budget_secs: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Entry {
    Engine {
        name: String,
        test: String,
        exponent: u32,
        backend: String,
        #[serde(default)]
        options: EngineOptions,
    },
    External {
        name: String,
        argv: Vec<String>,
        cwd: Option<PathBuf>,
        timeout_secs: Option<f64>,
    },
}

impl Entry {
    pub fn name(&self) -> &str {
        match self {
            Entry::Engine { name, .. } | Entry::External { name, .. } => name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub name: String,
    pub order: Order,
    pub seed: Option<u64>,
    pub repeats: u32,
    #[serde(default)]
    pub warmups: u32,
    pub entries: Vec<Entry>,
}

/// A number of seconds as a `Duration`: finite, positive, representable.
fn seconds(what: &str, s: f64) -> Result<Duration, String> {
    if !(s.is_finite() && s > 0.0) {
        return Err(format!(
            "{what} must be a finite positive number of seconds, got {s}"
        ));
    }
    Duration::try_from_secs_f64(s).map_err(|e| format!("{what} {s}: {e}"))
}

/// `now + secs`, or an error when the instant is not representable.
fn deadline_after(start: Instant, what: &str, s: f64) -> Result<Instant, String> {
    start
        .checked_add(seconds(what, s)?)
        .ok_or_else(|| format!("{what} {s} is too far in the future"))
}

impl Plan {
    /// Check every value that would otherwise fail or panic during the run.
    /// A plan with no entries or no measured repeats is refused: it would
    /// produce no observation, and an old output file must not pass for its
    /// result.
    pub fn validate(&self) -> Result<(), String> {
        if self.entries.is_empty() {
            return Err("the plan has no entries".into());
        }
        if self.repeats == 0 {
            return Err("repeats must be at least 1".into());
        }
        let now = Instant::now();
        for e in &self.entries {
            match e {
                Entry::Engine { name, options, .. } => {
                    if let Some(b) = options.budget_secs {
                        deadline_after(now, &format!("{name}: budget_secs"), b)?;
                    }
                }
                Entry::External {
                    name, timeout_secs, ..
                } => {
                    if let Some(t) = timeout_secs {
                        deadline_after(now, &format!("{name}: timeout_secs"), *t)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Check and fault counts of an engine run, from its evidence.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Reliability {
    pub gerbicz_passed: u64,
    pub gerbicz_failed: u64,
    pub jacobi_passed: u64,
    pub jacobi_failed: u64,
    pub jacobi_inconclusive: u64,
    pub backend_faults: u64,
    pub rollbacks: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Display name of the entry (not unique: see `entry_index`).
    pub entry: String,
    /// Position of the entry in the plan's `entries`: the stable identifier.
    pub entry_index: usize,
    pub kind: String,
    /// Engine runs: the workload.
    pub test: Option<String>,
    pub exponent: Option<u32>,
    /// External runs: the exact argv that ran.
    pub argv: Option<Vec<String>>,
    /// What the result rests on: "engine checked run" for engine entries;
    /// "external process timing; result not verified" for external ones,
    /// whose exit status says nothing about the correctness of their output.
    pub result_source: String,
    /// Engine runs start from iteration 0 with checkpointing disabled.
    pub start: String,
    /// Position in the executed sequence.
    pub sequence: usize,
    pub repeat: u32,
    pub warmup: bool,
    /// "completed", "incomplete", or "failed". An engine run is completed
    /// only when its requested proof stage also succeeded.
    pub status: String,
    pub error: Option<String>,
    pub wall_secs: f64,
    /// Engine runs: wall time split by activity.
    pub timings: Option<Timings>,
    pub verdict: Option<String>,
    pub res64: Option<String>,
    /// Engine runs: the backend that did the arithmetic.
    pub backend: Option<BackendIdentity>,
    /// Engine PRP runs: what happened to the proof.
    pub proof_stage: Option<ProofStage>,
    /// Engine PRP runs: the Gerbicz check step derived at the start.
    pub check_step: Option<u64>,
    /// Engine runs: check outcomes, faults, and rollbacks.
    pub reliability: Option<Reliability>,
    /// Engine runs: backend diagnostics read once after the run (for example
    /// the DWT maximum roundoff).
    pub diagnostics: Vec<(String, String)>,
    pub exit_code: Option<i32>,
    /// External runs: files holding the captured standard output and error.
    pub stdout: Option<PathBuf>,
    pub stderr: Option<PathBuf>,
    pub energy_joules: Option<f64>,
    pub telemetry: String,
}

impl Observation {
    fn failed(kind: &str, error: String, wall_secs: f64) -> Observation {
        Observation {
            entry: String::new(),
            entry_index: 0,
            kind: kind.into(),
            test: None,
            exponent: None,
            argv: None,
            result_source: String::new(),
            start: String::new(),
            sequence: 0,
            repeat: 0,
            warmup: false,
            status: "failed".into(),
            error: Some(error),
            wall_secs,
            timings: None,
            verdict: None,
            res64: None,
            backend: None,
            proof_stage: None,
            check_step: None,
            reliability: None,
            diagnostics: Vec::new(),
            exit_code: None,
            stdout: None,
            stderr: None,
            energy_joules: None,
            telemetry: "unavailable".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchOutput {
    pub plan: Plan,
    /// SHA3-256 of the plan as recorded in `plan` (compact JSON).
    pub plan_sha3: String,
    pub seed_used: Option<u64>,
    /// This invocation's directory: scratch space and external captures.
    pub execution_dir: PathBuf,
    /// Source build identity of the harness (see `build_identity`).
    pub build: String,
    pub crate_version: String,
    pub os: String,
    pub arch: String,
    pub observations: Vec<Observation>,
}

/// SplitMix64: a small, fully specified generator, so a recorded seed
/// reproduces the order on any platform.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The executed sequence: (entry index, repeat index, warmup).
pub fn schedule(plan: &Plan) -> (Vec<(usize, u32, bool)>, Option<u64>) {
    let n = plan.entries.len();
    let mut seq: Vec<(usize, u32, bool)> = Vec::new();
    for w in 0..plan.warmups {
        for i in 0..n {
            seq.push((i, w, true));
        }
    }
    let mut seed_used = None;
    match plan.order {
        Order::Sequential => {
            for i in 0..n {
                for r in 0..plan.repeats {
                    seq.push((i, r, false));
                }
            }
        }
        Order::Paired => {
            for r in 0..plan.repeats {
                for i in 0..n {
                    seq.push((i, r, false));
                }
            }
        }
        Order::Randomized => {
            let seed = plan.seed.unwrap_or(0x4d46_2d62_656e_6368);
            seed_used = Some(seed);
            let mut st = seed;
            for r in 0..plan.repeats {
                let mut idx: Vec<usize> = (0..n).collect();
                for j in (1..n).rev() {
                    let k = (splitmix(&mut st) % (j as u64 + 1)) as usize;
                    idx.swap(j, k);
                }
                seq.extend(idx.into_iter().map(|i| (i, r, false)));
            }
        }
    }
    (seq, seed_used)
}

fn engine_run(
    test: &str,
    p: u32,
    backend: &str,
    o: &EngineOptions,
    work: &Path,
    stop: Option<&AtomicBool>,
) -> Result<Observation, String> {
    let kind = BackendKind::parse(backend).ok_or_else(|| format!("unknown backend {backend}"))?;
    if kind.is_device() && o.budget_secs.is_none() {
        return Err(format!("{backend} entries need budget_secs"));
    }
    let start = Instant::now();
    let deadline = o
        .budget_secs
        .map(|s| deadline_after(start, "budget_secs", s))
        .transpose()?;
    let mut be = select::open(kind, p, o.device).map_err(|e| e.to_string())?;
    let init = start.elapsed().as_secs_f64();
    let mut ctl = Control {
        stop,
        deadline,
        ..Control::default()
    };
    let report = match test {
        "prp" => {
            let cfg = PrpConfig {
                block_size: o.block_size.unwrap_or(1000),
                check_step: o.check_step,
                proof_power: o.proof_power,
                proof_dir: o.proof_power.map(|_| work.join("proof")),
                verify_proof: o.verify_proof.unwrap_or(true),
                checkpoint: None,
                max_batch: 1000,
            };
            run_prp(&mut be, p, &cfg, serde_json::json!({}), None, &mut ctl)
        }
        "ll" => {
            let cfg = LlConfig {
                log_step: o.log_step.unwrap_or(20_000),
                jacobi: if o.jacobi_every_log == Some(true) {
                    JacobiCadence::EveryLog
                } else {
                    JacobiCadence::Auto
                },
                checkpoint: None,
                max_batch: 1000,
            };
            run_ll(&mut be, p, &cfg, serde_json::json!({}), None, &mut ctl)
        }
        other => return Err(format!("unknown test {other}")),
    }
    .map_err(|e| e.to_string())?;
    // Once, after the run: outside the arithmetic loop.
    let diagnostics = be.diagnostics().unwrap_or_default();
    let ev = &report.evidence;
    let reliability = Reliability {
        gerbicz_passed: ev.gerbicz_passed,
        gerbicz_failed: ev.gerbicz_failed,
        jacobi_passed: ev.jacobi_passed,
        jacobi_failed: ev.jacobi_failed,
        jacobi_inconclusive: ev.jacobi_inconclusive,
        backend_faults: ev.backend_faults,
        rollbacks: ev.rollbacks,
    };
    let mut timings = report.timings.clone();
    timings.init += init;
    let stage = report.evidence.proof_stage;
    let status = match (&report.process, stage) {
        (ProcessState::Completed, ProofStage::Failed | ProofStage::Rejected) => "failed",
        (ProcessState::Completed, ProofStage::Interrupted) => "incomplete",
        (ProcessState::Completed, _) => "completed",
        _ => "incomplete",
    };
    Ok(Observation {
        status: status.into(),
        error: (status == "failed").then(|| format!("proof stage {}", stage.name())),
        wall_secs: start.elapsed().as_secs_f64(),
        timings: Some(timings),
        verdict: report.result.as_ref().map(|r| format!("{:?}", r.verdict)),
        res64: report.result.map(|r| r.res64),
        backend: report.evidence.backend,
        proof_stage: (test == "prp").then_some(stage),
        check_step: report.evidence.check_step,
        reliability: Some(reliability),
        diagnostics,
        ..Observation::failed("engine", String::new(), 0.0)
    })
}

fn external_run(
    argv: &[String],
    cwd: Option<&Path>,
    timeout: Option<f64>,
    stop: Option<&AtomicBool>,
    capture: (&Path, &Path),
) -> Result<Observation, String> {
    let (prog, args) = argv
        .split_first()
        .ok_or("external entry needs a nonempty argv")?;
    let file = |p: &Path| fs::File::create(p).map_err(|e| format!("{}: {e}", p.display()));
    let mut cmd = std::process::Command::new(prog);
    cmd.args(args)
        .stdout(file(capture.0)?)
        .stderr(file(capture.1)?);
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let start = Instant::now();
    let deadline = timeout
        .map(|t| deadline_after(start, "timeout_secs", t))
        .transpose()?;
    let mut child = cmd.spawn().map_err(|e| format!("{prog}: {e}"))?;
    // Only the direct child is killed on a timeout or stop request; its own
    // descendants are not tracked.
    let mut stopped = false;
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(s);
        }
        stopped = stop.is_some_and(|s| s.load(Ordering::SeqCst));
        if stopped || deadline.is_some_and(|d| Instant::now() >= d) {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let wall = start.elapsed().as_secs_f64();
    let (st, code, error) = match status {
        Some(s) if s.success() => ("completed", s.code(), None),
        Some(s) => ("failed", s.code(), Some("nonzero exit".to_string())),
        None if stopped => ("incomplete", None, Some("stop requested".to_string())),
        None => ("incomplete", None, Some("timeout".to_string())),
    };
    Ok(Observation {
        status: st.into(),
        error,
        wall_secs: wall,
        exit_code: code,
        stdout: Some(capture.0.to_path_buf()),
        stderr: Some(capture.1.to_path_buf()),
        ..Observation::failed("external", String::new(), 0.0)
    })
}

/// Write the output as JSON, or CSV when the extension is `.csv`.
pub fn write_output(output: &Path, out: &BenchOutput) -> Result<(), String> {
    let body = if output.extension().is_some_and(|e| e == "csv") {
        to_csv(out)
    } else {
        serde_json::to_string_pretty(out).expect("serializes")
    };
    crate::checkpoint::write_atomic(output, body.as_bytes(), false)
        .map_err(|e| format!("{}: {e}", output.display()))
}

/// A new directory `<output>.runs/<seconds>-<pid>-<n>`, created exclusively,
/// owned by one invocation. Earlier invocations' directories are never
/// touched, so a report's capture files stay as it references them.
fn execution_dir(output: &Path) -> Result<PathBuf, String> {
    let mut name = output.as_os_str().to_owned();
    name.push(".runs");
    let runs = PathBuf::from(name);
    fs::create_dir_all(&runs).map_err(|e| format!("{}: {e}", runs.display()))?;
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    for n in 0u32.. {
        let dir = runs.join(format!("{secs}-{}-{n}", std::process::id()));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {e}", dir.display())),
        }
    }
    unreachable!("u32 names exhausted")
}

/// Run a plan, rewriting `output` after every observation (an existing
/// output file is replaced). The invocation owns a fresh directory
/// `<output>.runs/<id>` (see `execution_dir`): each run gets a scratch
/// directory `run-N` there, removed afterwards, and external output goes to
/// `N.stdout` and `N.stderr` there and is kept. A stop request ends the plan
/// after the current observation.
pub fn run_plan(
    plan: &Plan,
    output: &Path,
    stop: Option<&AtomicBool>,
    mut progress: impl FnMut(&Observation),
) -> Result<BenchOutput, String> {
    plan.validate()?;
    let (seq, seed_used) = schedule(plan);
    if let Some(dir) = output.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let work = execution_dir(output)?;
    let plan_json = serde_json::to_vec(plan).expect("plan serializes");
    // The exact bytes `plan_sha3` hashes, kept beside the captures.
    let manifest = work.join("plan.json");
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest)
        .and_then(|mut f| std::io::Write::write_all(&mut f, &plan_json))
        .map_err(|e| format!("{}: {e}", manifest.display()))?;
    let mut out = BenchOutput {
        plan: plan.clone(),
        plan_sha3: crate::proof::sha3_256(&[&plan_json])
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        seed_used,
        execution_dir: work.clone(),
        build: crate::build_identity().into(),
        crate_version: env!("CARGO_PKG_VERSION").into(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        observations: Vec::new(),
    };
    for (n, &(i, repeat, warmup)) in seq.iter().enumerate() {
        let entry = &plan.entries[i];
        let kind = match entry {
            Entry::Engine { .. } => "engine",
            Entry::External { .. } => "external",
        };
        let dir = work.join(format!("run-{n}"));
        let start = Instant::now();
        let res = match fs::create_dir(&dir) {
            Err(e) => Err(format!(
                "cannot create the scratch directory {}: {e}",
                dir.display()
            )),
            Ok(()) => {
                let r = match entry {
                    Entry::Engine {
                        test,
                        exponent,
                        backend,
                        options,
                        ..
                    } => engine_run(test, *exponent, backend, options, &dir, stop),
                    Entry::External {
                        argv,
                        cwd,
                        timeout_secs,
                        ..
                    } => external_run(
                        argv,
                        cwd.as_deref(),
                        *timeout_secs,
                        stop,
                        (
                            &work.join(format!("{n}.stdout")),
                            &work.join(format!("{n}.stderr")),
                        ),
                    ),
                };
                let _ = fs::remove_dir_all(&dir);
                r
            }
        };
        let mut obs =
            res.unwrap_or_else(|e| Observation::failed(kind, e, start.elapsed().as_secs_f64()));
        obs.entry = entry.name().to_string();
        obs.entry_index = i;
        match entry {
            Entry::Engine { test, exponent, .. } => {
                obs.test = Some(test.clone());
                obs.exponent = Some(*exponent);
                obs.result_source = "engine checked run".into();
                obs.start = "clean, checkpointing disabled".into();
            }
            Entry::External { argv, .. } => {
                obs.argv = Some(argv.clone());
                obs.result_source = "external process timing; result not verified".into();
                obs.start = "as the command decides".into();
            }
        }
        obs.sequence = n;
        obs.repeat = repeat;
        obs.warmup = warmup;
        progress(&obs);
        out.observations.push(obs);
        write_output(output, &out)?;
        if stop.is_some_and(|s| s.load(Ordering::SeqCst)) {
            break;
        }
    }
    Ok(out)
}

/// One CSV row per observation; timings and check counts flattened. Every
/// row repeats the provenance of the run (plan name and hash, seed, build,
/// platform, execution directory) and the entry's engine options; the exact
/// plan bytes that `plan_sha3` hashes are in `<execution_dir>/plan.json`.
/// `transform_plan` is the backend's arithmetic plan; `plan_name` and
/// `plan_sha3` identify the workload plan.
pub fn to_csv(out: &BenchOutput) -> String {
    const COLUMNS: &[&str] = &[
        "plan_name",
        "plan_sha3",
        "seed_used",
        "build",
        "os",
        "arch",
        "execution_dir",
        "entry",
        "entry_index",
        "kind",
        "test",
        "exponent",
        "options",
        "argv",
        "result_source",
        "start",
        "sequence",
        "repeat",
        "warmup",
        "status",
        "error",
        "wall_secs",
        "init",
        "iterations",
        "checks",
        "checkpoint_io",
        "proof_generation",
        "proof_verification",
        "verdict",
        "res64",
        "backend",
        "backend_build",
        "transform_plan",
        "device",
        "proof_stage",
        "check_step",
        "gerbicz_passed",
        "gerbicz_failed",
        "jacobi_passed",
        "jacobi_failed",
        "jacobi_inconclusive",
        "backend_faults",
        "rollbacks",
        "diagnostics",
        "exit_code",
        "stdout",
        "stderr",
        "energy_joules",
        "telemetry",
    ];
    let q = |v: &str| format!("\"{}\"", v.replace('"', "\"\""));
    let mut s = COLUMNS.join(",") + "\n";
    for o in &out.observations {
        let t = o.timings.as_ref();
        let f = |x: Option<f64>| x.map(|v| format!("{v:.6}")).unwrap_or_default();
        let r = o.reliability.as_ref();
        let c = |x: Option<u64>| x.map(|v| v.to_string()).unwrap_or_default();
        let b = o.backend.as_ref();
        let path = |p: &Option<PathBuf>| {
            q(&p.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default())
        };
        let row: Vec<String> = vec![
            q(&out.plan.name),
            out.plan_sha3.clone(),
            out.seed_used.map(|v| v.to_string()).unwrap_or_default(),
            q(&out.build),
            out.os.clone(),
            out.arch.clone(),
            q(&out.execution_dir.display().to_string()),
            q(&o.entry),
            o.entry_index.to_string(),
            o.kind.clone(),
            o.test.clone().unwrap_or_default(),
            o.exponent.map(|v| v.to_string()).unwrap_or_default(),
            q(&match out.plan.entries.get(o.entry_index) {
                Some(Entry::Engine { options, .. }) => {
                    serde_json::to_string(options).expect("options serialize")
                }
                _ => String::new(),
            }),
            q(&o.argv
                .as_ref()
                .map(|a| serde_json::to_string(a).expect("argv"))
                .unwrap_or_default()),
            q(&o.result_source),
            q(&o.start),
            o.sequence.to_string(),
            o.repeat.to_string(),
            o.warmup.to_string(),
            o.status.clone(),
            q(o.error.as_deref().unwrap_or("")),
            format!("{:.6}", o.wall_secs),
            f(t.map(|t| t.init)),
            f(t.map(|t| t.iterations)),
            f(t.map(|t| t.checks)),
            f(t.map(|t| t.checkpoint_io)),
            f(t.map(|t| t.proof_generation)),
            f(t.map(|t| t.proof_verification)),
            o.verdict.clone().unwrap_or_default(),
            o.res64.clone().unwrap_or_default(),
            b.map(|b| b.kind.clone()).unwrap_or_default(),
            q(b.map(|b| b.build.as_str()).unwrap_or("")),
            q(b.map(|b| b.plan.as_str()).unwrap_or("")),
            q(b.map(|b| b.device.as_str()).unwrap_or("")),
            o.proof_stage.map(|s| s.name()).unwrap_or("").into(),
            c(o.check_step),
            c(r.map(|r| r.gerbicz_passed)),
            c(r.map(|r| r.gerbicz_failed)),
            c(r.map(|r| r.jacobi_passed)),
            c(r.map(|r| r.jacobi_failed)),
            c(r.map(|r| r.jacobi_inconclusive)),
            c(r.map(|r| r.backend_faults)),
            c(r.map(|r| r.rollbacks)),
            q(&o.diagnostics
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(";")),
            o.exit_code.map(|c| c.to_string()).unwrap_or_default(),
            path(&o.stdout),
            path(&o.stderr),
            o.energy_joules.map(|e| e.to_string()).unwrap_or_default(),
            o.telemetry.clone(),
        ];
        debug_assert_eq!(row.len(), COLUMNS.len());
        s += &row.join(",");
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(order: Order) -> Plan {
        Plan {
            name: "t".into(),
            order,
            seed: Some(7),
            repeats: 3,
            warmups: 1,
            entries: ["a", "b", "c"]
                .iter()
                .map(|n| Entry::External {
                    name: n.to_string(),
                    argv: vec!["true".into()],
                    cwd: None,
                    timeout_secs: None,
                })
                .collect(),
        }
    }

    #[test]
    fn orders() {
        let (s, _) = schedule(&plan(Order::Paired));
        let ids: Vec<usize> = s.iter().map(|x| x.0).collect();
        assert_eq!(ids, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert!(s[..3].iter().all(|x| x.2) && s[3..].iter().all(|x| !x.2));
        let (s, _) = schedule(&plan(Order::Sequential));
        assert_eq!(
            s[3..].iter().map(|x| x.0).collect::<Vec<_>>(),
            [0, 0, 0, 1, 1, 1, 2, 2, 2]
        );
        let (a, seed) = schedule(&plan(Order::Randomized));
        let (b, _) = schedule(&plan(Order::Randomized));
        assert_eq!(seed, Some(7));
        assert_eq!(a, b, "the same seed gives the same order");
        for r in 0..3 {
            let mut ids: Vec<usize> = a[3..].iter().filter(|x| x.1 == r).map(|x| x.0).collect();
            ids.sort_unstable();
            assert_eq!(ids, [0, 1, 2]);
        }
    }
}
