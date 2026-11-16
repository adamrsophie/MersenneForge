// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Benchmark harness. A plan lists engine workloads (run in process on a
//! named backend) and external commands (timed as subprocesses, with argv
//! given verbatim so no flag of another program is invented here). Every run
//! is recorded as a raw observation, failures included. No energy value is
//! produced unless measured; this harness has no power telemetry, so energy is
//! always recorded as unavailable.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::ll::{run_ll, JacobiCadence, LlConfig};
use crate::prp::{run_prp, PrpConfig};
use crate::run::{Control, ProcessState, Timings};
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub entry: String,
    pub kind: String,
    /// Position in the executed sequence.
    pub sequence: usize,
    pub repeat: u32,
    pub warmup: bool,
    /// "completed", "incomplete", or "failed".
    pub status: String,
    pub error: Option<String>,
    pub wall_secs: f64,
    /// Engine runs: wall time split by activity.
    pub timings: Option<Timings>,
    pub verdict: Option<String>,
    pub res64: Option<String>,
    pub exit_code: Option<i32>,
    pub energy_joules: Option<f64>,
    pub telemetry: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchOutput {
    pub plan: Plan,
    pub seed_used: Option<u64>,
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
) -> Result<Observation, String> {
    let kind = BackendKind::parse(backend).ok_or_else(|| format!("unknown backend {backend}"))?;
    if kind.is_device() && o.budget_secs.is_none() {
        return Err(format!("{backend} entries need budget_secs"));
    }
    let start = Instant::now();
    let deadline = o.budget_secs.map(|s| start + Duration::from_secs_f64(s));
    let mut be = select::open(kind, p, o.device).map_err(|e| e.to_string())?;
    let init = start.elapsed().as_secs_f64();
    let mut ctl = Control {
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
    let mut timings = report.timings.clone();
    timings.init += init;
    let status = match report.process {
        ProcessState::Completed => "completed",
        _ => "incomplete",
    };
    Ok(Observation {
        entry: String::new(),
        kind: "engine".into(),
        sequence: 0,
        repeat: 0,
        warmup: false,
        status: status.into(),
        error: None,
        wall_secs: start.elapsed().as_secs_f64(),
        timings: Some(timings),
        verdict: report.result.as_ref().map(|r| format!("{:?}", r.verdict)),
        res64: report.result.map(|r| r.res64),
        exit_code: None,
        energy_joules: None,
        telemetry: "unavailable".into(),
    })
}

fn external_run(
    argv: &[String],
    cwd: Option<&Path>,
    timeout: Option<f64>,
) -> Result<Observation, String> {
    let (prog, args) = argv
        .split_first()
        .ok_or("external entry needs a nonempty argv")?;
    let mut cmd = std::process::Command::new(prog);
    cmd.args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let start = Instant::now();
    let mut child = cmd.spawn().map_err(|e| format!("{prog}: {e}"))?;
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(s);
        }
        if timeout.is_some_and(|t| start.elapsed().as_secs_f64() > t) {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let wall = start.elapsed().as_secs_f64();
    let (st, code) = match status {
        Some(s) if s.success() => ("completed", s.code()),
        Some(s) => ("failed", s.code()),
        None => ("incomplete", None),
    };
    Ok(Observation {
        entry: String::new(),
        kind: "external".into(),
        sequence: 0,
        repeat: 0,
        warmup: false,
        status: st.into(),
        error: if st == "completed" {
            None
        } else {
            Some("nonzero exit or timeout".into())
        },
        wall_secs: wall,
        timings: None,
        verdict: None,
        res64: None,
        exit_code: code,
        energy_joules: None,
        telemetry: "unavailable".into(),
    })
}

/// Run a plan. `work` holds per-run scratch directories (removed after each run).
pub fn run_plan(plan: &Plan, work: &Path, mut progress: impl FnMut(&Observation)) -> BenchOutput {
    let (seq, seed_used) = schedule(plan);
    let mut observations = Vec::new();
    for (n, &(i, repeat, warmup)) in seq.iter().enumerate() {
        let entry = &plan.entries[i];
        let dir = work.join(format!("run-{n}"));
        let _ = std::fs::create_dir_all(&dir);
        let start = Instant::now();
        let res = match entry {
            Entry::Engine {
                test,
                exponent,
                backend,
                options,
                ..
            } => engine_run(test, *exponent, backend, options, &dir),
            Entry::External {
                argv,
                cwd,
                timeout_secs,
                ..
            } => external_run(argv, cwd.as_deref(), *timeout_secs),
        };
        let _ = std::fs::remove_dir_all(&dir);
        let mut obs = res.unwrap_or_else(|e| Observation {
            entry: String::new(),
            kind: match entry {
                Entry::Engine { .. } => "engine".into(),
                Entry::External { .. } => "external".into(),
            },
            sequence: 0,
            repeat: 0,
            warmup: false,
            status: "failed".into(),
            error: Some(e),
            wall_secs: start.elapsed().as_secs_f64(),
            timings: None,
            verdict: None,
            res64: None,
            exit_code: None,
            energy_joules: None,
            telemetry: "unavailable".into(),
        });
        obs.entry = entry.name().to_string();
        obs.sequence = n;
        obs.repeat = repeat;
        obs.warmup = warmup;
        progress(&obs);
        observations.push(obs);
    }
    let _ = std::fs::remove_dir(work);
    BenchOutput {
        plan: plan.clone(),
        seed_used,
        crate_version: env!("CARGO_PKG_VERSION").into(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        observations,
    }
}

/// One CSV row per observation; timings flattened.
pub fn to_csv(out: &BenchOutput) -> String {
    let mut s = String::from(
        "entry,kind,sequence,repeat,warmup,status,error,wall_secs,init,iterations,checks,checkpoint_io,proof_generation,proof_verification,verdict,res64,exit_code,energy_joules,telemetry\n",
    );
    let q = |v: &str| format!("\"{}\"", v.replace('"', "\"\""));
    for o in &out.observations {
        let t = o.timings.clone();
        let f = |x: Option<f64>| x.map(|v| format!("{v:.6}")).unwrap_or_default();
        s += &format!(
            "{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{}\n",
            q(&o.entry),
            o.kind,
            o.sequence,
            o.repeat,
            o.warmup,
            o.status,
            q(o.error.as_deref().unwrap_or("")),
            o.wall_secs,
            f(t.as_ref().map(|t| t.init)),
            f(t.as_ref().map(|t| t.iterations)),
            f(t.as_ref().map(|t| t.checks)),
            f(t.as_ref().map(|t| t.checkpoint_io)),
            f(t.as_ref().map(|t| t.proof_generation)),
            f(t.as_ref().map(|t| t.proof_verification)),
            o.verdict.as_deref().unwrap_or(""),
            o.res64.as_deref().unwrap_or(""),
            o.exit_code.map(|c| c.to_string()).unwrap_or_default(),
            o.energy_joules.map(|e| e.to_string()).unwrap_or_default(),
            o.telemetry
        );
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
