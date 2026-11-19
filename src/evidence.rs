// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Benchmark evidence that is not arithmetic: the provenance of the harness
//! executable and host, optional power telemetry, and paired analysis of
//! recorded observations. Every value that cannot be determined is recorded
//! as unknown, never guessed.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::bench::{BenchOutput, Observation};

const UNKNOWN: &str = "unknown";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA3-256 of a file, streamed.
pub fn file_sha3(path: &std::path::Path) -> std::io::Result<String> {
    use sha3::{Digest, Sha3_256};
    let mut h = Sha3_256::new();
    let mut f = std::fs::File::open(path)?;
    std::io::copy(&mut f, &mut h)?;
    Ok(hex(&h.finalize()))
}

/// What ran the benchmark and where.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub executable: String,
    pub executable_sha3: String,
    /// Source identity (see `build_identity`).
    pub source_identity: String,
    pub crate_version: String,
    /// "debug" or "release", from `debug_assertions`.
    pub profile: String,
    pub target_arch: String,
    pub target_os: String,
    pub target_env: String,
    /// Instruction set features enabled at compile time, from a fixed list.
    pub target_features: Vec<String>,
    /// Cargo features compiled in.
    pub cargo_features: Vec<String>,
    pub lockfile_sha3: String,
    /// The compiler is not recorded in the executable: always unknown.
    pub rustc: String,
    pub cpu_model: String,
    pub logical_cpus: String,
    /// No power policy is read: always unknown.
    pub power_policy: String,
}

impl Provenance {
    pub fn collect() -> Provenance {
        let exe = std::env::current_exe().ok();
        let features: &[(&str, bool)] = &[
            ("sse4.2", cfg!(target_feature = "sse4.2")),
            ("avx2", cfg!(target_feature = "avx2")),
            ("fma", cfg!(target_feature = "fma")),
            ("avx512f", cfg!(target_feature = "avx512f")),
            ("neon", cfg!(target_feature = "neon")),
        ];
        Provenance {
            executable: exe
                .as_ref()
                .map_or(UNKNOWN.into(), |p| p.display().to_string()),
            executable_sha3: exe
                .as_deref()
                .and_then(|p| file_sha3(p).ok())
                .unwrap_or_else(|| UNKNOWN.into()),
            source_identity: crate::build_identity().into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .into(),
            target_arch: std::env::consts::ARCH.into(),
            target_os: std::env::consts::OS.into(),
            target_env: if cfg!(target_env = "gnu") {
                "gnu"
            } else if cfg!(target_env = "musl") {
                "musl"
            } else if cfg!(target_env = "msvc") {
                "msvc"
            } else {
                "none"
            }
            .into(),
            target_features: features
                .iter()
                .filter(|f| f.1)
                .map(|f| f.0.to_string())
                .collect(),
            cargo_features: [
                ("gpu", cfg!(feature = "gpu")),
                ("cuda", cfg!(feature = "cuda")),
            ]
            .iter()
            .filter(|f| f.1)
            .map(|f| f.0.to_string())
            .collect(),
            lockfile_sha3: hex(&crate::proof::sha3_256(&[include_bytes!("../Cargo.lock")])),
            rustc: env!("MERSENNE_FORGE_RUSTC").into(),
            cpu_model: cpu_model().unwrap_or_else(|| UNKNOWN.into()),
            logical_cpus: std::thread::available_parallelism()
                .map_or(UNKNOWN.into(), |n| n.to_string()),
            power_policy: power_policy(),
        }
    }
}

/// The CPU model string where the operating system reports one.
fn cpu_model() -> Option<String> {
    if cfg!(target_os = "macos") {
        let o = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        let s = String::from_utf8(o.stdout).ok()?.trim().to_string();
        return (o.status.success() && !s.is_empty()).then_some(s);
    }
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    text.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim().to_string())
}

/// A power sensor read by running a command. The command prints the current
/// power in watts as the first number on its standard output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetrySpec {
    pub argv: Vec<String>,
    /// Sampling interval in milliseconds (at least 10).
    pub interval_ms: u64,
    /// Free text naming the sensor (for example a device and its counter).
    pub sensor: String,
    /// "device" (one device's power) or "system" (wall power of the host).
    /// The two are different metrics and are never added together.
    pub scope: String,
}

impl TelemetrySpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.argv.is_empty() {
            return Err("telemetry argv is empty".into());
        }
        if self.interval_ms < 10 {
            return Err("telemetry interval_ms must be at least 10".into());
        }
        if !matches!(self.scope.as_str(), "device" | "system") {
            return Err(format!(
                "telemetry scope must be device or system, got {:?}",
                self.scope
            ));
        }
        Ok(())
    }
}

/// One reading: seconds since the observation started, and watts, or None
/// when the command failed or printed no number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerSample {
    pub t_secs: f64,
    pub watts: Option<f64>,
}

/// Power readings of one observation and the energy integrated from them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerRecord {
    pub sensor: String,
    pub scope: String,
    pub interval_ms: u64,
    pub unit: String,
    pub samples: Vec<PowerSample>,
    pub missing: usize,
    /// Trapezoidal integral of power over the sampled span, in joules. Null
    /// unless at least two readings exist and none is missing.
    pub energy_joules: Option<f64>,
    /// Seconds between the first and last reading: what the energy covers.
    pub span_secs: f64,
}

/// The host's power policy where it can be read without privileges: the
/// Linux CPU frequency governor, or the macOS low power mode setting.
fn power_policy() -> String {
    if let Ok(g) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    {
        return format!("cpufreq governor {}", g.trim());
    }
    if cfg!(target_os = "macos") {
        if let Ok(o) = std::process::Command::new("pmset").arg("-g").output() {
            let text = String::from_utf8_lossy(&o.stdout);
            if let Some(l) = text
                .lines()
                .find(|l| l.trim_start().starts_with("lowpowermode"))
            {
                return l.split_whitespace().collect::<Vec<_>>().join(" ");
            }
        }
    }
    UNKNOWN.into()
}

/// Read one power value in watts from a built-in source, for use as the
/// benchmark telemetry command (`mersenne-forge power --source ...`):
///
/// * `rapl`: Linux powercap package energy counter, sampled 100 ms apart
///   (`/sys/class/powercap/intel-rapl:0`, readable by root on most systems);
///   system CPU package scope.
/// * `nvidia`: `nvidia-smi --query-gpu=power.draw` for `device`; device scope.
/// * `powermetrics`: macOS `powermetrics` combined CPU, GPU, and ANE power
///   (needs root); system scope.
pub fn sample_power(source: &str, device: u32) -> Result<f64, String> {
    match source {
        "rapl" => {
            let base = Path::new("/sys/class/powercap/intel-rapl:0");
            let read = |f: &str| -> Result<u64, String> {
                std::fs::read_to_string(base.join(f))
                    .map_err(|e| format!("{}: {e}", base.join(f).display()))?
                    .trim()
                    .parse()
                    .map_err(|e| format!("{f}: {e}"))
            };
            let range = read("max_energy_range_uj")?;
            let (e0, t0) = (read("energy_uj")?, Instant::now());
            std::thread::sleep(Duration::from_millis(100));
            let (e1, dt) = (read("energy_uj")?, t0.elapsed().as_secs_f64());
            // The counter wraps at max_energy_range_uj.
            let de = if e1 >= e0 { e1 - e0 } else { e1 + range - e0 };
            Ok(de as f64 / 1e6 / dt)
        }
        "nvidia" => {
            let o = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                    "-i",
                    &device.to_string(),
                ])
                .output()
                .map_err(|e| format!("nvidia-smi: {e}"))?;
            if !o.status.success() {
                return Err(format!("nvidia-smi exited with {}", o.status));
            }
            parse_watts(&String::from_utf8_lossy(&o.stdout))
                .ok_or_else(|| "nvidia-smi printed no power value".into())
        }
        "powermetrics" => {
            let o = std::process::Command::new("powermetrics")
                .args(["-n", "1", "-i", "100", "--samplers", "cpu_power,gpu_power"])
                .output()
                .map_err(|e| format!("powermetrics: {e}"))?;
            if !o.status.success() {
                return Err(format!(
                    "powermetrics exited with {} (it needs root)",
                    o.status
                ));
            }
            let text = String::from_utf8_lossy(&o.stdout);
            let line = text
                .lines()
                .find(|l| l.contains("Combined Power"))
                .ok_or("powermetrics printed no combined power line")?;
            let mw = parse_watts(line.rsplit(':').next().unwrap_or(""))
                .ok_or("unreadable combined power value")?;
            Ok(mw / 1000.0)
        }
        other => Err(format!(
            "unknown power source {other:?}: use rapl, nvidia, or powermetrics"
        )),
    }
}

/// One power reading: run the sensor command for at most `limit`, read at
/// most 4 KiB of its output, and take the first number in it. A sensor that
/// hangs, fails, or prints no finite nonnegative number gives None.
fn read_watts(argv: &[String], limit: Duration) -> Option<f64> {
    use std::io::Read;
    let mut child = std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let mut out = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut out).take(4096).read_to_end(&mut buf);
        buf
    });
    let deadline = Instant::now() + limit;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(1)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let text = reader.join().ok()?;
    if !status?.success() {
        return None;
    }
    parse_watts(&String::from_utf8_lossy(&text))
}

/// The first number in `text`, including signed exponents (1.2e+03).
fn parse_watts(text: &str) -> Option<f64> {
    // Numbers, with the text that follows each. A number followed by a "W"
    // unit wins; otherwise the text must hold exactly one number, so a
    // label such as "GPU0" is never read as the value.
    let num = |c: char| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E');
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(|c: char| c.is_ascii_digit() || c == '.') {
        let tail = &rest[start..];
        let len = tail.find(|c: char| !num(c)).unwrap_or(tail.len());
        let (tok, after) = tail.split_at(len);
        let before = rest[..start].chars().next_back();
        let labeled = before.is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        let sign = if before == Some('-') { -1.0 } else { 1.0 };
        if let Ok(v) = tok.parse::<f64>().map(|v| sign * v) {
            if !labeled {
                found.push((v, after.trim_start().starts_with('W')));
            }
        }
        rest = after;
    }
    let pick = match found.iter().find(|(_, unit)| *unit) {
        Some(&(v, _)) => Some(v),
        None if found.len() == 1 => Some(found[0].0),
        None => None,
    };
    pick.filter(|w| w.is_finite() && *w >= 0.0)
}

/// A sampler running on its own thread while a workload runs.
pub struct Sampler {
    spec: TelemetrySpec,
    stop: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<Vec<PowerSample>>,
}

impl Sampler {
    pub fn start(spec: &TelemetrySpec) -> Sampler {
        let stop = Arc::new(AtomicBool::new(false));
        let (flag, argv, every) = (
            stop.clone(),
            spec.argv.clone(),
            Duration::from_millis(spec.interval_ms),
        );
        let handle = std::thread::spawn(move || {
            let t0 = Instant::now();
            let mut v = Vec::new();
            // A reading may take at most ten intervals, and at least 1 s.
            let limit = (every * 10).max(Duration::from_secs(1));
            let sample = |v: &mut Vec<PowerSample>| {
                v.push(PowerSample {
                    t_secs: t0.elapsed().as_secs_f64(),
                    watts: read_watts(&argv, limit),
                })
            };
            'outer: loop {
                if flag.load(Ordering::SeqCst) {
                    break;
                }
                sample(&mut v);
                // The next tick after now: ticks missed by a slow sensor are
                // skipped, not queued.
                let ticks = (t0.elapsed().as_nanos() / every.as_nanos().max(1)) as u32 + 1;
                let next = t0 + every * ticks;
                // Wait in short steps so a finished workload is not delayed.
                while Instant::now() < next {
                    if flag.load(Ordering::SeqCst) {
                        break 'outer;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
            // A closing reading, so the span reaches the end of the workload.
            sample(&mut v);
            v
        });
        Sampler {
            spec: spec.clone(),
            stop,
            handle,
        }
    }

    /// Stop sampling, take one last reading, and integrate.
    pub fn finish(self) -> PowerRecord {
        self.stop.store(true, Ordering::SeqCst);
        let samples = self.handle.join().unwrap_or_default();
        integrate(&self.spec, samples)
    }
}

pub fn integrate(spec: &TelemetrySpec, samples: Vec<PowerSample>) -> PowerRecord {
    let missing = samples.iter().filter(|s| s.watts.is_none()).count();
    let span = match (samples.first(), samples.last()) {
        (Some(a), Some(b)) => b.t_secs - a.t_secs,
        _ => 0.0,
    };
    let energy = (missing == 0 && samples.len() >= 2).then(|| {
        samples
            .windows(2)
            .map(|w| {
                let (a, b) = (&w[0], &w[1]);
                (b.t_secs - a.t_secs) * (a.watts.unwrap_or(0.0) + b.watts.unwrap_or(0.0)) / 2.0
            })
            .sum()
    });
    PowerRecord {
        sensor: spec.sensor.clone(),
        scope: spec.scope.clone(),
        interval_ms: spec.interval_ms,
        unit: "W".into(),
        samples,
        missing,
        energy_joules: energy,
        span_secs: span,
    }
}

/// The value of `metric` for one observation, when it has one.
fn metric_value(o: &Observation, metric: &str) -> Option<f64> {
    match metric {
        "wall_secs" => Some(o.wall_secs),
        "iterations" => o.timings.as_ref().map(|t| t.iterations),
        "energy_joules" => o.energy_joules,
        _ => None,
    }
}

/// Paired comparison of two entries of one recorded plan. Pairs are the
/// measured (not warmup) observations of the same repeat; a pair counts only
/// when both runs completed. The ratio is candidate / baseline. The median
/// ratio has a percentile bootstrap 95% interval from `resamples` draws with
/// a fixed seed, so the report is reproducible. Failed and incomplete runs
/// are listed with the wall time they cost; they are never dropped silently.
pub fn analyze(
    out: &BenchOutput,
    baseline: usize,
    candidate: usize,
    metric: &str,
    seed: u64,
) -> Result<serde_json::Value, String> {
    if !matches!(metric, "wall_secs" | "iterations" | "energy_joules") {
        return Err(format!(
            "unknown metric {metric:?}: use wall_secs, iterations, or energy_joules"
        ));
    }
    let n = out.plan.entries.len();
    for (what, i) in [("baseline", baseline), ("candidate", candidate)] {
        if i >= n {
            return Err(format!(
                "{what} index {i} is not an entry (the plan has {n})"
            ));
        }
    }
    if baseline == candidate {
        return Err("baseline and candidate are the same entry".into());
    }
    if out.schema_version > crate::bench::SCHEMA_VERSION {
        return Err(format!(
            "schema version {} is newer than this build reads ({})",
            out.schema_version,
            crate::bench::SCHEMA_VERSION
        ));
    }
    let measured = |i: usize| -> Vec<&Observation> {
        out.observations
            .iter()
            .filter(|o| o.entry_index == i && !o.warmup)
            .collect()
    };
    let (base, cand) = (measured(baseline), measured(candidate));
    for (what, v) in [("baseline", &base), ("candidate", &cand)] {
        let mut seen = std::collections::HashSet::new();
        if let Some(o) = v.iter().find(|o| !seen.insert(o.repeat)) {
            return Err(format!(
                "the {what} has more than one measured observation for repeat {}",
                o.repeat
            ));
        }
    }
    let cost = |v: &[&Observation]| {
        let not_done: Vec<_> = v
            .iter()
            .filter(|o| o.status != "completed")
            .map(|o| serde_json::json!({"sequence": o.sequence, "repeat": o.repeat, "status": o.status, "error": o.error, "wall_secs": o.wall_secs}))
            .collect();
        serde_json::json!({
            "runs": v.len(),
            "completed": v.iter().filter(|o| o.status == "completed").count(),
            "not_completed": not_done,
            "total_wall_secs": v.iter().map(|o| o.wall_secs).sum::<f64>(),
        })
    };
    let mut ratios = Vec::new();
    let mut pairs = Vec::new();
    let mut disagreements = 0usize;
    for b in &base {
        let Some(c) = cand.iter().find(|c| c.repeat == b.repeat) else {
            continue;
        };
        // A pair compares the same work: the same test and exponent where
        // both record them, the same result where both have one, and for
        // energy the same sensor scope.
        let both = |x: &Option<String>, y: &Option<String>| match (x, y) {
            (Some(x), Some(y)) => x == y,
            _ => true,
        };
        let same_work = both(&b.test, &c.test)
            && match (b.exponent, c.exponent) {
                (Some(x), Some(y)) => x == y,
                _ => true,
            };
        let same_result = both(&b.verdict, &c.verdict) && both(&b.res64, &c.res64);
        let same_scope = metric != "energy_joules"
            || matches!((&b.power, &c.power), (Some(x), Some(y)) if x.scope == y.scope);
        let excluded = if b.status != "completed" || c.status != "completed" {
            Some("not both completed")
        } else if !same_work {
            Some("different workloads")
        } else if !same_result {
            disagreements += 1;
            Some("results disagree")
        } else if !same_scope {
            Some("different power scopes")
        } else {
            None
        };
        let (vb, vc) = (metric_value(b, metric), metric_value(c, metric));
        let ratio = match (excluded, vb, vc) {
            (None, Some(x), Some(y)) if x > 0.0 => Some(y / x),
            _ => None,
        };
        if let Some(r) = ratio {
            ratios.push(r);
        }
        pairs.push(serde_json::json!({"repeat": b.repeat, "baseline": vb, "candidate": vc, "ratio": ratio, "excluded": excluded}));
    }
    let median = |v: &mut Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        let m = v.len() / 2;
        Some(if v.len() % 2 == 1 {
            v[m]
        } else {
            (v[m - 1] + v[m]) / 2.0
        })
    };
    let resamples = 2000;
    let interval = (ratios.len() >= 2).then(|| {
        let mut st = seed;
        let mut meds: Vec<f64> = (0..resamples)
            .map(|_| {
                let mut s: Vec<f64> = (0..ratios.len())
                    .map(|_| ratios[(next(&mut st) % ratios.len() as u64) as usize])
                    .collect();
                median(&mut s).expect("nonempty")
            })
            .collect();
        meds.sort_by(f64::total_cmp);
        [
            meds[resamples * 25 / 1000],
            meds[resamples * 975 / 1000 - 1],
        ]
    });
    let mut sorted = ratios.clone();
    let name = |i: usize| out.plan.entries[i].name().to_string();
    Ok(serde_json::json!({
        "metric": metric,
        "ratio": "candidate / baseline",
        "baseline": {"index": baseline, "name": name(baseline), "cost": cost(&base)},
        "candidate": {"index": candidate, "name": name(candidate), "cost": cost(&cand)},
        "pairs": pairs,
        "usable_pairs": ratios.len(),
        "result_disagreements": disagreements,
        "median_ratio": median(&mut sorted),
        "bootstrap": {"interval_95": interval, "resamples": resamples, "seed": seed},
        "note": if ratios.len() < 5 {
            "fewer than 5 usable pairs: the interval is not a meaningful uncertainty estimate"
        } else {
            "each pair is one complete run per entry in the same repeat; iterations are not samples"
        },
        "plan_sha3": out.plan_sha3,
        "source_identity": out.build,
    }))
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Where a retained or verified artifact is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    /// The retained copy, or None when it was not kept.
    pub path: Option<PathBuf>,
    pub bytes: u64,
    pub sha3: String,
    /// Why the file was not kept (for example the retention quota).
    pub not_retained: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> TelemetrySpec {
        TelemetrySpec {
            argv: vec!["x".into()],
            interval_ms: 100,
            sensor: "s".into(),
            scope: "device".into(),
        }
    }

    #[test]
    fn energy_is_the_trapezoid_and_missing_readings_void_it() {
        let s = |t: f64, w: Option<f64>| PowerSample {
            t_secs: t,
            watts: w,
        };
        let r = integrate(
            &spec(),
            vec![s(0.0, Some(10.0)), s(1.0, Some(20.0)), s(3.0, Some(20.0))],
        );
        assert_eq!(r.energy_joules, Some(15.0 + 40.0));
        assert_eq!(r.span_secs, 3.0);
        let r = integrate(
            &spec(),
            vec![s(0.0, Some(10.0)), s(1.0, None), s(2.0, Some(10.0))],
        );
        assert_eq!((r.energy_joules, r.missing), (None, 1));
        assert_eq!(
            integrate(&spec(), vec![s(0.0, Some(5.0))]).energy_joules,
            None
        );
    }

    #[test]
    fn power_readings_parse_signed_exponents() {
        assert_eq!(parse_watts("1.2e+03 W"), Some(1200.0));
        assert_eq!(parse_watts("power: 12.5 W"), Some(12.5));
        assert_eq!(parse_watts("5e-1"), Some(0.5));
        assert_eq!(parse_watts("-3 W"), None);
        assert_eq!(parse_watts("none"), None);
        assert_eq!(parse_watts("GPU0: 35 W"), Some(35.0));
        assert_eq!(parse_watts("GPU0: 35"), Some(35.0));
        assert_eq!(parse_watts("35 W (limit 300 W)"), Some(35.0));
        assert_eq!(parse_watts("35 then 300"), None);
    }

    #[test]
    fn telemetry_specs_are_checked() {
        assert!(spec().validate().is_ok());
        let mut t = spec();
        t.scope = "both".into();
        assert!(t.validate().is_err());
        let mut t = spec();
        t.interval_ms = 1;
        assert!(t.validate().is_err());
    }

    #[test]
    fn provenance_names_unknowns() {
        let p = Provenance::collect();
        assert!(p.rustc.starts_with("rustc ") || p.rustc == "unknown");
        assert!(!p.power_policy.is_empty());
        assert_eq!(p.source_identity, crate::build_identity());
        assert_eq!(p.lockfile_sha3.len(), 64);
        assert_eq!(p.executable_sha3.len(), 64);
    }
}
