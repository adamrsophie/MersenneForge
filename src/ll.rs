// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Lucas-Lehmer with the Jacobi check, written once over [`Backend`].
//!
//! s_0 = 4, s_(k+1) = s_k^2 - 2 mod N; M_p is prime iff s_(p-2) = 0.
//!
//! Assurance follows PRPLL (`Gpu::isPrimeLL`): with w = 2 + sqrt(3),
//! s_k - 2 = 3 t^2 for an integer t when k >= 1, and (3 | N) = -1, so the
//! Jacobi symbol (s_k - 2 | N) must be -1 for prime and composite N alike.
//! s = 2 and s = N - 2 (the path of a zeroed residue) count as failures. A
//! symbol of 0 is inconclusive. The check catches about half of all errors,
//! and an error is caught at the first check after it or never. It is not
//! the Gerbicz check: the LL recurrence has no such check.
//!
//! The symbol is computed on the host from an exported residue, at a log
//! step when the time since the last check is at least 100 times its cost,
//! at the first log step of a process, after a failure when the run returns
//! to the failed iteration, and always on the final residue. A residue that
//! passes becomes the verified state; a failure rolls back to it.

use std::path::PathBuf;
use std::time::Instant;

use num_bigint::BigUint;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

use crate::backend::{Backend, BackendError};
use crate::checkpoint::{Checkpoint, CheckpointError, CheckpointMeta, FORMAT_VERSION};
use crate::residue::{modulus, Residue};
use crate::run::{
    round_up, Control, Evidence, FaultTarget, MathResult, ProcessState, ProtocolError, RunReport,
    Stopwatch, Timings, Verdict,
};

pub const PROTOCOL: &str = "ll-jacobi/1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JacobiCadence {
    /// Upstream: cost aware (time since last check >= 100 x its cost).
    Auto,
    /// At every log step (tests and small exponents).
    EveryLog,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlConfig {
    pub log_step: u64,
    pub jacobi: JacobiCadence,
    pub checkpoint: Option<PathBuf>,
    pub max_batch: u64,
}

impl Default for LlConfig {
    fn default() -> Self {
        LlConfig {
            log_step: 20_000,
            jacobi: JacobiCadence::Auto,
            checkpoint: None,
            max_batch: 1000,
        }
    }
}

/// Jacobi symbol (a | n) for odd n > 0.
pub fn jacobi(a: &BigUint, n: &BigUint) -> i32 {
    assert!(n.bit(0), "the Jacobi symbol needs an odd modulus");
    let low = |x: &BigUint| x.iter_u32_digits().next().unwrap_or(0);
    let mut a = a % n;
    let mut n = n.clone();
    let mut t = 1;
    while !a.is_zero() {
        let tz = a.trailing_zeros().expect("nonzero");
        a >>= tz;
        if tz % 2 == 1 && matches!(low(&n) % 8, 3 | 5) {
            t = -t;
        }
        std::mem::swap(&mut a, &mut n);
        if low(&a) % 4 == 3 && low(&n) % 4 == 3 {
            t = -t;
        }
        a %= &n;
    }
    if n.is_one() {
        t
    } else {
        0
    }
}

/// Outcome of the Jacobi check on s_k (k >= 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobiOutcome {
    Pass,
    Fail,
    Inconclusive,
}

pub fn jacobi_check(s: &Residue) -> JacobiOutcome {
    let p = s.exponent();
    if *s == Residue::from_u64(p, 2) || *s == Residue::zero(p).minus_small(2) {
        return JacobiOutcome::Fail;
    }
    match jacobi(&s.minus_small(2).to_biguint(), &modulus(p)) {
        -1 => JacobiOutcome::Pass,
        0 => JacobiOutcome::Inconclusive,
        _ => JacobiOutcome::Fail,
    }
}

pub fn validate(p: u32, cfg: &LlConfig) -> Result<(), ProtocolError> {
    crate::exponent::validate(p).map_err(|e| ProtocolError::Config(e.to_string()))?;
    if cfg.log_step == 0 || cfg.max_batch == 0 {
        return Err(ProtocolError::Config(
            "log step and max batch must be positive".into(),
        ));
    }
    Ok(())
}

/// Run (or resume) a Lucas-Lehmer test of M_p.
pub fn run_ll<B: Backend>(
    be: &mut B,
    p: u32,
    cfg: &LlConfig,
    requested: serde_json::Value,
    resume: Option<&Checkpoint>,
    ctl: &mut Control,
) -> Result<RunReport, ProtocolError> {
    let init = Stopwatch::start();
    let mut timings = Timings::default();
    let mut ev = Evidence {
        backend: Some(be.identity()),
        ..Evidence::default()
    };
    validate(p, cfg)?;
    if be.exponent() != p {
        return Err(ProtocolError::Config(format!(
            "backend is for M{}, run is for M{p}",
            be.exponent()
        )));
    }
    if p == 2 {
        ev.notes
            .push("M2 = 3 is prime by inspection; the recurrence has no steps".into());
        let z = Residue::zero(2);
        return Ok(RunReport {
            process: ProcessState::Completed,
            result: Some(MathResult {
                test: "ll".into(),
                exponent: 2,
                verdict: Verdict::Prime,
                res64: format!("{:016x}", 0),
                res2048: z.res2048_hex(),
                special_case: Some("p = 2".into()),
            }),
            evidence: ev,
            timings,
        });
    }
    let config_json = serde_json::to_value(cfg).expect("config serializes");
    let kend = u64::from(p) - 2;
    let s0 = Residue::from_u64(p, 4);
    let (mut k, start_s, mut verified, elapsed_before) = match resume {
        None => (0, s0.clone(), (0u64, s0), 0.0),
        Some(ck) => {
            let m = &ck.meta;
            if m.protocol != PROTOCOL
                || m.exponent != p
                || ck.residues.len() != 2
                || m.verified_k > m.k
                || m.k >= kend
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
            if crate::prp::without_checkpoint(&m.config)
                != crate::prp::without_checkpoint(&config_json)
            {
                return Err(CheckpointError::Mismatch("run configuration differs".into()).into());
            }
            (
                m.k,
                ck.residues[0].clone(),
                (m.verified_k, ck.residues[1].clone()),
                m.elapsed_secs,
            )
        }
    };
    let s = be.alloc()?;
    be.import(s, &start_s)?;
    let start = Stopwatch::start();
    let mut last_check: Option<(Instant, f64)> = None;
    let mut recheck: Option<u64> = None;
    let mut n_seq = 0u32;
    let mut last_failed: Option<(u64, Residue)> = None;
    timings.init = init.secs();

    let outcome = loop {
        let mut next = round_up(k + 1, cfg.log_step)
            .min(kend)
            .min(k + cfg.max_batch);
        if let Some(h) = ctl.halt_at.filter(|&h| h > k) {
            next = next.min(h);
        }
        if let Some(f) = ctl.next_fault_after(k) {
            next = next.min(f);
        }
        let sw = Stopwatch::start();
        let mut step = be.ll_step(s, next - k).and_then(|()| be.sync());
        timings.iterations += sw.secs();
        k = next;
        if step.is_ok() {
            for (target, kind) in ctl.faults_at(k) {
                if target == FaultTarget::Data {
                    if let Err(e) = crate::prp::inject(be, s, kind, true) {
                        step = Err(e);
                        break;
                    }
                }
            }
        }
        if ctl.halt_at == Some(k) {
            break ProcessState::Halted { k };
        }
        let stop = ctl.stop_requested();
        let mut failed: Option<Option<Residue>> = None;
        let mut data = None;
        match step {
            Err(BackendError::Fault(_)) => {
                ev.backend_faults += 1;
                failed = Some(None);
            }
            Err(e) => return Err(e.into()),
            Ok(()) if !(k % cfg.log_step == 0 || k == kend || stop) => continue,
            Ok(()) => {
                let sw = Stopwatch::start();
                match be.export(s) {
                    Err(BackendError::Fault(_)) => {
                        ev.backend_faults += 1;
                        failed = Some(None);
                    }
                    Err(e) => return Err(e.into()),
                    Ok(v) => {
                        if v.is_zero() && k < kend {
                            failed = Some(Some(v.clone()));
                        }
                        data = Some(v);
                    }
                }
                let do_jacobi = k == kend
                    || recheck.is_some_and(|r| k >= r)
                    || cfg.jacobi == JacobiCadence::EveryLog
                    || last_check.is_none_or(|(t, cost)| t.elapsed().as_secs_f64() >= 100.0 * cost);
                if let (Some(v), true, None) = (&data, do_jacobi, &failed) {
                    let t0 = Instant::now();
                    let outcome = jacobi_check(v);
                    last_check = Some((Instant::now(), t0.elapsed().as_secs_f64()));
                    match outcome {
                        JacobiOutcome::Pass => {
                            ev.jacobi_passed += 1;
                            recheck = None;
                            n_seq = 0;
                            verified = (k, v.clone());
                        }
                        JacobiOutcome::Inconclusive => {
                            ev.jacobi_inconclusive += 1;
                            recheck = None;
                        }
                        JacobiOutcome::Fail => {
                            ev.jacobi_failed += 1;
                            failed = Some(Some(v.clone()));
                        }
                    }
                }
                timings.checks += sw.secs();
            }
        }
        if let Some(res) = failed {
            n_seq += 1;
            if n_seq > 2 {
                return Err(ProtocolError::TooManyErrors {
                    k,
                    errors: (ev.jacobi_failed + ev.backend_faults) as u32,
                });
            }
            if let Some(r) = &res {
                if last_failed.as_ref() == Some(&(k, r.clone())) {
                    return Err(ProtocolError::ConsistentError { k });
                }
                last_failed = Some((k, r.clone()));
            }
            recheck = Some(k);
            if stop {
                break ProcessState::Incomplete {
                    k: verified.0,
                    reason: "stop requested after a failed check".into(),
                    checkpoint: cfg.checkpoint.clone(),
                };
            }
            ev.rollbacks += 1;
            be.import(s, &verified.1)?;
            k = verified.0;
            continue;
        }
        let v = data.expect("a log step without failure has data");
        if k == kend {
            let prime = v.is_zero();
            be.free(s)?;
            ev.notes.push(
                "the Jacobi check catches about half of all errors; it is not a Gerbicz check"
                    .into(),
            );
            ev.notes.push(
                "a positive local result is not a discovery claim; compare with known results"
                    .into(),
            );
            return Ok(RunReport {
                process: ProcessState::Completed,
                result: Some(MathResult {
                    test: "ll".into(),
                    exponent: p,
                    verdict: if prime {
                        Verdict::Prime
                    } else {
                        Verdict::Composite
                    },
                    res64: format!("{:016x}", v.res64()),
                    res2048: v.res2048_hex(),
                    special_case: None,
                }),
                evidence: ev,
                timings,
            });
        }
        if let Some(path) = &cfg.checkpoint {
            let sw = Stopwatch::start();
            let ck = Checkpoint {
                meta: CheckpointMeta {
                    format_version: FORMAT_VERSION,
                    protocol: PROTOCOL.into(),
                    exponent: p,
                    k,
                    state: if verified.0 == k {
                        "verified".into()
                    } else {
                        "provisional".into()
                    },
                    verified_k: verified.0,
                    n_errors: (ev.jacobi_failed + ev.backend_faults) as u32,
                    proof_durable_through: None,
                    backend: be.identity(),
                    config: config_json.clone(),
                    requested: requested.clone(),
                    elapsed_secs: elapsed_before + start.secs(),
                    crate_version: env!("CARGO_PKG_VERSION").into(),
                },
                residues: vec![v.clone(), verified.1.clone()],
            };
            ck.save(path)?;
            timings.checkpoint_io += sw.secs();
        }
        if stop {
            break ProcessState::Incomplete {
                k,
                reason: "stop requested or budget exhausted".into(),
                checkpoint: cfg.checkpoint.clone(),
            };
        }
    };
    be.free(s)?;
    Ok(RunReport {
        process: outcome,
        result: None,
        evidence: ev,
        timings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_matches_euler_criterion_for_small_primes() {
        for n in [3u64, 5, 7, 11, 13, 101, 8191] {
            for a in 0..n.min(60) {
                let e = BigUint::from(a).modpow(&BigUint::from((n - 1) / 2), &BigUint::from(n));
                let want = if e.is_zero() {
                    0
                } else if e.is_one() {
                    1
                } else {
                    -1
                };
                assert_eq!(
                    jacobi(&BigUint::from(a), &BigUint::from(n)),
                    want,
                    "({a}|{n})"
                );
            }
        }
        // Composite modulus: (2 | 15) = (2|3)(2|5) = (-1)(-1) = 1.
        assert_eq!(jacobi(&BigUint::from(2u32), &BigUint::from(15u32)), 1);
    }

    #[test]
    fn ll_residues_have_symbol_minus_one() {
        for p in [5u32, 7, 11, 13, 23] {
            let mut s = Residue::from_u64(p, 4);
            for _ in 1..=(p - 2) {
                s = crate::oracle::ll_steps(&s, 1);
                assert_ne!(jacobi_check(&s), JacobiOutcome::Fail, "p={p}");
            }
        }
    }
}
