// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! A self test of a backend and the factoring code against answers known
//! independently, in the spirit of mfaktc's `-st` and Prime95's torture
//! test: LL and PRP tests of small exponents (prime and composite) checked
//! against the host reference in `oracle`, Gerbicz checked squaring chains
//! as CERT runs them, trial factoring of known factors, and P+1.

use serde::Serialize;

use crate::backend::Backend;
use crate::run::{Control, Verdict};

/// One self test case.
#[derive(Debug, Clone, Serialize)]
pub struct Case {
    pub name: String,
    /// None: skipped (the backend does not take this exponent).
    pub ok: Option<bool>,
    pub detail: String,
}

/// Exponents of the quick run: Mersenne primes and composites of both
/// kinds of LL ending. The full run adds larger ones.
const QUICK: &[u32] = &[3, 5, 7, 11, 13, 23, 31, 61, 89, 107, 127, 521, 607];
const FULL: &[u32] = &[1279, 2203, 2281, 3217, 4253, 4423, 4441, 9689, 9941];

/// Run the self test, opening a backend per exponent with `open`. `stop`
/// ends it early (the cases run so far are returned).
pub fn run(
    open: &mut dyn FnMut(u32) -> Result<Box<dyn Backend>, String>,
    full: bool,
    stop: &dyn Fn() -> bool,
) -> Vec<Case> {
    let mut out = Vec::new();
    let exps: Vec<u32> = QUICK
        .iter()
        .chain(if full { FULL } else { &[] })
        .copied()
        .collect();
    for &p in &exps {
        if stop() {
            return out;
        }
        for test in ["ll", "prp", "squarings"] {
            let name = format!("{test} M{p}");
            let mut be = match open(p) {
                Ok(b) => b,
                Err(e) => {
                    out.push(Case {
                        name,
                        ok: None,
                        detail: e,
                    });
                    continue;
                }
            };
            out.push(match one(&mut be, p, test) {
                Ok(()) => Case {
                    name,
                    ok: Some(true),
                    detail: String::new(),
                },
                Err(e) => Case {
                    name,
                    ok: Some(false),
                    detail: e,
                },
            });
        }
    }
    for &(p, q) in crate::factor::KNOWN {
        if stop() {
            return out;
        }
        let bits = 64 - q.leading_zeros();
        let name = format!("tf M{p} 2^{}..2^{bits}", bits - 1);
        let r = crate::factor::trial_factor(p, bits - 1, bits, None, &|| false, &mut |_| {});
        out.push(match r {
            Ok(o) if o.factors.contains(&q.to_string()) => Case {
                name,
                ok: Some(true),
                detail: String::new(),
            },
            Ok(o) => Case {
                name,
                ok: Some(false),
                detail: format!("{q} not found; found {:?}", o.factors),
            },
            Err(e) => Case {
                name,
                ok: Some(false),
                detail: e,
            },
        });
    }
    // P+1 on M29 with B1 = 13 finds 233 (see the factor tests).
    let r = crate::factor::pp1(29, 13, 13, 1, &|| false);
    out.push(Case {
        name: "p+1 M29 B1=13".into(),
        ok: Some(r.as_ref().is_ok_and(|o| o.factor.is_some())),
        detail: format!("{:?}", r.map(|o| o.factor)),
    });
    out
}

fn one(be: &mut Box<dyn Backend>, p: u32, test: &str) -> Result<(), String> {
    let mut ctl = Control::default();
    let e = |x: crate::run::ProtocolError| x.to_string();
    match test {
        "ll" => {
            let want = crate::oracle::ll(p);
            let r = crate::ll::run_ll(
                be,
                p,
                &crate::ll::LlConfig::default(),
                serde_json::json!({"selftest": p}),
                None,
                &mut ctl,
            )
            .map_err(e)?;
            let m = r.result.ok_or("no result")?;
            let prime = m.verdict == Verdict::Prime;
            let res64 = format!("{:016x}", want.terminal.res64());
            if prime != want.positive || m.res64 != res64 {
                return Err(format!(
                    "got {:?} res64 {}, want prime {} res64 {res64}",
                    m.verdict, m.res64, want.positive
                ));
            }
        }
        "prp" => {
            let want = crate::oracle::prp(p);
            let r = crate::prp::run_prp(
                be,
                p,
                &crate::prp::PrpConfig::default(),
                serde_json::json!({"selftest": p}),
                None,
                &mut ctl,
            )
            .map_err(e)?;
            let m = r.result.ok_or("no result")?;
            let pp = matches!(m.verdict, Verdict::ProbablePrime | Verdict::Prime);
            if pp != want.positive || m.final_residue.is_some_and(|x| x != want.terminal) {
                return Err(format!(
                    "got {:?}, want probable prime {}",
                    m.verdict, want.positive
                ));
            }
        }
        _ => {
            // A CERT style chain from a start value other than 3.
            let start = crate::residue::Residue::from_u64(p, 5);
            let n = 3 * u64::from(p) + 17;
            let (x, _) = crate::prp::gerbicz_squarings(be, &start, n, &mut ctl).map_err(e)?;
            let want = crate::oracle::square_n(&start, n);
            if x.as_ref() != Some(&want) {
                return Err("squaring chain differs from the host".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn quick_selftest_passes_on_the_cpu_backend() {
        let mut open = |p: u32| -> Result<Box<dyn crate::backend::Backend>, String> {
            crate::backend::cpu::CpuBackend::new(p)
                .map(|b| Box::new(b) as Box<dyn crate::backend::Backend>)
                .map_err(|e| e.to_string())
        };
        let cases = super::run(&mut open, false, &|| false);
        let bad: Vec<_> = cases.iter().filter(|c| c.ok == Some(false)).collect();
        assert!(bad.is_empty(), "{bad:?}");
        assert!(cases.iter().filter(|c| c.ok == Some(true)).count() > 40);
    }
}
