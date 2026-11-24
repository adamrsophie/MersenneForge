// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Base 3 PRP tests of N = k * b^n + c other than Mersenne numbers, as
//! Prime95 runs them from `PRP=k,b,n,c,...` lines. Fermat numbers are the
//! case k = 1, b = 2, n = 2^m, c = 1 (residue type 2 is then Pepin's test).
//!
//! The chain x_0 = 3^k, x_{i+1} = x_i^b is checked by Gerbicz's product
//! check, which holds for any b: with d the product of every L-th x,
//! d_new = x_0 * d_old^(b^L). The steps after the last full block are
//! computed twice and compared. All arithmetic is on the host in big
//! integers; ponytail: no FFT, so this is for numbers of up to some
//! hundred thousand bits, not wavefront sizes. Known factors (cofactor
//! tests) are refused.

use crate::checkpoint::write_atomic;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Numbers larger than this many bits are left to other programs.
pub const MAX_BITS: u64 = 1 << 20;

/// A `PRP=` or `PRPDC=` line for k * b^n + c.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericLine {
    pub k: u64,
    pub b: u32,
    pub n: u32,
    pub c: i64,
    pub residue_type: u8,
    pub aid: Option<String>,
    pub line: String,
}

impl GenericLine {
    /// k * b^n + c.
    pub fn number(&self) -> BigUint {
        let v = BigInt::from(self.k) * BigInt::from(self.b).pow(self.n) + self.c;
        v.to_biguint().unwrap_or_default()
    }

    /// The number as written, for messages: `k*b^n+c`.
    pub fn name(&self) -> String {
        let c = if self.c < 0 {
            format!("{}", self.c)
        } else {
            format!("+{}", self.c)
        };
        format!("{}*{}^{}{c}", self.k, self.b, self.n)
    }

    /// About how many bits the number has.
    pub fn bits(&self) -> u64 {
        let k = 64 - u64::from(self.k.leading_zeros());
        (f64::from(self.n) * f64::from(self.b).log2()).ceil() as u64 + k
    }
}

/// Read a `PRP=[AID,]k,b,n,c[,tf_bits,tests_saved[,base,residue_type]]`
/// line for a number other than 2^p - 1, or Mlucas' `Fermat,Test=m` (the
/// Pepin test of F_m, run as residue type 2 of 1,2,2^m,1). Ok(None) for
/// any other line (Mersenne lines included, which
/// [`crate::worktodo::parse_line`] reads).
pub fn parse_generic_line(raw: &str) -> Result<Option<GenericLine>, String> {
    let trimmed = raw.trim_end_matches(['\r', '\n']);
    if let Some(m) = trimmed
        .get(..12)
        .filter(|h| h.eq_ignore_ascii_case("Fermat,Test="))
        .map(|_| trimmed[12..].trim())
    {
        let m: u32 = m
            .parse()
            .ok()
            .filter(|&m| (1..=20).contains(&m))
            .ok_or_else(|| format!("Fermat numbers F1 to F20 only: {trimmed:?}"))?;
        return Ok(Some(GenericLine {
            k: 1,
            b: 2,
            n: 1 << m,
            c: 1,
            residue_type: 2,
            aid: None,
            line: trimmed.to_string(),
        }));
    }
    let Some(t) = crate::worktodo::tokenize(raw)? else {
        return Ok(None);
    };
    if !(t.is("PRP") || t.is("PRPDC")) || t.fields.len() < 4 {
        return Ok(None);
    }
    let line = raw.trim_end_matches(['\r', '\n']);
    let f = &t.fields;
    if f[0] == "1" && f[1] == "2" && f[3] == "-1" {
        return Ok(None);
    }
    let bad = |why: &str| format!("{why}: {line:?}");
    let int = |s: &str| s.bytes().all(|b| b.is_ascii_digit()) && !s.is_empty();
    let k: u64 = f[0]
        .parse()
        .ok()
        .filter(|&k| k >= 1 && int(f[0]))
        .ok_or_else(|| bad("bad k"))?;
    let b: u32 = f[1]
        .parse()
        .ok()
        .filter(|&b| b >= 2 && int(f[1]))
        .ok_or_else(|| bad("bad b"))?;
    let n: u32 = f[2]
        .parse()
        .ok()
        .filter(|&n| n >= 1 && int(f[2]))
        .ok_or_else(|| bad("bad n"))?;
    let c: i64 = f[3]
        .trim_start_matches('+')
        .parse()
        .ok()
        .filter(|&c: &i64| c != 0 && c.unsigned_abs() < 1 << 31)
        .ok_or_else(|| bad("bad c"))?;
    for x in f.iter().take(6).skip(4) {
        if !x.parse::<f64>().is_ok_and(|v| v.is_finite() && v >= 0.0) {
            return Err(bad("bad field"));
        }
    }
    if f.get(6).is_some_and(|b| *b != "3") {
        return Err(bad("only PRP base 3 is supported"));
    }
    let residue_type = match f.get(7).copied() {
        None | Some("1") => 1,
        Some("2") => 2,
        Some("3") => 3,
        Some("4") => 4,
        Some("5") => return Err(bad("residue type 5 is for cofactor tests")),
        Some(_) => return Err(bad("PRP residue types are 1 to 5")),
    };
    if f.len() > 8 {
        return Err(bad("too many fields"));
    }
    if !t.known_factors.is_empty() {
        return Err(bad("cofactor tests of k*b^n+c are not supported"));
    }
    let g = GenericLine {
        k,
        b,
        n,
        c,
        residue_type,
        aid: t.aid,
        line: line.to_string(),
    };
    if matches!(residue_type, 2 | 4) && !(b == 2 && c % 2 != 0) {
        return Err(bad("residue types 2 and 4 need b = 2 and an odd c"));
    }
    if g.bits() > MAX_BITS {
        return Err(bad(&format!("above {MAX_BITS} bits")));
    }
    let nn = g.number();
    if nn < BigUint::from(5u32) || (&nn % 3u32).is_zero() || nn.is_even() {
        return Err(bad("base 3 PRP needs an odd N > 3 prime to 3"));
    }
    Ok(Some(g))
}

/// The outcome of one test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericOutcome {
    pub probable_prime: bool,
    /// Low 64 bits of the reported residue, lowercase hex.
    pub res64: String,
    /// Low 2048 bits, hex, most significant first.
    pub res2048: String,
    /// Failed Gerbicz checks and tail mismatches, all recomputed.
    pub errors: u64,
    /// False when stopped; the checkpoint then holds the progress.
    pub complete: bool,
}

#[derive(Serialize, Deserialize)]
struct State {
    number: String,
    i: u64,
    x: String,
    d: String,
    errors: u64,
}

fn hex(v: &BigUint) -> String {
    v.to_str_radix(16)
}

fn unhex(s: &str) -> Result<BigUint, String> {
    BigUint::parse_bytes(s.as_bytes(), 16).ok_or_else(|| "bad checkpoint".to_string())
}

/// 3^e mod N for a signed e (N prime to 3).
fn pow3(e: i64, n: &BigUint) -> BigUint {
    let three = BigUint::from(3u32);
    let v = three.modpow(&BigUint::from(e.unsigned_abs()), n);
    if e >= 0 {
        v
    } else {
        let inv = BigInt::from(v)
            .extended_gcd(&BigInt::from(n.clone()))
            .x
            .mod_floor(&BigInt::from(n.clone()));
        inv.to_biguint().expect("non-negative")
    }
}

/// Run (or resume from `ckpt`) the test of `g`. `stop` is polled at every
/// check. Gerbicz failures roll back to the last verified state; more than
/// 10 in a row is an error.
pub fn prp_generic(
    g: &GenericLine,
    ckpt: Option<&Path>,
    stop: &dyn Fn() -> bool,
) -> Result<GenericOutcome, String> {
    prp_generic_with(g, ckpt, stop, &mut |_, _| {})
}

/// [`prp_generic`] with `corrupt(i, x)` called after every step, for fault
/// injection in tests.
pub fn prp_generic_with(
    g: &GenericLine,
    ckpt: Option<&Path>,
    stop: &dyn Fn() -> bool,
    corrupt: &mut dyn FnMut(u64, &mut BigUint),
) -> Result<GenericOutcome, String> {
    let n = g.number();
    let steps = u64::from(g.n) - u64::from(matches!(g.residue_type, 2 | 4));
    let b = BigUint::from(g.b);
    let step = |x: &BigUint| -> BigUint {
        if g.b == 2 {
            x * x % &n
        } else {
            x.modpow(&b, &n)
        }
    };
    // Block length L, checked every L blocks (about 16 checks per run
    // until L reaches upstream's 1000).
    let l = ((steps as f64).sqrt() / 4.0).clamp(1.0, 1000.0) as u64;
    let x0 = BigUint::from(3u32).modpow(&BigUint::from(g.k), &n);
    let (mut i, mut x, mut d, mut errors) = (0u64, x0.clone(), x0.clone(), 0u64);
    if let Some(path) = ckpt.filter(|p| p.exists()) {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let s: State =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if s.number != g.name() || s.i > steps {
            return Err(format!("{} is for another test", path.display()));
        }
        (i, x, d, errors) = (s.i, unhex(&s.x)?, unhex(&s.d)?, s.errors);
    }
    let save = |i: u64, x: &BigUint, d: &BigUint, errors: u64| -> Result<(), String> {
        let Some(path) = ckpt else { return Ok(()) };
        let s = State {
            number: g.name(),
            i,
            x: hex(x),
            d: hex(d),
            errors,
        };
        write_atomic(
            path,
            serde_json::to_string(&s).expect("serializes").as_bytes(),
            false,
        )
        .map_err(|e| format!("{}: {e}", path.display()))
    };
    let full = steps / l * l;
    let mut streak = 0;
    // Checked part: whole blocks, a check every L blocks or at the end.
    while i < full {
        if stop() {
            save(i, &x, &d, errors)?;
            return Ok(GenericOutcome {
                probable_prime: false,
                res64: String::new(),
                res2048: String::new(),
                errors,
                complete: false,
            });
        }
        let (vi, vx, vd) = (i, x.clone(), d.clone());
        let blocks = ((full - i) / l).min(l);
        let mut d_prev = d.clone();
        for _ in 0..blocks {
            for _ in 0..l {
                x = step(&x);
                i += 1;
                corrupt(i, &mut x);
            }
            d_prev = d.clone();
            d = &d * &x % &n;
        }
        // d = x_0 * d_prev^(b^L)
        let mut t = d_prev;
        for _ in 0..l {
            t = step(&t);
        }
        if &x0 * t % &n == d {
            streak = 0;
            save(i, &x, &d, errors)?;
        } else {
            errors += 1;
            streak += 1;
            if streak > 10 {
                return Err(format!(
                    "{}: Gerbicz check failed {streak} times in a row",
                    g.name()
                ));
            }
            (i, x, d) = (vi, vx, vd);
        }
    }
    // Tail: computed twice.
    loop {
        let (mut a, mut c) = (x.clone(), x.clone());
        for j in i..steps {
            a = step(&a);
            corrupt(j + 1, &mut a);
        }
        for _ in i..steps {
            c = step(&c);
        }
        if a == c {
            x = a;
            break;
        }
        errors += 1;
    }
    // y = 3^(k b^steps). Reported residue r and the prime test, by type.
    let (r, prime) = match g.residue_type {
        1 => {
            let r = &x * pow3(g.c - 1, &n) % &n;
            let p = r.is_one();
            (r, p)
        }
        3 => {
            let p = (&x * pow3(g.c - 1, &n) % &n).is_one();
            (x.clone(), p)
        }
        2 => {
            let r = &x * pow3((g.c - 1) / 2, &n) % &n;
            let p = r.is_one() || r == &n - 1u32;
            (r, p)
        }
        _ => {
            let t = &x * pow3((g.c - 1) / 2, &n) % &n;
            (x.clone(), t.is_one() || t == &n - 1u32)
        }
    };
    let lo = (&r & BigUint::from(u64::MAX)).to_u64().unwrap_or(0);
    let mask = (BigUint::one() << 2048u32) - 1u32;
    let w = format!("{:0>512}", (&r & mask).to_str_radix(16));
    Ok(GenericOutcome {
        probable_prime: prime,
        res64: format!("{lo:016x}"),
        res2048: w,
        errors,
        complete: true,
    })
}

/// The JSON result line, with Prime95's field names for k*b^n+c.
pub fn result_line(g: &GenericLine, o: &GenericOutcome, timestamp: &str) -> serde_json::Value {
    let mut v = serde_json::json!({
        "status": if o.probable_prime { "P" } else { "C" },
        "k": g.k, "b": g.b, "n": g.n, "c": g.c,
        "worktype": "PRP-3",
        "res64": o.res64,
        "res2048": o.res2048,
        "residue-type": g.residue_type,
        "errors": {"gerbicz": o.errors},
        "program": {
            "name": "mersenne-forge",
            "version": env!("CARGO_PKG_VERSION"),
            "port": crate::worktodo::port_code(),
            "backend": "host",
            "os": {"os": std::env::consts::OS, "architecture": std::env::consts::ARCH},
        },
        "timestamp": timestamp,
    });
    if let Some(aid) = &g.aid {
        v["aid"] = aid.clone().into();
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(s: &str) -> GenericLine {
        parse_generic_line(s).unwrap().unwrap()
    }

    fn run(s: &str) -> GenericOutcome {
        prp_generic(&line(s), None, &|| false).unwrap()
    }

    /// Verdicts and type 1 residues against a direct 3^(N-1) mod N.
    #[test]
    fn verdicts_and_residues_match_direct_powers() {
        // F4 = 65537 prime, F5 = 2^32 + 1 composite, 3*2^5+1 = 97 prime,
        // 2*10^3+3 = 2003 prime, 5*3^20-2 composite, 7*2^61-1 composite.
        for (s, prime) in [
            ("PRP=1,2,16,1", Some(true)),
            ("PRP=1,2,32,1", Some(false)),
            ("PRP=3,2,5,1", Some(true)),
            ("PRP=2,10,3,3", Some(true)),
            ("PRP=5,3,20,-2", None),
            ("PRP=7,2,61,-1", None),
            ("PRP=1,2,1024,1", Some(false)),
        ] {
            let g = line(s);
            let o = run(s);
            let n = g.number();
            let want = BigUint::from(3u32).modpow(&(&n - 1u32), &n);
            assert_eq!(o.probable_prime, want.is_one(), "{s}");
            if let Some(prime) = prime {
                assert_eq!(o.probable_prime, prime, "{s}");
            }
            let lo = (&want & BigUint::from(u64::MAX)).to_u64().unwrap();
            assert_eq!(o.res64, format!("{lo:016x}"), "{s}");
            assert!(o.complete && o.errors == 0);
        }
        // Pepin's test: 3^((F-1)/2) = -1 for F4 and type 3 / 4 verdicts.
        for t in ["2", "3", "4"] {
            assert!(
                run(&format!("PRP=1,2,16,1,0,0,3,{t}")).probable_prime,
                "type {t}"
            );
            assert!(
                !run(&format!("PRP=1,2,32,1,0,0,3,{t}")).probable_prime,
                "type {t}"
            );
        }
        let o = run("PRP=1,2,16,1,0,0,3,2");
        assert_eq!(o.res64, format!("{:016x}", 65536u64));
    }

    /// Injected errors are caught and recomputed; a stopped run resumes
    /// from its checkpoint to the same result.
    #[test]
    fn faults_roll_back_and_runs_resume() {
        let g = line("PRP=1,2,4100,1");
        let clean = prp_generic(&g, None, &|| false).unwrap();
        let mut hit = std::collections::HashSet::new();
        let bad = prp_generic_with(&g, None, &|| false, &mut |i, x| {
            if (i == 100 || i == 4098) && hit.insert(i) {
                *x += 1u32;
            }
        })
        .unwrap();
        assert_eq!(bad.res64, clean.res64);
        assert_eq!(bad.errors, 2);
        let dir = tempfile::tempdir().unwrap();
        let ck = dir.path().join("g.state");
        let calls = std::cell::Cell::new(0);
        let first = prp_generic(&g, Some(&ck), &|| {
            calls.set(calls.get() + 1);
            calls.get() > 2
        })
        .unwrap();
        assert!(!first.complete && ck.exists());
        let rest = prp_generic(&g, Some(&ck), &|| false).unwrap();
        assert_eq!(rest.res64, clean.res64);
        let other = line("PRP=1,2,2048,1");
        assert!(prp_generic(&other, Some(&ck), &|| false).is_err());
    }

    #[test]
    fn lines_are_read_or_refused() {
        assert_eq!(parse_generic_line("PRP=1,2,127,-1").unwrap(), None);
        assert_eq!(parse_generic_line("Factor=67,1,40").unwrap(), None);
        let g = line("PRP=0123456789ABCDEF0123456789ABCDEF,3,2,5,+1,60,1,3,1");
        assert_eq!((g.k, g.b, g.n, g.c), (3, 2, 5, 1));
        assert!(g.aid.is_some());
        assert_eq!(g.name(), "3*2^5+1");
        let f = line("Fermat,Test=4");
        assert_eq!((f.n, f.residue_type), (16, 2));
        assert!(run("Fermat,Test=4").probable_prime);
        assert!(run("Fermat,Test=1").probable_prime);
        assert!(!run("Fermat,Test=6").probable_prime);
        assert!(parse_generic_line("Fermat,Test=21").is_err());
        for bad in [
            "PRP=1,2,16,1,0,0,5,1",
            "PRP=1,2,16,1,0,0,3,5",
            "PRP=1,2,16,1,0,0,3,9",
            "PRP=1,3,5,1",
            "PRP=1,10,5,0",
            "PRP=1,2,16,1,\"65537\"",
            "PRP=2,10,3,3,0,0,3,2",
            "PRP=1,2,99999999,1",
        ] {
            assert!(parse_generic_line(bad).is_err(), "{bad}");
        }
        let o = run("PRP=3,2,5,1");
        let v = result_line(&line("PRP=3,2,5,1"), &o, "2026-11-24 00:00:00");
        assert_eq!(v["k"], 3);
        assert_eq!(v["status"], "P");
        assert_eq!(v["worktype"], "PRP-3");
    }
}
