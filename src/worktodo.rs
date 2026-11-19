// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Assignment lines in the worktodo format, result lines in the JSON form
//! upstream PRPLL writes, and checks of reported factors.
//!
//! The parsing rules and result fields follow the pinned upstream source
//! (`Worktodo.cpp`, `Task.cpp`, `Proof.cpp`), checked against fixtures in
//! `tests/worktodo.rs`. Acceptance of these lines by a PrimeNet server or
//! AutoPrimeNet has not been tested; nothing here contacts a server.

use num_bigint::BigUint;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

use crate::run::{MathResult, RunReport, Verdict};

/// What an assignment asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkKind {
    Prp,
    Ll,
}

/// One runnable assignment line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment {
    pub kind: WorkKind,
    pub exponent: u32,
    /// Assignment ID (32 hex digits), when the line has one.
    pub aid: Option<String>,
    /// The line as written, without its line ending.
    pub line: String,
}

/// Parse one worktodo line. `Ok(None)` for blank lines and `#` comments;
/// `Err` for anything this program cannot run, which stays in the file.
///
/// Accepted, as upstream: `PRP=` and `PRPDC=` with `[AID,]1,2,p,-1,...` or a
/// bare `[AID,]p`; `Test=` and `DoubleCheck=` with `[AID,]p,...`. An empty
/// or `N/A` first field is skipped. Other forms (k*2^n+c, P-1, CERT) and
/// composite exponents are refused rather than run as something else.
pub fn parse_line(raw: &str) -> Result<Option<Assignment>, String> {
    let line = raw.trim_end_matches(['\r', '\n']);
    if line.trim().is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let (kind, rest) = line
        .split_once('=')
        .ok_or_else(|| format!("not an assignment: {line:?}"))?;
    let kind = match kind {
        "PRP" | "PRPDC" => WorkKind::Prp,
        "Test" | "DoubleCheck" => WorkKind::Ll,
        other => return Err(format!("unsupported work type {other:?}: {line:?}")),
    };
    let mut parts: Vec<&str> = rest.split(',').collect();
    if parts.first().is_some_and(|f| f.is_empty() || *f == "N/A") {
        parts.remove(0);
    }
    let mut aid = None;
    if parts
        .first()
        .is_some_and(|f| f.len() == 32 && f.chars().all(|c| c.is_ascii_hexdigit()))
    {
        aid = Some(parts.remove(0).to_string());
    }
    let mersenne = parts.len() >= 4 && parts[0] == "1" && parts[1] == "2" && parts[3] == "-1";
    let field = match kind {
        WorkKind::Prp if mersenne => parts[2],
        WorkKind::Prp if parts.len() == 1 => parts[0],
        WorkKind::Ll if !parts.is_empty() => parts[0],
        _ => return Err(format!("not a Mersenne PRP line: {line:?}")),
    };
    let p: u32 = field
        .parse()
        .map_err(|_| format!("bad exponent {field:?}: {line:?}"))?;
    crate::exponent::validate(p).map_err(|e| format!("{e}: {line:?}"))?;
    Ok(Some(Assignment {
        kind,
        exponent: p,
        aid,
        line: line.to_string(),
    }))
}

/// MD5 (RFC 1321), used only because the result format names the proof file
/// by its MD5. Not used for integrity anywhere in this program.
pub fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32)
        .collect();
    let mut st: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_le_bytes());
    for chunk in msg.as_chunks::<64>().0 {
        let m: Vec<u32> = chunk
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let [mut a, mut b, mut c, mut d] = st;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let t = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(k[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = t;
        }
        for (s, v) in st.iter_mut().zip([a, b, c, d]) {
            *s = s.wrapping_add(v);
        }
    }
    let mut out = [0u8; 16];
    for (i, w) in st.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// Lowercase hex MD5 of a file, as upstream writes it.
pub fn file_md5(path: &std::path::Path) -> std::io::Result<String> {
    let b = std::fs::read(path)?;
    Ok(md5(&b).iter().map(|x| format!("{x:02x}")).collect())
}

/// UTC "YYYY-MM-DD HH:MM:SS" for seconds since 1970 (proleptic Gregorian).
pub fn utc_timestamp(secs: u64) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

/// The proof part of a result: power and the MD5 of the proof file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofInfo {
    pub power: u32,
    pub md5: String,
}

/// A result line in the JSON form upstream PRPLL appends to its results
/// file, with this program's name. Fields upstream writes that this program
/// has no counterpart for (`fft-type`, `fft-length`, `user`, `uid`) are
/// left out rather than invented. LL lines keep upstream's `shift-count` 0
/// and `error-code` "00000000".
pub fn result_line(
    a: &Assignment,
    r: &MathResult,
    report: &RunReport,
    proof: Option<&ProofInfo>,
    backend: &str,
    timestamp: &str,
) -> serde_json::Value {
    let prime = matches!(r.verdict, Verdict::Prime | Verdict::ProbablePrime);
    let ev = &report.evidence;
    let mut v = serde_json::Map::new();
    let mut put = |k: &str, x: serde_json::Value| {
        v.insert(k.into(), x);
    };
    put("status", (if prime { "P" } else { "C" }).into());
    put("exponent", a.exponent.into());
    match a.kind {
        WorkKind::Prp => {
            put("worktype", "PRP-3".into());
            put("res64", r.res64.clone().into());
            put("res2048", r.res2048.clone().into());
            put("residue-type", 1.into());
            put("errors", serde_json::json!({"gerbicz": ev.gerbicz_failed}));
            if let Some(pr) = proof {
                // Upstream writes version 1 here for its version 2 files.
                put(
                    "proof",
                    serde_json::json!({"version": 1, "power": pr.power, "hashsize": 64, "md5": pr.md5}),
                );
            }
        }
        WorkKind::Ll => {
            put("worktype", "LL".into());
            put("res64", r.res64.clone().into());
            put("shift-count", 0.into());
            put("error-code", "00000000".into());
            if ev.jacobi_failed > 0 {
                put("errors", serde_json::json!({"jacobi": ev.jacobi_failed}));
            }
        }
    }
    put(
        "program",
        serde_json::json!({
            "name": "mersenne-forge",
            "version": env!("CARGO_PKG_VERSION"),
            "port": backend,
            "os": {"os": std::env::consts::OS, "architecture": std::env::consts::ARCH},
        }),
    );
    if let Some(aid) = &a.aid {
        put("aid", aid.clone().into());
    }
    put("timestamp", timestamp.into());
    serde_json::Value::Object(v)
}

/// Free bytes on the file system holding `dir`, from `df -Pk`, or None when
/// that cannot be determined (the caller then says so instead of guessing).
pub fn free_bytes(dir: &std::path::Path) -> Option<u64> {
    let o = std::process::Command::new("df")
        .arg("-Pk")
        .arg(dir)
        .output()
        .ok()?;
    if !o.status.success() {
        return None;
    }
    let text = String::from_utf8(o.stdout).ok()?;
    let kib: u64 = text
        .lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse()
        .ok()?;
    kib.checked_mul(1024)
}

/// Disk bytes a PRP proof at `power` needs at its peak: the proof points
/// plus the proof file (power + 1 residues and a short header).
pub fn proof_disk_need(p: u32, power: u32) -> u64 {
    let residue = crate::residue::proof_byte_len(p) as u64;
    crate::proof::disk_usage_bytes(p, power) + residue * (u64::from(power) + 1) + 4096
}

/// Why a reported factor was accepted or refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FactorCheck {
    pub exponent: u32,
    pub factor: String,
    /// q divides 2^p - 1, by direct modular exponentiation.
    pub divides: bool,
    /// q = 2kp + 1 and q = +-1 mod 8. Every divisor of 2^p - 1 has this
    /// form, so a q that divides but fails it means an arithmetic error.
    pub form_ok: bool,
    /// q is neither 1 nor 2^p - 1 itself.
    pub proper: bool,
}

impl FactorCheck {
    pub fn accepted(&self) -> bool {
        self.divides && self.proper
    }
}

/// Check a reported factor of 2^p - 1 independently of whoever found it. A
/// factor is accepted when it divides 2^p - 1 and is proper; it need not be
/// prime (a composite factor still shows M_p composite).
pub fn check_factor(p: u32, factor: &str) -> Result<FactorCheck, String> {
    crate::exponent::validate(p).map_err(|e| e.to_string())?;
    let q = BigUint::parse_bytes(factor.trim().as_bytes(), 10)
        .ok_or_else(|| format!("not a decimal number: {factor:?}"))?;
    if q.is_zero() {
        return Err("the factor is zero".into());
    }
    let divides = BigUint::from(2u32).modpow(&BigUint::from(p), &q) == BigUint::one() % &q;
    let two_p = BigUint::from(2u64 * u64::from(p));
    let r8 = (&q % 8u32).to_u32_digits().first().copied().unwrap_or(0);
    let form_ok = (&q % &two_p) == BigUint::one() && (r8 == 1 || r8 == 7);
    Ok(FactorCheck {
        exponent: p,
        factor: q.to_string(),
        divides,
        form_ok,
        // q = 2^p - 1 exactly when it has p bits, all set.
        proper: !q.is_one() && !(q.bits() == u64::from(p) && q.count_ones() == u64::from(p)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn md5_matches_rfc_1321_vectors() {
        for (input, want) in [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ] {
            assert_eq!(hex(&md5(input.as_bytes())), want, "{input:?}");
        }
    }

    #[test]
    fn utc_timestamps() {
        assert_eq!(utc_timestamp(0), "1970-01-01 00:00:00");
        assert_eq!(utc_timestamp(951_782_400), "2000-02-29 00:00:00");
        assert_eq!(utc_timestamp(1_795_478_399), "2026-11-23 23:59:59");
    }

    #[test]
    fn disk_preflight_reads_free_space() {
        let d = tempfile::tempdir().unwrap();
        assert!(free_bytes(d.path()).unwrap() > 0);
        assert!(proof_disk_need(1279, 2) > crate::proof::disk_usage_bytes(1279, 2));
    }

    #[test]
    fn factors_are_checked_directly() {
        // M11 = 23 * 89.
        assert!(check_factor(11, "23").unwrap().accepted());
        assert!(check_factor(11, "89").unwrap().form_ok);
        assert!(!check_factor(11, "47").unwrap().accepted());
        assert!(!check_factor(11, "2047").unwrap().accepted(), "not proper");
        assert!(!check_factor(11, "1").unwrap().accepted());
        // A composite factor still divides: 23 * 89 = M11 is excluded, but
        // M29 = 233 * 1103 * 2089 has composite factors.
        assert!(check_factor(29, &(233u64 * 1103).to_string())
            .unwrap()
            .accepted());
        assert!(check_factor(29, "0").is_err());
        assert!(check_factor(29, "x").is_err());
        assert!(check_factor(30, "7").is_err());
    }
}
