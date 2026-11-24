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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    pub kind: WorkKind,
    pub exponent: u32,
    /// Assignment ID (32 hex digits), when the line has one.
    pub aid: Option<String>,
    /// The line as written, without its line ending.
    pub line: String,
    /// PRP: trial factoring depth in bits, when the line gives it.
    #[serde(default)]
    pub tf_bits: Option<f64>,
    /// PRP: primality tests a factor would save. Above 0, the assigner
    /// wants P-1 before the test, as upstream decides from these fields.
    #[serde(default)]
    pub tests_saved: Option<f64>,
    /// PRP: the residue type to report (1, 3, or 5; see `prp_outcome`).
    #[serde(default = "type_one")]
    pub residue_type: u8,
    /// PRP: known factors of 2^p - 1, decimal. With them the test is of
    /// the cofactor (2^p - 1) / (their product), residue type 5.
    #[serde(default)]
    pub known_factors: Vec<String>,
}

fn type_one() -> u8 {
    1
}

impl Assignment {
    /// True when the line asks for P-1 factoring before the PRP test.
    pub fn wants_pm1(&self) -> bool {
        self.kind == WorkKind::Prp && self.tests_saved.is_some_and(|t| t > 0.0)
    }

    /// The result `worktype` this assignment produces.
    pub fn worktype(&self) -> &'static str {
        match self.kind {
            WorkKind::Prp => "PRP-3",
            WorkKind::Ll => "LL",
        }
    }
}

/// Longest assignment line accepted; longer lines are refused, so a stored
/// assignment always fits checkpoint metadata.
pub const MAX_LINE: usize = 1024;

/// True for lines no parser runs: blank lines, comments (`#`, `;`, `//`,
/// `\\`), and `[Worker #n]` section headers. They stay in the file.
pub fn is_comment(line: &str) -> bool {
    let t = line.trim_start();
    t.is_empty() || t.starts_with(['#', ';', '[']) || t.starts_with("//") || t.starts_with("\\\\")
}

/// One worktodo line split the way Prime95's parser reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct Tokens<'a> {
    /// The work type as written (`PRP`, `Factor`, ...).
    pub work: &'a str,
    /// `key=value` pairs of a `B1=..,B2=..;` prefix, as AutoPrimeNet writes.
    pub prefix: Vec<(&'a str, &'a str)>,
    pub aid: Option<String>,
    /// The comma separated fields after the work type and the AID.
    pub fields: Vec<&'a str>,
    /// Decimal numbers from a quoted tail (`,"f1,f2"`). As upstream, any
    /// non-digit separates two numbers.
    pub known_factors: Vec<String>,
}

impl Tokens<'_> {
    /// The work type, compared without regard to case (mfaktc's rule).
    pub fn is(&self, name: &str) -> bool {
        self.work.eq_ignore_ascii_case(name)
    }
}

/// Split a worktodo line into its parts. `Ok(None)` for comments and blank
/// lines. Shared by every line parser in this crate.
pub fn tokenize(raw: &str) -> Result<Option<Tokens<'_>>, String> {
    let line = raw.trim_end_matches(['\r', '\n']);
    if is_comment(line) {
        return Ok(None);
    }
    if line.len() > MAX_LINE {
        return Err(format!("line longer than {MAX_LINE} bytes"));
    }
    let bad = |why: &str| Err(format!("{why}: {line:?}"));
    let (head, known_factors) = match line.split_once('"') {
        None => (line, Vec::new()),
        Some((h, q)) => {
            let Some(inner) = q.trim_end().strip_suffix('"') else {
                return bad("unterminated quoted field");
            };
            if inner.contains('"') {
                return bad("more than one quoted field");
            }
            let Some(h) = h.trim_end().strip_suffix(',') else {
                return bad("a quoted field must follow a comma");
            };
            let kf: Vec<String> = inner
                .split(|c: char| !c.is_ascii_digit())
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect();
            (h, kf)
        }
    };
    let (prefix_text, main) = match head.rsplit_once(';') {
        Some((pre, m)) => (Some(pre), m),
        None => (None, head),
    };
    let mut prefix = Vec::new();
    for kv in prefix_text.into_iter().flat_map(|t| t.split(',')) {
        let Some((k, v)) = kv.split_once('=') else {
            return bad("malformed prefix");
        };
        prefix.push((k.trim(), v.trim()));
    }
    let Some((work, rest)) = main.split_once('=') else {
        return bad("not an assignment");
    };
    if rest.contains('=') {
        return bad("more than one '='");
    }
    let mut fields: Vec<&str> = rest.split(',').map(str::trim).collect();
    if fields
        .first()
        .is_some_and(|f| f.is_empty() || f.eq_ignore_ascii_case("N/A"))
    {
        fields.remove(0);
    }
    let mut aid = None;
    if fields
        .first()
        .is_some_and(|f| f.len() == 32 && f.chars().all(|c| c.is_ascii_hexdigit()))
    {
        aid = Some(fields.remove(0).to_string());
    }
    Ok(Some(Tokens {
        work: work.trim(),
        prefix,
        aid,
        fields,
        known_factors,
    }))
}

/// Parse one worktodo line. `Ok(None)` for blank lines and comments;
/// `Err` for anything this parser cannot run, which stays in the file.
///
/// Accepted, as upstream: `PRP=` and `PRPDC=` with
/// `[AID,]1,2,p,-1[,tf_bits,tests_saved[,base,residue_type]][,"factors"]`
/// or a bare `[AID,]p`; `Test=` and `DoubleCheck=` with
/// `[AID,]1,2,p,-1[,...]` or `[AID,]p[,...]`. An empty or `N/A` first
/// field is skipped. PRP residue types 1, 3, and 5 are accepted; known
/// factors (each must divide 2^p - 1) make the test one of the cofactor,
/// reported as type 5 as Prime95 does. Refused and left in place: other
/// number forms (k*b^n+c), PRP bases other than 3, residue types 2 and 4,
/// known factors with a residue type other than 5, composite exponents,
/// and PRP of M2.
pub fn parse_line(raw: &str) -> Result<Option<Assignment>, String> {
    let Some(t) = tokenize(raw)? else {
        return Ok(None);
    };
    let line = raw.trim_end_matches(['\r', '\n']);
    let kind = if t.is("PRP") || t.is("PRPDC") {
        WorkKind::Prp
    } else if t.is("Test") || t.is("DoubleCheck") {
        WorkKind::Ll
    } else {
        return Err(format!("unsupported work type {:?}: {line:?}", t.work));
    };
    let parts = &t.fields;
    let refuse = |why: &str| Err(format!("{why}: {line:?}"));
    let mersenne = parts.len() >= 4 && parts[0] == "1" && parts[1] == "2" && parts[3] == "-1";
    // A k,b,n,c number line that is not 1,2,p,-1 names another number.
    if parts.len() >= 4 && matches!(parts[3], "-1" | "+1" | "1") && !mersenne {
        return refuse("not a Mersenne number line");
    }
    let field = match kind {
        _ if mersenne => parts[2],
        WorkKind::Prp if parts.len() == 1 => parts[0],
        WorkKind::Ll if !parts.is_empty() && !parts[0].contains('.') => parts[0],
        _ => return refuse("not a Mersenne number line"),
    };
    let (mut tf_bits, mut tests_saved, mut residue_type) = (None, None, 1u8);
    if kind == WorkKind::Prp && mersenne {
        let num = |i: usize| -> Result<Option<f64>, String> {
            parts
                .get(i)
                .map(|t| {
                    t.parse::<f64>()
                        .ok()
                        .filter(|x| x.is_finite() && *x >= 0.0)
                        .ok_or_else(|| format!("bad field {t:?}: {line:?}"))
                })
                .transpose()
        };
        tf_bits = num(4)?;
        tests_saved = num(5)?;
        if let Some(b) = parts.get(6) {
            if *b != "3" {
                return refuse("only PRP base 3 is supported");
            }
        }
        if let Some(t) = parts.get(7) {
            residue_type = match *t {
                "1" => 1,
                "3" => 3,
                "5" => 5,
                _ => return refuse("PRP residue types 2 and 4 are not supported"),
            };
        }
        if parts.len() > 8 {
            return refuse("too many fields");
        }
    }
    if kind == WorkKind::Ll && !t.known_factors.is_empty() {
        return refuse("an LL test has no known factors");
    }
    if kind == WorkKind::Prp && !t.known_factors.is_empty() {
        // Prime95 reports a cofactor test as type 5 when the line names
        // none; types 1 and 3 of a cofactor need powers this program does
        // not compute.
        if parts.len() <= 7 {
            residue_type = 5;
        }
        if residue_type != 5 {
            return refuse("known factors need residue type 5");
        }
    }
    if residue_type == 5 && t.known_factors.is_empty() {
        return refuse("residue type 5 needs known factors");
    }
    let p: u32 = field
        .parse()
        .ok()
        .filter(|_| field.bytes().all(|b| b.is_ascii_digit()))
        .ok_or_else(|| format!("bad exponent {field:?}: {line:?}"))?;
    crate::exponent::validate(p).map_err(|e| format!("{e}: {line:?}"))?;
    if kind == WorkKind::Prp && p < 3 {
        return refuse("base 3 PRP needs p >= 3");
    }
    for f in &t.known_factors {
        let c = check_factor(p, f).map_err(|e| format!("{e}: {line:?}"))?;
        if !c.accepted() {
            return refuse(&format!("{f} is not a proper factor of 2^{p} - 1"));
        }
    }
    Ok(Some(Assignment {
        kind,
        exponent: p,
        aid: t.aid,
        line: line.to_string(),
        tf_bits,
        tests_saved,
        residue_type,
        known_factors: t.known_factors,
    }))
}

/// The reported outcome of a base-3 PRP test, for one residue type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrpOutcome {
    pub probable_prime: bool,
    pub res64: String,
    pub res2048: String,
}

/// Decide a PRP result from x = 3^(2^p) mod 2^p - 1, for the assignment's
/// residue type (primenet.h's definitions, with N = 2^p - 1, a = 3):
///
/// - type 1: a^(N-1) = x / 9, probable prime when 1;
/// - type 3: a^(N+1) = x, probable prime when 9;
/// - type 5 (cofactor C = N / F, F the product of the known factors): the
///   reported residue is a^(N-1) mod N, as Prime95 reports it, and C is a
///   probable prime when a^(N-1) = a^(F-1) mod C (Fermat's test of C to
///   base a, since a^(N-1) = a^(F(C-1)) * a^(F-1)).
///
/// ponytail: type 5 runs on the host: one squaring of a p-bit number per
/// bit of F (BigUint multiplication, reduced modulo 2^p - 1 by shifts) and
/// one division by C, whose quotient is below F. Minutes at wavefront exponents; move the power to
/// a backend if that matters.
pub fn prp_outcome(a: &Assignment, x: &crate::residue::Residue) -> PrpOutcome {
    let p = a.exponent;
    let type1 = x.div3().div3();
    let shown = if a.residue_type == 3 { x } else { &type1 };
    let probable_prime = match a.residue_type {
        3 => *x == crate::residue::Residue::from_u64(p, 9),
        5 => {
            let n = crate::residue::modulus(p);
            let f = a
                .known_factors
                .iter()
                .filter_map(|s| BigUint::parse_bytes(s.as_bytes(), 10))
                .fold(BigUint::one(), |acc, q| acc * q);
            let c = &n / &f;
            if c <= BigUint::one() {
                false
            } else {
                // w = 3^(F-1) mod N by squarings modulo N (cheap reduction)
                // and multiplications by 3; then C | (v - w), since C | N.
                let fold = |x: BigUint| {
                    // x mod 2^p - 1 for x < 2^(2p + 2), by shifts and adds.
                    let mut r = (&x & &n) + (x >> p);
                    while r >= n {
                        r -= &n;
                    }
                    r
                };
                let e = &f - 1u32;
                let mut w = BigUint::one();
                for i in (0..e.bits()).rev() {
                    w = fold(&w * &w);
                    if e.bit(i) {
                        w = fold(w * 3u32);
                    }
                }
                let v = type1.to_biguint();
                let diff = if v >= w { v - w } else { v + &n - w };
                (diff % &c).is_zero()
            }
        }
        _ => type1 == crate::residue::Residue::from_u64(p, 1),
    };
    PrpOutcome {
        probable_prime,
        res64: format!("{:016x}", shown.res64()),
        res2048: shown.res2048_hex(),
    }
}

/// PrimeNet's numeric code for this platform, as upstream writes `port`.
pub fn port_code() -> u32 {
    let wide = cfg!(target_pointer_width = "64");
    match std::env::consts::OS {
        "windows" => {
            if wide {
                4
            } else {
                1
            }
        }
        "linux" => {
            if wide {
                8
            } else {
                2
            }
        }
        "macos" => {
            if wide {
                10
            } else {
                9
            }
        }
        "freebsd" => {
            if wide {
                12
            } else {
                6
            }
        }
        _ => 0,
    }
}

/// MD5 (RFC 1321), used only because the result format names the proof file
/// by its MD5. Not used for integrity anywhere in this program.
pub struct Md5 {
    st: [u32; 4],
    buf: Vec<u8>,
    len: u64,
}

impl Default for Md5 {
    fn default() -> Md5 {
        Md5 {
            st: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buf: Vec::with_capacity(64),
            len: 0,
        }
    }
}

impl Md5 {
    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if !self.buf.is_empty() {
            let take = (64 - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == 64 {
                let block: [u8; 64] = self.buf[..].try_into().expect("64");
                self.block(&block);
                self.buf.clear();
            }
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for b in blocks {
            self.block(b);
        }
        self.buf.extend_from_slice(rest);
    }

    pub fn finish(mut self) -> [u8; 16] {
        let bits = self.len.wrapping_mul(8);
        let mut tail = vec![0x80u8];
        while (self.buf.len() + tail.len()) % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&bits.to_le_bytes());
        let len = self.len;
        self.update(&tail);
        self.len = len;
        debug_assert!(self.buf.is_empty());
        let mut out = [0u8; 16];
        for (i, w) in self.st.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn block(&mut self, chunk: &[u8; 64]) {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
            5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        let m: Vec<u32> = chunk
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let [mut a, mut b, mut c, mut d] = self.st;
        for (i, &s) in S.iter().enumerate() {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let k = ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32;
            let t = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(k)
                    .wrapping_add(m[g])
                    .rotate_left(s),
            );
            a = t;
        }
        for (s, v) in self.st.iter_mut().zip([a, b, c, d]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// MD5 of a byte string.
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut h = Md5::default();
    h.update(data);
    h.finish()
}

/// Lowercase hex MD5 of a file, as upstream writes it, read in 64 KiB pieces.
pub fn file_md5(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Md5::default();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finish().iter().map(|x| format!("{x:02x}")).collect())
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
/// and `error-code` "00000000". `program.port` is PrimeNet's numeric
/// platform code, as upstream writes it; the backend is a separate
/// `program.backend` field. Error counts cover the whole run, earlier
/// sessions included.
pub fn result_line(
    a: &Assignment,
    r: &MathResult,
    report: &RunReport,
    proof: Option<&ProofInfo>,
    backend: &str,
    timestamp: &str,
) -> serde_json::Value {
    // PRP lines report the assignment's residue type, decided from the
    // final residue; without it (p = 2) the type 1 fields stand.
    let outcome = match (a.kind, &r.final_residue) {
        (WorkKind::Prp, Some(x)) => prp_outcome(a, x),
        _ => PrpOutcome {
            probable_prime: matches!(r.verdict, Verdict::Prime | Verdict::ProbablePrime),
            res64: r.res64.clone(),
            res2048: r.res2048.clone(),
        },
    };
    let prime = outcome.probable_prime;
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
            if !a.known_factors.is_empty() {
                put("known-factors", serde_json::json!(a.known_factors));
            }
            put("res64", outcome.res64.clone().into());
            put("res2048", outcome.res2048.clone().into());
            put("residue-type", a.residue_type.into());
            put("errors", serde_json::json!({"gerbicz": ev.errors_total}));
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
            if ev.errors_total > 0 {
                put("errors", serde_json::json!({"jacobi": ev.errors_total}));
            }
        }
    }
    put(
        "program",
        serde_json::json!({
            "name": "mersenne-forge",
            "version": env!("CARGO_PKG_VERSION"),
            "port": port_code(),
            "backend": backend,
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

/// Disk bytes a PRP proof at `power` needs at its peak: 2^power proof point
/// files (their exact encoded size, each rounded up to a 4 KiB allocation
/// unit), the staged proof file (power + 1 residues and a header), and room
/// for one temporary copy of the largest file being replaced.
pub fn proof_disk_need(p: u32, power: u32) -> u64 {
    let block = |n: u64| n.div_ceil(4096) * 4096;
    let point = block(16 + 4 + 8 + 4 * crate::residue::limb_count(p) as u64 + 32);
    let proof = block(200 + crate::residue::proof_byte_len(p) as u64 * (u64::from(power) + 1));
    (1u64 << power) * point + proof + proof.max(point)
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
    fn residue_types_and_cofactors() {
        use crate::residue::{modulus, Residue};
        let x = |p: u32| {
            let n = modulus(p);
            let e = BigUint::one() << p;
            Residue::from_biguint(p, &BigUint::from(3u32).modpow(&e, &n))
        };
        let a = |line: &str| parse_line(line).unwrap().unwrap();
        // M23 = 47 * 178481, and 178481 is prime.
        let o = prp_outcome(&a("PRP=1,2,23,-1,75,0,3,5,\"47\""), &x(23));
        assert!(o.probable_prime);
        // Type 5 reports a^(N-1) mod N unreduced: the type 1 residue.
        let t1 = prp_outcome(&a("PRP=1,2,23,-1,75,0"), &x(23));
        assert!(!t1.probable_prime);
        assert_eq!(o.res64, t1.res64);
        // M29 = 233 * 1103 * 2089: the cofactor 1103 * 2089 is composite.
        assert!(!prp_outcome(&a("PRP=1,2,29,-1,75,0,\"233\""), &x(29)).probable_prime);
        assert!(prp_outcome(&a("PRP=1,2,29,-1,75,0,\"233,1103\""), &x(29)).probable_prime);
        // Type 3 is x itself: 9 for a prime M31.
        let o = prp_outcome(&a("PRP=1,2,31,-1,75,0,3,3"), &x(31));
        assert!(o.probable_prime);
        assert_eq!(o.res64, format!("{:016x}", 9));
        assert!(prp_outcome(&a("PRP=1,2,31,-1,75,0"), &x(31)).probable_prime);
        assert!(!prp_outcome(&a("PRP=1,2,29,-1,75,0,3,3"), &x(29)).probable_prime);
    }

    #[test]
    fn lines_tokenize_as_prime95_reads_them() {
        let t = tokenize("B1=1000,B2=9000;Pminus1=N/A,1,2,67,-1,1000,3000,\"193707721\"\r\n")
            .unwrap()
            .unwrap();
        assert_eq!(t.work, "Pminus1");
        assert_eq!(t.prefix, [("B1", "1000"), ("B2", "9000")]);
        assert_eq!(t.aid, None);
        assert_eq!(t.fields, ["1", "2", "67", "-1", "1000", "3000"]);
        assert_eq!(t.known_factors, ["193707721"]);
        assert!(tokenize("PRP=1,2,23,-1,\"47").is_err());
        assert!(tokenize("PRP=1,2,23,-1\"47\"").is_err());
        assert_eq!(tokenize("[Worker #2]").unwrap(), None);
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
