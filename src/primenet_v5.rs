// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! A client for the PrimeNet v5 API, the one Prime95 and AutoPrimeNet use:
//! register a computer (`uc`), get assignments (`ga`), report progress
//! (`ap`), unreserve (`au`), and send results (`ar`). Transaction names,
//! parameters, reply format, work type and result type codes follow
//! AutoPrimeNet's `autoprimenet.py`.
//!
//! Requests are not signed with Prime95's key: like AutoPrimeNet for every
//! program other than Prime95, this client sends the fixed placeholder
//! `ss`/`sh` values, so the server knows the requests do not come from
//! Prime95.
//!
//! HTTP goes through the system `curl` program (see `primenet`).
//!
//! NOT TESTED against the PrimeNet server: this code has never contacted it.

use std::collections::BTreeMap;

use serde_json::Value;

pub const V5_URL: &str = "https://v5.mersenne.org/v5server/";
const API_VERSION: &str = "0.95";

/// PrimeNet's error names, by code.
pub fn error_name(code: i64) -> &'static str {
    match code {
        0 => "OK",
        3 => "server busy",
        4 => "invalid version",
        5 => "invalid transaction",
        7 => "invalid parameter",
        9 => "access denied",
        11 => "server database malfunction",
        13 => "server database full or broken",
        21 => "invalid user",
        30 => "CPU not registered",
        31 => "obsolete client",
        32 => "stale CPU info",
        33 => "CPU identity mismatch",
        34 => "CPU configuration mismatch",
        40 => "no assignment",
        43 => "invalid assignment key",
        44 => "invalid assignment type",
        45 => "invalid result type",
        46 => "invalid work type",
        47 => "work no longer needed",
        48 => "illegal residue",
        _ => "unknown error",
    }
}

/// The query string of one request: the protocol prefix, the transaction's
/// parameters in order, and AutoPrimeNet's placeholder security fields
/// when the request names a computer.
pub fn query(params: &[(&str, String)]) -> String {
    let mut q = vec![
        ("px".to_string(), "GIMPS".to_string()),
        ("v".to_string(), API_VERSION.to_string()),
    ];
    q.extend(params.iter().map(|(k, v)| (k.to_string(), v.clone())));
    if params.iter().any(|(k, _)| *k == "g") {
        q.push(("ss".into(), "19191919".into()));
        q.push(("sh".into(), "ABCDABCDABCDABCDABCDABCDABCDABCD".into()));
    }
    q.iter()
        .map(|(k, v)| format!("{k}={}", crate::primenet::urlencode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Parse a reply: `key=value` lines up to `==END==`. Err when the reply is
/// malformed or `pnErrorResult` is not 0, with the server's detail.
pub fn parse_reply(text: &str) -> Result<BTreeMap<String, String>, String> {
    let mut m = BTreeMap::new();
    let mut ended = false;
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line == "==END==" {
            ended = true;
            break;
        }
        let (k, v) = line.split_once('=').unwrap_or((line, ""));
        m.insert(k.to_string(), v.to_string());
    }
    if !ended {
        return Err("the reply has no ==END== line".into());
    }
    let code: i64 = m
        .get("pnErrorResult")
        .and_then(|c| c.trim().parse().ok())
        .ok_or("the reply has no pnErrorResult")?;
    if code != 0 {
        return Err(format!(
            "PrimeNet error {code} ({}): {}",
            error_name(code),
            m.get("pnErrorDetail").map_or("", String::as_str)
        ));
    }
    Ok(m)
}

/// Send one request. Only `V5_URL` is ever contacted.
pub fn send(params: &[(&str, String)]) -> Result<BTreeMap<String, String>, String> {
    let url = format!("{V5_URL}?{}", query(params));
    let text = crate::primenet::curl(&["--url".into(), url])?;
    parse_reply(&text)
}

/// Download the CERT start value of assignment `aid` to `dest`, as
/// AutoPrimeNet does: the reply is 32 hex digits of MD5 followed by the
/// value, and the value is kept only when its MD5 matches.
pub fn download_cert(aid: &str, dest: &std::path::Path) -> Result<(), String> {
    if aid.len() != 32 || !aid.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("bad assignment ID {aid:?}"));
    }
    let tmp = dest.with_extension("cert.part");
    crate::primenet::curl(&[
        "--output".into(),
        tmp.display().to_string(),
        "--url".into(),
        format!("{}proof_get_data/?aid={aid}", crate::primenet::BASE_URL),
    ])?;
    let bytes = std::fs::read(&tmp).map_err(|e| format!("{}: {e}", tmp.display()));
    let _ = std::fs::remove_file(&tmp);
    let bytes = bytes?;
    if bytes.len() <= 32 {
        return Err("the CERT start value reply is too short".into());
    }
    let (sum, value) = bytes.split_at(32);
    let want = String::from_utf8_lossy(sum).to_ascii_lowercase();
    let got: String = crate::worktodo::md5(value)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if want != got {
        return Err(format!(
            "MD5 of the start value is {got}, the server said {want}"
        ));
    }
    crate::checkpoint::write_atomic(dest, value, false)
        .map_err(|e| format!("{}: {e}", dest.display()))
}

/// A new computer GUID: 32 lowercase hex digits from the system's random
/// source.
pub fn new_guid() -> Result<String, String> {
    use std::io::Read;
    let mut b = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| format!("/dev/urandom: {e}"))?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

/// What `uc` reports about the computer. The server refuses LL work when
/// L2 is 256 KiB or less, as AutoPrimeNet notes.
#[derive(Debug, Clone)]
pub struct Machine {
    pub guid: String,
    pub hardware_guid: String,
    pub cpu_brand: String,
    pub cpu_features: String,
    pub l1_kib: u32,
    pub l2_kib: u32,
    pub cores: u32,
    pub threads_per_core: u32,
    pub memory_mib: u64,
    pub mhz: u32,
    pub hours_per_day: u32,
    pub user: Option<String>,
    pub computer_name: Option<String>,
}

/// Parameters of `uc` (update computer).
pub fn register_params(m: &Machine) -> Vec<(&'static str, String)> {
    let mut p = vec![
        ("t", "uc".to_string()),
        ("g", m.guid.clone()),
        ("hg", m.hardware_guid.clone()),
        ("wg", String::new()),
        (
            "a",
            format!(
                "{},mersenne-forge,v{}",
                std::env::consts::OS,
                env!("CARGO_PKG_VERSION")
            ),
        ),
        ("c", m.cpu_brand.clone()),
        ("f", m.cpu_features.clone()),
        ("L1", m.l1_kib.to_string()),
        ("L2", m.l2_kib.to_string()),
        ("np", m.cores.to_string()),
        ("hp", m.threads_per_core.to_string()),
        ("m", m.memory_mib.to_string()),
        ("s", m.mhz.to_string()),
        ("h", m.hours_per_day.to_string()),
        ("r", "1000".to_string()),
    ];
    if let Some(u) = &m.user {
        p.push(("u", u.clone()));
    }
    if let Some(c) = &m.computer_name {
        p.push(("cn", c.chars().take(20).collect()));
    }
    p
}

/// Parameters of `po` (program options) setting worker `worker`'s work
/// preference (see `primenet::worktype_code`).
pub fn options_params(guid: &str, worker: u32, preference: u32) -> Vec<(&'static str, String)> {
    vec![
        ("t", "po".to_string()),
        ("g", guid.to_string()),
        ("c", worker.to_string()),
        ("w", preference.to_string()),
    ]
}

/// Parameters of `ga` (get assignment) for worker `worker`; `cert` asks for
/// CERT work only.
pub fn assignment_params(guid: &str, worker: u32, cert: bool) -> Vec<(&'static str, String)> {
    let mut p = vec![
        ("t", "ga".to_string()),
        ("g", guid.to_string()),
        ("c", worker.to_string()),
    ];
    if cert {
        p.push(("cert", "1".to_string()));
    }
    p
}

/// The worktodo line for a `ga` reply, in the form AutoPrimeNet writes.
pub fn assignment_line(r: &BTreeMap<String, String>) -> Result<String, String> {
    let get = |k: &str| {
        r.get(k)
            .cloned()
            .ok_or_else(|| format!("the assignment reply has no {k}"))
    };
    let w: u32 = get("w")?.parse().map_err(|_| "bad work type".to_string())?;
    let aid = get("k")?;
    if aid.len() != 32 || !aid.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("bad assignment key {aid:?}"));
    }
    let n = get("n")?;
    let kbnc = || -> Result<String, String> {
        Ok(format!("{},{},{n},{}", get("A")?, get("b")?, get("c")?))
    };
    let kf = r.get("kf").map(|f| format!(",\"{f}\"")).unwrap_or_default();
    Ok(match w {
        100 | 101 => format!(
            "{}={aid},{n},{},{}",
            if w == 100 { "Test" } else { "DoubleCheck" },
            get("sf")?,
            get("p1")?
        ),
        150 => {
            let mut s = format!(
                "{}={aid},{}",
                if r.contains_key("dc") { "PRPDC" } else { "PRP" },
                kbnc()?
            );
            if let (Some(sf), Some(saved)) = (r.get("sf"), r.get("saved")) {
                s += &format!(",{sf},{saved}");
                if let (Some(base), Some(rt)) = (r.get("base"), r.get("rt")) {
                    s += &format!(",{base},{rt}");
                }
            }
            s + &kf
        }
        2 => format!("Factor={aid},{n},{},{}{kf}", get("sf")?, get("ef")?),
        3 => format!(
            "Pminus1={aid},{},{},{}{kf}",
            kbnc()?,
            get("B1")?,
            get("B2")?
        ),
        4 => format!(
            "Pfactor={aid},{},{},{}{kf}",
            kbnc()?,
            get("sf")?,
            get("saved")?
        ),
        5 => format!(
            "ECM2={aid},{},{},{},{}{kf}",
            kbnc()?,
            get("B1")?,
            get("B2")?,
            get("CR")?
        ),
        6 => format!(
            "Pplus1={aid},{},{},{},{}{kf}",
            kbnc()?,
            get("B1")?,
            get("B2")?,
            get("nr")?
        ),
        200 => format!("Cert={aid},{},{}", kbnc()?, get("ns")?),
        other => return Err(format!("unknown work type {other}")),
    })
}

/// Parameters of `ap` (assignment progress): `fraction` done (0 to 1),
/// expected seconds to completion, and seconds to the next check-in.
pub fn progress_params(
    guid: &str,
    aid: &str,
    worker: u32,
    fraction: f64,
    eta_secs: u64,
    checkin_secs: u64,
) -> Vec<(&'static str, String)> {
    vec![
        ("t", "ap".to_string()),
        ("g", guid.to_string()),
        ("k", aid.to_string()),
        ("p", format!("{:.4}", fraction.clamp(0.0, 1.0) * 100.0)),
        ("d", checkin_secs.to_string()),
        ("e", eta_secs.to_string()),
        ("c", worker.to_string()),
    ]
}

/// Parameters of `au` (unreserve an assignment).
pub fn unreserve_params(guid: &str, aid: &str) -> Vec<(&'static str, String)> {
    vec![
        ("t", "au".to_string()),
        ("g", guid.to_string()),
        ("k", aid.to_string()),
    ]
}

/// Parameters of `ar` (assignment result) for one JSON result line this
/// program wrote, mapped as AutoPrimeNet maps Prime95's lines. The whole
/// line goes in `m`. Err for a line without an assignment ID or of a work
/// type PrimeNet does not take here.
pub fn result_params(guid: &str, line: &str) -> Result<Vec<(&'static str, String)>, String> {
    let v: Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    let s = |k: &str| match &v[k] {
        Value::String(x) => Some(x.clone()),
        Value::Number(x) => Some(x.to_string()),
        Value::Bool(x) => Some(x.to_string()),
        _ => None,
    };
    let aid = s("aid").ok_or("the result has no assignment ID")?;
    let n = s("exponent").ok_or("the result has no exponent")?;
    let wt = s("worktype").ok_or("the result has no worktype")?;
    let found = matches!(s("status").as_deref(), Some("F"));
    let factors = || {
        v["factors"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|f| f.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default()
    };
    let mut p: Vec<(&'static str, String)> = vec![
        ("t", "ar".to_string()),
        ("g", guid.to_string()),
        ("k", aid),
        ("m", line.to_string()),
    ];
    let kbc = [
        ("A", "1".to_string()),
        ("b", "2".to_string()),
        ("c", "-1".to_string()),
    ];
    let b1 = || s("b1").or_else(|| s("B1")).unwrap_or_default();
    let b2 = || s("b2").or_else(|| s("B2"));
    let r: u32 = match wt.as_str() {
        "LL" => {
            let prime = s("status").as_deref() == Some("P");
            p.push(("d", "1".into()));
            if !prime {
                p.push(("rd", format!("{:0>16}", s("res64").unwrap_or_default())));
            }
            p.push(("sc", s("shift-count").unwrap_or_else(|| "0".into())));
            p.push(("ec", s("error-code").unwrap_or_else(|| "00000000".into())));
            if prime {
                101
            } else {
                100
            }
        }
        w if w.starts_with("PRP-") => {
            let prime = s("status").as_deref() == Some("P");
            p.push(("d", "1".into()));
            p.extend(kbc.clone());
            if !prime {
                p.push(("rd", format!("{:0>16}", s("res64").unwrap_or_default())));
                p.push(("rt", s("residue-type").unwrap_or_else(|| "1".into())));
            }
            p.push(("ec", s("error-code").unwrap_or_else(|| "00000000".into())));
            if let Some(kf) = v["known-factors"].as_array() {
                p.push(("nkf", kf.len().to_string()));
            }
            p.push(("base", w[4..].to_string()));
            if let Some(sc) = s("shift-count") {
                p.push(("sc", sc));
            }
            p.push(("gbz", "1".into()));
            if let Some(power) = v["proof"]["power"].as_u64().filter(|&x| x > 0) {
                p.push(("pp", power.to_string()));
                p.push((
                    "ph",
                    v["proof"]["md5"].as_str().unwrap_or_default().to_string(),
                ));
            }
            if prime {
                151
            } else {
                150
            }
        }
        "TF" => {
            p.push(("d", "1".into()));
            p.push(("sf", s("bitlo").unwrap_or_default()));
            if v["rangecomplete"].as_bool() == Some(true) {
                p.push(("ef", s("bithi").unwrap_or_default()));
            }
            if found {
                p.push(("f", factors()));
                1
            } else {
                4
            }
        }
        "PM1" | "P-1" => {
            p.push(("d", "1".into()));
            p.extend(kbc.clone());
            let b1 = b1();
            p.push(("B2", b2().unwrap_or_else(|| b1.clone())));
            p.push(("B1", b1));
            if found {
                p.push(("f", factors()));
                2
            } else {
                5
            }
        }
        "P+1" => {
            p.push(("d", "1".into()));
            p.extend(kbc.clone());
            let start = s("start").unwrap_or_default();
            let (num, den) = start
                .split_once('/')
                .ok_or("the P+1 result has no start value")?;
            p.push(("pp1n", num.to_string()));
            p.push(("pp1d", den.to_string()));
            let b1 = b1();
            p.push(("B2", b2().unwrap_or_else(|| b1.clone())));
            p.push(("B1", b1));
            if found {
                p.push(("f", factors()));
                7
            } else {
                8
            }
        }
        "ECM" => {
            p.push(("d", "1".into()));
            p.extend(kbc.clone());
            p.push(("CR", s("curves").unwrap_or_default()));
            p.push(("B1", b1()));
            if let Some(b2) = b2() {
                p.push(("B2", b2));
            }
            if found {
                p.push(("f", factors()));
                3
            } else {
                6
            }
        }
        "Cert" => {
            p.push(("d", "1".into()));
            p.extend(kbc.clone());
            p.push(("s3", s("sha3-hash").ok_or("the CERT result has no hash")?));
            p.push(("ec", s("error-code").unwrap_or_else(|| "00000000".into())));
            if let Some(sc) = s("shift-count") {
                p.push(("sc", sc));
            }
            200
        }
        other => {
            return Err(format!(
                "work type {other:?} is not sent through the v5 API"
            ))
        }
    };
    p.push(("r", r.to_string()));
    p.push(("n", n));
    if let Some(f) = s("fft-length") {
        p.push(("fftlen", f));
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replies_parse_and_errors_surface() {
        let ok = parse_reply("pnErrorResult=0\npnErrorDetail=SUCCESS\nw=150\n==END==\n").unwrap();
        assert_eq!(ok["w"], "150");
        let e = parse_reply("pnErrorResult=40\npnErrorDetail=none\n==END==").unwrap_err();
        assert!(e.contains("no assignment"), "{e}");
        assert!(parse_reply("pnErrorResult=0\n").is_err());
    }

    #[test]
    fn queries_carry_the_placeholder_security_fields() {
        let q = query(&unreserve_params("0123", "ABCD"));
        assert_eq!(
            q,
            "px=GIMPS&v=0.95&t=au&g=0123&k=ABCD&ss=19191919&sh=ABCDABCDABCDABCDABCDABCDABCDABCD"
        );
        assert_eq!(new_guid().unwrap().len(), 32);
    }

    #[test]
    fn assignment_replies_become_lines_this_program_parses() {
        let aid = "0123456789ABCDEF0123456789ABCDEF";
        let reply = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .chain([("k".to_string(), aid.to_string())])
                .collect()
        };
        let l = assignment_line(&reply(&[
            ("w", "150"),
            ("n", "4423"),
            ("A", "1"),
            ("b", "2"),
            ("c", "-1"),
            ("sf", "70"),
            ("saved", "0"),
            ("base", "3"),
            ("rt", "1"),
        ]))
        .unwrap();
        assert_eq!(l, format!("PRP={aid},1,2,4423,-1,70,0,3,1"));
        assert!(crate::worktodo::parse_line(&l).unwrap().is_some());
        let l = assignment_line(&reply(&[
            ("w", "200"),
            ("n", "4423"),
            ("A", "1"),
            ("b", "2"),
            ("c", "-1"),
            ("ns", "500"),
        ]))
        .unwrap();
        assert!(crate::cert::parse_cert_line(&l).unwrap().is_some());
        let l = assignment_line(&reply(&[
            ("w", "2"),
            ("n", "4423"),
            ("sf", "60"),
            ("ef", "61"),
        ]))
        .unwrap();
        assert!(crate::factor::parse_factor_line(&l).unwrap().is_some());
    }

    #[test]
    fn results_map_to_v5_fields() {
        let line = r#"{"status":"C","exponent":4423,"worktype":"PRP-3","res64":"abc","residue-type":1,"errors":{"gerbicz":0},"aid":"0123456789ABCDEF0123456789ABCDEF"}"#;
        let p = result_params("g", line).unwrap();
        let get = |k: &str| p.iter().find(|(x, _)| *x == k).map(|(_, v)| v.as_str());
        assert_eq!(get("r"), Some("150"));
        assert_eq!(get("rd"), Some("0000000000000abc"));
        assert_eq!(get("base"), Some("3"));
        assert_eq!(get("n"), Some("4423"));
        let tf = r#"{"status":"F","exponent":67,"worktype":"TF","bitlo":27,"bithi":28,"rangecomplete":true,"factors":["193707721"],"aid":"0123456789ABCDEF0123456789ABCDEF"}"#;
        let p = result_params("g", tf).unwrap();
        let get = |k: &str| p.iter().find(|(x, _)| *x == k).map(|(_, v)| v.as_str());
        assert_eq!(
            (get("r"), get("f"), get("ef")),
            (Some("1"), Some("193707721"), Some("28"))
        );
        assert!(result_params("g", r#"{"worktype":"TF","exponent":67}"#).is_err());
    }
}
