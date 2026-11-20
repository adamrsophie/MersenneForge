// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! CERT work: certify another tester's PRP proof by squaring the start
//! value the server sends a given number of times and reporting a hash of
//! the result, as Prime95 (`cert.c`) and gpuowl (`Gpu::isCERT`) do.
//!
//! The start value comes in a file `M<p>.cert` holding (p - 1) / 8 + 1
//! little-endian bytes (AutoPrimeNet downloads it before the run). The
//! squarings run on a backend with the Gerbicz check. The hash is SHA3-256
//! of the result in the same byte form, written as Prime95 writes it:
//! uppercase hex, last byte first. NOT TESTED against a PrimeNet server.

use std::path::{Path, PathBuf};

use sha3::{Digest, Sha3_256};

use crate::residue::Residue;

/// A `Cert=` line: `Cert=[AID,]1,2,p,-1,squarings`.
#[derive(Debug, Clone, PartialEq)]
pub struct CertLine {
    pub exponent: u32,
    pub squarings: u64,
    pub aid: Option<String>,
    pub line: String,
}

/// Parse a `Cert=` line. `Ok(None)` for any other line. Only Mersenne
/// numbers are accepted (Prime95 certifies only those too).
pub fn parse_cert_line(raw: &str) -> Result<Option<CertLine>, String> {
    let line = raw.trim_end_matches(['\r', '\n']);
    let head = line.split(['=', ';']).next().unwrap_or("").trim();
    if !head.eq_ignore_ascii_case("Cert") {
        return Ok(None);
    }
    let Some(t) = crate::worktodo::tokenize(raw)? else {
        return Ok(None);
    };
    let bad = || format!("malformed Cert line: {line:?}");
    let f = &t.fields;
    if f.len() != 5 || f[..2] != ["1", "2"] || f[3] != "-1" || !t.known_factors.is_empty() {
        return Err(bad());
    }
    let p: u32 = f[2].parse().map_err(|_| bad())?;
    let squarings: u64 = f[4].parse().map_err(|_| bad())?;
    crate::exponent::validate(p).map_err(|e| format!("{e}: {line:?}"))?;
    // gpuowl refuses chains of 100 squarings or fewer as malformed.
    if squarings <= 100 {
        return Err(bad());
    }
    Ok(Some(CertLine {
        exponent: p,
        squarings,
        aid: t.aid,
        line: line.to_string(),
    }))
}

/// The start value file for M_p in `dir`, named as AutoPrimeNet names it.
pub fn start_path(dir: &Path, p: u32) -> PathBuf {
    dir.join(format!("M{p}.cert"))
}

/// Read a start value: exactly (p - 1) / 8 + 1 bytes, no bits at or above p.
pub fn read_start(path: &Path, p: u32) -> Result<Residue, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Residue::from_proof_bytes(p, &bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// SHA3-256 of `x` in proof byte form, as Prime95 prints it: bytes 31 to 0,
/// uppercase hex.
pub fn hash(x: &Residue) -> String {
    let h = Sha3_256::digest(x.to_proof_bytes());
    h.iter().rev().map(|b| format!("{b:02X}")).collect()
}

/// The JSON result line, with the keys Prime95 writes and AutoPrimeNet
/// reads (`worktype` "Cert", `sha3-hash`, `squarings`).
pub fn result_line(
    c: &CertLine,
    hash: &str,
    errors: u64,
    backend: &str,
    timestamp: &str,
) -> serde_json::Value {
    let mut v = serde_json::json!({
        "exponent": c.exponent,
        "worktype": "Cert",
        "sha3-hash": hash,
        "squarings": c.squarings,
        "shift-count": 0,
        // Prime95 packs several error counters into error-code; this
        // program has no such counters, so it writes zero and reports the
        // Gerbicz check failures (all rolled back) the way its PRP lines do.
        "error-code": "00000000",
        "errors": {"gerbicz": errors},
        "program": {
            "name": "mersenne-forge",
            "version": env!("CARGO_PKG_VERSION"),
            "port": crate::worktodo::port_code(),
            "backend": backend,
            "os": {"os": std::env::consts::OS, "architecture": std::env::consts::ARCH},
        },
        "timestamp": timestamp,
    });
    if let Some(aid) = &c.aid {
        v["aid"] = aid.clone().into();
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    #[test]
    fn cert_lines_parse() {
        let c = parse_cert_line("Cert=B2EE67DC0A514753E488794C9DD6F6BD,1,2,82997591,-1,162105")
            .unwrap()
            .unwrap();
        assert_eq!((c.exponent, c.squarings), (82_997_591, 162_105));
        assert!(c.aid.is_some());
        assert_eq!(parse_cert_line("PRP=1,2,127,-1").unwrap(), None);
        assert!(parse_cert_line("Cert=1,2,127,-1,50").is_err());
        assert!(parse_cert_line("Cert=1,2,127,1,500").is_err());
        assert!(parse_cert_line("Cert=1,2,128,-1,500").is_err());
    }

    #[test]
    fn hash_is_sha3_of_the_bytes_last_byte_first() {
        let x = Residue::from_u64(127, 0x0102);
        let mut bytes = vec![0u8; 16];
        bytes[0] = 2;
        bytes[1] = 1;
        let want: String = Sha3_256::digest(&bytes)
            .iter()
            .rev()
            .map(|b| format!("{b:02X}"))
            .collect();
        assert_eq!(hash(&x), want);
        assert_eq!(hash(&x).len(), 64);
    }

    #[test]
    fn squarings_on_a_backend_match_the_host() {
        let p = 4423;
        let mut be = crate::backend::cpu::CpuBackend::new(p).unwrap();
        let start = Residue::from_u64(p, 123_456_789);
        let n = 2500u64;
        let mut ctl = crate::run::Control::default();
        let (x, ev) = crate::prp::gerbicz_squarings(&mut be, &start, n, &mut ctl).unwrap();
        let m = crate::residue::modulus(p);
        let want = BigUint::from(123_456_789u32).modpow(&(BigUint::from(1u32) << n), &m);
        assert_eq!(x.unwrap().to_biguint(), want);
        assert!(ev.gerbicz_passed > 0);
        let dir = tempfile::tempdir().unwrap();
        let path = start_path(dir.path(), p);
        std::fs::write(&path, start.to_proof_bytes()).unwrap();
        assert_eq!(read_start(&path, p).unwrap(), start);
        std::fs::write(&path, [0u8; 3]).unwrap();
        assert!(read_start(&path, p).is_err());
    }
}
