// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! PRP proofs: the point schedule, durable proof residues, generation, the
//! version 2 file format, and verification, following PRPLL's `Proof.cpp`.
//!
//! Schedule for power P: start with {0}; for each of P levels, with
//! span = (p + 1) / 2 for the first and span = (span + 1) / 2 after, add span
//! to each existing point; replace 0 by p; sort. The residue x_k is saved as
//! the PRP run passes each point k.
//!
//! File: the ASCII header `PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=P\nNUMBER=Mp\n`,
//! then B = x_p and the P middles, each as (p - 1) / 8 + 1 little-endian
//! bytes. Hash chain: h_0 = SHA3-256(B bytes), h_(i+1) = SHA3-256(h_i || M_i
//! bytes); the exponent used at level i is the first 64-bit little-endian
//! word of h_(i+1).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha3::{Digest, Sha3_256};

use crate::backend::{Backend, Reg};
use crate::prp::{checked_power, CheckedPowerConfig};
use crate::residue::{proof_byte_len, Residue};
use crate::run::{Control, ProtocolError};

/// Largest proof power upstream accepts.
pub const MAX_POWER: u32 = 13;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofError {
    /// The power is outside 1..=13 or gives colliding points for p.
    InvalidPower {
        p: u32,
        power: u32,
        reason: String,
    },
    /// Not a version 2 PRP proof header.
    BadHeader,
    /// The file names a different exponent than the one requested.
    WrongExponent {
        expected: u32,
        found: u32,
    },
    /// Fewer bytes than the header requires.
    Truncated {
        expected: u64,
        found: u64,
    },
    /// More bytes than the header requires.
    TrailingData {
        expected: u64,
        found: u64,
    },
    /// A residue has bits at or above bit p.
    NoncanonicalResidue {
        index: usize,
    },
    /// A middle residue is zero, which no honest computation produces.
    ZeroResidue {
        index: usize,
    },
    /// The file parsed, but the final check A^(2^span) == B failed.
    Rejected,
    /// A stored proof residue is missing or failed its integrity check.
    Residue {
        k: u64,
        message: String,
    },
    Io(String),
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofError::InvalidPower { p, power, reason } => {
                write!(
                    f,
                    "proof power {power} is invalid for exponent {p}: {reason}"
                )
            }
            ProofError::BadHeader => {
                write!(f, "not a PRP PROOF version 2 file (incompatible header)")
            }
            ProofError::WrongExponent { expected, found } => {
                write!(f, "proof is for M{found}, expected M{expected}")
            }
            ProofError::Truncated { expected, found } => {
                write!(
                    f,
                    "proof truncated: {found} bytes, header requires {expected}"
                )
            }
            ProofError::TrailingData { expected, found } => {
                write!(
                    f,
                    "proof has trailing data: {found} bytes, header requires {expected}"
                )
            }
            ProofError::NoncanonicalResidue { index } => {
                write!(f, "proof residue {index} has bits at or above bit p")
            }
            ProofError::ZeroResidue { index } => write!(f, "proof residue {index} is zero"),
            ProofError::Rejected => write!(f, "proof rejected: the final check does not hold"),
            ProofError::Residue { k, message } => write!(f, "proof residue {k}: {message}"),
            ProofError::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ProofError {}

/// SHA3-256 (standard FIPS 202 padding).
pub fn sha3_256(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

fn first_word(h: &[u8; 32]) -> u64 {
    u64::from_le_bytes(h[..8].try_into().expect("8 bytes"))
}

/// Upstream's default power for p (no disk constraint): 10 + floor(log4(p / 60e6)),
/// at most 13, then lowered until the schedule is valid for p.
pub fn default_power(p: u32) -> u32 {
    let raw = 10.0 + ((p as f64) / 60e6).log2().div_euclid(2.0);
    let mut power = (raw as i64).clamp(1, MAX_POWER as i64) as u32;
    while power > 1 && ProofSchedule::new(p, power).is_err() {
        power -= 1;
    }
    power
}

/// Disk needed for the proof residues of power `power`, in bytes.
pub fn disk_usage_bytes(p: u32, power: u32) -> u64 {
    (1u64 << power) * (proof_byte_len(p) as u64)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofSchedule {
    p: u32,
    power: u32,
    points: Vec<u64>,
}

impl ProofSchedule {
    pub fn new(p: u32, power: u32) -> Result<ProofSchedule, ProofError> {
        let bad = |reason: &str| ProofError::InvalidPower {
            p,
            power,
            reason: reason.into(),
        };
        if power == 0 || power > MAX_POWER {
            return Err(bad("power must be in 1..=13"));
        }
        if p < 3 || p % 2 == 0 {
            return Err(bad("proofs need an odd exponent"));
        }
        let mut points = vec![0u64];
        let mut span = u64::from(p).div_ceil(2);
        for _ in 0..power {
            let n = points.len();
            for i in 0..n {
                points.push(points[i] + span);
            }
            span = span.div_ceil(2);
        }
        points[0] = u64::from(p);
        points.sort_unstable();
        let distinct = points.windows(2).all(|w| w[0] < w[1]);
        if !distinct || points[0] == 0 || *points.last().expect("nonempty") != u64::from(p) {
            return Err(bad("the schedule has colliding points; use a lower power"));
        }
        Ok(ProofSchedule { p, power, points })
    }

    pub fn exponent(&self) -> u32 {
        self.p
    }
    pub fn power(&self) -> u32 {
        self.power
    }
    pub fn points(&self) -> &[u64] {
        &self.points
    }
    pub fn contains(&self, k: u64) -> bool {
        self.points.binary_search(&k).is_ok()
    }
    /// The first point strictly after k.
    pub fn next_after(&self, k: u64) -> Option<u64> {
        let i = self.points.partition_point(|&x| x <= k);
        self.points.get(i).copied()
    }
}

const RESIDUE_MAGIC: &[u8; 16] = b"MF-PROOF-RESIDUE";

/// Durable storage of proof residues: one file per point, written to a
/// temporary file, fsynced, renamed, the directory fsynced, then read back
/// and compared.
#[derive(Debug, Clone)]
pub struct ProofStore {
    pub schedule: ProofSchedule,
    pub dir: PathBuf,
}

impl ProofStore {
    pub fn residue_dir(proof_dir: &Path, p: u32) -> PathBuf {
        proof_dir.join(format!("M{p}.residues"))
    }

    pub fn open(proof_dir: &Path, schedule: ProofSchedule) -> std::io::Result<ProofStore> {
        let dir = Self::residue_dir(proof_dir, schedule.exponent());
        fs::create_dir_all(&dir)?;
        Ok(ProofStore { schedule, dir })
    }

    fn path(&self, k: u64) -> PathBuf {
        self.dir.join(k.to_string())
    }

    fn encode(&self, k: u64, x: &Residue) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(RESIDUE_MAGIC);
        b.extend_from_slice(&self.schedule.exponent().to_le_bytes());
        b.extend_from_slice(&k.to_le_bytes());
        b.extend_from_slice(&x.to_le_bytes());
        let h = sha3_256(&[&b]);
        b.extend_from_slice(&h);
        b
    }

    pub fn save(&self, k: u64, x: &Residue) -> std::io::Result<()> {
        debug_assert!(self.schedule.contains(k));
        let bytes = self.encode(k, x);
        crate::checkpoint::write_atomic(&self.path(k), &bytes, false)?;
        match self.load(k) {
            Ok(y) if y == *x => Ok(()),
            Ok(_) => Err(std::io::Error::other("read back differs")),
            Err(e) => Err(std::io::Error::other(e.to_string())),
        }
    }

    pub fn load(&self, k: u64) -> Result<Residue, ProofError> {
        let p = self.schedule.exponent();
        let err = |m: &str| ProofError::Residue {
            k,
            message: m.into(),
        };
        let expected = 16 + 4 + 8 + 4 * crate::residue::limb_count(p) + 32;
        let meta = fs::metadata(self.path(k)).map_err(|e| err(&e.to_string()))?;
        if meta.len() != expected as u64 {
            return Err(err("wrong size"));
        }
        let b = fs::read(self.path(k)).map_err(|e| err(&e.to_string()))?;
        let (body, h) = b.split_at(b.len() - 32);
        if sha3_256(&[body]) != h || &body[..16] != RESIDUE_MAGIC {
            return Err(err("integrity hash mismatch"));
        }
        if u32::from_le_bytes(body[16..20].try_into().expect("4")) != p
            || u64::from_le_bytes(body[20..28].try_into().expect("8")) != k
        {
            return Err(err("identity mismatch"));
        }
        Residue::from_le_bytes(p, &body[28..]).map_err(|e| err(&e.to_string()))
    }

    /// Every point at or below k is present and intact.
    pub fn check_through(&self, k: u64) -> Result<(), ProtocolError> {
        for &pt in self.schedule.points().iter().filter(|&&pt| pt <= k) {
            if self.load(pt).is_err() {
                return Err(ProtocolError::ProofResidueMissing { k: pt });
            }
        }
        Ok(())
    }
}

/// A parsed PRP proof (version 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof {
    pub p: u32,
    /// Raw bytes of B and of each middle, exactly as hashed.
    pub b: Vec<u8>,
    pub middles: Vec<Vec<u8>>,
}

fn header(power: usize, p: u32) -> String {
    format!("PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER={power}\nNUMBER=M{p}\n")
}

/// Parse "KEY=<digits>\n" at the start of `s`, returning the value and the rest.
fn take_field<'a>(s: &'a [u8], key: &[u8]) -> Option<(u32, &'a [u8])> {
    let rest = s.strip_prefix(key)?;
    let end = rest.iter().position(|&c| c == b'\n')?;
    let digits = &rest[..end];
    if digits.is_empty() || digits.len() > 10 || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let v: u64 = std::str::from_utf8(digits).ok()?.parse().ok()?;
    Some((u32::try_from(v).ok()?, &rest[end + 1..]))
}

/// Parse the header; returns (power, p, header length).
fn parse_header(bytes: &[u8]) -> Result<(u32, u32, usize), ProofError> {
    let rest = bytes
        .strip_prefix(b"PRP PROOF\nVERSION=2\nHASHSIZE=64\n".as_slice())
        .ok_or(ProofError::BadHeader)?;
    let (power, rest) = take_field(rest, b"POWER=").ok_or(ProofError::BadHeader)?;
    let (p, rest) = take_field(rest, b"NUMBER=M").ok_or(ProofError::BadHeader)?;
    Ok((power, p, bytes.len() - rest.len()))
}

impl Proof {
    pub fn power(&self) -> u32 {
        self.middles.len() as u32
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = header(self.middles.len(), self.p).into_bytes();
        out.extend_from_slice(&self.b);
        for m in &self.middles {
            out.extend_from_slice(m);
        }
        out
    }

    /// Strict parse. `expected` rejects a proof for another exponent.
    pub fn parse(bytes: &[u8], expected: Option<u32>) -> Result<Proof, ProofError> {
        let (power, p, hlen) = parse_header(bytes)?;
        if let Some(e) = expected {
            if e != p {
                return Err(ProofError::WrongExponent {
                    expected: e,
                    found: p,
                });
            }
        }
        if power == 0 || power > MAX_POWER || !crate::exponent::is_prime_u32(p) || p < 3 {
            return Err(ProofError::InvalidPower {
                p,
                power,
                reason: "outside the supported set".into(),
            });
        }
        ProofSchedule::new(p, power)?;
        let n = proof_byte_len(p);
        let want = hlen as u64 + (power as u64 + 1) * n as u64;
        let got = bytes.len() as u64;
        if got < want {
            return Err(ProofError::Truncated {
                expected: want,
                found: got,
            });
        }
        if got > want {
            return Err(ProofError::TrailingData {
                expected: want,
                found: got,
            });
        }
        let body = &bytes[hlen..];
        let mut chunks = body.chunks_exact(n).map(<[u8]>::to_vec);
        let b = chunks.next().expect("length checked");
        let middles: Vec<Vec<u8>> = chunks.collect();
        for (i, r) in std::iter::once(&b).chain(middles.iter()).enumerate() {
            let x = Residue::from_proof_bytes(p, r)
                .map_err(|_| ProofError::NoncanonicalResidue { index: i })?;
            if i > 0 && x.is_zero() {
                return Err(ProofError::ZeroResidue { index: i });
            }
        }
        Ok(Proof { p, b, middles })
    }

    /// Read a proof file, checking its size against the header before
    /// reading the body.
    pub fn read(path: &Path, expected: Option<u32>) -> Result<Proof, ProofError> {
        let io = |e: std::io::Error| ProofError::Io(format!("{}: {e}", path.display()));
        let mut f = fs::File::open(path).map_err(io)?;
        let len = f.metadata().map_err(io)?.len();
        let mut head = vec![0u8; len.min(128) as usize];
        f.read_exact(&mut head).map_err(io)?;
        let (power, p, hlen) = parse_header(&head)?;
        if let Some(e) = expected {
            if e != p {
                return Err(ProofError::WrongExponent {
                    expected: e,
                    found: p,
                });
            }
        }
        let want =
            hlen as u64 + (u64::from(power.min(MAX_POWER)) + 1) * proof_byte_len(p.max(2)) as u64;
        if len > want {
            return Err(ProofError::TrailingData {
                expected: want,
                found: len,
            });
        }
        let mut bytes = head;
        f.read_to_end(&mut bytes).map_err(io)?;
        Proof::parse(&bytes, expected)
    }

    pub fn file_name(&self) -> String {
        format!("{}-{}.proof", self.p, self.power())
    }

    pub fn write(&self, dir: &Path) -> std::io::Result<PathBuf> {
        let path = dir.join(self.file_name());
        crate::checkpoint::write_atomic(&path, &self.to_bytes(), false)?;
        Ok(path)
    }

    /// B is 9: the proven computation says probable prime.
    pub fn claims_probable_prime(&self) -> bool {
        Residue::from_proof_bytes(self.p, &self.b).is_ok_and(|b| b == Residue::from_u64(self.p, 9))
    }

    /// The hash chain exponents h_1..h_P.
    pub fn hashes(&self) -> Vec<u64> {
        let mut h = sha3_256(&[&self.b]);
        self.middles
            .iter()
            .map(|m| {
                h = sha3_256(&[&h, m]);
                first_word(&h)
            })
            .collect()
    }
}

/// a <- a^h * b (left to right binary exponentiation), using `tmp`.
fn exp_mul<B: Backend>(be: &mut B, a: Reg, h: u64, b: Reg, tmp: Reg) -> Result<(), ProtocolError> {
    if h == 0 {
        be.import(a, &Residue::from_u64(be.exponent(), 1))?;
    } else {
        be.copy(tmp, a)?;
        for bit in (0..63 - h.leading_zeros()).rev() {
            be.square(a, 1)?;
            if h >> bit & 1 == 1 {
                be.mul(a, tmp)?;
            }
        }
    }
    be.mul(a, b)?;
    Ok(())
}

/// Build the proof from the stored residues, as upstream `computeProof`.
/// This step is not checked; verify the result afterwards.
pub fn generate<B: Backend>(be: &mut B, store: &ProofStore) -> Result<Proof, ProtocolError> {
    let s = &store.schedule;
    let p = s.exponent();
    let power = s.power();
    let load = |k: u64| {
        store
            .load(k)
            .map_err(|_| ProtocolError::ProofResidueMissing { k })
    };
    let b = load(u64::from(p))?;
    let mut hash = sha3_256(&[&b.to_proof_bytes()]);
    let mut hashes: Vec<u64> = Vec::new();
    let mut middles = Vec::new();
    let regs: Vec<Reg> = (0..=power).map(|_| be.alloc()).collect::<Result<_, _>>()?;
    let tmp = be.alloc()?;
    let result = (|| -> Result<(), ProtocolError> {
        for lvl in 0..power {
            let step = 1usize << (power - lvl - 1);
            let mut top = 0usize;
            for i in 0..(1usize << lvl) {
                let w = load(s.points()[step * (2 * i + 1) - 1])?;
                be.import(regs[top], &w)?;
                top += 1;
                let mut j = 0;
                while i & (1 << j) != 0 {
                    top -= 1;
                    let h = hashes[lvl as usize - 1 - j];
                    exp_mul(be, regs[top - 1], h, regs[top], tmp)?;
                    j += 1;
                }
            }
            debug_assert_eq!(top, 1);
            let m = be.export(regs[0])?;
            if m.is_zero() {
                return Err(ProtocolError::Proof(ProofError::ZeroResidue {
                    index: lvl as usize + 1,
                }));
            }
            let mb = m.to_proof_bytes();
            hash = sha3_256(&[&hash, &mb]);
            hashes.push(first_word(&hash));
            middles.push(mb);
        }
        Ok(())
    })();
    for r in regs.into_iter().chain([tmp]) {
        be.free(r)?;
    }
    result?;
    Ok(Proof {
        p,
        b: b.to_proof_bytes(),
        middles,
    })
}

/// Outcome of a successful verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verified {
    pub probable_prime: bool,
    pub squarings: u64,
    pub gerbicz_passed: u64,
    pub gerbicz_failed: u64,
}

/// Verify a proof as upstream `Proof::verify`, with the final long squaring
/// run protected by the generic Gerbicz check (start value A).
pub fn verify<B: Backend>(
    be: &mut B,
    proof: &Proof,
    ctl: &mut Control,
) -> Result<Verified, ProtocolError> {
    let p = proof.p;
    if be.exponent() != p {
        return Err(ProtocolError::Config(format!(
            "backend is for M{}, proof for M{p}",
            be.exponent()
        )));
    }
    let res = |bytes: &[u8]| Residue::from_proof_bytes(p, bytes).map_err(|_| ProofError::BadHeader);
    let hashes = proof.hashes();
    let (a, b, m, tmp) = (be.alloc()?, be.alloc()?, be.alloc()?, be.alloc()?);
    be.import(a, &Residue::from_u64(p, 3))?;
    be.import(b, &res(&proof.b)?)?;
    let mut span = u64::from(p);
    for (i, mid) in proof.middles.iter().enumerate() {
        be.import(m, &res(mid)?)?;
        let h = hashes[i];
        if span % 2 == 1 {
            be.square(b, 1)?;
        }
        // B <- M^h * B, computed in m then copied.
        exp_mul(be, m, h, b, tmp)?;
        be.copy(b, m)?;
        be.import(m, &res(mid)?)?;
        exp_mul(be, a, h, m, tmp)?;
        span = span.div_ceil(2);
    }
    let a_res = be.export(a)?;
    let b_res = be.export(b)?;
    for r in [a, b, m, tmp] {
        be.free(r)?;
    }
    let cfg = CheckedPowerConfig::for_length(span);
    let out = checked_power(be, &a_res, span, &cfg, ctl)?;
    if out.residue != b_res {
        return Err(ProtocolError::Proof(ProofError::Rejected));
    }
    Ok(Verified {
        probable_prime: proof.claims_probable_prime(),
        squarings: span,
        gerbicz_passed: out.passed,
        gerbicz_failed: out.failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_definition() {
        // p = 13, spans 7, 4: {0} -> {0, 7} -> {0, 7, 4, 11}, then 0 -> 13.
        assert_eq!(ProofSchedule::new(13, 2).unwrap().points(), &[4, 7, 11, 13]);
        // Power 3 adds span 2: 11 + 2 = 13 collides with the final point.
        assert!(ProofSchedule::new(13, 3).is_err());
        let s = ProofSchedule::new(1279, 3).unwrap();
        assert_eq!(s.points(), &[160, 320, 480, 640, 800, 960, 1120, 1279]);
        assert_eq!(s.next_after(0), Some(160));
        assert_eq!(s.next_after(160), Some(320));
        assert_eq!(s.next_after(1279), None);
    }

    #[test]
    fn colliding_powers_rejected() {
        assert!(ProofSchedule::new(3, 1).is_ok());
        assert!(ProofSchedule::new(3, 2).is_err());
        assert!(ProofSchedule::new(1279, 9).is_ok());
        // Spans 640 + 320 + ... + 2 = 1280 > 1279: the top point leaves the range.
        assert!(ProofSchedule::new(1279, 10).is_err());
    }

    #[test]
    fn sha3_is_standard() {
        // FIPS 202 SHA3-256 of the empty string.
        let h = sha3_256(&[]);
        assert_eq!(h[..4], [0xa7, 0xff, 0xc6, 0xf8]);
    }

    #[test]
    fn header_parsing_is_strict() {
        assert_eq!(
            parse_header(b"PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=8\nNUMBER=M216091\n")
                .unwrap()
                .0,
            8
        );
        assert_eq!(
            parse_header(b"PRP PROOF\nVERSION=1\nHASHSIZE=64\nPOWER=8\nNUMBER=M216091\n"),
            Err(ProofError::BadHeader)
        );
        assert_eq!(
            parse_header(b"PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=x\nNUMBER=M216091\n"),
            Err(ProofError::BadHeader)
        );
    }
}
