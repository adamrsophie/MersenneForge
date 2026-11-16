// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Project checkpoint format (not upstream savefile compatible).
//!
//! Layout, all integers little-endian:
//!
//! ```text
//! magic        20 bytes  "MERSENNE-FORGE-CKPT\n"
//! version      u32       FORMAT_VERSION
//! meta_len     u64       at most MAX_META
//! meta         JSON      CheckpointMeta
//! count        u32       number of residues, at most 4
//! residues     count * (u32 exponent, u64 byte_len, byte_len bytes of limbs)
//! hash         32 bytes  SHA3-256 of everything above
//! ```
//!
//! The total size is computed from the header and metadata and compared with
//! the file size before any residue is read. The hash detects corrupted bytes;
//! it is not mathematical verification. The state machines check the loaded
//! residues mathematically (PRP: a Gerbicz check on load; LL: the residue is
//! resumed and the next Jacobi check covers it).
//!
//! Durability: the new file is written to `<path>.tmp` and fsynced; the
//! existing checkpoint is renamed to `<path>.prev`; the temporary file is
//! renamed to `<path>`; the directory is fsynced. A crash at any point leaves
//! either `<path>` or `<path>.prev` intact, and the loader falls back to
//! `<path>.prev` when `<path>` is missing or corrupt.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::backend::BackendIdentity;
use crate::proof::sha3_256;
use crate::residue::{limb_count, Residue};

pub const FORMAT_VERSION: u32 = 1;
const MAGIC: &[u8; 20] = b"MERSENNE-FORGE-CKPT\n";
const MAX_META: u64 = 1 << 20;
const MAX_RESIDUES: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointError {
    Io(String),
    /// Wrong magic, sizes, or hash: corrupted or not a checkpoint.
    Corrupt(String),
    UnsupportedVersion(u32),
    /// Metadata that does not match the run being resumed.
    Mismatch(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(m) => write!(f, "{m}"),
            CheckpointError::Corrupt(m) => write!(f, "corrupt checkpoint: {m}"),
            CheckpointError::UnsupportedVersion(v) => {
                write!(f, "unsupported checkpoint format version {v}")
            }
            CheckpointError::Mismatch(m) => write!(f, "checkpoint does not match: {m}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

/// Everything needed to resume. Residues are stored separately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub format_version: u32,
    /// "prp-base3-gerbicz/1" or "ll-jacobi/1".
    pub protocol: String,
    pub exponent: u32,
    /// Iteration of the stored state.
    pub k: u64,
    /// "verified" (PRP: passed a Gerbicz check; LL: a provisional residue at
    /// k plus the last Jacobi verified residue at verified_k).
    pub state: String,
    pub verified_k: u64,
    pub n_errors: u32,
    /// Proof points with k at or below this are durable (PRP only).
    pub proof_durable_through: Option<u64>,
    pub backend: BackendIdentity,
    /// The full run configuration, as effective for this run.
    pub config: serde_json::Value,
    /// The configuration as requested on the command line.
    pub requested: serde_json::Value,
    pub elapsed_secs: f64,
    pub crate_version: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    pub meta: CheckpointMeta,
    pub residues: Vec<Residue>,
}

pub fn prev_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".prev");
    PathBuf::from(s)
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

fn sync_dir(path: &Path) -> std::io::Result<()> {
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::File::open(dir)?.sync_all()
}

/// Write `bytes` to `path` atomically: temporary file, fsync, optional
/// rotation of the old file to `.prev`, rename, directory fsync.
pub fn write_atomic(path: &Path, bytes: &[u8], keep_prev: bool) -> std::io::Result<()> {
    let tmp = tmp_path(path);
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    if keep_prev && path.exists() {
        fs::rename(path, prev_path(path))?;
    }
    fs::rename(&tmp, path)?;
    sync_dir(path)
}

impl Checkpoint {
    pub fn to_bytes(&self) -> Vec<u8> {
        let meta = serde_json::to_vec(&self.meta).expect("metadata serializes");
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        b.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        b.extend_from_slice(&meta);
        b.extend_from_slice(&(self.residues.len() as u32).to_le_bytes());
        for r in &self.residues {
            let bytes = r.to_le_bytes();
            b.extend_from_slice(&r.exponent().to_le_bytes());
            b.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            b.extend_from_slice(&bytes);
        }
        let h = sha3_256(&[&b]);
        b.extend_from_slice(&h);
        b
    }

    pub fn save(&self, path: &Path) -> Result<(), CheckpointError> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir).map_err(|e| CheckpointError::Io(e.to_string()))?;
        }
        write_atomic(path, &self.to_bytes(), true)
            .map_err(|e| CheckpointError::Io(format!("{}: {e}", path.display())))
    }

    /// Load one file, validating sizes before reading residues.
    pub fn load_file(path: &Path) -> Result<Checkpoint, CheckpointError> {
        let io = |e: std::io::Error| CheckpointError::Io(format!("{}: {e}", path.display()));
        let corrupt = |m: &str| CheckpointError::Corrupt(m.to_string());
        let mut f = fs::File::open(path).map_err(io)?;
        let file_len = f.metadata().map_err(io)?.len();
        let mut head = [0u8; 32];
        if file_len < 32 + 4 + 32 {
            return Err(corrupt("file too short"));
        }
        f.read_exact(&mut head).map_err(io)?;
        if &head[..20] != MAGIC {
            return Err(corrupt("bad magic"));
        }
        let version = u32::from_le_bytes(head[20..24].try_into().expect("4"));
        let meta_len = u64::from_le_bytes(head[24..32].try_into().expect("8"));
        if meta_len > MAX_META || 32 + meta_len + 4 + 32 > file_len {
            return Err(corrupt("metadata length out of range"));
        }
        let mut meta_bytes = vec![0u8; meta_len as usize];
        f.read_exact(&mut meta_bytes).map_err(io)?;
        let mut cnt = [0u8; 4];
        f.read_exact(&mut cnt).map_err(io)?;
        let count = u32::from_le_bytes(cnt);
        let meta: CheckpointMeta =
            serde_json::from_slice(&meta_bytes).map_err(|e| corrupt(&format!("metadata: {e}")))?;
        if count > MAX_RESIDUES || meta.exponent < 2 {
            return Err(corrupt("residue count or exponent out of range"));
        }
        let rlen = 4 * limb_count(meta.exponent) as u64;
        let want = 32 + meta_len + 4 + u64::from(count) * (12 + rlen) + 32;
        if want != file_len {
            return Err(corrupt(&format!(
                "size {file_len} does not match the header ({want})"
            )));
        }
        let mut rest = Vec::with_capacity((file_len - 32 - meta_len - 4) as usize);
        f.read_to_end(&mut rest).map_err(io)?;
        let mut all = Vec::with_capacity(file_len as usize);
        all.extend_from_slice(&head);
        all.extend_from_slice(&meta_bytes);
        all.extend_from_slice(&cnt);
        all.extend_from_slice(&rest);
        let (body, hash) = all.split_at(all.len() - 32);
        if sha3_256(&[body]) != hash {
            return Err(corrupt("integrity hash mismatch"));
        }
        if version != FORMAT_VERSION || meta.format_version != FORMAT_VERSION {
            return Err(CheckpointError::UnsupportedVersion(
                version.max(meta.format_version),
            ));
        }
        let mut residues = Vec::new();
        let mut at = 0usize;
        for _ in 0..count {
            let p = u32::from_le_bytes(rest[at..at + 4].try_into().expect("4"));
            let len = u64::from_le_bytes(rest[at + 4..at + 12].try_into().expect("8"));
            if p != meta.exponent || len != rlen {
                return Err(corrupt("residue header mismatch"));
            }
            let r = Residue::from_le_bytes(p, &rest[at + 12..at + 12 + len as usize])
                .map_err(|e| corrupt(&format!("residue: {e}")))?;
            residues.push(r);
            at += 12 + len as usize;
        }
        Ok(Checkpoint { meta, residues })
    }

    /// Load `path`, falling back to `path.prev` when `path` is missing or
    /// corrupt. Returns the checkpoint and the file actually used.
    pub fn load_latest(path: &Path) -> Result<(Checkpoint, PathBuf), CheckpointError> {
        match Checkpoint::load_file(path) {
            Ok(c) => Ok((c, path.to_path_buf())),
            Err(first) => {
                let prev = prev_path(path);
                match Checkpoint::load_file(&prev) {
                    Ok(c) => Ok((c, prev)),
                    Err(_) => Err(first),
                }
            }
        }
    }
}
