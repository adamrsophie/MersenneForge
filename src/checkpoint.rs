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
//! the file size before any residue is read. The hash detects damaged bytes;
//! it authenticates nothing (anyone can recompute it). The PRP on-load check
//! detects a damaged residue pair; the LL residue is covered by the next
//! Jacobi check. The iteration count and the configuration are trusted
//! metadata that no local check can authenticate.
//!
//! Durability: the new file is written to a temporary file of its own,
//! `<path>.tmp.<pid>.<n>`, created exclusively and fsynced; the
//! existing `<path>` is renamed to `<path>.prev` only if it loads and passes
//! its hash, and is otherwise removed so that a good `<path>.prev` is kept;
//! the temporary file is renamed to `<path>`; the directory is fsynced. If
//! `<path>` or `<path>.prev` loaded before a save, a crash at any point of
//! the save leaves one of them loadable, and the loader falls back to
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
/// Backend kind of a portable checkpoint: one any backend may resume.
pub const PORTABLE: &str = "portable";

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
    /// "verified" or "completed" (PRP: passed a Gerbicz check), or
    /// "provisional" (LL: residue at k plus the last Jacobi verified residue
    /// at verified_k). A completed PRP state also stores the terminal x_p.
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

/// Counter for temporary names, unique within this process.
static NEXT_TMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A temporary name next to `path`: `<path>.tmp.<pid>.<n>`. A file left at
/// that name by someone else is possible; `write_atomic` then picks another.
fn tmp_path(path: &Path) -> PathBuf {
    use std::sync::atomic::Ordering;
    let mut s = path.as_os_str().to_owned();
    s.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        NEXT_TMP.fetch_add(1, Ordering::Relaxed)
    ));
    PathBuf::from(s)
}

/// Fsync the directory that contains `path`, so a new or renamed entry in
/// it is durable. Windows has no portable directory fsync (opening a
/// directory needs a special flag); there renames are already journaled by
/// NTFS, so this is a no-op.
pub fn sync_dir(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let dir = path
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::File::open(dir)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// `create_dir_all`, then fsync the parent of every directory it created,
/// so the new directory entries are durable before files inside them are
/// relied on.
pub fn create_dir_all_durable(dir: &Path) -> std::io::Result<()> {
    let mut missing = Vec::new();
    let mut d = dir;
    while !d.as_os_str().is_empty() && !d.exists() {
        missing.push(d.to_path_buf());
        match d.parent() {
            Some(parent) => d = parent,
            None => break,
        }
    }
    fs::create_dir_all(dir)?;
    for m in missing.iter().rev() {
        sync_dir(m)?;
    }
    Ok(())
}

/// Write `bytes` to `path` atomically: temporary file, fsync, rename,
/// directory fsync. With `checkpoint_rotation`, the existing `path` is first
/// renamed to `.prev` if it is a checkpoint that loads, and removed if it
/// does not, so a corrupt file never replaces a good `.prev`.
pub fn write_atomic(path: &Path, bytes: &[u8], checkpoint_rotation: bool) -> std::io::Result<()> {
    // Exclusive creation: the file is this writer's alone. A name that
    // already exists belongs to someone else, is left untouched, and another
    // name is tried.
    let (tmp, mut f) = (0..1000)
        .find_map(|_| {
            let tmp = tmp_path(path);
            let mut o = fs::OpenOptions::new();
            o.write(true).create_new(true);
            // Readable by the owner only until it takes the target's mode.
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
            match o.open(&tmp) {
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                r => Some(r.map(|f| (tmp, f))),
            }
        })
        .unwrap_or_else(|| {
            Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("{}: no free temporary name", path.display()),
            ))
        })?;
    // From here the file is ours: on any failure it is removed. A file that
    // replaces another keeps the other's permissions.
    let keep_mode = fs::metadata(path).map(|m| m.permissions());
    let result = f
        .write_all(bytes)
        .and_then(|()| match keep_mode {
            Ok(perm) => f.set_permissions(perm),
            Err(_) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    // A new file gets the usual mode under the umask.
                    let umask = current_umask();
                    f.set_permissions(fs::Permissions::from_mode(0o666 & !umask))
                }
                #[cfg(not(unix))]
                Ok(())
            }
        })
        .and_then(|()| f.sync_all())
        .and_then(|()| publish(path, &tmp, checkpoint_rotation));
    drop(f);
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    } else if checkpoint_rotation {
        remove_stale_temporaries(path);
    }
    result
}

/// Remove temporary files a crashed writer left next to a checkpoint:
/// `<name>.tmp.<pid>.<n>` from another process, untouched for an hour. A
/// checkpoint has one writer at a time (it holds the checkpoint lock), so
/// such a file cannot belong to a live write.
fn remove_stale_temporaries(path: &Path) {
    let (Some(dir), Some(name)) = (path.parent(), path.file_name()) else {
        return;
    };
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let prefix = format!("{}.tmp.", name.to_string_lossy());
    let ours = format!("{prefix}{}.", std::process::id());
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        let old = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age.as_secs() > 3600);
        if n.starts_with(&prefix) && !n.starts_with(&ours) && old {
            let _ = fs::remove_file(e.path());
        }
    }
}

/// The process umask, read without changing it for longer than a moment.
#[cfg(unix)]
fn current_umask() -> u32 {
    // There is no portable way to read the umask without setting it; the
    // value is restored at once. Reading it from a fresh file avoids that.
    let dir = std::env::temp_dir();
    for n in 0..100u32 {
        let probe = dir.join(format!(".mf-umask.{}.{n}", std::process::id()));
        if let Ok(f) = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = f
                .metadata()
                .map(|m| m.permissions().mode())
                .unwrap_or(0o644);
            let _ = fs::remove_file(&probe);
            return !mode & 0o666;
        }
    }
    0o022
}

fn publish(path: &Path, tmp: &Path, checkpoint_rotation: bool) -> std::io::Result<()> {
    if checkpoint_rotation && path.exists() {
        match Checkpoint::load_file(path) {
            Ok(_) => fs::rename(path, prev_path(path))?,
            // Proven damage: the file cannot be anyone's state.
            Err(CheckpointError::Corrupt(_)) => fs::remove_file(path)?,
            // A newer format or a file this process cannot read now may still
            // be good: keep it under another name instead of deleting it.
            Err(_) => {
                let mut aside = path.as_os_str().to_owned();
                aside.push(".unloadable");
                fs::rename(path, aside)?;
            }
        }
    }
    fs::rename(tmp, path)?;
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

    /// This checkpoint's last verified state, not tied to a backend. A PRP
    /// state is verified as stored. An LL state rewinds to its last Jacobi
    /// verified residue, so nothing unverified moves to another backend.
    pub fn portable(mut self) -> Result<Checkpoint, CheckpointError> {
        match (self.meta.protocol.as_str(), self.meta.state.as_str()) {
            ("ll-jacobi/1", _) if self.residues.len() == 2 => {
                self.meta.k = self.meta.verified_k;
                self.residues[0] = self.residues[1].clone();
            }
            (p, "verified" | "completed") if p.starts_with("prp-") => {}
            (p, st) => {
                return Err(CheckpointError::Mismatch(format!(
                    "no portable form for a {p} checkpoint in state {st:?}"
                )))
            }
        }
        self.meta.backend = BackendIdentity {
            kind: PORTABLE.into(),
            build: String::new(),
            plan: String::new(),
            device: String::new(),
        };
        Ok(self)
    }

    pub fn save(&self, path: &Path) -> Result<(), CheckpointError> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            create_dir_all_durable(dir).map_err(|e| CheckpointError::Io(e.to_string()))?;
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
        let rest_len = usize::try_from(file_len - 32 - meta_len - 4)
            .map_err(|_| corrupt("residues too large for this platform"))?;
        let mut rest = Vec::new();
        rest.try_reserve_exact(rest_len).map_err(|e| {
            CheckpointError::Io(format!("{}: cannot allocate: {e}", path.display()))
        })?;
        // Never more than one byte past the expected length, so a file that
        // grows while it is read costs no more memory than the check allows.
        (&mut f)
            .take(rest_len as u64 + 1)
            .read_to_end(&mut rest)
            .map_err(io)?;
        if rest.len() != rest_len {
            return Err(corrupt("file changed while it was read"));
        }
        let (residue_bytes, hash) = rest.split_at(rest_len - 32);
        if sha3_256(&[&head, &meta_bytes, &cnt, residue_bytes]) != hash {
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

    /// Load `path`, falling back to `path.prev` only when `path` is missing
    /// or proven corrupt. Any other failure (a newer format, a read error)
    /// is returned, so an older `.prev` never silently replaces a primary
    /// that may still be good. Returns the checkpoint and the file used.
    pub fn load_latest(path: &Path) -> Result<(Checkpoint, PathBuf), CheckpointError> {
        match Checkpoint::load_file(path) {
            Ok(c) => Ok((c, path.to_path_buf())),
            Err(first) if path.exists() && !matches!(first, CheckpointError::Corrupt(_)) => {
                Err(first)
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    /// Files at the next temporary names belong to someone else: they keep
    /// their bytes, and the write is published through a name of its own.
    #[test]
    fn existing_temporary_names_are_skipped_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        // Other tests may take names concurrently; cover a wide window.
        let start = NEXT_TMP.load(Ordering::SeqCst);
        let sentinels: Vec<PathBuf> = (start..start + 64)
            .map(|n| {
                let mut s = path.as_os_str().to_owned();
                s.push(format!(".tmp.{}.{n}", std::process::id()));
                PathBuf::from(s)
            })
            .collect();
        for s in &sentinels {
            fs::write(s, b"someone else's").unwrap();
        }
        write_atomic(&path, b"new contents", false).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new contents");
        assert!(
            NEXT_TMP.load(Ordering::SeqCst) > start + 1,
            "a collision was retried"
        );
        for s in &sentinels {
            assert_eq!(fs::read(s).unwrap(), b"someone else's", "{}", s.display());
        }
        // Nothing else is left behind.
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            sentinels.len() + 1
        );
    }

    /// A temporary file this writer created is removed when publication
    /// fails (here the target is a nonempty directory, so the rename fails).
    #[test]
    fn a_failed_publication_removes_only_its_own_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("target");
        fs::create_dir(&path).unwrap();
        fs::write(path.join("inside"), b"x").unwrap();
        assert!(write_atomic(&path, b"data", false).is_err());
        let names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["target"]);
        assert_eq!(fs::read(path.join("inside")).unwrap(), b"x");
    }
}
