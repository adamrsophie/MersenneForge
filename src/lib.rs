// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! MersenneForge: a Rust engine for Lucas-Lehmer and base-3 PRP tests of
//! Mersenne numbers M_p = 2^p - 1.
//!
//! The LL, PRP, Gerbicz check, proof, and checkpoint state machines are
//! written once over [`backend::Backend`]; backends only do arithmetic.

pub mod backend;
pub mod bench;
pub mod checkpoint;
pub mod evidence;
pub mod exponent;
pub mod ll;
pub mod oracle;
pub mod proof;
pub mod prp;
pub mod residue;
pub mod run;
pub mod select;
pub mod worktodo;

// Include source bytes so packaged builds and uncommitted changes have an
// identity without relying on a Git checkout or a native build script.
const BUILD_SOURCES: &[(&str, &[u8])] = &[
    ("Cargo.lock", include_bytes!("../Cargo.lock")),
    ("Cargo.toml", include_bytes!("../Cargo.toml")),
    ("src/backend/cpu.rs", include_bytes!("backend/cpu.rs")),
    (
        "src/backend/cuda/kernels.cu",
        include_bytes!("backend/cuda/kernels.cu"),
    ),
    (
        "src/backend/cuda/mod.rs",
        include_bytes!("backend/cuda/mod.rs"),
    ),
    (
        "src/backend/cuda/model.rs",
        include_bytes!("backend/cuda/model.rs"),
    ),
    ("src/backend/dwt.rs", include_bytes!("backend/dwt.rs")),
    (
        "src/backend/gpu/kernels.wgsl",
        include_bytes!("backend/gpu/kernels.wgsl"),
    ),
    (
        "src/backend/gpu/mod.rs",
        include_bytes!("backend/gpu/mod.rs"),
    ),
    (
        "src/backend/gpu/model.rs",
        include_bytes!("backend/gpu/model.rs"),
    ),
    ("src/backend/mod.rs", include_bytes!("backend/mod.rs")),
    ("src/bench.rs", include_bytes!("bench.rs")),
    ("src/checkpoint.rs", include_bytes!("checkpoint.rs")),
    ("src/evidence.rs", include_bytes!("evidence.rs")),
    ("src/exponent.rs", include_bytes!("exponent.rs")),
    ("src/lib.rs", include_bytes!("lib.rs")),
    ("src/ll.rs", include_bytes!("ll.rs")),
    ("src/main.rs", include_bytes!("main.rs")),
    ("src/oracle.rs", include_bytes!("oracle.rs")),
    ("src/proof.rs", include_bytes!("proof.rs")),
    ("src/prp.rs", include_bytes!("prp.rs")),
    ("src/residue.rs", include_bytes!("residue.rs")),
    ("src/run.rs", include_bytes!("run.rs")),
    ("src/select.rs", include_bytes!("select.rs")),
    ("src/worktodo.rs", include_bytes!("worktodo.rs")),
];

/// Crate version and SHA3-256 of the source files, manifest, and lockfile.
/// This identifies source content, not the compiler, build flags, or hardware.
pub fn build_identity() -> &'static str {
    use sha3::{Digest, Sha3_256};
    use std::sync::OnceLock;

    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let mut hash = Sha3_256::new();
        for (path, bytes) in BUILD_SOURCES {
            hash.update((path.len() as u64).to_le_bytes());
            hash.update(path.as_bytes());
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        format!(
            "mersenne-forge {} source-sha3-256 {:x}",
            env!("CARGO_PKG_VERSION"),
            hash.finalize()
        )
    })
}

#[cfg(test)]
mod build_tests {
    use super::*;
    use crate::backend::Backend;
    use std::path::Path;

    #[test]
    fn build_identity_covers_all_implementation_files() {
        fn collect(root: &Path, dir: &Path, files: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    collect(root, &path, files);
                } else if matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("rs" | "cu" | "wgsl")
                ) {
                    files.push(
                        path.strip_prefix(root)
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .replace('\\', "/"),
                    );
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut expected = vec!["Cargo.lock".into(), "Cargo.toml".into()];
        collect(root, &root.join("src"), &mut expected);
        expected.sort();
        assert_eq!(
            expected,
            BUILD_SOURCES
                .iter()
                .map(|(path, _)| path.to_string())
                .collect::<Vec<_>>()
        );
        for (path, bytes) in BUILD_SOURCES {
            assert_eq!(std::fs::read(root.join(path)).unwrap(), *bytes, "{path}");
        }
        let id = build_identity();
        let hash = id.rsplit(' ').next().unwrap();
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            id,
            backend::cpu::CpuBackend::new(127).unwrap().identity().build
        );
    }
}
