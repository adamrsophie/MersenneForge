// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Interoperability in the other direction: a proof generated here is given
//! to an unmodified upstream PRPLL verifier (`prpll -verify <file>`), which
//! must accept it, and must reject a copy with one bit flipped. This has not
//! been run unless a validation record says so.
//!
//! The test fails (never skips) when the verifier is not given:
//!
//! ```text
//! MERSENNE_FORGE_UPSTREAM_VERIFIER=<path to prpll> \
//!     cargo test --release --test upstream_verifier -- --ignored
//! ```
//!
//! The exponent defaults to 786433 (power 10), the size of the proof shipped
//! in the PRPLL source tree, so it is inside the upstream FFT range;
//! `MERSENNE_FORGE_UPSTREAM_EXPONENT` overrides it. The proof is generated on
//! the `dwt` backend, which takes a while at this size.

use std::path::Path;
use std::process::Command;

use mersenne_forge::backend::dwt::DwtBackend;
use mersenne_forge::prp::{run_prp, PrpConfig};
use mersenne_forge::run::{Control, ProofStage};

/// Run the upstream verifier on `proof` in `dir`; true when it reports the
/// proof as verified and exits successfully. Upstream logs
/// `proof '<file>' verified` or `... failed`, and a failed proof ends the run
/// with an error.
fn upstream_accepts(verifier: &str, proof: &Path, dir: &Path) -> bool {
    let out = Command::new(verifier)
        .arg("-verify")
        .arg(proof)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("{verifier}: {e}"));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    println!("{text}");
    out.status.success() && text.contains("' verified")
}

#[test]
#[ignore = "needs MERSENNE_FORGE_UPSTREAM_VERIFIER (an upstream prpll binary); slow"]
fn upstream_verifier_accepts_our_proof() {
    let verifier = std::env::var("MERSENNE_FORGE_UPSTREAM_VERIFIER")
        .expect("set MERSENNE_FORGE_UPSTREAM_VERIFIER to the path of an upstream prpll binary");
    let p: u32 = std::env::var("MERSENNE_FORGE_UPSTREAM_EXPONENT")
        .map(|v| v.parse().expect("an exponent"))
        .unwrap_or(786_433);
    let dir = tempfile::tempdir().unwrap();
    let cfg = PrpConfig {
        proof_power: Some(10),
        proof_dir: Some(dir.path().join("proof")),
        ..PrpConfig::default()
    };
    let mut be = DwtBackend::new(p).expect("exponent in the dwt range");
    let r = run_prp(
        &mut be,
        p,
        &cfg,
        serde_json::json!({}),
        None,
        &mut Control::default(),
    )
    .expect("PRP run");
    assert_eq!(r.evidence.proof_stage, ProofStage::Verified);
    let proof = r.evidence.proof.unwrap().path;

    assert!(
        upstream_accepts(&verifier, &proof, dir.path()),
        "the upstream verifier did not accept {}",
        proof.display()
    );
    let mut bytes = std::fs::read(&proof).unwrap();
    let n = bytes.len();
    bytes[n - 7] ^= 1;
    let bad = dir.path().join(format!(
        "bad-{}",
        proof.file_name().unwrap().to_string_lossy()
    ));
    std::fs::write(&bad, &bytes).unwrap();
    assert!(
        !upstream_accepts(&verifier, &bad, dir.path()),
        "the upstream verifier accepted a corrupted proof"
    );
}
