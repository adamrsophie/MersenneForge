// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Format evidence in one direction: a proof file produced by upstream PRPLL
//! is parsed and verified by this project's verifier. This does not test the
//! other direction (an upstream verifier accepting our proofs).
//!
//! Run with the path of an upstream proof, for example the `test/786433-10.proof`
//! fixture shipped in the PRPLL source tree at the pinned commit:
//!
//! ```text
//! MERSENNE_FORGE_UPSTREAM_PROOF=<path>/786433-10.proof \
//!     cargo test --release --test upstream_proof -- --ignored
//! ```

use mersenne_forge::backend::dwt::DwtBackend;
use mersenne_forge::proof::{self, Proof};
use mersenne_forge::run::Control;

#[test]
#[ignore = "needs MERSENNE_FORGE_UPSTREAM_PROOF; slow outside release builds"]
fn verifies_an_upstream_proof() {
    let path = std::env::var("MERSENNE_FORGE_UPSTREAM_PROOF")
        .expect("set MERSENNE_FORGE_UPSTREAM_PROOF to an upstream PRPLL proof file");
    let pr = Proof::read(std::path::Path::new(&path), None).expect("upstream proof parses");
    let mut be = DwtBackend::new(pr.p).expect("exponent in the DWT range");
    let v = proof::verify(&mut be, &pr, &mut Control::default()).expect("upstream proof verifies");
    println!(
        "M{} power {}: accepted, claims {}",
        pr.p,
        pr.power(),
        if v.probable_prime {
            "probable prime"
        } else {
            "composite"
        }
    );

    let mut bytes = std::fs::read(&path).unwrap();
    let n = bytes.len();
    bytes[n - 7] ^= 1;
    let bad = Proof::parse(&bytes, Some(pr.p)).unwrap();
    assert!(proof::verify(&mut be, &bad, &mut Control::default()).is_err());
}
