// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! MersenneForge: a Rust engine for Lucas-Lehmer and base-3 PRP tests of
//! Mersenne numbers M_p = 2^p - 1.
//!
//! The LL, PRP, Gerbicz check, proof, and checkpoint state machines are
//! written once over [`backend::Backend`]; backends only do arithmetic.

pub mod backend;
pub mod bench;
pub mod checkpoint;
pub mod exponent;
pub mod ll;
pub mod oracle;
pub mod proof;
pub mod prp;
pub mod residue;
pub mod run;
pub mod select;
