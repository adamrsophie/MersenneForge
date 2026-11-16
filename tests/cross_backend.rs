// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The same exponent on different backends gives identical full final
//! residues and byte-identical proof files.

mod common;

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::backend::dwt::DwtBackend;

#[test]
fn cpu_and_dwt_identical() {
    common::cross_backend_identical(
        |p| CpuBackend::new(p).unwrap(),
        |p| DwtBackend::new(p).unwrap(),
    );
}
