// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The generic protocol suite on the exact CPU backend.

mod common;

use mersenne_forge::backend::cpu::CpuBackend;

fn mk(p: u32) -> CpuBackend {
    CpuBackend::new(p).expect("CPU backend")
}

protocol_suite!(mk);
