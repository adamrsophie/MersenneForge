// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The generic protocol suite on the wgpu GPU backend. Needs a GPU adapter:
//! every test fails (never skips) when none exists. Not run in CI.

#![cfg(feature = "gpu")]

mod common;

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::backend::gpu::{GpuBackend, GpuOptions};

fn mk(p: u32) -> GpuBackend {
    GpuBackend::new(p, &GpuOptions { device: None })
        .expect("a GPU adapter is required for these tests")
}

protocol_suite!(mk);

#[test]
fn cpu_dwt_and_gpu_identical() {
    common::cross_backend_identical(|p| CpuBackend::new(p).unwrap(), mk);
    common::cross_backend_identical(
        |p| mersenne_forge::backend::dwt::DwtBackend::new(p).unwrap(),
        mk,
    );
}
