// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The generic protocol suite on the CUDA backend. Needs a CUDA device:
//! every test fails (never skips) when none exists. Not run in CI.

#![cfg(feature = "cuda")]

mod common;

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::backend::cuda::{CudaBackend, CudaOptions};

fn mk(p: u32) -> CudaBackend {
    CudaBackend::new(
        p,
        &CudaOptions {
            device: None,
            ..CudaOptions::default()
        },
    )
    .expect("a CUDA device is required for these tests")
}

protocol_suite!(mk);

#[test]
fn cpu_dwt_and_cuda_identical() {
    common::cross_backend_identical(|p| CpuBackend::new(p).unwrap(), mk);
    common::cross_backend_identical(
        |p| mersenne_forge::backend::dwt::DwtBackend::new(p).unwrap(),
        mk,
    );
}
