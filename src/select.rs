// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Backend selection by name, used by the CLI, the bench runner, and tests.
//! Every backend kind is constructed here, so feature gating lives in one
//! place.

use serde::{Deserialize, Serialize};

use crate::backend::{Backend, BackendError, BackendIdentity, BackendResult, MemoryEstimate, Reg};
use crate::residue::Residue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Cpu,
    Dwt,
    Gpu,
    Cuda,
}

impl BackendKind {
    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Dwt => "dwt",
            BackendKind::Gpu => "gpu",
            BackendKind::Cuda => "cuda",
        }
    }

    pub fn parse(s: &str) -> Option<BackendKind> {
        match s {
            "cpu" => Some(BackendKind::Cpu),
            "dwt" => Some(BackendKind::Dwt),
            "gpu" => Some(BackendKind::Gpu),
            "cuda" => Some(BackendKind::Cuda),
            _ => None,
        }
    }

    /// GPU kinds need a bounded budget.
    pub fn is_device(self) -> bool {
        matches!(self, BackendKind::Gpu | BackendKind::Cuda)
    }

    /// Whether this binary contains the backend.
    pub fn built(self) -> bool {
        match self {
            BackendKind::Cpu | BackendKind::Dwt => true,
            BackendKind::Gpu => cfg!(feature = "gpu"),
            BackendKind::Cuda => cfg!(feature = "cuda"),
        }
    }

    /// Inclusive supported exponent range, if built.
    pub fn supported_exponents(self) -> Option<(u32, u32)> {
        match self {
            BackendKind::Cpu => Some(crate::backend::cpu::supported_exponents()),
            BackendKind::Dwt => Some(crate::backend::dwt::supported_exponents()),
            #[cfg(feature = "gpu")]
            BackendKind::Gpu => Some(crate::backend::gpu::supported_exponents()),
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => Some(crate::backend::cuda::supported_exponents()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// Memory figures for exponent p from the plan this backend would choose,
    /// if built. Pure host computation; no device is opened.
    pub fn memory_estimate(self, p: u32) -> Option<BackendResult<MemoryEstimate>> {
        match self {
            BackendKind::Cpu => Some(crate::backend::cpu::memory_estimate(p)),
            BackendKind::Dwt => Some(crate::backend::dwt::memory_estimate(p)),
            #[cfg(feature = "gpu")]
            BackendKind::Gpu => Some(crate::backend::gpu::memory_estimate(p)),
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => Some(crate::backend::cuda::memory_estimate(p)),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

/// Construct a backend. A kind not compiled into this binary is an explicit
/// `Unsupported` error naming the Cargo feature.
pub fn open(kind: BackendKind, p: u32, device: Option<u32>) -> BackendResult<Box<dyn Backend>> {
    if !kind.is_device() && device.is_some() {
        return Err(BackendError::Unsupported(format!(
            "--device does not apply to the {} backend",
            kind.name()
        )));
    }
    if let Some((lo, hi)) = kind.supported_exponents() {
        if p < lo || p > hi {
            return Err(BackendError::Unsupported(format!(
                "exponent {p} is outside the {} backend range {lo}..={hi}",
                kind.name()
            )));
        }
    }
    match kind {
        BackendKind::Cpu => Ok(Box::new(crate::backend::cpu::CpuBackend::new(p)?)),
        BackendKind::Dwt => Ok(Box::new(crate::backend::dwt::DwtBackend::new(p)?)),
        BackendKind::Gpu => open_gpu(p, device),
        BackendKind::Cuda => open_cuda(p, device),
    }
}

#[cfg(feature = "gpu")]
fn open_gpu(p: u32, device: Option<u32>) -> BackendResult<Box<dyn Backend>> {
    use crate::backend::gpu::{GpuBackend, GpuOptions};
    Ok(Box::new(GpuBackend::new(
        p,
        &GpuOptions {
            device,
            ..GpuOptions::default()
        },
    )?))
}

#[cfg(not(feature = "gpu"))]
fn open_gpu(_p: u32, _device: Option<u32>) -> BackendResult<Box<dyn Backend>> {
    Err(BackendError::Unsupported(
        "this binary was built without GPU support; rebuild with `cargo build --release --features gpu`".into(),
    ))
}

#[cfg(feature = "cuda")]
fn open_cuda(p: u32, device: Option<u32>) -> BackendResult<Box<dyn Backend>> {
    use crate::backend::cuda::{CudaBackend, CudaOptions};
    Ok(Box::new(CudaBackend::new(
        p,
        &CudaOptions {
            device,
            ..CudaOptions::default()
        },
    )?))
}

#[cfg(not(feature = "cuda"))]
fn open_cuda(_p: u32, _device: Option<u32>) -> BackendResult<Box<dyn Backend>> {
    Err(BackendError::Unsupported(
        "this binary was built without CUDA support; rebuild with `cargo build --release --features cuda`".into(),
    ))
}

/// Forwarding impl so the state machines run on a selected backend.
impl Backend for Box<dyn Backend> {
    fn exponent(&self) -> u32 {
        (**self).exponent()
    }
    fn identity(&self) -> BackendIdentity {
        (**self).identity()
    }
    fn alloc(&mut self) -> BackendResult<Reg> {
        (**self).alloc()
    }
    fn free(&mut self, r: Reg) -> BackendResult<()> {
        (**self).free(r)
    }
    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        (**self).import(r, x)
    }
    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        (**self).export(r)
    }
    fn copy(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        (**self).copy(dst, src)
    }
    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        (**self).square(r, n)
    }
    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        (**self).ll_step(r, n)
    }
    fn mul(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        (**self).mul(dst, src)
    }
    fn sub(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        (**self).sub(dst, src)
    }
    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        (**self).equal(a, b)
    }
    fn sync(&mut self) -> BackendResult<()> {
        (**self).sync()
    }
    fn diagnostics(&mut self) -> BackendResult<Vec<(String, String)>> {
        (**self).diagnostics()
    }
}
