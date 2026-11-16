//! Arithmetic backend contract.
//!
//! The LL, PRP, proof, and checkpoint state machines are written once, in Rust,
//! against this trait. A backend only does modular arithmetic modulo
//! N = 2^p - 1 on resident residues. It never owns the iteration count, the
//! check boundaries, the checkpoint decisions, or the result classification.
//!
//! Implementations:
//! - `cpu`: exact host bigint arithmetic with Mersenne folding, always available.
//! - `dwt`: f64 irrational-base discrete weighted transform on the host, pure
//!   Rust, with a roundoff check that reports `BackendError::Fault`.
//! - `gpu`: exact integer NTT over GF(M31^2) on the GPU via `wgpu`, behind the
//!   `gpu` Cargo feature.
//! - `cuda`: NVIDIA GPU backend on `cudarc` with CUDA C kernels authored in
//!   this repository, behind the `cuda` Cargo feature.
//!
//! Representation rule: the only representation that crosses this trait is the
//! canonical [`Residue`] (little-endian 32-bit limbs, 0 <= x < N, unique zero).
//! Balanced mixed-radix words, transformed buffers, and device memory stay
//! inside the backend.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dwt;
#[cfg(feature = "gpu")]
pub mod gpu;

use crate::residue::Residue;

/// Handle of a resident residue owned by one backend instance.
///
/// A handle is only valid for the backend that returned it, until `free` or
/// until the backend is dropped. Using it anywhere else is `InvalidHandle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

/// Failure of a backend operation. None of these is a mathematical result:
/// a caller must never turn one into "composite".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// The exponent, plan, or operation is outside the documented range.
    Unsupported(String),
    /// A handle that this backend did not issue, or that was freed.
    InvalidHandle(u32),
    /// Host or device allocation failed.
    Allocation(String),
    /// A recoverable arithmetic or transfer fault that the backend detected
    /// itself (for example a failed checked read). State may be stale; the
    /// protocol must roll back to its last verified state.
    Fault(String),
    /// The device context is unusable for the rest of this process.
    Poisoned(String),
    /// Any other error reported by native code, with its code and message.
    Native { code: i32, message: String },
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Unsupported(m) => write!(f, "unsupported: {m}"),
            BackendError::InvalidHandle(h) => write!(f, "invalid residue handle {h}"),
            BackendError::Allocation(m) => write!(f, "allocation failed: {m}"),
            BackendError::Fault(m) => write!(f, "arithmetic or transfer fault: {m}"),
            BackendError::Poisoned(m) => write!(f, "backend context poisoned: {m}"),
            BackendError::Native { code, message } => {
                write!(f, "native error {code}: {message}")
            }
        }
    }
}

impl std::error::Error for BackendError {}

pub type BackendResult<T> = Result<T, BackendError>;

/// What produced the arithmetic. Recorded in checkpoints and results so that
/// state never silently crosses incompatible configurations.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BackendIdentity {
    /// "cpu" or "cuda".
    pub kind: String,
    /// Build identity: crate version plus, for native backends, the pinned
    /// upstream revision and the kernel source hash.
    pub build: String,
    /// Arithmetic plan: for example "bigint" or the FFT spec in use.
    pub plan: String,
    /// Device description, empty for the CPU backend.
    pub device: String,
}

/// Modular arithmetic modulo N = 2^p - 1 on resident residues.
///
/// Every method that returns `Ok` has completed, or has been queued such that
/// a later `sync`, `export`, or `equal` reports its failure. `export`, `equal`
/// and `sync` are the only operations required to synchronize with a device.
/// Batched methods (`square`, `ll_step`) must not transfer the residue to the
/// host or force a synchronization per iteration.
pub trait Backend {
    /// The prime exponent p of the modulus N = 2^p - 1.
    fn exponent(&self) -> u32;

    fn identity(&self) -> BackendIdentity;

    /// Allocate a resident residue. Its initial value is unspecified.
    fn alloc(&mut self) -> BackendResult<Reg>;

    fn free(&mut self, r: Reg) -> BackendResult<()>;

    /// Load a canonical residue. The residue's exponent must equal `exponent()`.
    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()>;

    /// Read back the complete canonical residue (never a fingerprint).
    fn export(&mut self, r: Reg) -> BackendResult<Residue>;

    /// dst <- src. `dst == src` is a no-op.
    fn copy(&mut self, dst: Reg, src: Reg) -> BackendResult<()>;

    /// r <- r^(2^n) mod N: n modular squarings. `n == 0` is a no-op.
    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()>;

    /// n Lucas-Lehmer steps: r <- r^2 - 2 mod N, n times.
    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()>;

    /// dst <- dst * src mod N. `dst` and `src` must be different handles.
    fn mul(&mut self, dst: Reg, src: Reg) -> BackendResult<()>;

    /// Whether the two residues are equal as integers modulo N. Compares the
    /// complete values, not a fingerprint.
    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool>;

    /// Wait for all queued work and report any deferred launch or completion
    /// error.
    fn sync(&mut self) -> BackendResult<()>;

    /// Backend diagnostics as key and value pairs (for example roundoff
    /// statistics). Informational: never a substitute for the protocol checks.
    fn diagnostics(&mut self) -> BackendResult<Vec<(String, String)>> {
        Ok(Vec::new())
    }
}
