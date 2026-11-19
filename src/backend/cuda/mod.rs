// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! CUDA backend: exact integer IBDWT over GF(q^2), q = 2^61 - 1, on one
//! NVIDIA device through the CUDA driver API and NVRTC, via `cudarc` with
//! dynamic loading (no CUDA SDK at build time).
//!
//! Evidence level: the arithmetic is validated by the Rust model in
//! [`model`] against `num-bigint`, and the kernel source in `kernels.cu` is
//! validated by host emulation (compiled as C++ on the CPU). This host code
//! has never been run on CUDA hardware. See docs/CUDA.md.
//!
//! Process-global state: the CUDA driver keeps one primary context per
//! device per process, and `cudarc` keeps the loaded libraries and symbols in
//! process-wide `OnceLock`s. Every `CudaBackend` retains the device's primary
//! context and owns its own stream, module, and buffers. Two live backends on
//! one device are safe (the driver reference counts the primary context), but
//! a sticky device error raised by either poisons both, and both then report
//! `Poisoned`. One engine per device is the supported configuration.

pub mod model;

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{CompileError, CompileOptions};
use sha3::{Digest, Sha3_256};

use super::{
    new_handle, Backend, BackendError, BackendIdentity, BackendResult, MemoryEstimate, Reg,
};
use crate::residue::Residue;
use model::{Arg, Buf, Kernel, Launch, Plan};

/// The CUDA C source of every kernel, compiled at context creation by NVRTC.
pub const KERNEL_SOURCE: &str = include_str!("kernels.cu");

/// Transform description reported by `probe()`.
pub const TRANSFORM: &str = "IBDWT NTT GF(M61^2), exact integer, power-of-two weights";

/// NVRTC options other than the architecture, which comes from the device
/// (`--gpu-architecture=compute_XY`).
pub const NVRTC_BASE_OPTIONS: &[&str] = &["--std=c++17"];

/// Threads per block. The kernels do not depend on it.
const BLOCK: u32 = 256;

/// SHA3-256 of [`KERNEL_SOURCE`], lowercase hex.
pub fn kernel_source_hash() -> String {
    Sha3_256::digest(KERNEL_SOURCE.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Inclusive exponent range with a transform plan; see [`model::supported_exponents`].
pub fn supported_exponents() -> (u32, u32) {
    model::supported_exponents()
}

fn plan_name(n: u64) -> String {
    format!("ntt-gf-m61sq n={n}")
}

/// Device buffers allocated once per backend, besides the registers.
const FIXED_BUFS: [Buf; 5] = [Buf::T, Buf::T2, Buf::Tw, Buf::Z, Buf::A];

/// Device memory for exponent p, without loading the driver. A register is
/// `n` words of 8 bytes. The fixed figure counts the two transform buffers
/// (2n words each), the twiddle table, and the two carry buffers (n words
/// each). The host staging copies of `import` and `export` are not counted.
pub fn memory_estimate(p: u32) -> BackendResult<MemoryEstimate> {
    let (lo, hi) = supported_exponents();
    let plan = model::plan(p).ok_or_else(|| {
        BackendError::Unsupported(format!(
            "exponent {p} is outside the CUDA range {lo}..={hi}"
        ))
    })?;
    let bytes = |b: Buf| 8 * plan.buf_len(b) as u64;
    Ok(MemoryEstimate {
        plan: plan_name(plan.n),
        bytes_per_register: bytes(Buf::X),
        fixed_bytes: FIXED_BUFS.iter().map(|&b| bytes(b)).sum(),
        on_device: true,
    })
}

/// The complete NVRTC option list for a device of compute capability major.minor.
pub fn nvrtc_options(major: i32, minor: i32) -> Vec<String> {
    let mut v: Vec<String> = NVRTC_BASE_OPTIONS.iter().map(|s| s.to_string()).collect();
    v.push(format!("--gpu-architecture=compute_{major}{minor}"));
    v
}

/// Squarings per captured CUDA graph (opt-in graph replay).
const GRAPH_ITERS: u64 = 32;

pub struct CudaOptions {
    /// CUDA device ordinal as listed by probe(). None: device 0.
    pub device: Option<u32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CudaDevice {
    pub id: u32,
    pub name: String,
    pub memory_bytes: u64,
    pub compute_capability: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CudaProbe {
    /// libcuda and NVRTC found and initialized.
    pub driver_loaded: bool,
    /// Enumeration is not validation.
    pub devices: Vec<CudaDevice>,
    pub transform: String,
    /// SHA3-256 hex of the embedded CUDA C source.
    pub kernel_source_hash: String,
    /// Missing libraries, versions, errors.
    pub notes: Vec<String>,
}

// ------------------------------------------------------------- errors ----

/// Map a CUDA driver result code to a backend error. Pure, so it is unit
/// tested without a driver.
pub fn classify(code: i32, message: String) -> BackendError {
    match code {
        // CUDA_ERROR_OUT_OF_MEMORY
        2 => BackendError::Allocation(message),
        // NOT_INITIALIZED, STUB_LIBRARY, DEVICE_UNAVAILABLE, NO_DEVICE,
        // INVALID_DEVICE, NO_BINARY_FOR_GPU, JIT_COMPILER_NOT_FOUND,
        // UNSUPPORTED_PTX_VERSION, SYSTEM_DRIVER_MISMATCH,
        // COMPAT_NOT_SUPPORTED_ON_DEVICE: this host cannot run the backend.
        3 | 34 | 46 | 100 | 101 | 209 | 221 | 222 | 803 | 804 => BackendError::Unsupported(message),
        // DEINITIALIZED, INVALID_CONTEXT, ECC_UNCORRECTABLE,
        // NVLINK_UNCORRECTABLE, ILLEGAL_ADDRESS, LAUNCH_TIMEOUT,
        // CONTEXT_IS_DESTROYED, ASSERT, HARDWARE_STACK_ERROR,
        // ILLEGAL_INSTRUCTION, MISALIGNED_ADDRESS, INVALID_ADDRESS_SPACE,
        // INVALID_PC, LAUNCH_FAILED: sticky or context destroying.
        4 | 201 | 214 | 220 | 700 | 702 | 709 | 710 | 714..=719 => BackendError::Poisoned(message),
        _ => BackendError::Native { code, message },
    }
}

fn driver_err(what: &str, e: DriverError) -> BackendError {
    let name = e
        .error_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let text = e
        .error_string()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    classify(e.0 as i32, format!("{what}: {name}: {text}"))
}

fn compile_err(e: CompileError) -> BackendError {
    match e {
        CompileError::CompileError {
            nvrtc,
            options,
            log,
        } => BackendError::Native {
            code: nvrtc.0 as i32,
            message: format!(
                "NVRTC compile failed with options {options:?}: {}",
                log.to_string_lossy()
            ),
        },
        CompileError::CreationError(n)
        | CompileError::GetLogError(n)
        | CompileError::GetPtxError(n)
        | CompileError::DestroyError(n) => BackendError::Native {
            code: n.0 as i32,
            message: format!("NVRTC: {n:?}"),
        },
    }
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

/// Whether libcuda and libnvrtc can be loaded, with the searched names when not.
/// `cudarc` panics on the first call into a missing library, so this runs first.
fn libraries_present() -> Result<(), String> {
    // SAFETY: `is_culib_present` only calls `libloading::Library::new` on the
    // candidate file names and drops the handle; it calls no symbol. Loading
    // runs the library's initializers, which the CUDA driver and NVRTC
    // libraries are designed to allow. No Rust invariant depends on it.
    let driver = unsafe { cudarc::driver::sys::is_culib_present() };
    // SAFETY: as above, for the NVRTC library.
    let nvrtc = unsafe { cudarc::nvrtc::sys::is_culib_present() };
    let mut missing = Vec::new();
    if !driver {
        missing.push(format!(
            "CUDA driver library not found (searched {:?})",
            cudarc::get_lib_name_candidates("cuda")
        ));
    }
    if !nvrtc {
        missing.push(format!(
            "NVRTC library not found (searched {:?})",
            cudarc::get_lib_name_candidates("nvrtc")
        ));
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing.join("; "))
    }
}

fn nvrtc_version() -> String {
    let (mut major, mut minor) = (0i32, 0i32);
    // SAFETY: called only after `libraries_present` succeeded; nvrtcVersion
    // writes two ints through valid, live pointers.
    let r = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    if r == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        format!("{major}.{minor}")
    } else {
        format!("unknown ({r:?})")
    }
}

fn driver_version() -> String {
    let mut v = 0i32;
    // SAFETY: called only after `libraries_present` succeeded; writes one int
    // through a valid, live pointer.
    let r = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut v) };
    if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        format!("{}.{}", v / 1000, (v % 1000) / 10)
    } else {
        format!("unknown ({r:?})")
    }
}

// -------------------------------------------------------------- probe ----

/// Never panics: a missing driver or NVRTC library is reported in `notes`.
pub fn probe() -> CudaProbe {
    let mut pr = CudaProbe {
        driver_loaded: false,
        devices: Vec::new(),
        transform: TRANSFORM.into(),
        kernel_source_hash: kernel_source_hash(),
        notes: vec![
            "never run on CUDA hardware: kernel arithmetic is validated only by host emulation"
                .into(),
        ],
    };
    let r = catch_unwind(|| -> Result<(Vec<CudaDevice>, Vec<String>), String> {
        libraries_present()?;
        let mut notes = vec![format!(
            "driver API {}, NVRTC {}",
            driver_version(),
            nvrtc_version()
        )];
        let count = CudaContext::device_count()
            .map_err(|e| driver_err("cuDeviceGetCount", e).to_string())?;
        let mut devices = Vec::new();
        for id in 0..count.max(0) as u32 {
            let info = CudaContext::new(id as usize).and_then(|c| {
                let (maj, min) = c.compute_capability()?;
                Ok(CudaDevice {
                    id,
                    name: c.name()?,
                    memory_bytes: c.total_mem()? as u64,
                    compute_capability: format!("{maj}.{min}"),
                })
            });
            match info {
                Ok(d) => devices.push(d),
                Err(e) => notes.push(format!("device {id}: {}", driver_err("query", e))),
            }
        }
        Ok((devices, notes))
    });
    match r {
        Ok(Ok((devices, notes))) => {
            pr.driver_loaded = true;
            pr.devices = devices;
            pr.notes.extend(notes);
        }
        Ok(Err(m)) => pr.notes.push(m),
        Err(p) => pr
            .notes
            .push(format!("CUDA library failure: {}", panic_message(p))),
    }
    pr
}

// ------------------------------------------------------------ backend ----

/// One device, one engine owner. All work goes to one stream, in order.
pub struct CudaBackend {
    plan: Plan,
    square_seq: Vec<Launch>,
    ll_seq: Vec<Launch>,
    mul_seq: Vec<Launch>,
    stream: Arc<CudaStream>,
    /// Opt-in graph replay (MERSENNE_FORGE_CUDA_GRAPHS=1): one captured
    /// graph of GRAPH_ITERS squarings (or LL steps) per register, replayed
    /// instead of relaunching each kernel. Declared before `funcs`, so the
    /// graphs are destroyed before the module that holds their kernels.
    /// UNTESTED: no NVIDIA device has run this.
    graphs: Option<HashMap<(u32, bool), CudaGraph>>,
    funcs: Vec<CudaFunction>,
    t: CudaSlice<u64>,
    t2: CudaSlice<u64>,
    tw: CudaSlice<u64>,
    z: CudaSlice<u64>,
    a: CudaSlice<u64>,
    regs: HashMap<u32, CudaSlice<u64>>,
    poisoned: Option<String>,
    device: String,
    build: String,
}

impl CudaBackend {
    pub fn new(p: u32, opts: &CudaOptions) -> BackendResult<CudaBackend> {
        let (lo, hi) = supported_exponents();
        let plan = model::plan(p).ok_or_else(|| {
            BackendError::Unsupported(format!(
                "exponent {p} is outside the CUDA range {lo}..={hi}"
            ))
        })?;
        let ordinal = opts.device.unwrap_or(0);
        match catch_unwind(AssertUnwindSafe(|| Self::open(plan, ordinal))) {
            Ok(r) => r,
            Err(p) => Err(BackendError::Unsupported(format!(
                "CUDA library failure: {}",
                panic_message(p)
            ))),
        }
    }

    fn open(plan: Plan, ordinal: u32) -> BackendResult<CudaBackend> {
        libraries_present().map_err(BackendError::Unsupported)?;
        let ctx = CudaContext::new(ordinal as usize)
            .map_err(|e| driver_err(&format!("open CUDA device {ordinal}"), e))?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| driver_err("compute capability", e))?;
        let name = ctx.name().map_err(|e| driver_err("device name", e))?;
        let options = nvrtc_options(major, minor);
        // ponytail: no kernel binary cache; NVRTC compiles at every context
        // creation. A cache must be keyed by the source hash, the options,
        // the architecture, and the NVRTC and driver versions.
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            KERNEL_SOURCE,
            CompileOptions {
                options: options.clone(),
                name: Some("mersenne_forge.cu".into()),
                ..Default::default()
            },
        )
        .map_err(compile_err)?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| driver_err("load module", e))?;
        let funcs = Kernel::ALL
            .iter()
            .map(|k| {
                module
                    .load_function(k.name())
                    .map_err(|e| driver_err(k.name(), e))
            })
            .collect::<BackendResult<Vec<_>>>()?;
        let stream = ctx
            .new_stream()
            .map_err(|e| driver_err("create stream", e))?;
        let zeros = |b: Buf| {
            stream
                .alloc_zeros::<u64>(plan.buf_len(b))
                .map_err(|e| driver_err("allocate", e))
        };
        let (t, t2, z, a) = (
            zeros(Buf::T)?,
            zeros(Buf::T2)?,
            zeros(Buf::Z)?,
            zeros(Buf::A)?,
        );
        let host_tw = plan.twiddles();
        let tw = stream
            .clone_htod(&host_tw)
            .map_err(|e| driver_err("upload twiddles", e))?;
        // The host table must outlive the copy.
        stream
            .synchronize()
            .map_err(|e| driver_err("upload twiddles", e))?;
        drop(host_tw);
        let build = format!(
            "{}; cuda kernels sha3-256 {}; nvrtc {} options [{}]; driver {}; no kernel cache",
            crate::build_identity(),
            kernel_source_hash(),
            nvrtc_version(),
            options.join(" "),
            driver_version()
        );
        Ok(CudaBackend {
            square_seq: plan.square_launches(0),
            ll_seq: plan.square_launches(-2),
            mul_seq: plan.mul_launches(),
            plan,
            stream,
            graphs: std::env::var("MERSENNE_FORGE_CUDA_GRAPHS")
                .is_ok_and(|v| v == "1")
                .then(HashMap::new),
            funcs,
            t,
            t2,
            tw,
            z,
            a,
            regs: HashMap::new(),
            poisoned: None,
            device: format!("{name} (CUDA {ordinal}, cc {major}.{minor})"),
            build,
        })
    }

    fn live(&self) -> BackendResult<()> {
        match &self.poisoned {
            Some(m) => Err(BackendError::Poisoned(m.clone())),
            None => Ok(()),
        }
    }

    /// Latch a poisoned context: every later operation reports it.
    fn note<T>(&mut self, r: BackendResult<T>) -> BackendResult<T> {
        if let Err(BackendError::Poisoned(m)) = &r {
            self.poisoned = Some(m.clone());
        }
        r
    }

    fn reg(&self, r: Reg) -> BackendResult<&CudaSlice<u64>> {
        self.regs.get(&r.0).ok_or(BackendError::InvalidHandle(r.0))
    }

    /// Queue one launch sequence on the stream. No synchronization.
    fn submit(&self, seq: &[Launch], x: Reg, src: Option<Reg>, times: u64) -> BackendResult<()> {
        let xs = self.reg(x)?;
        let ss = src.map(|s| self.reg(s)).transpose()?;
        for _ in 0..times {
            for l in seq {
                let mut b = self.stream.launch_builder(&self.funcs[l.kernel as usize]);
                for arg in &l.args {
                    match arg {
                        Arg::Buf(Buf::X) => b.arg(xs),
                        Arg::Buf(Buf::Src) => b.arg(ss.ok_or(BackendError::InvalidHandle(x.0))?),
                        Arg::Buf(Buf::T) => b.arg(&self.t),
                        Arg::Buf(Buf::T2) => b.arg(&self.t2),
                        Arg::Buf(Buf::Tw) => b.arg(&self.tw),
                        Arg::Buf(Buf::Z) => b.arg(&self.z),
                        Arg::Buf(Buf::A) => b.arg(&self.a),
                        Arg::U64(v) => b.arg(v),
                    };
                }
                let cfg = LaunchConfig {
                    grid_dim: (l.threads.div_ceil(u64::from(BLOCK)) as u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: (1) the argument list matches the kernel's CUDA
                // signature: `model::Kernel::params` is checked against every
                // launch by a unit test, and the host emulation test compiles
                // the same source against those argument kinds. Every pointer
                // is a `CudaSlice<u64>` and every scalar a `u64`, matching the
                // `u64*` and `unsigned long long` parameters. (2) Every buffer
                // has the length `Plan::buf_len` gives, which is what the
                // kernels index (n or 2n elements, guarded by the thread
                // index check). (3) All slices are owned by `self` and live
                // until `self` drops; their drop is ordered on this stream
                // (cudarc frees asynchronously on the owning stream or
                // synchronizes first), so queued kernels never see freed memory.
                // The module that holds the kernels is unloaded only after
                // `Drop for CudaBackend` has waited for the stream.
                unsafe { b.launch(cfg) }.map_err(|e| driver_err(l.kernel.name(), e))?;
            }
        }
        Ok(())
    }

    /// `n` squarings (or LL steps) of `r`: replayed from a captured graph
    /// when graphs are on, with the remainder launched directly.
    fn run_square(&mut self, ll: bool, r: Reg, n: u64) -> BackendResult<()> {
        let direct =
            |s: &Self, n: u64| s.submit(if ll { &s.ll_seq } else { &s.square_seq }, r, None, n);
        let Some(graphs) = &self.graphs else {
            return direct(self, n);
        };
        if n < GRAPH_ITERS {
            return direct(self, n);
        }
        let key = (r.0, ll);
        if !graphs.contains_key(&key) {
            use cudarc::driver::sys;
            self.stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| driver_err("begin graph capture", e))?;
            let captured = direct(self, GRAPH_ITERS);
            let graph = self
                .stream
                .end_capture(sys::CUgraphInstantiate_flags(0))
                .map_err(|e| driver_err("end graph capture", e));
            captured?;
            match graph? {
                Some(g) => {
                    self.graphs.as_mut().expect("graphs are on").insert(key, g);
                }
                None => return direct(self, n),
            }
        }
        let g = &self.graphs.as_ref().expect("graphs are on")[&key];
        for _ in 0..n / GRAPH_ITERS {
            g.launch().map_err(|e| driver_err("graph launch", e))?;
        }
        direct(self, n % GRAPH_ITERS)
    }

    /// Graphs hold the device addresses of one register: drop them when the
    /// register is replaced or freed.
    fn forget_graphs(&mut self, r: u32) {
        if let Some(g) = self.graphs.as_mut() {
            g.retain(|&(reg, _), _| reg != r);
        }
    }

    fn read_words(&self, r: Reg) -> BackendResult<Vec<u64>> {
        let words = self
            .stream
            .clone_dtoh(self.reg(r)?)
            .map_err(|e| driver_err("copy to host", e))?;
        // The copy is complete before the host reads the words.
        self.stream
            .synchronize()
            .map_err(|e| driver_err("synchronize", e))?;
        Ok(words)
    }

    fn read_residue(&mut self, r: Reg) -> BackendResult<Residue> {
        let words = self.read_words(r);
        let words = self.note(words)?;
        self.plan
            .from_words(&words)
            .map_err(|e| BackendError::Fault(format!("export: {e}")))
    }
}

impl Drop for CudaBackend {
    /// Wait for queued kernels before the fields drop: dropping `funcs`
    /// releases the module, and `cudarc` unloads it immediately. A failure here
    /// cannot be reported and is ignored; the buffers still free on the stream.
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl Backend for CudaBackend {
    fn exponent(&self) -> u32 {
        self.plan.p
    }

    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            kind: "cuda".into(),
            build: self.build.clone(),
            plan: if self.graphs.is_some() {
                format!("{} graphs", plan_name(self.plan.n))
            } else {
                plan_name(self.plan.n)
            },
            device: self.device.clone(),
        }
    }

    fn alloc(&mut self) -> BackendResult<Reg> {
        self.live()?;
        let r = self
            .stream
            .alloc_zeros::<u64>(self.plan.buf_len(Buf::X))
            .map_err(|e| driver_err("allocate residue", e));
        let slice = self.note(r)?;
        let h = new_handle()?;
        self.regs.insert(h.0, slice);
        Ok(h)
    }

    fn free(&mut self, r: Reg) -> BackendResult<()> {
        self.forget_graphs(r.0);
        self.regs
            .remove(&r.0)
            .map(|_| ())
            .ok_or(BackendError::InvalidHandle(r.0))
    }

    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        self.live()?;
        if x.exponent() != self.plan.p {
            return Err(BackendError::Unsupported(format!(
                "residue for exponent {} imported into backend for {}",
                x.exponent(),
                self.plan.p
            )));
        }
        self.reg(r)?;
        let words = self.plan.to_words(x);
        let mut dst = self
            .regs
            .remove(&r.0)
            .ok_or(BackendError::InvalidHandle(r.0))?;
        let res = self
            .stream
            .memcpy_htod(&words, &mut dst)
            .and_then(|_| self.stream.synchronize())
            .map_err(|e| driver_err("copy to device", e));
        // `words` stays alive until the synchronize above has returned.
        self.forget_graphs(r.0);
        self.regs.insert(r.0, dst);
        drop(words);
        self.note(res)
    }

    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        self.live()?;
        self.read_residue(r)
    }

    fn copy(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        self.live()?;
        self.reg(dst)?;
        self.reg(src)?;
        if dst == src {
            return Ok(());
        }
        let mut d = self
            .regs
            .remove(&dst.0)
            .ok_or(BackendError::InvalidHandle(dst.0))?;
        let res = self
            .stream
            .memcpy_dtod(self.reg(src)?, &mut d)
            .map_err(|e| driver_err("device copy", e));
        self.forget_graphs(dst.0);
        self.regs.insert(dst.0, d);
        self.note(res)
    }

    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.live()?;
        self.reg(r)?;
        let res = self.run_square(false, r, n);
        self.note(res)
    }

    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.live()?;
        self.reg(r)?;
        let res = self.run_square(true, r, n);
        self.note(res)
    }

    fn mul(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        self.live()?;
        self.reg(dst)?;
        self.reg(src)?;
        if dst == src {
            return Err(BackendError::Unsupported(
                "mul needs distinct handles".into(),
            ));
        }
        let res = self.submit(&self.mul_seq, dst, Some(src), 1);
        self.note(res)
    }

    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        self.live()?;
        let x = self.read_residue(a)?;
        let y = self.read_residue(b)?;
        Ok(x == y)
    }

    fn sync(&mut self) -> BackendResult<()> {
        self.live()?;
        let res = self
            .stream
            .synchronize()
            .and_then(|_| self.stream.context().check_err())
            .map_err(|e| driver_err("synchronize", e));
        self.note(res)
    }

    fn diagnostics(&mut self) -> BackendResult<Vec<(String, String)>> {
        let pl = &self.plan;
        Ok(vec![
            ("transform_length".into(), pl.n.to_string()),
            ("word_width".into(), format!("{}..{}", pl.w_min, pl.w_max)),
            ("carry_pieces".into(), pl.pieces.to_string()),
            ("word_bound".into(), pl.word_bound.to_string()),
            ("coefficient_bound".into(), pl.coeff_bound.to_string()),
            (
                "coefficient_limit".into(),
                model::CAPACITY_LIMIT.to_string(),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_mapping() {
        assert!(matches!(
            classify(2, "oom".into()),
            BackendError::Allocation(_)
        ));
        for c in [100, 101, 34, 209, 222, 803] {
            assert!(
                matches!(classify(c, String::new()), BackendError::Unsupported(_)),
                "{c}"
            );
        }
        for c in [700, 702, 709, 714, 715, 716, 717, 718, 719, 201, 4] {
            assert!(
                matches!(classify(c, String::new()), BackendError::Poisoned(_)),
                "{c}"
            );
        }
        assert_eq!(
            classify(701, "x".into()),
            BackendError::Native {
                code: 701,
                message: "x".into()
            }
        );
        assert_eq!(
            classify(999, "x".into()),
            BackendError::Native {
                code: 999,
                message: "x".into()
            }
        );
    }

    #[test]
    fn out_of_range_exponent_is_unsupported_without_a_driver() {
        let (lo, hi) = supported_exponents();
        for p in [2, 7, lo - 1, hi + 1] {
            let e = CudaBackend::new(p, &CudaOptions { device: None })
                .err()
                .unwrap();
            assert!(
                matches!(e, BackendError::Unsupported(ref m) if m.contains("outside")),
                "{p}: {e}"
            );
        }
    }

    /// The device buffers cannot be measured without a driver; the estimate
    /// is checked against the lengths `open` and `alloc` pass to the driver.
    #[test]
    fn memory_estimate_matches_the_buffer_lengths() {
        let (lo, hi) = supported_exponents();
        for p in [lo - 1, hi + 1] {
            assert!(matches!(
                memory_estimate(p),
                Err(BackendError::Unsupported(_))
            ));
        }
        assert!(memory_estimate(hi).is_ok());
        for p in [lo, 127, 86243, 1_257_787] {
            let m = memory_estimate(p).unwrap();
            let pl = model::plan(p).unwrap();
            let n = pl.n;
            assert_eq!(m.plan, format!("ntt-gf-m61sq n={n}"));
            assert!(m.on_device);
            assert_eq!(m.bytes_per_register, 8 * n);
            assert_eq!(pl.twiddles().len(), pl.buf_len(Buf::Tw), "p={p}");
            // T and T2 hold 2n words, twiddles, Z, and A hold n words.
            assert_eq!(m.fixed_bytes, 8 * (2 * n + 2 * n + 3 * n), "p={p}");
        }
    }

    #[test]
    fn source_identity() {
        assert_eq!(kernel_source_hash().len(), 64);
        assert!(!KERNEL_SOURCE.contains("#include"));
        for k in Kernel::ALL {
            assert!(
                KERNEL_SOURCE.contains(&format!("extern \"C\" __global__ void {}(", k.name())),
                "{k:?}"
            );
        }
        assert_eq!(
            nvrtc_options(8, 9),
            vec![
                "--std=c++17".to_string(),
                "--gpu-architecture=compute_89".into()
            ]
        );
    }
}
