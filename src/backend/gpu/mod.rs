// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! GPU backend: exact NTT arithmetic over GF(q^2), q = 2^31 - 1, in WGSL
//! compute shaders run through `wgpu` (Metal, Vulkan, or DX12).
//!
//! The mathematics, the capacity bound, and the carry proof are documented in
//! [`model`], which is also the host model the kernels are tested against.
//! Residues stay on the device as balanced words; only `export` and `equal`
//! read them back, and only `import` writes them.

pub mod model;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use sha3::{Digest, Sha3_256};
use wgpu::util::DeviceExt;

use super::{
    new_handle, Backend, BackendError, BackendIdentity, BackendResult, MemoryEstimate, Reg,
};
use crate::residue::Residue;
use model::Plan;

/// The embedded WGSL source. Its SHA3-256 is part of the backend identity.
pub const KERNELS: &str = include_str!("kernels.wgsl");

const WG: u32 = 256;
/// Work budget per submission, in transform elements times dispatches. On the
/// Apple M2 this is a few tens of milliseconds of GPU time.
const SUBMIT_BUDGET: u64 = 1 << 26;
/// Fewest and most iterations encoded in one command buffer. At n = 2^22 the
/// floor gives submissions of roughly 0.25 s on the Apple M2.
const MIN_ITERS_PER_SUBMIT: u64 = 4;
const MAX_ITERS_PER_SUBMIT: u64 = 256;
/// Submissions allowed in flight before the host waits for the oldest.
const MAX_IN_FLIGHT: usize = 2;

#[derive(Debug, Clone, Default)]
pub struct GpuOptions {
    /// Adapter index as listed by probe(). None: the first hardware adapter.
    pub device: Option<u32>,
    /// Fused radix 4 stages (also enabled by MERSENNE_FORGE_GPU_RADIX4=1).
    pub radix4: bool,
    /// Two words per element and a half length transform (also enabled by
    /// MERSENNE_FORGE_GPU_PACKED=1); see `weight_packed` in the kernels.
    pub packed: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuDevice {
    pub id: u32,
    pub name: String,
    pub api: String,
    pub device_type: String,
    pub driver: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuProbe {
    pub devices: Vec<GpuDevice>,
    pub transform: String,
    pub kernel_source_hash: String,
    pub notes: Vec<String>,
}

/// Inclusive range of exponents with a plan (see [`model::supported_range`]).
pub fn supported_exponents() -> (u32, u32) {
    model::supported_range()
}

/// Whether fused radix 4 stages were asked for (MERSENNE_FORGE_GPU_RADIX4=1).
fn radix4_requested() -> bool {
    std::env::var("MERSENNE_FORGE_GPU_RADIX4").is_ok_and(|v| v == "1")
}

/// Whether the packed transform was asked for (MERSENNE_FORGE_GPU_PACKED=1).
fn packed_requested() -> bool {
    std::env::var("MERSENNE_FORGE_GPU_PACKED").is_ok_and(|v| v == "1")
}

fn plan_name(n: u32) -> String {
    format!("ntt-gf-m31sq n={n}")
}

/// Uniform parameter blocks of 16 bytes: the base and LL blocks plus one per
/// forward and one per inverse transform stage.
fn uniform_bytes(log_n: u32) -> u64 {
    16 * (2 + 2 * u64::from(log_n))
}

/// Device memory for exponent p, without opening a device. A register is
/// `n` balanced `i32` words. The fixed figure counts the two transform
/// buffers (8n bytes each), the two carry buffers (4n each), the twiddle
/// table (n / 2 elements of 8 bytes), the word table (4n), the fault flags,
/// the uniform blocks, and the 4n readback staging buffer that `export`
/// creates for the time of one read.
pub fn memory_estimate(p: u32) -> BackendResult<MemoryEstimate> {
    let (lo, hi) = supported_exponents();
    let n = model::choose_n(p).ok_or_else(|| {
        BackendError::Unsupported(format!(
            "exponent {p} is outside the GPU backend range {lo}..={hi}"
        ))
    })?;
    let n64 = u64::from(n);
    Ok(MemoryEstimate {
        plan: plan_name(n),
        bytes_per_register: 4 * n64,
        fixed_bytes: 2 * 8 * n64
            + 2 * 4 * n64
            + 4 * n64
            + 4 * n64
            + 16
            + uniform_bytes(n.trailing_zeros())
            + 4 * n64,
        on_device: true,
    })
}

pub fn kernel_source_hash() -> String {
    Sha3_256::digest(KERNELS.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const BACKENDS: wgpu::Backends = wgpu::Backends::METAL
    .union(wgpu::Backends::VULKAN)
    .union(wgpu::Backends::DX12);

fn adapters() -> Vec<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: BACKENDS,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    pollster::block_on(instance.enumerate_adapters(BACKENDS))
}

fn describe(id: u32, a: &wgpu::Adapter) -> GpuDevice {
    let i = a.get_info();
    GpuDevice {
        id,
        name: i.name,
        api: format!("{:?}", i.backend),
        device_type: format!("{:?}", i.device_type),
        driver: format!("{} {}", i.driver, i.driver_info).trim().to_string(),
    }
}

/// Enumerate adapters. Enumeration is not validation.
pub fn probe() -> GpuProbe {
    let devices = adapters()
        .iter()
        .zip(0..)
        .map(|(a, i)| describe(i, a))
        .collect();
    let (lo, hi) = supported_exponents();
    GpuProbe {
        devices,
        transform: "NTT GF(M31^2)".into(),
        kernel_source_hash: kernel_source_hash(),
        notes: vec![
            format!("supported exponents {lo}..={hi}"),
            "validated only on an Apple GPU through Metal; Vulkan and DX12 are untested".into(),
            "software (Cpu type) adapters are used only when selected by index".into(),
        ],
    }
}

/// Bind group resources shared by all kernels.
struct Shared {
    device: wgpu::Device,
    layout: wgpu::BindGroupLayout,
    t: wgpu::Buffer,
    t2: wgpu::Buffer,
    tw: wgpu::Buffer,
    wt: wgpu::Buffer,
    err: wgpu::Buffer,
}

impl Shared {
    fn bind(
        &self,
        params: &wgpu::Buffer,
        win: &wgpu::Buffer,
        wout: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bufs = [
            params, &self.t, &self.t2, &self.tw, win, wout, &self.wt, &self.err,
        ];
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .zip(0..)
            .map(|(b, i)| wgpu::BindGroupEntry {
                binding: i,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        })
    }

    fn params(&self, v: [u32; 4]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }
}

struct Pipes {
    weight: wgpu::ComputePipeline,
    dif: wgpu::ComputePipeline,
    dit: wgpu::ComputePipeline,
    dif4: wgpu::ComputePipeline,
    dit4: wgpu::ComputePipeline,
    sqr: wgpu::ComputePipeline,
    pmul: wgpu::ComputePipeline,
    unweight: wgpu::ComputePipeline,
    weight_packed: wgpu::ComputePipeline,
    sqr_packed: wgpu::ComputePipeline,
    pmul_packed: wgpu::ComputePipeline,
    unweight_packed: wgpu::ComputePipeline,
    carry: wgpu::ComputePipeline,
    #[cfg_attr(not(test), allow(dead_code))]
    mulmod_test: wgpu::ComputePipeline,
}

struct RegBuf {
    buf: wgpu::Buffer,
    /// Reads this register into the transform buffer.
    load: wgpu::BindGroup,
    /// Last carry pass, writing this register.
    finish: wgpu::BindGroup,
}

pub struct GpuBackend {
    plan: Plan,
    sh: Shared,
    queue: wgpu::Queue,
    pipes: Pipes,
    s: [wgpu::Buffer; 2],
    base: wgpu::Buffer,
    bg_base: wgpu::BindGroup,
    bg_ll: wgpu::BindGroup,
    bg_dif: Vec<wgpu::BindGroup>,
    bg_dit: Vec<wgpu::BindGroup>,
    /// Opt-in fused stages: (radix 4, bind group) per pass, forward and
    /// inverse. None: one pass per stage.
    fused: Option<(Vec<(bool, wgpu::BindGroup)>, Vec<(bool, wgpu::BindGroup)>)>,
    /// Packed transform: the NTT runs at length n / 2.
    packed: bool,
    bg_carry: [wgpu::BindGroup; 2],
    regs: HashMap<u32, RegBuf>,
    /// First uncaptured device error or device loss. Sticky.
    sticky: Arc<Mutex<Option<BackendError>>>,
    in_flight: VecDeque<wgpu::SubmissionIndex>,
    max_groups: u32,
    iters_per_submit: u64,
    device_name: String,
}

fn lock(m: &Mutex<Option<BackendError>>) -> std::sync::MutexGuard<'_, Option<BackendError>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl GpuBackend {
    pub fn new(p: u32, opts: &GpuOptions) -> BackendResult<GpuBackend> {
        let (lo, hi) = supported_exponents();
        let plan = Plan::new(p).ok_or_else(|| {
            BackendError::Unsupported(format!(
                "exponent {p} is outside the GPU backend range {lo}..={hi}"
            ))
        })?;
        let mut all = adapters();
        let adapter = match opts.device {
            Some(i) => {
                if (i as usize) >= all.len() {
                    return Err(BackendError::Unsupported(format!(
                        "GPU adapter index {i} does not exist ({} found)",
                        all.len()
                    )));
                }
                all.swap_remove(i as usize)
            }
            None => all
                .into_iter()
                .find(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
                .ok_or_else(|| {
                    BackendError::Unsupported(
                        "no hardware GPU adapter found through Metal, Vulkan, or DX12 \
                         (software adapters are used only when selected by index)"
                            .into(),
                    )
                })?,
        };
        let info = adapter.get_info();
        let device_name = format!("{} ({:?})", info.name, info.backend);
        let limits = adapter.limits();
        let n = u64::from(plan.n);
        let need = 8 * n;
        if need > limits.max_storage_buffer_binding_size
            || need > limits.max_buffer_size
            || limits.max_storage_buffers_per_shader_stage < 7
            || limits.max_compute_invocations_per_workgroup < WG
            || limits.max_compute_workgroup_size_x < WG
        {
            return Err(BackendError::Unsupported(format!(
                "adapter {device_name} limits are too small for n = {n}"
            )));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mersenne-forge"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| BackendError::Unsupported(format!("cannot open {device_name}: {e}")))?;

        let sticky = Arc::new(Mutex::new(None));
        let s2 = sticky.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let err = match e {
                wgpu::Error::OutOfMemory { .. } => BackendError::Allocation(e.to_string()),
                _ => BackendError::Poisoned(format!("device error: {e}")),
            };
            lock(&s2).get_or_insert(err);
        }));
        let s3 = sticky.clone();
        device.set_device_lost_callback(move |reason, msg| {
            lock(&s3).get_or_insert(BackendError::Poisoned(format!(
                "device lost ({reason:?}): {msg}"
            )));
        });

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kernels"),
            source: wgpu::ShaderSource::Wgsl(KERNELS.into()),
        });
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let types = [
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            storage(false),
            storage(false),
            storage(true),
            storage(false),
            storage(false),
            storage(true),
            storage(false),
        ];
        let entries: Vec<wgpu::BindGroupLayoutEntry> = types
            .iter()
            .zip(0..)
            .map(|(&ty, binding)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = Pipes {
            weight: pipe("weight"),
            dif: pipe("dif"),
            dit: pipe("dit"),
            dif4: pipe("dif4"),
            dit4: pipe("dit4"),
            sqr: pipe("sqr"),
            pmul: pipe("pmul"),
            unweight: pipe("unweight"),
            weight_packed: pipe("weight_packed"),
            sqr_packed: pipe("sqr_packed"),
            pmul_packed: pipe("pmul_packed"),
            unweight_packed: pipe("unweight_packed"),
            carry: pipe("carry"),
            mulmod_test: pipe("mulmod_test"),
        };
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(BackendError::Unsupported(format!(
                "kernels do not compile on {device_name}: {e}"
            )));
        }

        let storage_buf = |label, size: u64, extra| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let init = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let copy = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let tw: Vec<u32> = model::twiddles(plan.n)
            .iter()
            .flat_map(|w| [w.re, w.im])
            .collect();
        let (t, t2) = (
            storage_buf("t", 8 * n, copy),
            storage_buf("t2", 8 * n, copy),
        );
        let s = [
            storage_buf("s0", 4 * n, copy),
            storage_buf("s1", 4 * n, copy),
        ];
        let tw = init("twiddles", bytemuck::cast_slice(&tw));
        let wt = init("word table", bytemuck::cast_slice(&plan.word_table()));
        let err = init("error flags", bytemuck::cast_slice(&[0u32; 4]));
        let sh = Shared {
            device,
            layout,
            t,
            t2,
            tw,
            wt,
            err,
        };
        let base = sh.params([plan.n, 0, 0, 0]);
        let ll = sh.params([plan.n, 0, 0, 2]);
        // Transform length: n, or n / 2 when packed. The twiddle stride
        // stays n / (2 m): a length L stage with half block m uses w_L^j =
        // w_n^(j n / L), and the table holds powers of w_n.
        let packed = opts.packed || packed_requested();
        let tlen = if packed { plan.n / 2 } else { plan.n };
        let mut bg_dif = Vec::new();
        let mut bg_dit = Vec::new();
        let mut m = tlen / 2;
        while m >= 1 {
            let params = sh.params([tlen, m, plan.n / (2 * m), 0]);
            bg_dif.push(sh.bind(&params, &s[1], &s[0]));
            m /= 2;
        }
        let mut m = 1;
        while m < tlen {
            let params = sh.params([tlen, m, plan.n / (2 * m), 0]);
            bg_dit.push(sh.bind(&params, &s[1], &s[0]));
            m *= 2;
        }
        // Fused stages: radix 4 passes from the first stage on, and one radix
        // 2 pass when the stage count is odd (the last DIF stage, the last
        // DIT stage).
        let fused = (opts.radix4 || radix4_requested()).then(|| {
            let mut f = Vec::new();
            let mut m = tlen / 2;
            while m >= 1 {
                let params = sh.params([tlen, m, plan.n / (2 * m), 0]);
                let four = m >= 2;
                f.push((four, sh.bind(&params, &s[1], &s[0])));
                m /= if four { 4 } else { 2 };
            }
            let mut i = Vec::new();
            let mut h = 1;
            while h < tlen {
                let params = sh.params([tlen, h, plan.n / (2 * h), 0]);
                let four = 4 * h <= tlen;
                i.push((four, sh.bind(&params, &s[1], &s[0])));
                h *= if four { 4 } else { 2 };
            }
            (f, i)
        });
        let max_groups = limits.max_compute_workgroups_per_dimension.max(1);
        let work = n * u64::from(2 * plan.log_n + plan.passes + 3);
        let g = GpuBackend {
            bg_base: sh.bind(&base, &s[1], &s[0]),
            bg_ll: sh.bind(&ll, &s[1], &s[0]),
            bg_carry: [sh.bind(&base, &s[0], &s[1]), sh.bind(&base, &s[1], &s[0])],
            bg_dif,
            bg_dit,
            fused,
            packed,
            base,
            s,
            sh,
            queue,
            pipes,
            regs: HashMap::new(),
            sticky,
            in_flight: VecDeque::new(),
            max_groups,
            iters_per_submit: (SUBMIT_BUDGET / work)
                .clamp(MIN_ITERS_PER_SUBMIT, MAX_ITERS_PER_SUBMIT),
            device_name,
            plan,
        };
        g.check()?;
        Ok(g)
    }

    fn check(&self) -> BackendResult<()> {
        match lock(&self.sticky).clone() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn reg(&self, r: Reg) -> BackendResult<&RegBuf> {
        self.regs.get(&r.0).ok_or(BackendError::InvalidHandle(r.0))
    }

    fn dispatch(
        &self,
        pass: &mut wgpu::ComputePass,
        pipe: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        count: u32,
    ) {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(count.div_ceil(WG).clamp(1, self.max_groups), 1, 1);
    }

    /// Weight `load`'s register into T and transform it.
    fn encode_forward(&self, pass: &mut wgpu::ComputePass, load: &wgpu::BindGroup) {
        let n = self.tlen();
        if self.packed {
            self.dispatch(pass, &self.pipes.weight_packed, load, n);
        } else {
            self.dispatch(pass, &self.pipes.weight, load, n);
        }
        if let Some((fwd, _)) = &self.fused {
            for (four, bg) in fwd {
                if *four {
                    self.dispatch(pass, &self.pipes.dif4, bg, n / 4);
                } else {
                    self.dispatch(pass, &self.pipes.dif, bg, n / 2);
                }
            }
            return;
        }
        for bg in &self.bg_dif {
            self.dispatch(pass, &self.pipes.dif, bg, n / 2);
        }
    }

    /// Inverse transform T, unweight, carry, and store into `dst`.
    /// Transform length: n, or n / 2 when packed.
    fn tlen(&self) -> u32 {
        if self.packed {
            self.plan.n / 2
        } else {
            self.plan.n
        }
    }

    /// Pointwise square of T, or product with T2.
    fn encode_pointwise(&self, pass: &mut wgpu::ComputePass, product: bool) {
        let (pipe, count) = match (self.packed, product) {
            (false, false) => (&self.pipes.sqr, self.plan.n),
            (false, true) => (&self.pipes.pmul, self.plan.n),
            (true, false) => (&self.pipes.sqr_packed, self.plan.n / 4 + 1),
            (true, true) => (&self.pipes.pmul_packed, self.plan.n / 4 + 1),
        };
        self.dispatch(pass, pipe, &self.bg_base, count);
    }

    fn encode_back(&self, pass: &mut wgpu::ComputePass, unweight: &wgpu::BindGroup, dst: &RegBuf) {
        let n = self.tlen();
        match &self.fused {
            Some((_, inv)) => {
                for (four, bg) in inv {
                    if *four {
                        self.dispatch(pass, &self.pipes.dit4, bg, n / 4);
                    } else {
                        self.dispatch(pass, &self.pipes.dit, bg, n / 2);
                    }
                }
            }
            None => {
                for bg in &self.bg_dit {
                    self.dispatch(pass, &self.pipes.dit, bg, n / 2);
                }
            }
        }
        let n = self.plan.n;
        if self.packed {
            self.dispatch(pass, &self.pipes.unweight_packed, unweight, n / 2);
        } else {
            self.dispatch(pass, &self.pipes.unweight, unweight, n);
        }
        for t in 1..self.plan.passes {
            self.dispatch(
                pass,
                &self.pipes.carry,
                &self.bg_carry[(t as usize - 1) % 2],
                n,
            );
        }
        self.dispatch(pass, &self.pipes.carry, &dst.finish, n);
    }

    fn submit(&mut self, cmd: wgpu::CommandBuffer) -> BackendResult<()> {
        self.in_flight.push_back(self.queue.submit(Some(cmd)));
        while self.in_flight.len() > MAX_IN_FLIGHT {
            let idx = self.in_flight.pop_front().unwrap();
            self.wait(Some(idx))?;
        }
        self.check()
    }

    fn wait(&self, idx: Option<wgpu::SubmissionIndex>) -> BackendResult<()> {
        self.sh
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: idx,
                timeout: None,
            })
            .map_err(|e| BackendError::Poisoned(format!("device poll failed: {e}")))?;
        self.check()
    }

    fn repeat_square(&mut self, r: Reg, n: u64, ll: bool) -> BackendResult<()> {
        self.check()?;
        self.reg(r)?;
        let mut left = n;
        while left > 0 {
            let k = left.min(self.iters_per_submit);
            let mut enc = self.sh.device.create_command_encoder(&Default::default());
            {
                let reg = self.reg(r)?;
                let mut pass = enc.begin_compute_pass(&Default::default());
                for _ in 0..k {
                    self.encode_forward(&mut pass, &reg.load);
                    self.encode_pointwise(&mut pass, false);
                    self.encode_back(&mut pass, if ll { &self.bg_ll } else { &self.bg_base }, reg);
                }
            }
            self.submit(enc.finish())?;
            left -= k;
        }
        Ok(())
    }

    fn read(&mut self, src: &wgpu::Buffer, size: u64) -> BackendResult<Vec<u8>> {
        let staging = self.sh.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.sh.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        let idx = self.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |res| {
            let _ = tx.send(res);
        });
        self.wait(Some(idx))?;
        self.in_flight.clear();
        match rx.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(BackendError::Fault(format!("readback map failed: {e}"))),
            Err(_) => return Err(BackendError::Fault("readback did not complete".into())),
        }
        let data = staging
            .get_mapped_range(..)
            .map_err(|e| BackendError::Fault(format!("readback view failed: {e}")))?
            .to_vec();
        staging.unmap();
        Ok(data)
    }

    /// Report and clear the kernel fault flag (nonzero imaginary part).
    fn check_fault_flag(&mut self) -> BackendResult<()> {
        let err = self.sh.err.clone();
        let flags = self.read(&err, 16)?;
        if flags.iter().any(|&b| b != 0) {
            self.queue.write_buffer(&self.sh.err, 0, &[0u8; 16]);
            return Err(BackendError::Fault(
                "a GPU transform produced a nonzero imaginary part".into(),
            ));
        }
        self.check()
    }
}

impl Backend for GpuBackend {
    fn exponent(&self) -> u32 {
        self.plan.p
    }

    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            kind: "gpu".into(),
            build: format!(
                "{} wgsl-sha3-256 {}",
                crate::build_identity(),
                kernel_source_hash()
            ),
            plan: {
                let mut name = plan_name(self.plan.n);
                if self.packed {
                    name.push_str(" packed");
                }
                if self.fused.is_some() {
                    name.push_str(" radix4");
                }
                name
            },
            device: self.device_name.clone(),
        }
    }

    fn alloc(&mut self) -> BackendResult<Reg> {
        self.check()?;
        let buf = self.sh.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("residue"),
            size: 4 * u64::from(self.plan.n),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let last = &self.s[(self.plan.passes as usize - 1) % 2];
        let reg = RegBuf {
            load: self.sh.bind(&self.base, &buf, &self.s[0]),
            finish: self.sh.bind(&self.base, last, &buf),
            buf,
        };
        let r = new_handle()?;
        self.regs.insert(r.0, reg);
        Ok(r)
    }

    fn free(&mut self, r: Reg) -> BackendResult<()> {
        self.regs
            .remove(&r.0)
            .map(|_| ())
            .ok_or(BackendError::InvalidHandle(r.0))
    }

    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        self.check()?;
        if x.exponent() != self.plan.p {
            return Err(BackendError::Unsupported(format!(
                "residue exponent {} does not match backend exponent {}",
                x.exponent(),
                self.plan.p
            )));
        }
        let words = self.plan.words_from_residue(x);
        let reg = self.reg(r)?;
        self.queue
            .write_buffer(&reg.buf, 0, bytemuck::cast_slice(&words));
        Ok(())
    }

    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        self.check()?;
        let buf = self.reg(r)?.buf.clone();
        let bytes = self.read(&buf, 4 * u64::from(self.plan.n))?;
        self.check_fault_flag()?;
        let words: Vec<i32> = bytemuck::pod_collect_to_vec(&bytes);
        self.plan.residue_from_words(&words)
    }

    fn copy(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        self.check()?;
        let (d, s) = (self.reg(dst)?, self.reg(src)?);
        if dst == src {
            return Ok(());
        }
        let mut enc = self.sh.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&s.buf, 0, &d.buf, 0, None);
        self.submit(enc.finish())
    }

    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.repeat_square(r, n, false)
    }

    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        self.repeat_square(r, n, true)
    }

    fn mul(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        self.check()?;
        let (d, s) = (self.reg(dst)?, self.reg(src)?);
        if dst == src {
            return Err(BackendError::Unsupported(
                "mul needs two different handles".into(),
            ));
        }
        let mut enc = self.sh.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            self.encode_forward(&mut pass, &s.load);
        }
        enc.copy_buffer_to_buffer(&self.sh.t, 0, &self.sh.t2, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            self.encode_forward(&mut pass, &d.load);
            self.encode_pointwise(&mut pass, true);
            self.encode_back(&mut pass, &self.bg_base, d);
        }
        self.submit(enc.finish())
    }

    /// Full-value comparison on the host: both residues are exported and
    /// their canonical values compared.
    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        Ok(self.export(a)? == self.export(b)?)
    }

    fn sync(&mut self) -> BackendResult<()> {
        self.check()?;
        self.wait(None)?;
        self.in_flight.clear();
        self.check_fault_flag()
    }

    fn diagnostics(&mut self) -> BackendResult<Vec<(String, String)>> {
        let p = &self.plan;
        Ok(vec![
            ("transform_length".into(), p.n.to_string()),
            ("word_bits".into(), format!("{}..{}", p.w_min, p.w_max)),
            (
                "coefficient_bound".into(),
                format!("{} (limit {})", p.z_bound, model::Z_LIMIT),
            ),
            ("carry_passes".into(), p.passes.to_string()),
            (
                "iterations_per_submit".into(),
                self.iters_per_submit.to_string(),
            ),
            ("adapter".into(), self.device_name.clone()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::model::{self, F2, Q};
    use super::*;

    fn backend(p: u32) -> GpuBackend {
        GpuBackend::new(p, &GpuOptions::default()).expect("a hardware GPU adapter is required")
    }

    /// Fused radix 4 stages give the same residues as one pass per stage,
    /// at odd and even stage counts.
    #[test]
    fn fused_stages_match_single_stages() {
        let opts = GpuOptions {
            radix4: true,
            ..GpuOptions::default()
        };
        for p in [127u32, 521, 4423, 86243] {
            let mut a = backend(p);
            let mut b = GpuBackend::new(p, &opts).unwrap();
            assert!(b.identity().plan.ends_with(" radix4"));
            let x = Residue::from_u64(p, 3);
            let (ra, rb) = (a.alloc().unwrap(), b.alloc().unwrap());
            a.import(ra, &x).unwrap();
            b.import(rb, &x).unwrap();
            a.square(ra, 40).unwrap();
            b.square(rb, 40).unwrap();
            assert_eq!(a.export(ra).unwrap(), b.export(rb).unwrap(), "p = {p}");
        }
    }

    /// The packed transform gives the same residues as the unpacked one,
    /// for squaring and for multiplication, alone and with fused stages.
    #[test]
    fn packed_transform_matches_unpacked() {
        for radix4 in [false, true] {
            let opts = GpuOptions {
                packed: true,
                radix4,
                ..GpuOptions::default()
            };
            for p in [127u32, 521, 4423, 86243] {
                let mut a = backend(p);
                let mut b = GpuBackend::new(p, &opts).unwrap();
                assert!(b.identity().plan.contains(" packed"));
                let x = Residue::from_u64(p, 3);
                let (ra, rb) = (a.alloc().unwrap(), b.alloc().unwrap());
                let (sa, sb) = (a.alloc().unwrap(), b.alloc().unwrap());
                a.import(ra, &x).unwrap();
                b.import(rb, &x).unwrap();
                a.square(ra, 40).unwrap();
                b.square(rb, 40).unwrap();
                a.copy(sa, ra).unwrap();
                b.copy(sb, rb).unwrap();
                a.square(ra, 3).unwrap();
                b.square(rb, 3).unwrap();
                a.mul(ra, sa).unwrap();
                b.mul(rb, sb).unwrap();
                b.ll_step(rb, 2).unwrap();
                a.ll_step(ra, 2).unwrap();
                assert_eq!(a.export(ra).unwrap(), b.export(rb).unwrap(), "p = {p}");
            }
        }
    }

    impl GpuBackend {
        fn read_t(&mut self) -> Vec<F2> {
            let t = self.sh.t.clone();
            let b = self.read(&t, 8 * u64::from(self.plan.n)).unwrap();
            let v: Vec<u32> = bytemuck::pod_collect_to_vec(&b);
            v.chunks(2).map(|c| F2 { re: c[0], im: c[1] }).collect()
        }

        fn mulmod_on_gpu(&mut self, pairs: &[(u32, u32)]) -> Vec<u32> {
            let n = self.plan.n as usize;
            let mut out = Vec::new();
            for chunk in pairs.chunks(n) {
                let mut data = vec![0u32; 2 * n];
                for (i, &(a, b)) in chunk.iter().enumerate() {
                    data[2 * i] = a;
                    data[2 * i + 1] = b;
                }
                self.queue
                    .write_buffer(&self.sh.t, 0, bytemuck::cast_slice(&data));
                let mut enc = self.sh.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    self.dispatch(
                        &mut pass,
                        &self.pipes.mulmod_test,
                        &self.bg_base,
                        self.plan.n,
                    );
                }
                self.submit(enc.finish()).unwrap();
                let t2 = self.sh.t2.clone();
                let b = self.read(&t2, 8 * n as u64).unwrap();
                let v: Vec<u32> = bytemuck::pod_collect_to_vec(&b);
                out.extend(v.chunks(2).take(chunk.len()).map(|c| c[0]));
            }
            out
        }
    }

    #[test]
    fn gpu_mulmod_matches_exact_products_on_edge_values() {
        let mut g = backend(1 << 16); // n = 2^13 elements per batch
        let mut vals = vec![
            0, 1, 2, 3, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff, 0x1_0000, 0x1_0001,
        ];
        vals.extend([
            0x3fff_ffff,
            0x4000_0000,
            0x4000_0001,
            0x7ffe_ffff,
            0x7fff_0000,
            Q - 2,
            Q - 1,
        ]);
        let mut x = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..40 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            vals.push((x % u64::from(Q)) as u32);
        }
        let pairs: Vec<(u32, u32)> = vals
            .iter()
            .flat_map(|&a| vals.iter().map(move |&b| (a, b)))
            .collect();
        let got = g.mulmod_on_gpu(&pairs);
        for (&(a, b), &r) in pairs.iter().zip(&got) {
            assert_eq!(
                u64::from(r),
                u64::from(a) * u64::from(b) % u64::from(Q),
                "{a} * {b}"
            );
        }
    }

    #[test]
    fn memory_estimate_matches_the_device_buffers() {
        let (lo, hi) = supported_exponents();
        for p in [lo - 1, hi + 1] {
            assert!(matches!(
                memory_estimate(p),
                Err(BackendError::Unsupported(_))
            ));
        }
        for p in [lo, 127, 4423, 86243, model::edge(1 << 16).unwrap()] {
            let m = memory_estimate(p).unwrap();
            let mut g = backend(p);
            assert_eq!(m.plan, g.identity().plan, "p={p}");
            assert!(m.on_device);
            let r = g.alloc().unwrap();
            assert_eq!(m.bytes_per_register, g.reg(r).unwrap().buf.size(), "p={p}");
            let sh = &g.sh;
            let resident: u64 = [&sh.t, &sh.t2, &g.s[0], &g.s[1], &sh.tw, &sh.wt, &sh.err]
                .iter()
                .map(|b| b.size())
                .sum();
            // Uniform blocks: the stored base block plus the LL block and one
            // per stage, which only the bind groups hold.
            let stages = (g.bg_dif.len() + g.bg_dit.len()) as u64;
            let uniforms = g.base.size() + 16 + 16 * stages;
            let staging = 4 * u64::from(g.plan.n);
            assert_eq!(m.fixed_bytes, resident + uniforms + staging, "p={p}");
        }
    }

    #[test]
    fn batches_hold_several_iterations_at_every_length() {
        for p in [127, model::edge(1 << 16).unwrap(), supported_exponents().1] {
            assert!(backend(p).iters_per_submit >= MIN_ITERS_PER_SUBMIT, "p={p}");
        }
    }

    /// A nonzero imaginary part after the inverse transform is reported as a
    /// Fault at the next sync, and the flag is cleared.
    #[test]
    fn imaginary_part_fault_is_reported_once() {
        let mut g = backend(521);
        let r = g.alloc().unwrap();
        g.import(r, &Residue::from_u64(521, 3)).unwrap();
        g.sync().unwrap();
        let mut bad = vec![0u32; 2 * g.plan.n as usize];
        bad[7] = 1;
        g.queue.write_buffer(&g.sh.t, 0, bytemuck::cast_slice(&bad));
        let mut enc = g.sh.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            g.dispatch(&mut pass, &g.pipes.unweight, &g.bg_base, g.plan.n);
        }
        g.submit(enc.finish()).unwrap();
        assert!(matches!(g.sync(), Err(BackendError::Fault(_))));
        g.sync().unwrap();
        g.square(r, 1).unwrap();
        assert_eq!(g.export(r).unwrap(), Residue::from_u64(521, 9));
    }

    #[test]
    fn destroyed_device_is_poisoned() {
        let mut g = backend(521);
        let r = g.alloc().unwrap();
        g.sh.device.destroy();
        let _ = g.sh.device.poll(wgpu::PollType::Poll);
        assert!(
            matches!(g.sync(), Err(BackendError::Poisoned(_))),
            "{:?}",
            g.sync()
        );
        assert!(matches!(g.export(r), Err(BackendError::Poisoned(_))));
    }

    /// Stage by stage: the GPU forward transform equals the model's.
    #[test]
    fn gpu_forward_transform_matches_model() {
        for p in [127, 4423, 86243] {
            let mut g = backend(p);
            let r = g.alloc().unwrap();
            let x = Residue::from_u64(p, 0x1234_5678_9abc_def1).mul_small(987_654_321);
            g.import(r, &x).unwrap();
            let mut enc = g.sh.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                g.encode_forward(&mut pass, &g.reg(r).unwrap().load);
            }
            g.submit(enc.finish()).unwrap();
            let gpu = g.read_t();
            let plan = &g.plan;
            let mut t = plan.weight(&plan.words_from_residue(&x));
            plan.forward(&mut t, &model::twiddles(plan.n));
            assert_eq!(gpu, t, "p={p}");
        }
    }
}
