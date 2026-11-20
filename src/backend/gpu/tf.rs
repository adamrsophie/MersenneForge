// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Trial factoring on the GPU: the kernel in `tf.wgsl` tests a batch of
//! candidate divisors q < 2^95 of 2^p - 1 (mfaktc's bound), and the host
//! sieves the candidates (see [`crate::factor::trial_factor_with`]) and
//! checks every hit again exactly. Larger candidates are tested on the host.

use wgpu::util::DeviceExt;

/// The trial factoring kernel source.
pub const TF_KERNEL: &str = include_str!("tf.wgsl");

/// Largest candidate the kernel takes (exclusive).
pub const Q_LIMIT: u128 = 1 << 95;

/// Hit slots per batch; a batch with more hits is tested on the host.
const CAP: u32 = 64;

pub struct GpuTf {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    max_groups: u32,
    pub device_name: String,
}

impl GpuTf {
    pub fn new(device: Option<u32>) -> Result<GpuTf, String> {
        let adapter = super::pick_adapter(device).map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        let device_name = format!("{} ({:?})", info.name, info.backend);
        let limits = adapter.limits();
        let (dev, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mersenne-forge tf"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| format!("cannot open {device_name}: {e}"))?;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tf"),
            source: wgpu::ShaderSource::Wgsl(TF_KERNEL.into()),
        });
        let buffer = |ty| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let types = [
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ];
        let entries: Vec<wgpu::BindGroupLayoutEntry> = types
            .iter()
            .zip(0..)
            .map(|(&ty, binding)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: buffer(ty),
                count: None,
            })
            .collect();
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tf"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("tf"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!(
                "the TF kernel does not compile on {device_name}: {e}"
            ));
        }
        Ok(GpuTf {
            device: dev,
            queue,
            pipe,
            layout,
            max_groups: limits.max_compute_workgroups_per_dimension.max(1),
            device_name,
        })
    }

    /// The candidates in `qs` (odd, below [`Q_LIMIT`]) for which
    /// 2^p = 1 mod q, by the kernel. None when the hit slots overflowed;
    /// the caller then tests the batch on the host.
    pub fn test(&self, p: u32, qs: &[u128]) -> Result<Option<Vec<u128>>, String> {
        if qs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if qs.iter().any(|&q| q >= Q_LIMIT || q & 1 == 0) {
            return Err("GPU trial factoring takes odd candidates below 2^95".into());
        }
        let count = u32::try_from(qs.len()).map_err(|_| "batch too large".to_string())?;
        let words: Vec<u32> = qs
            .iter()
            .flat_map(|&q| [q as u32, (q >> 32) as u32, (q >> 64) as u32])
            .collect();
        let init = |label, contents: &[u8], usage| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        let params = init(
            "tf params",
            bytemuck::cast_slice(&[p, count, 31 - p.leading_zeros(), CAP]),
            wgpu::BufferUsages::UNIFORM,
        );
        let q = init(
            "tf candidates",
            bytemuck::cast_slice(&words),
            wgpu::BufferUsages::STORAGE,
        );
        let hits_size = 4 * u64::from(CAP + 1);
        let hits = init(
            "tf hits",
            &vec![0u8; hits_size as usize],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tf readback"),
            size: hits_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: hits.as_entire_binding(),
                },
            ],
        });
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(count.div_ceil(256).clamp(1, self.max_groups), 1, 1);
        }
        enc.copy_buffer_to_buffer(&hits, 0, &staging, 0, hits_size);
        let idx = self.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .map_err(|e| format!("device poll failed: {e}"))?;
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("TF kernel error: {e}"));
        }
        match rx.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(format!("TF readback failed: {e}")),
            Err(_) => return Err("TF readback did not complete".into()),
        }
        let out: Vec<u32> = bytemuck::pod_collect_to_vec(
            &staging
                .get_mapped_range(..)
                .map_err(|e| format!("TF readback view failed: {e}"))?,
        );
        staging.unmap();
        let n = out[0];
        if n > CAP {
            return Ok(None);
        }
        let mut found = Vec::new();
        for &i in &out[1..=n as usize] {
            let q = *qs
                .get(i as usize)
                .ok_or_else(|| format!("TF kernel returned index {i} of {count}"))?;
            found.push(q);
        }
        found.sort_unstable();
        found.dedup();
        Ok(Some(found))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known factors are hits and their neighbors are not, against the
    /// host's exact test, across all three words and near 2^95.
    #[test]
    fn kernel_matches_the_host_test() {
        let tf = GpuTf::new(None).expect("a hardware GPU adapter is required");
        // 2^67 - 1 = 193707721 * 761838257287; 2^11 - 1 = 23 * 89;
        // 2^83 - 1 has the factor 167 and 57912614113275649087721 (76 bits).
        let cases: [(u32, &[u128]); 3] = [
            (67, &[193707721, 761838257287]),
            (11, &[23, 89]),
            (83, &[167, 57912614113275649087721]),
        ];
        for (p, factors) in cases {
            let mut qs: Vec<u128> = Vec::new();
            for &f in factors {
                for d in [0u128, 2, 4, 2 * u128::from(p)] {
                    qs.push(f + d);
                }
            }
            // Large odd candidates in the second and third words.
            let two_p = 2 * u128::from(p);
            for k in [
                (1u128 << 61) / two_p,
                (1u128 << 62) / two_p - 7,
                (1u128 << 80) / two_p + 3,
                (1u128 << 94) / two_p - 1,
            ] {
                qs.push(two_p * k + 1);
            }
            let got = tf.test(p, &qs).unwrap().unwrap();
            let want: Vec<u128> = {
                let mut w: Vec<u128> = qs
                    .iter()
                    .copied()
                    .filter(|&q| {
                        num_bigint::BigUint::from(2u32).modpow(&p.into(), &q.into())
                            == num_bigint::BigUint::from(1u32)
                    })
                    .collect();
                w.sort_unstable();
                w.dedup();
                w
            };
            assert_eq!(got, want, "p = {p}");
            assert!(factors.iter().all(|f| got.contains(f)));
        }
    }
}
