# GPU backend (`--features gpu`)

`backend::gpu` performs exact modular squaring and multiplication modulo
N = 2^p - 1 on the GPU. It is written in Rust on `wgpu` 30 with WGSL compute
shaders authored in this repository (`src/backend/gpu/kernels.wgsl`, embedded
with `include_str!`). There is no floating point anywhere in the arithmetic.
This backend is about correctness, not speed: it has had no performance
tuning.

The default build does not depend on any GPU crate. With `--features gpu` the
backend adds `wgpu` (features `metal`, `vulkan`, `dx12`, `wgsl`), `pollster`
and `bytemuck`.

## Transform

* Field GF(q^2) with q = 2^31 - 1 (M31) and i^2 = -1. Since q = 3 mod 4,
  x^2 + 1 is irreducible and GF(q^2) is a field. Its multiplicative group has
  order q^2 - 1 = 2^32 (2^30 - 1), so a primitive 2^32-th root h exists. The
  backend takes the first element k + i (k = 1, 2, ...) whose (2^30 - 1)-th
  power has exact order 2^32; a unit test checks the order. The n-th root is
  w_n = h^(2^32 / n). For n <= 2^31 the norm of w_n is 1, so its inverse is
  its conjugate and one twiddle table serves both directions.
* Forward transform: radix 2 decimation in frequency (natural order in,
  bit-reversed order out). Inverse: radix 2 decimation in time with conjugate
  twiddles (bit-reversed in, natural out). No bit-reversal pass is needed.
  Each stage is one dispatch with its own precreated uniform buffer and bind
  group.
* GF(q) products of two 31-bit values use four 16-bit limb products in u32,
  a 62-bit sum with explicit carry, and Mersenne folding. The WGSL function is
  tested on the GPU against exact u64 products on all pairs of 58 values
  (edges such as 0, 1, 2^15, 2^16 +- 1, 2^30, q - 1, plus random values).
* Known ceiling: the input to the transform is real (zero imaginary part), so
  half of each GF(q^2) element is unused. The ceiling is a factor of about 2
  in length and memory; the upgrade is a right-angle style packing of two
  words per element, or a real-input transform.

## Word layout and weights

A residue lives on the device as n signed 32-bit balanced words. Word k sits
at bit b_k = ceil(p k / n) and has w_k = b_(k+1) - b_k bits, which is
floor(p / n) or ceil(p / n). The value is sum x_k 2^(b_k) mod N.

The irrational-base weights are exact here. Word k gets weight r^(e_k) with
e_k = n b_k - p k in [0, n) and r an n-th root of 2. In GF(q) the element 2
has odd order 31, so r = 2^(n^-1 mod 31) satisfies r^n = 2, and every weight
is a power of two modulo q: weighting and unweighting (together with the
n^-1 scale) are 31-bit rotations. For a product term x_i y_j with
i + j = k mod n the weights combine to exactly 2^c, c = b_i + b_j - b_k
(minus p on wraparound), and c is 0 or 1. So the inverse transform returns
the integers z_k = sum x_i y_j 2^c reduced mod q, and they are recovered
exactly when |z_k| <= (q - 1) / 2. The per-word widths and rotations come
from a host table, so no p k product is formed in the shader.

## Capacity bound

Between operations every word satisfies |x_k| <= W_k = 2^(w_k - 1) + 1 (the
+1 is the lazy carry slack, see below). Because i -> k - i is a permutation
and 2^c <= 2, Cauchy and Schwarz give

    |z_k| <= 2 sum_i |x_i| |y_(k-i)| <= 2 S,   S = sum_k W_k^2.

A plan (p, n) is accepted only if 2 S <= 2^29, which is half of the exact
recovery limit 2^30 - 1: one bit of explicit margin. The backend uses the
smallest power-of-two n in [4, 2^24] that satisfies the bound with every
word at least 2 bits wide. Unit tests check, at the largest p of several n,
that words all at +W_k or all at -W_k give exactly the i64 convolution, and
that slightly larger words give wrong coefficients (so the check is not
vacuous).

| n | largest p | word bits | carry passes |
|---|---|---|---|
| 4 | 55 | 13..14 | 3 |
| 64 | 767 | 11..12 | 3 |
| 1024 | 10234 | 9..10 | 4 |
| 16384 | 130732 | 7..8 | 5 |
| 65536 | 456056 | 6..7 | 5 |
| 262144 | 1551564 | 5..6 | 6 |
| 1048576 | 5076519 | 4..5 | 8 |
| 4194304 | 15503945 | 3..4 | 10 |
| 8388608 | 26214400 | 3..4 | 10 |
| 16777216 | 40894464 | 2..3 | 15 |

Supported exponents: **8 ..= 40894464** (`supported_exponents()`), proven
contiguous by an exhaustive unit test. The cap n <= 2^24 keeps the largest
buffer (n elements of 8 bytes, 128 MiB) inside the WebGPU default storage
binding limit of 128 MiB. It is also the end of this field: above 2^24 the
bound leaves words narrower than 2 bits, so wavefront exponents (above
100,000,000) need a different modulus or a CRT of two. Lengths 2^23 and 2^24
are written but have not been run on a device. `GpuBackend::new` also checks the real adapter
limits (storage binding size, buffer size, storage buffers per stage,
workgroup size) and returns `Unsupported` if the plan does not fit. The
backend does not check that p is prime; the protocol layer does.

## Memory

`gpu::memory_estimate(p)` returns these figures without opening a device
(the `doctor` command shows them), and a unit test checks them against the
sizes of the buffers a real backend creates:

* per resident register: n words of 4 bytes, **4 n bytes** (16 MiB at
  n = 2^22, 64 MiB at n = 2^24).
* fixed per backend: two transform buffers (8 n bytes each), two carry
  buffers (4 n each), the twiddle table (n / 2 elements of 8 bytes), the word
  table (4 n), 16 bytes of fault flags, 16 bytes of uniform parameters for
  each of the 2 + 2 log2(n) parameter blocks, and the 4 n byte staging buffer
  that `export` creates for one read: **36 n + 16 (3 + 2 log2(n)) bytes**
  (144 MiB plus 752 bytes at n = 2^22, 576 MiB plus 816 bytes at
  n = 2^24).

Memory the driver and `wgpu` use internally (pipelines, bind groups, command
buffers) is not counted.

## Carry

After unweighting, word 0 also receives the Lucas-Lehmer subtraction of 2
(for `ll_step`). Then a fixed number of parallel passes runs, each mapping

    x'_k = lo(v_k) + hi(v_(k-1)),  hi(v) = floor((v + 2^(w-1)) / 2^w),
    lo(v) = v - hi(v) 2^w in [-2^(w-1), 2^(w-1)),

with the carry out of word n - 1 entering word 0 because 2^p = 1 mod N. Each
pass preserves the value mod N. Passes ping-pong between two scratch buffers
and the last one writes the register.

Pass count proof: if every carry in pass t has magnitude at most H_t, then
afterwards |x_k| <= 2^(w_k - 1) + H_t, and the next carries satisfy
H_(t+1) <= floor((2^w + H_t) / 2^w) = 1 + floor(H_t / 2^w_min). The first
pass sees |v| <= 2 S + 2, so H_1 <= floor((2 S + 2 + 2^(w-1)) / 2^w). The plan
iterates this recurrence and uses the first pass t with H_t <= 1, which is
exactly the invariant |x_k| <= 2^(w_k - 1) + 1 that the capacity bound
assumes. A carry of magnitude 1 can always remain, so this +1 slack is
inherent to a fixed number of parallel passes. A unit test runs the passes
from the worst inputs (all +2S, all -(2S + 2), and mixed patterns) and checks
the invariant.

## Host conversions

* `import`: the canonical limbs are cut into w_k-bit digits and balanced
  sequentially; the final carry enters word 0. Uploaded with
  `Queue::write_buffer`.
* `export`: the word buffer is copied to a staging buffer and mapped. The
  host checks every word against its range (a word outside it is
  `BackendError::Fault`), performs a full carry normalization with the
  cyclic wraparound, packs the bits, and builds the residue with
  `Residue::from_limbs_normalizing`, which maps the value N to the unique
  zero.
* `equal`: exports both registers and compares the canonical values on the
  host. It is a full-value comparison, never a fingerprint.
* `copy`: a device buffer copy. `mul(dst, src)`: transform `src`, copy the
  transform aside, transform `dst`, pointwise multiply, then the same inverse,
  unweight, and carry path as squaring.

## Batching and synchronization

`square(r, n)` and `ll_step(r, n)` encode many iterations into one command
buffer and one compute pass. One iteration is 2 log2(n) + passes + 3
dispatches. Iterations per submission are
clamp(2^26 / (n (2 log2(n) + passes + 3)), 4, 256): for example 113 at
n = 2^14, 25 at n = 2^16, and the floor of 4 from n = 2^20 up (about 0.25 s of
GPU time per submission at n = 2^22 on the Apple M2). Large n are split into
several submissions. At most 2 submissions are in flight: before a third one
is queued, the host waits for the oldest to finish. That wait is per batch of
at least 4 iterations, never per iteration, and never transfers the residue.
Only `export`, `equal`, and `sync` read data back.

## Error reporting

* An uncaptured `wgpu` error (validation, internal, out of memory) is caught
  by the device error callback and stored. It is sticky: every later call
  returns it. Out of memory becomes `Allocation`, anything else `Poisoned`.
* A lost device (device lost callback) is `Poisoned`.
* Kernel compilation is checked with an error scope in `GpuBackend::new`
  (`Unsupported` on failure).
* The inverse transform of a real convolution is real. The unweight kernel
  flags any nonzero imaginary part in a device error word; the flag is read
  at the next `sync`, `export`, or `equal`, reported as `Fault`, and cleared.
* Handles come from one process-wide counter shared by every backend, so
  freed handles and handles of another backend instance are `InvalidHandle`;
  exhausting the counter is `Allocation`. A residue of the wrong
  exponent is `Unsupported`, as is `mul` with the same handle twice.

## Identity and device selection

* `kind = "gpu"`, `plan = "ntt-gf-m31sq n=<len>"`,
  `device = "<adapter name> (<api>)"`, `build` contains the crate version,
  the implementation source fingerprint, and the SHA3-256 of the embedded WGSL.
* `probe()` enumerates Metal, Vulkan, and DX12 adapters. Enumeration is not
  validation. `GpuOptions { device: None }` uses the first adapter that is not
  a software (`Cpu` type) adapter; a software adapter is used only when
  selected by index. With no usable adapter, `GpuBackend::new` returns
  `Unsupported` with a message saying so.
* No `unsafe` code.

## What was validated

Only this machine: **Apple M2, Metal backend** (macOS arm64, Rust 1.100.0,
`wgpu` 30.0.2). Observed on this machine:

* `cargo build --features gpu`: builds. `cargo build`: builds, and
  `cargo tree -e normal` lists no `wgpu` crate.
* `cargo test --features gpu --lib backend::gpu`: 16 passed (10 host model
  tests against `num-bigint` and exact i64 or u64 arithmetic, including the
  export range check at `i32::MIN` and `i32::MAX`; 6 GPU tests: WGSL mulmod
  on edge pairs, the GPU forward transform equal to the model for p = 127,
  4423, 86243, a forced nonzero imaginary part reported once as `Fault`, a
  destroyed device reported as `Poisoned`, the batch floor, and the memory
  estimate equal to the created buffer sizes).
* `cargo test --features gpu --test gpu_backend`: 9 passed, 1 ignored, about
  21 s. Every expected value is plain `num-bigint` arithmetic in the test.
  Covered: import and export round trips (0, 1, 2, N - 1, N - 2, long runs of
  ones, top bit, all-ones limbs, alternating bits, random, and values whose
  words all have maximal magnitude) for p from 8 to 15503945, including the
  largest p of n = 4, 64, 1024, 16384, 65536, and lengths 2^18 and 2^22;
  single squarings of all of those; batches of 1, 2, 7, 100, 1000 squarings in
  one call and split across calls; 20 squarings from maximal words at the
  bound edges of n = 1024 and 65536, and 5 at the largest supported exponent; `mul`, `copy`, `equal`, `ll_step`;
  complete LL runs ending at zero for p = 521, 607, 1279, 2203, 2281, 4423,
  9689; complete LL runs for p = 1277 and 4421 equal to bigint and nonzero;
  error paths.
* Raw observed timings, debug test profile (opt-level 2), Apple M2, single
  runs, not benchmarks: complete LL p = 9689 about 1.0 s to 1.2 s; complete
  LL p = 110503 (`cargo test --features gpu --test gpu_backend -- --ignored`)
  12.1 s, ending at zero; one squaring about 0.2 ms at n = 2^14, 0.5 ms at
  2^16, 1.0 ms at 2^18, 11 ms at 2^20, 63 ms at 2^22.
* Mutation checks: removing the carry term from the WGSL mulmod fails 6 of
  the 9 GPU integration tests; using one carry pass fewer than the proof
  requires is caught by the export range check.

## What was NOT validated

* Every other vendor and API: NVIDIA, AMD, and Intel GPUs, Vulkan, DX12, and
  software adapters. The code is written for them but has never run on them.
* Complete LL or PRP runs at transform lengths above 2^17. Lengths 2^18
  through 2^22 are covered by single and short repeated squarings only. The
  largest complete run on this machine is the M786433 PRP with a power 10
  proof (n = 2^17) recorded in `docs/VALIDATION.md`; the complete runs listed
  above use n = 2^14 and below. The protocol tests in `tests/protocol_gpu.rs` belong
  to the core and are not described here.
* Performance tuning (workgroup size, memory layout). Two opt-in variants
  exist and are written but not yet validated or timed:
  `MERSENNE_FORGE_GPU_RADIX4=1` runs two NTT stages per pass (`dif4`,
  `dit4`, the same field operations in the same order, so bit identical),
  and `MERSENNE_FORGE_GPU_PACKED=1` packs words 2j and 2j + 1 as the real
  and imaginary part of element j and runs a half length transform with an
  untangle, pointwise, and retangle pass (`sqr_packed`, `pmul_packed`; the
  derivation is in `kernels.wgsl`). The packed path cannot use the
  imaginary part fault check. `tf.wgsl` is a separate trial factoring
  kernel for candidates below 2^95 (three word Montgomery arithmetic). No speed claims are made.
* Long-run behavior under device resets, thermal throttling, or driver
  timeouts beyond what the error paths above report.
