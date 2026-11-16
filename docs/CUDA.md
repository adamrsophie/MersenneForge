# CUDA backend (`--features cuda`)

`backend::cuda` performs exact modular squaring and multiplication modulo
N = 2^p - 1 on one NVIDIA GPU. The host code is Rust on the CUDA driver API
and NVRTC through `cudarc` 0.19.10 with dynamic loading, so building needs no
CUDA SDK, driver, or GPU. The kernels are CUDA C authored in this repository
(`src/backend/cuda/kernels.cu`, embedded with `include_str!`) and compiled at
run time by NVRTC. There is no floating point anywhere in the arithmetic.

**Status: implemented, compile checked, kernel logic host emulated, never run
on CUDA hardware.** Nothing in this document is a performance claim, and no
timing of any kind has been taken.

## Evidence levels

The three levels are kept apart. Each claim below names its level.

| Level | What it covers | How | Run here |
|---|---|---|---|
| (a) mathematics | the exact algorithm: field, transform, weights, capacity bound, carry, conversions | Rust CPU model (`src/backend/cuda/model.rs`) against `num-bigint` | yes, passes |
| (b) kernel logic | the very same CUDA C source, its arithmetic and indexing, and the launch sequence the host submits | host emulation: the source compiled by the host C++ compiler and run on the CPU, compared word for word with the model | yes, passes |
| (c) CUDA execution | NVRTC compilation, module loading, launches, device memory, streams, error paths, results on a GPU | `tests/cuda_backend.rs` on an NVIDIA host | **no: never run on CUDA hardware** |

What (b) proves: the kernel arithmetic and indexing as C++ semantics on a CPU,
every thread of every launch, in forward and reversed thread order, with
several block sizes that do not divide n, under AddressSanitizer and
UndefinedBehaviorSanitizer with exactly sized buffers. What (b) does not prove:
that NVRTC accepts the source, that the launch configuration is valid on a
device, device memory management, concurrency between GPU threads, or any
device behavior. The host `unsigned __int128` stands in for `__umul64hi`.

Commands run on the development machine (macOS arm64, Apple GPU, no NVIDIA
hardware, Rust 1.100) and their observed results:

* `cargo build --features cuda`: builds.
* `cargo clippy --features cuda --all-targets -- -D warnings`: no issues.
* `cargo check --features cuda --all-targets --target x86_64-unknown-linux-gnu`: compiles (a compile check for Linux; not linked or run).
* `cargo test --features cuda --lib backend::cuda`: 18 model and host unit tests pass (level a). The largest transform length executed by the model is n = 2^18 (p = 5242880, one squaring, one LL step, one multiplication against `num-bigint`).
* `cargo test --features cuda --test cuda_kernel_emulation`: 5 tests pass (level b), with Apple clang as `c++`; the largest length emulated is n = 2^12. GCC on Linux has not been tried here. As a check that the emulation detects faults, three deliberate kernel bugs were each caught: an off by one index guard (AddressSanitizer heap overflow), a wrong neighbor width in carry pass B, and a missing Mersenne fold in the modular multiply (both word for word mismatches against the model).
* `cargo test --features cuda --test cuda_backend`: all 7 tests fail, as required, with "no CUDA driver: this suite needs an NVIDIA GPU with libcuda and NVRTC" and the list of library names searched.
* `probe()` on this Mac returns `driver_loaded: false`, no devices, and notes naming the missing `libcuda` and `libnvrtc` libraries with every file name searched. The kernel source hash at the time of writing is `2cc097172814308d421731e706e5692646c993432dfaa823cc58338219246c4c`.
* `cargo tree -e normal` without the feature contains no `cudarc`.

The range up to p = 285212672 rests on the capacity proof below, not on a run at that size.

## Transform

* Field GF(q^2), q = 2^61 - 1 (M61), elements a + b i with i^2 = -1. Since
  q = 3 mod 4, x^2 + 1 is irreducible and GF(q^2) is a field. Its group has
  order q^2 - 1 = (q - 1)(q + 1) = 2^62 (2^60 - 1).
* The norm-1 subgroup has order q + 1 = 2^61. The host finds the first
  g = a + i (a = 1, 2, ...) for which h = g^(q - 1) has order exactly 2^61
  (h^(2^60) = -1; unit tested). The n-th root is w_n = h^(2^61 / n). Every
  w_n lies in the norm-1 subgroup, so w_n^-1 = w_n^q = conj(w_n): one twiddle
  table of n/2 entries serves both directions. GF(q) alone has no roots of
  order above 2, which is why the transform runs in GF(q^2).
* Forward: radix-2 decimation in frequency, natural order in, bit-reversed
  out. Inverse: radix-2 decimation in time with conjugate twiddles,
  bit-reversed in, natural out. No bit-reversal pass. The 1/n scale is folded
  into unweighting. The model checks that forward, pointwise product, inverse
  equals n times the cyclic convolution.
* GF(q) multiply (`mf_mulmod`): the 128-bit product from `a * b` and
  `__umul64hi(a, b)`, then Mersenne reduction. For a, b < 2^61 the product is
  below 2^122, so hi < 2^58 and `(hi << 3) | (lo >> 61)` is the product shifted
  by 61 without overflow; two folds and one conditional subtraction leave
  [0, q). The proof is in the kernel comment; the model checks it against
  `u128 %` on all pairs of 208 edge and random values.
* `ponytail:` the transform input has zero imaginary part, a 2x ceiling on
  transform length per word. The upgrade is the right-angle convolution
  (word pairs packed as re + i im, twisted by a 4n-th root of unity).
* `ponytail:` radix-2 stages with one kernel launch per stage, and separate
  weight, unweight, and carry launches. The upgrade is radix-4 or radix-8
  stages in shared memory and fused kernels. Transform lengths are powers of
  two only; non power of two lengths (3 or 5 times 2^k) would fill the gaps
  where an exponent just above a length's capacity doubles n.

## Word layout and weights

A residue lives on the device as n balanced signed words, stored as 64-bit
two's complement. Word i sits at bit b_i = ceil(p i / n) and has width
w_i = b_(i+1) - b_i, which is floor(p / n) or ceil(p / n); b_n = p.

2 has order 61 modulo q, and n is a power of two, so k = n^-1 mod 61 exists
and r = 2^k satisfies r^n = 2^(kn) = 2 mod q: r is an exact n-th root of 2.
Word i gets weight r^(e_i) with e_i = ceil(p i / n) n - p i = (-(p i)) mod n,
so every weight is a power of two and weighting is the 61-bit rotation by
(k e_i) mod 61. Unweighting with the 1/n scale is the rotation by
(122 - (k e_i mod 61) - log2 n) mod 61. Both shifts are computed from the
thread index; there is no weight table.

## Capacity bound

Let every input word satisfy |x_i|, |y_j| <= W. After weighting, transform,
pointwise product, inverse, and unweighting, coefficient k equals, modulo q,
the integer

    z_k = sum over i + j = k (mod n) of x_i y_j 2^(d_ij),   d_ij in {0, 1}.

Proof sketch: the weight exponents give e_i + e_j - e_k = n (b_i + b_j - b_k)
when i + j = k, and n (b_i + b_j - b_k - p) when i + j = k + n. In both cases
the bracket is ceil(u) + ceil(v) - ceil(u + v) for real u, v, which is 0 or 1,
and r^(n d) = 2^d. The same identity gives 2^(b_i) 2^(b_j) = 2^d 2^(b_k), times
2^p = 1 mod N on wraparound, so sum z_k 2^(b_k) = X Y mod N. The factor 2 is
the weight wraparound factor. Hence

    |z_k| <= 2 n W^2.

Reading z_k back as a balanced integer in [-(q - 1)/2, (q - 1)/2] is exact if
and only if |z_k| <= (q - 1)/2 = 2^60 - 1. A plan is accepted only if

    2 n W^2 <= 2^59,

one bit of margin. W includes the lazy carry slack:
W = 2^(w_max - 1) + floor(m / 2), where m is the carry piece count below. The
model tests feed words at exactly +W, at -W, and in alternating patterns at the
capacity edge of lengths 4 to 1024, compare every unweighted coefficient
with an i128 schoolbook evaluation of the formula above, and check the final
residue against `num-bigint`.

## Carry

Coefficients can be near 2^57 while words are 17 to 28 bits, so a single
"low bits plus one carry" step is not enough. The carry is two fixed,
data-parallel passes; neither has a sequential chain.

Balanced split for width w: low(v) = v - round(v / 2^w) 2^w, in
[-2^(w-1), 2^(w-1)), and carry(v) = (v - low(v)) / 2^w, exact.

**Pass A** (`mf_carry_a`, a gather): coefficient z_s is split along the widths
of words s, s + 1, ... into m parts: m - 1 balanced digits and a final
remainder. Part j lands on word s + j mod n (the wrap from word n - 1 to word 0
has weight 2^p = 1 mod N). Thread k sums the parts landing on word k from
sources k, k - 1, ..., k - m + 1, plus the LL subtraction at word 0.

The remainder after t splits satisfies |v_(t+1)| <= (|v_t| + 2^(w-1)) / 2^w,
so by induction |v_t| < Z / 2^(t w_min) + 1 for |z| <= Z. The plan chooses the
smallest m with 2^((m - 1) w_min) >= Z = 2 n W(m)^2; then the remainder has
magnitude at most 1. Every other part landing on word k is a digit of width
w_k. Therefore

    |a_k| <= (m - 1) 2^(w_k - 1) + 1 + 2   (the 2 is the LL subtraction).

**Pass B** (`mf_carry_b`): x_k = low(a_k) + carry(a_(k-1)), each thread
recomputing its neighbor's carry. |carry(a)| <= (|a| + 2^(w-1)) / 2^w
<= m/2 + 3/2^w, and with w >= 3 the term 3/2^w is at most 3/8, so the integer
carry is at most floor(m / 2). Hence after every squaring

    |x_k| <= 2^(w_k - 1) + floor(m / 2),

which is the bound W the capacity condition assumed. Imported words have
|x_0| <= 2^(w_0 - 1) + 1, within W since m >= 2. Two passes are enough because
the bound is self consistent: words never need full normalization between
squarings. A unit test brute forces both passes on extreme and random
coefficients at widths 3 to 10 and checks the output bound and that the value
modulo N is preserved, with and without the LL subtraction. Export verifies the
bound on every word (a violation is `Fault`) and then normalizes exactly on
the host.

For tiny exponents m can exceed n (p = 13: n = 4, m = 5); the gather then
wraps around the ring more than once, which the index arithmetic handles and
the tests exercise. In practice m is 4 to 6.

`ll_step` is a squaring with -2 added at word 0 in pass A. `mul(dst, src)`
weights and transforms both operands, multiplies pointwise, and shares the
back end.

## Supported range and memory

Plan rule: the smallest n = 2^L, 2 <= L <= 24, with w_min >= 3, w_max <= 30,
and 2 n W^2 <= 2^59. Validity depends on p only through (floor(p/n),
ceil(p/n)); a unit test checks that the range has no gaps (every length's
classes are contiguous, the next length takes over exactly where one stops,
and every p below 200000 has a plan).

`supported_exponents()` = **(12, 285212672)**, inclusive.

Largest exponent per length (all at m = 4 or 5, slack 2):

| n | max p | n | max p |
|---|---|---|---|
| 2^2 | 112 | 2^14 | 360448 |
| 2^4 | 432 | 2^16 | 1376256 |
| 2^6 | 1664 | 2^18 | 5242880 |
| 2^8 | 6400 | 2^20 | 19922944 |
| 2^10 | 24576 | 2^22 | 75497472 |
| 2^12 | 94208 | 2^24 | 285212672 |

Examples: p = 1279 uses n = 2^6 (widths 19..20, 2 n W^2 = 2^45);
p = 136279841 uses n = 2^23 (widths 16..17, 2^56).

Device memory, all buffers of 64-bit elements:

* per context: two transform buffers (16 n bytes each), twiddles, carry
  scratch Z and A (8 n bytes each): **56 n bytes**, 896 MiB at n = 2^24,
  224 MiB at n = 2^22.
* per resident residue: **8 n bytes**, 128 MiB at n = 2^24.

`ponytail:` words are stored as 64-bit values although they fit in 32 bits;
the upgrade halves residue and scratch memory.

## Host, batching, and synchronization

* One context per backend: `CudaContext::new(ordinal)` retains the device's
  primary context. One non-blocking stream carries all work in order. NVRTC
  compiles the source at context creation; the module and all eight function
  handles are loaded once.
* NVRTC options: `--std=c++17 --gpu-architecture=compute_XY`, XY from the
  device's compute capability. They are recorded in the identity.
* No kernel binary cache in 0.1: every context creation compiles. A future
  cache must be keyed by the source hash, the options, the architecture, and
  the NVRTC and driver versions.
* `square(r, n)` and `ll_step(r, n)` queue n copies of the launch sequence
  (2 log2 n + 5 launches each) with no host transfer and no synchronization.
  Launch errors returned at submission are checked on every launch; deferred
  execution errors surface at the next `sync`, `export`, or `equal`.
* `export` copies the words to the host, synchronizes the stream, checks the
  word bound, and converts exactly to the canonical `Residue` with
  `Residue::from_limbs_normalizing` (the all-ones pattern N becomes the unique
  zero). `equal` exports both and compares canonical values. `import`
  converts exactly to balanced words, copies, and synchronizes so the host
  buffer outlives the copy. `copy` is a device to device copy on the stream.
  `sync` synchronizes the stream and reads `cudarc`'s recorded error state.
* Lifetimes: every device buffer is owned by the backend; `cudarc` frees a
  buffer on its owning stream (or synchronizes first), so queued kernels never
  see freed memory. Dropping the backend first synchronizes its stream, so the
  module is never unloaded under queued kernels. The one `unsafe` launch call documents its invariants;
  there is no `unsafe impl Send` or `Sync`.
* Identity: `kind = "cuda"`, `plan = "ntt-gf-m61sq n=<n>"`,
  `device = "<name> (CUDA <ordinal>, cc <major.minor>)"`, and `build` lists the
  crate version, the SHA3-256 of the kernel source, the NVRTC version and
  options, and the driver version.
* Process-global state: the CUDA driver keeps one primary context per device
  per process, and `cudarc` keeps the loaded libraries and symbols in
  process-wide `OnceLock`s. Two live backends on one device are safe (the
  primary context is reference counted), but a sticky error in either poisons
  both. One engine per device is the supported configuration; the hardware
  test suite serializes on a lock.

## Error mapping

Every driver and NVRTC call is checked.

| Condition | `BackendError` |
|---|---|
| exponent outside (12, 285212672); checked before the driver is touched | `Unsupported` |
| libcuda or libnvrtc missing (names searched listed); missing symbol (driver older than the bindings; `cudarc` panics, caught) | `Unsupported` |
| no device, invalid ordinal, stub library, driver mismatch, unsupported PTX version, no binary for GPU | `Unsupported` |
| out of memory | `Allocation` |
| illegal address, launch failed or timed out, illegal instruction, misaligned address, ECC, context destroyed, and other sticky errors | `Poisoned`, latched: every later call returns it |
| NVRTC compile failure | `Native` with the NVRTC code and the full compile log |
| any other driver code | `Native` with code, name, and message |
| exported word outside the proven bound | `Fault` |
| unknown or freed handle | `InvalidHandle` |

None of these is a mathematical result; the protocol never turns one into
"composite". The code to error mapping is unit tested.

## Runtime prerequisites

* Linux x86_64 (other targets untested), an NVIDIA GPU, and the NVIDIA driver
  providing `libcuda.so.1`.
* NVRTC from a CUDA 12 toolkit or runtime package: `libnvrtc.so.12` (or
  `libnvrtc.so`), plus its companion `libnvrtc-builtins.so.12.*`, both on the
  loader path (`LD_LIBRARY_PATH` if not in a system directory). The bindings
  are pinned with the `cuda-12080` feature (CUDA 12.8 headers). With that pin
  `cudarc` searches, among others, `libnvrtc.so`, `libnvrtc.so.12`,
  `libnvrtc.so.11`, and `libcuda.so`, `libcuda.so.1`; it does **not** search
  `libnvrtc.so.13`, so a CUDA 13 only installation needs a `libnvrtc.so`
  link. `probe()` lists the exact names searched when a library is missing.
* The driver must accept the PTX that NVRTC emits: the driver's CUDA version
  should be at least the NVRTC version.

## Validating on an NVIDIA Linux host

    cargo test --features cuda --lib backend::cuda
    cargo test --features cuda --test cuda_kernel_emulation
    cargo test --features cuda --test cuda_backend
    cargo test --features cuda --test protocol_cuda
    cargo build --release --features cuda
    target/release/mersenne-forge doctor --json
    target/release/mersenne-forge run --test ll --exponent 44497 --backend cuda --device 0 --budget 10m
    target/release/mersenne-forge run --test prp --exponent 86243 --backend cuda --device 0 --budget 10m --proof-dir proofs --checkpoint-dir ckpt
    target/release/mersenne-forge verify --proof PATH --backend cuda

`cargo test --features cuda --test cuda_backend` covers import and export
round trips, single and batched squarings, LL steps, multiplication, copy,
equal, complete LL runs (Mersenne primes up to M44497 and composites with
prime exponents), and error paths, all compared as complete canonical
residues against `num-bigint`. Set `MERSENNE_FORGE_CUDA_DEVICE` to test a
device other than 0. Until these pass on real hardware, the CUDA backend has
evidence levels (a) and (b) only.

## Most likely to break on the first real CUDA run

1. Library discovery: NVRTC installed as `libnvrtc.so.13` only, or
   `libnvrtc-builtins` not on the loader path.
2. NVRTC and driver version skew: PTX too new for the driver
   (`UNSUPPORTED_PTX_VERSION`, reported as `Unsupported`), or an option the
   installed NVRTC rejects (reported as `Native` with the log).
3. NVRTC rejecting something the host compiler accepted: emulation does not
   exercise NVRTC's front end at all.
4. A `cudarc` symbol missing from an older driver (panic caught and reported
   as `Unsupported`).
5. Kernel argument passing: emulation checks arity and pointer versus scalar
   kinds against the same signatures, but not the driver's argument
   marshaling.
6. Memory: a large exponent needs 56 n bytes per context plus 8 n bytes per
   residue on the device.
