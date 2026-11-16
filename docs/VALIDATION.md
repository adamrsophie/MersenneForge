# Validation report

Host for every result below: Apple M2 (macOS 27.2, arm64), Rust 1.100.0 stable. No NVIDIA, AMD, or Intel GPU was available. Nothing here was run on CUDA hardware.

## Levels

| Item | Implemented | Compiled | CPU tested | GPU tested | Interoperability tested | Performance measured |
|---|---|---|---|---|---|---|
| Residue, exponent domain, oracle, reference LL and PRP | yes | yes | yes | n/a | n/a | no |
| `cpu` backend (bigint, Mersenne folding) | yes | yes | yes | n/a | n/a | no |
| `dwt` backend (f64 DWT) | yes | yes | yes | n/a | n/a | no |
| `gpu` backend (wgpu, NTT over GF(M31^2)) | yes | yes | n/a | Apple M2 through Metal only; NOT validated on NVIDIA, AMD, or Intel | n/a | no |
| `cuda` backend | yes | yes (compiles; loads libcuda at run time) | n/a | NOT validated: never run on CUDA hardware | n/a | no |
| PRP with Gerbicz check, rollback, stop conditions | yes | yes | yes (`cpu`, `dwt`) | Metal only | n/a | no |
| LL with Jacobi check | yes | yes | yes (`cpu`, `dwt`) | Metal only | n/a | no |
| Checkpoint format, atomic write, resume | yes | yes | yes | Metal only | no upstream savefile compatibility (by design) | no |
| PRP proof generation and verification (v2) | yes | yes | yes | Metal only | one direction: an upstream generated proof verifies here and our M786433 proof is byte-identical to it; our proofs have NOT been checked by an upstream verifier | no |
| Bench harness | yes | yes | yes (`bench/cpu-smoke.json`) | no | n/a | no comparison measured |

Continuous integration (`.github/workflows/ci.yml`) runs only the CPU checks; it compiles the `gpu` and `cuda` features but never runs GPU or CUDA tests. Every GPU result in this report was observed by hand on the host above.

"GPU tested" means the generic protocol suite and the CLI runs below passed on the Apple GPU through Metal. Backend internals and their own tests are documented by the GPU engineer in [GPU.md](GPU.md).

## Default build checks

| Command | Result |
|---|---|
| `cargo build` | ok |
| `cargo test` | ok: 74 tests passed, 0 failed, 4 ignored (slow or needing an external file; listed below) |
| `cargo clippy --all-targets -- -D warnings` | ok, no warnings |
| `cargo clippy --features gpu --all-targets -- -D warnings` | ok, no warnings |
| `cargo fmt --check` | ok |
| `cargo check --features gpu --all-targets` | ok |
| `cargo check --features cuda --all-targets` | ok |
| `cargo build --locked --all-targets` | ok |
| `cargo +1.85 build --locked` | ok (MSRV 1.85 for the library and binary) |
| `cargo +1.85 test --no-run` | fails: `proptest 1.12` requires Rust 1.88, so the tests need 1.88 |

Per suite (`cargo test`, debug profile with opt-level 2): lib unit tests 13, main 1, `arithmetic` 7, `backend_fault` 2, `checkpoint` 5, `cli` 5, `cross_backend` 1, `ll_jacobi` 1, `protocol_cpu` 10, `protocol_dwt` 10, `reference` 3 (1 ignored), `upstream_proof` 0 (1 ignored), plus the DWT engineer's `dwt_backend` 16 (2 ignored). The `gpu` and `cuda` test files compile to nothing without their features.

Ignored tests, run explicitly:

| Command | Result |
|---|---|
| `cargo test --release --test reference -- --ignored` | ok: M9689, M9941, M11213, M19937 prime by LL and probable prime by PRP on `dwt`; M9689 on `cpu` |
| `MERSENNE_FORGE_UPSTREAM_PROOF=<path>/786433-10.proof cargo test --release --test upstream_proof -- --ignored` | ok: the proof file shipped in the PRPLL source tree at the pinned commit (`test/786433-10.proof`) parses and verifies (M786433, power 10, composite); a copy with one bit flipped is rejected |

The same upstream proof also verifies from the CLI on `dwt` and on `gpu` (Metal): `mersenne-forge verify --proof work/upstream-786433-10.proof --backend gpu --budget 30m` reported `accepted`, `M786433 composite`.

Generation agrees with upstream on this exponent: `mersenne-forge run --test prp --exponent 786433 --backend gpu --proof-power 10 --proof-dir work/m786433/proof --checkpoint-dir work/m786433/ckpt --budget 40m` (Metal, 970 s wall) reported `composite, res64 bfb6553c004222ba`, 11 Gerbicz checks passed, self verification accepted, and `cmp work/m786433/proof/786433-10.proof <upstream>/test/786433-10.proof` exits 0: the file is byte-identical to the proof shipped upstream. This is format and generation evidence; it is not a test of an upstream verifier accepting our files.

## What the tests establish

- Mathematics: reference vectors for p = 2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423 (prime) and 11, 23, 29, 37, 41, 1277 (composite) by the oracle and by the checked engines; base 2 Fermat passes the composite 2047 = 23 * 89 while base 3 rejects it; the PRP terminal residue by repeated `%` squaring equals the `modpow` (Montgomery) result.
- Representation: canonical encoding at 0, 1, 2, N - 1, N - 2, N / 2, around every 32-bit limb boundary, all-ones words, the value N, bits above p, wrong lengths; word aligned and non aligned exponents; randomized residues (proptest).
- Differential arithmetic: `cpu` (folding) and `dwt` (transform) against the oracle (division) for mul, square, ll_step, copy, equal.
- PRP protocol on `cpu`, `dwt`, and (with `--features gpu`) `gpu`: complete prime and composite runs with full 2048-bit residues equal to the oracle; data and check value corruption (bit flip, zero, triple) detected and rolled back with a correct result; an engine injected `BackendError::Fault` rolled back; the same wrong residue twice stops; three failures in a row stop.
- Late backend faults (`tests/backend_fault.rs`, a wrapper around the `cpu` backend only): faults raised from `equal`, `export`, and `ll_step`, including at the initial load, are rolled back with a correct result; persistent faults end in an execution error, never a verdict.
- Interruption: simulated abrupt termination at k = 77 (mid block), 128 and 129 (at and after a check), 639, 640, 641 (before, at, after a proof point), 1279 (the final residue), 1280 (the final check), and a budget stop; every resumed run reached the same result and a byte-identical, accepted proof.
- Proof durability: a proof residue write failure at k = 680 stops the run; the checkpoint on disk is at 640; resume completes with an accepted proof. A deleted durable residue makes resume refuse.
- Proof rejection: altered B or middles (rejected by the final check), truncated, trailing data, wrong exponent, unknown version, bits above p; each a distinct error.
- Checkpoints: corrupted bytes anywhere fall back to `.prev`; a crash between rotation and rename falls back to `.prev`; oversized metadata is refused before allocation; a different backend, configuration, exponent, or protocol is refused; a checkpoint with a valid hash but a wrong residue is refused by the on-load Gerbicz check.
- LL Jacobi: a symbol flipping fault at a check is always caught; over 64 single bit flips on M521, the detected ones recover and the undetected ones end with a wrong final residue (the test asserts both), which is the documented limit of the LL check.
- Cross backend: `cpu`, `dwt`, and `gpu` give identical full final residues and byte-identical proof files for M1279 and M1277.

## End-to-end CLI demonstration (default build, release)

Commands run from the repository root after `cargo build --release`. Output pasted verbatim.

```text
$ ./target/release/mersenne-forge run --test ll --exponent 4423 --backend cpu --log-step 1000 --jacobi every-log
M4423 LL: prime (LL criterion satisfied), res64 0000000000000000
jacobi checks: 5 passed, 0 failed and rolled back, 0 inconclusive
note: the Jacobi check catches about half of all errors; it is not a Gerbicz check
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test ll --exponent 4421 --backend cpu --log-step 1000 --jacobi every-log
M4421 LL: composite, res64 436652647e1e860b
jacobi checks: 5 passed, 0 failed and rolled back, 0 inconclusive
note: the Jacobi check catches about half of all errors; it is not a Gerbicz check
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 4423 --backend cpu --block-size 200 --check-step 400 --proof-power 6 --proof-dir work/proof --checkpoint-dir work/ckpt
M4423 PRP: probable prime (base 3), res64 0000000000000001
gerbicz checks: 12 passed, 0 failed and rolled back
proof: work/proof/4423-6.proof (power 6, self verification accepted)
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 4421 --backend cpu --block-size 200 --check-step 400 --proof-power 6 --proof-dir work/proof --checkpoint-dir work/ckpt
M4421 PRP: composite, res64 b45cf72f1682620c
gerbicz checks: 12 passed, 0 failed and rolled back
proof: work/proof/4421-6.proof (power 6, self verification accepted)
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 44497 --backend dwt --block-size 200 --check-step 2000 --proof-power 8 --proof-dir work/proof --checkpoint-dir work/ckpt --budget 2s
incomplete at iteration 16600: stop requested or budget exhausted
resume with: mersenne-forge resume --checkpoint work/ckpt/M44497-prp.ckpt
gerbicz checks: 10 passed, 0 failed and rolled back
[exit 3]

$ ./target/release/mersenne-forge resume --checkpoint work/ckpt/M44497-prp.ckpt --budget 10m
M44497 PRP: probable prime (base 3), res64 0000000000000001
gerbicz checks: 16 passed, 0 failed and rolled back
proof: work/proof/44497-8.proof (power 8, self verification accepted)
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 44497 --backend dwt --block-size 200 --check-step 2000 --proof-power 8 --proof-dir work/proof-uninterrupted
M44497 PRP: probable prime (base 3), res64 0000000000000001
gerbicz checks: 24 passed, 0 failed and rolled back
proof: work/proof-uninterrupted/44497-8.proof (power 8, self verification accepted)
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ cmp work/proof/44497-8.proof work/proof-uninterrupted/44497-8.proof
[exit 0]

$ ./target/release/mersenne-forge verify --proof work/proof/44497-8.proof --backend cpu
proof work/proof/44497-8.proof: accepted
M44497 probable prime: the proof verifies the PRP computation, not deterministic primality
[exit 0]

$ python3 (flip one bit of a middle residue into work/proof/corrupt.proof)

$ ./target/release/mersenne-forge verify --proof work/proof/corrupt.proof --backend cpu
proof work/proof/corrupt.proof: rejected
proof rejected: the final check does not hold
[exit 7]

$ ./target/release/mersenne-forge verify --proof work/proof/44497-8.proof --backend cpu --exponent 44483
proof work/proof/44497-8.proof: rejected
proof is for M44497, expected M44483
[exit 7]

$ ./target/release/mersenne-forge verify --proof work/proof/truncated.proof --backend cpu
proof work/proof/truncated.proof: rejected
proof truncated: 3000 bytes, header requires 3924
[exit 7]
```

(`work/proof/truncated.proof` is the first 3000 bytes of `work/proof/4423-6.proof`.)

Ctrl-C: an LL run of M86243 on `dwt` with `--checkpoint-dir work/ctrlc` received SIGINT after 2 seconds, printed `stop requested; finishing at the next valid boundary`, exited 3 with JSON process state `{"state": "incomplete", "k": 16000, "checkpoint": "work/ctrlc/M86243-ll.ckpt"}`, and `mersenne-forge resume --checkpoint work/ctrlc/M86243-ll.ckpt` then reported `M86243 LL: prime (LL criterion satisfied)` with exit 0.

Bench: `mersenne-forge bench --plan bench/cpu-smoke.json --output work/bench/cpu-smoke.json` wrote 16 observations (4 warmups, 12 measured), all completed. These timings are a harness check, not a performance result.

## GPU demonstration (Apple M2, Metal, `--features gpu`)

`cargo test --release --features gpu --test protocol_gpu`: 11 passed, 0 failed (the full generic protocol suite above on the `gpu` backend, plus the cpu, dwt, gpu identity test).

```text
$ ./target/release/mersenne-forge doctor
mersenne-forge 0.1.0 (release)
  cpu   exact bigint, Mersenne folding                exponents 2..=4294967295
  dwt   f64 irrational-base DWT, roundoff checked     exponents 2..=50331648
  gpu   exact integer NTT on the GPU via wgpu         exponents 8..=15503945
        device 0: Apple M2
  cuda  CUDA kernels via cudarc                       not built
note: listing a device is not validation of its arithmetic
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 44497 --backend gpu --block-size 200 --check-step 2000 --proof-power 8 --proof-dir work/gpu-proof --budget 10m
M44497 PRP: probable prime (base 3), res64 0000000000000001
gerbicz checks: 24 passed, 0 failed and rolled back
proof: work/gpu-proof/44497-8.proof (power 8, self verification accepted)
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ cmp work/gpu-proof/44497-8.proof work/proof-uninterrupted/44497-8.proof
[exit 0]

$ ./target/release/mersenne-forge run --test ll --exponent 44497 --backend gpu --log-step 5000 --jacobi every-log --budget 10m
M44497 LL: prime (LL criterion satisfied), res64 0000000000000000
jacobi checks: 9 passed, 0 failed and rolled back, 0 inconclusive
note: the Jacobi check catches about half of all errors; it is not a Gerbicz check
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 1279 --backend gpu
error: the gpu backend requires a bounded --budget
[exit 4]

$ ./target/release/mersenne-forge verify --proof work/proof/44497-8.proof --backend gpu --budget 5m
proof work/proof/44497-8.proof: accepted
M44497 probable prime: the proof verifies the PRP computation, not deterministic primality
[exit 0]
```

Composite check: PRP of M44483 with the same settings gave `composite`, res64 `752add05405ed00b`, 24 Gerbicz checks passed, on both `gpu` and `dwt`.

## CUDA

`cargo build --release --features cuda` succeeds here. `mersenne-forge run --test prp --exponent 44497 --backend cuda --budget 1m` exits 4 with `unsupported: CUDA driver library not found ...; NVRTC library not found ...`. `cargo test --features cuda --test protocol_cuda` fails all 11 tests (by design: they fail, never skip, without a device). No CUDA arithmetic has ever run.

## NOT validated, and the commands to run

Every item below is unvalidated. Run on a Linux host with an NVIDIA GPU, a current driver, and the CUDA toolkit (for NVRTC) installed:

```sh
cargo build --release --features cuda,gpu
./target/release/mersenne-forge doctor --json
cargo test --release --features cuda --test protocol_cuda
cargo test --release --features cuda --test cuda_backend
cargo test --release --features gpu --test protocol_gpu          # wgpu on Vulkan
./target/release/mersenne-forge run --test prp --exponent 44497 --backend cuda --device 0 \
    --block-size 200 --check-step 2000 --proof-power 8 --proof-dir work/cuda-proof --budget 30m
cmp work/cuda-proof/44497-8.proof work/proof-uninterrupted/44497-8.proof   # after the dwt run above
./target/release/mersenne-forge run --test prp --exponent 1257787 --backend cuda --device 0 \
    --proof-power 8 --proof-dir work/cuda-proof --checkpoint-dir work/cuda-ckpt --budget 4h
./target/release/mersenne-forge verify --proof work/cuda-proof/1257787-8.proof --backend dwt
```

For AMD or Intel GPUs, run the `gpu` lines (Vulkan) the same way.

Upstream verifier interoperability (our proofs accepted by PRPLL): build PRPLL at commit `90a3d8db84e742773f45e2adfba3131199fa8101` per its README, generate a proof here for an exponent inside the upstream FFT range (for example the M1257787 run above), and run the upstream verifier with its documented interface `prpll -verify <file>`. This has not been done.

Performance: no measurement exists. See [BENCHMARK.md](BENCHMARK.md).
