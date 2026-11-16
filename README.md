<p align="center"><img src="assets/logo.svg" alt="MersenneForge" width="160"></p>

# MersenneForge

MersenneForge is a Rust engine for Lucas-Lehmer (LL) and base-3 probable prime (PRP) tests of Mersenne numbers M_p = 2^p - 1, with Gerbicz error checking, PRP proofs, and durable checkpoints.

The LL, PRP, Gerbicz check, proof, and checkpoint state machines are written once in Rust over a small arithmetic trait (`Backend`). Backends only do modular arithmetic; Rust owns the iteration count, check boundaries, rollback, checkpoint decisions, and result classification.

Everything in this repository is Rust (plus WGSL and CUDA C kernel sources embedded by the GPU backends). There is no linked native library and no upstream code. The protocol semantics, the Gerbicz check, and the PRP proof file format follow PRPLL (gpuowl), studied at commit `90a3d8db84e742773f45e2adfba3131199fa8101`; that reference checkout lives outside this repository.

## Backends

| Backend | Feature | Arithmetic | Status |
|---|---|---|---|
| `cpu` | default | exact bigint, Mersenne folding | implemented; CPU tested |
| `dwt` | default | f64 irrational-base DWT, roundoff checked | implemented; CPU tested |
| `gpu` | `gpu` | exact integer NTT over GF(M31^2) via `wgpu` | implemented; validated only on an Apple GPU through Metal (see [docs/GPU.md](docs/GPU.md)); not validated on NVIDIA, AMD, or Intel |
| `cuda` | `cuda` | CUDA C kernels via `cudarc` | implemented; never run on CUDA hardware |

## Status

| Area | Status |
|---|---|
| Exact CPU oracle, reference LL and PRP | implemented, CPU tested |
| Checked PRP (Gerbicz), rollback, stop conditions | implemented, CPU tested on `cpu` and `dwt`, run on `gpu` (Metal) |
| LL with Jacobi check | implemented, CPU tested |
| Checkpoints (atomic, versioned, hashed) and resume | implemented, CPU tested |
| PRP proof generation and verification (format v2) | implemented, CPU tested; an upstream generated proof verifies here |
| Acceptance of our proofs by an upstream verifier | not tested |
| GPU arithmetic | implemented; validated only on an Apple GPU through Metal; not validated on NVIDIA, AMD, or Intel |
| CUDA arithmetic | implemented; never run on CUDA hardware |
| Performance | harness only; no performance comparison has been measured |

CI runs the CPU checks and only compiles the `gpu` and `cuda` features; GPU and CUDA tests are run by hand. Details, commands, and observed results: [docs/VALIDATION.md](docs/VALIDATION.md). Protocol and formats: [docs/PROTOCOL.md](docs/PROTOCOL.md). Benchmark method: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Build

Rust 1.85 or newer builds the library and binary (the test suite needs 1.88 because of `proptest`).

```sh
cargo build --release                  # cpu and dwt backends
cargo build --release --features gpu   # adds the wgpu backend
cargo build --release --features cuda  # adds the CUDA backend (loads libcuda and NVRTC at run time)
cargo test
```

## Usage

```sh
mersenne-forge doctor --json
mersenne-forge run --test ll --exponent 4423 --backend cpu
mersenne-forge run --test prp --exponent 44497 --backend dwt --proof-dir work/proof --checkpoint-dir work/ckpt --budget 1h
mersenne-forge resume --checkpoint work/ckpt/M44497-prp.ckpt --budget 1h
mersenne-forge verify --proof work/proof/44497-8.proof --backend cpu
mersenne-forge bench --plan bench/cpu-smoke.json --output work/bench/cpu-smoke.json
```

`--backend gpu` and `--backend cuda` require `--budget`; `--device N` selects the adapter index or CUDA ordinal. Add `--json` for machine-readable output. Exit codes: 0 success, 2 usage, 3 incomplete (budget or Ctrl-C; a checkpoint is written when a checkpoint directory was given), 4 unsupported, 5 execution failure, 6 checkpoint or file error, 7 proof rejected.

A PRP pass means probable prime. A verified proof establishes that the PRP computation was done, not that M_p is prime. An LL zero residue satisfies the LL criterion for the arithmetic used. No local result is a discovery claim.

## License and credits

GPL-3.0-only (see [LICENSE](LICENSE)). The protocol logic was written by studying the GPLv3 source of PRPLL/gpuowl by Mihai Preda and George Woltman, including its CUDA backend by "Sherpa". The SHA3 hash is the `sha3` crate; big integers are `num-bigint`; FFTs are `rustfft`.

Copyright (C) 2026 Adam R Sophie.
