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

CI runs CPU tests, CUDA model and kernel host emulation, strict documentation checks, and feature compilation. GPU hardware tests are run by hand. Details, commands, and observed results: [docs/VALIDATION.md](docs/VALIDATION.md). Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Protocol and formats: [docs/PROTOCOL.md](docs/PROTOCOL.md). Benchmark method: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Build

Rust 1.89 or newer builds and tests every feature (1.89 is the first version with `std::fs::File::try_lock`, used for the checkpoint lock; it also covers the minimums of `wgpu` and `proptest`). CI builds and tests at 1.89.

```sh
cargo build --release                  # cpu and dwt backends
cargo build --release --features gpu   # adds the wgpu backend
cargo build --release --features cuda  # adds the CUDA backend (loads libcuda and NVRTC at run time)
cargo test
```

## Usage

```sh
mersenne-forge doctor --json --exponent 44497
mersenne-forge run --test ll --exponent 4423 --backend cpu
mersenne-forge run --test prp --exponent 44497 --backend dwt --proof-power 8 --proof-dir work/proof --checkpoint-dir work/ckpt --budget 1h
mersenne-forge resume --checkpoint work/ckpt/M44497-prp.ckpt --budget 1h
mersenne-forge resume --checkpoint moved/ckpt/M44497-prp.ckpt --proof-dir moved/proof --budget 1h
mersenne-forge verify --proof work/proof/44497-8.proof --backend cpu
mersenne-forge bench --plan bench/cpu-smoke.json --output work/bench/cpu-smoke.json
```

`--backend gpu` and `--backend cuda` require `--budget`; `--device N` selects the adapter index or CUDA ordinal. Add `--json` for machine-readable output (failures too, as a JSON object with `kind`, `message`, and `exit_code`). Exit codes: 0 success, 2 usage, 3 incomplete (budget or Ctrl-C, also during the proof stage; a checkpoint is written when a checkpoint directory was given), 4 unsupported, 5 execution failure (also a failed proof generation or write), 6 checkpoint or file error, 7 proof rejected (also a rejected self verification). `run` refuses to start over an existing checkpoint: continue with `resume`, or delete the checkpoint and its `.prev` to start over. A run or resume holds a lock on `<checkpoint>.lock`, and a PRP run with a proof directory also locks `M<p>.residues/.lock` there, so a second process on the same checkpoint or proof store is refused.

PRP runs save a completed checkpoint before proof generation. If proof generation, writing, or verification fails or stops, `resume` validates that completed state and restarts the proof stage without repeating the final PRP check interval. Move the entire proof directory, including `M<p>.residues`, when relocating work; `resume --proof-dir` updates its location and checks all required saved residues. It preserves the original proof power and cannot add a proof schedule to a run that had none.

Build identities include a SHA3-256 fingerprint of the implementation, manifest, and lockfile, including uncommitted source changes. Checkpoints require the same build identity. Finish existing runs with their original binary before upgrading; checkpoints written by older version-only builds are refused. Proof files remain portable across backends and builds.

A PRP pass means probable prime. A proof accepted by `verify` below p = 2^17 has had its terminal residue recomputed directly. At or above that, acceptance means the file satisfies the version 2 proof equation, the relation the upstream verifier checks: that is sound against computation errors and against a prover who skipped the work, but not against a prover who knows a factor of M_p or elements of small order modulo M_p (the challenge is 64 bits), so a probable prime claim from a proof needs an independent test (see [docs/PROTOCOL.md](docs/PROTOCOL.md)). No proof establishes that M_p is prime. An LL zero residue satisfies the LL criterion for the arithmetic used. No local result is a discovery claim.

## License and credits

GPL-3.0-only (see [LICENSE](LICENSE)). The protocol logic was written by studying the GPLv3 source of PRPLL/gpuowl by Mihai Preda and George Woltman, including its CUDA backend by "Sherpa". The SHA3 hash is the `sha3` crate; big integers are `num-bigint`; FFTs are `rustfft`.

Copyright (C) 2026 Adam R Sophie.
