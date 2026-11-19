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
| Assignment queue (`work`), worktodo and result lines, factor checks | implemented, CPU tested; never submitted to PrimeNet |
| Trial factoring, P-1, ECM | implemented on the CPU (TF), any backend plus host (P-1 stage 1 and stage 2), and host (ECM stage 1); suited to small exponents; result line keys for TF not checked against another program |
| PrimeNet client (`primenet`) | written, never run against the server |
| Several workers (`supervise`), status file, `status` | implemented |
| Opt-in transform variants (packed DWT, fused radix 4 GPU stages, CUDA graph replay) and `tune` | written; the CUDA variant has never run |
| Performance | harness with production mode, retained artifacts, power telemetry, and paired analysis; no performance comparison has been measured |

CI runs CPU tests, CUDA model and kernel host emulation, strict documentation checks, and feature compilation. GPU hardware tests are run by hand. Details, commands, and observed results: [docs/VALIDATION.md](docs/VALIDATION.md). Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Protocol and formats: [docs/PROTOCOL.md](docs/PROTOCOL.md). Benchmark method: [docs/BENCHMARK.md](docs/BENCHMARK.md). Every capability and what is missing: [docs/STATUS.md](docs/STATUS.md).

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
mersenne-forge bench-render --input work/bench/cpu-smoke.json --output work/bench/cpu-smoke.csv
mersenne-forge bench-analyze --input work/bench/cpu-smoke.json --baseline 0 --candidate 1
mersenne-forge work --worktodo worktodo.txt --results results.json.txt --checkpoint-dir work/ckpt --proof-dir work/proof --backend dwt
mersenne-forge check-factor --exponent 11 --factor 23
mersenne-forge tf --exponent 67 --from 27 --to 28
mersenne-forge pm1 --exponent 67 --b1 1000 --b2 3000
mersenne-forge ecm --exponent 67 --b1 2000 --curves 40
mersenne-forge checkpoint-export --input work/ckpt/M44497-prp.ckpt --output M44497.portable.ckpt
mersenne-forge resume --checkpoint M44497.portable.ckpt --backend cpu
mersenne-forge supervise --config workers.json
mersenne-forge status --worktodo worktodo.txt --results results.json.txt --checkpoint-dir work/ckpt
mersenne-forge tune --exponent 1000003 --backend dwt
mersenne-forge power --source rapl
```

`work` takes `PRP=`, `PRPDC=`, `Test=`, and `DoubleCheck=` lines for Mersenne numbers (the forms upstream PRPLL accepts) and `Factor=`, `Pminus1=`, and `Pfactor=` lines, runs or resumes each in turn, appends a result line in upstream's JSON form to the results file and a journal entry to `<results>.done` (both fsynced, with their directory), and only then removes the assignment. A PRP line whose tests_saved field is above 0 gets P-1 stage 1 first, as upstream does: a factor ends the assignment, otherwise the line is rewritten with tests_saved 0 and the test follows. Lines it cannot run (CERT, other number forms, PRP bases other than 3, residue types other than 1, known factor lists, composite exponents, P-1 above p = 10,000,000) stay in the file. A line already in the journal, or whose assignment ID already has a result, is removed without running again; a fragment a crash left at the end of the results file is moved to `<results>.torn` before the next line is written. Before starting a PRP with a proof it checks that the proof points and the proof file fit on disk. An assignment that cannot start or fails (the backend refuses it, too little disk, a checkpoint from another build or another assignment, a missing proof point, a protocol error) stays in the file with a message, later assignments still run, and the session exits 6. A proof that fails or is rejected does not withhold a verified result: the line is reported without a proof, as upstream does. Checkpoints are named by assignment ID (or a hash of the line), and an interrupted assignment resumes from its own checkpoint, on the backend that wrote it, with its original settings. After a durable result it deletes that assignment's checkpoints and, once the proof is verified, its proof points; the proof file stays for upload and an existing proof file is never overwritten. The worktodo and results files are edited under `<file>.lck` locks in AutoPrimeNet's convention (interoperation with AutoPrimeNet has not been tested). One process per worktodo file (it is locked); `supervise` runs one per device from a JSON list of workers. `--status-file` keeps a JSON health state. `work` sends nothing to any server; `primenet` (separate, never run against the server) sends result lines, fetches assignments, and uploads proofs, through the system `curl`, and only to mersenne.org over https.

`--backend gpu` and `--backend cuda` require `--budget`; `--device N` selects the adapter index or CUDA ordinal. Add `--json` for machine-readable output (failures too, as a JSON object with `kind`, `message`, and `exit_code`). Exit codes: 0 success, 2 usage, 3 incomplete (budget or Ctrl-C, also during the proof stage; a checkpoint is written when a checkpoint directory was given), 4 unsupported, 5 execution failure (also a failed proof generation or write), 6 checkpoint or file error, 7 proof rejected (also a rejected self verification). `run` refuses to start over an existing checkpoint: continue with `resume`, or delete the checkpoint and its `.prev` to start over. A run or resume holds a lock on `<checkpoint>.lock`, and a PRP run with a proof directory also locks `M<p>.residues/.lock` there, so a second process on the same checkpoint or proof store is refused.

PRP runs save a completed checkpoint before proof generation. If proof generation, writing, or verification fails or stops, `resume` validates that completed state and restarts the proof stage without repeating the final PRP check interval. Move the entire proof directory, including `M<p>.residues`, when relocating work; `resume --proof-dir` updates its location and checks all required saved residues. It preserves the original proof power and cannot add a proof schedule to a run that had none.

Build identities include a SHA3-256 fingerprint of the implementation, manifest, build script, and lockfile, including uncommitted source changes. Checkpoints require the same build identity and backend. To move a run to another build or backend, `checkpoint-export` writes its last verified state (an LL state rewinds to its last Jacobi verified residue) without that binding, and `resume --backend` (or `work`, from the assignment's checkpoint path) continues it. Proof files remain portable across backends and builds.

Opt-in transform variants, each with its own plan name so checkpoints never mix them: `MERSENNE_FORGE_DWT_PACKED=1` (DWT packed real transform, about half the transform work), `MERSENNE_FORGE_GPU_RADIX4=1` (two NTT stages per GPU pass, bit identical to one pass per stage), `MERSENNE_FORGE_DWT_THREADS=N` (DWT four step FFT on N threads), `MERSENNE_FORGE_GPU_PACKED=1` (two words per GPU transform element, half length transform), `MERSENNE_FORGE_CUDA_GRAPHS=1` (CUDA graph replay of squaring batches; never run), `MERSENNE_FORGE_CUDA_NARROW=1` (32-bit CUDA registers; never run). `tf --gpu` tests trial factoring candidates below 2^63 on the GPU. `tune` times the default against the variants and checks that their residues agree.

A PRP pass means probable prime. A proof accepted by `verify` below p = 2^17 has had its terminal residue recomputed directly. At or above that, acceptance means the file satisfies the version 2 proof equation, the relation the upstream verifier checks: that is sound against computation errors and against a prover who skipped the work, but not against a prover who knows a factor of M_p or elements of small order modulo M_p (the challenge is 64 bits), so a probable prime claim from a proof needs an independent test (see [docs/PROTOCOL.md](docs/PROTOCOL.md)). The proof equation accepts a terminal residue B and its negation alike, so when B is 9 or N - 9 the sign decides the claim and `verify` reports it as undetermined unless the terminal residue was recomputed. No proof establishes that M_p is prime. An LL zero residue satisfies the LL criterion for the arithmetic used. No local result is a discovery claim.

## License and credits

GPL-3.0-only (see [LICENSE](LICENSE)). The protocol logic was written by studying the GPLv3 source of PRPLL/gpuowl by Mihai Preda and George Woltman, including its CUDA backend by "Sherpa". The SHA3 hash is the `sha3` crate; big integers are `num-bigint`; FFTs are `rustfft`.

Copyright (C) 2026 Adam R Sophie.
