# Architecture

MersenneForge is one Rust crate with a library and a CLI. Rust owns protocol state, checking, rollback, durability, and result classification. Arithmetic backends share the same protocol code.

| Module | Responsibility |
|---|---|
| `residue` | Canonical little-endian residues, unique zero, proof encoding, reported residues |
| `exponent`, `oracle` | Prime-exponent validation and independent exact reference arithmetic using generic modular reduction |
| `backend` | Resident-register arithmetic contract and CPU, DWT, wgpu, and CUDA implementations |
| `select` | Feature-gated backend construction and forwarding through `Box<dyn Backend>` |
| `prp` | Base-3 recurrence, Gerbicz checks, rollback, proof-point saves, and completed-state recovery |
| `ll` | Lucas-Lehmer recurrence, Jacobi checks, and rollback |
| `proof` | Proof schedules, durable residue storage, version 2 generation, parsing, and verification |
| `checkpoint` | Bounded, hashed project files, atomic writes, previous-file rotation, and directory durability |
| `run` | Stop control, fault injection, process state, mathematical results, evidence, and timings |
| `bench` | Declared workload plans, engine and external observations, incremental JSON or CSV output, steady state and production modes, retention, external proof verification |
| `evidence` | Benchmark provenance, power telemetry sampling and energy, paired analysis with a bootstrap interval |
| `worktodo` | Assignment lines, result lines in the upstream JSON form, proof MD5, disk preflight, factor checks |
| `main` | CLI parsing, checkpoint locks, Ctrl-C handling, JSON output, and exit codes |

Backends allocate instance-owned register handles and expose import, export, copy, squaring, multiplication, LL steps, equality, and synchronization. Only canonical `Residue` values cross this boundary. Transform words and device memory stay inside the backend. Recoverable backend faults trigger protocol rollback; unsupported operations and execution failures never become composite verdicts.

The CPU backend uses exact bigint arithmetic with Mersenne folding. The independent oracle uses generic `%` reduction and `modpow`. DWT uses an f64 irrational-base transform and rejects excessive roundoff. The wgpu backend uses exact integer transforms over GF((2^31 - 1)^2); CUDA uses GF((2^61 - 1)^2). GPU host code is Rust; WGSL and CUDA C kernels are authored here and embedded in the crate. No upstream source or linked upstream library is required.

For PRP, a proof point must become durable before a later checked checkpoint can claim it. A completed checkpoint binds the terminal residue to the final checked state, allowing proof-stage recovery without repeating the recurrence. Files carry integrity hashes, which detect damage but authenticate neither the producer nor trusted iteration metadata. Build identities include implementation source fingerprints and prevent automatic checkpoint reuse across source changes.

Protocol and format details, including mathematical assurance limits: [PROTOCOL.md](PROTOCOL.md). Hardware evidence and remaining validation gates: [VALIDATION.md](VALIDATION.md). Backend details: [GPU.md](GPU.md), [CUDA.md](CUDA.md). Benchmark rules: [BENCHMARK.md](BENCHMARK.md).
