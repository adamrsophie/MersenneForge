# Capability status

Each capability is marked **done** (implemented and tested on this project's own hosts), **partial**, **blocked** (implemented as far as possible without hardware or a tool this project has not had access to), or **not started**. "Tested" names the evidence in [VALIDATION.md](VALIDATION.md). Nothing marked blocked or not started is claimed to work.

| ID | Capability | Status | Evidence or what is missing |
|---|---|---|---|
| F01 | Checked base 3 PRP | done | CPU, DWT, and Metal tests; independent big-integer sweep |
| F02 | LL with Jacobi check | done | the Jacobi check covers fewer error kinds than Gerbicz (see PROTOCOL.md) |
| F03 | Independent oracle and differential tests | done | `oracle`, reference vectors, 688 case sweep |
| F04 | Proof generation and local verification | done | format version 2; upstream generated proof verifies here |
| F05 | Our proofs accepted by an upstream verifier | blocked | `tests/upstream_verifier.rs` and the benchmark `verify_with` adapter exist; the upstream verifier does not run on the available Mac (no FP64, OpenCL compiler abort) |
| F06 | Atomic writes, file ownership, store locks | done | exclusive temporary files, proof store and checkpoint locks |
| F07 | Stop, restart, proof stage recovery | done | kill and resume tests; completed state resume |
| F08 | Fault detection and rollback; trustworthy harnesses | done | fault injection; emulation drivers must exit 0 |
| F09 | Bounded parsing and allocation errors | done | fault injecting allocator test |
| F10 | Benchmark JSON and CSV records | done | strict CSV parse test; failures kept |
| F11 | Benchmark provenance | partial | executable hash, source identity, profile, target, features, lockfile hash, CPU model; compiler version, driver, NVRTC, kernel binary, and power policy are recorded as unknown |
| F12 | Retained proofs and verified outputs | partial | `retain` keeps proof and report with hashes under a quota; external verification adapter tested with a stand in verifier only |
| F13 | Production workload mode | done | `mode: production`: checkpoints, proof points, proof, self verification |
| F14 | Real device validation per platform | blocked | Apple M2 through Metal only; no NVIDIA, AMD, or Intel GPU |
| F15 | Backend ranges and capacity | done | `doctor`, memory estimates |
| F16 | Tuned baseline and statistical comparison | partial | paired bootstrap analysis (`bench-analyze`); no tuned PRPLL baseline has been run |
| F17 | Power and energy telemetry | partial | sampled power command with trapezoid energy, scope, and missing readings; no real sensor has been attached (none readable without root on the Mac) |
| F18 | CUDA graph replay | blocked | needs an NVIDIA device and profiles |
| F19 | Fused and higher radix GPU stages | blocked | needs measurement on the target device |
| F20 | Narrower CUDA buffers | blocked | needs hardware comparison before any change |
| F21 | Packed (real or right angle) transforms | not started | needs a design note and exact model first |
| F22 | Plan and kernel autotuning | not started | depends on F18 to F21 |
| F23 | Faster CPU transform (packing, threads) | not started | |
| F24 | Asynchronous proof and checkpoint writes | not started | proof point writes are synchronous and now timed separately (`proof_points`) |
| F25 | Several GPUs, one assignment each | partial | one `work` process per device and worktodo file (each worktodo is locked); no supervisor |
| F26 | Durable queue and result outbox | done | `work`: results fsynced before the assignment is removed; an assignment already in the results file is not run again |
| F27 | Worktodo and result formats, PrimeNet | partial | upstream line forms and result fields from the pinned source, with fixtures; never submitted to a server; no AutoPrimeNet test |
| F28 | Independent factor checks | done | `check-factor` |
| F29 | Trial factoring adapters | not started | |
| F30 | P-1 | not started | P-1 lines are left in the worktodo file |
| F31 | ECM | not started | |
| F32 | Native factoring kernels | not started | |
| F33 | Disk preflight, retention, cleanup | partial | `work` refuses an assignment whose proof does not fit on disk, keeps proof files, and removes checkpoints and proof points after a durable result; no upload state |
| F34 | Structured logs and health state | partial | JSON output and failures; no metrics endpoint |
| F35 | Cross version checkpoint migration | not started | intentionally refused: checkpoints need the same source identity |
| F36 | Releases and platform CI | partial | Linux CI and minimum Rust version; macOS CI runners rarely start; no release builds |
| F37 | TUI or web interface | not started | optional |
| F38 | Search policy research | not started | optional |
