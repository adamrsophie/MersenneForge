# Capability status

Each capability is marked **done** (implemented and tested on this project's own hosts), **written** (code and tests written, not yet compiled or run; see the note below), **partial**, **blocked** (implemented as far as possible without hardware or a tool this project has not had access to), or **not started**. "Tested" names the evidence in [VALIDATION.md](VALIDATION.md). Nothing marked written, blocked, or not started is claimed to work.

Note: the changes after the validation record of 2026-11-19 (the rows marked written, and the fixes they list) were written while the build host was too loaded to compile or test. Until a new validation record lists them, treat them as unverified.

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
| F11 | Benchmark provenance | written | adds the compiler version (build script) and the power policy where readable; driver, NVRTC, and kernel binary details stay with the backend identity |
| F12 | Retained proofs and verified outputs | partial | `retain` keeps proof and report with hashes under a quota; external verification adapter tested with a stand in verifier only |
| F13 | Production workload mode | done | `mode: production`: checkpoints, proof points, proof, self verification |
| F14 | Real device validation per platform | blocked | Apple M2 through Metal only; no NVIDIA, AMD, or Intel GPU |
| F15 | Backend ranges and capacity | done | `doctor`, memory estimates |
| F16 | Tuned baseline and statistical comparison | partial | paired bootstrap analysis (`bench-analyze`); no tuned PRPLL baseline has been run |
| F17 | Power and energy telemetry | written | built in sensors (`power --source rapl, nvidia, powermetrics`); sampling bounded and limited to the workload; no sensor has been run (RAPL and powermetrics need root here, no NVIDIA device) |
| F18 | CUDA graph replay | blocked | written behind `MERSENNE_FORGE_CUDA_GRAPHS=1`; never run (no NVIDIA device) |
| F19 | Fused and higher radix GPU stages | written | radix 4 DIF and DIT passes behind `MERSENNE_FORGE_GPU_RADIX4=1`, bit identical by construction, with a test against the single stage path; not yet run or timed on Metal |
| F20 | Narrower CUDA buffers | blocked | not changed: needs an NVIDIA device to compare before and after |
| F21 | Packed (real or right angle) transforms | partial | DWT packed real transform written (see F23); the GPU NTT does not pack yet |
| F22 | Plan and kernel autotuning | written | `tune` times the default against the opt-in variants and checks that residues agree; it recommends a setting, it does not switch automatically |
| F23 | Faster CPU transform (packing, threads) | partial | packed real transform behind `MERSENNE_FORGE_DWT_PACKED=1`, tested against the CPU reference (written); no threads (the transform library is single threaded) |
| F24 | Asynchronous proof and checkpoint writes | written | proof points are written by a background thread with a bounded queue, flushed before every checkpoint and the proof stage; checkpoint writes stay synchronous |
| F25 | Several GPUs, one assignment each | written | `supervise` runs one `work` process per configured worker, refuses shared queues and devices, restarts crashed workers |
| F26 | Durable queue and result outbox | done, fixes written | done as validated; the later fixes (completion journal, keyed checkpoints, `.lck` locks, per assignment failures, reporting without a failed proof) are written |
| F27 | Worktodo and result formats, PrimeNet | partial | upstream line forms and result fields from the pinned source, with fixtures; `primenet` client written from upstream's former scripts, never run against the server; no AutoPrimeNet test |
| F28 | Independent factor checks | done | `check-factor` |
| F29 | Trial factoring | written | `tf` and `Factor=` lines on the CPU, sieved, with saved progress; slow above 2^64 |
| F30 | P-1 | written | `pm1`, `Pminus1=`, `Pfactor=`, and P-1 before PRP lines with tests_saved; stage 1 on any backend with saved progress, stage 2 on the host; `work` limits P-1 to p <= 10,000,000 |
| F31 | ECM | written | `ecm`: stage 1 on Montgomery curves on the host, small exponents only; no stage 2 |
| F32 | Native factoring kernels | not started | factoring runs on the CPU (TF, ECM) or the host plus a backend (P-1); no GPU TF kernel |
| F33 | Disk preflight, retention, cleanup | written | as before, plus upload state: `primenet` moves uploaded proofs to `uploaded/` and keeps sent and refused result logs |
| F34 | Structured logs and health state | written | `work --status-file` JSON health state and `status`; no metrics endpoint |
| F35 | Cross version checkpoint migration | written | `checkpoint-export` writes the last verified state without its backend binding; `resume --backend` and `work` continue it on any backend or build |
| F36 | Releases and platform CI | written | release workflow builds, tests, and drafts tagged releases for Linux and macOS (cpu and gpu builds); not yet run |
| F37 | TUI or web interface | partial | `status --watch` text view of queue, checkpoints, and workers; no TUI |
| F38 | Search policy research | not started | optional |
