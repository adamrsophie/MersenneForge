# Validation report

Historical results below used Apple M2 (macOS 27.2, arm64), Rust 1.100.0 stable. The completion checks dated 2026-11-16 used the same hardware with macOS 27.2.1. No NVIDIA, AMD, or Intel GPU was available. Nothing here was run on CUDA hardware.

## Continuous integration (2026-11-24)

The build host was too loaded to compile or test after 2026-11-19, so GitHub Actions did both. Runs (repository `adamrsophie/MersenneForge`, workflow `ci`): 45443574126 (the backlog, after its fixes), 45455447642 (P-1 stage 2 pairing), 45456831356 (`ECMSTAGE2=`), 45457319409 and 45457651572 (`status --serve`), and 45458391415 (device sieve, commit `f9a0d94`). The `ubuntu-latest` and `msrv` jobs passed in every one; the `macos-latest` job was never given a runner and proves nothing.

| Command (ubuntu-latest, Rust 1.100.0, run 45458391415) | Observed result |
|---|---|
| `cargo build --locked --all-targets`; `cargo test --locked` | 176 passed, 0 failed, 5 ignored by default |
| `cargo clippy --locked --all-targets -- -D warnings`; `cargo fmt --check`; `cargo doc --locked --no-deps --features gpu,cuda` | passed (clippy on default features only) |
| `cargo check --locked --features gpu --all-targets`; same with `--features cuda` | passed; compiled only |
| `cargo test --locked --features cuda --lib backend::cuda`; `--test cuda_kernel_emulation` | 20 and 6 passed; CPU model and emulation, not CUDA hardware |
| `msrv` job: `cargo test --locked` and `cargo check --locked --all-targets --features gpu,cuda` on Rust 1.89 | passed |

New since 2026-11-19 and covered by these tests: PRP residue types 2 and 4, P-1 stage 2 prime pairing, `ECMSTAGE2=` from GMP-ECM save files, `status --serve` (token and Host check), and the backlog rows marked CI tested in STATUS.md.

Not covered: no GPU test ran (`tests/gpu_backend.rs`, `tests/protocol_gpu.rs`, the TF kernel tests including `segment_kernel_matches_the_host`, lengths 2^23 and 2^24); no CUDA kernel ran on NVIDIA hardware; no sensor, release workflow, or PrimeNet exchange ran.

## Benchmark evidence, work queue, and factor checks (2026-11-19)

Implementation source SHA3-256: `390fa29f7066fc0c46b4e48831a7d1076c4221adc5f7c818157ad7980758c914`; debug executable SHA-256 `79b8d86d8292c0132b8b712310721f1b226d4926860bbdd806144b6f2b40d4af` (Apple M2, macOS 27.2.1, Rust 1.100.0).

New: benchmark modes, retention with a quota, the external proof verification adapter, provenance, the `proof_points` timer and unclassified remainder, power telemetry, `bench-render`, `bench-analyze`, the `work` queue, worktodo and result lines, proof MD5, disk preflight, and `check-factor`. The capability list is in [STATUS.md](STATUS.md). The external verification adapter was exercised only with this program's own `verify` standing in for an independent verifier, and telemetry only with a command printing a fixed value; no real power sensor was read.

| Command | Observed result |
|---|---|
| `cargo test --locked` | 137 passed, 0 failed, 5 ignored by default |
| `cargo clippy --locked --all-targets -- -D warnings`, also with `--features gpu`, `--features cuda`, `--features gpu,cuda` | passed |
| `cargo fmt --check`; `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --features gpu,cuda`; `cargo +1.89 check --locked --all-targets --features gpu,cuda` | passed |
| `cargo test --locked --features gpu --lib --test gpu_backend --test protocol_gpu -- --test-threads=1` | 65 passed, 0 failed, 1 ignored; real Apple M2 through Metal |
| `cargo test --locked --features cuda --lib backend::cuda` | 19 passed; CPU model and host code only |
| `cargo test --locked --features cuda --test cuda_kernel_emulation` | 6 passed; CPU emulation, not CUDA hardware |
| `cargo test --locked --test reference -- --ignored`; upstream fixture test | 1 passed each |
| Independent Python big-integer sweep against the debug CLI | 688 of 688 agree (172 prime exponents up to 1024, LL and PRP, `cpu` and `dwt`) |

Still open: no CUDA kernel has run on NVIDIA hardware; no upstream verifier has checked a proof generated here; no result line has been submitted to PrimeNet; no performance comparison or energy measurement.

## Temporary file ownership and CSV quoting (2026-11-17)

Implementation source SHA3-256: `2283ef92ecfac447ef2a74d6fca7f43b11e8c6b8209b47911633aa0ee060d87f`; debug executable SHA-256 `c5e77bd5bface421bcc432b3d5f098de6d931ce871598438a6b9200b44a77131` (Apple M2, macOS 27.2.1, Rust 1.100.0).

| Problem | Change | Regression test |
|---|---|---|
| When the generated temporary name already existed, `write_atomic` failed and then deleted that file, which it never owned | The writer removes only a file it created; an existing name is left untouched and another name is tried | `existing_temporary_names_are_skipped_and_kept` (fails on the previous code), `a_failed_publication_removes_only_its_own_temporary_file` |
| A test name or other text with a comma, quote, CR, or newline broke the CSV row that recorded its failure | Every CSV field is quoted, quotes doubled | `bench_csv_quotes_every_text_field`: a strict RFC 4180 parse gives one full width record per observation; the failed entry, a following valid entry, and an external argv keep their values (the previous code produced 6 records for 3 observations) |

| Command | Observed result |
|---|---|
| `cargo test --locked` | 122 passed, 0 failed, 5 ignored by default |
| `cargo clippy --locked --all-targets -- -D warnings`, also with `--features gpu`, `--features cuda`, `--features gpu,cuda` | passed |
| `cargo fmt --check`; `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --features gpu,cuda`; `cargo +1.89 check --locked --all-targets --features gpu,cuda` | passed |
| `cargo test --locked --features gpu --lib --test gpu_backend --test protocol_gpu -- --test-threads=1` | 58 passed, 0 failed, 1 ignored; real Apple M2 through Metal |
| `cargo test --locked --features cuda --lib backend::cuda` | 19 passed; CPU model and host code only |
| `cargo test --locked --features cuda --test cuda_kernel_emulation` | 6 passed under AddressSanitizer and UndefinedBehaviorSanitizer; CPU emulation, not CUDA hardware; LeakSanitizer unsupported on this host |
| `cargo test --locked --test reference -- --ignored`; upstream fixture test as below | 1 passed each |
| Independent Python big-integer sweep against the debug CLI | 688 of 688 agree (172 prime exponents up to 1024, LL and PRP, `cpu` and `dwt`) |

Still open: no CUDA kernel has run on NVIDIA hardware; no upstream verifier has checked a proof generated here; no Vulkan or discrete GPU run; no performance comparison or energy measurement.

## Storage, proof reading, and benchmark checks (2026-11-17)

Implementation source SHA3-256: `6e50fefa7278c56f6e78146fb1ac2b30986c82c6c22de8b81d01b9cefbe8c4ca` (Apple M2, macOS 27.2.1, Rust 1.100.0, Apple clang 21.0.0). Checkpoints from earlier builds are refused, as intended. These regression tests were also run against the previous code and failed there: `concurrent_atomic_writers_never_share_a_temporary_file`, `a_held_proof_directory_refuses_a_second_writer`, `huge_batch_sizes_stop_at_the_boundaries`, `proof_allocation` (SIGABRT), and `dwt_allocation`. A driver forced to exit 86 after its last correct response passed the previous emulation harness and now fails `thread_order_and_block_size_do_not_change_results`. Rerunning the earlier benchmark scenarios (two entries named `candidate` at p = 127 and 521; `report.json` then `report.csv`; an empty plan) on the new build gave distinct CSV rows with exponent, seed, options, and build, an unchanged first capture, and exit 2 with no output file. The other tests listed cover new behavior.

| Problem | Change | Regression test |
|---|---|---|
| Two writers of the same file shared `<path>.tmp`; one could truncate or rename the other's temporary file | Each write uses its own exclusively created `<path>.tmp.<pid>.<n>` | `concurrent_atomic_writers_never_share_a_temporary_file` (8 threads) |
| Two runs on one proof store (different or no checkpoint directories) could overwrite each other's residues | `ProofStore::open` holds an advisory lock on `M<p>.residues/.lock` for the whole PRP run; a second `run` or `resume` exits 6 before writing, and `resume` does not move the checkpoint aside | `a_held_proof_directory_refuses_a_second_writer`, `resume_refuses_a_held_proof_store_without_touching_the_checkpoint`, `a_store_owns_its_directory` |
| LL `k + max_batch` overflowed for a batch limit near `u64::MAX` | Saturating boundary; `max_batch` 0 is still a configuration error | `huge_batch_sizes_stop_at_the_boundaries` |
| Proof validation copied every residue with infallible allocation, so a large proof could abort the process | Canonicality and zero checks on borrowed bytes; residue buffers reserved fallibly | `proof_allocation` (a fault injecting allocator: the old code aborts with SIGABRT, the new code returns a typed error), `proof_bytes_check_matches_the_parser` (exhaustive for p <= 13) |
| CUDA emulation ignored the driver's exit status, so a sanitizer report at shutdown passed | Checked shutdown on every success path; `Drop` also fails a nonzero exit unless already unwinding | `a_failed_driver_exit_fails_the_test` |
| `report.json` and `report.csv` shared capture files `report.0.stdout`; a later run overwrote an earlier report's evidence | Each invocation owns `<output>.runs/<id>`; captures go there | `bench_captures_belong_to_their_run` |
| CSV rows lost provenance and could not tell entries of the same name apart | Entry index, workload, engine options, argv, check counts, diagnostics, plan hash, build, platform, and execution directory in every row; the exact hashed plan bytes in `<execution_dir>/plan.json`; transform plan named apart from workload plan | `bench_csv_keeps_the_provenance` |
| A proof residue file that changed between its size check and its read panicked | One handle, bounded read, size mismatch is an error | `residue_reads_are_bounded_and_exact`, `damaged_proof_residue_files_are_errors` |
| DWT allocated a word vector per squaring, contrary to its comment | `finish` writes into a reused buffer; the batch still commits only after every step passes | `dwt_allocation` (old code: 501 allocations for 500 squarings; new: equal to one squaring). No speed claim: nothing was timed |
| An empty plan exited 0 without writing output, leaving a stale report in place | Empty plans and `repeats` 0 are usage errors (exit 2); nothing is written | `bench_refuses_an_empty_plan` |

Benchmark observations now also record Gerbicz and Jacobi outcomes, faults, rollbacks, backend diagnostics (DWT maximum roundoff), whether the run started clean (engine runs always do, with checkpointing disabled), and label external observations as process timings whose results are not verified. Energy stays null.

| Command | Observed result |
|---|---|
| `cargo test --locked` | 119 passed, 0 failed, 5 ignored by default |
| `cargo clippy --locked --all-targets -- -D warnings`, also with `--features gpu`, `--features cuda`, `--features gpu,cuda` | passed |
| `cargo fmt --check` | passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --features gpu,cuda` | passed |
| `cargo +1.89 check --locked --all-targets --features gpu,cuda` | passed |
| `cargo test --locked --features gpu --lib --test gpu_backend --test protocol_gpu -- --test-threads=1` | 56 passed, 0 failed, 1 ignored; real Apple M2 through Metal |
| `cargo test --locked --features cuda --lib backend::cuda` | 19 passed; CPU model and host code only |
| `cargo test --locked --features cuda --test cuda_kernel_emulation` | 6 passed under AddressSanitizer and UndefinedBehaviorSanitizer; CPU emulation of the kernel source, not CUDA hardware. LeakSanitizer is unsupported by Apple clang on macOS (`detect_leaks is not supported on this platform`), so leaks are not checked |
| `cargo test --locked --test reference -- --ignored` | 1 passed |
| `MERSENNE_FORGE_UPSTREAM_PROOF=documentation/upstream/gpuowl/test/786433-10.proof cargo test --locked --test upstream_proof -- --ignored` | 1 passed |
| An independent Python big-integer sweep (generic modular arithmetic, no shared code) against the debug CLI | 688 of 688 agree: complete LL and PRP residues on `cpu` and `dwt` for all 172 prime exponents up to 1024 |

Still open: no CUDA kernel has run on NVIDIA hardware; no upstream verifier has checked a proof generated here; no Vulkan or discrete GPU run; no performance comparison.

## Completion checks (2026-11-16)

Implementation source SHA3-256: `6fda9a8a4729907828eacdc6dd17a9aa003de729b9a754ae10017f7ba114e018`. The source identity covers the implementation, manifest, and lockfile as specified in [PROTOCOL.md](PROTOCOL.md). It does not cover this report or the test files.

PRP now saves a completed checkpoint before the proof stage. Tests stop exactly between the final check and proof generation, recover after an unwritable proof file, reject damaged completion residues and metadata, and resume prime and composite runs without repeating PRP iterations. CPU, DWT, and Metal runs require identical regenerated proof bytes. CLI tests relocate both unfinished and completed work, verify effective paths, persist the new proof directory, and reject missing directories without changing the checkpoint.

Build identities now distinguish source changes even when the crate version is unchanged. A filesystem-based test requires every implementation file to appear in the embedded source manifest. Existing identity-mismatch tests still reject incompatible checkpoints. Earlier version-only checkpoint identities are refused; complete their runs with the original binary. Public checked-power calls also reject counts that overflow check-boundary arithmetic, and a maximum u64 batch limit completes correctly.

| Command | Observed result |
|---|---|
| `cargo test --locked` | 105 passed, 0 failed, 5 ignored by default |
| `cargo clippy --locked --all-targets --features gpu,cuda -- -D warnings` | passed |
| `cargo fmt --check` | passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --features gpu,cuda` | passed; strict documentation checks also added to CI |
| `cargo +1.89 check --locked --all-targets --features gpu,cuda` | passed |
| `cargo test --locked --features gpu --lib --test gpu_backend --test protocol_gpu -- --test-threads=1` | 53 passed, 0 failed, 1 ignored; real Apple M2 through Metal |
| `cargo test --locked --features cuda --lib backend::cuda` | 19 passed; CPU model and host code only |
| `cargo test --locked --features cuda --test cuda_kernel_emulation` | 5 passed; CUDA kernel source executed by CPU emulation |
| `cargo test --locked --test reference -- --ignored` | passed; larger reference vectors |
| `MERSENNE_FORGE_UPSTREAM_PROOF=documentation/upstream/gpuowl/test/786433-10.proof cargo test --locked --test upstream_proof -- --ignored` | passed; upstream fixture accepted, altered copy rejected; use the fixture from the pinned reference checkout |
| `cargo build --locked --release` | passed; default CPU/DWT executable |

Release smoke commands below all exited 0. Use a fresh work directory. M4423 LL returned zero; M44497 PRP returned type 1 res64 `0000000000000001` with an accepted proof. After relocating the work directory, resume loaded the completed state at k = 44600, reported zero PRP iteration time and zero new PRP Gerbicz checks, and regenerated byte-identical proof data. Verification on the exact CPU backend accepted it after direct terminal recomputation. The benchmark harness completed all 16 observations; this is a harness check, not a performance comparison.

```sh
target/release/mersenne-forge doctor --json --exponent 44497
target/release/mersenne-forge run --test ll --exponent 4423 --backend cpu --json
target/release/mersenne-forge run --test prp --exponent 44497 --backend dwt \
    --block-size 200 --check-step 2000 --proof-power 8 \
    --proof-dir work/completion-20261116/original/proof \
    --checkpoint-dir work/completion-20261116/original/ckpt --budget 1m --json
mv work/completion-20261116/original work/completion-20261116/relocated
target/release/mersenne-forge resume \
    --checkpoint work/completion-20261116/relocated/ckpt/M44497-prp.ckpt \
    --proof-dir work/completion-20261116/relocated/proof --budget 1m --json
target/release/mersenne-forge verify \
    --proof work/completion-20261116/relocated/proof/44497-8.proof \
    --backend cpu --budget 1m --json
target/release/mersenne-forge bench --plan bench/cpu-smoke.json \
    --output work/completion-20261116/bench.json
```

The regenerated M44497 proof SHA-256 was `db2ca14f48dbc81bf7471498be2ea63c5e28a6a5a7b92eb9b7bce3f33c3ed269` before and after relocation. Python's independent SHA3 implementation also reproduced the source fingerprint reported by the release executable.

The two ignored DWT stress tests and the external upstream-verifier test were not run in this completion pass. CUDA hardware, Vulkan devices, upstream acceptance of our generated proofs, and comparative performance remain unvalidated. No performance improvement is claimed.

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
| PRP proof generation and verification (v2) | yes | yes | yes | Metal only | one direction: an upstream generated proof verifies here, and our M786433 proof was byte-identical to it at commit `4042941`; our proofs have NOT been checked by an upstream verifier (`tests/upstream_verifier.rs` exists and has not been run) | no |
| Bench harness | yes | yes | yes (`bench/cpu-smoke.json`) | no | n/a | no comparison measured |

Continuous integration (`.github/workflows/ci.yml`) runs only the CPU checks; it compiles the `gpu` and `cuda` features but never runs GPU or CUDA tests. Every GPU result in this report was observed by hand on the host above.

"GPU tested" means the generic protocol suite and the CLI runs below passed on the Apple GPU through Metal. Backend internals and their own tests are documented by the GPU engineer in [GPU.md](GPU.md).

## Default build checks

| Command | Result |
|---|---|
| `cargo build` | ok |
| `cargo test --locked` | ok: 105 tests passed, 0 failed, 5 ignored (slow or needing an external file or program; listed below) |
| `cargo clippy --all-targets -- -D warnings` | ok, no warnings |
| `cargo clippy --features gpu --all-targets -- -D warnings` | ok, no warnings |
| `cargo fmt --check` | ok |
| `cargo check --features gpu --all-targets` | ok |
| `cargo check --features cuda --all-targets` | ok |
| `cargo build --locked --all-targets` | ok |
| `cargo +1.89 check --locked --all-targets --features gpu,cuda` | ok (MSRV 1.89 for every feature and the tests) |
| `cargo +1.88 check --locked --ignore-rust-version` | fails: `std::fs::File::try_lock` (the checkpoint lock) is stable from 1.89 |

Per suite (`cargo test --locked`, debug profile with opt-level 2): lib unit tests 17, main 1, `arithmetic` 9, `backend_fault` 4, `checkpoint` 9, `cli` 15, `cross_backend` 1, `dwt_backend` 16 (2 ignored), `ll_jacobi` 1, `proof_forgery` 5, `proof_stage` 4, `protocol_cpu` 10, `protocol_dwt` 10, `reference` 3 (1 ignored), `upstream_proof` 0 (1 ignored), `upstream_verifier` 0 (1 ignored). The `gpu` and `cuda` test files compile to nothing without their features.

Ignored tests, run explicitly:

| Command | Result |
|---|---|
| `cargo test --release --test reference -- --ignored` | ok: M9689, M9941, M11213, M19937 prime by LL and probable prime by PRP on `dwt`; M9689 on `cpu` |
| `MERSENNE_FORGE_UPSTREAM_PROOF=<path>/786433-10.proof cargo test --release --test upstream_proof -- --ignored` | ok: the proof file shipped in the PRPLL source tree at the pinned commit (`test/786433-10.proof`) parses and verifies (M786433, power 10, composite); a copy with one bit flipped is rejected |
| `MERSENNE_FORGE_UPSTREAM_VERIFIER=<prpll> cargo test --release --test upstream_verifier -- --ignored` | NOT run: the upstream build cannot execute on this machine (see the end of this document). It generates a proof (M786433, power 10, `dwt`), runs `prpll -verify` on it and on a corrupted copy, and fails when the variable is unset |

The same upstream proof also verifies from the CLI on `dwt`. M786433 is above the direct recomputation cutoff (2^17), so the acceptance is by the proof equation alone, and the output says so:

```text
$ ./target/release/mersenne-forge verify --proof work/upstream-786433-10.proof --backend dwt
proof work/upstream-786433-10.proof: accepted
M786433 composite: the file satisfies the version 2 proof equation (the relation the upstream verifier checks). That is sound against computation errors and against a prover who skipped the work, not against a prover who knows a factor of M786433 or elements of small order modulo M786433; the challenge is 64 bits. A probable prime claim from a proof needs an independent test
[exit 0]
```

Generation agreed with upstream on this exponent at commit `4042941` (not regenerated since): `mersenne-forge run --test prp --exponent 786433 --backend gpu --proof-power 10 --proof-dir work/m786433/proof --checkpoint-dir work/m786433/ckpt --budget 40m` (Metal, 970 s wall) reported `composite, res64 bfb6553c004222ba`, 11 Gerbicz checks passed, self verification accepted, and `cmp work/m786433/proof/786433-10.proof <upstream>/test/786433-10.proof` exited 0: the file was byte-identical to the proof shipped upstream. Later changes to the run (the terminal round trip, the proof stage) do not change the residues a run stores, and the cross backend and interrupted run tests still give byte-identical proofs, but this exponent was not rerun. This is format and generation evidence; it is not a test of an upstream verifier accepting our files.

## What the tests establish

- Mathematics: reference vectors for p = 2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423 (prime) and 11, 23, 29, 37, 41, 1277 (composite) by the oracle and by the checked engines; base 2 Fermat passes the composite 2047 = 23 * 89 while base 3 rejects it; the PRP terminal residue by repeated `%` squaring equals the `modpow` (Montgomery) result.
- Representation: canonical encoding at 0, 1, 2, N - 1, N - 2, N / 2, around every 32-bit limb boundary, all-ones words, the value N, bits above p, wrong lengths; word aligned and non aligned exponents; randomized residues (proptest).
- Differential arithmetic: `cpu` (folding) and `dwt` (transform) against the oracle (division) for mul, square, ll_step, copy, equal.
- PRP protocol on `cpu`, `dwt`, and (with `--features gpu`) `gpu`: complete prime and composite runs with full 2048-bit residues equal to the oracle; data and check value corruption (bit flip, zero, triple) detected and rolled back with a correct result; an engine injected `BackendError::Fault` rolled back; the same wrong residue twice stops; three failures in a row stop.
- Late backend faults (`tests/backend_fault.rs`, a wrapper around the `cpu` backend only): faults raised from `equal`, `export`, and `ll_step`, including at the initial load, are rolled back with a correct result; persistent faults end in an execution error, never a verdict; every register is freed on those error paths (PRP, LL, proof verification).
- Terminal export: a wrapper that corrupts only the host copy of the terminal residue (PRP x_p = 9 for M1279; LL s_(p-2) = 0, corrupted to 1, which passes the Jacobi check) is caught by the round trip after the export, counted as one fault, and the run ends with the correct result. With the round trip disabled the same test fails.
- Interruption: simulated abrupt termination at k = 77 (mid block), 128 and 129 (at and after a check), 639, 640, 641 (before, at, after a proof point), 1279 (the final residue), 1280 (the final check), and a budget stop; every resumed run reached the same result and a byte-identical, accepted proof.
- Proof durability: a proof residue write failure at k = 680 stops the run; the checkpoint on disk is at 640; resume completes with an accepted proof. A deleted durable residue makes resume refuse.
- Proof rejection: altered B or middles (rejected both by `verify` and by the proof equation alone), truncated, trailing data, wrong exponent, unknown version, bits above p; each a distinct error.
- Chosen proofs for small moduli (`tests/proof_forgery.rs`): the proof equation alone accepts the wrong terminal values M13 B = 8182 (= -9) with middle 7434, and M11 B = 9 with middles 9 (a unit) and 1513 (a zero divisor); `verify` rejects all three by recomputing the terminal residue. Over every (B, middle) pair for M11 at power 1, `verify` accepts only B = 929, the true terminal residue, while the equation alone accepts some middle for 1992 of the 2046 wrong values of B. Middles 23 and 89 for M11 fold to zero and are rejected; a zero middle in a directly constructed proof is rejected; M13 at power 3 (colliding generation schedule) parses and verifies.
- Proof stage (`tests/proof_stage.rs`, `tests/cli.rs`): `not_requested`, `generated`, `verified`, `rejected` (a stored proof residue replaced with a different intact value before resume), `failed` (a directory in place of the proof file; CLI exit 5), `interrupted` (a stop request after the final check); the PRP result is kept in every case.
- Checkpoints: corrupted bytes anywhere fall back to `.prev`; a crash between rotation and rename falls back to `.prev`; after falling back, the next save removes the corrupt primary instead of rotating it over the good `.prev`; oversized metadata is refused before allocation; a different backend, configuration, exponent, or protocol is refused; forged `state`, `verified_k`, or `proof_durable_through` is refused; a checkpoint with a valid hash but a damaged residue pair is refused by the on-load Gerbicz check; a stop or fault before the first save reports no checkpoint.
- CLI: `run` over an existing checkpoint (or only its `.prev`) exits 6 naming `resume` and leaves the file unchanged; a lock held by another process makes `run` and `resume` exit 6; a primary with a valid hash and a damaged residue pair makes `resume` fall back to `.prev` (reported as `loaded_from`); an unrepresentable `--budget` exits 2; JSON failures are JSON objects; a process killed with SIGKILL after its first checkpoint resumes to the same result and a byte-identical proof as an uninterrupted run; `doctor --exponent` reports per backend memory figures.
- Bench: a preexisting scratch directory is a failed observation and is left untouched; external output is saved to files; the output file is rewritten after every observation; a SIGINT during an external command stops the plan (exit 3); invalid plan budgets exit 2.
- Not observable in a test: the fsync of newly created directories.
- LL Jacobi: a symbol flipping fault at a check is always caught; over 64 single bit flips on M521, the detected ones recover and the undetected ones end with a wrong final residue (the test asserts both), which is the documented limit of the LL check.
- Cross backend: `cpu`, `dwt`, and `gpu` give identical full final residues and byte-identical proof files for M1279 and M1277.

## End-to-end CLI demonstration (default build, release)

Commands run from the repository root after `cargo build --release`, starting from an empty `work/`. Output pasted verbatim. A completed run keeps its checkpoint, so a second pass of the same commands is refused by `run` (exit 6) until `work/` is emptied.

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
gerbicz check step: 400
proof: work/proof/4423-6.proof (power 6, self verification accepted)
proof stage: verified
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 4421 --backend cpu --block-size 200 --check-step 400 --proof-power 6 --proof-dir work/proof --checkpoint-dir work/ckpt
M4421 PRP: composite, res64 b45cf72f1682620c
gerbicz checks: 12 passed, 0 failed and rolled back
gerbicz check step: 400
proof: work/proof/4421-6.proof (power 6, self verification accepted)
proof stage: verified
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 44497 --backend dwt --block-size 200 --check-step 2000 --proof-power 8 --proof-dir work/proof --checkpoint-dir work/ckpt --budget 2s
incomplete at iteration 18200: stop requested or budget exhausted
resume with: mersenne-forge resume --checkpoint work/ckpt/M44497-prp.ckpt
gerbicz checks: 11 passed, 0 failed and rolled back
gerbicz check step: 2000
[exit 3]

$ ./target/release/mersenne-forge run --test prp --exponent 44497 --backend dwt --block-size 200 --check-step 2000 --proof-power 8 --proof-dir work/proof --checkpoint-dir work/ckpt
error: a checkpoint for this run already exists at work/ckpt/M44497-prp.ckpt: continue it with `mersenne-forge resume --checkpoint work/ckpt/M44497-prp.ckpt`, or delete work/ckpt/M44497-prp.ckpt and work/ckpt/M44497-prp.ckpt.prev to start over
[exit 6]

$ ./target/release/mersenne-forge resume --checkpoint work/ckpt/M44497-prp.ckpt --budget 10m
resumed from work/ckpt/M44497-prp.ckpt at iteration 18200
M44497 PRP: probable prime (base 3), res64 0000000000000001
gerbicz checks: 15 passed, 0 failed and rolled back
gerbicz check step: 2000
proof: work/proof/44497-8.proof (power 8, self verification accepted)
proof stage: verified
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ ./target/release/mersenne-forge run --test prp --exponent 44497 --backend dwt --block-size 200 --check-step 2000 --proof-power 8 --proof-dir work/proof-uninterrupted
M44497 PRP: probable prime (base 3), res64 0000000000000001
gerbicz checks: 24 passed, 0 failed and rolled back
gerbicz check step: 2000
proof: work/proof-uninterrupted/44497-8.proof (power 8, self verification accepted)
proof stage: verified
note: a positive local result is not a discovery claim; compare with known results
[exit 0]

$ cmp work/proof/44497-8.proof work/proof-uninterrupted/44497-8.proof
[exit 0]

$ ./target/release/mersenne-forge verify --proof work/proof/44497-8.proof --backend cpu
proof work/proof/44497-8.proof: accepted
M44497 probable prime: the terminal residue was recomputed directly and matches the proof; this verifies the PRP computation, not deterministic primality
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

$ ./target/release/mersenne-forge verify --proof work/proof/4423-6.proof --backend cpu
proof work/proof/4423-6.proof: accepted
M4423 probable prime: the terminal residue was recomputed directly and matches the proof; this verifies the PRP computation, not deterministic primality
[exit 0]
```

(`work/proof/truncated.proof` is the first 3000 bytes of `work/proof/4423-6.proof`.)

Ctrl-C: an LL run of M86243 on `dwt` with `--checkpoint-dir work/ctrlc --json` received SIGINT after 2 seconds, printed `stop requested; finishing at the next valid boundary (press Ctrl-C again to abort)`, exited 3 with JSON process state `{"state": "incomplete", "k": 21000, "checkpoint": "work/ctrlc/M86243-ll.ckpt", ...}`, and `mersenne-forge resume --checkpoint work/ctrlc/M86243-ll.ckpt` then printed `resumed from work/ctrlc/M86243-ll.ckpt at iteration 21000` and `M86243 LL: prime (LL criterion satisfied)` with exit 0.

Bench: `mersenne-forge bench --plan bench/cpu-smoke.json --output work/bench/cpu-smoke.json` wrote 16 observations (4 warmups, 12 measured), all completed. These timings are a harness check, not a performance result.

## GPU demonstration (Apple M2, Metal, `--features gpu`)

`cargo test --features gpu --test protocol_gpu` (debug profile with opt-level 2): 11 passed, 0 failed (the full generic protocol suite above on the `gpu` backend, plus the cpu, dwt, gpu identity test).

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
gerbicz check step: 2000
proof: work/gpu-proof/44497-8.proof (power 8, self verification accepted)
proof stage: verified
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
M44497 probable prime: the terminal residue was recomputed directly and matches the proof; this verifies the PRP computation, not deterministic primality
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

Upstream verifier interoperability (our proofs accepted by PRPLL): build PRPLL at commit `90a3d8db84e742773f45e2adfba3131199fa8101` per its README, generate a proof here for an exponent inside the upstream FFT range (for example the M1257787 run above), and run the upstream verifier with its documented interface `prpll -verify <file>`. This has not been done. One attempt was made on the development machine (Apple M2, macOS, 2026-11-16): PRPLL at that commit builds with Apple clang and lists the Apple M2 as an OpenCL 1.2 device, but `prpll -verify test/786433-10.proof` cannot run there. With the default FP64 transform it stops with "FP64 not supported"; with the integer transforms (`-fft 1:256:1:256`, `-fft 3:256:1:256`) the process aborts inside the Apple OpenCL compiler during `clCompileProgram`, before any arithmetic. So the upstream verifier needs a host with a working OpenCL or CUDA driver.

Performance: no measurement exists. See [BENCHMARK.md](BENCHMARK.md).

Before and after optimization on NVIDIA (not done): build at a fixed commit, write one plan with the upstream PRPLL command as an `external` entry and this program's `cuda` engine entry in `production` mode with `retain` and `verify_with` naming the upstream verifier, add a `telemetry` command that prints device watts (for example `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0`), run it, and compare with `bench-analyze`. Change one kernel at a time, keep the old variant selectable, and repeat the same frozen plan. No such run exists, and no optimization is claimed.

Work queue: `tests/worktodo.rs` checks the upstream assignment forms and refusals, the result fields against the proof file's MD5, outbox idempotency, and resumption of a budget stopped assignment. No result line has been submitted to PrimeNet or read by AutoPrimeNet.
