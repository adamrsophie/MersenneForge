# Benchmark method

No performance comparison has been measured. No GPU measurement and no upstream baseline measurement exists yet. This document describes the harness, its formats, and the commands; it reports no numbers.

## Harness

`mersenne-forge bench --plan <plan.json> --output <out.json|out.csv>` runs a plan and writes one raw observation per run. Failures are kept as observations with status `failed` and the error text; nothing is dropped. The output file is rewritten after every observation, so an interrupted plan leaves the observations made so far.

The whole plan is validated before anything runs: a plan with no entries or with `repeats` 0 is a usage error (exit 2) and no output is written, so an old output file is never mistaken for a new result; a budget or timeout that is not a finite positive number of seconds, or that cannot be represented as a deadline, is a usage error (exit 2).

Execution directory: every invocation creates its own directory `<output>.runs/<seconds>-<pid>-<n>` exclusively and records it as `execution_dir`. Each run gets a scratch directory `run-N` there, removed after the run. Directories of earlier invocations are never reused, changed, or deleted, so the capture files a report references stay as they were when it was written, even when a later plan writes `report.csv` beside `report.json` or reruns the same output. A rerun replaces the output file itself.

Stopping: Ctrl-C is passed to engine runs (they stop at their next valid boundary) and checked while an external command runs (the command is killed); the plan then ends after the current observation, and the command exits 3. Only the direct child process of an external entry is killed, on a stop request or a timeout; processes it started itself are not tracked or killed.

Plan (JSON):

```json
{
  "name": "cpu-smoke",
  "order": "randomized",
  "seed": 20261115,
  "repeats": 3,
  "warmups": 1,
  "entries": [
    { "kind": "engine", "name": "prp-4423-cpu", "test": "prp", "exponent": 4423, "backend": "cpu",
      "options": { "block_size": 200, "proof_power": 6 } },
    { "kind": "external", "name": "baseline", "argv": ["<program>", "<arg>"], "cwd": "<dir>", "timeout_secs": 7200 }
  ]
}
```

- `engine` entries run in process on `cpu`, `dwt`, `gpu`, or `cuda`. Options: `block_size`, `check_step`, `proof_power`, `verify_proof`, `log_step`, `jacobi_every_log`, `device`, `budget_secs` (required for `gpu` and `cuda`).
- `external` entries run `argv` verbatim as a subprocess and record wall time and exit code. Their standard output and error are saved in the execution directory as `N.stdout` and `N.stderr` (N is the sequence number) and referenced from the observation. An external observation is a process timing: its `result_source` is `external process timing; result not verified`, since an exit status says nothing about the correctness of the program's output. This is how a tuned upstream `prpll` baseline is timed: the plan author writes its exact command line. The harness invents no flags for any external program.
- `order`: `sequential` (all repeats of one entry, then the next), `paired` (each repeat runs every entry in declared order), `randomized` (each repeat runs every entry in an order shuffled by SplitMix64 from `seed`; the seed used is recorded in the output).
- `warmups`: that many runs of every entry before the measured runs, recorded with `warmup: true`.

Observation fields: `entry` (display name, not unique), `entry_index` (position in the plan's `entries`, the stable identifier), `kind`, `test` and `exponent` (engine runs), `argv` (external runs), `result_source`, `start` (engine runs: `clean, checkpointing disabled`; every engine run starts at iteration 0 and writes no checkpoint), `sequence` (position in the executed order), `repeat`, `warmup`, `status` (`completed`, `incomplete`, `failed`; an engine PRP run counts as completed only when its requested proof stage also succeeded: a `failed` or `rejected` proof stage makes the observation `failed`, an `interrupted` one `incomplete`), `error`, `wall_secs`, `timings` (engine runs: `init`, `iterations`, `checks`, `checkpoint_io`, `proof_generation`, `proof_verification`, all seconds), `verdict`, `res64`, `backend` (engine runs: kind, build, plan, device), `proof_stage` (engine PRP runs), `check_step` (engine PRP runs: the Gerbicz check step derived at the start), `reliability` (engine runs: Gerbicz and Jacobi outcomes, backend faults, rollbacks), `diagnostics` (engine runs: backend diagnostics read once after the run, such as the DWT maximum roundoff), `exit_code` (external runs), `stdout` and `stderr` (external runs: the capture files), `energy_joules`, `telemetry`. The output also records the plan, `plan_sha3` (SHA3-256 of the plan as recorded, compact JSON), the seed used, `execution_dir`, `build` (the source build identity), the crate version, OS, and architecture. CSV output has one row per observation with the timings and check counts flattened, and every row repeats that provenance (`plan_name`, `plan_sha3`, `seed_used`, `build`, `os`, `arch`, `execution_dir`) together with the entry's engine `options` as JSON. The exact plan bytes that `plan_sha3` hashes are written to `<execution_dir>/plan.json`, so a CSV file and its execution directory together identify the whole plan. In CSV, `transform_plan` is the backend's arithmetic plan and `plan_name` with `plan_sha3` identify the workload plan; `argv` is a JSON array and `diagnostics` is `key=value` pairs joined by `;`. Every CSV field is quoted (RFC 4180, a quote inside a field is doubled), so commas, quotes, carriage returns, and newlines in names, test strings, argv, or errors stay inside their field and each observation is one record.

Accounting boundaries: `iterations` is time inside backend squaring or LL step calls between checks, followed by a backend `sync` once per segment of at most 1000 iterations, so queued GPU work is counted here and not in `checks`; `checks` includes Gerbicz checks (with their residue exports) or Jacobi checks; `proof_generation` and `proof_verification` are the proof steps after the run; `init` includes backend construction. `wall_secs` is the complete run. Kernel compilation, when a GPU backend does it, falls in `init`.

Energy: the harness reads no power sensor, so `energy_joules` is always null and `telemetry` is `unavailable`. Missing telemetry is never replaced by an estimate.

## Method for a comparison (not yet done)

1. Freeze the plan (exponent cohort, settings, repeats, order, seed) before looking at results.
2. Record hardware, driver, backend identity (`mersenne-forge doctor --json`), and for the upstream baseline its revision, build, and tuning.
3. Use paired or randomized order with warmups. The statistical unit is one complete run, not an iteration.
4. Compare complete workloads (completed, checked tests including proof generation and verification), and separately steady state `iterations` time. Failed or incomplete runs are reported, not counted as completed work.
5. Report paired differences with uncertainty; an inconclusive result or a regression is a valid outcome.

## Commands

```sh
cargo build --release
./target/release/mersenne-forge bench --plan bench/cpu-smoke.json --output work/bench/cpu-smoke.json
./target/release/mersenne-forge bench --plan bench/cpu-smoke.json --output work/bench/cpu-smoke.csv

# On a GPU host (edit bench/gpu-example.json first: exponent range, device, and the exact upstream command):
cargo build --release --features gpu
./target/release/mersenne-forge bench --plan bench/gpu-example.json --output work/bench/gpu.json
```

`bench/cpu-smoke.json` runs on any machine and checks that the harness works; its timings are not a performance claim. `bench/gpu-example.json` is a template: its external entry is a placeholder that must be replaced with the real baseline command.
