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

Observation fields: `entry` (display name, not unique), `entry_index` (position in the plan's `entries`, the stable identifier), `kind`, `test` and `exponent` (engine runs), `argv` (external runs), `result_source`, `start` (engine runs: every run starts at iteration 0; `clean, checkpointing disabled` in steady state mode, `clean, checkpoints written at every passed check` in production mode), `sequence` (position in the executed order), `repeat`, `warmup`, `status` (`completed`, `incomplete`, `failed`; an engine PRP run counts as completed only when its requested proof stage also succeeded: a `failed` or `rejected` proof stage makes the observation `failed`, an `interrupted` one `incomplete`), `error`, `wall_secs`, `timings` (engine runs: `init`, `iterations`, `checks`, `checkpoint_io`, `proof_generation`, `proof_verification`, all seconds), `verdict`, `res64`, `backend` (engine runs: kind, build, plan, device), `proof_stage` (engine PRP runs), `check_step` (engine PRP runs: the Gerbicz check step derived at the start), `reliability` (engine runs: Gerbicz and Jacobi outcomes, backend faults, rollbacks), `diagnostics` (engine runs: backend diagnostics read once after the run, such as the DWT maximum roundoff), `exit_code` (external runs), `stdout` and `stderr` (external runs: the capture files), `energy_joules`, `telemetry`. The output also records the plan, `plan_sha3` (SHA3-256 of the plan as recorded, compact JSON), the seed used, `execution_dir`, `build` (the source build identity), the crate version, OS, and architecture. CSV output has one row per observation with the timings and check counts flattened, and every row repeats that provenance (`plan_name`, `plan_sha3`, `seed_used`, `build`, `os`, `arch`, `execution_dir`) together with the entry's engine `options` as JSON. The exact plan bytes that `plan_sha3` hashes are written to `<execution_dir>/plan.json`, so a CSV file and its execution directory together identify the whole plan. In CSV, `transform_plan` is the backend's arithmetic plan and `plan_name` with `plan_sha3` identify the workload plan; `argv` is a JSON array and `diagnostics` is `key=value` pairs joined by `;`. Every CSV field is quoted (RFC 4180, a quote inside a field is doubled), so commas, quotes, carriage returns, and newlines in names, test strings, argv, or errors stay inside their field and each observation is one record.

Accounting boundaries: `iterations` is time inside backend squaring or LL step calls between checks, followed by a backend `sync` once per segment of at most 1000 iterations, so queued GPU work is counted here and not in `checks`; `checks` includes Gerbicz checks (with their residue exports) or Jacobi checks; `proof_generation` and `proof_verification` are the proof steps after the run; `init` includes backend construction. `wall_secs` is the complete run. Kernel compilation, when a GPU backend does it, falls in `init`.

Energy: without a `telemetry` block in the plan, `energy_joules` is null and `telemetry` is `unavailable`. Missing telemetry is never replaced by an estimate.

## Evidence options (output schema version 2)

Outputs carry `schema_version` 2 and a `provenance` block: executable path and SHA3-256, source identity, crate version, profile, target architecture, OS, and environment, compiled instruction set features, Cargo features, lockfile SHA3-256, CPU model, and logical CPU count. The compiler version comes from the build script (`rustc -V` of the compiler that built the binary, or `unknown`); the power policy is the Linux CPU frequency governor or the macOS low power mode setting where readable, otherwise `unknown`. Outputs with a newer `schema_version` are refused. Version 1 outputs (no `schema_version`) are meant to be read by `bench-render` and `bench-analyze`; no version 1 file is part of the tests.

Engine options:

- `mode`: `steady_state` (default; no checkpoints, proof only when `proof_power` is given) or `production` (checkpoints at every passed check; PRP entries must give `proof_power`, and proof self verification cannot be turned off). Both modes include startup; `timings.init` shows it.
- `retain`: keep the run report and the proof file in `<execution_dir>/artifacts/N/`, each recorded with size and SHA3-256. The plan field `retain_max_bytes` (default 1 GiB) caps the total: a file that does not fit is hashed and recorded with `not_retained` giving the reason, but not kept. A retention error marks the observation failed with the reason and keeps its measurements.
- `verify_with`: `{argv, accept_text, timeout_secs}`. After a completed PRP with a proof, `argv` runs with `{proof}` replaced by the proof path, then again on a copy with one bit flipped. The verifier program is resolved through PATH once, and that file is both hashed and run. Each run gets an outcome: `accepted` (exit 0 and `accept_text` printed), `rejected` (a normal nonzero exit), `unrecognized` (exit 0 without `accept_text`), `timeout`, `signal`, or `stopped`; a timeout kills the verifier's whole process group. The proof counts as verified only when the first run is `accepted` and the altered copy is explicitly `rejected` (`altered_outcome`; `altered_rejected` is true only then). The record keeps the exact argv, exit codes, outcomes, captures, the verifier's SHA3-256, and the time; anything else marks the observation failed. The adapter has been tested only with this program's own `verify` standing in for an independent verifier.

Timing: `timings.proof_points` is the export, durable write, and read back of PRP proof points, which were previously only part of the wall time. `unclassified_secs` is wall time not covered by any timer (for example the terminal residue round trip and report assembly). The timers do not overlap.

Telemetry: the plan field `telemetry` is `{argv, interval_ms, sensor, scope}`. The command runs every `interval_ms` (at least 10) while the workload runs, plus once at the end; sampling stops when the engine returns, before retention and external verification, so energy covers the timed workload only. A slow reading skips the ticks it overran instead of queuing them, and each reading is limited to ten intervals (at least one second) and 4 KiB of output; a reading over the limit is killed and counts as missing. The output is read in watts: a number followed by a `W` unit, or else the only number in the output (a number inside a label such as `GPU0` is ignored); anything else is a missing reading. `scope` is `device` or `system`; the two are different metrics. `power` keeps every reading with its time; `energy_joules` is the trapezoidal integral over the sampled span, and stays null when any reading is missing or fewer than two exist. Built in sensors for the `argv` (`mersenne-forge power --source rapl|nvidia|powermetrics [--device N]`): Linux RAPL package energy (system CPU package; often root only), `nvidia-smi` power draw (device), and macOS `powermetrics` combined power (needs root). None has been run on this project's hosts, so energy has not been measured.

Rendering and analysis: `bench-render --input out.json --output out.csv` (or `.json`) writes a recorded execution again without running anything. `bench-analyze --input out.json --baseline I --candidate J [--metric wall_secs|iterations|energy_joules] [--seed S]` pairs the measured runs of two entries by repeat, counts a pair only when both completed with the same test and exponent, the same result (a disagreement is counted in `result_disagreements`), and the same power scope (each excluded pair is listed with its reason in `excluded`), refuses duplicate repeats and newer schema versions, and reports the per pair ratios, the median ratio with a 95% percentile bootstrap interval (2000 resamples, fixed seed), and the full cost of each entry including failed runs. With fewer than 5 usable pairs it says the interval is not meaningful. Tune on one plan and evaluate on a separate, frozen plan with other exponents.

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
