# Benchmark method

No performance comparison has been measured. No GPU measurement and no upstream baseline measurement exists yet. This document describes the harness, its formats, and the commands; it reports no numbers.

## Harness

`mersenne-forge bench --plan <plan.json> --output <out.json|out.csv>` runs a plan and writes one raw observation per run. Failures are kept as observations with status `failed` and the error text; nothing is dropped. The output file is rewritten after every observation, so an interrupted plan leaves the observations made so far.

The whole plan is validated before anything runs: a budget or timeout that is not a finite positive number of seconds, or that cannot be represented as a deadline, is a usage error (exit 2).

Scratch directories: each run gets `<output>.work/run-N`, created exclusively. An existing path there is reported as a failed observation and left alone; the harness deletes only directories this invocation created.

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
- `external` entries run `argv` verbatim as a subprocess and record wall time and exit code. Their standard output and error are saved next to the output as `<output>.N.stdout` and `<output>.N.stderr` (N is the sequence number) and referenced from the observation. This is how a tuned upstream `prpll` baseline is timed: the plan author writes its exact command line. The harness invents no flags for any external program.
- `order`: `sequential` (all repeats of one entry, then the next), `paired` (each repeat runs every entry in declared order), `randomized` (each repeat runs every entry in an order shuffled by SplitMix64 from `seed`; the seed used is recorded in the output).
- `warmups`: that many runs of every entry before the measured runs, recorded with `warmup: true`.

Observation fields: `entry`, `kind`, `sequence` (position in the executed order), `repeat`, `warmup`, `status` (`completed`, `incomplete`, `failed`; an engine PRP run counts as completed only when its requested proof stage also succeeded: a `failed` or `rejected` proof stage makes the observation `failed`, an `interrupted` one `incomplete`), `error`, `wall_secs`, `timings` (engine runs: `init`, `iterations`, `checks`, `checkpoint_io`, `proof_generation`, `proof_verification`, all seconds), `verdict`, `res64`, `backend` (engine runs: kind, build, plan, device), `proof_stage` (engine PRP runs), `check_step` (engine PRP runs: the Gerbicz check step derived at the start), `exit_code` (external runs), `stdout` and `stderr` (external runs: the capture files), `energy_joules`, `telemetry`. The output also records the plan, the seed used, the crate version, OS, and architecture. CSV output has one row per observation with the timings flattened.

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
