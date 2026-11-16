# Protocol and formats

This document specifies what the Rust state machines in `src/prp.rs`, `src/ll.rs`, `src/proof.rs`, and `src/checkpoint.rs` compute. The semantics follow PRPLL (gpuowl) at commit `90a3d8db84e742773f45e2adfba3131199fa8101` (`Gpu::isPrimePRP`, `Gpu::doCheck`, `Gpu::isPrimeLL`, `Proof.cpp`); that reference checkout is kept outside this repository. Every deliberate difference is listed at the end.

Notation: p is a prime exponent, N = 2^p - 1, all arithmetic is modulo N.

## Canonical residue

`Residue`: exponent p, exactly ceil(p / 32) little-endian `u32` limbs, value 0 <= x < N, bits at positions >= p zero, unique zero (the all-ones pattern, which equals N, is rejected by `from_limbs` and normalized to zero by `from_limbs_normalizing`). Files store limbs little-endian. The PRP proof format stores (p - 1) / 8 + 1 little-endian bytes per residue.

## Exponent domain

Only prime p is accepted. Primality of p is decided by trial division up to sqrt(p); for p <= 2^32 - 1 that bound is below 65536, so the routine is a complete proof over the whole `u32` range. Each backend also has a supported exponent range (`mersenne-forge doctor`).

## PRP (base 3) with the Gerbicz check

Iteration indexing: x_0 = A with A = 3; x_(k+1) = x_k^2. After p squarings, x_p = 3^(2^p) = 3^(N+1). M_p is a base-3 probable prime iff x_p = 9 (for p = 3, 9 mod 7 = 2). The reported res64 is the low 64 bits of the "type 1" residue x_p / 9 (division by 3 is defined because 3 does not divide N for odd p); res2048 is the low 2048 bits of the same value. p = 2 is a special case: N = 3 equals the base, and M_2 is reported prime by inspection.

Check value C with block size B (default 1000; upstream set 200, 400, 500, 1000):

- C starts at 1. Before squaring at every k that is a multiple of B, C <- C * x_k. So after k = 0, C = x_0 = A.
- A check at a multiple k of B uses C_old, the value of C before the update at k: it computes T = C_old^(2^B) * A and C <- C_old * x_k, and passes iff T == C and C_old != 0 and x_k != 0. This holds because C_old = x_0 x_B ... x_(k-B), so C_old^(2^B) = x_B ... x_k.
- The check performs the update at k, so the update at the start of the next block is skipped after a check.
- Checks run at every multiple of the check step S, at start + 2B (after every load or rollback), when a stop is requested (budget or Ctrl-C), and at roundUp(p, B). The run continues past p to roundUp(p, B) so the final residue is covered by a check; x_p is recorded on the way and reported only after that final check passes.
- Check step: S = 40000, 160000, 200000, 1000000 for B = 200, 400, 500, 1000; halved after any error; and, as upstream, when S > p / 8 it becomes max(L, floor(min(S, p / 8) / L) * L) with L = 20000 (or L = B when B does not divide 20000). An explicit `--check-step` (a multiple of B) replaces this rule and is still halved after an error (rounded down to a multiple of B, at least B).

Verified state: (k, C_old, x_k). Rollback boundary: the last verified state. Loading a state (start, rollback, or resume) imports C_old and x_k and runs the check equation on the pair, which both verifies the loaded state mathematically and leaves C = C_old * x_k.

Failure handling: a failed check or a detected backend fault (`BackendError::Fault`, for example the DWT roundoff check) counts as an error and rolls back. The run stops with an execution error after 3 failures in a row (a passed check resets the count) or when the same wrong residue x_k is produced twice. Neither is ever reported as composite.

Stops: a stop request is honored at the next block boundary, after a check there; a passed check writes a checkpoint, then the run reports `incomplete`.

## Proof points and durability

Schedule for power P (1 <= P <= 13): start with {0}; for each of P levels, with span = (p + 1) / 2 at the first level and span = (span + 1) / 2 after, add span to every existing point; replace 0 by p; sort. A power is accepted only if the 2^P points are distinct and the largest is p (so p = 13 accepts P <= 2, p = 1279 accepts P <= 9). The default power is upstream's 10 + floor(log4(p / 6e7)), at most 13, lowered until valid.

As the PRP run passes each point k it exports x_k and writes it durably: temporary file, fsync, rename, directory fsync, then read back and compare. Only after that does the run continue. A checkpoint is written only at a later passed check, so a checkpoint never claims progress past a proof residue that failed to become durable: a failed write stops the run with an error, the last checkpoint is earlier, and resume recomputes the residue. On resume, every point at or below the checkpoint's k must be present and intact, or resume refuses. A rollback past a point rewrites it.

Proof residue file (`<proof-dir>/M<p>.residues/<k>`): `"MF-PROOF-RESIDUE"` (16 bytes), u32 p, u64 k, the limbs, SHA3-256 of the preceding bytes.

## Proof file (PRPLL version 2)

```text
PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=<P>\nNUMBER=M<p>\n
B = x_p                     (p - 1) / 8 + 1 bytes, little-endian
M_1 ... M_P                 (p - 1) / 8 + 1 bytes each
```

File name `<p>-<P>.proof`. Hash chain: h_0 = SHA3-256(bytes of B); h_i = SHA3-256(h_(i-1) || bytes of M_i); the exponent e_i is the first 8 bytes of h_i read as a little-endian u64. SHA3-256 is the standard FIPS 202 function (padding 0x06).

Generation (`computeProof`): at level i (0 based), with s = 2^(P - i - 1), the residues at points[s (2j + 1) - 1] for j = 0 .. 2^i - 1 are combined left to right with a stack: push each, and while bit t of j is set, pop the top T and replace the new top U by U^(e_(i - t)) * T. The single remaining value is M_(i+1). A zero middle is an error. Generation is not checked; it is verified afterwards.

Verification (`Proof::verify`): A = 3, B from the file, span = p. For each level i: B <- M^e * (span odd ? B^2 : B); A <- A^e * M; span <- (span + 1) / 2. Finally the proof is accepted iff A^(2^span) == B. A proof whose B is 9 proves the computation of a probable prime result; otherwise of a composite result.

The final A^(2^span) uses the same Gerbicz engine as PRP with start value A in place of 3: C^(2^B) * A == C * x_k. Block B = clamp(round(sqrt(span / 10)), 2, 1000), check step about span / 10 (rounded to a multiple of B).

Rejections, each a distinct error: `BadHeader` (not version 2 or malformed), `WrongExponent` (with `--exponent`), `InvalidPower`, `Truncated`, `TrailingData`, `NoncanonicalResidue` (bits at or above p), `ZeroResidue`, `Rejected` (parses, but the final equation fails). The file size is checked against the header before the body is read.

## Lucas-Lehmer with the Jacobi check

s_0 = 4, s_(k+1) = s_k^2 - 2; M_p is prime iff s_(p-2) = 0. p = 2 is a special case (prime). The reported res64 is the low 64 bits of s_(p-2).

With w = 2 + sqrt(3), s_k = w^(2^k) + w^(-2^k), so for k >= 1, s_k - 2 = 3 t^2 for an integer t, and (3 | N) = -1. So the Jacobi symbol (s_k - 2 | N) must be -1 for prime and composite N alike. s = 2 and s = N - 2 (the path of a zeroed residue) count as failures; a symbol of 0 is inconclusive (neither pass nor failure). A residue that passes becomes the verified state. A zero residue before k = p - 2 is a failure.

This check catches about half of all errors (`tests/ll_jacobi.rs` measures this on single bit flips), and an error is caught at the first check after it or never. It is not the Gerbicz check; an undetected LL error yields a wrong final residue.

Cadence: the symbol is computed on the host from an exported residue at a log step (default 20000) when the time since the last check is at least 100 times its measured cost, at the first log step of the process, at the first log step at or after a failed iteration, and always on the final residue (`--jacobi every-log` checks every log step). Failure handling: roll back to the last verified residue; stop after 3 failures in a row or the same wrong residue at the same k twice.

## Checkpoint format (project format, version 1)

```text
"MERSENNE-FORGE-CKPT\n"   20 bytes
u32 format version        1
u64 metadata length       at most 1 MiB
metadata                  JSON (below)
u32 residue count         at most 4
per residue               u32 p, u64 byte length (= 4 ceil(p / 32)), limb bytes
SHA3-256                  over all preceding bytes
```

Metadata: `format_version`, `protocol` (`prp-base3-gerbicz/1` or `ll-jacobi/1`), `exponent`, `k`, `state` (`verified` or `provisional`), `verified_k`, `n_errors`, `proof_durable_through` (largest proof point at or below k), `backend` (kind, build, plan, device), `config` (effective configuration, including block size, check step, proof power and directory), `requested` (configuration as requested), `elapsed_secs`, `crate_version`.

Residues: PRP stores [C_old, x_k] of a verified state. LL stores [s_k, s_verified]: the provisional residue at k and the last Jacobi verified residue at `verified_k`.

Write: `<path>.tmp`, fsync; rename the old `<path>` to `<path>.prev`; rename the temporary file to `<path>`; fsync the directory. Load: sizes are checked against the header before residues are read; the hash is checked; a missing or corrupt `<path>` falls back to `<path>.prev`. A checkpoint is refused when its protocol, exponent, backend identity, or configuration (apart from the checkpoint path) differs from the run. The hash only detects corrupted bytes; the PRP on-load check is the mathematical verification.

There is no compatibility with upstream savefiles.

## Differences from upstream PRPLL

1. Checkpoints store both C_old and x_k; upstream stores C only and rebuilds x_k = 3 C^(2^B - 1) (a chain that needs B not to be a power of two). Ours loads with the check equation instead.
2. Only verified PRP states are checkpointed; upstream also writes unverified saves at log steps. A crash can lose up to one check interval.
3. Proof verification uses the PRP style Gerbicz check with start A; upstream checks the verification squarings CERT style (`certVerify`) and saves verification progress. A stopped verification here restarts.
4. Block size and check step are configurable (for small exponents and tests); defaults are upstream's.
5. Proof generation is attempted once at the configured power; on failure the result is reported without a proof and the failure is recorded. Upstream retries at lower powers.
6. The default proof power is lowered until the schedule is valid for p; upstream does not handle very small p.
7. The "same wrong residue twice" rule compares complete residues, not res64.
8. Backend faults count as failed checks. A failed on-load check is retried once before the state is rejected.
9. The Jacobi symbol is computed on the host; upstream uses GPU GCD kernels.
10. LL stores the provisional and verified residues in one checkpoint file; upstream uses two savefile series.
11. P-1 inside PRP (upstream `Pm1Prp`) is not implemented.
12. A stop request ends an LL run after the current batch of at most 1000 iterations; PRP stops at the next block boundary as upstream.
