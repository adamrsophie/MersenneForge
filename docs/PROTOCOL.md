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
- Checks run at every multiple of the check step S, at start + 2B (after every load or rollback), when a stop is requested (budget or Ctrl-C), and at roundUp(p, B). The run continues past p to roundUp(p, B) so the computation through p is covered by a check; x_p is recorded on the way and reported only after that final check passes.
- The recorded x_p is a host copy made by `export`. Right after that export the copy is imported into a spare register and compared with the resident value (`equal`); a mismatch counts as an error and rolls back like a failed check. The later check covers the resident value, so this binds the reported copy to it.
- x_(p-1) is exported too, and x_p must equal its square. The Gerbicz check cannot see a sign flip at iteration p (squaring at p + 1 removes it, and M_p = 3 mod 4 makes -x a non-square), so this binds the reported x_p to the checked chain.
- Remaining limit, shared with upstream: the later check covers x_p only through its square. A device error at exactly iteration p that maps x_p to another square root of x_p^2 (for example -x_p) leaves every later value unchanged and is not seen by the final check.
- Check step: S = 40000, 160000, 200000, 1000000 for B = 200, 400, 500, 1000; halved after any error; and, as upstream, when S > p / 8 it becomes max(L, floor(min(S, p / 8) / L) * L) with L = 20000 (or L = B when B does not divide 20000). An explicit `--check-step` (a multiple of B) replaces this rule and is still halved after an error (rounded down to a multiple of B, at least B).

Verified state: (k, C_old, x_k). Rollback boundary: the last verified state. Loading a state (start, rollback, or resume) imports C_old and x_k and runs the check equation on the pair, which detects a damaged residue pair and leaves C = C_old * x_k. The equation does not contain k: a consistent pair labelled with a wrong iteration count passes it. The iteration count is trusted metadata (see the checkpoint section).

Failure handling: a failed check or a detected backend fault (`BackendError::Fault`, for example the DWT roundoff check) counts as an error and rolls back. The run stops with an execution error after 3 failures in a row (a passed check resets the count) or when the same wrong residue x_k is produced twice. Neither is ever reported as composite.

Stops: a stop request is honored at the next block boundary, after a check there; a passed check writes a checkpoint, then the run reports `incomplete`.

## Proof points and durability

Schedule for power P (1 <= P <= 13): start with {0}; for each of P levels, with span = (p + 1) / 2 at the first level and span = (span + 1) / 2 after, add span to every existing point; replace 0 by p; sort. A power is accepted only if the 2^P points are distinct and the largest is p (so p = 13 accepts P <= 2, p = 1279 accepts P <= 9). The default power is upstream's 10 + floor(log4(p / 6e7)), at most 13, lowered until valid.

As the PRP run passes each point k it exports x_k and hands it to a writer thread, which writes it durably (temporary file, fsync, rename, directory fsync) and reads it back. At most two points wait in its queue; a further one blocks the run until the writer catches up. Before every checkpoint and before the proof stage the run waits until every submitted point is durable, so a checkpoint never claims progress past a proof residue that failed to become durable: a failed write stops the run with an error at that wait, the last checkpoint is earlier, and resume recomputes the residue. On resume, every point at or below the checkpoint's k must be present and intact, or resume refuses. A rollback past a point rewrites it.

Directories created for proof residues or checkpoints are made durable: after creating them, the parent of every new directory is fsynced.

Ownership: opening the proof store takes an advisory lock on `<proof-dir>/M<p>.residues/.lock` (`std::fs::File::try_lock`) and holds it until the PRP run returns, through the recurrence, the proof stage, and verification, with or without a checkpoint. A second run on the same proof store for the same exponent is refused (exit 6) before it writes anything, whatever checkpoint directory it uses. The library takes the lock in `ProofStore::open`, so the rule also holds outside the CLI; clones of a store share its lock. Lock order: the checkpoint lock first, then the proof store lock.

A proof residue file is read through one open handle: its size is checked on that handle, the read is bounded to the expected size, and a file that grows, shrinks, or changes between the size check and the read is an error, never a panic.

Proof residue file (`<proof-dir>/M<p>.residues/<k>`): `"MF-PROOF-RESIDUE"` (16 bytes), u32 p, u64 k, the limbs, SHA3-256 of the preceding bytes.

## Proof file (PRPLL version 2)

```text
PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=<P>\nNUMBER=M<p>\n
B = x_p                     (p - 1) / 8 + 1 bytes, little-endian
M_1 ... M_P                 (p - 1) / 8 + 1 bytes each
```

File name `<p>-<P>.proof`. Hash chain: h_0 = SHA3-256(bytes of B); h_i = SHA3-256(h_(i-1) || bytes of M_i); the exponent e_i is the first 8 bytes of h_i read as a little-endian u64. SHA3-256 is the standard FIPS 202 function (padding 0x06).

Generation (`computeProof`): at level i (0 based), with s = 2^(P - i - 1), the residues at points[s (2j + 1) - 1] for j = 0 .. 2^i - 1 are combined left to right with a stack: push each, and while bit t of j is set, pop the top T and replace the new top U by U^(e_(i - t)) * T. The single remaining value is M_(i+1). A zero middle is an error. Generation is not checked; it is verified afterwards. The stop flag and the budget are checked before every combination step, so a stop request or an exhausted budget ends generation promptly.

Validation, shared by parsing and verification (a directly constructed `Proof` gets the same checks): p prime and at least 3, power in 1..=13, every residue (p - 1) / 8 + 1 bytes and canonical, every middle nonzero. The generation schedule is not required: a file whose power gives colliding points for p (for example M13 at power 3) is parsed and verified.

Proof equation (`verify_equation`, as upstream `Proof::verify`): A = 3, B from the file, span = p. For each level i: B <- M^e * (span odd ? B^2 : B); A <- A^e * M; span <- (span + 1) / 2. The proof is rejected if the folded A or B is zero, and otherwise accepted iff A^(2^span) == B. The final A^(2^span) uses the same Gerbicz engine as PRP with start value A in place of 3: C^(2^B) * A == C * x_k. Block B = clamp(round(sqrt(span / 10)), 2, 1000), check step about span / 10 (rounded to a multiple of B).

Verification (`verify`, used by the CLI and by self verification): validation; then, for p below 2^17 (`DIRECT_RECOMPUTE_BELOW`), a direct checked recomputation of 3^(2^p) that must equal B; then the proof equation. The output says which mode ran (`terminal_recomputed`). The recomputation is the cost of the PRP test itself, seconds below the cutoff; it is needed there because for a small modulus the equation alone accepts many wrong terminal values (for M11 at power 1, 1992 of the 2046 wrong terminal values have a middle that the equation accepts, and accepted middles include units, so no gcd test would help).

What an accepted proof means at or above the cutoff: the file satisfies the version 2 proof equation, the same relation the upstream verifier checks. That is sound against computation errors and against a prover who skipped the work. It is not sound against a prover who knows a factor of M_p, or elements of small order modulo M_p: for a prime M_p the group order is known, and the sign of B is never bound because the first odd span squares B. The challenge e_i is 64 bits. A probable prime claim from a proof needs an independent test. A proof whose B is 9 claims a probable prime result; otherwise a composite result. Because the equation accepts B and N - B alike, a file whose B is 9 or N - 9 has its claim decided by a sign the equation does not bind: without a direct recomputation `verify` reports such a claim as `undetermined`.

Proof generation stages the file as `<name>.staged`, reads it back, requires its terminal residue to equal the run's own x_p, and verifies it; only a verified (or, without self verification, generated) proof is renamed to its final name. A rejected or failed proof stays staged for inspection and never replaces an earlier file.

Rejections, each a distinct error: `BadHeader` (not version 2 or malformed), `WrongExponent` (with `--exponent`), `InvalidPower` (power, or an exponent that is not a prime of at least 3), `Truncated`, `TrailingData`, `NoncanonicalResidue` (bits at or above p, or a wrong length), `ZeroResidue`, `Rejected` (the recomputation or the final equation fails, or a folded value is zero). The header is checked and the file size compared with it before any residue is read; format validation (canonical bytes, nonzero middles) works on borrowed bytes and allocates nothing, and the buffers that hold the residues are reserved fallibly, so an allocation failure is an error, not an abort. Peak memory for reading a proof is the residue bytes held once. Verification then builds residues and backend registers as computation requires.

Proof stage of a PRP run, reported separately from the result (`proof_stage`): `not_requested`, `generated` (no self verification requested), `verified`, `rejected`, `failed` (generation, write, or verification error), `interrupted` (budget or stop request). The PRP result is kept in every case. Exit codes: `failed` 5, `rejected` 7, `interrupted` 3. After the final Gerbicz check passes, the run writes a completed checkpoint before starting the proof stage. A later `resume` validates the saved state and restarts the whole proof stage without repeating the final PRP check interval. Generation and verification progress within the proof stage is not saved.

## Lucas-Lehmer with the Jacobi check

s_0 = 4, s_(k+1) = s_k^2 - 2; M_p is prime iff s_(p-2) = 0. p = 2 is a special case (prime). The reported res64 is the low 64 bits of s_(p-2).

With w = 2 + sqrt(3), s_k = w^(2^k) + w^(-2^k), so for k >= 1, s_k - 2 = 3 t^2 for an integer t, and (3 | N) = -1. So the Jacobi symbol (s_k - 2 | N) must be -1 for prime and composite N alike. s = 2 and s = N - 2 (the path of a zeroed residue) count as failures; a symbol of 0 is inconclusive (neither pass nor failure). A residue that passes becomes the verified state. A zero residue before k = p - 2 is a failure.

This check catches about half of all errors (`tests/ll_jacobi.rs` measures this on single bit flips), and an error is caught at the first check after it or never. It is not the Gerbicz check; an undetected LL error yields a wrong final residue.

Every exported residue that can become a verified state, a checkpoint, or the final result is compared with the resident value right after its export; a mismatch counts as a detected fault and rolls back. A zero final residue (the prime verdict) is recomputed from the last verified residue before it is reported. Error counts (Jacobi failures and backend faults) are stored in the checkpoint and carried across sessions.

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

Metadata: `format_version`, `protocol` (`prp-base3-gerbicz/1` or `ll-jacobi/1`), `exponent`, `k`, `state` (`verified`, `completed`, or `provisional`), `verified_k`, `n_errors`, `proof_durable_through` (largest proof point at or below k), `backend` (kind, build, plan, device), `config` (effective configuration, including block size, check step, proof power and directory), `requested` (configuration as requested), `elapsed_secs`, `crate_version`.

Residues: PRP stores [C_old, x_k] of a verified state at k < p. A completed PRP checkpoint stores [C_old, x_k, x_p], with k = roundUp(p, B). Resume checks the Gerbicz equation on the pair and requires x_p != 0 and x_p^(2^(k-p)) == x_k. This retains the square-root ambiguity described above. No PRP iterations are repeated after these checks; the iteration timing and passed-check count in that resume report are zero. LL stores [s_k, s_verified]: the provisional residue at k and the last Jacobi verified residue at `verified_k`.

Write: a temporary file of its own, `<path>.tmp.<pid>.<n>`, created exclusively with owner only permissions, written, given the permissions of the file it replaces (or the default mode under the umask), and fsynced (concurrent writers never share or truncate each other's temporary file; a file already present at a candidate name belongs to someone else, is left untouched, and another name is tried; a failed write removes only the file it created); if the old `<path>` loads and passes its hash, rename it to `<path>.prev`, otherwise remove it (so a corrupt file never replaces a good `.prev`); rename the temporary file to `<path>`; fsync the directory. If `<path>` or `<path>.prev` was loadable before a save, a crash at any point of the save leaves one of them loadable. On a save, a `<path>` that loads becomes `.prev`, a `<path>` proven corrupt is removed, and one that cannot be read for another reason (an I/O error, a newer format) is kept as `<path>.unloadable`. Temporary files another process left more than an hour ago are removed. Load: reads are bounded by the header sizes; the hash is checked; only a missing or corrupt `<path>` falls back to `<path>.prev` (a version or I/O error is reported, not skipped). `resume` also falls back to `<path>.prev` when `<path>` loads but fails the resume validation (the PRP on-load check, or a missing proof residue); the rejected file is renamed to `<path>.rejected` and the report names the file used, and if `.prev` fails too the newer file is put back. It does not fall back on an identity or configuration mismatch.

Portable form (`checkpoint-export`): the last verified state with backend kind `portable` and an empty build, plan, and device. An LL state rewinds to its last Jacobi verified residue first, so nothing unverified changes backend. Any backend resumes a portable file (`resume --backend`, or `work` from the assignment's checkpoint path); its next save carries that backend's identity. The configuration must still match.

A checkpoint is refused when its protocol, exponent, backend identity, or configuration (apart from the checkpoint path and an existing PRP proof directory) differs from the run, and, for PRP, when its state, residue count, or iteration range is invalid, `verified_k` differs from `k`, or `proof_durable_through` is not the largest proof point at or below `k`. What each check establishes: the hash detects damaged bytes (it is not keyed and authenticates nothing); the PRP on-load equation detects a damaged residue pair; the iteration count and the configuration are trusted metadata that no local check can authenticate.

`resume --proof-dir <directory>` relocates an existing PRP proof schedule. The directory must contain `M<p>.residues` with every point required by the checkpoint. The effective configuration and the next saved checkpoint record the new path, including when resuming a completed state. Missing directories are refused before changing the checkpoint. LL checkpoints and PRP checkpoints without a proof schedule refuse this option. Both two-residue verified states and three-residue completed states are supported when the build identity matches; older programs reject the completed state.

`run` refuses to start when its checkpoint or `.prev` file already exists, and names `resume` (or deleting both files to start over). `run` and `resume` with a checkpoint hold an advisory lock on `<path>.lock` (`std::fs::File::try_lock`) for their whole life and refuse when another process holds it. A report names a checkpoint to resume from only when that file (or its `.prev`) exists.

Checkpoint compatibility is defined by the format version and the backend identity (kind, build, plan, device). The build identity contains the crate version and SHA3-256 of every implementation file under `src/`, `Cargo.toml`, `Cargo.lock`, and `build.rs`. Each file contributes its path length, path bytes, content length, and content bytes in sorted path order, with lengths encoded as little-endian u64. Source content is included at compilation, so no Git checkout, build script, or native compiler is needed. This detects uncommitted source changes and changes between revisions that kept the same crate version. It is a content fingerprint, not a Git revision, compiler identity, or promise of identical machine code. GPU backends additionally record the kernel source hash, and CUDA includes compiler and driver details. `doctor` reports the source identity.

Checkpoints from older version-only builds are refused. Complete existing work with the original binary before upgrading. Proof files are unaffected by this checkpoint identity change.

There is no compatibility with upstream savefiles.

## Differences from upstream PRPLL

1. Checkpoints store both C_old and x_k; upstream stores C only and rebuilds x_k = 3 C^(2^B - 1) (a chain that needs B not to be a power of two). Ours loads with the check equation instead.
2. Only verified PRP states are checkpointed; upstream also writes unverified saves at log steps. A crash can lose up to one check interval.
3. Proof verification uses the PRP style Gerbicz check with start A; upstream checks the verification squarings CERT style (`certVerify`) and saves verification progress. A stopped verification here restarts.
4. Block size and check step are configurable (for small exponents and tests); defaults are upstream's.
5. Proof generation is attempted once at the configured power; on failure the result is reported without a proof and the proof stage is `failed` (a rejected proof is `rejected`). `work` reports such a result without a proof, as upstream does. Upstream retries at lower powers.
6. The default proof power is lowered until the schedule is valid for p; upstream does not handle very small p.
7. The "same wrong residue twice" rule compares complete residues, not res64.
8. Backend faults count as failed checks. A failed on-load check is retried once before the state is rejected.
9. The Jacobi symbol is computed on the host; upstream uses GPU GCD kernels.
10. LL stores the provisional and verified residues in one checkpoint file; upstream uses two savefile series.
11. P-1 inside PRP (upstream `Pm1Prp`) is not implemented; P-1 runs as a separate step before the test (see below).
12. A stop request ends an LL run after the current batch of at most 1000 iterations; PRP stops at the next block boundary as upstream.
13. Verification is stricter than upstream: zero middles, noncanonical residues, and a folded A or B of zero are rejected, and for p below 2^17 the terminal residue is recomputed directly. Upstream checks the proof equation only.
14. Exported residues (PRP x_p, LL anchors, checkpoints, and s_(p-2)) are compared with the resident value after export, PRP x_p must be the square of x_(p-1), and an LL zero is replayed.

## Assignment and result lines

`work` reads the worktodo forms of the pinned upstream parser: `PRP=` and `PRPDC=` with `[AID,]1,2,p,-1[,tf,saved[,base,residue_type]]` or a bare `[AID,]p`, and `Test=` and `DoubleCheck=` with `[AID,]1,2,p,-1[,...]` or `[AID,]p[,...]`. An empty or `N/A` first field is skipped; a 32 digit hex field is the assignment ID. The tokenizer follows Prime95's grammar: a trailing quoted list of known factors (any non-digit separates them), a `key=value,...;` prefix, case insensitive work types, and `#`, `;`, `//`, and `[Worker #n]` lines ignored. PRP residue types 1, 3, and 5 are accepted; known factors (each must divide 2^p - 1) make the test one of the cofactor C = N / F, reported as type 5: the residue is 3^(N-1) mod N as Prime95 reports it, and C is a probable prime when 3^(N-1) = 3^(F-1) mod C. `Cert=[AID,]1,2,p,-1,squarings` lines run first, as gpuowl orders them. Refused and left in the file: any other number form (k other than 1, base other than 2, c other than -1), a PRP base other than 3, residue types 2 and 4, known factors with a residue type other than 5, a non numeric tf or saved field, a composite exponent, a signed exponent, a line with a second `=`, and PRP of M2.

Factoring lines: `Factor=[AID,]p,bit_lo,bit_hi` (trial factoring), `Pminus1=[AID,]1,2,p,-1,B1,B2`, and `Pfactor=[AID,]1,2,p,-1,tf,saved` (bounds B1 = p / 10 between 1,000 and 1,000,000, stage 1 only). A PRP line with saved above 0 gets the Pfactor treatment before the test: a factor ends the assignment; otherwise a `PM1` result is written, the journal records `pm1:<line>`, and the line is rewritten with saved 0. P-1 in `work` saves stage 1 progress (`M<p>-pm1-<key>.ckpt`) and is limited to p <= 10,000,000 (the final gcd runs on the host); trial factoring saves the next untested k (`M<p>-tf-<key>.state`).

Durability: the result line and a journal entry (`<results>.done`, the line itself) are appended and fsynced with their directory, under the results file's `.lck` lock, before the line is removed from the worktodo file under its own `.lck` lock. A fragment a crash left at the end of the results file is moved to `<results>.torn` first. A line is treated as done when its text is in the journal, or when its assignment ID has a result of the same exponent and worktype with status P or C; its checkpoints are then removed. Checkpoints are named `M<p>-<prp|ll>-<key>.ckpt`, where the key is the assignment ID in lowercase or the first 8 bytes of SHA3-256 of the line, and a checkpoint is adopted only when its stored assignment equals the line.

Each finished assignment appends one JSON line with the fields upstream writes: `status` (`P` or `C`), `exponent`, `worktype` (`PRP-3` or `LL`), `res64`; for PRP also `res2048`, `residue-type` 1, `errors.gerbicz`, and `proof` with `version` 1 (the value upstream writes for its version 2 files), `power`, `hashsize` 64, and the lowercase hex MD5 of the proof file; for LL `shift-count` 0, `error-code` "00000000", and `errors.jacobi` when nonzero; then `program` (name, version, `port`, the PrimeNet numeric platform code upstream writes, for example 10 on 64 bit macOS and 8 on 64 bit Linux, `backend`, and os), `aid` when present, and a UTC `timestamp`. Error counts cover every session of the run. Upstream fields with no counterpart here (`fft-type`, `fft-length`, `user`, `uid`) are omitted. MD5 is used only to name the proof file in this line.

Factoring results: TF lines carry `worktype` `TF`, `status` `F` or `NF`, `bitlo`, `bithi`, `rangecomplete`, and `factors` (keys as recalled from mfaktc, not checked against its output); P-1 lines carry `worktype` `PM1`, `B1`, `B2` when stage 2 ran, and `factors`, the keys upstream PRPLL's P-1 wrote. Every factor is checked with `check-factor`'s test before it is reported.

`check-factor` accepts q when 2^p mod q is 1 and q is neither 1 nor M_p; a composite divisor is accepted too. It also reports whether q has the form 2kp+1 with q = +-1 mod 8. For odd p every divisor of M_p has that form, so a divisor without it could only come from an arithmetic error.
