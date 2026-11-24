# Upstream feature map

This page maps what the GIMPS programs do to what this program does. The
programs are the ones the GIMPS download page lists: Prime95 (mprime),
PRPLL (gpuowl), mfaktc, mfakto, Mlucas, and AutoPrimeNet. It was written
from their source: Prime95 30.19 (`commonc.c`, `commonb.c`,
`cert.c`, `ecm.cpp`, `primenet.c`, `primenet.h`, `proof_getdata.c`),
gpuowl/PRPLL, mfaktc, mfakto, Mlucas, and AutoPrimeNet. Prime95's source is
under the GIMPS license and was read for behavior and file formats only;
no code was copied. Status words are those of [STATUS.md](STATUS.md):
**done** (tested here), **written** (code and tests written, not yet run),
**not planned** (with the reason).

## Work types and line grammar

| Upstream feature | Where | Here |
|---|---|---|
| `PRP=`, `PRPDC=` with k,b,n,c, tf_bits, tests_saved, base, residue type | Prime95, gpuowl, Mlucas | done for 1,2,p,-1 base 3 type 1; written for types 3 and 5 |
| Known factor lists (`"f1,f2"`), cofactor PRP (type 5) | Prime95, Mlucas, PrimeNet | written: each factor checked to divide 2^p - 1, cofactor decided on the host |
| Residue types 2 and 4 | Prime95 | not planned: they need x_(p-1), which the engine does not keep, and PrimeNet assigns types 1 and 5 |
| k*b^n+c numbers | Prime95 | written: `PRP=k,b,n,c` lines (residue types 1 to 4, base 3) run on the host in big integers with a Gerbicz product check over the chain of b-th powers, up to 2^20 bits; result lines carry `k`, `b`, `n`, `c`; no FFT backend, no cofactor tests, no other PRP bases |
| `Test=`, `DoubleCheck=` (LL) | Prime95, gpuowl, Mlucas | done |
| `Cert=` | Prime95, gpuowl | written: run first (gpuowl's order), Gerbicz checked |
| `Factor=` with optional `M` prefix, `//` comments | mfaktc, mfakto | written |
| `Pminus1=` with tf_bits, B2_start, known factors | Prime95, Mlucas | written (B2_start is covered: stage 2 starts at B1) |
| `Pfactor=`, old and new forms | Prime95 | written (bounds from a fixed rule, not Prime95's cost model) |
| `Pplus1=` with nth_run | Prime95 | written |
| `ECM=`, `ECM2=` with B2, curves, sigma | Prime95 | written (host arithmetic, small exponents) |
| `ECMSTAGE2=` with GMP-ECM save files | Prime95 | written: Suyama (PARAM=0) curves, Z normalized, skip and count fields, each N must divide 2^p - 1; `ECMSTAGE2N=` (no number given) is refused |
| `B1=..,B2=..;` prefixes, `[Worker #n]` sections, `;` and `#` comments, case insensitive work types | Prime95, AutoPrimeNet, mfaktc | written |
| `AdvancedTest=`, `AdvancedFactor=` | Prime95 | not planned: test and debugging forms |
| Fermat numbers (Pépin test, `Fermat` lines) | Mlucas | written: `Fermat,Test=m` (F1 to F20) and `PRP=1,2,2^m,1` run as the host k*b^n+c test, where residue type 2 is Pépin's test; no negacyclic transform |

## Result lines

| Upstream feature | Here |
|---|---|
| PRP JSON (`res64`, `res2048`, `residue-type`, `known-factors`, `errors`, `proof`) | done for type 1; written for types 3 and 5 and `known-factors` |
| LL JSON | done |
| TF JSON (`bitlo`, `bithi`, `rangecomplete`, quoted `factors`) as mfaktc writes it | written |
| P-1 JSON | written (`worktype` "PM1" with `B1`/`B2`, which AutoPrimeNet accepts) |
| P+1 JSON (`worktype` "P+1", `b1`, `b2`, `start`) | written |
| ECM JSON (`worktype` "ECM", `b1`, `b2`, `curves`, `sigma`) | written |
| CERT JSON (`worktype` "Cert", `sha3-hash`, `squarings`) | written |
| Security codes (`security-code`) and mfaktc's checksum | not planned: they are keyed to the upstream programs |
| Shift counts | written for PRP: `run --shift S` and `work --shift S` hold every value rotated by a tracked per-register shift (squaring doubles it, multiplication adds), with Gerbicz checks, proofs, and checkpoints on the true values; PRP result lines carry `shift-count` when it is not 0; LL is refused with a shift |

## Algorithms and kernels

| Upstream feature | Where | Here |
|---|---|---|
| Gerbicz checked PRP, proofs (version 2), LL with Jacobi check | Prime95, gpuowl | done |
| Wavefront exponents (above 100,000,000) on a GPU | gpuowl | not yet: the wgpu backend tops out at p = 40,894,464 (transform length 2^24, the end of its GF((2^31 - 1)^2) field; lengths above 2^22 written, not run); reaching 100,000,000 needs another modulus. The CUDA backend reaches further, unrun |
| Speed comparable to gpuowl and Prime95 | gpuowl, Prime95 | not measured: no tuned baseline has been run (STATUS F16) |
| CERT squarings with SHA3-256 hash | Prime95, gpuowl | written |
| GPU trial factoring to 2^95 | mfaktc, mfakto | written (WGSL, three word Montgomery); no CUDA or OpenCL kernel |
| GPU sieving | mfaktc, mfakto | written: per segment, the device strikes the sieve primes below 2^16 from host computed roots and tests the survivors; no 4620 class split or `MoreClasses` (one class per segment instead) |
| P-1 stage 2 with prime pairing and polynomial multiplication | Prime95, gpuowl | prime pairing written: baby step giant step with D = 210 and f(t) = t^2, one subtraction and one multiplication per pair; subtraction is native on the CPU backend and a host round trip elsewhere; no polynomial multiplication (FFT continuation) |
| ECM on Edwards curves, GMP-ECM param 3, big stage 2 | Prime95 | not planned now: Montgomery curves, Suyama sigma, standard continuation |
| P+1 stage 2 | Prime95 | written (one Lucas ladder per prime) |
| Self test (mfaktc `-st`, Prime95 torture test) | mfaktc, Prime95 | written: `selftest` checks correctness; it is not a stress test with heat and memory load |
| FFT tuning (`-tune`, `-ctune`) | gpuowl | written: `tune` over the opt-in variants |
| Several workers, one per device (`-workers`, `-pool`) | gpuowl | written: `supervise` runs one `work` per device; a shared pool file is not planned |

## PrimeNet

| Upstream feature | Where | Here |
|---|---|---|
| v5 API: `uc`, `po`, `ga`, `ap`, `au`, `ar` | Prime95, AutoPrimeNet | written: `primenet-v5`, with AutoPrimeNet's placeholder security fields for programs other than Prime95; never run against the server |
| CERT start value download (`proof_get_data`, MD5 checked) | Prime95, AutoPrimeNet | written |
| Proof upload | Prime95, AutoPrimeNet | written in `primenet` (never run against the server) |
| Manual web forms | former gpuowl scripts | written: `primenet` |
| Email notices, mersenne.ca submission, rolling averages, GPU detection | AutoPrimeNet | not planned |
