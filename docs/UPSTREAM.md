# Upstream feature map

This page maps what the GIMPS programs do to what this program does. It was
written from their source: Prime95 30.19 (`commonc.c`, `commonb.c`,
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
| PRP bases other than 3, k*b^n+c numbers | Prime95 | not planned: every backend here is built on arithmetic modulo 2^p - 1 |
| `Test=`, `DoubleCheck=` (LL) | Prime95, gpuowl, Mlucas | done |
| `Cert=` | Prime95, gpuowl | written: run first (gpuowl's order), Gerbicz checked |
| `Factor=` with optional `M` prefix, `//` comments | mfaktc, mfakto | written |
| `Pminus1=` with tf_bits, B2_start, known factors | Prime95, Mlucas | written (B2_start is covered: stage 2 starts at B1) |
| `Pfactor=`, old and new forms | Prime95 | written (bounds from a fixed rule, not Prime95's cost model) |
| `Pplus1=` with nth_run | Prime95 | written |
| `ECM=`, `ECM2=` with B2, curves, sigma | Prime95 | written (host arithmetic, small exponents) |
| `ECMSTAGE2=` with GMP-ECM save files | Prime95 | not planned |
| `B1=..,B2=..;` prefixes, `[Worker #n]` sections, `;` and `#` comments, case insensitive work types | Prime95, AutoPrimeNet, mfaktc | written |
| `AdvancedTest=`, `AdvancedFactor=` | Prime95 | not planned: test and debugging forms |
| Fermat numbers (Pépin test, `Fermat` lines) | Mlucas | not planned: arithmetic modulo 2^(2^m) + 1 needs other backends |

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
| Shift counts | not planned: every run here starts unshifted, reported as `shift-count` 0 |

## Algorithms and kernels

| Upstream feature | Where | Here |
|---|---|---|
| Gerbicz checked PRP, proofs (version 2), LL with Jacobi check | Prime95, gpuowl | done |
| CERT squarings with SHA3-256 hash | Prime95, gpuowl | written |
| GPU trial factoring to 2^95 | mfaktc, mfakto | written (WGSL, three word Montgomery); no CUDA or OpenCL kernel |
| GPU sieving, 4620 class split, `MoreClasses` | mfaktc, mfakto | not planned: the host sieves and batches candidates |
| P-1 stage 2 with prime pairing and polynomial multiplication | Prime95, gpuowl | not planned now: stage 2 here is one prime at a time |
| ECM on Edwards curves, GMP-ECM param 3, big stage 2 | Prime95 | not planned now: Montgomery curves, Suyama sigma, standard continuation |
| P+1 stage 2 | Prime95 | written (one Lucas ladder per prime) |
| Self test (mfaktc `-st`, Prime95 torture test) | mfaktc, Prime95 | written: `selftest` |
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
