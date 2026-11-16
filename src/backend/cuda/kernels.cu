// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.
//
// MersenneForge CUDA kernels: squaring and multiplication modulo N = 2^p - 1
// by an irrational-base discrete weighted transform over GF(q^2), q = 2^61 - 1.
// Compiled at run time by NVRTC. The Rust model in model.rs mirrors every
// kernel stage for stage; docs/CUDA.md has the derivations.
//
// Rules for this file:
// - No include directive: it is compiled both by NVRTC and, for host emulation, by a
//   host C++ compiler behind a shim that defines __global__, __device__,
//   __forceinline__, __umul64hi, threadIdx, blockIdx and blockDim.
// - Every kernel is plain data parallel: a thread computes from its global
//   index only. No shared memory, no __syncthreads, no atomics, no
//   dependence on the block size. A thread writes only its own outputs, and
//   kernels that read neighbors (the carry passes) are out of place.
// - Every pointer parameter addresses 64-bit elements; every scalar parameter
//   is unsigned long long. Signed values travel as their two's complement bits.
//
// ponytail: radix-2 butterflies, one kernel launch per transform stage, and
// separate weight, unweight and carry launches. The ceiling is memory
// bandwidth and launch overhead; the upgrade path is radix-4 or radix-8
// stages over shared memory and fused weight, carry and transform kernels.

typedef unsigned long long u64;
typedef long long i64;

#define MF_Q 0x1fffffffffffffffULL      /* q = 2^61 - 1 */
#define MF_HALF_Q 0x0fffffffffffffffULL /* (q - 1) / 2 */

// a + b mod q. Input: a, b in [0, q). Output: [0, q). a + b < 2q < 2^62.
static __device__ __forceinline__ u64 mf_add(u64 a, u64 b) {
    u64 s = a + b;
    return s >= MF_Q ? s - MF_Q : s;
}

// a - b mod q. Input: a, b in [0, q). Output: [0, q).
static __device__ __forceinline__ u64 mf_sub(u64 a, u64 b) {
    return a >= b ? a - b : a + MF_Q - b;
}

// a * b mod q.
// Input: a, b < 2^61. Output: [0, q).
// Proof: the product P = a * b < 2^122, so hi = P >> 64 < 2^58 and
// (hi << 3) | (lo >> 61) = P >> 61 < 2^61 without overflow. Since
// 2^61 = 1 mod q, P = (P >> 61) + (P mod 2^61) mod q; that sum r1 < 2^62.
// Folding again, r2 = (r1 mod 2^61) + (r1 >> 61) <= (2^61 - 1) + 1 = q + 1,
// still congruent to P. One conditional subtraction leaves [0, q).
static __device__ __forceinline__ u64 mf_mulmod(u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 r = (lo & MF_Q) + ((hi << 3) | (lo >> 61));
    r = (r & MF_Q) + (r >> 61);
    return r >= MF_Q ? r - MF_Q : r;
}

// x * 2^s mod q as a 61-bit rotation. Input: x in [0, q), s in [0, 61).
// Output: [0, q). A rotation of a 61-bit value other than all ones is not all
// ones, so the result never equals q. For s = 0, x >> 61 = 0.
static __device__ __forceinline__ u64 mf_rotl(u64 x, u64 s) {
    return ((x << s) & MF_Q) | (x >> (61 - s));
}

// Bit offset of word i: ceil(p * i / n). p * i < 2^30 * 2^24 fits easily.
static __device__ __forceinline__ u64 mf_offset(u64 p, u64 log_n, u64 i) {
    return (p * i + (1ULL << log_n) - 1) >> log_n;
}

// Width of word i (i in [0, n)): offset(i + 1) - offset(i), floor or ceil of p / n.
static __device__ __forceinline__ u64 mf_width(u64 p, u64 log_n, u64 i) {
    return mf_offset(p, log_n, i + 1) - mf_offset(p, log_n, i);
}

// Weight of word i is r^e with r = 2^kinv the n-th root of 2 modulo q and
// e = ceil(p i / n) n - p i = (-(p i)) mod n. Returned as the rotation k e mod 61.
static __device__ __forceinline__ u64 mf_weight_shift(u64 n, u64 p, u64 kinv, u64 i) {
    u64 e = (n - ((p * i) & (n - 1))) & (n - 1);
    return (kinv * e) % 61;
}

// Balanced low digit of v for width w: v - round(v / 2^w) 2^w, in
// [-2^(w-1), 2^(w-1)). Uses two's complement & and arithmetic >>, which NVRTC
// and the host compilers provide (and C++20 guarantees).
static __device__ __forceinline__ i64 mf_low(i64 v, u64 w) {
    i64 half = (i64)1 << (w - 1);
    i64 mask = ((i64)1 << w) - 1;
    return ((v + half) & mask) - half;
}

// Carry of v for width w: (v - low(v)) / 2^w, an exact division.
static __device__ __forceinline__ i64 mf_carry(i64 v, u64 w) {
    return (v - mf_low(v, w)) >> w;
}

// Weighted input: t[i] = x[i] * weight(i) as an element of GF(q^2) with zero
// imaginary part. n threads. x holds balanced words, |x| < 2^30.
// ponytail: zero imaginary input wastes half of each GF(q^2) element, a 2x
// ceiling on transform length per word; the upgrade is the right-angle
// convolution (pack word pairs as re + i im and twist by a 4n-th root of unity).
extern "C" __global__ void mf_weight(u64* t, const u64* x, u64 n, u64 p, u64 kinv) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    i64 v = (i64)x[i];
    u64 m = v < 0 ? (u64)(v + (i64)MF_Q) : (u64)v;
    t[2 * i] = mf_rotl(m, mf_weight_shift(n, p, kinv, i));
    t[2 * i + 1] = 0;
}

// One decimation-in-frequency radix-2 stage (Gentleman-Sande), natural order
// in, bit-reversed order out after all stages. h is the half span; stride is
// n / (2 h), so tw[stride * j] = w_(2h)^j. n / 2 threads, one butterfly each;
// a thread touches only the pair (i0, i0 + h).
extern "C" __global__ void mf_fwd_stage(u64* t, const u64* tw, u64 n, u64 h, u64 stride) {
    u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n / 2) return;
    u64 j = k & (h - 1);
    u64 i0 = 2 * (k - j) + j;
    u64 i1 = i0 + h;
    u64 ur = t[2 * i0], ui = t[2 * i0 + 1];
    u64 vr = t[2 * i1], vi = t[2 * i1 + 1];
    u64 wr = tw[2 * (stride * j)], wi = tw[2 * (stride * j) + 1];
    t[2 * i0] = mf_add(ur, vr);
    t[2 * i0 + 1] = mf_add(ui, vi);
    u64 dr = mf_sub(ur, vr), di = mf_sub(ui, vi);
    t[2 * i1] = mf_sub(mf_mulmod(dr, wr), mf_mulmod(di, wi));
    t[2 * i1 + 1] = mf_add(mf_mulmod(dr, wi), mf_mulmod(di, wr));
}

// One decimation-in-time radix-2 stage (Cooley-Tukey) with the inverse root,
// bit-reversed order in, natural order out after all stages. The twiddles lie
// in the norm-1 subgroup of GF(q^2), where the inverse is the conjugate
// (a + b i)^q = a - b i. n / 2 threads. The 1/n scale is applied by mf_unweight.
extern "C" __global__ void mf_inv_stage(u64* t, const u64* tw, u64 n, u64 h, u64 stride) {
    u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n / 2) return;
    u64 j = k & (h - 1);
    u64 i0 = 2 * (k - j) + j;
    u64 i1 = i0 + h;
    u64 wr = tw[2 * (stride * j)], wi = tw[2 * (stride * j) + 1];
    u64 xr = t[2 * i1], xi = t[2 * i1 + 1];
    // v = x * conj(w) = (xr wr + xi wi) + (xi wr - xr wi) i
    u64 vr = mf_add(mf_mulmod(xr, wr), mf_mulmod(xi, wi));
    u64 vi = mf_sub(mf_mulmod(xi, wr), mf_mulmod(xr, wi));
    u64 ur = t[2 * i0], ui = t[2 * i0 + 1];
    t[2 * i0] = mf_add(ur, vr);
    t[2 * i0 + 1] = mf_add(ui, vi);
    t[2 * i1] = mf_sub(ur, vr);
    t[2 * i1 + 1] = mf_sub(ui, vi);
}

// Pointwise square in GF(q^2): (a + b i)^2 = (a^2 - b^2) + 2ab i. n threads.
extern "C" __global__ void mf_sqr(u64* t, u64 n) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 a = t[2 * i], b = t[2 * i + 1];
    t[2 * i] = mf_sub(mf_mulmod(a, a), mf_mulmod(b, b));
    t[2 * i + 1] = mf_mulmod(mf_add(a, a), b);
}

// Pointwise product t <- t * t2 in GF(q^2). n threads.
extern "C" __global__ void mf_mul(u64* t, const u64* t2, u64 n) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 a = t[2 * i], b = t[2 * i + 1];
    u64 c = t2[2 * i], d = t2[2 * i + 1];
    t[2 * i] = mf_sub(mf_mulmod(a, c), mf_mulmod(b, d));
    t[2 * i + 1] = mf_add(mf_mulmod(a, d), mf_mulmod(b, c));
}

// Unweight and scale: z[i] = re(t[i]) * weight(i)^-1 * n^-1, read as the
// balanced integer in [-(q-1)/2, (q-1)/2]. n^-1 = 2^(-log_n) mod q, so the
// whole factor is the rotation (122 - s - log_n) mod 61, with s < 61 and
// log_n <= 24 keeping the left side positive. The imaginary part of an exact
// result is zero and is not read. n threads.
extern "C" __global__ void mf_unweight(u64* z, const u64* t, u64 n, u64 p, u64 log_n, u64 kinv) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 s = mf_weight_shift(n, p, kinv, i);
    u64 c = mf_rotl(t[2 * i], (122 - s - log_n) % 61);
    i64 v = c > MF_HALF_Q ? (i64)c - (i64)MF_Q : (i64)c;
    z[i] = (u64)v;
}

// Carry pass A, a gather. Each coefficient z[s] is split, along the word
// widths starting at word s, into `pieces` parts: pieces - 1 balanced digits
// and a final remainder. Part j lands on word s + j (mod n; 2^p = 1 mod N, so
// the wrap from word n - 1 to word 0 carries weight 1). Thread k sums the
// parts that land on word k, from sources k, k - 1, ..., k - pieces + 1, and
// adds add0 (the LL subtraction, as two's complement bits) at word 0.
// The host chooses pieces so that the remainder has magnitude at most 1, which
// bounds |a[k]| <= (pieces - 1) 2^(w_k - 1) + 3; see docs/CUDA.md.
// pieces may exceed n for tiny exponents; the index arithmetic wraps. n threads.
extern "C" __global__ void mf_carry_a(u64* a, const u64* z, u64 n, u64 p, u64 log_n,
                                      u64 pieces, u64 add0) {
    u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    i64 acc = k == 0 ? (i64)add0 : 0;
    for (u64 j = 0; j < pieces; j++) {
        u64 s = (k - j) & (n - 1);
        i64 v = (i64)z[s];
        for (u64 t = 0; t < j; t++) {
            v = mf_carry(v, mf_width(p, log_n, (s + t) & (n - 1)));
        }
        acc += j + 1 < pieces ? mf_low(v, mf_width(p, log_n, k)) : v;
    }
    a[k] = (u64)acc;
}

// Carry pass B: x[k] = low(a[k]) + carry(a[k - 1]), with word 0 receiving the
// carry of word n - 1. Each thread recomputes its neighbor's carry, so there is
// no cross-thread dependency. Output |x[k]| <= 2^(w_k - 1) + floor(pieces / 2).
// n threads.
extern "C" __global__ void mf_carry_b(u64* x, const u64* a, u64 n, u64 p, u64 log_n) {
    u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    u64 prev = (k - 1) & (n - 1);
    i64 d = mf_low((i64)a[k], mf_width(p, log_n, k));
    i64 c = mf_carry((i64)a[prev], mf_width(p, log_n, prev));
    x[k] = (u64)(d + c);
}
