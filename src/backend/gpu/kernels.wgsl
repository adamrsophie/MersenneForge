// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.
//
// Exact NTT arithmetic modulo 2^p - 1 over GF(q^2), q = 2^31 - 1.
// Every function here has a line by line mirror in model.rs; see that file
// for the transform, the capacity bound, and the carry proof.

const Q: u32 = 0x7fffffffu;
const WG: u32 = 256u;

struct Params {
    n: u32,      // transform length
    m: u32,      // butterfly half block for NTT stages
    stride: u32, // twiddle index stride for NTT stages
    sub: u32,    // amount subtracted from word 0 by unweight (0 or 2)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> T: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> T2: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> TW: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> WIN: array<i32>;
@group(0) @binding(5) var<storage, read_write> WOUT: array<i32>;
@group(0) @binding(6) var<storage, read> WT: array<u32>;
@group(0) @binding(7) var<storage, read_write> ERR: array<atomic<u32>>;

fn fadd(a: u32, b: u32) -> u32 {
    let s = a + b;
    return select(s, s - Q, s >= Q);
}

fn fsub(a: u32, b: u32) -> u32 {
    return select(a + Q - b, a - b, a >= b);
}

// a b mod q for a, b < q from 16-bit limb products.
fn fmul(a: u32, b: u32) -> u32 {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let lo = a0 * b0;
    let mid = a1 * b0 + a0 * b1;
    let hi = a1 * b1;
    let l = lo + (mid << 16u);
    let c = select(0u, 1u, l < lo);
    let h = hi + (mid >> 16u) + c;
    var s = ((h << 1u) | (l >> 31u)) + (l & Q);
    s = (s & Q) + (s >> 31u);
    return select(s, s - Q, s >= Q);
}

fn rot(v: u32, s: u32) -> u32 {
    return ((v << s) | (v >> (31u - s))) & Q;
}

fn cadd(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(fadd(a.x, b.x), fadd(a.y, b.y));
}

fn csub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(fsub(a.x, b.x), fsub(a.y, b.y));
}

fn cmul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(fsub(fmul(a.x, b.x), fmul(a.y, b.y)), fadd(fmul(a.x, b.y), fmul(a.y, b.x)));
}

fn csqr(a: vec2<u32>) -> vec2<u32> {
    let t = fmul(a.x, a.y);
    return vec2<u32>(fsub(fmul(a.x, a.x), fmul(a.y, a.y)), fadd(t, t));
}

fn conj(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x, fsub(0u, a.y));
}

fn to_field(x: i32) -> u32 {
    return select(u32(x + i32(Q)), u32(x), x >= 0);
}

fn balanced(v: u32) -> i32 {
    return select(i32(v), i32(v) - i32(Q), v > Q / 2u);
}

fn carry_hi(v: i32, w: u32) -> i32 {
    return (v + (1i << (w - 1u))) >> w;
}

// Grid-stride loops: any dispatch size covers any element count.

// ponytail: real input (zero imaginary part) wastes half of each GF(q^2)
// element, a 2x ceiling on length and memory; upgrade with a right-angle
// packing of two words per element or a real-input transform.
@compute @workgroup_size(256)
fn weight(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        let s = (WT[i] >> 8u) & 31u;
        T[i] = vec2<u32>(rot(to_field(WIN[i]), s), 0u);
    }
}

@compute @workgroup_size(256)
fn dif(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let m = P.m;
    for (var u = g.x; u < P.n / 2u; u += nw.x * WG) {
        let j = u % m;
        let i0 = (u / m) * 2u * m + j;
        let a = T[i0];
        let b = T[i0 + m];
        T[i0] = cadd(a, b);
        T[i0 + m] = cmul(csub(a, b), TW[j * P.stride]);
    }
}

@compute @workgroup_size(256)
fn dit(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let m = P.m;
    for (var u = g.x; u < P.n / 2u; u += nw.x * WG) {
        let j = u % m;
        let i0 = (u / m) * 2u * m + j;
        let a = T[i0];
        let b = cmul(T[i0 + m], conj(TW[j * P.stride]));
        T[i0] = cadd(a, b);
        T[i0 + m] = csub(a, b);
    }
}

// Two DIF stages in one pass (opt-in, MERSENNE_FORGE_GPU_RADIX4=1): the
// stage with half block m (twiddle stride s = P.stride) then the stage with
// half block m / 2 (stride 2s), on the four elements b + j + {0, 1, 2, 3} m/2.
// The same field operations as two `dif` passes, in the same order, so the
// result is identical; only memory traffic and dispatches halve.
@compute @workgroup_size(256)
fn dif4(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let m = P.m;
    let h = m / 2u;
    let s = P.stride;
    for (var u = g.x; u < P.n / 4u; u += nw.x * WG) {
        let j = u % h;
        let i0 = (u / h) * 2u * m + j;
        let e0 = T[i0];
        let e1 = T[i0 + h];
        let e2 = T[i0 + m];
        let e3 = T[i0 + m + h];
        let a0 = cadd(e0, e2);
        let a2 = cmul(csub(e0, e2), TW[j * s]);
        let a1 = cadd(e1, e3);
        let a3 = cmul(csub(e1, e3), TW[(j + h) * s]);
        let t = TW[j * 2u * s];
        T[i0] = cadd(a0, a1);
        T[i0 + h] = cmul(csub(a0, a1), t);
        T[i0 + m] = cadd(a2, a3);
        T[i0 + m + h] = cmul(csub(a2, a3), t);
    }
}

// Two DIT stages in one pass: half block h (stride s = P.stride) then half
// block 2h (stride s / 2), on the elements b + j + {0, 1, 2, 3} h. Identical
// to two `dit` passes.
@compute @workgroup_size(256)
fn dit4(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let h = P.m;
    let s = P.stride;
    for (var u = g.x; u < P.n / 4u; u += nw.x * WG) {
        let j = u % h;
        let i0 = (u / h) * 4u * h + j;
        let ca = conj(TW[j * s]);
        let x1 = cmul(T[i0 + h], ca);
        let x3 = cmul(T[i0 + 3u * h], ca);
        let e0 = cadd(T[i0], x1);
        let e1 = csub(T[i0], x1);
        let e2 = cadd(T[i0 + 2u * h], x3);
        let e3 = csub(T[i0 + 2u * h], x3);
        let y2 = cmul(e2, conj(TW[j * (s / 2u)]));
        let y3 = cmul(e3, conj(TW[(j + h) * (s / 2u)]));
        T[i0] = cadd(e0, y2);
        T[i0 + 2u * h] = csub(e0, y2);
        T[i0 + h] = cadd(e1, y3);
        T[i0 + 3u * h] = csub(e1, y3);
    }
}

@compute @workgroup_size(256)
fn sqr(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        T[i] = csqr(T[i]);
    }
}

@compute @workgroup_size(256)
fn pmul(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        T[i] = cmul(T[i], T2[i]);
    }
}

// The inverse of a real input convolution is real. A nonzero imaginary part
// can only come from a hardware or kernel fault; it is flagged in ERR[0].
@compute @workgroup_size(256)
fn unweight(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        let t = T[i];
        if (t.y != 0u) {
            atomicOr(&ERR[0], 1u);
        }
        var z = balanced(rot(t.x, (WT[i] >> 16u) & 31u));
        if (i == 0u) {
            z -= i32(P.sub);
        }
        WOUT[i] = z;
    }
}

@compute @workgroup_size(256)
fn carry(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        let e = WT[i];
        let w = e & 31u;
        let wp = e >> 24u;
        let v = WIN[i];
        let hi = carry_hi(v, w);
        let lo = v - hi * (1i << w);
        WOUT[i] = lo + carry_hi(WIN[(i + P.n - 1u) % P.n], wp);
    }
}

// Test kernel: T2[i].x = T[i].x T[i].y mod q.
@compute @workgroup_size(256)
fn mulmod_test(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        T2[i] = vec2<u32>(fmul(T[i].x, T[i].y), 0u);
    }
}
