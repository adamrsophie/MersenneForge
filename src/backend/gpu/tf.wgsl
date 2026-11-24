// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.
//
// Trial factoring of 2^p - 1: for each candidate q < 2^95 (odd), decide
// whether 2^p = 1 mod q. Montgomery arithmetic with R = 2^96 on three
// 32-bit words (CIOS), 32 x 32 bit products from 16-bit limbs. The 2^95
// bound is mfaktc's and mfakto's (bit_max 95). Every hit is checked
// again on the host before it is reported, so a kernel error can lose a
// factor but never report a false one.

struct Params {
    p: u32,     // exponent
    count: u32, // candidates in Q
    top: u32,   // index of the highest set bit of p
    cap: u32,   // hit slots after HITS[0]
}

@group(0) @binding(0) var<uniform> P: Params;
// Candidates as three words each, low word first.
@group(0) @binding(1) var<storage, read> Q: array<u32>;
@group(0) @binding(2) var<storage, read_write> HITS: array<atomic<u32>>;

const WG: u32 = 256u;

// a b as (low word, high word).
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let lo = a0 * b0;
    let m1 = a1 * b0;
    let m2 = a0 * b1;
    let mid = m1 + m2;
    let cmid = select(0u, 1u, mid < m1);
    let l = lo + (mid << 16u);
    let c = select(0u, 1u, l < lo);
    let h = a1 * b1 + (mid >> 16u) + (cmid << 16u) + c;
    return vec2<u32>(l, h);
}

// t + a b + c as (low word, carry word); fits since
// (2^32 - 1)^2 + 2 (2^32 - 1) = 2^64 - 1.
fn mac(t: u32, a: u32, b: u32, c: u32) -> vec2<u32> {
    let pr = mul32(a, b);
    let s = pr.x + t;
    var h = pr.y + select(0u, 1u, s < pr.x);
    let s2 = s + c;
    h += select(0u, 1u, s2 < s);
    return vec2<u32>(s2, h);
}

fn ge(a: vec3<u32>, b: vec3<u32>) -> bool {
    if (a.z != b.z) {
        return a.z > b.z;
    }
    if (a.y != b.y) {
        return a.y > b.y;
    }
    return a.x >= b.x;
}

fn sub96(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
    let x = a.x - b.x;
    let b0 = select(0u, 1u, a.x < b.x);
    let y0 = a.y - b.y;
    let y = y0 - b0;
    let b1 = select(0u, 1u, a.y < b.y || y0 < b0);
    return vec3<u32>(x, y, a.z - b.z - b1);
}

// 2 x mod q for x < q < 2^95 (2 x < 2^96 does not overflow).
fn dbl(x: vec3<u32>, q: vec3<u32>) -> vec3<u32> {
    let d = vec3<u32>(x.x << 1u, (x.y << 1u) | (x.x >> 31u), (x.z << 1u) | (x.y >> 31u));
    return select(d, sub96(d, q), ge(d, q));
}

// a b R^-1 mod q (CIOS, three words); ninv = -q^-1 mod 2^32.
fn mont_mul(a: vec3<u32>, b: vec3<u32>, q: vec3<u32>, ninv: u32) -> vec3<u32> {
    var t = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
    for (var i = 0u; i < 3u; i++) {
        let bi = b[i];
        var c = 0u;
        for (var j = 0u; j < 3u; j++) {
            let r = mac(t[j], a[j], bi, c);
            t[j] = r.x;
            c = r.y;
        }
        let s = t[3] + c;
        t[4] = select(0u, 1u, s < c);
        t[3] = s;
        let m = t[0] * ninv;
        var r = mac(t[0], m, q.x, 0u);
        c = r.y;
        for (var j = 1u; j < 3u; j++) {
            r = mac(t[j], m, q[j], c);
            t[j - 1u] = r.x;
            c = r.y;
        }
        let s2 = t[3] + c;
        t[2] = s2;
        t[3] = t[4] + select(0u, 1u, s2 < c);
    }
    let v = vec3<u32>(t[0], t[1], t[2]);
    if (t[3] != 0u || ge(v, q)) {
        return sub96(v, q);
    }
    return v;
}

// Whether 2^p = 1 mod q, for odd q < 2^95; top is the highest set bit of p.
fn pow2_is_one(q: vec3<u32>, p: u32, top: u32) -> bool {
    // -q^-1 mod 2^32 by Newton's iteration (q odd: q q = 1 mod 8).
    var x = q.x;
    for (var j = 0u; j < 4u; j++) {
        x = x * (2u - q.x * x);
    }
    let ninv = 0u - x;
    // R mod q = 2^96 mod q by 96 doublings of 1.
    var one = vec3<u32>(1u, 0u, 0u);
    for (var j = 0u; j < 96u; j++) {
        one = dbl(one, q);
    }
    // Montgomery form throughout: v = 2^e R mod q, from the top bit.
    var v = one;
    for (var b = i32(top); b >= 0; b--) {
        v = mont_mul(v, v, q, ninv);
        if (((p >> u32(b)) & 1u) == 1u) {
            v = dbl(v, q);
        }
    }
    return all(v == one);
}

@compute @workgroup_size(256)
fn tf(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.count; i += nw.x * WG) {
        let q = vec3<u32>(Q[3u * i], Q[3u * i + 1u], Q[3u * i + 2u]);
        if (pow2_is_one(q, P.p, P.top)) {
            let slot = atomicAdd(&HITS[0], 1u);
            if (slot < P.cap) {
                atomicStore(&HITS[slot + 1u], i);
            }
        }
    }
}

// Sieving on the device. One segment is the k values k0 + i, i < len; a
// candidate is q = 2 k p + 1. `strike` marks every i with s | q for each
// sieve prime s (from its first such i), and `tf_seg` tests the unmarked
// i whose q is +-1 mod 8, recording i for each hit. The host only sends
// the roots and checks every hit again. Separate bindings (3 to 6) so the
// segment entry points share this module with `tf`.
struct SegParams {
    p: u32,
    top: u32,
    len: u32,    // k values in the segment
    nroots: u32, // sieve primes
    k0x: u32,    // first k, low word first
    k0y: u32,
    k0z: u32,
    cap: u32,    // hit slots after SHITS[0]
}

@group(0) @binding(3) var<uniform> S: SegParams;
// (s, first index) pairs.
@group(0) @binding(4) var<storage, read> ROOTS: array<u32>;
// One bit per k: set when sieved out. Zeroed by the host.
@group(0) @binding(5) var<storage, read_write> BITS: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> SHITS: array<atomic<u32>>;

// ponytail: one thread per sieve prime, so the threads for 3, 5, 7 strike
// far more than the rest; split small primes over threads if it shows up.
@compute @workgroup_size(256)
fn strike(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var r = g.x; r < S.nroots; r += nw.x * WG) {
        let s = ROOTS[2u * r];
        for (var i = ROOTS[2u * r + 1u]; i < S.len; i += s) {
            atomicOr(&BITS[i >> 5u], 1u << (i & 31u));
        }
    }
}

@compute @workgroup_size(256)
fn tf_seg(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < S.len; i += nw.x * WG) {
        if ((atomicLoad(&BITS[i >> 5u]) & (1u << (i & 31u))) != 0u) {
            continue;
        }
        // k = k0 + i, then k p in three words (the host keeps q < 2^95).
        let kx = S.k0x + i;
        let c0 = select(0u, 1u, kx < i);
        let ky = S.k0y + c0;
        let c1 = select(0u, 1u, ky < c0);
        let kz = S.k0z + c1;
        let a = mul32(kx, S.p);
        let b = mul32(ky, S.p);
        let y = a.y + b.x;
        let cy = select(0u, 1u, y < a.y);
        let z = kz * S.p + b.y + cy;
        let q = vec3<u32>((a.x << 1u) | 1u, (y << 1u) | (a.x >> 31u), (z << 1u) | (y >> 31u));
        let m = q.x & 7u;
        if (m != 1u && m != 7u) {
            continue;
        }
        if (pow2_is_one(q, S.p, S.top)) {
            let slot = atomicAdd(&SHITS[0], 1u);
            if (slot < S.cap) {
                atomicStore(&SHITS[slot + 1u], i);
            }
        }
    }
}
