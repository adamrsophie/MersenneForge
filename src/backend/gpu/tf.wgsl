// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.
//
// Trial factoring of 2^p - 1: for each candidate q < 2^63 (odd), decide
// whether 2^p = 1 mod q. Montgomery arithmetic with R = 2^64 on two 32-bit
// words (CIOS), 32 x 32 bit products from 16-bit limbs. Every hit is checked
// again on the host before it is reported, so a kernel error can lose a
// factor but never report a false one.

struct Params {
    p: u32,     // exponent
    count: u32, // candidates in Q
    top: u32,   // index of the highest set bit of p
    cap: u32,   // hit slots after HITS[0]
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec2<u32>>;
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

fn ge(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y > b.y || (a.y == b.y && a.x >= b.x);
}

fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let borrow = select(0u, 1u, a.x < b.x);
    return vec2<u32>(a.x - b.x, a.y - b.y - borrow);
}

// 2 x mod q for x < q < 2^63 (2 x < 2^64 does not overflow).
fn dbl(x: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let d = vec2<u32>(x.x << 1u, (x.y << 1u) | (x.x >> 31u));
    return select(d, sub64(d, q), ge(d, q));
}

// a b R^-1 mod q (CIOS, two words); ninv = -q^-1 mod 2^32.
fn mont_mul(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>, ninv: u32) -> vec2<u32> {
    var t0 = 0u;
    var t1 = 0u;
    var t2 = 0u;
    var t3 = 0u;
    for (var i = 0u; i < 2u; i++) {
        let bi = select(b.x, b.y, i == 1u);
        var r = mac(t0, a.x, bi, 0u);
        t0 = r.x;
        r = mac(t1, a.y, bi, r.y);
        t1 = r.x;
        let s = t2 + r.y;
        t3 = select(0u, 1u, s < t2);
        t2 = s;
        let m = t0 * ninv;
        r = mac(t0, m, q.x, 0u);
        r = mac(t1, m, q.y, r.y);
        t0 = r.x;
        let s2 = t2 + r.y;
        t1 = s2;
        t2 = t3 + select(0u, 1u, s2 < t2);
    }
    let t = vec2<u32>(t0, t1);
    if (t2 != 0u || ge(t, q)) {
        return sub64(t, q);
    }
    return t;
}

@compute @workgroup_size(256)
fn tf(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.count; i += nw.x * WG) {
        let q = Q[i];
        // -q^-1 mod 2^32 by Newton's iteration (q odd: q q = 1 mod 8).
        var x = q.x;
        for (var j = 0u; j < 4u; j++) {
            x = x * (2u - q.x * x);
        }
        let ninv = 0u - x;
        // R mod q = 2^64 mod q by 64 doublings of 1.
        var one = vec2<u32>(1u, 0u);
        for (var j = 0u; j < 64u; j++) {
            one = dbl(one, q);
        }
        // Montgomery form throughout: v = 2^e R mod q, from the top bit.
        var v = one;
        for (var b = i32(P.top); b >= 0; b--) {
            v = mont_mul(v, v, q, ninv);
            if (((P.p >> u32(b)) & 1u) == 1u) {
                v = dbl(v, q);
            }
        }
        if (v.x == one.x && v.y == one.y) {
            let slot = atomicAdd(&HITS[0], 1u);
            if (slot < P.cap) {
                atomicStore(&HITS[slot + 1u], i);
            }
        }
    }
}
