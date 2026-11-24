// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.
//
// Exact NTT arithmetic modulo 2^p - 1 over GF(q^2), q = 2^61 - 1: the wide
// field of wide.rs, which holds the model these kernels are compared
// against, the capacity bound, and the digit split. Same bindings and pass
// structure as kernels.wgsl, without the packed kernels, and with `clear`
// before `unweight`, which adds into its output. A field element is two u32
// words (low, high) of a value below q; a GF(q^2) element is four words
// (re low, re high, im low, im high).

const WG: u32 = 256u;
const QF: vec2<u32> = vec2<u32>(0xffffffffu, 0x1fffffffu);
const QH: u32 = 0x1fffffffu;
const HALF_Q: vec2<u32> = vec2<u32>(0xffffffffu, 0x0fffffffu);

struct Params {
    n: u32,      // transform length
    m: u32,      // butterfly half block for NTT stages; digit count D for unweight
    stride: u32, // twiddle index stride for NTT stages
    sub: u32,    // amount subtracted from word 0 by unweight (0 or 2)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> T: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> T2: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> TW: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> WIN: array<i32>;
@group(0) @binding(5) var<storage, read_write> WOUT: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read> WT: array<u32>;
@group(0) @binding(7) var<storage, read_write> ERR: array<atomic<u32>>;

// 64-bit helpers on (low, high) pairs.

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn ge64(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y > b.y || (a.y == b.y && a.x >= b.x);
}

// Shifts by 0 to 63 (a WGSL shift by 32 or more would wrap).
fn shl64(v: vec2<u32>, s: u32) -> vec2<u32> {
    if (s == 0u) {
        return v;
    }
    if (s >= 32u) {
        return vec2<u32>(0u, v.x << (s - 32u));
    }
    return vec2<u32>(v.x << s, (v.y << s) | (v.x >> (32u - s)));
}

fn shr64(v: vec2<u32>, s: u32) -> vec2<u32> {
    if (s == 0u) {
        return v;
    }
    if (s >= 32u) {
        return vec2<u32>(v.y >> (s - 32u), 0u);
    }
    return vec2<u32>((v.x >> s) | (v.y << (32u - s)), v.y >> s);
}

// Arithmetic right shift of a two's complement value by 1 to 31.
fn sar64(v: vec2<u32>, s: u32) -> vec2<u32> {
    return vec2<u32>((v.x >> s) | (v.y << (32u - s)), u32(i32(v.y) >> s));
}

fn sext(x: i32) -> vec2<u32> {
    return vec2<u32>(u32(x), select(0u, 0xffffffffu, x < 0));
}

// a b as (low, high), from 16-bit limb products.
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

// GF(q).

fn fadd(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let s = add64(a, b);
    return select(s, sub64(s, QF), ge64(s, QF));
}

fn fsub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return select(sub64(add64(a, QF), b), sub64(a, b), ge64(a, b));
}

// a b mod q: the 122-bit product in four words, then low 61 bits plus the
// rest, folded once more.
fn fmul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul32(a.x, b.x);
    let lh = mul32(a.x, b.y);
    let hl = mul32(a.y, b.x);
    let hh = mul32(a.y, b.y);
    let s1 = ll.y + lh.x;
    var c1 = select(0u, 1u, s1 < ll.y);
    let p1 = s1 + hl.x;
    c1 += select(0u, 1u, p1 < s1);
    let s2 = lh.y + hl.y;
    var c2 = select(0u, 1u, s2 < lh.y);
    let s3 = s2 + hh.x;
    c2 += select(0u, 1u, s3 < s2);
    let p2 = s3 + c1;
    c2 += select(0u, 1u, p2 < s3);
    let p3 = hh.y + c2;
    let lo = vec2<u32>(ll.x, p1 & QH);
    let hi = vec2<u32>((p1 >> 29u) | (p2 << 3u), (p2 >> 29u) | (p3 << 3u));
    var s = add64(lo, hi);
    s = add64(vec2<u32>(s.x, s.y & QH), vec2<u32>(s.y >> 29u, 0u));
    return select(s, sub64(s, QF), ge64(s, QF));
}

// v 2^s mod q for v < q, s < 61: a 61-bit rotation.
fn rot(v: vec2<u32>, s: u32) -> vec2<u32> {
    let a = shl64(v, s);
    let b = shr64(v, 61u - s);
    return vec2<u32>(a.x | b.x, (a.y | b.y) & QH);
}

fn to_field(x: i32) -> vec2<u32> {
    if (x >= 0) {
        return vec2<u32>(u32(x), 0u);
    }
    return fsub(vec2<u32>(0u, 0u), vec2<u32>(u32(-x), 0u));
}

// Canonical v to the balanced integer, two's complement.
fn balanced(v: vec2<u32>) -> vec2<u32> {
    return select(v, sub64(v, QF), !ge64(HALF_Q, v));
}

// GF(q^2).

fn mk(re: vec2<u32>, im: vec2<u32>) -> vec4<u32> {
    return vec4<u32>(re.x, re.y, im.x, im.y);
}

fn cadd(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return mk(fadd(a.xy, b.xy), fadd(a.zw, b.zw));
}

fn csub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return mk(fsub(a.xy, b.xy), fsub(a.zw, b.zw));
}

fn cmul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return mk(
        fsub(fmul(a.xy, b.xy), fmul(a.zw, b.zw)),
        fadd(fmul(a.xy, b.zw), fmul(a.zw, b.xy)),
    );
}

fn csqr(a: vec4<u32>) -> vec4<u32> {
    let t = fmul(a.xy, a.zw);
    return mk(fsub(fmul(a.xy, a.xy), fmul(a.zw, a.zw)), fadd(t, t));
}

fn conj(a: vec4<u32>) -> vec4<u32> {
    return mk(a.xy, fsub(vec2<u32>(0u, 0u), a.zw));
}

fn carry_hi(v: i32, w: u32) -> i32 {
    return (v + (1i << (w - 1u))) >> w;
}

@compute @workgroup_size(256)
fn weight(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        let s = (WT[i] >> 8u) & 0xffu;
        T[i] = mk(rot(to_field(WIN[i]), s), vec2<u32>(0u, 0u));
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

// The fused radix 4 stages of kernels.wgsl, in this field.
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

// Zero the words the unweight pass adds into.
@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        atomicStore(&WOUT[i], 0i);
    }
}

// Unweight, scale, and split coefficient i into P.m balanced digits added
// into words i, i + 1, ... (see wide.rs). A nonzero imaginary part can only
// come from a fault; it is flagged in ERR[0].
@compute @workgroup_size(256)
fn unweight(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let d = P.m;
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        let t = T[i];
        if (t.z != 0u || t.w != 0u) {
            atomicOr(&ERR[0], 1u);
        }
        var v = balanced(rot(t.xy, (WT[i] >> 16u) & 0xffu));
        for (var j = 0u; j + 1u < d; j++) {
            let idx = (i + j) % P.n;
            let w = WT[idx] & 0xffu;
            let half = 1u << (w - 1u);
            let lo = i32((v.x + half) & ((1u << w) - 1u)) - i32(half);
            atomicAdd(&WOUT[idx], lo);
            v = sar64(sub64(v, sext(lo)), w);
        }
        atomicAdd(&WOUT[(i + d - 1u) % P.n], i32(v.x));
        if (i == 0u) {
            atomicAdd(&WOUT[0], -i32(P.sub));
        }
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
        atomicStore(&WOUT[i], lo + carry_hi(WIN[(i + P.n - 1u) % P.n], wp));
    }
}

// Test kernel: T2[i].re = T[i].re T[i].im mod q.
@compute @workgroup_size(256)
fn mulmod_test(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = g.x; i < P.n; i += nw.x * WG) {
        T2[i] = mk(fmul(T[i].xy, T[i].zw), vec2<u32>(0u, 0u));
    }
}
