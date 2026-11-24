// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Polynomial arithmetic over Z/(2^p - 1) on the host, for the polynomial
//! P-1 stage 2 (Montgomery and Silverman's "FFT continuation"): evaluate
//! F(X) = prod (X - b_j) at many points at once with a subproduct tree and
//! a remainder tree, in O(M(k) log k) ring operations per k points instead
//! of k^2.
//!
//! Coefficients are little endian `BigUint`s below N = 2^p - 1. Products
//! use Kronecker substitution: every coefficient gets a slot of whole
//! 32-bit digits wide enough for a full convolution sum, the two packed
//! integers are multiplied once, and the slots are read back and reduced.
//! ponytail: the big integer product is num-bigint's (Karatsuba and Toom 3,
//! no FFT) and every coefficient is resident on the host, so this suits
//! exponents of a few million bits at most; a device polynomial product
//! would lift both limits.

use num_bigint::BigUint;
use num_traits::Zero;

/// Z/(2^p - 1).
pub struct Ring {
    p: u32,
    n: BigUint,
}

pub type Poly = Vec<BigUint>;

impl Ring {
    pub fn new(p: u32) -> Ring {
        Ring {
            p,
            n: crate::residue::modulus(p),
        }
    }

    /// x mod 2^p - 1 by folding the high bits onto the low ones.
    pub fn reduce(&self, mut x: BigUint) -> BigUint {
        while x.bits() > u64::from(self.p) {
            x = (&x & &self.n) + (x >> self.p);
        }
        if x == self.n {
            BigUint::zero()
        } else {
            x
        }
    }

    pub fn mul_elt(&self, a: &BigUint, b: &BigUint) -> BigUint {
        self.reduce(a * b)
    }

    fn sub_elt(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a >= b {
            a - b
        } else {
            a + &self.n - b
        }
    }

    /// a * b.
    pub fn mul(&self, a: &[BigUint], b: &[BigUint]) -> Poly {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let terms = a.len().min(b.len()) as u64;
        // Each output coefficient is a sum of at most `terms` products
        // below N^2.
        let bits = 2 * u64::from(self.p) + (64 - terms.leading_zeros() as u64) + 1;
        let w = bits.div_ceil(32) as usize;
        let pack = |v: &[BigUint]| {
            let mut d = vec![0u32; v.len() * w];
            for (i, c) in v.iter().enumerate() {
                for (k, x) in c.to_u32_digits().into_iter().enumerate() {
                    d[i * w + k] = x;
                }
            }
            BigUint::new(d)
        };
        let prod = (pack(a) * pack(b)).to_u32_digits();
        let len = a.len() + b.len() - 1;
        (0..len)
            .map(|i| {
                let lo = (i * w).min(prod.len());
                let hi = ((i + 1) * w).min(prod.len());
                self.reduce(BigUint::from_slice(&prod[lo..hi]))
            })
            .collect()
    }

    /// 1 / f mod X^l, for f with constant term 1 (Newton iteration).
    fn inverse(&self, f: &[BigUint], l: usize) -> Poly {
        let mut g: Poly = vec![BigUint::from(1u32)];
        let mut have = 1;
        while have < l {
            have = (2 * have).min(l);
            let mut fg = self.mul(&f[..f.len().min(have)], &g);
            fg.truncate(have);
            // e = 2 - f g.
            let mut e: Poly = fg
                .iter()
                .map(|c| self.sub_elt(&BigUint::zero(), c))
                .collect();
            e[0] = self.reduce(&e[0] + 2u32);
            g = self.mul(&g, &e);
            g.truncate(have);
        }
        g
    }

    /// a mod b, for monic b of degree at least 1.
    pub fn rem(&self, a: &[BigUint], b: &[BigUint]) -> Poly {
        let db = b.len() - 1;
        if a.len() <= db {
            return a.to_vec();
        }
        let l = a.len() - db;
        let ra: Poly = a.iter().rev().take(l).cloned().collect();
        let rb: Poly = b.iter().rev().cloned().collect();
        let mut q = self.mul(&ra, &self.inverse(&rb, l));
        q.resize(l, BigUint::zero());
        q.reverse();
        let qb = self.mul(&q, b);
        (0..db).map(|i| self.sub_elt(&a[i], &qb[i])).collect()
    }

    /// The subproduct tree of (X - x_i): level 0 holds the linear factors,
    /// the last level the whole product. Every node is monic.
    pub fn tree(&self, xs: &[BigUint]) -> Vec<Vec<Poly>> {
        let one = BigUint::from(1u32);
        let mut levels: Vec<Vec<Poly>> = vec![xs
            .iter()
            .map(|x| vec![self.sub_elt(&BigUint::zero(), x), one.clone()])
            .collect()];
        while levels.last().expect("one level").len() > 1 {
            let up = levels
                .last()
                .expect("one level")
                .chunks(2)
                .map(|c| match c {
                    [a, b] => self.mul(a, b),
                    [a] => a.clone(),
                    _ => unreachable!(),
                })
                .collect();
            levels.push(up);
        }
        levels
    }

    /// The monic product of (X - x_i).
    pub fn from_roots(&self, xs: &[BigUint]) -> Poly {
        self.tree(xs).pop().expect("one level").remove(0)
    }

    /// f(x_i) for every point, through the remainder tree.
    pub fn eval(&self, f: &[BigUint], xs: &[BigUint]) -> Vec<BigUint> {
        if xs.is_empty() {
            return Vec::new();
        }
        let tree = self.tree(xs);
        let mut rems = vec![self.rem(f, &tree[tree.len() - 1][0])];
        for level in tree.iter().rev().skip(1) {
            rems = level
                .iter()
                .enumerate()
                .map(|(i, node)| self.rem(&rems[i / 2], node))
                .collect();
        }
        rems.into_iter()
            .map(|r| r.into_iter().next().unwrap_or_default())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn horner(r: &Ring, f: &[BigUint], x: &BigUint) -> BigUint {
        f.iter()
            .rev()
            .fold(BigUint::zero(), |acc, c| r.reduce(acc * x + c))
    }

    /// The remainder tree gives the values Horner's rule gives, for root
    /// counts and point counts that are not powers of two.
    #[test]
    fn tree_evaluation_matches_horner() {
        let p = 127;
        let r = Ring::new(p);
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            r.reduce(BigUint::from(s) * BigUint::from(s.rotate_left(17)) * (s >> 3))
        };
        for (k, m) in [(1, 1), (5, 3), (24, 24), (13, 40), (33, 7)] {
            let roots: Vec<BigUint> = (0..k).map(|_| next()).collect();
            let xs: Vec<BigUint> = (0..m).map(|_| next()).collect();
            let f = r.from_roots(&roots);
            assert_eq!(f.len(), k + 1);
            let got = r.eval(&f, &xs);
            for (x, v) in xs.iter().zip(&got) {
                assert_eq!(*v, horner(&r, &f, x), "k={k} m={m}");
                let direct = roots
                    .iter()
                    .fold(BigUint::from(1u32), |a, b| r.mul_elt(&a, &r.sub_elt(x, b)));
                assert_eq!(*v, direct);
            }
        }
        // A root is a zero.
        let f = r.from_roots(&[BigUint::from(5u32), BigUint::from(9u32)]);
        assert!(r.eval(&f, &[BigUint::from(9u32)])[0].is_zero());
    }
}
