// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Shifted residues, as Prime95's shift count: every value v is held by the
//! wrapped backend as v * 2^s mod 2^p - 1 (a rotation of its p bits), with
//! s tracked per register. Squaring doubles s and multiplication adds the
//! two counts, so the device works on rotated data while everything
//! outside sees the true values: `export` rotates back and `import`
//! rotates by the starting shift. A second run with another shift puts
//! every bit in another place, so a systematic arithmetic error cannot
//! give both runs the same residue.
//!
//! LL works too: after a squaring the register holds v^2 * 2^(2s), so
//! the `- 2` becomes `- 2^(2s + 1)`, which is applied to the rotated value
//! between squarings.

use super::{Backend, BackendError, BackendIdentity, BackendResult, Reg};
use crate::residue::Residue;
use num_bigint::BigUint;
use std::collections::HashMap;

pub struct Shifted<'a, B: Backend + ?Sized> {
    inner: &'a mut B,
    start: u32,
    shift: HashMap<Reg, u32>,
}

impl<'a, B: Backend + ?Sized> Shifted<'a, B> {
    /// `start` must be below p.
    pub fn new(inner: &'a mut B, start: u32) -> BackendResult<Self> {
        if start >= inner.exponent() {
            return Err(BackendError::Unsupported(format!(
                "shift count {start} must be below p = {}",
                inner.exponent()
            )));
        }
        Ok(Shifted {
            inner,
            start,
            shift: HashMap::new(),
        })
    }

    fn of(&self, r: Reg) -> BackendResult<u32> {
        self.shift
            .get(&r)
            .copied()
            .ok_or(BackendError::InvalidHandle(r.0))
    }
}

/// x * 2^s mod 2^p - 1, for s < p: a left rotation of the p bits.
pub fn rotate_left(x: &Residue, s: u32) -> Residue {
    let p = x.exponent();
    if s == 0 {
        return x.clone();
    }
    let v = x.to_biguint();
    let mask = (BigUint::from(1u32) << p) - 1u32;
    let r = ((&v << s) & &mask) | (v >> (p - s));
    Residue::from_biguint(p, &r)
}

impl<B: Backend + ?Sized> Backend for Shifted<'_, B> {
    fn exponent(&self) -> u32 {
        self.inner.exponent()
    }

    fn identity(&self) -> BackendIdentity {
        self.inner.identity()
    }

    fn alloc(&mut self) -> BackendResult<Reg> {
        let r = self.inner.alloc()?;
        self.shift.insert(r, 0);
        Ok(r)
    }

    fn free(&mut self, r: Reg) -> BackendResult<()> {
        self.shift.remove(&r);
        self.inner.free(r)
    }

    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        self.of(r)?;
        self.inner.import(r, &rotate_left(x, self.start))?;
        self.shift.insert(r, self.start);
        Ok(())
    }

    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        let s = self.of(r)?;
        let x = self.inner.export(r)?;
        let p = x.exponent();
        Ok(rotate_left(&x, (p - s) % p))
    }

    fn copy(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        let s = self.of(src)?;
        self.of(dst)?;
        self.inner.copy(dst, src)?;
        self.shift.insert(dst, s);
        Ok(())
    }

    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        let s = self.of(r)?;
        self.inner.square(r, n)?;
        // s 2^n mod p.
        let p = u64::from(self.exponent());
        let two_n = BigUint::from(2u32).modpow(&BigUint::from(n), &BigUint::from(p));
        let t = (u64::from(s) * two_n.iter_u64_digits().next().unwrap_or(0)) % p;
        self.shift.insert(r, t as u32);
        Ok(())
    }

    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        let p = self.exponent();
        let modulus = crate::residue::modulus(p);
        for _ in 0..n {
            let s = self.of(r)?;
            if s == 0 {
                // Unrotated (a run started without a shift reached here
                // only through a zero register): the plain step is exact.
                self.inner.ll_step(r, 1)?;
                continue;
            }
            self.inner.square(r, 1)?;
            let t = ((u64::from(s) * 2) % u64::from(p)) as u32;
            // ponytail: one export and import per iteration, a host round
            // trip on the device backends; a device side subtraction of a
            // single power of two would remove it.
            let k = (t + 1) % p;
            let x = self.inner.export(r)?.to_biguint();
            let c = BigUint::from(1u32) << k;
            let d = if x >= c { x - c } else { x + &modulus - c };
            self.inner.import(r, &Residue::from_biguint(p, &d))?;
            self.shift.insert(r, t);
        }
        Ok(())
    }

    fn mul(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        let (a, b) = (self.of(dst)?, self.of(src)?);
        self.inner.mul(dst, src)?;
        let p = u64::from(self.exponent());
        self.shift
            .insert(dst, ((u64::from(a) + u64::from(b)) % p) as u32);
        Ok(())
    }

    fn sub(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        let (a, b) = (self.of(dst)?, self.of(src)?);
        if dst != src && a == b {
            // Both rotated alike: the difference is rotated alike too.
            return self.inner.sub(dst, src);
        }
        if dst == src {
            return Err(BackendError::Unsupported(
                "sub needs distinct handles".into(),
            ));
        }
        let p = self.exponent();
        let n = crate::residue::modulus(p);
        let x = self.export(dst)?.to_biguint();
        let y = self.export(src)?.to_biguint();
        let d = if x >= y { x - y } else { x + &n - y };
        self.import(dst, &Residue::from_biguint(p, &d))
    }

    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        if self.of(a)? == self.of(b)? {
            return self.inner.equal(a, b);
        }
        Ok(self.export(a)? == self.export(b)?)
    }

    fn sync(&mut self) -> BackendResult<()> {
        self.inner.sync()
    }

    fn diagnostics(&mut self) -> BackendResult<Vec<(String, String)>> {
        self.inner.diagnostics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;

    /// Squarings, products, copies, and comparisons through a shift give
    /// the same values as without one, while the stored data differs.
    #[test]
    fn shifted_arithmetic_matches_plain_arithmetic() {
        let p = 127;
        let mut plain = CpuBackend::new(p).unwrap();
        let mut base = CpuBackend::new(p).unwrap();
        let mut sh = Shifted::new(&mut base, 45).unwrap();
        let x = Residue::from_u64(p, 3);
        let run = |be: &mut dyn Backend| {
            let (a, b) = (be.alloc().unwrap(), be.alloc().unwrap());
            be.import(a, &x).unwrap();
            be.copy(b, a).unwrap();
            be.square(a, 77).unwrap();
            be.mul(b, a).unwrap();
            be.square(b, 3).unwrap();
            let eq = be.equal(a, b).unwrap();
            be.copy(a, b).unwrap();
            be.square(a, 0).unwrap();
            let eq2 = be.equal(a, b).unwrap();
            be.sub(a, b).unwrap();
            (be.export(b).unwrap(), be.export(a).unwrap(), eq, eq2)
        };
        let want = run(&mut plain);
        let got = run(&mut sh);
        assert_eq!(got, want);
        assert!(want.3 && !want.2);
        assert!(want.1.is_zero());
        assert!(Shifted::new(&mut plain, 127).is_err());
        // LL steps through every starting shift give the plain residues.
        for start in [0, 1, 63, 126] {
            let mut base = CpuBackend::new(p).unwrap();
            let mut sh = Shifted::new(&mut base, start).unwrap();
            let mut plain = CpuBackend::new(p).unwrap();
            let four = Residue::from_u64(p, 4);
            let (a, b) = (sh.alloc().unwrap(), plain.alloc().unwrap());
            sh.import(a, &four).unwrap();
            plain.import(b, &four).unwrap();
            sh.ll_step(a, 125).unwrap();
            plain.ll_step(b, 125).unwrap();
            assert_eq!(sh.export(a).unwrap(), plain.export(b).unwrap());
            assert!(plain.export(b).unwrap().is_zero(), "M127 is prime");
        }
        let y = Residue::from_u64(p, 0x1234_5678_9abc);
        assert_eq!(rotate_left(&rotate_left(&y, 100), 27), y);
    }
}
