// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Exact CPU backend.
//!
//! Multiplication is `num_bigint` multiplication. Reduction modulo
//! N = 2^p - 1 is Mersenne folding: since 2^p = 1 mod N, a value
//! v = hi * 2^p + lo reduces to hi + lo; repeat until v < 2^p, then map the
//! value N to zero. This deliberately differs from the oracle, which reduces
//! with generic division, so differential tests compare two reduction
//! methods. Both share `num_bigint` multiplication, a stated residual risk.

use std::collections::HashMap;

use num_bigint::BigUint;
use num_traits::Zero;

use super::{Backend, BackendError, BackendIdentity, BackendResult, Reg};
use crate::residue::{modulus, Residue};

/// Largest exponent the CPU backend accepts (residues of 512 MiB each).
pub const MAX_EXPONENT: u32 = u32::MAX;

/// Inclusive exponent range of this backend.
pub fn supported_exponents() -> (u32, u32) {
    (2, MAX_EXPONENT)
}

/// A corruption applied directly to a resident residue, for protocol tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corruption {
    FlipBit(u32),
    Zero,
}

pub struct CpuBackend {
    p: u32,
    n: BigUint,
    regs: HashMap<u32, BigUint>,
    next: u32,
}

impl CpuBackend {
    pub fn new(p: u32) -> BackendResult<CpuBackend> {
        if p < 2 {
            return Err(BackendError::Unsupported(format!(
                "exponent {p} defines no modulus"
            )));
        }
        Ok(CpuBackend {
            p,
            n: modulus(p),
            regs: HashMap::new(),
            next: 0,
        })
    }

    fn get(&self, r: Reg) -> BackendResult<&BigUint> {
        self.regs.get(&r.0).ok_or(BackendError::InvalidHandle(r.0))
    }

    fn get_mut(&mut self, r: Reg) -> BackendResult<&mut BigUint> {
        self.regs
            .get_mut(&r.0)
            .ok_or(BackendError::InvalidHandle(r.0))
    }

    /// Mersenne folding to the canonical range [0, N).
    fn fold(&self, mut v: BigUint) -> BigUint {
        while v.bits() > u64::from(self.p) {
            let hi = &v >> self.p;
            let lo = v & &self.n;
            v = hi + lo;
        }
        if v == self.n {
            v.set_zero();
        }
        v
    }

    /// Test hook: corrupt a resident residue in place.
    pub fn corrupt(&mut self, r: Reg, c: Corruption) -> BackendResult<()> {
        let p = self.p;
        let v = self.get_mut(r)?;
        match c {
            Corruption::Zero => v.set_zero(),
            Corruption::FlipBit(b) => {
                let b = u64::from(b % p);
                let bit = v.bit(b);
                v.set_bit(b, !bit);
            }
        }
        let taken = std::mem::take(self.get_mut(r)?);
        let folded = self.fold(taken);
        *self.get_mut(r)? = folded;
        Ok(())
    }
}

impl Backend for CpuBackend {
    fn exponent(&self) -> u32 {
        self.p
    }

    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            kind: "cpu".into(),
            build: format!("mersenne-forge {}", env!("CARGO_PKG_VERSION")),
            plan: "bigint-mersenne-fold".into(),
            device: String::new(),
        }
    }

    fn alloc(&mut self) -> BackendResult<Reg> {
        let id = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| BackendError::Allocation("handle space exhausted".into()))?;
        self.regs.insert(id, BigUint::zero());
        Ok(Reg(id))
    }

    fn free(&mut self, r: Reg) -> BackendResult<()> {
        self.regs
            .remove(&r.0)
            .map(|_| ())
            .ok_or(BackendError::InvalidHandle(r.0))
    }

    fn import(&mut self, r: Reg, x: &Residue) -> BackendResult<()> {
        if x.exponent() != self.p {
            return Err(BackendError::Unsupported(format!(
                "residue for exponent {} imported into backend for {}",
                x.exponent(),
                self.p
            )));
        }
        *self.get_mut(r)? = x.to_biguint();
        Ok(())
    }

    fn export(&mut self, r: Reg) -> BackendResult<Residue> {
        let p = self.p;
        let mut limbs = self.get(r)?.to_u32_digits();
        limbs.resize(crate::residue::limb_count(p), 0);
        Residue::from_limbs(p, limbs).map_err(|e| BackendError::Fault(format!("export: {e}")))
    }

    fn copy(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        let v = self.get(src)?.clone();
        *self.get_mut(dst)? = v;
        Ok(())
    }

    fn square(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        let mut v = std::mem::take(self.get_mut(r)?);
        for _ in 0..n {
            v = self.fold(&v * &v);
        }
        *self.get_mut(r)? = v;
        Ok(())
    }

    fn ll_step(&mut self, r: Reg, n: u64) -> BackendResult<()> {
        let mut v = std::mem::take(self.get_mut(r)?);
        let two = BigUint::from(2u32);
        for _ in 0..n {
            // s^2 - 2 = s^2 + (N - 2) mod N, which stays nonnegative.
            v = self.fold(&v * &v + &self.n - &two);
        }
        *self.get_mut(r)? = v;
        Ok(())
    }

    fn mul(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        if dst == src {
            return Err(BackendError::Unsupported(
                "mul needs distinct handles".into(),
            ));
        }
        let b = self.get(src)?.clone();
        let a = std::mem::take(self.get_mut(dst)?);
        let v = self.fold(a * b);
        *self.get_mut(dst)? = v;
        Ok(())
    }

    fn equal(&mut self, a: Reg, b: Reg) -> BackendResult<bool> {
        Ok(self.get(a)? == self.get(b)?)
    }

    fn sync(&mut self) -> BackendResult<()> {
        Ok(())
    }
}
