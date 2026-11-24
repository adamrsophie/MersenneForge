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

use super::{
    new_handle, Backend, BackendError, BackendIdentity, BackendResult, MemoryEstimate, Reg,
};
use crate::residue::{modulus, Residue};

/// Largest exponent the CPU backend accepts (residues of 512 MiB each).
pub const MAX_EXPONENT: u32 = u32::MAX;

/// Inclusive exponent range of this backend.
pub fn supported_exponents() -> (u32, u32) {
    (2, MAX_EXPONENT)
}

const PLAN: &str = "bigint-mersenne-fold";

/// Host memory for exponent p. A register is a `BigUint` of `ceil(p / 64)`
/// 64-bit digits. The fixed figure counts the modulus and the peak of one
/// operation: the double-width product, the two fold halves, and the copy of
/// the second operand in `mul`, six register sizes in all. Allocator
/// overhead is not counted.
pub fn memory_estimate(p: u32) -> BackendResult<MemoryEstimate> {
    let (lo, hi) = supported_exponents();
    if p < lo || p > hi {
        return Err(BackendError::Unsupported(format!(
            "exponent {p} defines no modulus"
        )));
    }
    let reg = 8 * u64::from(p).div_ceil(64);
    Ok(MemoryEstimate {
        plan: PLAN.into(),
        bytes_per_register: reg,
        fixed_bytes: 6 * reg,
        on_device: false,
    })
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
            build: crate::build_identity().into(),
            plan: PLAN.into(),
            device: String::new(),
        }
    }

    fn alloc(&mut self) -> BackendResult<Reg> {
        let r = new_handle()?;
        self.regs.insert(r.0, BigUint::zero());
        Ok(r)
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

    fn sub(&mut self, dst: Reg, src: Reg) -> BackendResult<()> {
        if dst == src {
            return Err(BackendError::Unsupported(
                "sub needs distinct handles".into(),
            ));
        }
        let b = self.get(src)?.clone();
        let a = std::mem::take(self.get_mut(dst)?);
        let v = if a >= b { a - b } else { a + &self.n - b };
        *self.get_mut(dst)? = v;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_estimate_matches_the_resident_digits() {
        assert!(matches!(
            memory_estimate(1),
            Err(BackendError::Unsupported(_))
        ));
        for p in [2u32, 61, 64, 65, 127, 86243, u32::MAX] {
            let m = memory_estimate(p).unwrap();
            assert_eq!(m.plan, PLAN);
            assert!(!m.on_device);
            assert_eq!(m.fixed_bytes, 6 * m.bytes_per_register);
            if p > 200_000 {
                continue;
            }
            let mut be = CpuBackend::new(p).unwrap();
            assert_eq!(be.identity().plan, m.plan);
            // The largest residue, N - 1, occupies every digit.
            let r = be.alloc().unwrap();
            let x = Residue::from_biguint(p, &(&be.n - 1u32));
            be.import(r, &x).unwrap();
            let digits = be.regs[&r.0].to_u64_digits().len() as u64;
            assert_eq!(8 * digits, m.bytes_per_register, "p={p}");
        }
    }
}
