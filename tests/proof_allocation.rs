// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Reading and validating a proof never aborts on a failed allocation.
//!
//! This test binary installs an allocator that, when armed, grants only a
//! fixed number of large allocations and then returns null for the rest. A
//! null from an infallible allocation (`Vec::to_vec`, `resize`, `collect`)
//! aborts the whole process, so the test only passes when every large
//! allocation on the read, parse, and validate path is fallible or absent.
//! This covers the program's own allocations, not an operating system that
//! overcommits memory and later kills the process.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use mersenne_forge::proof::{Proof, ProofError};

/// Allocations at least this large are counted when armed.
const LARGE: usize = 1 << 20;

struct Limited;
static ARMED: AtomicBool = AtomicBool::new(false);
static GRANTS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Limited {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= LARGE && ARMED.load(Ordering::SeqCst) {
            let left = GRANTS.load(Ordering::SeqCst);
            if left == 0 {
                return std::ptr::null_mut();
            }
            GRANTS.store(left - 1, Ordering::SeqCst);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Limited = Limited;

/// One armed section at a time: the counter is process wide.
static SERIAL: Mutex<()> = Mutex::new(());

fn armed<T>(grants: usize, f: impl FnOnce() -> T) -> T {
    GRANTS.store(grants, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    let out = f();
    ARMED.store(false, Ordering::SeqCst);
    out
}

/// A format-valid power 1 proof for a prime p near 2^23: residues of about
/// 1 MiB each (B = 9, middle = 1).
fn large_proof() -> (u32, Vec<u8>) {
    let p = 8_388_617u32;
    let n = ((p - 1) / 8 + 1) as usize;
    assert!(n >= LARGE);
    let mut bytes =
        format!("PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=1\nNUMBER=M{p}\n").into_bytes();
    let mut b = vec![0u8; n];
    b[0] = 9;
    let mut m = vec![0u8; n];
    m[0] = 1;
    bytes.extend_from_slice(&b);
    bytes.extend_from_slice(&m);
    (p, bytes)
}

#[test]
fn reading_needs_only_the_residue_buffers() {
    let _g = SERIAL.lock().unwrap();
    let (p, bytes) = large_proof();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("{p}-1.proof"));
    std::fs::write(&path, &bytes).unwrap();
    // Two large buffers (B and the middle) are enough: validation adds none.
    let proof = armed(2, || Proof::read(&path, Some(p))).unwrap();
    assert_eq!(proof.power(), 1);
    // With fewer grants the read fails with a typed error, never an abort.
    for grants in [0, 1] {
        let e = armed(grants, || Proof::read(&path, Some(p))).unwrap_err();
        assert!(
            matches!(e, ProofError::Io(ref m) if m.contains("allocate")),
            "{e}"
        );
    }
}

#[test]
fn parsing_and_validating_need_only_the_copies() {
    let _g = SERIAL.lock().unwrap();
    let (p, bytes) = large_proof();
    let proof = armed(2, || Proof::parse(&bytes, Some(p))).unwrap();
    armed(0, || proof.validate()).unwrap();
    for grants in [0, 1] {
        let e = armed(grants, || Proof::parse(&bytes, Some(p))).unwrap_err();
        assert!(matches!(e, ProofError::Io(_)), "{e}");
    }
}
