// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The DWT backend allocates per batch, never per iteration: a batch of many
//! squarings, LL steps, or a multiplication makes no more heap allocations
//! than a batch of one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use mersenne_forge::backend::dwt::DwtBackend;
use mersenne_forge::backend::Backend;
use mersenne_forge::residue::Residue;

struct Counting;
static COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.fetch_add(1, Ordering::SeqCst);
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations(f: impl FnOnce()) -> usize {
    let before = COUNT.load(Ordering::SeqCst);
    f();
    COUNT.load(Ordering::SeqCst) - before
}

// One test only: the counter is process wide.
#[test]
fn batches_allocate_independently_of_their_length() {
    let p = 4423;
    let mut be = DwtBackend::new(p).unwrap();
    let r = be.alloc().unwrap();
    let s = be.alloc().unwrap();
    be.import(r, &Residue::from_u64(p, 3)).unwrap();
    be.import(s, &Residue::from_u64(p, 7)).unwrap();
    be.square(r, 1).unwrap();
    be.ll_step(r, 1).unwrap();
    be.mul(r, s).unwrap();
    let one = allocations(|| be.square(r, 1).unwrap());
    let many = allocations(|| be.square(r, 500).unwrap());
    assert_eq!(one, many, "square: {one} for 1 iteration, {many} for 500");
    let one = allocations(|| be.ll_step(r, 1).unwrap());
    let many = allocations(|| be.ll_step(r, 500).unwrap());
    assert_eq!(one, many, "ll_step: {one} for 1 iteration, {many} for 500");
    assert_eq!(
        allocations(|| be.mul(r, s).unwrap()),
        0,
        "mul allocates nothing"
    );
}
