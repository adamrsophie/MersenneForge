// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The generic protocol suite on the f64 DWT backend.

mod common;

use mersenne_forge::backend::dwt::DwtBackend;

fn mk(p: u32) -> DwtBackend {
    DwtBackend::new(p).expect("DWT backend")
}

protocol_suite!(mk);
