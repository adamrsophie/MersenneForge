// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The Jacobi check catches only some LL errors. This test flips single bits
//! between checks and shows both outcomes: detected flips are rolled back to
//! a correct result; undetected flips produce a wrong final residue that no
//! LL check can notice. That is why the LL assurance is weaker than Gerbicz.

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::ll::{run_ll, JacobiCadence, LlConfig};
use mersenne_forge::oracle;
use mersenne_forge::run::{Control, FaultKind, FaultPlan, FaultTarget};

#[test]
fn jacobi_catches_about_half_of_bit_flips() {
    let p = 521;
    let reference = oracle::ll(p).terminal;
    let cfg = LlConfig {
        log_step: 100,
        jacobi: JacobiCadence::EveryLog,
        checkpoint: None,
        max_batch: 1000,
    };
    let (mut detected, mut missed) = (0, 0);
    for bit in 0..64u32 {
        let mut ctl = Control {
            faults: vec![FaultPlan::once(
                250,
                FaultTarget::Data,
                FaultKind::FlipBit(bit * 8),
            )],
            ..Control::default()
        };
        let r = run_ll(
            &mut CpuBackend::new(p).unwrap(),
            p,
            &cfg,
            serde_json::json!({}),
            None,
            &mut ctl,
        )
        .unwrap();
        let res = r.result.unwrap();
        if r.evidence.jacobi_failed > 0 {
            detected += 1;
            assert_eq!(
                res.res2048,
                reference.res2048_hex(),
                "detected flip {bit} must recover"
            );
        } else {
            missed += 1;
            assert_ne!(
                res.res2048,
                reference.res2048_hex(),
                "an undetected flip leaves a wrong residue"
            );
        }
    }
    assert!(
        (16..=48).contains(&detected),
        "detected {detected} of 64, missed {missed}"
    );
}
