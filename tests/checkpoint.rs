// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Checkpoint corruption, mismatched identity, interrupted writes, resume.

mod common;

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::backend::dwt::DwtBackend;
use mersenne_forge::checkpoint::{create_dir_all_durable, prev_path, Checkpoint, CheckpointError};
use mersenne_forge::ll::run_ll;
use mersenne_forge::prp::run_prp;
use mersenne_forge::run::{
    Control, FaultKind, FaultPlan, FaultTarget, ProcessState, ProtocolError,
};

use common::{ll_cfg, prp_cfg, PRIME_P};

/// A PRP run halted at k = 1000 leaves checkpoints at 960 and (prev) 896.
fn halted_prp(dir: &std::path::Path) -> std::path::PathBuf {
    let cfg = prp_cfg(dir, None);
    let mut ctl = Control {
        halt_at: Some(1000),
        ..Control::default()
    };
    run_prp(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &cfg,
        serde_json::json!({}),
        None,
        &mut ctl,
    )
    .unwrap();
    cfg.checkpoint.unwrap()
}

#[test]
fn corrupted_bytes_fall_back_to_previous() {
    let dir = tempfile::tempdir().unwrap();
    let path = halted_prp(dir.path());
    assert_eq!(Checkpoint::load_file(&path).unwrap().meta.k, 960);
    assert_eq!(
        Checkpoint::load_file(&prev_path(&path)).unwrap().meta.k,
        896
    );
    let good = std::fs::read(&path).unwrap();
    for at in [
        0usize,
        25,
        40,
        good.len() / 2,
        good.len() - 40,
        good.len() - 1,
    ] {
        let mut bad = good.clone();
        bad[at] ^= 0x01;
        std::fs::write(&path, &bad).unwrap();
        assert!(
            matches!(
                Checkpoint::load_file(&path),
                Err(CheckpointError::Corrupt(_))
            ),
            "byte {at}"
        );
        let (ck, used) = Checkpoint::load_latest(&path).unwrap();
        assert_eq!((ck.meta.k, used), (896, prev_path(&path)));
    }
    // Resume from the previous state still finishes correctly.
    std::fs::write(&path, &good[..good.len() / 3]).unwrap();
    let (ck, _) = Checkpoint::load_latest(&path).unwrap();
    let cfg = prp_cfg(dir.path(), None);
    let r = run_prp(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &cfg,
        serde_json::json!({}),
        Some(&ck),
        &mut Control::default(),
    )
    .unwrap();
    assert_eq!(r.result.unwrap().res64, common::oracle_prp_res64(PRIME_P));
}

#[test]
fn interrupted_write_keeps_last_usable_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = halted_prp(dir.path());
    // A crash between rotating to .prev and renaming the new file leaves no
    // current file and a stray temporary file.
    std::fs::rename(&path, dir.path().join("moved")).unwrap();
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    std::fs::write(&tmp, b"MERSENNE-FORGE-CKPT\npartial").unwrap();
    let (ck, used) = Checkpoint::load_latest(&path).unwrap();
    assert_eq!((ck.meta.k, used), (896, prev_path(&path)));
    // Both unusable: a clear error, never a fresh start.
    std::fs::write(prev_path(&path), b"junk").unwrap();
    assert!(Checkpoint::load_latest(&path).is_err());
}

#[test]
fn oversized_metadata_is_rejected_before_allocation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.ckpt");
    let mut b = b"MERSENNE-FORGE-CKPT\n".to_vec();
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&u64::MAX.to_le_bytes());
    b.extend_from_slice(&[0u8; 64]);
    std::fs::write(&path, &b).unwrap();
    assert!(matches!(
        Checkpoint::load_file(&path),
        Err(CheckpointError::Corrupt(_))
    ));
}

#[test]
fn mismatched_identity_or_configuration_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = halted_prp(dir.path());
    let (ck, _) = Checkpoint::load_latest(&path).unwrap();
    let cfg = prp_cfg(dir.path(), None);
    // Another backend.
    let e = run_prp(
        &mut DwtBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &cfg,
        serde_json::json!({}),
        Some(&ck),
        &mut Control::default(),
    )
    .unwrap_err();
    assert!(
        matches!(e, ProtocolError::Checkpoint(CheckpointError::Mismatch(_))),
        "{e}"
    );
    // Another configuration.
    let mut other = cfg.clone();
    other.check_step = Some(128);
    let e = run_prp(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &other,
        serde_json::json!({}),
        Some(&ck),
        &mut Control::default(),
    )
    .unwrap_err();
    assert!(
        matches!(e, ProtocolError::Checkpoint(CheckpointError::Mismatch(_))),
        "{e}"
    );
    // Another exponent.
    let e = run_prp(
        &mut CpuBackend::new(1277).unwrap(),
        1277,
        &cfg,
        serde_json::json!({}),
        Some(&ck),
        &mut Control::default(),
    )
    .unwrap_err();
    assert!(
        matches!(e, ProtocolError::Checkpoint(CheckpointError::Mismatch(_))),
        "{e}"
    );
    // An iteration out of range (valid hash, rewritten metadata).
    for bad_k in [u64::from(PRIME_P), 961, u64::MAX] {
        let mut forged = ck.clone();
        forged.meta.k = bad_k;
        let e = run_prp(
            &mut CpuBackend::new(PRIME_P).unwrap(),
            PRIME_P,
            &cfg,
            serde_json::json!({}),
            Some(&forged),
            &mut Control::default(),
        )
        .unwrap_err();
        assert!(
            matches!(e, ProtocolError::Checkpoint(CheckpointError::Mismatch(_))),
            "{e}"
        );
    }
    // Metadata fields with one valid value for a PRP state.
    let forgeries: [fn(&mut Checkpoint); 3] = [
        |c| c.meta.state = "provisional".into(),
        |c| c.meta.verified_k = 896,
        |c| c.meta.proof_durable_through = Some(640),
    ];
    for forge in forgeries {
        let mut forged = ck.clone();
        forge(&mut forged);
        let e = run_prp(
            &mut CpuBackend::new(PRIME_P).unwrap(),
            PRIME_P,
            &cfg,
            serde_json::json!({}),
            Some(&forged),
            &mut Control::default(),
        )
        .unwrap_err();
        assert!(
            matches!(e, ProtocolError::Checkpoint(CheckpointError::Mismatch(_))),
            "{e}"
        );
    }
    // An LL checkpoint given to PRP.
    let mut ctl = Control {
        halt_at: Some(300),
        ..Control::default()
    };
    let lcfg = ll_cfg(dir.path());
    run_ll(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &lcfg,
        serde_json::json!({}),
        None,
        &mut ctl,
    )
    .unwrap();
    let (lck, _) = Checkpoint::load_latest(lcfg.checkpoint.as_ref().unwrap()).unwrap();
    let e = run_prp(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &cfg,
        serde_json::json!({}),
        Some(&lck),
        &mut Control::default(),
    )
    .unwrap_err();
    assert!(
        matches!(e, ProtocolError::Checkpoint(CheckpointError::Mismatch(_))),
        "{e}"
    );
}

#[test]
fn mathematically_wrong_checkpoint_is_rejected_on_load() {
    // Valid bytes and hash, damaged residue pair: the on-load Gerbicz check
    // catches what the integrity hash cannot. It cannot catch a consistent
    // pair under a wrong iteration count: k and the configuration are trusted
    // metadata, and the hash only detects damaged bytes.
    let dir = tempfile::tempdir().unwrap();
    let path = halted_prp(dir.path());
    let (mut ck, _) = Checkpoint::load_latest(&path).unwrap();
    ck.residues[1] = ck.residues[1].with_bit_flipped(5);
    ck.save(&path).unwrap();
    let (ck, _) = Checkpoint::load_latest(&path).unwrap();
    let cfg = prp_cfg(dir.path(), None);
    let e = run_prp(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &cfg,
        serde_json::json!({}),
        Some(&ck),
        &mut Control::default(),
    )
    .unwrap_err();
    assert!(
        matches!(e, ProtocolError::CheckpointRejected { k: 960 }),
        "{e}"
    );
}

#[test]
fn a_save_after_falling_back_keeps_the_good_previous_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = halted_prp(dir.path());
    let mut bad = std::fs::read(&path).unwrap();
    let n = bad.len();
    bad[n / 2] ^= 0x01;
    std::fs::write(&path, &bad).unwrap();
    let (ck, used) = Checkpoint::load_latest(&path).unwrap();
    assert_eq!((ck.meta.k, &used), (896, &prev_path(&path)));
    // Resume from 896 and stop right after the one save, at the early check
    // at 896 + 2 * 16 = 928.
    let mut ctl = Control {
        halt_at: Some(929),
        ..Control::default()
    };
    run_prp(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &prp_cfg(dir.path(), None),
        serde_json::json!({}),
        Some(&ck),
        &mut ctl,
    )
    .unwrap();
    assert_eq!(Checkpoint::load_file(&path).unwrap().meta.k, 928);
    // The corrupt primary was removed, not rotated over the good .prev.
    assert_eq!(
        Checkpoint::load_file(&prev_path(&path)).unwrap().meta.k,
        896
    );
}

#[test]
fn a_report_names_only_a_checkpoint_that_exists() {
    // A stop request and a fault before the first checkpoint: nothing was
    // written, so no resume command may be offered.
    let dir = tempfile::tempdir().unwrap();
    let cfg = prp_cfg(dir.path(), None);
    let mut ctl = Control {
        deadline: Some(std::time::Instant::now()),
        faults: vec![FaultPlan::once(
            16,
            FaultTarget::Data,
            FaultKind::BackendFault,
        )],
        ..Control::default()
    };
    let r = run_prp(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &cfg,
        serde_json::json!({}),
        None,
        &mut ctl,
    )
    .unwrap();
    assert!(!cfg.checkpoint.as_ref().unwrap().exists());
    assert!(
        matches!(
            r.process,
            ProcessState::Incomplete {
                checkpoint: None,
                ..
            }
        ),
        "{:?}",
        r.process
    );

    let lcfg = ll_cfg(dir.path());
    let mut ctl = Control {
        deadline: Some(std::time::Instant::now()),
        faults: vec![FaultPlan::once(
            50,
            FaultTarget::Data,
            FaultKind::BackendFault,
        )],
        ..Control::default()
    };
    let r = run_ll(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &lcfg,
        serde_json::json!({}),
        None,
        &mut ctl,
    )
    .unwrap();
    assert!(!lcfg.checkpoint.as_ref().unwrap().exists());
    assert!(
        matches!(
            r.process,
            ProcessState::Incomplete {
                checkpoint: None,
                ..
            }
        ),
        "{:?}",
        r.process
    );

    // After a save, the report names the file.
    let mut ctl = Control {
        deadline: Some(std::time::Instant::now()),
        ..Control::default()
    };
    let r = run_prp(
        &mut CpuBackend::new(PRIME_P).unwrap(),
        PRIME_P,
        &cfg,
        serde_json::json!({}),
        None,
        &mut ctl,
    )
    .unwrap();
    assert!(
        matches!(&r.process, ProcessState::Incomplete { checkpoint: Some(c), .. } if c.exists()),
        "{:?}",
        r.process
    );
}

#[test]
fn nested_directories_are_created() {
    let dir = tempfile::tempdir().unwrap();
    let deep = dir.path().join("a").join("b").join("c");
    create_dir_all_durable(&deep).unwrap();
    assert!(deep.is_dir());
    create_dir_all_durable(&deep).unwrap();
}
