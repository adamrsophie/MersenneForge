// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! CLI integration: exit codes and machine-readable JSON.

mod common;

use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use mersenne_forge::backend::cpu::CpuBackend;
use mersenne_forge::checkpoint::{prev_path, Checkpoint};
use mersenne_forge::run::Control;

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
        .args(args)
        .output()
        .expect("binary runs")
}

fn json(o: &Output) -> serde_json::Value {
    serde_json::from_slice(&o.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&o.stdout)))
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn doctor_reports_backends() {
    let o = cli(&["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0));
    let v = json(&o);
    assert_eq!(v["build"], mersenne_forge::build_identity());
    let kinds: Vec<&str> = v["backends"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["cpu", "dwt", "gpu", "cuda"]);
    assert_eq!(v["backends"][0]["built"], true);
    assert!(v["notes"][0].as_str().unwrap().contains("not validation"));

    // Per backend memory figures for one exponent.
    let o = cli(&["doctor", "--exponent", "1279", "--json"]);
    assert_eq!(o.status.code(), Some(0));
    let v = json(&o);
    let cpu = &v["backends"][0]["memory"];
    let want = mersenne_forge::backend::cpu::memory_estimate(1279).unwrap();
    assert_eq!(cpu["bytes_per_register"], want.bytes_per_register);
    assert_eq!(cpu["fixed_bytes"], want.fixed_bytes);
    assert_eq!(cpu["canonical_residue_bytes"], 160);
    let power = cpu["default_proof_power"].as_u64().unwrap();
    assert_eq!(cpu["registers"]["proof_generation"], power + 2);
    assert_eq!(
        cpu["peak_bytes"]["proof_generation"],
        want.fixed_bytes + (power + 2) * want.bytes_per_register
    );
    assert!(v["backends"][1]["memory"]["bytes_per_register"].is_u64());
    // Outside a backend range: an error entry, not a figure.
    let o = cli(&["doctor", "--exponent", "1", "--json"]);
    assert!(json(&o)["backends"][0]["memory"]["error"].is_string());
}

#[test]
fn run_ll_and_prp_with_json() {
    let o = cli(&["run", "--test", "ll", "--exponent", "127", "--json"]);
    assert_eq!(o.status.code(), Some(0));
    let v = json(&o);
    assert_eq!(v["report"]["result"]["verdict"], "prime");
    assert_eq!(v["report"]["process"]["state"], "completed");
    assert_eq!(v["requested"]["exponent"], 127);

    let o = cli(&[
        "run",
        "--test",
        "prp",
        "--exponent",
        "1277",
        "--block-size",
        "200",
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(0));
    let v = json(&o);
    assert_eq!(v["report"]["result"]["verdict"], "composite");
    // Upstream rule scaled down for a small exponent: requested none, effective recorded.
    assert_eq!(v["requested"]["check_step"], serde_json::Value::Null);
    assert_eq!(v["effective"]["block_size"], 200);
    // The derived check step is reported, outside the compared configuration.
    assert_eq!(v["effective"]["check_step"], serde_json::Value::Null);
    assert_eq!(v["report"]["evidence"]["check_step"], 20_000);
    assert_eq!(v["report"]["evidence"]["proof_stage"], "not_requested");
}

#[test]
fn json_failures_are_json_objects() {
    let o = cli(&["run", "--test", "prp", "--exponent", "1281", "--json"]);
    assert_eq!(o.status.code(), Some(4));
    let v = json(&o);
    assert_eq!(v["kind"], "unsupported");
    assert_eq!(v["exit_code"], 4);
    assert!(v["message"].as_str().unwrap().contains("1281"));
    let o = cli(&["run", "--test", "prp", "--exponent", "x", "--json"]);
    assert_eq!(o.status.code(), Some(2));
    assert_eq!(json(&o)["kind"], "usage");
    let o = cli(&["verify", "--proof", "/nonexistent/x.proof", "--json"]);
    assert_eq!(o.status.code(), Some(6));
    assert_eq!(json(&o)["status"], "error");
}

#[test]
fn unrepresentable_budgets_are_usage_errors() {
    for b in [
        "18446744073709551616s",
        "1e400s",
        "-5s",
        "99999999999999999999d",
    ] {
        let o = cli(&["run", "--test", "ll", "--exponent", "127", "--budget", b]);
        assert_eq!(
            o.status.code(),
            Some(2),
            "{b}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
}

#[test]
fn usage_and_unsupported_exit_codes() {
    assert_eq!(
        cli(&[
            "run",
            "--test",
            "prp",
            "--exponent",
            "1279",
            "--backend",
            "bogus"
        ])
        .status
        .code(),
        Some(2)
    );
    assert_eq!(
        cli(&["run", "--test", "prp", "--exponent", "1281"])
            .status
            .code(),
        Some(4)
    );
    assert_eq!(
        cli(&[
            "run",
            "--test",
            "prp",
            "--exponent",
            "1279",
            "--block-size",
            "7"
        ])
        .status
        .code(),
        Some(4)
    );
    let o = cli(&[
        "run",
        "--test",
        "prp",
        "--exponent",
        "1279",
        "--backend",
        "gpu",
    ]);
    assert_eq!(o.status.code(), Some(4));
    assert!(String::from_utf8_lossy(&o.stderr).contains("--budget"));
    if !cfg!(feature = "gpu") {
        let o = cli(&[
            "run",
            "--test",
            "prp",
            "--exponent",
            "1279",
            "--backend",
            "gpu",
            "--budget",
            "10s",
        ]);
        assert_eq!(o.status.code(), Some(4));
        assert!(String::from_utf8_lossy(&o.stderr).contains("built without GPU support"));
    }
    if !cfg!(feature = "cuda") {
        let o = cli(&[
            "run",
            "--test",
            "prp",
            "--exponent",
            "1279",
            "--backend",
            "cuda",
            "--budget",
            "10s",
        ]);
        assert_eq!(o.status.code(), Some(4));
        assert!(String::from_utf8_lossy(&o.stderr).contains("--features cuda"));
    }
}

#[test]
fn budget_incomplete_then_resume_then_verify() {
    let dir = tempfile::tempdir().unwrap();
    let (ck, pf) = (dir.path().join("ck"), dir.path().join("proof"));
    let o = cli(&[
        "run",
        "--test",
        "prp",
        "--exponent",
        "4423",
        "--block-size",
        "200",
        "--check-step",
        "400",
        "--proof-dir",
        s(&pf),
        "--proof-power",
        "6",
        "--checkpoint-dir",
        s(&ck),
        "--budget",
        "0.001s",
        "--json",
    ]);
    assert_eq!(
        o.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    let v = json(&o);
    assert_eq!(v["report"]["process"]["state"], "incomplete");
    assert!(
        v["report"]["result"].is_null(),
        "an incomplete run has no result"
    );
    let ckpt = ck.join("M4423-prp.ckpt");
    let o = cli(&["resume", "--checkpoint", s(&ckpt), "--json"]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v = json(&o);
    assert_eq!(v["report"]["result"]["verdict"], "probable_prime");
    assert_eq!(
        v["report"]["evidence"]["proof"]["self_verification"],
        "accepted"
    );
    let proof = pf.join("4423-6.proof");

    let o = cli(&["verify", "--proof", s(&proof), "--json"]);
    assert_eq!(o.status.code(), Some(0));
    let v = json(&o);
    assert_eq!(v["status"], "accepted");
    assert_eq!(v["detail"]["terminal_recomputed"], true);
    let o = cli(&[
        "verify",
        "--proof",
        s(&proof),
        "--backend",
        "dwt",
        "--exponent",
        "4423",
    ]);
    assert_eq!(o.status.code(), Some(0));

    let mut bytes = std::fs::read(&proof).unwrap();
    let n = bytes.len();
    bytes[n - 100] ^= 4;
    let bad = dir.path().join("bad.proof");
    std::fs::write(&bad, &bytes).unwrap();
    let o = cli(&["verify", "--proof", s(&bad), "--json"]);
    assert_eq!(o.status.code(), Some(7));
    assert_eq!(json(&o)["detail"]["error"], "Rejected");
    std::fs::write(&bad, &bytes[..n - 1]).unwrap();
    let o = cli(&["verify", "--proof", s(&bad), "--json"]);
    assert_eq!(o.status.code(), Some(7));
    assert!(json(&o)["detail"]["error"]
        .as_str()
        .unwrap()
        .starts_with("Truncated"));
    let o = cli(&[
        "verify",
        "--proof",
        s(&proof),
        "--exponent",
        "4253",
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(7));
    assert!(json(&o)["detail"]["error"]
        .as_str()
        .unwrap()
        .starts_with("WrongExponent"));
    assert_eq!(
        cli(&["verify", "--proof", s(&dir.path().join("none"))])
            .status
            .code(),
        Some(6)
    );
    assert_eq!(
        cli(&["resume", "--checkpoint", s(&dir.path().join("none"))])
            .status
            .code(),
        Some(6)
    );
}

#[test]
fn bench_writes_observations() {
    let dir = tempfile::tempdir().unwrap();
    let plan = dir.path().join("plan.json");
    std::fs::write(
        &plan,
        r#"{"name":"t","order":"randomized","seed":3,"repeats":2,"warmups":1,"entries":[
            {"kind":"engine","name":"prp-127","test":"prp","exponent":127,"backend":"cpu","options":{"block_size":200}},
            {"kind":"external","name":"true","argv":["true"]},
            {"kind":"external","name":"bad","argv":["false"]}]}"#,
    )
    .unwrap();
    let out = dir.path().join("out.json");
    // Directories of other invocations are never reused or deleted.
    let mine = dir.path().join("out.json.runs").join("earlier");
    std::fs::create_dir_all(&mine).unwrap();
    std::fs::write(mine.join("keep"), b"user data").unwrap();
    let o = cli(&["bench", "--plan", s(&plan), "--output", s(&out)]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let obs = v["observations"].as_array().unwrap();
    assert_eq!(obs.len(), 9);
    assert_eq!(v["seed_used"], 3);
    assert!(obs
        .iter()
        .all(|o| o["telemetry"] == "unavailable" && o["energy_joules"].is_null()));
    assert!(obs
        .iter()
        .filter(|o| o["entry"] == "bad")
        .all(|o| o["status"] == "failed"));
    assert!(obs
        .iter()
        .filter(|o| o["entry"] == "prp-127")
        .all(|o| o["verdict"] == "ProbablePrime"
            && o["backend"]["kind"] == "cpu"
            && o["proof_stage"] == "not_requested"
            && o["check_step"] == 20_000
            && o["test"] == "prp"
            && o["exponent"] == 127
            && o["entry_index"] == 0
            && o["result_source"] == "engine checked run"
            && o["reliability"]["gerbicz_passed"].as_u64().unwrap() > 0));
    assert_eq!(std::fs::read(mine.join("keep")).unwrap(), b"user data");
    assert_eq!(v["build"], mersenne_forge::build_identity());
    assert_eq!(v["plan_sha3"].as_str().unwrap().len(), 64);
    for o in obs.iter().filter(|o| o["entry"] == "true") {
        assert_eq!(o["argv"], serde_json::json!(["true"]));
        assert_eq!(
            o["result_source"],
            "external process timing; result not verified"
        );
        assert!(Path::new(o["stdout"].as_str().unwrap()).is_file());
        assert!(Path::new(o["stderr"].as_str().unwrap()).is_file());
    }
    let csv = dir.path().join("out.csv");
    assert_eq!(
        cli(&["bench", "--plan", s(&plan), "--output", s(&csv)])
            .status
            .code(),
        Some(0)
    );
    assert_eq!(std::fs::read_to_string(&csv).unwrap().lines().count(), 10);
}

#[test]
fn bench_rewrites_its_output_after_every_observation() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.json");
    let plan = dir.path().join("plan.json");
    // The second entry prints the output file as it stands after the first.
    let text = serde_json::json!({
        "name": "t", "order": "sequential", "repeats": 1,
        "entries": [
            {"kind": "external", "name": "first", "argv": ["true"]},
            {"kind": "external", "name": "second", "argv": ["cat", s(&out)]},
        ]
    });
    std::fs::write(&plan, text.to_string()).unwrap();
    let o = cli(&["bench", "--plan", s(&plan), "--output", s(&out)]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let seen = std::fs::read(v["observations"][1]["stdout"].as_str().unwrap()).unwrap();
    let seen: serde_json::Value = serde_json::from_slice(&seen).unwrap();
    assert_eq!(seen["observations"].as_array().unwrap().len(), 1);
    assert_eq!(seen["observations"][0]["entry"], "first");

    // Invalid budgets in a plan are usage errors, before anything runs.
    for budget in ["-1", "1e300"] {
        let bad = format!(
            r#"{{"name":"t","order":"sequential","repeats":1,"entries":[{{"kind":"engine","name":"x","test":"ll","exponent":127,"backend":"cpu","options":{{"budget_secs":{budget}}}}}]}}"#
        );
        std::fs::write(&plan, bad).unwrap();
        let o = cli(&[
            "bench",
            "--plan",
            s(&plan),
            "--output",
            s(&dir.path().join("b.json")),
        ]);
        assert_eq!(
            o.status.code(),
            Some(2),
            "{budget}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(!dir.path().join("b.json").exists());
    }
}

#[cfg(unix)]
#[test]
fn bench_stops_an_external_command_on_ctrl_c() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.json");
    let plan = dir.path().join("plan.json");
    std::fs::write(
        &plan,
        r#"{"name":"t","order":"sequential","repeats":3,"entries":[{"kind":"external","name":"sleep","argv":["sleep","30"]}]}"#,
    )
    .unwrap();
    let start = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
        .args(["bench", "--plan", s(&plan), "--output", s(&out)])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // Wait until the first observation is running, then send SIGINT once.
    let started = || {
        std::fs::read_dir(dir.path().join("out.json.runs"))
            .into_iter()
            .flatten()
            .any(|e| e.unwrap().path().join("0.stdout").exists())
    };
    while !started() {
        assert!(child.try_wait().unwrap().is_none(), "bench ended early");
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "no observation started"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(200));
    let kill = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(kill.success());
    let status = child.wait().unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "the plan was not stopped"
    );
    assert_eq!(status.code(), Some(3));
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let obs = v["observations"].as_array().unwrap();
    assert_eq!(obs.len(), 1, "the plan ends after the stopped observation");
    assert_eq!(obs[0]["status"], "incomplete");
    assert_eq!(obs[0]["error"], "stop requested");
}

/// PRP of M1279 halted at k = 1000 through the library, leaving checkpoints
/// at 960 and (.prev) 896.
fn library_checkpoint(dir: &Path) -> std::path::PathBuf {
    let cfg = common::prp_cfg(dir, None);
    let mut ctl = Control {
        halt_at: Some(1000),
        ..Control::default()
    };
    common::prp_run(&mut CpuBackend::new(1279).unwrap(), 1279, &cfg, &mut ctl).unwrap();
    cfg.checkpoint.unwrap()
}

#[test]
fn run_refuses_an_existing_checkpoint_and_a_held_lock() {
    let dir = tempfile::tempdir().unwrap();
    let ck = dir.path().join("ck");
    let args = [
        "run",
        "--test",
        "prp",
        "--exponent",
        "4423",
        "--block-size",
        "200",
        "--check-step",
        "400",
        "--checkpoint-dir",
        s(&ck),
        "--budget",
        "0.001s",
    ];
    assert_eq!(cli(&args).status.code(), Some(3));
    let ckpt = ck.join("M4423-prp.ckpt");
    let before = std::fs::read(&ckpt).unwrap();
    let o = cli(&args);
    assert_eq!(o.status.code(), Some(6));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("resume --checkpoint") && err.contains("delete"),
        "{err}"
    );
    assert_eq!(
        std::fs::read(&ckpt).unwrap(),
        before,
        "the checkpoint was touched"
    );
    // Only .prev left: still refused.
    std::fs::rename(&ckpt, prev_path(&ckpt)).unwrap();
    assert_eq!(cli(&args).status.code(), Some(6));

    // Another process holds the lock: run and resume refuse.
    let other = dir.path().join("other");
    std::fs::create_dir_all(&other).unwrap();
    let lock = std::fs::File::create(other.join("M4423-ll.ckpt.lock")).unwrap();
    lock.try_lock().unwrap();
    let o = cli(&[
        "run",
        "--test",
        "ll",
        "--exponent",
        "4423",
        "--checkpoint-dir",
        s(&other),
    ]);
    assert_eq!(o.status.code(), Some(6));
    assert!(String::from_utf8_lossy(&o.stderr).contains("held by another"));
    let o = cli(&["resume", "--checkpoint", s(&other.join("M4423-ll.ckpt"))]);
    assert_eq!(o.status.code(), Some(6));
    assert!(String::from_utf8_lossy(&o.stderr).contains("held by another"));
    drop(lock);
}

#[test]
fn resume_falls_back_when_the_primary_fails_its_on_load_check() {
    let dir = tempfile::tempdir().unwrap();
    let path = library_checkpoint(dir.path());
    // A primary with a valid hash and a damaged residue pair.
    let mut ck = Checkpoint::load_file(&path).unwrap();
    ck.residues[1] = ck.residues[1].with_bit_flipped(5);
    std::fs::write(&path, ck.to_bytes()).unwrap();
    let o = cli(&["resume", "--checkpoint", s(&path), "--json"]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    let v = json(&o);
    assert_eq!(v["loaded_from"], s(&prev_path(&path)));
    assert_eq!(v["resumed_at"], 896);
    assert_eq!(
        v["report"]["result"]["res64"],
        common::oracle_prp_res64(1279)
    );
    let mut aside = path.as_os_str().to_owned();
    aside.push(".rejected");
    assert!(
        Path::new(&aside).exists(),
        "the rejected primary is kept aside"
    );

    // An identity mismatch does not fall back.
    let dir = tempfile::tempdir().unwrap();
    let path = library_checkpoint(dir.path());
    let mut ck = Checkpoint::load_file(&path).unwrap();
    ck.meta.backend.plan = "another plan".into();
    std::fs::write(&path, ck.to_bytes()).unwrap();
    assert_eq!(
        cli(&["resume", "--checkpoint", s(&path)]).status.code(),
        Some(6)
    );
}

#[test]
fn an_unwritable_proof_file_exits_5_and_keeps_the_result() {
    let dir = tempfile::tempdir().unwrap();
    let pf = dir.path().join("proof");
    let ckpt_dir = dir.path().join("ckpt");
    std::fs::create_dir_all(pf.join("1279-4.proof")).unwrap();
    let o = cli(&[
        "run",
        "--test",
        "prp",
        "--exponent",
        "1279",
        "--block-size",
        "200",
        "--proof-dir",
        s(&pf),
        "--proof-power",
        "4",
        "--checkpoint-dir",
        s(&ckpt_dir),
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(5));
    let v = json(&o);
    assert_eq!(v["report"]["result"]["verdict"], "probable_prime");
    assert_eq!(v["report"]["evidence"]["proof_stage"], "failed");
    let ckpt = ckpt_dir.join("M1279-prp.ckpt");
    assert_eq!(
        Checkpoint::load_file(&ckpt).unwrap().meta.state,
        "completed"
    );
    std::fs::remove_dir(pf.join("1279-4.proof")).unwrap();
    let resumed = cli(&["resume", "--checkpoint", s(&ckpt), "--json"]);
    assert_eq!(resumed.status.code(), Some(0));
    let resumed = json(&resumed);
    assert_eq!(resumed["report"]["result"], v["report"]["result"]);
    assert_eq!(resumed["report"]["timings"]["iterations"], 0.0);
    assert_eq!(resumed["report"]["evidence"]["proof_stage"], "verified");
}

#[test]
fn a_killed_run_resumes_to_the_uninterrupted_result() {
    let dir = tempfile::tempdir().unwrap();
    let run = |ck: Option<&Path>, pf: &Path| {
        let mut args: Vec<String> = [
            "run",
            "--test",
            "prp",
            "--exponent",
            "44497",
            "--backend",
            "dwt",
            "--block-size",
            "200",
            "--check-step",
            "2000",
            "--proof-power",
            "6",
            "--proof-dir",
            s(pf),
            "--json",
        ]
        .iter()
        .map(|a| a.to_string())
        .collect();
        if let Some(c) = ck {
            args.extend(["--checkpoint-dir".into(), s(c).into()]);
        }
        Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    };
    let (ck, pf) = (dir.path().join("ck"), dir.path().join("proof"));
    let ckpt = ck.join("M44497-prp.ckpt");
    let mut child = run(Some(&ck), &pf);
    while !ckpt.exists() {
        assert!(
            child.try_wait().unwrap().is_none(),
            "finished before a checkpoint"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        child.try_wait().unwrap().is_none(),
        "the run finished before the kill"
    );
    child.kill().unwrap();
    child.wait().unwrap();
    let killed_at = Checkpoint::load_latest(&ckpt).unwrap().0.meta.k;
    assert!(killed_at < 44497);

    let o = cli(&["resume", "--checkpoint", s(&ckpt), "--json"]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    let resumed = json(&o);

    let base_pf = dir.path().join("base-proof");
    let base = run(None, &base_pf).wait_with_output().unwrap();
    assert_eq!(base.status.code(), Some(0));
    let base = json(&base);
    assert_eq!(resumed["report"]["result"], base["report"]["result"]);
    assert_eq!(resumed["report"]["result"]["res64"], "0000000000000001");
    assert_eq!(
        std::fs::read(pf.join("44497-6.proof")).unwrap(),
        std::fs::read(base_pf.join("44497-6.proof")).unwrap()
    );
}

#[test]
fn resume_uses_relocated_checkpoints_and_proof_residues() {
    for completed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("original");
        let moved = dir.path().join("moved");
        let cfg = common::prp_cfg(&original, Some(4));
        let mut ctl = Control {
            halt_at: (!completed).then_some(1000),
            ..Control::default()
        };
        common::prp_run(&mut CpuBackend::new(1279).unwrap(), 1279, &cfg, &mut ctl).unwrap();
        std::fs::rename(&original, &moved).unwrap();
        let ckpt = moved.join("ckpt/prp.ckpt");
        let proof_dir = moved.join("proof");
        let before = std::fs::read(&ckpt).unwrap();
        let missing = dir.path().join("missing");
        let o = cli(&[
            "resume",
            "--checkpoint",
            s(&ckpt),
            "--proof-dir",
            s(&missing),
            "--json",
        ]);
        assert_eq!(o.status.code(), Some(6));
        assert_eq!(std::fs::read(&ckpt).unwrap(), before);
        assert!(!missing.exists());
        let o = cli(&[
            "resume",
            "--checkpoint",
            s(&ckpt),
            "--proof-dir",
            s(&proof_dir),
            "--json",
        ]);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&o.stdout)
        );
        let v = json(&o);
        assert_eq!(
            v["report"]["result"]["res64"],
            common::oracle_prp_res64(1279)
        );
        assert_eq!(v["report"]["evidence"]["proof_stage"], "verified");
        assert_eq!(v["effective"]["proof_dir"], s(&proof_dir));
        assert_eq!(v["effective"]["checkpoint"], s(&ckpt));
        if completed {
            assert_eq!(v["resumed_at"], 1280);
            assert_eq!(v["report"]["timings"]["iterations"], 0.0);
        }
        assert!(!original.exists(), "resume recreated the old paths");
        let ck = Checkpoint::load_file(&ckpt).unwrap();
        assert_eq!(ck.meta.config["proof_dir"], s(&proof_dir));
        // The new directory is durable configuration, so no override is needed again.
        let before = std::fs::read(proof_dir.join("1279-4.proof")).unwrap();
        let o = cli(&["resume", "--checkpoint", s(&ckpt), "--json"]);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&o.stdout)
        );
        assert_eq!(json(&o)["report"]["timings"]["iterations"], 0.0);
        assert_eq!(
            std::fs::read(proof_dir.join("1279-4.proof")).unwrap(),
            before
        );
    }
}

#[test]
fn resume_refuses_to_add_a_proof_schedule() {
    let dir = tempfile::tempdir().unwrap();
    let path = library_checkpoint(dir.path());
    let before = std::fs::read(&path).unwrap();
    let o = cli(&[
        "resume",
        "--checkpoint",
        s(&path),
        "--proof-dir",
        s(&dir.path().join("proof")),
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(2));
    assert!(json(&o)["message"]
        .as_str()
        .unwrap()
        .contains("proof schedule"));
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// A run that holds a proof directory owns it: a second run with the same
/// exponent and proof directory is refused, with or without a checkpoint and
/// whatever its checkpoint path, and the first run's residue files stay
/// byte for byte as they were.
#[test]
fn a_held_proof_directory_refuses_a_second_writer() {
    use mersenne_forge::proof::{ProofSchedule, ProofStore};
    let dir = tempfile::tempdir().unwrap();
    let proof = dir.path().join("proof");
    let store = ProofStore::open(&proof, ProofSchedule::new(1279, 4).unwrap()).unwrap();
    let x = mersenne_forge::residue::Residue::from_u64(1279, 12345);
    let k = store.schedule.points()[0];
    store.save(k, &x).unwrap();
    let rdir = ProofStore::residue_dir(&proof, 1279);
    let snapshot = |d: &Path| {
        let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(d)
            .unwrap()
            .map(|e| e.unwrap().path())
            .map(|p| (p.display().to_string(), std::fs::read(&p).unwrap()))
            .collect();
        v.sort();
        v
    };
    let before = snapshot(&rdir);
    let base = [
        "run",
        "--test",
        "prp",
        "--exponent",
        "1279",
        "--backend",
        "cpu",
    ];
    let other_ckpt = dir.path().join("other");
    let cases: [Vec<&str>; 2] = [
        vec!["--proof-dir", s(&proof), "--proof-power", "4", "--json"],
        vec![
            "--proof-dir",
            s(&proof),
            "--proof-power",
            "4",
            "--checkpoint-dir",
            s(&other_ckpt),
            "--json",
        ],
    ];
    for extra in cases {
        let args: Vec<&str> = base.iter().copied().chain(extra).collect();
        let o = cli(&args);
        assert_eq!(
            o.status.code(),
            Some(6),
            "{}",
            String::from_utf8_lossy(&o.stdout)
        );
        assert!(json(&o)["message"].as_str().unwrap().contains(".lock"));
        assert_eq!(snapshot(&rdir), before, "the held store was modified");
    }
    assert_eq!(store.load(k).unwrap(), x);
    drop(store);
    // Released: the same command now runs to completion.
    let args: Vec<&str> = base
        .iter()
        .copied()
        .chain(["--proof-dir", s(&proof), "--proof-power", "4", "--json"])
        .collect();
    let o = cli(&args);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    assert_eq!(json(&o)["report"]["evidence"]["proof_stage"], "verified");
}

fn bench(plan: &serde_json::Value, out: &Path) -> Output {
    let path = out.with_extension("plan");
    std::fs::write(&path, plan.to_string()).unwrap();
    cli(&["bench", "--plan", s(&path), "--output", s(out)])
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

/// Reports that share a stem (report.json, report.csv) and a rerun of the
/// same output each keep their own capture files; an older report's files
/// are never changed by a later run.
#[test]
fn bench_captures_belong_to_their_run() {
    let dir = tempfile::tempdir().unwrap();
    let plan = |word: &str| {
        serde_json::json!({"name": "t", "order": "sequential", "repeats": 1, "entries": [
            {"kind": "external", "name": "echo", "argv": ["echo", word]}]})
    };
    let json_out = dir.path().join("report.json");
    let csv_out = dir.path().join("report.csv");
    assert_eq!(bench(&plan("FIRST_RUN"), &json_out).status.code(), Some(0));
    assert_eq!(bench(&plan("SECOND_RUN"), &csv_out).status.code(), Some(0));
    let first = read_json(&json_out);
    let first_capture = first["observations"][0]["stdout"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        std::fs::read_to_string(&first_capture).unwrap(),
        "FIRST_RUN\n"
    );
    let csv = std::fs::read_to_string(&csv_out).unwrap();
    assert!(!csv.contains(&first_capture));
    // A rerun to the same output replaces the report, not the old captures.
    assert_eq!(bench(&plan("THIRD_RUN"), &json_out).status.code(), Some(0));
    let third = read_json(&json_out);
    assert_ne!(third["execution_dir"], first["execution_dir"]);
    assert_eq!(
        std::fs::read_to_string(&first_capture).unwrap(),
        "FIRST_RUN\n"
    );
    let third_capture = third["observations"][0]["stdout"].as_str().unwrap();
    assert_eq!(
        std::fs::read_to_string(third_capture).unwrap(),
        "THIRD_RUN\n"
    );
}

/// CSV rows carry the same provenance as the JSON output, and entries that
/// share a display name stay distinguishable by index and workload.
#[test]
fn bench_csv_keeps_the_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let plan = serde_json::json!({"name": "dup", "order": "sequential", "repeats": 1, "entries": [
        {"kind": "engine", "name": "same", "test": "ll", "exponent": 89, "backend": "cpu"},
        {"kind": "engine", "name": "same", "test": "ll", "exponent": 127, "backend": "dwt"}]});
    let json_out = dir.path().join("p.json");
    let csv_out = dir.path().join("p.csv");
    assert_eq!(bench(&plan, &json_out).status.code(), Some(0));
    assert_eq!(bench(&plan, &csv_out).status.code(), Some(0));
    let v = read_json(&json_out);
    let csv = std::fs::read_to_string(&csv_out).unwrap();
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("{name}"))
    };
    // Quote-aware split: "" inside quotes is one quote.
    let split = |l: &str| -> Vec<String> {
        let (mut out, mut cur, mut quoted) = (Vec::new(), String::new(), false);
        let mut it = l.chars().peekable();
        while let Some(c) = it.next() {
            match (c, quoted) {
                ('"', true) if it.peek() == Some(&'"') => {
                    cur.push('"');
                    it.next();
                }
                ('"', _) => quoted = !quoted,
                (',', false) => out.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            }
        }
        out.push(cur);
        out
    };
    let rows: Vec<Vec<String>> = lines.map(split).collect();
    assert_eq!(rows.len(), 2);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), header.len());
        assert_eq!(row[col("plan_name")], "dup");
        assert_eq!(row[col("plan_sha3")], v["plan_sha3"].as_str().unwrap());
        assert_eq!(row[col("build")], mersenne_forge::build_identity());
        assert_eq!(row[col("entry")], "same");
        assert_eq!(row[col("entry_index")], i.to_string());
        assert_eq!(row[col("test")], "ll");
        assert_eq!(row[col("exponent")], ["89", "127"][i]);
        assert_eq!(row[col("backend")], ["cpu", "dwt"][i]);
        assert_eq!(row[col("verdict")], "Prime");
        let obs = &v["observations"][i];
        assert_eq!(
            row[col("transform_plan")],
            obs["backend"]["plan"].as_str().unwrap()
        );
        assert_eq!(row[col("exponent")], obs["exponent"].to_string());
        assert_eq!(
            row[col("options")],
            r#"{"block_size":null,"check_step":null,"proof_power":null,"verify_proof":null,"log_step":null,"jacobi_every_log":null,"device":null,"budget_secs":null}"#
        );
    }
    // The manifest beside the captures holds the exact hashed plan bytes.
    use sha3::Digest;
    let manifest = Path::new(&rows[0][col("execution_dir")]).join("plan.json");
    let bytes = std::fs::read(&manifest).unwrap();
    let digest: String = sha3::Sha3_256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(digest, rows[0][col("plan_sha3")]);
    let recorded: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(recorded["entries"][1]["exponent"], 127);
    // DWT reports its roundoff as a diagnostic.
    assert!(
        rows[1][col("diagnostics")].contains("roundoff"),
        "{}",
        rows[1][col("diagnostics")]
    );
}

/// An empty plan, or one without measured repeats, is a usage error and
/// writes nothing; a stale report is left alone, not passed off as new.
#[test]
fn bench_refuses_an_empty_plan() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("e.json");
    std::fs::write(&out, b"stale").unwrap();
    for plan in [
        serde_json::json!({"name": "e", "order": "sequential", "repeats": 1, "entries": []}),
        serde_json::json!({"name": "e", "order": "sequential", "repeats": 0, "entries": [
            {"kind": "external", "name": "t", "argv": ["true"]}]}),
    ] {
        let o = bench(&plan, &out);
        assert_eq!(
            o.status.code(),
            Some(2),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert_eq!(std::fs::read(&out).unwrap(), b"stale");
    }
    assert!(!dir.path().join("e.json.runs").exists());
}

/// `resume` on a proof store held by another run is refused like `run`: the
/// checkpoint is not moved aside and no file changes.
#[test]
fn resume_refuses_a_held_proof_store_without_touching_the_checkpoint() {
    use mersenne_forge::proof::{ProofSchedule, ProofStore};
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::prp_cfg(dir.path(), Some(4));
    let mut ctl = Control {
        halt_at: Some(1000),
        ..Control::default()
    };
    common::prp_run(&mut CpuBackend::new(1279).unwrap(), 1279, &cfg, &mut ctl).unwrap();
    let ckpt = cfg.checkpoint.unwrap();
    let proof = cfg.proof_dir.unwrap();
    let snapshot = || {
        let mut v = Vec::new();
        for d in [
            ckpt.parent().unwrap().to_path_buf(),
            ProofStore::residue_dir(&proof, 1279),
        ] {
            for e in std::fs::read_dir(d).unwrap() {
                let p = e.unwrap().path();
                // Lock files are empty and created on demand.
                if p.extension().is_some_and(|x| x == "lock") || p.ends_with(".lock") {
                    continue;
                }
                v.push((p.display().to_string(), std::fs::read(&p).unwrap()));
            }
        }
        v.sort();
        v
    };
    let held = ProofStore::open(&proof, ProofSchedule::new(1279, 4).unwrap()).unwrap();
    let before = snapshot();
    let o = cli(&["resume", "--checkpoint", s(&ckpt), "--json"]);
    assert_eq!(
        o.status.code(),
        Some(6),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    assert!(json(&o)["message"].as_str().unwrap().contains(".lock"));
    assert_eq!(snapshot(), before, "nothing was renamed or written");
    drop(held);
    let o = cli(&["resume", "--checkpoint", s(&ckpt), "--json"]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    assert_eq!(json(&o)["report"]["evidence"]["proof_stage"], "verified");
}

/// RFC 4180 records: quoted fields may hold commas, quotes ("" inside), CR,
/// and LF. Panics on a quote in an unquoted field.
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let (mut records, mut record, mut field) = (Vec::new(), Vec::new(), String::new());
    let (mut quoted, mut was_quoted) = (false, false);
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        if quoted {
            match c {
                '"' if it.peek() == Some(&'"') => {
                    field.push('"');
                    it.next();
                }
                '"' => quoted = false,
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' => {
                assert!(field.is_empty() && !was_quoted, "stray quote");
                quoted = true;
                was_quoted = true;
            }
            ',' => {
                record.push(std::mem::take(&mut field));
                was_quoted = false;
            }
            '\n' => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
                was_quoted = false;
            }
            _ => {
                assert!(!was_quoted, "text after a closing quote");
                field.push(c);
            }
        }
    }
    assert!(
        !quoted && field.is_empty() && record.is_empty(),
        "unterminated record"
    );
    records
}

/// Arbitrary text in a plan (here names and argv) cannot
/// break the CSV: one record per observation, every record full width, and a
/// failed observation followed by a valid one keeps both.
#[test]
fn bench_csv_quotes_every_text_field() {
    let dir = tempfile::tempdir().unwrap();
    let nasty = "ll,bogus\r\n\"quoted\"\nend";
    let plan = serde_json::json!({"name": "csv, \"edge\"\ncases", "order": "sequential", "repeats": 1, "entries": [
        {"kind": "external", "name": "bad,name", "argv": ["false", nasty]},
        {"kind": "engine", "name": "good", "test": "ll", "exponent": 89, "backend": "cpu"},
        {"kind": "external", "name": "echo", "argv": ["echo", "a,b\n\"c\""]}]});
    let out = dir.path().join("r.csv");
    let o = bench(&plan, &out);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let records = parse_csv(&std::fs::read_to_string(&out).unwrap());
    let header = &records[0];
    let col = |name: &str| header.iter().position(|h| h == name).unwrap();
    assert_eq!(records.len(), 4, "header and three observations");
    for r in &records[1..] {
        assert_eq!(r.len(), header.len());
        assert_eq!(r[col("plan_name")], "csv, \"edge\"\ncases");
    }
    let (bad, good, ext) = (&records[1], &records[2], &records[3]);
    assert_eq!(bad[col("entry")], "bad,name");
    assert_eq!(
        bad[col("argv")],
        serde_json::to_string(&["false", nasty]).unwrap()
    );
    assert_eq!(bad[col("status")], "failed");
    assert_eq!(bad[col("entry_index")], "0");
    assert_eq!(good[col("status")], "completed");
    assert_eq!(good[col("verdict")], "Prime");
    assert_eq!(good[col("entry_index")], "1");
    assert_eq!(ext[col("argv")], r#"["echo","a,b\n\"c\""]"#);
    assert_eq!(ext[col("status")], "completed");
}

/// Production mode, retained artifacts, an external verifier (this binary's
/// own `verify`, standing in for an independent one), and a sampled power
/// command. The verifier must accept the proof and reject an altered copy.
#[test]
fn bench_production_runs_keep_verified_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let exe = env!("CARGO_BIN_EXE_mersenne-forge");
    let plan = serde_json::json!({
        "name": "evidence", "order": "sequential", "repeats": 1,
        "telemetry": {"argv": ["echo", "12.5 W"], "interval_ms": 20, "sensor": "fixed test value", "scope": "system"},
        "entries": [
            {"kind": "engine", "name": "prod", "test": "prp", "exponent": 1279, "backend": "cpu",
             "options": {"block_size": 32, "check_step": 128, "proof_power": 3, "mode": "production", "retain": true,
                         "verify_with": {"argv": [exe, "verify", "--proof", "{proof}"], "accept_text": ": accepted"}}},
            {"kind": "engine", "name": "steady", "test": "ll", "exponent": 1279, "backend": "dwt"}]});
    let out = dir.path().join("e.json");
    let o = bench(&plan, &out);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v = read_json(&out);
    assert_eq!(v["schema_version"], 2);
    assert_eq!(
        v["provenance"]["source_identity"],
        mersenne_forge::build_identity()
    );
    assert!(v["provenance"]["rustc"]
        .as_str()
        .unwrap()
        .starts_with("rustc "));
    let prod = &v["observations"][0];
    assert_eq!(prod["status"], "completed", "{prod}");
    assert_eq!(prod["mode"], "production");
    assert!(prod["reliability"]["gerbicz_passed"].as_u64().unwrap() > 0);
    assert!(prod["timings"]["checkpoint_io"].as_f64().unwrap() > 0.0);
    assert!(prod["timings"]["proof_points"].as_f64().unwrap() > 0.0);
    assert!(prod["unclassified_secs"].as_f64().unwrap() >= 0.0);
    let ev = &prod["external_verification"];
    assert_eq!(ev["status"], "verified");
    assert_eq!(
        (ev["accepted"].as_bool(), ev["altered_rejected"].as_bool()),
        (Some(true), Some(true))
    );
    assert_ne!(ev["altered_exit_code"], 0);
    assert_eq!(ev["verifier_sha3"].as_str().unwrap().len(), 64);
    // Retained files exist after the scratch directory is gone, with the
    // recorded hashes.
    let arts = prod["artifacts"].as_array().unwrap();
    assert_eq!(arts.len(), 2);
    for a in arts {
        let p = Path::new(a["path"].as_str().unwrap());
        assert_eq!(
            mersenne_forge::evidence::file_sha3(p).unwrap(),
            a["sha3"].as_str().unwrap()
        );
    }
    assert!(arts.iter().any(|a| a["name"] == "1279-3.proof"));
    // Power: a fixed 12.5 W reading integrates to 12.5 W times the span.
    for o in v["observations"].as_array().unwrap() {
        let pw = &o["power"];
        assert_eq!(pw["missing"], 0);
        assert_eq!(pw["scope"], "system");
        let e = o["energy_joules"].as_f64().unwrap();
        assert!((e - 12.5 * pw["span_secs"].as_f64().unwrap()).abs() < 1e-9);
    }
    assert_eq!(v["observations"][1]["mode"], "steady_state");
    assert!(v["observations"][1]["artifacts"]
        .as_array()
        .unwrap()
        .is_empty());

    // The same execution renders to CSV without rerunning; the hashes and
    // statuses carry over.
    let csv = dir.path().join("again.csv");
    let r = cli(&["bench-render", "--input", s(&out), "--output", s(&csv)]);
    assert_eq!(
        r.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&r.stderr)
    );
    let records = parse_csv(&std::fs::read_to_string(&csv).unwrap());
    let col = |n: &str| records[0].iter().position(|h| h == n).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(records[1][col("external_verification")], "verified");
    assert!(records[1][col("artifacts")].contains(arts[0]["sha3"].as_str().unwrap()));
    assert_eq!(records[1][col("schema_version")], "2");
}

/// A quota smaller than the proof keeps the hash but not the file; a
/// verifier that accepts everything fails the altered copy check.
#[test]
fn bench_quota_and_a_verifier_that_accepts_everything() {
    let dir = tempfile::tempdir().unwrap();
    let plan = serde_json::json!({
        "name": "q", "order": "sequential", "repeats": 1, "retain_max_bytes": 10,
        "entries": [{"kind": "engine", "name": "p", "test": "prp", "exponent": 127, "backend": "cpu",
            "options": {"block_size": 16, "check_step": 64, "proof_power": 2, "retain": true,
                        "verify_with": {"argv": ["echo", "{proof} ok"], "accept_text": "ok"}}}]});
    let out = dir.path().join("q.json");
    assert_eq!(bench(&plan, &out).status.code(), Some(0));
    let o = &read_json(&out)["observations"][0];
    assert_eq!(o["status"], "failed");
    assert_eq!(o["error"], "external verification failed");
    assert_eq!(o["external_verification"]["altered_rejected"], false);
    for a in o["artifacts"].as_array().unwrap() {
        assert!(a["path"].is_null());
        assert!(a["not_retained"].as_str().unwrap().contains("quota"));
        assert_eq!(a["sha3"].as_str().unwrap().len(), 64);
    }
    // Invalid evidence settings are usage errors before anything runs.
    for bad in [
        serde_json::json!({"mode": "production"}),
        serde_json::json!({"mode": "fast"}),
        serde_json::json!({"verify_with": {"argv": ["x"], "accept_text": "ok"}, "proof_power": 2}),
    ] {
        let plan = serde_json::json!({"name": "b", "order": "sequential", "repeats": 1,
            "entries": [{"kind": "engine", "name": "p", "test": "prp", "exponent": 127, "backend": "cpu", "options": bad}]});
        assert_eq!(
            bench(&plan, &dir.path().join("b.json")).status.code(),
            Some(2),
            "{bad}"
        );
    }
}

/// Paired analysis over a recorded plan: one pair per repeat, failures kept
/// in the cost accounting, and an interval from a fixed seed.
#[test]
fn bench_analyze_pairs_by_repeat() {
    let dir = tempfile::tempdir().unwrap();
    let plan = serde_json::json!({"name": "a", "order": "randomized", "seed": 5, "repeats": 5, "warmups": 1,
        "entries": [
            {"kind": "engine", "name": "cpu", "test": "ll", "exponent": 521, "backend": "cpu"},
            {"kind": "engine", "name": "dwt", "test": "ll", "exponent": 521, "backend": "dwt"},
            {"kind": "external", "name": "bad", "argv": ["false"]}]});
    let out = dir.path().join("a.json");
    assert_eq!(bench(&plan, &out).status.code(), Some(0));
    let run = |b: &str, c: &str| {
        cli(&[
            "bench-analyze",
            "--input",
            s(&out),
            "--baseline",
            b,
            "--candidate",
            c,
        ])
    };
    let o = run("0", "1");
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["usable_pairs"], 5);
    let [lo, hi] = [
        v["bootstrap"]["interval_95"][0].as_f64().unwrap(),
        v["bootstrap"]["interval_95"][1].as_f64().unwrap(),
    ];
    let m = v["median_ratio"].as_f64().unwrap();
    assert!(lo <= m && m <= hi);
    assert_eq!(
        o.stdout,
        run("0", "1").stdout,
        "the same seed gives the same report"
    );
    let v: serde_json::Value = serde_json::from_slice(&run("0", "2").stdout).unwrap();
    assert_eq!(v["usable_pairs"], 0);
    assert_eq!(
        v["candidate"]["cost"]["not_completed"]
            .as_array()
            .unwrap()
            .len(),
        5
    );
    assert!(v["median_ratio"].is_null());
    assert_eq!(run("0", "0").status.code(), Some(2));
    assert_eq!(run("0", "9").status.code(), Some(2));
}

#[test]
fn tune_compares_dwt_variants_and_they_agree() {
    let o = cli(&[
        "tune",
        "--exponent",
        "21701",
        "--backend",
        "dwt",
        "--iters",
        "20",
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let rows = v["variants"].as_array().unwrap();
    // The default and the packed transform, plus threaded variants on a
    // machine with more than one core.
    assert!(rows.len() >= 2, "{rows:?}");
    assert!(rows.iter().all(|r| r["agrees_with_default"] == true));
    assert!(v["fastest"]["ms_per_iteration"].as_f64().unwrap() > 0.0);
}

/// `status --serve` answers on loopback with the page and the plain text,
/// and refuses a public address.
#[test]
fn status_serves_a_local_page() {
    use std::io::{BufRead, Read, Write};
    let dir = tempfile::tempdir().unwrap();
    let todo = dir.path().join("worktodo.txt");
    std::fs::write(&todo, "PRP=127\n<script>\n").unwrap();
    let exe = env!("CARGO_BIN_EXE_mersenne-forge");
    let o = std::process::Command::new(exe)
        .args(["status", "--worktodo", s(&todo), "--serve", "0.0.0.0:0"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    let mut child = std::process::Command::new(exe)
        .args(["status", "--worktodo", s(&todo), "--serve", "127.0.0.1:0"])
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    std::io::BufReader::new(child.stderr.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let url = line
        .split("http://")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap();
    let (addr, page_path) = url.split_once('/').unwrap();
    let page_path = format!("/{page_path}");
    let get = |path: &str, host: &str| {
        let mut c = std::net::TcpStream::connect(addr).unwrap();
        write!(c, "GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n").unwrap();
        let mut r = String::new();
        c.read_to_string(&mut r).unwrap();
        r
    };
    let page = get(&page_path, addr);
    let text = get(&format!("{page_path}status.txt"), addr);
    let missing = get("/", addr);
    let rebound = get(&page_path, "attacker.example:80");
    let _ = child.kill();
    let _ = child.wait();
    assert!(page.starts_with("HTTP/1.1 200"), "{page}");
    assert!(page.contains("<pre>") && page.contains("PRP=127"));
    assert!(page.contains("1 runnable"));
    assert!(text.starts_with("HTTP/1.1 200") && text.contains("1 runnable"));
    assert!(missing.starts_with("HTTP/1.1 404"), "the token is required");
    assert!(
        rebound.starts_with("HTTP/1.1 421"),
        "a foreign Host is refused"
    );
}
