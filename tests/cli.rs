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
    assert_eq!(json(&o)["status"], "rejected");
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
            {"kind":"engine","name":"bad","test":"prp","exponent":128,"backend":"cpu"}]}"#,
    )
    .unwrap();
    let out = dir.path().join("out.json");
    // A preexisting scratch directory is reported, never reused or deleted.
    let mine = dir.path().join("out.work").join("run-0");
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
        .filter(|o| o["entry"] == "prp-127" && o["sequence"] != 0)
        .all(|o| o["verdict"] == "ProbablePrime"
            && o["backend"]["kind"] == "cpu"
            && o["proof_stage"] == "not_requested"
            && o["check_step"] == 20_000));
    assert_eq!(obs[0]["status"], "failed");
    assert!(obs[0]["error"].as_str().unwrap().contains("run-0"));
    assert_eq!(std::fs::read(mine.join("keep")).unwrap(), b"user data");
    for o in obs
        .iter()
        .filter(|o| o["entry"] == "true" && o["sequence"] != 0)
    {
        assert!(Path::new(o["stdout"].as_str().unwrap()).is_file());
        assert!(Path::new(o["stderr"].as_str().unwrap()).is_file());
    }
    std::fs::remove_dir_all(dir.path().join("out.work")).unwrap();
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
    while !dir.path().join("out.0.stdout").exists() {
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
        "--json",
    ]);
    assert_eq!(o.status.code(), Some(5));
    let v = json(&o);
    assert_eq!(v["report"]["result"]["verdict"], "probable_prime");
    assert_eq!(v["report"]["evidence"]["proof_stage"], "failed");
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
