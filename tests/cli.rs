// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! CLI integration: exit codes and machine-readable JSON.

use std::path::Path;
use std::process::{Command, Output};

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
    assert_eq!(json(&o)["status"], "accepted");
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
        .all(|o| o["verdict"] == "ProbablePrime"));
    let csv = dir.path().join("out.csv");
    assert_eq!(
        cli(&["bench", "--plan", s(&plan), "--output", s(&csv)])
            .status
            .code(),
        Some(0)
    );
    assert_eq!(std::fs::read_to_string(&csv).unwrap().lines().count(), 10);
}
