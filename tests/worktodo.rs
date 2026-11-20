// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Worktodo lines and the `work` command. The accepted line forms are the
//! examples and rules of the pinned upstream parser (`Worktodo.cpp`).

use std::path::Path;
use std::process::Command;

use mersenne_forge::worktodo::{parse_line, Assignment, WorkKind};

fn ok(line: &str) -> Assignment {
    parse_line(line).unwrap().unwrap()
}

#[test]
fn upstream_line_forms() {
    let a = ok("PRP=FEEE9DCD59A0855711265C1165C4C693,1,2,124647911,-1,77,0\r\n");
    assert_eq!(a.kind, WorkKind::Prp);
    assert_eq!(a.exponent, 124_647_911);
    assert_eq!(a.aid.as_deref(), Some("FEEE9DCD59A0855711265C1165C4C693"));
    assert_eq!(
        a.line,
        "PRP=FEEE9DCD59A0855711265C1165C4C693,1,2,124647911,-1,77,0"
    );
    let a = ok("DoubleCheck=E0F583710728343C61643028FBDBA0FB,70198703,75,1");
    assert_eq!((a.kind, a.exponent), (WorkKind::Ll, 70_198_703));
    assert_eq!(ok("PRP=118063003").aid, None);
    assert_eq!(ok("PRPDC=N/A,1,2,4423,-1,60,0").exponent, 4423);
    assert_eq!(ok("Test=,4423").exponent, 4423);
    assert_eq!(ok("Test=1,2,4423,-1").exponent, 4423);
    let a = ok("PRP=1,2,4423,-1,60,1.3");
    assert_eq!((a.tf_bits, a.tests_saved), (Some(60.0), Some(1.3)));
    assert!(a.wants_pm1());
    assert!(!ok("PRP=1,2,4423,-1,60,0,3,1").wants_pm1());
    // Cofactor lines, as PrimeNet and AutoPrimeNet write them.
    let a = ok("PRP=N/A,1,2,23,-1,75,0,3,5,\"47\"");
    assert_eq!(
        (a.residue_type, a.known_factors.clone()),
        (5, vec!["47".to_string()])
    );
    let a = ok("PRP=1,2,29,-1,75,0,\"233,1103\"");
    assert_eq!(a.residue_type, 5);
    assert_eq!(a.known_factors, ["233", "1103"]);
    assert_eq!(ok("PRP=1,2,31,-1,75,0,3,3").residue_type, 3);
    assert_eq!(
        ok("B1=50000,B2=5000000;PRP=1,2,4423,-1,75,0").exponent,
        4423
    );
    assert_eq!(ok("prp=1,2,4423,-1").exponent, 4423);
    for skip in [
        "",
        "   ",
        "# comment",
        "; comment",
        "[Worker #1]",
        "// comment",
    ] {
        assert_eq!(parse_line(skip).unwrap(), None);
    }
    for bad in [
        "Cert=B2EE67DC0A514753E488794C9DD6F6BD,1,2,82997591,-1,162105",
        "Pminus1=B2EE67DC0A514753E488794C9DD6F6BD,1,2,82997591,-1,1000000,30000000",
        "Pfactor=B2EE67DC0A514753E488794C9DD6F6BD,1,2,82997591,-1,76,1.3",
        "PRP=1,2,4423,1",  // 2^n + 1
        "PRP=3,2,4423,-1", // k * 2^n - 1
        "PRP=1,3,4423,-1", // base 3
        "PRP=4425",        // composite
        "Test=4423x",
        "Test=3,2,1279,-1",              // 3*2^1279-1
        "PRPDC=1,2,1279,-1,75,0,2,1",    // base 2
        "PRP=1,2,1279,-1,75,0,3,5",      // residue type 5
        "PRP=1,2,23,-1,75,0,3,2",        // residue type 2
        "PRP=1,2,23,-1,75,0,3,1,\"47\"", // known factors, type 1
        "PRP=1,2,23,-1,75,0,3,5,\"7\"",  // 7 does not divide M23
        "Test=23,\"47\"",                // LL has no known factors
        "PRP=2",                         // base 3 needs p >= 3
        "PRP=+127",                      // signed exponent
        "Test=127,=x",                   // second '='
        "garbage",
    ] {
        assert!(parse_line(bad).is_err(), "{bad}");
    }
}

fn work(dir: &Path, extra: &[&str]) -> std::process::Output {
    let s = |p: &Path| p.to_str().unwrap().to_string();
    let mut args = vec![
        "work".to_string(),
        "--worktodo".into(),
        s(&dir.join("worktodo.txt")),
        "--results".into(),
        s(&dir.join("results.json.txt")),
        "--checkpoint-dir".into(),
        s(&dir.join("ckpt")),
        "--proof-dir".into(),
        s(&dir.join("proof")),
        "--proof-power".into(),
        "2".into(),
    ];
    args.extend(extra.iter().map(|x| x.to_string()));
    Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
        .args(args)
        .output()
        .unwrap()
}

fn results(dir: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(dir.join("results.json.txt"))
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn work_runs_assignments_and_keeps_what_it_cannot_run() {
    let dir = tempfile::tempdir().unwrap();
    let todo = dir.path().join("worktodo.txt");
    std::fs::write(
        &todo,
        "# queue\n\
         Pminus1=B2EE67DC0A514753E488794C9DD6F6BD,1,2,82997591,-1,1000000,30000000\n\
         PRP=0123456789ABCDEF0123456789ABCDEF,1,2,1279,-1,60,0\n\
         DoubleCheck=89\n\
         PRP=1,2,1277,-1,60,0\n",
    )
    .unwrap();
    let o = work(dir.path(), &[]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(&todo).unwrap(),
        "# queue\nPminus1=B2EE67DC0A514753E488794C9DD6F6BD,1,2,82997591,-1,1000000,30000000\n"
    );
    let r = results(dir.path());
    assert_eq!(r.len(), 3);
    let prp = &r[0];
    assert_eq!(prp["status"], "P");
    assert_eq!(prp["worktype"], "PRP-3");
    assert_eq!(prp["exponent"], 1279);
    assert_eq!(prp["aid"], "0123456789ABCDEF0123456789ABCDEF");
    assert_eq!(prp["res64"], "0000000000000001");
    assert_eq!(prp["residue-type"], 1);
    assert_eq!(prp["res2048"].as_str().unwrap().len(), 512);
    let proof = dir.path().join("proof").join("1279-2.proof");
    assert_eq!(
        prp["proof"]["md5"],
        mersenne_forge::worktodo::file_md5(&proof).unwrap()
    );
    assert_eq!(prp["proof"]["power"], 2);
    assert_eq!(
        (r[1]["worktype"].as_str(), r[1]["status"].as_str()),
        (Some("LL"), Some("P"))
    );
    assert_eq!(r[1]["shift-count"], 0);
    assert_eq!(
        (r[2]["exponent"].as_u64(), r[2]["status"].as_str()),
        (Some(1277), Some("C"))
    );
    // The proof file stays; checkpoints and proof points are cleaned up.
    assert!(proof.is_file());
    assert!(!dir.path().join("proof").join("M1279.residues").exists());
    let left: Vec<_> = std::fs::read_dir(dir.path().join("ckpt"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("M1279-"))
        .collect();
    assert!(left.is_empty(), "{left:?}");
    // A results line torn by a crash is closed before the next one.
    let res = dir.path().join("results.json.txt");
    let mut torn = std::fs::read_to_string(&res).unwrap();
    torn.push_str("{\"status\":\"C\",\"expo");
    std::fs::write(&res, torn).unwrap();
    std::fs::write(&todo, "Test=127\n").unwrap();
    assert_eq!(work(dir.path(), &[]).status.code(), Some(0));
    let text = std::fs::read_to_string(&res).unwrap();
    let last: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert_eq!(last["exponent"], 127);
    // The fragment moved aside whole; every results line is complete JSON.
    assert_eq!(
        std::fs::read_to_string(dir.path().join("results.json.txt.torn")).unwrap(),
        "{\"status\":\"C\",\"expo\n"
    );
    assert_eq!(text.lines().count(), 4);
    std::fs::write(
        &res,
        text.lines()
            .take(3)
            .map(|l| format!("{l}\n"))
            .collect::<String>(),
    )
    .unwrap();
    // The outbox is idempotent: an assignment whose result is already there
    // (as after a crash between the two writes) is removed, not run again.
    std::fs::write(
        &todo,
        "PRP=0123456789ABCDEF0123456789ABCDEF,1,2,1279,-1,60,0\n",
    )
    .unwrap();
    let o = work(dir.path(), &[]);
    assert_eq!(o.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&o.stderr).contains("already in"));
    assert_eq!(results(dir.path()).len(), 3);
    assert_eq!(std::fs::read_to_string(&todo).unwrap(), "");
}

/// A session stopped by its budget keeps the assignment and its checkpoint;
/// the next session resumes it and reports the same result as a clean run.
#[test]
fn an_interrupted_assignment_resumes_in_the_next_session() {
    let dir = tempfile::tempdir().unwrap();
    let todo = dir.path().join("worktodo.txt");
    std::fs::write(&todo, "PRP=1,2,21701,-1,60,0\n").unwrap();
    let o = work(dir.path(), &["--budget", "1s"]);
    assert_eq!(
        o.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    let ckpts: Vec<String> = std::fs::read_dir(dir.path().join("ckpt"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        ckpts
            .iter()
            .any(|n| n.starts_with("M21701-prp-") && n.ends_with(".ckpt")),
        "{ckpts:?}"
    );
    assert!(results(dir.path()).is_empty());
    let o = work(dir.path(), &[]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let r = results(dir.path());
    assert_eq!(r.len(), 1);
    assert_eq!(
        (r[0]["status"].as_str(), r[0]["res64"].as_str()),
        (Some("P"), Some("0000000000000001"))
    );
}

/// A line this program must not run as given (here a PRP residue type it
/// does not compute) stays in place, and later assignments still run.
#[test]
fn an_unrunnable_assignment_does_not_block_the_queue() {
    let dir = tempfile::tempdir().unwrap();
    let todo = dir.path().join("worktodo.txt");
    std::fs::write(&todo, "PRP=1,2,1279,-1,75,0,3,2\nTest=127\n").unwrap();
    let o = work(dir.path(), &[]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("residue type"));
    assert_eq!(
        std::fs::read_to_string(&todo).unwrap(),
        "PRP=1,2,1279,-1,75,0,3,2\n"
    );
    assert_eq!(results(dir.path())[0]["exponent"], 127);
}

/// A PRP line that asks for P-1 gets it first: a PM1 result, then the PRP
/// test of the rewritten line.
#[test]
fn p_minus_1_runs_before_a_prp_test_that_asks_for_it() {
    let dir = tempfile::tempdir().unwrap();
    let todo = dir.path().join("worktodo.txt");
    std::fs::write(&todo, "PRP=1,2,1279,-1,60,1\n").unwrap();
    assert_eq!(work(dir.path(), &[]).status.code(), Some(0));
    let r = results(dir.path());
    assert_eq!(
        (r[0]["worktype"].as_str(), r[0]["status"].as_str()),
        (Some("PM1"), Some("NF"))
    );
    assert_eq!(r[1]["worktype"], "PRP-3");
    assert_eq!(std::fs::read_to_string(&todo).unwrap(), "");
}

#[test]
fn factoring_lines_report_factors() {
    let dir = tempfile::tempdir().unwrap();
    let todo = dir.path().join("worktodo.txt");
    std::fs::write(&todo, "Factor=67,27,28\nPminus1=1,2,67,-1,2677,0\n").unwrap();
    assert_eq!(work(dir.path(), &[]).status.code(), Some(0));
    let r = results(dir.path());
    assert_eq!(
        (r[0]["worktype"].as_str(), r[0]["status"].as_str()),
        (Some("TF"), Some("F"))
    );
    assert_eq!(r[0]["factors"][0], "193707721");
    assert_eq!(r[0]["rangecomplete"], true);
    assert_eq!(
        (r[1]["worktype"].as_str(), r[1]["factors"][0].as_str()),
        (Some("PM1"), Some("193707721"))
    );
    assert_eq!(std::fs::read_to_string(&todo).unwrap(), "");
}

/// Without an assignment ID, a second test of the same exponent is new work:
/// only the completion journal marks a line done.
#[test]
fn a_double_check_without_an_id_still_runs() {
    let dir = tempfile::tempdir().unwrap();
    let todo = dir.path().join("worktodo.txt");
    std::fs::write(&todo, "Test=127\n").unwrap();
    assert_eq!(work(dir.path(), &[]).status.code(), Some(0));
    std::fs::write(&todo, "DoubleCheck=127\n").unwrap();
    assert_eq!(work(dir.path(), &[]).status.code(), Some(0));
    assert_eq!(results(dir.path()).len(), 2);
    // The same line again is recognized from the journal and not rerun.
    std::fs::write(&todo, "DoubleCheck=127\n").unwrap();
    assert_eq!(work(dir.path(), &[]).status.code(), Some(0));
    assert_eq!(results(dir.path()).len(), 2);
}

#[test]
fn work_refuses_paths_that_would_lose_work() {
    let dir = tempfile::tempdir().unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
        .args([
            "work",
            "--worktodo",
            "w.txt",
            "--results",
            "w.txt",
            "--checkpoint-dir",
            "c",
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    let o = Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
        .args([
            "work",
            "--worktodo",
            "w.txt",
            "--results",
            "r.txt",
            "--checkpoint-dir",
            "c",
            "--proof-power",
            "2",
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn check_factor_command() {
    let run = |q: &str| {
        Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
            .args(["check-factor", "--exponent", "11", "--factor", q, "--json"])
            .output()
            .unwrap()
    };
    let o = run("23");
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(
        (v["divides"].as_bool(), v["form_ok"].as_bool()),
        (Some(true), Some(true))
    );
    assert_eq!(run("47").status.code(), Some(7));
    assert_eq!(run("abc").status.code(), Some(2));
}

/// Two workers under `supervise`, each with its own queue, both finish and
/// write their status files.
#[test]
fn supervise_runs_one_worker_per_queue() {
    let dir = tempfile::tempdir().unwrap();
    let mut workers = Vec::new();
    for (i, p) in [(0, 127), (1, 521)] {
        let d = dir.path().join(format!("w{i}"));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("worktodo.txt"), format!("Test={p}\n")).unwrap();
        workers.push(serde_json::json!({
            "worktodo": d.join("worktodo.txt"),
            "results": d.join("results.txt"),
            "checkpoint_dir": d.join("ckpt"),
            "status_file": d.join("status.json"),
        }));
    }
    let cfg = dir.path().join("workers.json");
    std::fs::write(&cfg, serde_json::to_string(&workers).unwrap()).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
        .args(["supervise", "--config"])
        .arg(&cfg)
        .output()
        .unwrap();
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    for (i, p) in [(0, 127), (1, 521)] {
        let d = dir.path().join(format!("w{i}"));
        let r = std::fs::read_to_string(d.join("results.txt")).unwrap();
        assert!(r.contains(&format!("\"exponent\":{p}")), "{r}");
        let st: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(d.join("status.json")).unwrap()).unwrap();
        assert_eq!(
            (st["state"].as_str(), st["exit_code"].as_u64()),
            (Some("exited"), Some(0))
        );
    }
    // A shared queue is refused before anything starts.
    let same = serde_json::json!([workers[0].clone(), workers[0].clone()]);
    std::fs::write(&cfg, same.to_string()).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_mersenne-forge"))
        .args(["supervise", "--config"])
        .arg(&cfg)
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
}

/// A CERT line runs from the start value file next to the worktodo file,
/// reports the hash of the squared value, and removes the file; a cofactor
/// PRP line reports type 5 with its known factors.
#[test]
fn cert_and_cofactor_lines_run() {
    use mersenne_forge::residue::Residue;
    let dir = tempfile::tempdir().unwrap();
    let todo = dir.path().join("worktodo.txt");
    let start = Residue::from_u64(127, 0xdead_beef);
    let cert = dir.path().join("M127.cert");
    std::fs::write(&cert, start.to_proof_bytes()).unwrap();
    std::fs::write(
        &todo,
        "PRP=1,2,23,-1,60,0,\"47\"\nCert=0123456789ABCDEF0123456789ABCDEF,1,2,127,-1,500\n",
    )
    .unwrap();
    let o = work(dir.path(), &[]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let r = results(dir.path());
    // CERT first.
    assert_eq!(r[0]["worktype"], "Cert");
    let want = mersenne_forge::cert::hash(&mersenne_forge::oracle::square_n(&start, 500));
    assert_eq!(r[0]["sha3-hash"].as_str(), Some(want.as_str()));
    assert!(!cert.exists());
    assert_eq!(r[1]["worktype"], "PRP-3");
    assert_eq!(r[1]["status"], "P", "M23 / 47 = 178481 is prime");
    assert_eq!(r[1]["residue-type"], 5);
    assert_eq!(r[1]["known-factors"][0], "47");
    assert_eq!(std::fs::read_to_string(&todo).unwrap(), "");
}
