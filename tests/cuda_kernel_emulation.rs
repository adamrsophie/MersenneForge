// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! Host emulation of the CUDA kernels.
//!
//! The exact embedded kernel source (`backend::cuda::KERNEL_SOURCE`) is
//! compiled by the host C++ compiler (`c++` on PATH) behind a shim header
//! that defines `__global__`, `__device__`, `__forceinline__`, `__umul64hi`
//! (through `unsigned __int128`), and the thread and block index variables.
//! A generated driver runs each launch of the shared launch plan by looping
//! over every thread index of the rounded-up grid, in forward or reversed
//! order, with exactly sized heap buffers, under AddressSanitizer and
//! UndefinedBehaviorSanitizer. Outputs are compared word for word against the
//! Rust model, and complete squarings, multiplications, and LL steps against
//! `num-bigint`.
//!
//! What this proves: the kernel arithmetic and indexing, as C++ semantics on
//! a CPU, for the launch sequence the CUDA host submits. What it does not
//! prove: that NVRTC accepts the source, the launch configuration on a GPU,
//! device memory management, concurrency between GPU threads, or any device
//! behavior. If no host compiler is found this test fails; it never skips.

#![cfg(feature = "cuda")]

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::OnceLock;

use mersenne_forge::backend::cuda::model::{self, Arg, Buf, Buffers, Kernel, Launch, Param, Plan};
use mersenne_forge::backend::cuda::KERNEL_SOURCE;
use mersenne_forge::residue::{modulus, Residue};
use num_bigint::BigUint;

const SHIM: &str = r#"
#pragma once
#define __global__
#define __device__
#define __forceinline__ inline
struct mf_dim3 { unsigned int x, y, z; };
static mf_dim3 threadIdx, blockIdx, blockDim, gridDim;
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return (unsigned long long)(((unsigned __int128)a * b) >> 64);
}
"#;

const DRIVER_HEAD: &str = r#"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "shim.h"
#include "kernels.cu"

static unsigned long long* bufs[8];
static unsigned long long lens[8];
struct Arg { bool ptr; unsigned long long v; };
static std::vector<Arg> args;

static unsigned long long* P(size_t k) {
    if (k >= args.size() || !args[k].ptr || args[k].v >= 8 || !bufs[args[k].v]) { std::fprintf(stderr, "bad pointer arg %zu\n", k); std::abort(); }
    return bufs[args[k].v];
}
static unsigned long long V(size_t k) {
    if (k >= args.size() || args[k].ptr) { std::fprintf(stderr, "bad scalar arg %zu\n", k); std::abort(); }
    return args[k].v;
}
"#;

const DRIVER_MAIN: &str = r#"
int main() {
    std::ios::sync_with_stdio(false);
    std::string cmd;
    while (std::cin >> cmd) {
        if (cmd == "S") {
            unsigned long long id, len;
            std::cin >> id >> len;
            delete[] bufs[id];
            bufs[id] = new unsigned long long[len];
            lens[id] = len;
            for (unsigned long long i = 0; i < len; i++) std::cin >> bufs[id][i];
        } else if (cmd == "G") {
            unsigned long long id;
            std::cin >> id;
            std::cout << lens[id];
            for (unsigned long long i = 0; i < lens[id]; i++) std::cout << ' ' << bufs[id][i];
            std::cout << '\n' << std::flush;
        } else if (cmd == "L") {
            std::string name;
            unsigned int grid, block, rev;
            size_t na;
            std::cin >> name >> grid >> block >> rev >> na;
            args.assign(na, Arg{false, 0});
            for (size_t k = 0; k < na; k++) {
                std::string tok;
                std::cin >> tok;
                args[k].ptr = tok[0] == 'b';
                args[k].v = std::strtoull(tok.c_str() + 1, nullptr, 10);
            }
            blockDim = mf_dim3{block, 1, 1};
            gridDim = mf_dim3{grid, 1, 1};
            unsigned long long total = (unsigned long long)grid * block;
            for (unsigned long long s = 0; s < total; s++) {
                unsigned long long g = rev ? total - 1 - s : s;
                blockIdx = mf_dim3{(unsigned int)(g / block), 0, 0};
                threadIdx = mf_dim3{(unsigned int)(g % block), 0, 0};
                dispatch(name);
            }
            std::cout << "ok\n" << std::flush;
        } else if (cmd == "Q") {
            return 0;
        } else if (cmd == "E") {
            // Harness self-test: answer, then exit with the given status.
            int status = 0;
            std::cin >> status;
            std::cout << "ok\n" << std::flush;
            return status;
        } else {
            std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
            return 2;
        }
    }
    return 0;
}
"#;

/// Dispatch generated from the Rust kernel signatures: a mismatch in arity
/// or in pointer versus scalar kind against kernels.cu is a compile error.
fn dispatch_source() -> String {
    let mut s = String::from("static void dispatch(const std::string& name) {\n");
    for k in Kernel::ALL {
        let call: Vec<String> = k
            .params()
            .iter()
            .enumerate()
            .map(|(i, p)| match p {
                Param::Ptr => format!("P({i})"),
                Param::Scalar => format!("V({i})"),
            })
            .collect();
        s += &format!(
            "    if (name == \"{}\") {{ {}({}); return; }}\n",
            k.name(),
            k.name(),
            call.join(", ")
        );
    }
    s +=
        "    std::fprintf(stderr, \"unknown kernel %s\\n\", name.c_str());\n    std::abort();\n}\n";
    s
}

/// Compile the driver once per test process.
fn driver() -> &'static PathBuf {
    static EXE: OnceLock<PathBuf> = OnceLock::new();
    EXE.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cuda_kernel_emulation");
        std::fs::create_dir_all(&dir).expect("create emulation directory");
        std::fs::write(dir.join("shim.h"), SHIM).unwrap();
        std::fs::write(dir.join("kernels.cu"), KERNEL_SOURCE).unwrap();
        std::fs::write(dir.join("driver.cpp"), format!("{DRIVER_HEAD}{}{DRIVER_MAIN}", dispatch_source())).unwrap();
        let exe = dir.join("driver");
        let out = Command::new("c++")
            .args(["-std=c++17", "-O1", "-g", "-Wall", "-Wextra"])
            .args(["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"])
            .arg("-I")
            .arg(&dir)
            .arg("-o")
            .arg(&exe)
            .arg(dir.join("driver.cpp"))
            .output()
            .unwrap_or_else(|e| {
                panic!("host emulation needs a C++ compiler: running `c++` failed ({e}); install one (clang++ or g++) as `c++` on PATH")
            });
        assert!(
            out.status.success(),
            "compiling the CUDA kernel source as host C++ failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        if !out.stderr.is_empty() {
            eprintln!("host compiler warnings:\n{}", String::from_utf8_lossy(&out.stderr));
        }
        exe
    })
}

fn buf_id(b: Buf) -> usize {
    match b {
        Buf::X => 0,
        Buf::Src => 1,
        Buf::T => 2,
        Buf::T2 => 3,
        Buf::Tw => 4,
        Buf::Z => 5,
        Buf::A => 6,
    }
}

const ALL_BUFS: [Buf; 7] = [Buf::X, Buf::Src, Buf::T, Buf::T2, Buf::Tw, Buf::Z, Buf::A];

/// One emulated device: a driver process holding the buffers.
struct Emu {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    finished: bool,
}

impl Emu {
    fn new() -> Emu {
        let mut child = Command::new(driver())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start the emulation driver");
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        Emu {
            child,
            input,
            output,
            finished: false,
        }
    }

    /// Stop the driver and require a successful exit. A sanitizer report at
    /// shutdown (for example a leak) or any nonzero status fails the test.
    fn finish(mut self) {
        self.finished = true;
        writeln!(self.input, "Q").unwrap();
        self.input.flush().unwrap();
        let status = self.child.wait().expect("wait for the emulation driver");
        assert!(
            status.success(),
            "emulation driver exited with {status} after its last response (details on stderr)"
        );
    }

    fn line(&mut self) -> String {
        let mut s = String::new();
        let n = self.output.read_line(&mut s).unwrap();
        assert!(
            n > 0,
            "emulation driver exited (a sanitizer report or abort is on stderr)"
        );
        s
    }

    fn set(&mut self, b: Buf, data: &[u64]) {
        let mut s = format!("S {} {}", buf_id(b), data.len());
        for v in data {
            s += &format!(" {v}");
        }
        writeln!(self.input, "{s}").unwrap();
    }

    fn get(&mut self, b: Buf) -> Vec<u64> {
        writeln!(self.input, "G {}", buf_id(b)).unwrap();
        self.input.flush().unwrap();
        let line = self.line();
        let mut it = line.split_whitespace().map(|t| t.parse::<u64>().unwrap());
        let len = it.next().unwrap() as usize;
        let v: Vec<u64> = it.collect();
        assert_eq!(v.len(), len);
        v
    }

    fn load(&mut self, bufs: &Buffers) {
        let mut b = bufs.clone();
        for id in ALL_BUFS {
            let data = b.get_mut(id).clone();
            self.set(id, &data);
        }
    }

    fn launch(&mut self, l: &Launch, block: u32, reverse: bool) {
        let grid = l.threads.div_ceil(u64::from(block));
        let mut s = format!(
            "L {} {grid} {block} {} {}",
            l.kernel.name(),
            u8::from(reverse),
            l.args.len()
        );
        for a in &l.args {
            match a {
                Arg::Buf(b) => s += &format!(" b{}", buf_id(*b)),
                Arg::U64(v) => s += &format!(" u{v}"),
            }
        }
        writeln!(self.input, "{s}").unwrap();
        self.input.flush().unwrap();
        assert_eq!(self.line().trim(), "ok", "launch of {}", l.kernel.name());
    }
}

/// Cleanup for paths that do not reach `finish`. While unwinding it only
/// reaps the child; otherwise a failed exit still fails the test.
impl Drop for Emu {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let _ = writeln!(self.input, "Q");
        let _ = self.input.flush();
        let status = self.child.wait();
        if !std::thread::panicking() {
            let status = status.expect("wait for the emulation driver");
            assert!(status.success(), "emulation driver exited with {status}");
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn random(p: u32, rng: &mut Rng) -> BigUint {
    let limbs: Vec<u32> = (0..(p as usize).div_ceil(32))
        .map(|_| rng.next() as u32)
        .collect();
    BigUint::from_slice(&limbs) % modulus(p)
}

fn words(plan: &Plan, x: &BigUint) -> Vec<u64> {
    plan.to_words(&Residue::from_biguint(plan.p, x))
}

fn first_ptr(l: &Launch) -> Buf {
    match l.args[0] {
        Arg::Buf(b) => b,
        Arg::U64(_) => panic!("the first parameter is always the output pointer"),
    }
}

/// Run a launch sequence on the emulator and the model side by side and
/// compare the written buffer after every launch, alternating block sizes
/// that do not divide n and thread orders.
fn lockstep(plan: &Plan, emu: &mut Emu, model_bufs: &mut Buffers, seq: &[Launch]) {
    for (i, l) in seq.iter().enumerate() {
        let (block, reverse) = [(256, false), (97, true), (1, false), (33, true)][i % 4];
        emu.launch(l, block, reverse);
        model::execute(l, model_bufs);
        let out = first_ptr(l);
        let want = model_bufs.get_mut(out).clone();
        assert_eq!(
            emu.get(out),
            want,
            "p={} launch {i} {:?} differs from the model",
            plan.p,
            l.kernel
        );
    }
}

fn plans() -> Vec<Plan> {
    let mut v: Vec<Plan> = [12u32, 13, 31, 127, 1279, 9941, 86243]
        .iter()
        .map(|&p| model::plan(p).unwrap())
        .collect();
    // Exponents at the capacity edge of their length.
    for log_n in [2u32, 5, 9, 12] {
        let p = model::max_exponent_for_len(log_n).unwrap();
        v.push(model::plan_for_len(p, log_n).unwrap());
    }
    v
}

#[test]
fn every_kernel_matches_the_model_word_for_word() {
    let mut rng = Rng(21);
    for plan in plans() {
        let mut emu = Emu::new();
        let mut m = Buffers::new(&plan);
        m.x = words(&plan, &random(plan.p, &mut rng));
        m.src = words(&plan, &random(plan.p, &mut rng));
        emu.load(&m);
        lockstep(&plan, &mut emu, &mut m, &plan.square_launches(0));
        lockstep(&plan, &mut emu, &mut m, &plan.square_launches(-2));
        lockstep(&plan, &mut emu, &mut m, &plan.mul_launches());
        // Every buffer, including scratch and the twiddle table, agrees.
        for b in ALL_BUFS {
            assert_eq!(emu.get(b), m.get_mut(b).clone(), "p={} {b:?}", plan.p);
        }
        emu.finish();
    }
}

#[test]
fn worst_case_words_match_the_model() {
    for log_n in [2u32, 4, 8] {
        let p = model::max_exponent_for_len(log_n).unwrap();
        let plan = model::plan_for_len(p, log_n).unwrap();
        let wb = plan.word_bound as i64;
        for sign in [1i64, -1] {
            let mut emu = Emu::new();
            let mut m = Buffers::new(&plan);
            m.x = vec![(sign * wb) as u64; plan.n as usize];
            m.src = vec![(-sign * wb) as u64; plan.n as usize];
            emu.load(&m);
            lockstep(&plan, &mut emu, &mut m, &plan.square_launches(-2));
            m.x = vec![(sign * wb) as u64; plan.n as usize];
            emu.set(Buf::X, &m.x);
            lockstep(&plan, &mut emu, &mut m, &plan.mul_launches());
            emu.finish();
        }
    }
}

#[test]
fn thread_order_and_block_size_do_not_change_results() {
    let mut rng = Rng(23);
    let plan = model::plan(9941).unwrap();
    let x = words(&plan, &random(plan.p, &mut rng));
    let mut results = Vec::new();
    for (block, reverse) in [
        (1u32, false),
        (1, true),
        (7, false),
        (1000, true),
        (plan.n as u32, false),
    ] {
        let mut emu = Emu::new();
        let mut m = Buffers::new(&plan);
        m.x = x.clone();
        emu.load(&m);
        for l in plan.square_launches(-2) {
            emu.launch(&l, block, reverse);
        }
        results.push(emu.get(Buf::X));
        emu.finish();
    }
    assert!(results.windows(2).all(|w| w[0] == w[1]));
}

/// Complete operations through the emulator only, read back with the exact
/// host conversion, against num-bigint.
#[test]
fn emulated_operations_match_bigint() {
    let mut rng = Rng(29);
    for p in [12u32, 13, 89, 127, 607, 4423, 23209] {
        let plan = model::plan(p).unwrap();
        let n = modulus(p);
        let edges = [
            BigUint::from(0u32),
            BigUint::from(1u32),
            &n - 1u32,
            &n - 2u32,
            random(p, &mut rng),
        ];
        for x in &edges {
            let y = random(p, &mut rng);
            let mut emu = Emu::new();
            let mut m = Buffers::new(&plan);
            m.x = words(&plan, x);
            m.src = words(&plan, &y);
            emu.load(&m);
            let mut want = x.clone();
            for _ in 0..3 {
                for l in plan.square_launches(0) {
                    emu.launch(&l, 128, false);
                }
                want = (&want * &want) % &n;
            }
            for _ in 0..3 {
                for l in plan.square_launches(-2) {
                    emu.launch(&l, 128, true);
                }
                want = (&want * &want + &n - 2u32) % &n;
            }
            for l in plan.mul_launches() {
                emu.launch(&l, 64, false);
            }
            want = (&want * &y) % &n;
            let got = plan
                .from_words(&emu.get(Buf::X))
                .expect("words within the proven bound");
            assert_eq!(got.to_biguint(), want, "p={p}");
            emu.finish();
        }
    }
}

#[test]
fn emulated_complete_lucas_lehmer_runs() {
    for (p, prime) in [
        (13u32, true),
        (61, true),
        (89, true),
        (101, false),
        (127, true),
        (131, false),
        (521, true),
        (523, false),
    ] {
        let plan = model::plan(p).unwrap();
        let mut emu = Emu::new();
        let mut m = Buffers::new(&plan);
        m.x = words(&plan, &BigUint::from(4u32));
        emu.load(&m);
        let seq = plan.square_launches(-2);
        for _ in 0..p - 2 {
            for l in &seq {
                emu.launch(l, 256, false);
            }
        }
        let r = plan.from_words(&emu.get(Buf::X)).unwrap();
        assert_eq!(r.is_zero(), prime, "M{p}");
        emu.finish();
    }
}

/// The harness itself: a driver that answers correctly and then exits with a
/// failure status makes the enclosing test fail, through `finish` and
/// through `Drop`.
#[test]
fn a_failed_driver_exit_fails_the_test() {
    for explicit in [true, false] {
        let r = std::panic::catch_unwind(|| {
            let mut emu = Emu::new();
            writeln!(emu.input, "E 86").unwrap();
            emu.input.flush().unwrap();
            assert_eq!(emu.line().trim(), "ok");
            if explicit {
                emu.finish();
            }
        });
        let msg = r.expect_err("a nonzero driver exit must fail");
        let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(msg.contains("exit status: 86"), "{msg}");
    }
    // A clean exit passes.
    Emu::new().finish();
}
