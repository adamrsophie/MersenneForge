// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! The local web servers: `status --serve` (a read only report) and `ui` (a
//! control page). Both bind only loopback addresses, put every route under a
//! random per-run token that is printed at start, and refuse a request whose
//! Host header is not this address or localhost (which stops DNS
//! rebinding). `ui` also accepts one kind of write, appending a validated
//! line to the worktodo file, and refuses a POST whose Origin is another
//! site.
//!
//! ponytail: one request at a time, no keep-alive, bodies capped at 16 KiB;
//! a slow client delays the next one by at most the 5 s read timeout. Enough
//! for one person's browser on the same machine.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::{json, Value};

use crate::{fail, lck, Failure, EXIT_STATE, STOP};

/// The control page, served as is. Everything it shows comes from the JSON
/// routes, and it escapes every value before inserting it.
const UI_HTML: &str = include_str!("ui.html");

const MAX_REQUEST: usize = 16 * 1024;

pub struct Request {
    pub method: String,
    /// The path after `/<token>/`, without the query.
    pub path: String,
    pub query: String,
    pub body: Vec<u8>,
}

pub struct Response {
    pub code: &'static str,
    pub kind: &'static str,
    pub body: String,
}

impl Response {
    pub fn new(code: &'static str, kind: &'static str, body: impl Into<String>) -> Response {
        Response {
            code,
            kind,
            body: body.into(),
        }
    }
    fn json(code: &'static str, v: &Value) -> Response {
        Response::new(code, "application/json", v.to_string())
    }
    fn text(code: &'static str, body: &str) -> Response {
        Response::new(code, "text/plain; charset=utf-8", format!("{body}\n"))
    }
}

/// A random token for the URL: other local users and web pages cannot guess
/// it.
fn token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut t = String::new();
    for i in 0..2u64 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(i);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        h.write_u32(std::process::id());
        t.push_str(&format!("{:016x}", h.finish()));
    }
    t
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|l| {
        l.split_once(':')
            .filter(|(k, _)| k.trim().eq_ignore_ascii_case(name))
            .map(|(_, v)| v.trim())
    })
}

/// Serve `handle` on a loopback address until Ctrl-C. `flag` names the
/// option in messages; `what` is printed with the URL. With a `launcher`
/// path, a busy port falls back to a free one, a private page that
/// redirects to the URL is written there, and the browser opens it; the
/// page is removed when the server stops.
pub fn serve(
    addr: std::net::SocketAddr,
    flag: &str,
    what: &str,
    launcher: Option<PathBuf>,
    mut handle: impl FnMut(&Request) -> Response,
) -> Result<u8, Failure> {
    // The pages show local paths and queue contents: never on a public
    // interface.
    if !addr.ip().is_loopback() {
        return Err(fail(
            2,
            format!("{flag} {addr}: only loopback addresses are allowed"),
        ));
    }
    let listener = match std::net::TcpListener::bind(addr) {
        Err(e) if launcher.is_some() && e.kind() == std::io::ErrorKind::AddrInUse => {
            eprintln!("{addr} is in use; picking a free port");
            std::net::TcpListener::bind((addr.ip(), 0))
        }
        r => r,
    }
    .map_err(|e| fail(EXIT_STATE, format!("{flag} {addr}: {e}")))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| fail(EXIT_STATE, e.to_string()))?;
    let addr = listener.local_addr().unwrap_or(addr);
    let token = token();
    let hosts = [addr.to_string(), format!("localhost:{}", addr.port())];
    let prefix = format!("/{token}/");
    let url = format!("http://{addr}{prefix}");
    eprintln!("serving {what} at {url} (Ctrl-C to stop)");
    // A second secret in the private launcher page: the server returns it
    // only to requests that carry the token, so a later launch can tell
    // this server from another program squatting on the same port.
    let key = self::token();
    let launcher = launcher.filter(|page| open_browser(&url, &key, page));
    let key_header = match &launcher {
        Some(_) => format!("X-Launcher-Key: {key}\r\n"),
        None => String::new(),
    };
    while !STOP.load(Ordering::SeqCst) {
        let mut stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            Err(e) => {
                eprintln!("{flag}: {e}");
                continue;
            }
        };
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let r = respond(&mut stream, &prefix, &hosts, &mut handle);
        let head = format!(
            "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\n\
             X-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n\
             Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\n\
             {}Connection: close\r\n\r\n",
            r.0.code,
            r.0.kind,
            r.0.body.len(),
            if r.0.code.starts_with("200") { key_header.as_str() } else { "" }
        );
        let _ = stream.write_all(head.as_bytes());
        if !r.1 {
            let _ = stream.write_all(r.0.body.as_bytes());
        }
    }
    if let Some(f) = launcher {
        let _ = std::fs::remove_file(f);
    }
    Ok(0)
}

/// Read one request and answer it; the flag is true for HEAD.
fn respond(
    stream: &mut std::net::TcpStream,
    prefix: &str,
    hosts: &[String],
    handle: &mut impl FnMut(&Request) -> Response,
) -> (Response, bool) {
    let mut buf = [0u8; 4096];
    let mut req = Vec::new();
    let head_end = loop {
        if let Some(i) = req.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if req.len() >= MAX_REQUEST {
            return (
                Response::text("431 Request Header Fields Too Large", "too large"),
                false,
            );
        }
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => {
                return (
                    Response::text("400 Bad Request", "incomplete request"),
                    false,
                )
            }
            Ok(n) => req.extend_from_slice(&buf[..n]),
        }
    };
    let head = String::from_utf8_lossy(&req[..head_end]).into_owned();
    let mut words = head.lines().next().unwrap_or("").split(' ');
    let (method, target) = (words.next().unwrap_or(""), words.next().unwrap_or(""));
    let is_head = method == "HEAD";
    if !header(&head, "host").is_some_and(|h| hosts.iter().any(|x| x == h)) {
        return (
            Response::text("421 Misdirected Request", "unexpected Host"),
            is_head,
        );
    }
    if let Some(origin) = header(&head, "origin") {
        if !hosts.iter().any(|h| origin == format!("http://{h}")) {
            return (
                Response::text("403 Forbidden", "cross-origin request"),
                is_head,
            );
        }
    }
    let len: usize = match header(&head, "content-length").map(str::parse::<usize>) {
        None => 0,
        Some(Ok(n)) if n <= MAX_REQUEST => n,
        Some(_) => {
            return (
                Response::text("413 Content Too Large", "body too large"),
                is_head,
            )
        }
    };
    while req.len() < head_end + len {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => {
                return (
                    Response::text("400 Bad Request", "incomplete body"),
                    is_head,
                )
            }
            Ok(n) => req.extend_from_slice(&buf[..n]),
        }
    }
    let Some(rest) = target.strip_prefix(prefix) else {
        return (Response::text("404 Not Found", "not found"), is_head);
    };
    let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
    let r = Request {
        method: if is_head { "GET".into() } else { method.into() },
        path: path.into(),
        query: query.into(),
        body: req[head_end..head_end + len].to_vec(),
    };
    (handle(&r), is_head)
}

/// What `ui` reads.
pub struct UiFiles {
    pub worktodo: PathBuf,
    pub results: PathBuf,
    pub checkpoint_dir: Option<PathBuf>,
    pub status_files: Vec<PathBuf>,
}

/// `mersenne-forge ui`.
pub fn run_ui(
    addr: std::net::SocketAddr,
    files: UiFiles,
    launcher: Option<PathBuf>,
) -> Result<u8, Failure> {
    serve(addr, "ui --addr", "the control page", launcher, |r| match (
        r.method.as_str(),
        r.path.as_str(),
    ) {
        ("GET", "") => Response::new("200 OK", "text/html; charset=utf-8", UI_HTML),
        ("GET", "api/state") => Response::json("200 OK", &state(&files)),
        ("GET", "api/doctor") => {
            let p = r
                .query
                .split('&')
                .find_map(|kv| kv.strip_prefix("exponent="))
                .and_then(|v| v.parse::<u32>().ok());
            Response::json("200 OK", &crate::doctor_json(p))
        }
        ("POST", "api/queue") => match append_line(&files.worktodo, &r.body) {
            Ok(line) => Response::json("200 OK", &json!({ "ok": true, "line": line })),
            Err(e) => Response::json("400 Bad Request", &json!({ "ok": false, "error": e })),
        },
        ("POST", "api/quit") => {
            STOP.store(true, Ordering::SeqCst);
            Response::json("200 OK", &json!({ "ok": true }))
        }
        ("GET", _) | ("POST", _) => Response::text("404 Not Found", "not found"),
        _ => Response::text("405 Method Not Allowed", "only GET, HEAD, and POST"),
    })
}

/// A fresh launcher page path in the temp directory.
pub fn temp_launcher() -> PathBuf {
    std::env::temp_dir().join(format!("mersenne-forge-{}.html", token()))
}

/// Open `url` with the desktop's handler. The URL carries the access
/// token, and process arguments are visible to other local users, so the
/// handler gets `page`, a private (0600) local page that redirects to it.
/// False when the page could not be written; a failed open only prints a
/// note, since the URL is already on stderr.
fn open_browser(url: &str, key: &str, page: &Path) -> bool {
    if !open_browser_page(url, key, page) {
        eprintln!("could not open a browser; open the address above by hand");
        return false;
    }
    if open_path(page).is_err() {
        eprintln!("could not open a browser; open the address above by hand");
    }
    true
}

/// Write the private launcher page for `url`, with `key`.
fn open_browser_page(url: &str, key: &str, page: &Path) -> bool {
    let _ = std::fs::remove_file(page);
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    let written = o.open(page).and_then(|mut f| {
        write!(
            f,
            "<!doctype html><!--url:{url}--><!--key:{key}--><meta charset=\"utf-8\">\
             <meta name=\"referrer\" content=\"no-referrer\"><title>MersenneForge</title>\
             <script>location.replace({url:?})</script>\
             <a href=\"{url}\">Open the MersenneForge control page</a>"
        )
    });
    written.is_ok()
}

fn open_path(page: &Path) -> std::io::Result<std::process::Child> {
    let mut c = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    } else {
        std::process::Command::new("xdg-open")
    };
    c.arg(page)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
}

/// Where a launch without a command keeps its files: the platform's
/// per-user application data directory.
fn data_dir() -> Option<PathBuf> {
    let var = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if cfg!(target_os = "macos") {
        var("HOME").map(|h| h.join("Library/Application Support/MersenneForge"))
    } else if cfg!(windows) {
        var("APPDATA").map(|a| a.join("MersenneForge"))
    } else {
        var("XDG_DATA_HOME")
            .or_else(|| var("HOME").map(|h| h.join(".local/share")))
            .map(|d| d.join("mersenne-forge"))
    }
}

/// True when the launcher page points at a server that answers.
fn alive(page: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(page) else {
        return false;
    };
    let field = |name: &str| {
        text.split_once(name)
            .and_then(|(_, r)| r.split_once("-->"))
            .map(|(v, _)| v)
    };
    let (Some(url), Some(key)) = (field("<!--url:http://"), field("<!--key:")) else {
        return false;
    };
    let Some((host, path)) = url.split_once('/') else {
        return false;
    };
    let Ok(addr) = host.parse::<std::net::SocketAddr>() else {
        return false;
    };
    let Ok(mut s) = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(1)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    if write!(s, "HEAD /{path} HTTP/1.1\r\nHost: {host}\r\n\r\n").is_err() {
        return false;
    }
    let mut head = Vec::new();
    let _ = s.take(16 * 1024).read_to_end(&mut head);
    let head = String::from_utf8_lossy(&head);
    head.starts_with("HTTP/1.1 200")
        && head
            .lines()
            .any(|l| l.trim_end() == format!("X-Launcher-Key: {key}"))
}

/// A launch with no command, as from a double click: keep the files in the
/// per-user data directory, reuse a control page that is already running,
/// or start one in the background (no terminal needed) and open it.
pub fn launch_app() -> u8 {
    let Some(dir) = data_dir() else {
        eprintln!("no home directory; run mersenne-forge ui instead");
        return 2;
    };
    if let Err(e) = std::fs::create_dir_all(dir.join("checkpoints")) {
        eprintln!("{}: {e}", dir.display());
        return EXIT_STATE;
    }
    let page = dir.join("open.html");
    if alive(&page) {
        return if open_path(&page).is_ok() {
            0
        } else {
            EXIT_STATE
        };
    }
    // Left over from a server that did not stop cleanly, or answered by
    // something else: never open it.
    let _ = std::fs::remove_file(&page);
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("ui.log"));
    let Ok(exe) = std::env::current_exe() else {
        eprintln!("cannot find this program's path; run mersenne-forge ui instead");
        return EXIT_STATE;
    };
    let mut c = std::process::Command::new(exe);
    c.arg("ui")
        .arg("--worktodo")
        .arg(dir.join("worktodo.txt"))
        .arg("--results")
        .arg(dir.join("results.txt"))
        .arg("--checkpoint-dir")
        .arg(dir.join("checkpoints"))
        .arg("--launcher")
        .arg(&page)
        .current_dir(&dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null());
    match log {
        Ok(f) => c.stderr(f),
        Err(_) => c.stderr(std::process::Stdio::null()),
    };
    // Its own process group, so closing whatever launched it does not stop
    // the server; the page's Stop button does.
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut c, 0);
    match c.spawn() {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("could not start the control page: {e}");
            EXIT_STATE
        }
    }
}

/// How the worktodo parsers read a line: Some((kind, exponent, error)) for
/// an assignment line (error set when this program will not run it), None
/// for blank lines, comments, and headers.
fn classify(l: &str) -> Option<(String, Option<u32>, Option<String>)> {
    use mersenne_forge::{factor, generic, worktodo as wt};
    let kind = l.split_once('=').map(|(k, _)| k.trim().to_string())?;
    if let Ok(Some(_)) = generic::parse_generic_line(l) {
        return Some((kind, None, None));
    }
    match (factor::parse_factor_line(l), wt::parse_line(l)) {
        (Ok(Some(f)), _) => Some((kind, Some(f.exponent), None)),
        (Ok(None), Ok(Some(a))) => Some((kind, Some(a.exponent), None)),
        (Ok(None), Ok(None)) => None,
        (Err(e), _) | (_, Err(e)) => Some((kind, None, Some(e))),
    }
}

/// Everything the page shows, read fresh on every call.
fn state(f: &UiFiles) -> Value {
    let text = std::fs::read_to_string(&f.worktodo).unwrap_or_default();
    let queue: Vec<Value> = text
        .lines()
        .filter_map(|l| {
            classify(l).map(|(kind, exponent, error)| {
                json!({ "line": l, "kind": kind, "exponent": exponent, "runnable": error.is_none(), "error": error })
            })
        })
        .collect();
    let results_text = std::fs::read_to_string(&f.results).unwrap_or_default();
    let results: Vec<Value> = results_text
        .lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .take(500)
        .collect();
    let mut checkpoints = Vec::new();
    if let Some(d) = &f.checkpoint_dir {
        for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
            let path = e.path();
            if path.extension().is_some_and(|x| x == "ckpt") {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
                checkpoints.push(
                    match mersenne_forge::checkpoint::Checkpoint::load_file(&path) {
                        Ok(ck) => json!({
                            "file": name,
                            "protocol": ck.meta.protocol,
                            "exponent": ck.meta.exponent,
                            "k": ck.meta.k,
                            "verified_k": ck.meta.verified_k,
                            "errors": ck.meta.n_errors,
                            "elapsed_secs": ck.meta.elapsed_secs,
                            "backend": ck.meta.backend,
                        }),
                        Err(e) => json!({ "file": name, "error": e.to_string() }),
                    },
                );
            }
        }
    }
    let workers: Vec<Value> = f
        .status_files
        .iter()
        .map(|s| {
            let mut v = std::fs::read_to_string(s)
                .map_err(|e| e.to_string())
                .and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string()))
                .unwrap_or_else(|e| json!({ "state": "unreadable", "error": e }));
            v["file"] = json!(s.display().to_string());
            v
        })
        .collect();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "build": mersenne_forge::build_identity(),
        "worktodo": f.worktodo.display().to_string(),
        "worktodo_text": text.chars().take(64 * 1024).collect::<String>(),
        "results_file": f.results.display().to_string(),
        "checkpoint_dir": f.checkpoint_dir.as_ref().map(|d| d.display().to_string()),
        "queue": queue,
        "results": results,
        "checkpoints": checkpoints,
        "workers": workers,
    })
}

/// Append one assignment line to the worktodo file, under its `.lck` lock,
/// after checking that the parsers accept it and this program would run it.
fn append_line(worktodo: &Path, body: &[u8]) -> Result<String, String> {
    let line = std::str::from_utf8(body)
        .map_err(|_| "the line is not UTF-8".to_string())?
        .trim();
    if line.is_empty() || line.len() > 512 || line.contains(['\r', '\n']) {
        return Err("send one line of at most 512 bytes".into());
    }
    match classify(line) {
        None => return Err(format!("not an assignment line: {line}")),
        Some((_, _, Some(e))) => return Err(e),
        Some(_) => {}
    }
    let _guard = lck(worktodo).map_err(|f| f.message)?;
    let existing = std::fs::read(worktodo).unwrap_or_default();
    let mut out = String::new();
    if !existing.is_empty() && !existing.ends_with(b"\n") {
        out.push('\n');
    }
    out.push_str(line);
    out.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(worktodo)
        .map_err(|e| format!("{}: {e}", worktodo.display()))?;
    file.write_all(out.as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(|e| format!("{}: {e}", worktodo.display()))?;
    Ok(line.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A launcher page is trusted only when the server returns its key: a
    /// program answering 200 on the same port without it is not reused.
    #[test]
    fn launcher_pages_need_the_servers_key() {
        let dir = tempfile::tempdir().unwrap();
        let page = dir.path().join("open.html");
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let reply = |l: &std::net::TcpListener, key: &str| {
            let (mut c, _) = l.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = c.read(&mut buf);
            write!(c, "HTTP/1.1 200 OK\r\n{key}Content-Length: 0\r\n\r\n").unwrap();
        };
        assert!(open_browser_page(&format!("http://{addr}/t/"), "k1", &page));
        let t = std::thread::spawn(move || {
            reply(&l, "");
            reply(&l, "X-Launcher-Key: k1\r\n");
        });
        assert!(!alive(&page), "a 200 without the key is not this server");
        assert!(alive(&page), "the key proves it");
        t.join().unwrap();
    }

    #[test]
    fn appended_lines_are_checked_and_kept_whole() {
        let dir = std::env::temp_dir().join(format!("mf-ui-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let w = dir.join("worktodo.txt");
        std::fs::write(&w, "Test=127").unwrap();
        assert_eq!(
            append_line(&w, b"PRP=1,2,4423,-1\n").unwrap(),
            "PRP=1,2,4423,-1"
        );
        assert_eq!(
            std::fs::read_to_string(&w).unwrap(),
            "Test=127\nPRP=1,2,4423,-1\n"
        );
        assert!(append_line(&w, b"hello").is_err());
        assert!(append_line(&w, b"Test=127\nTest=521").is_err());
        assert!(append_line(&w, b"Test=128").is_err());
        assert_eq!(
            std::fs::read_to_string(&w).unwrap(),
            "Test=127\nPRP=1,2,4423,-1\n"
        );
        let s = state(&UiFiles {
            worktodo: w.clone(),
            results: dir.join("missing.txt"),
            checkpoint_dir: None,
            status_files: vec![dir.join("none.json")],
        });
        assert_eq!(s["queue"].as_array().unwrap().len(), 2);
        assert_eq!(s["queue"][1]["exponent"], 4423);
        assert_eq!(s["workers"][0]["state"], "unreadable");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn requests_from_other_hosts_and_origins_are_refused() {
        use std::net::{TcpListener, TcpStream};
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = l.local_addr().unwrap();
        let hosts = [a.to_string(), format!("localhost:{}", a.port())];
        let ask = |raw: String| {
            let mut c = TcpStream::connect(a).unwrap();
            c.write_all(raw.as_bytes()).unwrap();
            let (mut s, _) = l.accept().unwrap();
            let mut seen = None;
            let (r, _) = respond(&mut s, "/tok/", &hosts, &mut |q: &Request| {
                seen = Some((
                    q.method.clone(),
                    q.path.clone(),
                    q.query.clone(),
                    q.body.clone(),
                ));
                Response::text("200 OK", "ok")
            });
            (r.code, seen)
        };
        let (code, seen) = ask(format!(
            "POST /tok/api/queue?x=1 HTTP/1.1\r\nHost: {a}\r\nOrigin: http://{a}\r\nContent-Length: 4\r\n\r\nTest"
        ));
        assert_eq!(code, "200 OK");
        assert_eq!(
            seen,
            Some((
                "POST".into(),
                "api/queue".into(),
                "x=1".into(),
                b"Test".to_vec()
            ))
        );
        let (code, _) = ask(format!(
            "GET /tok/ HTTP/1.1\r\nHost: evil.example:{}\r\n\r\n",
            a.port()
        ));
        assert_eq!(code, "421 Misdirected Request");
        let (code, _) = ask(format!(
            "POST /tok/api/queue HTTP/1.1\r\nHost: {a}\r\nOrigin: http://evil.example\r\nContent-Length: 0\r\n\r\n"
        ));
        assert_eq!(code, "403 Forbidden");
        let (code, _) = ask(format!("GET /other/ HTTP/1.1\r\nHost: {a}\r\n\r\n"));
        assert_eq!(code, "404 Not Found");
    }
}
