// Copyright (C) 2026 Adam R Sophie. GPL-3.0-only.

//! A PrimeNet client for the manual web interface: log in, fetch
//! assignments into a worktodo file, send result lines, and upload proof
//! files. The endpoints, form fields, and replies follow the client scripts
//! upstream PRPLL shipped (`tools/primenet.py` and `tools/upload.py`, removed
//! upstream in favor of AutoPrimeNet).
//!
//! HTTP goes through the system `curl` program, so this crate needs no
//! network dependency. The password is passed to curl in a file readable
//! only by the owner, never on the command line.
//!
//! NOT TESTED against the PrimeNet server: this code has never contacted it.
//! Use it only after checking a first exchange by hand.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

pub const BASE_URL: &str = "https://www.mersenne.org/";
const PROOF_UPLOAD_URL: &str = "https://mersenne.org/proof_upload/";
const CHUNK: usize = 5 * 1024 * 1024;

/// PrimeNet work preference codes, as the upstream script named them.
pub fn worktype_code(name: &str) -> Option<u32> {
    Some(match name {
        "PRP" => 150,
        "PM1" => 4,
        "LL_DC" => 106,
        "PRP_DC" => 151,
        "PRP_WORLD_RECORD" => 152,
        "PRP_100M" => 153,
        "PRP_P1" => 154,
        other => other.parse().ok()?,
    })
}

/// A logged in session: a private state directory holding curl's cookie jar.
pub struct Session {
    dir: PathBuf,
    jar: PathBuf,
}

fn private_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut o = fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    o.open(path)?.write_all(bytes)
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".into(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Run curl with `args`; return its stdout. `--fail` is not used, because
/// PrimeNet reports errors in the page body.
fn curl(args: &[String]) -> Result<String, String> {
    let o = std::process::Command::new("curl")
        .args(["--silent", "--show-error", "--max-time", "120"])
        .args(args)
        .output()
        .map_err(|e| format!("curl: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "curl exited with {}: {}",
            o.status,
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&o.stdout).into_owned())
}

impl Session {
    /// Log in. `dir` holds the cookie jar and temporary form files.
    pub fn login(dir: &Path, user: &str, password: &str) -> Result<Session, String> {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let s = Session {
            dir: dir.to_path_buf(),
            jar: dir.join("primenet.cookies"),
        };
        private_file(&s.jar, b"").map_err(|e| e.to_string())?;
        let form = s.dir.join("login.form");
        private_file(
            &form,
            format!(
                "user_login={}&user_password={}",
                urlencode(user),
                urlencode(password)
            )
            .as_bytes(),
        )
        .map_err(|e| e.to_string())?;
        let page = s.post_form(BASE_URL, &form);
        let _ = fs::remove_file(&form);
        let page = page?;
        if !page.contains(&format!("{user}<br>logged in")) {
            return Err("PrimeNet login failed".into());
        }
        Ok(s)
    }

    fn jar_args(&self) -> Vec<String> {
        let j = self.jar.display().to_string();
        vec!["--cookie".into(), j.clone(), "--cookie-jar".into(), j]
    }

    fn post_form(&self, url: &str, form: &Path) -> Result<String, String> {
        let mut a = self.jar_args();
        a.extend([
            "--data-binary".into(),
            format!("@{}", form.display()),
            url.into(),
        ]);
        curl(&a)
    }

    /// Send one result line through the manual results form.
    pub fn send_result(&self, line: &str) -> Result<SendOutcome, String> {
        let form = self.dir.join("result.form");
        private_file(&form, format!("data={}", urlencode(line)).as_bytes())
            .map_err(|e| e.to_string())?;
        let page = self.post_form(&format!("{BASE_URL}manual_result/default.php"), &form);
        let _ = fs::remove_file(&form);
        Ok(classify_result_page(&page?))
    }

    /// Fetch one assignment line of work preference `code`.
    pub fn fetch_assignment(&self, code: u32) -> Result<String, String> {
        let form = self.dir.join("assignment.form");
        private_file(
            &form,
            format!("cores=1&num_to_get=1&pref={code}").as_bytes(),
        )
        .map_err(|e| e.to_string())?;
        let page = self.post_form(&format!("{BASE_URL}manual_assignment/"), &form);
        let _ = fs::remove_file(&form);
        parse_assignment_page(&page?)
    }
}

/// What the results form said about one line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum SendOutcome {
    Accepted(String),
    AlreadySent,
    Refused(String),
    Unrecognized,
}

/// Read the reply page of the manual results form, as the upstream script
/// did: "Error code" introduces a refusal (code 40 meaning already sent),
/// and "CPU credit is" an acceptance.
pub fn classify_result_page(page: &str) -> SendOutcome {
    let between = |start: &str| {
        page.find(start).map(|b| {
            let e = page[b..].find("</div>").map_or(page.len(), |e| b + e);
            page[b..e].trim().to_string()
        })
    };
    if let Some(err) = between("Error code") {
        if err.starts_with("Error code: 40,") {
            return SendOutcome::AlreadySent;
        }
        return SendOutcome::Refused(err);
    }
    match between("CPU credit is") {
        Some(c) => SendOutcome::Accepted(c),
        None => SendOutcome::Unrecognized,
    }
}

/// Take the assignment line out of the manual assignment page.
pub fn parse_assignment_page(page: &str) -> Result<String, String> {
    let begin = [
        "PRP=",
        "LL=",
        "Test=",
        "DoubleCheck=",
        "PRPDC=",
        "Pminus1=",
        "Pfactor=",
    ]
    .iter()
    .filter_map(|m| page.find(&format!(">{m}")))
    .min()
    .ok_or("no assignment in the PrimeNet reply")?
        + 1;
    let end = page[begin..]
        .find("</")
        .map(|e| begin + e)
        .ok_or("unterminated assignment in the PrimeNet reply")?;
    Ok(page[begin..end].trim().to_string())
}

/// Upload one proof file with the proof upload API: ask which byte range the
/// server needs, then post it in chunks of at most 5 MiB, each with its MD5.
/// Returns true when the server has the whole file (including when it
/// already had it).
pub fn upload_proof(dir: &Path, user: &str, proof: &Path) -> Result<bool, String> {
    let data = fs::read(proof).map_err(|e| format!("{}: {e}", proof.display()))?;
    let exponent = proof_exponent(&data).ok_or("not a version 2 PRP proof file")?;
    let md5hex = |b: &[u8]| -> String {
        crate::worktodo::md5(b)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect()
    };
    let file_md5 = md5hex(&data);
    let ask = format!(
        "{PROOF_UPLOAD_URL}?UserID={}&Exponent={exponent}&FileSize={}&FileMD5={file_md5}",
        urlencode(user),
        data.len()
    );
    loop {
        let reply: serde_json::Value = serde_json::from_str(&curl(&[ask.clone()])?)
            .map_err(|e| format!("proof upload reply: {e}"))?;
        if reply.get("error_status").is_some() {
            return Ok(reply["error_status"] == 409
                && reply["error_description"] == "Proof already uploaded");
        }
        let (Some(url), Some(need)) = (reply["URLToUse"].as_str(), reply["need"].as_object())
        else {
            return Err(format!("unexpected proof upload reply: {reply}"));
        };
        let url = match url.strip_prefix("http:") {
            Some(rest) => format!("https:{rest}"),
            None => url.to_string(),
        };
        let base = format!("{url}&FileMD5={file_md5}");
        // The first region the server still needs; its end is inclusive.
        let (mut pos, end) = need
            .iter()
            .filter_map(|(a, b)| {
                Some((
                    a.parse::<usize>().ok()?,
                    b.as_u64()
                        .map(|x| x as usize)
                        .or_else(|| b.as_str()?.parse().ok())?
                        + 1,
                ))
            })
            .min()
            .ok_or("the proof upload reply names no region")?;
        let chunk_file = dir.join("proof.chunk");
        while pos < end.min(data.len()) {
            let size = (end - pos).min(CHUNK).min(data.len() - pos);
            let chunk = &data[pos..pos + size];
            private_file(&chunk_file, chunk).map_err(|e| e.to_string())?;
            let page = curl(&[
                "--form".into(),
                format!("Data=<{}", chunk_file.display()),
                format!(
                    "{base}&DataOffset={pos}&DataSize={size}&DataMD5={}",
                    md5hex(chunk)
                ),
            ]);
            let _ = fs::remove_file(&chunk_file);
            let page = page?;
            pos += size;
            if page.contains("FileUploaded") {
                return Ok(true);
            }
        }
        // The region was sent without the final acknowledgment: ask again.
    }
}

/// The exponent in a version 2 proof header ("PRP PROOF\nVERSION=2\n...
/// NUMBER=M<p>\n").
fn proof_exponent(data: &[u8]) -> Option<u32> {
    let head = String::from_utf8_lossy(&data[..data.len().min(200)]);
    let mut lines = head.lines();
    if lines.next()? != "PRP PROOF" {
        return None;
    }
    lines
        .find_map(|l| l.strip_prefix("NUMBER=M"))
        .and_then(|n| n.parse().ok())
}

/// Lines of `path`, trimmed, without blanks.
fn lines_of(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

pub fn append_line(path: &Path, line: &str) -> Result<(), String> {
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(format!("{line}\n").as_bytes())
        .and_then(|()| f.sync_all())
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Paths of one worker directory's exchange state.
pub struct Exchange {
    pub results: PathBuf,
    /// Result lines PrimeNet accepted or already had.
    pub sent: PathBuf,
    /// Result lines PrimeNet refused, kept for a person to look at.
    pub refused: PathBuf,
    pub proof_dir: Option<PathBuf>,
}

impl Exchange {
    /// Result lines not yet sent (and not refused).
    pub fn unsent(&self) -> Vec<String> {
        let done: std::collections::HashSet<String> = lines_of(&self.sent)
            .into_iter()
            .chain(lines_of(&self.refused))
            .collect();
        lines_of(&self.results)
            .into_iter()
            .filter(|l| l.starts_with('{') && !done.contains(l))
            .collect()
    }

    /// Proof files waiting for upload (uploaded ones move to `uploaded/`).
    pub fn unuploaded(&self) -> Vec<PathBuf> {
        let Some(d) = &self.proof_dir else {
            return Vec::new();
        };
        let mut v: Vec<PathBuf> = fs::read_dir(d)
            .map(|r| {
                r.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|e| e == "proof"))
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }
}

/// What one exchange pass did.
#[derive(Debug, Default, Serialize)]
pub struct ExchangeReport {
    pub sent: usize,
    pub already_sent: usize,
    pub refused: usize,
    pub unrecognized: usize,
    pub fetched: Vec<String>,
    pub uploaded: Vec<PathBuf>,
    pub upload_failed: Vec<String>,
}

/// Lines of a worktodo file that are not comments.
pub fn runnable_lines(worktodo: &Path) -> usize {
    lines_of(worktodo)
        .iter()
        .filter(|l| !l.starts_with('#'))
        .count()
}

/// One pass: send unsent results and upload proofs. Fetching work is left
/// to the caller, which holds the worktodo lock while it appends.
pub fn exchange(s: &Session, user: &str, x: &Exchange) -> Result<ExchangeReport, String> {
    let mut rep = ExchangeReport::default();
    for line in x.unsent() {
        match s.send_result(&line)? {
            SendOutcome::Accepted(_) => {
                rep.sent += 1;
                append_line(&x.sent, &line)?;
            }
            SendOutcome::AlreadySent => {
                rep.already_sent += 1;
                append_line(&x.sent, &line)?;
            }
            SendOutcome::Refused(why) => {
                rep.refused += 1;
                append_line(&x.refused, &format!("{line}\t{why}"))?;
            }
            // Unknown reply: try again next pass.
            SendOutcome::Unrecognized => rep.unrecognized += 1,
        }
    }
    if let Some(d) = &x.proof_dir {
        let done = d.join("uploaded");
        for p in x.unuploaded() {
            match upload_proof(&s.dir, user, &p) {
                Ok(true) => {
                    fs::create_dir_all(&done).map_err(|e| e.to_string())?;
                    let to = done.join(p.file_name().expect("a file"));
                    fs::rename(&p, &to).map_err(|e| format!("{}: {e}", p.display()))?;
                    rep.uploaded.push(to);
                }
                Ok(false) => rep.upload_failed.push(format!("{}: refused", p.display())),
                Err(e) => rep.upload_failed.push(format!("{}: {e}", p.display())),
            }
        }
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reply_pages_are_classified() {
        assert_eq!(
            classify_result_page("<div>Error code: 40, error text: This computer has already sent in this PRP result</div>"),
            SendOutcome::AlreadySent
        );
        assert!(matches!(
            classify_result_page("<div>Error code: 12, bad</div>"),
            SendOutcome::Refused(_)
        ));
        assert!(matches!(
            classify_result_page("<div>CPU credit is 1.2 GHz-days.</div>"),
            SendOutcome::Accepted(_)
        ));
        assert_eq!(
            classify_result_page("<html></html>"),
            SendOutcome::Unrecognized
        );
    }

    #[test]
    fn assignment_lines_are_extracted() {
        let page =
            "<pre><!--BEGIN_ASSIGNMENTS_BLOCK-->\n<b>PRP=ABC,1,2,124647911,-1,77,0</b></pre>";
        assert_eq!(
            parse_assignment_page(page).unwrap(),
            "PRP=ABC,1,2,124647911,-1,77,0"
        );
        assert!(parse_assignment_page("nothing").is_err());
    }

    #[test]
    fn forms_are_url_encoded() {
        assert_eq!(urlencode("a b&c=d"), "a+b%26c%3Dd");
        assert_eq!(worktype_code("PRP"), Some(150));
        assert_eq!(worktype_code("151"), Some(151));
        assert_eq!(worktype_code("x"), None);
    }

    #[test]
    fn proof_headers_name_the_exponent() {
        assert_eq!(
            proof_exponent(b"PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=8\nNUMBER=M44497\n"),
            Some(44497)
        );
        assert_eq!(proof_exponent(b"junk"), None);
    }
}
