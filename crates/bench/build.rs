use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>, Box<dyn Error>> {
    let output = Command::new("git")
        .arg("-c")
        .arg(format!("safe.directory={}", root.display()))
        .arg("-c")
        .arg("core.autocrlf=true")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(format!("git {:?}: {}", args, String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(output.stdout)
}

fn git_path(root: &Path, path: &str) -> Result<PathBuf, Box<dyn Error>> {
    let output = git(root, &["rev-parse", "--git-path", path])?;
    let value = String::from_utf8(output)?.trim().to_owned();
    let path = PathBuf::from(value);
    Ok(if path.is_absolute() { path } else { root.join(path) })
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?)
        .join("../..")
        .canonicalize()?;
    for path in ["Cargo.toml", "Cargo.lock", "crates", "native", "scripts/build_native_icicle.sh", "env"] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    println!("cargo:rerun-if-changed={}", git_path(&root, "HEAD")?.display());
    println!("cargo:rerun-if-changed={}", git_path(&root, "packed-refs")?.display());
    let branch = git(&root, &["symbolic-ref", "-q", "HEAD"]);
    if let Ok(branch) = branch {
        println!(
            "cargo:rerun-if-changed={}",
            git_path(&root, String::from_utf8(branch)?.trim())?.display()
        );
    }

    let head = String::from_utf8(git(&root, &["rev-parse", "HEAD"])?)?
        .trim()
        .to_owned();
    let diff = git(
        &root,
        &["diff", "--no-ext-diff", "--binary", "HEAD", "--", "Cargo.toml", "Cargo.lock", "crates", "native", "scripts/build_native_icicle.sh", "env"],
    )?;
    let untracked = git(
        &root,
        &["ls-files", "--others", "--exclude-standard", "-z", "--", "Cargo.toml", "Cargo.lock", "crates", "native", "scripts/build_native_icicle.sh", "env"],
    )?;
    let dirty = !diff.is_empty() || !untracked.is_empty();
    let mut material = Vec::new();
    material.extend_from_slice(head.as_bytes());
    material.extend_from_slice(&diff);
    for path in untracked.split(|&byte| byte == 0).filter(|path| !path.is_empty()) {
        material.extend_from_slice(path);
        material.push(0);
        material.extend_from_slice(&fs::read(root.join(String::from_utf8(path.to_vec())?))?);
        material.push(0);
    }
    let mut hash = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    hash.stdin.take().ok_or("git hash-object stdin unavailable")?.write_all(&material)?;
    let output = hash.wait_with_output()?;
    if !output.status.success() {
        return Err("git hash-object failed".into());
    }
    let source_hash = String::from_utf8(output.stdout)?.trim().to_owned();
    println!("cargo:rustc-env=DYNAVERSE_BUILD_COMMIT={head}");
    println!("cargo:rustc-env=DYNAVERSE_SOURCE_HASH={source_hash}");
    println!("cargo:rustc-env=DYNAVERSE_SOURCE_DIRTY={dirty}");
    Ok(())
}
