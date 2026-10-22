//! Probe output routing only; never changes arithmetic or operator timers.
use anyhow::{ensure,Result};
use std::path::PathBuf;

pub fn resolve(requested:Option<&str>)->Result<PathBuf> {
    if let Some(value)=requested {
        let path=PathBuf::from(value);
        ensure!(path.is_absolute(),"DVGPU_G2_OUTPUT_DIR must be absolute");
        return Ok(path);
    }
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/g2-validation-20261017"))
}

pub fn directory()->Result<PathBuf> {
    let configured=std::env::var("DVGPU_G2_OUTPUT_DIR").ok();
    let path=resolve(configured.as_deref())?;
    std::fs::create_dir_all(&path)?;
    Ok(path)
}
