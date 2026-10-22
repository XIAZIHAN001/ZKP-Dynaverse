use std::path::PathBuf;

fn required_dir(name: &str) -> PathBuf {
    let value = std::env::var_os(name).unwrap_or_else(|| panic!("set {name} for the icicle feature"));
    let path = PathBuf::from(value);
    assert!(path.is_absolute() && path.is_dir(), "{name} must be an existing absolute directory");
    path
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ICICLE");
    if std::env::var_os("CARGO_FEATURE_ICICLE").is_none() { return; }
    assert_eq!(std::env::var("CARGO_CFG_TARGET_OS").unwrap(), "linux",
        "icicle feature currently supports Linux/WSL CUDA builds");
    for name in ["DYNAVERSE_NATIVE_LIB_DIR", "ICICLE_LIB_DIR", "CUDA_HOME"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let native = required_dir("DYNAVERSE_NATIVE_LIB_DIR");
    let icicle = required_dir("ICICLE_LIB_DIR");
    for (directory, library) in [(&native, "dynaverse_icicle"),
        (&icicle, "ingo_curve_bn254"), (&icicle, "ingo_field_bn254")] {
        let archive = directory.join(format!("lib{library}.a"));
        assert!(archive.is_file(), "missing {}: build the native bridge first", archive.display());
        println!("cargo:rerun-if-changed={}", archive.display());
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib=static={library}");
    }
    let cuda = PathBuf::from(std::env::var_os("CUDA_HOME").unwrap_or_else(|| "/usr/local/cuda".into()));
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    // libcuda is supplied at runtime by the driver, including Docker's GPU
    // mounts. The toolkit stub is used only by the link editor.
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64/stubs").display());
    for library in ["stdc++", "cudart", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
}
