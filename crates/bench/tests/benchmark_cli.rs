use std::path::PathBuf;
use std::process::Command;

fn output_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "dynaverse_bench_{label}_{}_{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn cli_writes_reproducible_small_dynamo_matrix() {
    let output = output_path("smoke");
    let result = Command::new(env!("CARGO_BIN_EXE_dynaverse-bench"))
        .args([
            "--m", "8", "--k-powers", "0..2", "--samples", "3", "--warmup", "1", "--output",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let csv = std::fs::read_to_string(&output).unwrap();
    std::fs::remove_file(&output).unwrap();
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows[0], "commit,cpu,m,k,phase,samples,median_us,p95_us,source_hash,source_dirty");
    assert_eq!(rows.len(), 11); // header, setup, then 3 phases for k=1/2/4
    for k in [1usize, 2, 4] {
        for phase in ["derive_h", "prove_sparse", "verify"] {
            let marker = format!(",8,{k},{phase},3,");
            assert!(rows.iter().any(|row| row.contains(&marker)), "missing {marker}");
        }
    }
    assert!(rows.iter().any(|row| row.contains(",8,0,setup,1,")));
    for row in &rows[1..] {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.len(), 10, "{row}");
        assert_eq!(fields[0].len(), 40, "commit hash missing: {row}");
        assert!(!fields[1].is_empty(), "CPU model missing: {row}");
        assert!(fields[6].parse::<f64>().unwrap() >= 0.0);
        assert!(fields[7].parse::<f64>().unwrap() >= 0.0);
        assert_eq!(fields[8].len(), 40, "build source hash missing: {row}");
        assert!(matches!(fields[9], "true" | "false"), "build dirty flag missing: {row}");
    }
}

#[test]
fn cli_records_build_source_when_run_outside_checkout() {
    let output = output_path("outside_checkout");
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let expected = Command::new("git")
        .arg("-c")
        .arg(format!("safe.directory={}", source_root.canonicalize().unwrap().display()))
        .arg("-C")
        .arg(&source_root)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(expected.status.success());
    let expected_commit = String::from_utf8(expected.stdout).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_dynaverse-bench"))
        .current_dir(std::env::temp_dir())
        .args(["--m", "8", "--k-powers", "0..0", "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let csv = std::fs::read_to_string(&output).unwrap();
    std::fs::remove_file(&output).unwrap();
    let first_data_row = csv.lines().nth(1).unwrap();
    assert_eq!(first_data_row.split(',').next().unwrap(), expected_commit.trim());
}

#[test]
fn cli_rejects_non_radix_two_m_without_output() {
    let output = output_path("invalid");
    let result = Command::new(env!("CARGO_BIN_EXE_dynaverse-bench"))
        .args(["--m", "3", "--k-powers", "0..1", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!output.exists());
}
