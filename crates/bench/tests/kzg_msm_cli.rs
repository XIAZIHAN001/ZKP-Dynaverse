#![cfg(feature = "icicle")]
use std::process::Command;

#[test]
fn kzg_cli_keeps_normal_samples_separate_from_profile_spans() {
    let directory = std::env::temp_dir();
    let output = directory.join(format!("kzg_msm_{}.csv", std::process::id()));
    let profile = directory.join(format!("kzg_msm_profile_{}.csv", std::process::id()));
    let result = Command::new(env!("CARGO_BIN_EXE_bench_kzg_msm"))
        .args(["--lengths", "1,17", "--warmup", "0", "--samples", "2", "--output"])
        .arg(&output).arg("--profile-output").arg(&profile).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let regular_csv = std::fs::read_to_string(&output).unwrap();
    let profile_csv = std::fs::read_to_string(&profile).unwrap();
    std::fs::remove_file(&output).unwrap();
    std::fs::remove_file(&profile).unwrap();
    // Each row borrows the CSV strings, which stay owned in this test scope.
    let mut regular = Vec::new();
    for line in regular_csv.lines().skip(1) { regular.push(line.split(',').collect::<Vec<&str>>()); }
    let mut sample_count = 0;
    for row in &regular { if row[5] == "sample" { sample_count += 1; } }
    assert_eq!(sample_count, 8);
    let mut diagnostic = Vec::new();
    for line in profile_csv.lines().skip(1) { diagnostic.push(line.split(',').collect::<Vec<&str>>()); }
    assert_eq!(diagnostic.len(), 32);
    for length in ["1", "17"] {
        for sample in ["0", "1"] {
            let mut spans = std::collections::HashMap::new();
            for row in &diagnostic {
                assert_eq!(row.len(), 10);
                assert_eq!(row[4], "icicle");
                if row[3] == length && row[6] == sample {
                    assert!(spans.insert(row[5], row[7].parse::<f64>().unwrap()).is_none());
                }
            }
            for name in ["total", "validation", "encoding", "native", "decode",
                "native_conversion", "icicle_inclusive", "native_output"] {
                assert!(spans.contains_key(name), "missing {name}");
            }
            // Allow only the CSV's nanosecond rounding, not overlapping spans.
            assert!(spans["total"] + 0.004 >= spans["validation"] + spans["encoding"] + spans["native"] + spans["decode"]);
            assert!(spans["native"] + 0.004 >= spans["native_conversion"] + spans["icicle_inclusive"] + spans["native_output"]);
            assert!(spans["icicle_inclusive"] > 0.0);
        }
    }
}

#[test]
fn kzg_cli_rejects_output_alias_without_truncating_previous_results() {
    let directory = std::env::temp_dir();
    let output = directory.join(format!("kzg_alias_{}.csv", std::process::id()));
    let alias = directory.join(format!("kzg_alias_link_{}.csv", std::process::id()));
    std::fs::write(&output, "previous measurement\n").unwrap();
    std::fs::hard_link(&output, &alias).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_bench_kzg_msm"))
        .args(["--lengths", "1", "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output).arg("--profile-output").arg(&alias).output().unwrap();
    assert!(!result.status.success());
    let saved = std::fs::read_to_string(&output).unwrap();
    std::fs::remove_file(&alias).unwrap();
    std::fs::remove_file(&output).unwrap();
    assert_eq!(saved, "previous measurement\n");
}
