use std::process::Command;

#[test]
fn cache_benchmark_reports_precompute_and_both_opening_paths() {
    let output = std::env::temp_dir().join(format!("kzg_cache_{}.csv", std::process::id()));
    let result = Command::new(env!("CARGO_BIN_EXE_kzg_cache"))
        .current_dir(std::env::temp_dir())
        .args(["--n", "8", "--ks", "1,4", "--warmup", "0", "--samples", "3", "--output"])
        .arg(&output).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let csv = std::fs::read_to_string(&output).unwrap();
    std::fs::remove_file(&output).unwrap();
    let rows: Vec<_> = csv.lines().skip(1).map(|row| row.split(',').collect::<Vec<_>>()).collect();
    assert_eq!(rows.len(), 5);
    for (k, phase) in [(0, "precompute_fft"), (1, "dense_batch"), (1, "cached_combine"),
        (4, "dense_batch"), (4, "cached_combine")] {
        let row = rows.iter().find(|row| row[3] == k.to_string() && row[4] == phase).unwrap();
        assert_eq!(row[0].len(), 40);
        assert!(!row[1].is_empty());
        assert_eq!(row[2], "8");
        assert_eq!(row[5], "3");
        assert!(row[6].parse::<f64>().unwrap() >= 0.0);
        assert!(row[7].parse::<f64>().unwrap() >= row[6].parse::<f64>().unwrap());
        assert_eq!(row[8].len(), 40);
        assert!(matches!(row[9], "true" | "false"));
    }
}

#[test]
fn cache_benchmark_rejects_invalid_shapes_before_writing() {
    for (i, args) in [vec!["--n", "3"], vec!["--n", "8", "--ks", "0"],
        vec!["--n", "8", "--ks", "9"], vec!["--n", "8", "--samples", "0"]].iter().enumerate() {
        let output = std::env::temp_dir().join(format!("kzg_cache_bad_{}_{i}.csv", std::process::id()));
        let result = Command::new(env!("CARGO_BIN_EXE_kzg_cache"))
            .args(args).arg("--output").arg(&output).output().unwrap();
        assert!(!result.status.success());
        assert!(!output.exists());
    }
}
