#![cfg(feature = "icicle")]
use std::process::Command;

#[test]
fn operator_profile_roundtrips_to_sequence_and_rejects_stale_identity_without_truncation() {
    let directory = std::env::temp_dir().join(format!("operator_profile_{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let raw = directory.join("raw.csv");
    let profile = directory.join("profile.json");
    let output = directory.join("sequence.csv");
    let calibrated = Command::new(env!("CARGO_BIN_EXE_bench_operators"))
        .args(["--lengths", "16,32,64", "--warmup", "2", "--samples", "6", "--output"])
        .arg(&raw).arg("--profile-output").arg(&profile).output().unwrap();
    assert!(calibrated.status.success(), "{}", String::from_utf8_lossy(&calibrated.stderr));
    let csv = std::fs::read_to_string(&raw).unwrap();
    assert_eq!(csv.lines().skip(1).count(), 378);
    for line in csv.lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[14], "true");
    }
    for length in ["16", "32", "64"] {
        for operation in ["msm_dense", "msm_sparse_single", "msm_sparse_64",
            "msm_sparse_16", "msm_binary", "ntt_forward", "ntt_inverse"] {
            for (phase, expected) in [("first_call", 2), ("warmup", 4), ("sample", 12)] {
                let count = csv.lines().skip(1).filter(|line| {
                    let fields: Vec<_> = line.split(',').collect();
                    fields[5] == operation && fields[6] == length && fields[8] == phase
                }).count();
                assert_eq!(count, expected, "{operation}/{length}/{phase}");
            }
            // Each backend runs first in exactly half of the paired steady trials.
            for backend in ["cpu", "icicle"] {
                let first = csv.lines().skip(1).filter(|line| {
                    let fields: Vec<_> = line.split(',').collect();
                    fields[5] == operation && fields[6] == length && fields[7] == backend
                        && fields[8] == "sample" && fields[10] == "0"
                }).count();
                assert_eq!(first, 3, "unbalanced order for {operation}/{length}/{backend}");
            }
        }
    }
    let run = || Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "2", "--policies", "dense", "--workloads", "spreading",
            "--kzg-backends", "cpu,hybrid", "--icicle-large-bucket-factor", "10",
            "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output).arg("--dispatch-calibration").arg(&profile).output().unwrap();
    let result = run();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    assert!(std::fs::read_to_string(&output).unwrap().contains("hybrid_calibrated"));
    let original = std::fs::read_to_string(&profile).unwrap();
    let value: serde_json::Value = serde_json::from_str(&original).unwrap();
    for (key, bad) in [("source_hash", serde_json::json!("stale")),
        ("device", serde_json::json!("other-gpu")), ("large_bucket_factor", serde_json::json!(0))] {
        let mut invalid = value.clone();
        invalid[key] = bad;
        std::fs::write(&profile, serde_json::to_string(&invalid).unwrap()).unwrap();
        std::fs::write(&output, "preserve previous result\n").unwrap();
        assert!(!run().status.success());
        assert_eq!(std::fs::read_to_string(&output).unwrap(), "preserve previous result\n");
    }
    std::fs::write(&profile, &original).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--kzg-backends", "hybrid", "--icicle-large-bucket-factor", "10", "--output"])
        .arg(&profile).arg("--dispatch-calibration").arg(&profile).output().unwrap();
    assert!(!result.status.success());
    assert_eq!(std::fs::read_to_string(&profile).unwrap(), original);
    for path in [&raw, &profile, &output] { std::fs::remove_file(path).unwrap(); }
    std::fs::remove_dir(&directory).unwrap();
}
