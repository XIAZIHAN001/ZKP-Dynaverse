use std::path::PathBuf;
use std::process::Command;

fn output_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("cpu_dynaverse_{label}_{}.csv", std::process::id()))
}

#[cfg(feature = "icicle")]
#[test]
fn pipeline_explicit_large_bucket_config_records_effective_factor_and_raw_samples() {
    // A dropped CLI->owner setting or lost metadata makes this fail. Exercise
    // real protocol GPU calls and preserve the raw complete-sequence samples.
    let output = output_path("large_factor");
    let raw = output_path("large_factor_samples");
    let profile = output_path("large_factor_profile");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "1", "--policies", "dense", "--workloads", "hotspot",
            "--kzg-backends", "cpu,icicle,hybrid", "--gpu-min-points", "2",
            "--icicle-large-bucket-factor", "10", "--warmup", "0", "--samples", "2", "--output"])
        .arg(&output).arg("--sequence-samples-output").arg(&raw)
        .arg("--msm-profile-output").arg(&profile).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    for path in [&output, &raw, &profile] {
        let csv = std::fs::read_to_string(path).unwrap();
        let mut lines = csv.lines();
        let header: Vec<&str> = lines.next().unwrap().split(',').collect();
        let factor = header.iter().position(|&field| field == "icicle_large_bucket_factor").unwrap();
        let backend = header.iter().position(|&field| field == "kzg_backend").unwrap();
        let dispatch = header.iter().position(|&field| field == "dispatch");
        let mut gpu_rows = 0;
        for line in lines {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), header.len());
            let gpu = fields[backend] != "cpu" && fields[backend] != "shared"
                && dispatch.map(|index| fields[index] != "cpu").unwrap_or(true);
            assert_eq!(fields[factor], if gpu { "10" } else { "0" });
            if gpu { gpu_rows += 1; }
        }
        assert!(gpu_rows > 0);
    }
    let raw_csv = std::fs::read_to_string(&raw).unwrap();
    assert_eq!(raw_csv.lines().skip(1).count(), 6);
    for path in [&output, &raw, &profile] { std::fs::remove_file(path).unwrap(); }
}

#[test]
fn pipeline_rejects_unused_or_invalid_large_bucket_factor_before_creating_outputs() {
    for (label, modes, factor) in [("unused", "cpu", "10"), ("invalid", "hybrid", "101")] {
        let output = output_path(label);
        let raw = output_path(&format!("{label}_raw"));
        let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
            .args(["--n", "8", "--kzg-backends", modes, "--icicle-large-bucket-factor", factor, "--output"])
            .arg(&output).arg("--sequence-samples-output").arg(&raw).output().unwrap();
        assert!(!result.status.success());
        assert!(!output.exists() && !raw.exists());
    }
}

#[test]
fn pipeline_raw_sequence_samples_preserve_aggregates_and_alias_safety() {
    let output = output_path("raw_sequence_aggregates");
    let raw = output_path("raw_sequence_aggregates_trials");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "1", "--policies", "dense", "--workloads", "hotspot",
            "--warmup", "1", "--samples", "2", "--output"])
        .arg(&output).arg("--sequence-samples-output").arg(&raw).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let raw_csv = std::fs::read_to_string(&raw).unwrap();
    let mut samples = Vec::new();
    assert!(raw_csv.lines().next().unwrap().ends_with(",updates,warmup,samples"));
    for line in raw_csv.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields[8], "cpu");
        assert_eq!(fields[9], "0");
        assert_eq!(&fields[12..], ["1", "1", "2"]);
        samples.push(fields[11].parse::<f64>().unwrap());
    }
    assert_eq!(samples.len(), 2);
    let csv = std::fs::read_to_string(&output).unwrap();
    let total = csv.lines().skip(1).find(|line| {
        let fields: Vec<&str> = line.split(',').collect();
        fields[5] == "sequence" && fields[9] == "total"
    }).unwrap();
    let fields: Vec<&str> = total.split(',').collect();
    assert!((fields[11].parse::<f64>().unwrap() - (samples[0] + samples[1]) / 2.0).abs() <= 0.0011);
    assert!((fields[12].parse::<f64>().unwrap() - samples[0].max(samples[1])).abs() <= 0.0011);
    std::fs::write(&output, "preserve prior result\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--output"]).arg(&output)
        .arg("--sequence-samples-output").arg(&output).output().unwrap();
    assert!(!result.status.success());
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "preserve prior result\n");
    for path in [&output, &raw] { std::fs::remove_file(path).unwrap(); }
}

#[test]
fn pipeline_timeline_requires_gpu_mode_before_creating_outputs() {
    let output = output_path("timeline_no_gpu");
    let timeline = output_path("timeline_no_gpu_calls");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "1", "--policies", "dense", "--workloads", "hotspot", "--output"])
        .arg(&output).arg("--msm-timeline-output").arg(&timeline).output().unwrap();
    assert!(!result.status.success());
    assert!(!output.exists() && !timeline.exists());
}

#[cfg(feature = "icicle")]
#[test]
fn pipeline_timeline_alias_preserves_each_existing_output() {
    for (index, victim) in ["timing", "inventory", "profile"].iter().enumerate() {
        let output = output_path(&format!("timeline_alias_{victim}_timing"));
        let inventory = output_path(&format!("timeline_alias_{victim}_inventory"));
        let profile = output_path(&format!("timeline_alias_{victim}_profile"));
        let missing = std::env::temp_dir().join(format!("timeline_alias_parent_{}_{index}", std::process::id()));
        assert!(!missing.exists());
        for path in [&output, &inventory, &profile] { std::fs::write(path, "previous output\n").unwrap(); }
        let target = match *victim { "timing" => &output, "inventory" => &inventory, _ => &profile };
        let alias = missing.join("..").join(target.file_name().unwrap());
        let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
            .args(["--n", "8", "--updates", "1", "--policies", "dense", "--workloads", "hotspot",
                "--kzg-backends", "hybrid", "--output"])
            .arg(&output).arg("--msm-output").arg(&inventory).arg("--msm-profile-output").arg(&profile)
            .arg("--msm-timeline-output").arg(&alias).output().unwrap();
        assert!(!result.status.success());
        for path in [output, inventory, profile] {
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "previous output\n");
            std::fs::remove_file(path).unwrap();
        }
        if missing.exists() { std::fs::remove_dir(missing).unwrap(); }
    }
}

#[cfg(feature = "icicle")]
#[test]
fn pipeline_timeline_preserves_real_call_identity_and_separates_clock_domains() {
    // Catch missing/mislabeled protocol calls, fabricated CPU GPU phases,
    // incorrect encoded copy sizes, and event/wall clock mixing.
    let output = output_path("timeline_timing");
    let inventory = output_path("timeline_inventory");
    let profile = output_path("timeline_host_profile");
    let timeline = output_path("timeline_events");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "2", "--policies", "dense", "--workloads", "spreading",
            "--kzg-backends", "cpu,icicle,hybrid", "--gpu-min-points", "16",
            "--warmup", "1", "--samples", "2", "--output"])
        .arg(&output).arg("--msm-output").arg(&inventory)
        .arg("--msm-profile-output").arg(&profile)
        .arg("--msm-timeline-output").arg(&timeline).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let ordinary = std::fs::read_to_string(&output).unwrap();
    for row in ordinary.lines().skip(1) { assert_eq!(row.split(',').count(), 23); }
    assert!(!std::fs::read_to_string(&profile).unwrap().contains("gpu_event"));
    let inventory_csv = std::fs::read_to_string(&inventory).unwrap();
    let timeline_csv = std::fs::read_to_string(&timeline).unwrap();
    let mut rows = Vec::new();
    for row in timeline_csv.lines().skip(1) { rows.push(row.split(',').collect::<Vec<_>>()); }
    let mut cpu_calls = 0;
    let mut gpu_calls = 0;
    for backend in ["icicle", "hybrid_16"] {
        for sample in ["0", "1"] {
            for call_line in inventory_csv.lines().skip(1) {
                let call: Vec<_> = call_line.split(',').collect();
                let mut spans = std::collections::HashMap::new();
                let mut route = "";
                for row in &rows {
                    assert_eq!(row.len(), 31);
                    assert_eq!(row[22], "true");
                    assert_eq!(&row[24..26], &["1", "2"]);
                    if row[8] != backend || row[18] != sample || row[9] != call[8]
                        || row[10] != call[9] || row[11] != call[10] { continue; }
                    assert_eq!(&row[12..15], &call[11..14]);
                    assert!(row[23].contains(&format!("backend={backend}/sample={sample}/operation={}/step={}/call={}/size={}",
                        call[8],call[9],call[10],call[11])));
                    route = row[15];
                    // Borrow the parsed row; the CSV/rows stay owned by this test.
                    assert!(spans.insert((row[16], row[17]), row.as_slice()).is_none());
                }
                assert!(spans.contains_key(&("wall", "total")));
                if route == "cpu" {
                    assert_eq!(backend, "hybrid_16");
                    assert_eq!(spans.len(), 1);
                    cpu_calls += 1;
                    continue;
                }
                assert_eq!(route, "icicle");
                let point_copy = spans[&("gpu_event", "point_h2d")];
                let scalar_copy = spans[&("gpu_event", "scalar_h2d")];
                assert_eq!(point_copy[20], "1");
                assert_eq!(scalar_copy[20], "1");
                let size: usize = call[11].parse().unwrap();
                assert_eq!(point_copy[21].parse::<usize>().unwrap(), size * 64);
                assert_eq!(scalar_copy[21].parse::<usize>().unwrap(), size * 32);
                assert!(spans[&("gpu_event", "kernel_execution")][19].parse::<f64>().unwrap() > 0.0);
                assert!(spans[&("wall", "synchronization")][20].parse::<usize>().unwrap() >= 2);
                let mut wall_sum = 0.0;
                for phase in ["allocation_preparation", "point_h2d", "scalar_h2d", "kernel_submission",
                    "synchronization", "output_d2h", "metadata_d2h", "free", "remaining_overhead", "instrumentation_inclusive"] {
                    wall_sum += spans[&("wall", phase)][19].parse::<f64>().unwrap();
                }
                let inclusive = spans[&("wall", "icicle_inclusive")][19].parse::<f64>().unwrap();
                assert!((wall_sum - inclusive).abs() < 0.011);
                assert!(!spans.contains_key(&("gpu_event", "synchronization")));
                gpu_calls += 1;
            }
        }
    }
    assert!(cpu_calls > 0 && gpu_calls > 0);
    for path in [output, inventory, profile, timeline] { std::fs::remove_file(path).unwrap(); }
}

#[test]
fn pipeline_missing_parent_alias_preserves_timing_before_inventory_creation() {
    let output = output_path("missing_parent_inventory_timing");
    let missing = std::env::temp_dir().join(format!("missing_inventory_parent_{}", std::process::id()));
    assert!(!missing.exists());
    let alias = missing.join("..").join(output.file_name().unwrap());
    std::fs::write(&output, "previous timing\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "1", "--policies", "dense", "--workloads", "hotspot",
            "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output).arg("--msm-output").arg(&alias).output().unwrap();
    let saved = std::fs::read_to_string(&output).unwrap();
    std::fs::remove_file(output).unwrap();
    if missing.exists() { std::fs::remove_dir(missing).unwrap(); }
    assert!(!result.status.success());
    assert!(saved == "previous timing\n", "old timing was truncated through missing/../ alias");
}

#[cfg(feature = "icicle")]
#[test]
fn pipeline_missing_parent_alias_preserves_inventory_before_profile_creation() {
    let output = output_path("missing_parent_profile_timing");
    let inventory = output_path("missing_parent_profile_inventory");
    let missing = std::env::temp_dir().join(format!("missing_profile_parent_{}", std::process::id()));
    assert!(!missing.exists());
    let alias = missing.join("..").join(inventory.file_name().unwrap());
    std::fs::write(&output, "previous timing\n").unwrap();
    std::fs::write(&inventory, "previous inventory\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "1", "--policies", "dense", "--workloads", "hotspot",
            "--kzg-backends", "hybrid", "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output).arg("--msm-output").arg(&inventory)
        .arg("--msm-profile-output").arg(&alias).output().unwrap();
    let saved_timing = std::fs::read_to_string(&output).unwrap();
    let saved_inventory = std::fs::read_to_string(&inventory).unwrap();
    for path in [output, inventory] { std::fs::remove_file(path).unwrap(); }
    if missing.exists() { std::fs::remove_dir(missing).unwrap(); }
    assert!(!result.status.success());
    assert!(saved_timing == "previous timing\n" && saved_inventory == "previous inventory\n",
        "old CSV was truncated through missing/../ profile alias");
}

#[cfg(feature = "icicle")]
#[test]
fn pipeline_profiles_real_protocol_calls_separately_and_checks_cpu_results() {
    // Catch dropped/mislabeled calls, altered timing schema and overlapping
    // native spans. The CPU inventory is an independent diagnostic sequence.
    let output = output_path("profile_timing");
    let inventory = output_path("profile_inventory");
    let profile = output_path("profile_calls");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "2", "--policies", "dense", "--workloads", "spreading",
            "--kzg-backends", "cpu,icicle,hybrid", "--gpu-min-points", "16",
            "--warmup", "1", "--samples", "2", "--output"])
        .arg(&output).arg("--msm-output").arg(&inventory)
        .arg("--msm-profile-output").arg(&profile).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let timing = std::fs::read_to_string(&output).unwrap();
    for row in timing.lines().skip(1) { assert_eq!(row.split(',').count(), 23); }
    let inventory_csv = std::fs::read_to_string(&inventory).unwrap();
    let profile_csv = std::fs::read_to_string(&profile).unwrap();
    let mut calls = Vec::new();
    for row in inventory_csv.lines().skip(1) { calls.push(row.split(',').collect::<Vec<&str>>()); }
    let mut rows = Vec::new();
    for row in profile_csv.lines().skip(1) { rows.push(row.split(',').collect::<Vec<&str>>()); }
    let mut cpu_routes = 0;
    let mut gpu_routes = 0;
    for backend in ["icicle", "hybrid_16"] {
        for sample in ["0", "1"] {
            for call in &calls {
                let mut spans = std::collections::HashMap::new();
                let mut dispatch = "";
                for row in &rows {
                    assert_eq!(row.len(), 23);
                    assert_eq!(row[21], "true");
                    assert_eq!(row[19], "1");
                    assert_eq!(row[20], "2");
                    if row[8] != backend || row[17] != sample || row[9] != call[8]
                        || row[10] != call[9] || row[11] != call[10] { continue; }
                    assert_eq!(&row[12..15], &call[11..14]);
                    dispatch = row[15];
                    assert!(spans.insert(row[16], row[18].parse::<f64>().unwrap()).is_none());
                }
                assert!(spans.contains_key("total"), "missing call {backend}/{sample}/{}", call[10]);
                if dispatch == "cpu" {
                    assert_eq!(backend, "hybrid_16");
                    assert_eq!(spans.len(), 1);
                    cpu_routes += 1;
                } else {
                    assert_eq!(dispatch, "icicle");
                    assert_eq!(spans.len(), 8);
                    assert!(spans["total"] + 0.004 >= spans["validation"] + spans["encoding"] + spans["native"] + spans["decode"]);
                    assert!(spans["native"] + 0.004 >= spans["native_conversion"] + spans["icicle_inclusive"] + spans["native_output"]);
                    assert!(spans["icicle_inclusive"] > 0.0);
                    gpu_routes += 1;
                }
            }
        }
    }
    assert!(cpu_routes > 0 && gpu_routes > 0);
    assert_eq!(rows.len(), cpu_routes + 8 * gpu_routes);
    for path in [output, inventory, profile] { std::fs::remove_file(path).unwrap(); }
}

#[test]
fn pipeline_profile_requires_gpu_mode_before_creating_outputs() {
    let output = output_path("profile_without_gpu");
    let profile = output_path("profile_without_gpu_calls");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--output"]).arg(&output)
        .arg("--msm-profile-output").arg(&profile).output().unwrap();
    assert!(!result.status.success());
    assert!(!output.exists() && !profile.exists());
}

#[cfg(all(unix, feature = "icicle"))]
#[test]
fn pipeline_profile_rejects_aliases_before_truncating_previous_csv() {
    for alias_target in ["timing", "inventory"] {
        let output = output_path(&format!("profile_alias_{alias_target}_timing"));
        let inventory = output_path(&format!("profile_alias_{alias_target}_inventory"));
        let profile = output_path(&format!("profile_alias_{alias_target}_calls"));
        std::fs::write(&output, "previous timing\n").unwrap();
        std::fs::write(&inventory, "previous inventory\n").unwrap();
        std::fs::hard_link(if alias_target == "timing" { &output } else { &inventory }, &profile).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
            .args(["--n", "8", "--kzg-backends", "hybrid", "--output"])
            .arg(&output).arg("--msm-output").arg(&inventory)
            .arg("--msm-profile-output").arg(&profile).output().unwrap();
        assert!(!result.status.success());
        assert_eq!(std::fs::read_to_string(&output).unwrap(), "previous timing\n");
        assert_eq!(std::fs::read_to_string(&inventory).unwrap(), "previous inventory\n");
        for path in [profile, inventory, output] { std::fs::remove_file(path).unwrap(); }
    }
}

#[test]
fn pipeline_inventory_records_operations_without_changing_timing_schema() {
    let output = output_path("inventory_timing");
    let inventory = output_path("inventory_calls");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "2", "--policies", "dense", "--workloads", "spreading",
            "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output).arg("--msm-output").arg(&inventory).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let timing = std::fs::read_to_string(&output).unwrap();
    for row in timing.lines().skip(1) { assert_eq!(row.split(',').count(), 23); }
    let calls = std::fs::read_to_string(&inventory).unwrap();
    assert_eq!(calls.lines().next().unwrap(), "commit,source_hash,source_dirty,n,policy,workload,verifier,batch_open_mode,operation,step,call,length,nonzero_scalars,max_scalar_bits");
    let mut counters = [0usize; 3];
    for line in calls.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 14);
        let step = fields[9].parse::<usize>().unwrap();
        assert!(step <= 2);
        assert_eq!(fields[8], if step == 0 { "prove" } else { "update" });
        assert_eq!(fields[10].parse::<usize>().unwrap(), counters[step]);
        counters[step] += 1;
        let length = fields[11].parse::<usize>().unwrap();
        let nonzero = fields[12].parse::<usize>().unwrap();
        assert!(length > 0 && nonzero <= length);
        assert!(fields[13].parse::<u32>().unwrap() <= 254);
    }
    for count in counters { assert!(count > 0); }
    std::fs::remove_file(output).unwrap();
    std::fs::remove_file(inventory).unwrap();
}

#[cfg(unix)]
#[test]
fn pipeline_inventory_rejects_alias_before_truncating_previous_csv() {
    let output = output_path("inventory_previous");
    let alias = output_path("inventory_alias");
    std::fs::write(&output, "previous measurement\n").unwrap();
    std::fs::hard_link(&output, &alias).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--output"]).arg(&output).arg("--msm-output").arg(&alias).output().unwrap();
    assert!(!result.status.success());
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "previous measurement\n");
    std::fs::remove_file(alias).unwrap();
    std::fs::remove_file(output).unwrap();
}

#[test]
fn pipeline_cli_covers_policies_workloads_and_rebuilds() {
    let output = output_path("matrix");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .current_dir(std::env::temp_dir())
        .args(["--n", "8", "--updates", "2", "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let csv = std::fs::read_to_string(&output).unwrap();
    std::fs::remove_file(&output).unwrap();
    let mut rows = Vec::new();
    for row in csv.lines().skip(1) {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.len(), 23);
        assert_eq!(fields[0].len(), 40);
        assert_eq!(fields[13].len(), 40);
        assert!(matches!(fields[14], "true" | "false"));
        assert!(fields[12].parse::<f64>().unwrap() >= fields[11].parse::<f64>().unwrap());
        rows.push(fields);
    }
    for policy in ["dense", "precompute"] {
        for workload in ["hotspot", "spreading"] {
            for operation in ["prove", "update", "verify", "sequence"] {
                assert!(rows.iter().any(|r| r[3] == policy && r[4] == workload && r[5] == operation && r[9] == "total"));
            }
            let second = rows.iter().find(|r| r[3] == policy && r[4] == workload && r[5] == "update" && r[6] == "2" && r[9] == "total").unwrap();
            if workload == "spreading" {
                assert_eq!(second[7], "2");
                assert_eq!(second[8], "1");
            } else {
                assert_eq!(second[7], "1");
                assert_eq!(second[8], "0");
            }
            let precompute = rows.iter().find(|r| r[3] == policy && r[4] == workload && r[5] == "prove" && r[9] == "cache_precompute").unwrap();
            let measured = precompute[11].parse::<f64>().unwrap();
            if policy == "dense" { assert_eq!(measured, 0.0); }
            else { assert!(measured > 0.0); }
            let verify = rows.iter().find(|r| r[3] == policy && r[4] == workload
                && r[5] == "verify" && r[6] == "1" && r[9] == "total").unwrap();
            assert_eq!(verify[16], "192");
            assert_eq!(verify[19], "86");
            assert_eq!(verify[15], "product");
            assert_eq!(verify[20], "prepared");
            assert_eq!(verify[21], "cpu");
            assert_eq!(verify[17], "86");
            assert_eq!(verify[18], "86");
            let consistency = rows.iter().find(|r| r[3] == policy && r[4] == workload
                && r[5] == "verify" && r[6] == "1" && r[9] == "verify_delta_consistency").unwrap();
            assert_eq!(consistency[16], "66");
            assert_eq!(consistency[19], "30");
            let detail = rows.iter().find(|r| r[3] == policy && r[4] == workload
                && r[5] == "update" && r[6] == "1" && r[9] == "batch_open_public_quotient").unwrap();
            assert!(detail[11].parse::<f64>().unwrap() > 0.0);
            let empty = rows.iter().find(|r| r[3] == policy && r[4] == workload
                && r[5] == "prove" && r[9] == "batch_open_public_quotient").unwrap();
            assert_eq!(empty[11].parse::<f64>().unwrap(), 0.0);
        }
    }
}

#[test]
fn pipeline_cli_rejects_invalid_inputs_before_creating_csv() {
    let cases = [vec!["--n", "1"], vec!["--n", "3"], vec!["--n", "8", "--updates", "0"],
        vec!["--n", "8", "--samples", "0"], vec!["--n", "8", "--policies", "invalid"],
        vec!["--n", "8", "--verifiers", "invalid"], vec!["--n", "8", "--batch-open-modes", "invalid"],
        vec!["--n", "8", "--kzg-backends", "invalid"]];
    for (index, args) in cases.iter().enumerate() {
        let output = output_path(&format!("invalid_{index}"));
        let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
            .args(args).arg("--output").arg(&output).output().unwrap();
        assert!(!result.status.success());
        assert!(!output.exists());
    }
}

#[cfg(feature = "icicle")]
#[test]
fn pipeline_cli_compares_explicit_hybrid_and_records_its_threshold() {
    let output = output_path("hybrid");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "2", "--policies", "dense", "--workloads", "spreading",
            "--kzg-backends", "cpu,icicle,hybrid", "--gpu-min-points", "16",
            "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let csv = std::fs::read_to_string(&output).unwrap();
    for backend in ["cpu", "icicle", "hybrid_16"] {
        let mut found_sequence = false;
        let mut found_rebuild = false;
        for line in csv.lines().skip(1) {
            let row: Vec<&str> = line.split(',').collect();
            assert_eq!(row.len(), 23);
            if row[21] != backend { continue; }
            if row[5] == "sequence" && row[9] == "total" { found_sequence = true; }
            if row[5] == "update" && row[6] == "2" && row[9] == "total" && row[8] == "1" { found_rebuild = true; }
        }
        assert!(found_sequence && found_rebuild);
    }
    std::fs::remove_file(output).unwrap();
}

#[test]
fn pipeline_cli_rejects_invalid_hybrid_threshold_before_creating_csv() {
    for threshold in ["0", "1048577"] {
        let output = output_path(&format!("hybrid_invalid_{threshold}"));
        let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
            .args(["--n", "8", "--kzg-backends", "hybrid", "--gpu-min-points", threshold, "--output"])
            .arg(&output).output().unwrap();
        assert!(!result.status.success());
        assert!(!output.exists());
    }
}

#[cfg(feature = "icicle")]
#[test]
fn pipeline_cli_records_cpu_gpu_sequences_and_initialization_separately() {
    let output = output_path("kzg_backends");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "2", "--policies", "dense", "--workloads", "spreading",
            "--kzg-backends", "cpu,icicle", "--warmup", "0", "--samples", "1", "--output"])
        .arg(&output).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let csv = std::fs::read_to_string(&output).unwrap();
    std::fs::remove_file(&output).unwrap();
    let rows: Vec<Vec<&str>> = csv.lines().skip(1).map(|line| line.split(',').collect()).collect();
    assert!(rows.iter().any(|row| row[5] == "backend_init" && row[21] == "icicle"));
    for backend in ["cpu", "icicle"] {
        assert!(rows.iter().any(|row| row[5] == "first_sequence" && row[21] == backend && row[10] == "1"));
        assert!(rows.iter().any(|row| row[5] == "sequence" && row[21] == backend && row[8] == "1"));
        assert!(rows.iter().any(|row| row[5] == "verify" && row[6] == "2" && row[21] == backend));
    }
}

#[test]
fn pipeline_cli_compares_candidate_and_independent_verifier_modes() {
    let output = output_path("verifiers");
    let result = Command::new(env!("CARGO_BIN_EXE_cpu_dynaverse"))
        .args(["--n", "8", "--updates", "1", "--policies", "dense",
            "--workloads", "hotspot", "--verifiers", "reference,product",
            "--batch-open-modes", "reference,prepared",
            "--warmup", "0", "--samples", "2", "--output"])
        .arg(&output).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let csv = std::fs::read_to_string(&output).unwrap();
    std::fs::remove_file(&output).unwrap();
    for mode in ["reference", "product"] {
        for batch_mode in ["reference", "prepared"] {
            let wanted_calls = if mode == "reference" { "192" } else { "86" };
            let mut found_candidate = false;
            let mut found_independent = false;
            for line in csv.lines().skip(1) {
                let fields: Vec<&str> = line.split(',').collect();
                if fields[15] != mode || fields[6] != "1" || fields[20] != batch_mode { continue; }
                if (fields[5] == "update" && fields[9] == "pairing_verify")
                    || (fields[5] == "verify" && fields[9] == "total") {
                    assert_eq!(fields[10], "2");
                    assert_eq!(fields[16], "192");
                    assert_eq!(fields[17], wanted_calls);
                    assert_eq!(fields[18], wanted_calls);
                    assert_eq!(fields[19], "86");
                    if fields[5] == "update" { found_candidate = true; }
                    else { found_independent = true; }
                }
            }
            assert!(found_candidate && found_independent);
        }
    }
    let preparation = csv.lines().skip(1).find(|line| line.split(',').nth(9) == Some("public_division_precompute")).unwrap();
    assert!(preparation.split(',').nth(11).unwrap().parse::<f64>().unwrap() > 0.0);
}
