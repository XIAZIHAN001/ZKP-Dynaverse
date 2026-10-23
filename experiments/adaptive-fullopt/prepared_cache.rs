//! Bounded prepared-cache experiment, separate from the frozen R5 probe.
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::{BigInteger, PrimeField, UniformRand, Zero};
use dynaverse_gpu::{G2ValidationMode, IcicleBackend, SetupMode};
use dynaverse_protocol::{Update, Witness, dynaverse::DynaverseParams,
    toy_circuit::ToyCircuit, trace::CpuTrace,
    incremental_rebuild::{UpdateTimings, LinearTimings, CacheCheckTimings,
        SparseProofTimings, operators::RebuildBackend}};
use rand::{rngs::StdRng, SeedableRng};
use serde_json::{json, Value};
use std::time::Instant;
mod strategy;
mod prepared_policy;
mod readback;

fn emit(row: Value) { println!("ROW {row}"); }

// Same owner options as R5. The shared calibration module always uses the
// original 64-point Strategy, independently of the prepared table capacity.
fn fullopt_backend(n: usize, optimized: bool, affine: bool) -> Result<IcicleBackend> {
    let policy = strategy::Strategy::for_size(n).map_err(anyhow::Error::msg)?;
    let mut backend = IcicleBackend::with_resident_limits(policy.limit_bytes, policy.undo_bytes)?;
    backend.set_bounded_g1_decode(optimized);
    backend.set_device_affine_readback(optimized && affine);
    backend.set_setup_mode(SetupMode::TableBatchNormalize)?;
    backend.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63)?;
    Ok(backend)
}

fn edits(w: &Witness, n: usize, step: usize) -> Result<Update> {
    // Match R5's deterministic inputs and its two 16-wire edit families.
    let mut rng = StdRng::seed_from_u64(22100 + n as u64 * 100 + step as u64);
    let addition = step % 2 == 1;
    let roots = if addition {4} else {8};
    let mut update = Update { indices: Vec::new(), values: Vec::new() };
    for i in 1..=roots {
        let value = Fr::rand(&mut rng);
        let old = if addition {w.wires[n+i]} else {w.wires[4*n+i]};
        ensure!((value-old).into_bigint().num_bits() > 128, "eta is not full width");
        if addition {
            let sum = w.wires[i] + value;
            update.indices.extend([n+i, 2*n+i, 3*n+i, 5*n+i]);
            update.values.extend([value, sum, sum, sum*w.wires[4*n+i]]);
        } else {
            update.indices.extend([4*n+i, 5*n+i]);
            update.values.extend([value, w.wires[3*n+i]*value]);
        }
    }
    Ok(update)
}

fn named_ns(names: &[&str], values: &[u128]) -> Value {
    let mut map = serde_json::Map::new();
    for (name, ns) in names.iter().zip(values) { map.insert((*name).into(), json!(ns)); }
    Value::Object(map)
}

fn timings(t: &UpdateTimings) -> Value {
    json!({"enabled":t.enabled,"total_ns":t.total_ns,
        "normalization_ns":t.normalization_ns,"constraints_ns":t.constraints_ns,
        "preparation_ns":t.preparation_ns,"publication_ns":t.publication_ns,
        "live_linear_ns":t.live_linear_ns,"delta_proof_ns":t.delta_proof_ns,"rebase_ns":t.rebase_ns,
        "rebuild_ns":named_ns(&UpdateTimings::REBUILD_NAMES,&t.rebuild_ns),
        "rebuild_calls":t.rebuild_calls,
        "cache_ns":named_ns(&CacheCheckTimings::NAMES,&t.cache_detail.ns),
        "linear_ns":named_ns(&LinearTimings::NAMES,&t.linear_detail.ns),
        "delta_detail_ns":named_ns(&SparseProofTimings::NAMES,&t.delta_detail.ns),
        "rebase_detail_ns":named_ns(&SparseProofTimings::NAMES,&t.rebase_detail.ns)})
}

fn run(n: usize, caps: &[usize]) -> Result<()> {
    let policy = strategy::Strategy::for_size(n).map_err(anyhow::Error::msg)?;
    ensure!(caps.len() >= 2 && caps[0] == 64, "comparison requires the full-optimized 64-root control");
    let mut actual_caps = Vec::new();
    for &cap in caps {
        prepared_policy::config(n, cap)?;
        let actual = n.min(cap);
        ensure!(!actual_caps.contains(&actual), "duplicate actual capacity");
        actual_caps.push(actual);
    }
    emit(json!({"kind":"prepared_start","schema":1,"n":n,"caps":caps,
        "actual_caps":actual_caps,"steps":4,"measured_steps":2,
        "window":policy.window,"readback_roots":64,"calibration_points":64,
        "row_batch":16,"packet_cap":policy.packet_cap,"native_owner_limit_bytes":policy.limit_bytes,
        "undo_bytes":policy.undo_bytes,"setup_seed":2026102220+n as u64}));
    let circuit = ToyCircuit::new(n)?;
    let (instance, witness) = circuit.valid_instance();
    let setup_backend = fullopt_backend(n, false, false)?;
    let mut setup_trace = CpuTrace::new();
    let started = Instant::now();
    let params = DynaverseParams::setup_with_backend_traced(circuit,
        &mut StdRng::seed_from_u64(2026102220+n as u64), &setup_backend, &mut setup_trace)?;
    emit(json!({"kind":"prepared_setup","n":n,"ns":started.elapsed().as_nanos(),
        "trace_recorded_ns":setup_trace.recorded_total()}));
    drop(setup_backend);
    let vk = params.verifying_key();
    let affine = readback::calibrate(n, policy)?;
    let mut sessions = Vec::new();
    let mut allocations = Vec::new();
    for &cap in caps {
        let started = Instant::now();
        let s = params.start_incremental_session(&instance, &witness,
            prepared_policy::config(n, cap)?, fullopt_backend(n, true, affine)?)?;
        let ns = started.elapsed().as_nanos();
        let p = s.resource_plan();
        let c = s.cache_preparation_stats().ok_or_else(||anyhow::anyhow!("missing prepared table"))?;
        let budget = s.budget();
        ensure!(p.upper_bytes <= policy.limit_bytes && p.transaction_bytes <= policy.undo_bytes,
            "resource or undo envelope exceeds owner admission");
        ensure!(c.roots == n.min(cap) && c.retained_bytes <= c.host_upper_bytes,
            "prepared allocation differs from plan");
        let started = Instant::now();
        ensure!(vk.verify_reference(&instance, s.proof())?, "initial Reference rejected");
        let verify_ns = started.elapsed().as_nanos();
        allocations.push(s.backend().stats().bytes_allocated);
        emit(json!({"kind":"prepared_initial","n":n,"cap":cap,"ns":ns,"verify_ns":verify_ns,
            "affine":affine,"correct":true,"roots":c.roots,"prepared_entries":c.prepared_entries,
            "retained_bytes":c.retained_bytes,"host_upper_bytes":c.host_upper_bytes,
            "precompute_ns":c.precompute_ns,"resident_bytes":s.backend().stats().bytes_allocated,
            "budget":{"q":budget.q.to_string(),"total":budget.total.to_string(),
                "max_chunk":budget.max_chunk.to_string(),"large":budget.large.to_string()},
            "resource":{"upper_bytes":p.upper_bytes,"resident_bytes":p.resident_bytes,
                "workspace_reserve":p.workspace_reserve,"transaction_bytes":p.transaction_bytes,
                "tile_len":p.tile_len,"frozen_linear_bytes":p.frozen_linear_bytes,
                "global_g2_host_bytes":p.global_g2_host_bytes,
                "global_g2_workspace_bytes":p.global_g2_workspace_bytes,
                "cache_preparation_host_bytes":p.cache_preparation_host_bytes,
                "cache_readback_host_bytes":p.cache_readback_host_bytes}}));
        sessions.push(s);
    }
    for s in &sessions {
        ensure!(s.proof() == sessions[0].proof() && s.current_witness() == &witness,
            "initial paired proof/witness differs");
    }
    emit(json!({"kind":"prepared_pair","n":n,"step":-1,"proof_equal":true,"witness_equal":true}));
    for step in 0..4 {
        let update = edits(sessions[0].current_witness(), n, step)?;
        let mut work = Vec::new();
        // Reverse the order for odd steps. Each family is first/last across
        // warmup versus sample, avoiding a permanently privileged control.
        for offset in 0..caps.len() {
            let index = if (step/2 + step) % 2 == 0 {offset} else {caps.len()-1-offset};
            let cap = caps[index];
            let s = &mut sessions[index];
            let before = s.backend().stats();
            let sparse_before = s.sparse_dispatch_stats();
            let started = Instant::now();
            let proof = s.update(&update)?;
            let wall = started.elapsed().as_nanos();
            let r = *s.last_receipt();
            let t = s.update_timings();
            let c = t.cache_detail;
            let after = s.backend().stats();
            let sparse = s.sparse_dispatch_stats();
            ensure!(r.changed == 16 && r.large_batch && r.swapped && !r.restarted
                && s.rebuild_progress().is_none(), "incomplete paid rebuild");
            ensure!(r.rebuild_units <= r.allowance, "work quota exceeded");
            ensure!(after.bytes_allocated == allocations[index], "resident buffers grew");
            ensure!(c.equations == (2*n) as u64 && c.final_exponentiations == c.equations
                && c.miller_calls == c.equations && c.pairing_terms == 2*c.equations,
                "independent Product equations changed");
            let fallback = (2*(n-n.min(cap))) as u64;
            // Each four-root fallback checker prepares one generator in
            // addition to its four divisors. Prepared-prefix calls build none.
            ensure!(c.divisors_formed == fallback && c.g2_prepare_builds == fallback*5/4
                && c.g2_prepare_hits + c.g2_prepare_builds == 4*n as u64,
                "prepared prefix counts differ: divisors={} builds={} hits={}",
                c.divisors_formed, c.g2_prepare_builds, c.g2_prepare_hits);
            ensure!(t.enabled && t.total_ns <= wall && c.recorded_ns() <= t.rebuild_ns[6]
                && t.linear_detail.recorded_ns() <= t.rebuild_ns[1]
                && t.delta_detail.recorded_ns() <= t.delta_proof_ns
                && t.rebase_detail.recorded_ns() <= t.rebase_ns,
                "nested timing exceeds parent scope");
            ensure!(r.frozen_linear_captured && r.foreground_linear_reused && r.global_g2_reused
                && r.cache_reused_points > 0, "full optimization mechanism missed");
            ensure!(sparse.cpu.g1_points == sparse_before.cpu.g1_points,
                "ForceGpu dispatched CPU sparse work");
            ensure!((r.transform_packet_calls > 0) == policy.packet_cap.is_some(),
                "transform strategy differs");
            let fr = after.fr_transform_elements-before.fr_transform_elements;
            let g1 = after.g1_transform_elements-before.g1_transform_elements;
            work.push((fr, g1, r.rebuild_units, r.allowance, c.equations));
            let started = Instant::now();
            ensure!(vk.verify_reference(&instance, &proof)?, "updated Reference rejected");
            let verify_ns = started.elapsed().as_nanos();
            let cache = json!({"equations":c.equations,"pairing_terms":c.pairing_terms,
                "miller_calls":c.miller_calls,"final_exponentiations":c.final_exponentiations,
                "g2_prepare_builds":c.g2_prepare_builds,"g2_prepare_hits":c.g2_prepare_hits,
                "divisors_formed":c.divisors_formed,"prepared_peak_entries":c.prepared_peak_entries,
                "read_batches":c.read_batches,"read_points":c.read_points,"max_read_roots":c.max_read_roots});
            emit(json!({"kind":"prepared_update","n":n,"cap":cap,"step":step,
                "order":offset,"measured":step>=2,"family":if step%2==1 {"addition"} else {"multiplier"},
                "ns":wall,"verify_ns":verify_ns,"correct":true,"changed":r.changed,
                "generation":proof.generation,"anchor_generation":proof.anchor_generation,
                "large_batch":r.large_batch,"swapped":r.swapped,"restarted":r.restarted,
                "pending":s.rebuild_progress().is_some(),"resident_bytes":after.bytes_allocated,
                "rebuild_units":r.rebuild_units.to_string(),"allowance":r.allowance.to_string(),
                "fr_elements":fr,"g1_elements":g1,
                "foreground_linear_work":r.foreground_linear_work.units()?.to_string(),
                "frozen_linear_captured":r.frozen_linear_captured,
                "foreground_linear_reused":r.foreground_linear_reused,"global_g2_reused":r.global_g2_reused,
                "global_g2_inputs":r.global_g2_inputs,"cache_reused_points":r.cache_reused_points,
                "support_geometry_reused":r.support_geometry_reused,"linear_reused":r.linear_reused,
                "transform_packet_calls":r.transform_packet_calls,"transform_packet_elements":r.transform_packet_elements,
                "sparse_gpu_g1_points":sparse.gpu.g1_points-sparse_before.gpu.g1_points,
                "sparse_cpu_g1_points":sparse.cpu.g1_points-sparse_before.cpu.g1_points,
                "cache":cache,"timings":timings(t)}));
        }
        for s in &sessions {
            ensure!(s.proof() == sessions[0].proof() && s.current_witness() == sessions[0].current_witness(),
                "paired proof/witness differs");
        }
        ensure!(work.iter().all(|item| *item == work[0]), "paired arithmetic/work differs");
        emit(json!({"kind":"prepared_pair","n":n,"step":step,"proof_equal":true,
            "witness_equal":true,"work_equal":true}));
    }
    // Independent dense CPU anchor at the completed endpoint, outside timing.
    let started = Instant::now();
    let expected = params.prove_anchor(&instance, sessions[0].current_witness())?;
    for s in &sessions { ensure!(s.proof().anchor == expected.anchor, "CPU dense anchor differs"); }
    emit(json!({"kind":"prepared_dense","n":n,"ns":started.elapsed().as_nanos(),"correct":true}));
    println!("PREPARED_CACHE_OK n={n} routes={} measured_steps=2", caps.len());
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: prepared_cache_probe <n> <scan|candidate-cap>");
    let n: usize = args[1].parse()?;
    let caps = if args[2] == "scan" { prepared_policy::scan_caps(n)? }
        else { vec![64, args[2].parse()?] };
    run(n, &caps)
}
