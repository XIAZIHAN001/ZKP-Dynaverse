#![cfg(feature="icicle")]
use ark_bn254::Fr;
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,UpdateTimings,
        operators::RebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

fn local_scales(sizes:&[usize],paired:bool) {
    for &n in sizes {
        let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let started=Instant::now();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18300+n as u64)).unwrap();
        eprintln!("SCALE_CASE n={n} gates={} setup_ns={}",2*n,started.elapsed().as_nanos());
        for workload in ["multiplier","public_preserve"] {
            let mut sessions=Vec::new();
            let mut allocations=Vec::new();
            for route in 0..if paired {2} else {1} {
            let gpu=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
            let started=Instant::now();
            let session=params.start_incremental_session(&instance,&witness,
                IncrementalConfig {linear_rebuild_reuse:true,trace_updates:true,
                    unchanged_cache_reuse:route==1,
                    sparse:SparseDispatchPolicy::ForceGpu,..Default::default()},gpu).unwrap();
            let allocated=session.backend().stats().bytes_allocated;
            eprintln!("SCALE_INITIAL n={n} workload={workload} route={route} ns={} resident={allocated}",started.elapsed().as_nanos());
            sessions.push(session);allocations.push(allocated);
            }
            // n1024 has H=16; 32 calls cross multiple paid rebuild epochs.
            for step in 0..if n==1024 {32} else {8} {
                let i=step+1;
                let session=&sessions[0];
                let (update,public)=if workload=="multiplier" {
                    let next=session.current_witness().wires[4*n+i]+Fr::from(1u64);
                    (Update {indices:vec![4*n+i,5*n+i],values:vec![next,session.current_witness().wires[3*n+i]*next]},
                        Update {indices:vec![],values:vec![]})
                } else {
                    let left=session.current_witness().wires[i]+Fr::from(1u64);
                    let right=session.current_witness().wires[n+i]-Fr::from(1u64);
                    (Update {indices:vec![i,n+i],values:vec![left,right]},Update {indices:vec![i],values:vec![left]})
                };
                // Same edits for both owners, alternating which route runs first.
                let order=if paired {vec![step%2,1-step%2]} else {vec![0]};
                for route in order {
                let session=&mut sessions[route];
                let before=session.backend().stats();let sb=session.sparse_dispatch_stats();
                let started=Instant::now();let proof=session.update_with_public_inputs(&update,&public).unwrap();
                let wall=started.elapsed().as_nanos();
                let r=session.last_receipt();let t=session.update_timings();
                let after=session.backend().stats();let sa=session.sparse_dispatch_stats();
                eprintln!("SCALE_UPDATE n={n} workload={workload} route={route} step={step} r={} k={} a={} d={} ds={} o={} wires={} frozen_wires={} start={} swap={} restart={} linear={} wall_ns={wall} total_ns={} norm_ns={} constraints_ns={} prepare_ns={} publish_ns={} delta_ns={} rebase_ns={} allowance={} work={} resident_g1={} resident_g2={} sparse_g1={} sparse_g2={} fr_elements={} g1_elements={} h2d={} d2h={} cache_mask={} cache_points={}",
                    r.raw_edits,r.changed,r.affected_gates,r.delta_entries,r.frozen_entries,r.overlay_entries,
                    r.changed_wire_mask,r.frozen_wire_mask,r.started,r.swapped,r.restarted,r.linear_reused,
                    t.total_ns,t.normalization_ns,t.constraints_ns,t.preparation_ns,t.publication_ns,
                    t.delta_proof_ns,t.rebase_ns,r.allowance,r.rebuild_units,
                    after.msm_g1_points-before.msm_g1_points,after.msm_g2_points-before.msm_g2_points,
                    sa.cpu.g1_points+sa.gpu.g1_points-sb.cpu.g1_points-sb.gpu.g1_points,
                    sa.cpu.g2_points+sa.gpu.g2_points-sb.cpu.g2_points-sb.gpu.g2_points,
                    after.fr_transform_elements-before.fr_transform_elements,
                    after.g1_transform_elements-before.g1_transform_elements,
                    after.h2d_bytes-before.h2d_bytes,after.d2h_bytes-before.d2h_bytes,
                    r.cache_reused_wire_mask,r.cache_reused_points);
                for (phase,name) in UpdateTimings::REBUILD_NAMES.iter().enumerate() {
                    eprintln!("SCALE_PHASE n={n} workload={workload} route={route} step={step} phase={name} ns={} calls={}",t.rebuild_ns[phase],t.rebuild_calls[phase]);
                }
                let started=Instant::now();
                assert!(params.verifying_key().verify_reference(session.current_instance(),&proof).unwrap());
                eprintln!("SCALE_VERIFY n={n} workload={workload} route={route} step={step} ns={}",started.elapsed().as_nanos());
                assert_eq!(session.backend().stats().bytes_allocated,allocations[route]);
                }
                if paired {assert_eq!(sessions[0].current_witness(),sessions[1].current_witness());}
            }
        }
    }
}

#[test]
#[ignore="explicit local n64/256 dynamic baseline, no H200 claim"]
fn local_dynamic_scale_baseline() {local_scales(&[64,256],false);}

#[test]
#[ignore="explicit paired R1a vs R1a+R2a diagnostic, not H200"]
fn local_dynamic_scale_cache_comparison() {local_scales(&[64,256],true);}

#[test]
#[ignore="explicit n1024 enlargement only after n256 admission"]
fn local_dynamic_scale_1024() {local_scales(&[1024],true);}
