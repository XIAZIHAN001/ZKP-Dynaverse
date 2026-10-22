#![cfg(feature="icicle")]
use ark_bn254::{Fr,G1Affine,G1Projective};
use ark_ec::{AffineRepr,CurveGroup,Group};
use ark_ff::{Field,UniformRand,Zero};
use ark_poly::{EvaluationDomain,Radix2EvaluationDomain};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::incremental_rebuild::operators::{BufferKind,RebuildBackend};
use rand::{rngs::StdRng,SeedableRng};
#[path="support/transform_boundary.rs"]
mod transform_boundary;

#[test]
fn resident_profile_preserves_output_and_splits_clocks_without_post_seal_allocations() {
    let mut gpu=IcicleBackend::with_resident_limits(256<<20,1<<20).unwrap();
    let input=gpu.alloc(BufferKind::G1,32,true).unwrap();
    let output=gpu.alloc(BufferKind::G1,32,true).unwrap();
    let mut rng=StdRng::seed_from_u64(22000);
    let values:Vec<_>=(0..32).map(|i|if i%7==0 {G1Affine::identity()} else {(G1Projective::generator()*Fr::rand(&mut rng)).into_affine()}).collect();
    let upload=gpu.profile_write_points(input,0,&values).unwrap();
    gpu.seal();let before=gpu.resident_diagnostics().unwrap();let allocated=gpu.stats().bytes_allocated;
    gpu.begin_transaction().unwrap();
    let profile=gpu.profile_transform_rows(BufferKind::G1,input,output,8,1,2,false).unwrap();
    let mut actual=vec![G1Affine::identity();32];let download=gpu.profile_read_points(output,0,&mut actual).unwrap();
    let mut expected=vec![G1Affine::identity();32];let domain=Radix2EvaluationDomain::<Fr>::new(8).unwrap();
    for row in 1..3 {
        let mut projective:Vec<G1Projective>=values[row*8..row*8+8].iter().map(|p|(*p).into()).collect();
        domain.fft_in_place(&mut projective);expected[row*8..row*8+8].copy_from_slice(&G1Projective::normalize_batch(&projective));
    }
    assert_eq!(actual,expected);
    assert_eq!(gpu.stats().bytes_allocated,allocated);
    assert_eq!(gpu.resident_diagnostics().unwrap().resident_allocations,before.resident_allocations);
    assert_ne!(profile.native.event_mask,0,"profile has no native clock split");
    assert!(profile.native.native_wall_ns>0 && profile.native.wait_host_ns>0);
    assert!(profile.native.event_total_ns>=profile.native.event_operation_ns);
    assert_eq!((profile.native.h2d_bytes,profile.native.d2h_bytes),(0,0));
    assert_eq!(profile.native.d2d_bytes,16*96);
    assert_eq!(upload.native.h2d_bytes,32*96);
    assert_eq!(download.native.d2h_bytes,32*96);
    assert!(profile.ffi_ns>=profile.native.native_wall_ns && profile.total_ns>=profile.ffi_ns);
    gpu.rollback_transaction().unwrap();gpu.read_points(output,0,&mut actual).unwrap();
    assert!(actual.iter().all(AffineRepr::is_zero));
    assert!(gpu.profile_transform_rows(BufferKind::G1,input,input,8,0,1,false).is_err());
    assert!(gpu.memory_info().is_ok());
    eprintln!("TP_PROFILE_EQUAL profile={profile:?} upload={upload:?} download={download:?}");
}

#[test]
fn resident_profile_zero_and_rejected_calls_keep_owner_usable() {
    let mut gpu=IcicleBackend::with_resident_limits(256<<20,1<<20).unwrap();
    let fields=gpu.alloc(BufferKind::Fr,32,true).unwrap();let out=gpu.alloc(BufferKind::Fr,32,true).unwrap();
    let packed=gpu.register_msm_bases(&[G1Affine::identity();32]).unwrap();
    let values:Vec<_>=(0..32).map(|i|Fr::from(i as u64+11)).collect();
    let upload=gpu.profile_write_fields(fields,0,&values).unwrap();assert_eq!(upload.native.h2d_bytes,32*32);
    gpu.seal();let bytes=gpu.stats().bytes_allocated;let mut recovered=vec![Fr::zero();32];
    let empty=gpu.profile_read_fields(fields,32,&mut[]).unwrap();assert_eq!(empty.native.event_mask,0);
    let empty=gpu.profile_transform_rows(BufferKind::Fr,fields,out,8,4,0,false).unwrap();assert_eq!(empty.native.event_mask,0);
    assert!(gpu.profile_transform_rows(BufferKind::Fr,fields,out,3,0,1,false).is_err());
    assert!(gpu.profile_transform_rows(BufferKind::Fr,fields,fields,8,0,1,false).is_err());
    assert!(gpu.profile_transform_rows(BufferKind::G1,packed,packed,8,0,1,false).is_err());
    assert!(gpu.profile_transpose(BufferKind::Fr,fields,out,4,8,31,2,None).is_err());
    let mut other=IcicleBackend::with_resident_limits(256<<20,1<<20).unwrap();
    let foreign=other.alloc(BufferKind::Fr,32,true).unwrap();
    assert!(gpu.profile_transform_rows(BufferKind::Fr,foreign,out,8,0,1,false).is_err());
    let read=gpu.profile_read_fields(fields,0,&mut recovered).unwrap();
    assert_eq!(recovered,values);assert_eq!(read.native.d2h_bytes,32*32);
    gpu.begin_transaction().unwrap();
    let trace=gpu.profile_transform_rows(BufferKind::Fr,fields,out,8,1,2,true).unwrap();
    gpu.profile_read_fields(out,0,&mut recovered).unwrap();
    let domain=Radix2EvaluationDomain::<Fr>::new(8).unwrap();let mut expected=vec![Fr::zero();32];
    for row in 1..3 {expected[row*8..row*8+8].copy_from_slice(&domain.ifft(&values[row*8..row*8+8]));}
    assert_eq!(recovered,expected);assert_eq!(trace.native.d2d_bytes,16*32);
    gpu.profile_transpose(BufferKind::Fr,fields,out,4,8,0,32,Some(domain.group_gen)).unwrap();
    gpu.profile_read_fields(out,0,&mut recovered).unwrap();
    for i in 0..32 {expected[(i%8)*4+i/8]=values[i]*domain.group_gen.pow([((i/8)*(i%8)) as u64]);}
    assert_eq!(recovered,expected);gpu.rollback_transaction().unwrap();
    assert!(gpu.memory_info().is_ok());assert_eq!(gpu.stats().bytes_allocated,bytes);
}

#[test]
#[ignore="explicit serial resident boundary profile; normal/profile clocks kept separate"]
fn local_resident_boundary_profile() {transform_boundary::run();}

#[test]
fn native_packet_matches_independent_fft_in_one_submission() {
    let mut gpu=IcicleBackend::with_resident_limits(512<<20,4<<20).unwrap();
    let mut buffers=Vec::new();
    for kind in [BufferKind::Fr,BufferKind::G1] {
        for slot in 0..3 {buffers.push(gpu.alloc(kind,4096,slot!=0).unwrap());}
    }
    let mut rng=StdRng::seed_from_u64(23200);
    let mut values:Vec<Fr>=(0..4096).map(|_|Fr::rand(&mut rng)).collect();values[3]=-Fr::from(1u64);values[7]=Fr::zero();
    let points:Vec<_>=(0..4096).map(|i|if i%17==0 {G1Affine::identity()} else {(G1Projective::generator()*Fr::rand(&mut rng)).into_affine()}).collect();
    gpu.write_fields(buffers[0],0,&values).unwrap();gpu.write_points(buffers[3],0,&points).unwrap();gpu.seal();
    let allocated=gpu.stats().bytes_allocated;
    for len in [1usize,2,8,32,128,1024,4096] {for inverse in [false,true] {for (index,kind) in [BufferKind::Fr,BufferKind::G1].into_iter().enumerate() {
        let ids=&buffers[index*3..index*3+3];gpu.begin_transaction().unwrap();let before=gpu.stats();let diag=gpu.resident_diagnostics().unwrap();
        gpu.transform_four_step(kind,ids[0],ids[1],ids[2],len,inverse).unwrap();let after=gpu.stats();let now=gpu.resident_diagnostics().unwrap();
        let domain=Radix2EvaluationDomain::<Fr>::new(len).unwrap();
        if kind==BufferKind::Fr {
            let expected=if inverse {domain.ifft(&values[..len])} else {domain.fft(&values[..len])};
            let mut actual=vec![Fr::zero();len];gpu.read_fields(ids[2],0,&mut actual).unwrap();assert_eq!(actual,expected);
            assert_eq!(after.fr_transform_elements-before.fr_transform_elements,2*len as u64);
        } else {
            let mut expected:Vec<G1Projective>=points[..len].iter().map(|p|(*p).into()).collect();
            if inverse {domain.ifft_in_place(&mut expected)} else {domain.fft_in_place(&mut expected)}
            let mut actual=vec![G1Affine::identity();len];gpu.read_points(ids[2],0,&mut actual).unwrap();assert_eq!(actual,G1Projective::normalize_batch(&expected));
            assert_eq!(after.g1_transform_elements-before.g1_transform_elements,2*len as u64);
        }
        assert_eq!(after.calls-before.calls,1,"transform is not one native packet");
        assert_eq!((now.physical_h2d_bytes-diag.physical_h2d_bytes,now.physical_d2h_bytes-diag.physical_d2h_bytes),(0,0));
        // Singleton rows enqueue explicit copies in addition to pair journaling.
        let row_copies=if len==1 {2} else if len==2 {1} else {0};
        assert_eq!(now.explicit_d2d_bytes-diag.explicit_d2d_bytes,(2+row_copies)*len as u64*if kind==BufferKind::Fr {32} else {96});
        gpu.rollback_transaction().unwrap();assert_eq!(gpu.stats().bytes_allocated,allocated);
        eprintln!("TP_NATIVE_EQUAL len={len} inverse={} kind={kind:?}",usize::from(inverse));
    }}}
}

#[test]
fn native_packet_preflights_remaining_journal_before_either_output_changes() {
    let mut gpu=IcicleBackend::with_resident_limits(256<<20,2*32*96).unwrap();
    let input=gpu.alloc(BufferKind::G1,32,false).unwrap();let scratch=gpu.alloc(BufferKind::G1,32,true).unwrap();
    let output=gpu.alloc(BufferKind::G1,32,true).unwrap();let prior=gpu.alloc(BufferKind::Fr,64,true).unwrap();
    let values:Vec<_>=(0..32).map(|i|(G1Projective::generator()*Fr::from(i as u64+1)).into_affine()).collect();
    let sent_s=vec![(G1Projective::generator()*Fr::from(37u64)).into_affine();32];
    let sent_o=vec![(G1Projective::generator()*Fr::from(41u64)).into_affine();32];
    gpu.write_points(input,0,&values).unwrap();gpu.write_points(scratch,0,&sent_s).unwrap();gpu.write_points(output,0,&sent_o).unwrap();gpu.seal();
    gpu.begin_transaction().unwrap();gpu.write_fields(prior,0,&[Fr::from(9u64);64]).unwrap();
    let before=gpu.resident_diagnostics().unwrap();
    assert!(gpu.transform_four_step(BufferKind::G1,input,scratch,output,32,false).is_err());
    assert_eq!(gpu.resident_diagnostics().unwrap().journal_saved_bytes,before.journal_saved_bytes,"packet saved/mutated a partial output before whole preflight");
    let mut actual=vec![G1Affine::identity();32];gpu.read_points(scratch,0,&mut actual).unwrap();assert_eq!(actual,sent_s);
    gpu.read_points(output,0,&mut actual).unwrap();assert_eq!(actual,sent_o);assert!(gpu.memory_info().is_ok());
    gpu.rollback_transaction().unwrap();gpu.begin_transaction().unwrap();
    gpu.transform_four_step(BufferKind::G1,input,scratch,output,32,false).unwrap();gpu.rollback_transaction().unwrap();
    gpu.read_points(output,0,&mut actual).unwrap();assert_eq!(actual,sent_o);
}

#[test]
fn native_packet_rejections_and_profile_keep_owner_and_outputs_usable() {
    let mut gpu=IcicleBackend::with_resident_limits(256<<20,1<<20).unwrap();
    let input=gpu.alloc(BufferKind::Fr,128,false).unwrap();let scratch=gpu.alloc(BufferKind::Fr,128,true).unwrap();
    let output=gpu.alloc(BufferKind::Fr,128,true).unwrap();let readonly=gpu.alloc(BufferKind::Fr,128,false).unwrap();
    let packed=gpu.register_msm_bases(&[G1Affine::identity();128]).unwrap();
    let values:Vec<_>=(0..128).map(|i|Fr::from(i as u64+7)).collect();gpu.write_fields(input,0,&values).unwrap();gpu.seal();
    let mut other=IcicleBackend::with_resident_limits(256<<20,1<<20).unwrap();
    let foreign=other.alloc(BufferKind::Fr,128,true).unwrap();
    gpu.begin_transaction().unwrap();let before=gpu.resident_diagnostics().unwrap();
    for len in [0,3,256,8192] {assert!(gpu.transform_four_step(BufferKind::Fr,input,scratch,output,len,false).is_err());}
    for ids in [[input,input,output],[input,scratch,input],[input,output,output],
        [foreign,scratch,output],[input,foreign,output],[input,scratch,foreign],
        [input,readonly,output],[input,scratch,readonly],[packed,scratch,output],
        [input,packed,output],[input,scratch,packed]] {
        assert!(gpu.transform_four_step(BufferKind::Fr,ids[0],ids[1],ids[2],128,false).is_err());
    }
    assert_eq!(gpu.resident_diagnostics().unwrap().journal_saved_bytes,before.journal_saved_bytes);
    let trace=gpu.profile_transform_four_step(BufferKind::Fr,input,scratch,output,128,true).unwrap();
    assert_eq!(trace.native.event_mask,7);assert_eq!(trace.native.d2d_bytes,2*128*32);
    assert_eq!((trace.native.h2d_bytes,trace.native.d2h_bytes),(0,0));
    assert!(trace.native.event_total_ns>=trace.native.event_operation_ns && trace.native.wait_host_ns>0);
    let mut actual=vec![Fr::zero();128];gpu.read_fields(output,0,&mut actual).unwrap();
    assert_eq!(actual,Radix2EvaluationDomain::<Fr>::new(128).unwrap().ifft(&values));
    gpu.rollback_transaction().unwrap();
    for id in [scratch,output] {gpu.read_fields(id,0,&mut actual).unwrap();assert!(actual.iter().all(Zero::is_zero));}
    assert_eq!(gpu.resident_diagnostics().unwrap().resident_allocations,before.resident_allocations);
    assert!(gpu.memory_info().is_ok());
}

#[path="../../protocol/tests/support/packet_profile.rs"]
mod packet_profile;

#[test]
fn native_packet_full_proof_admission_matches_cpu_and_original_schedule() {
    use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
        incremental_rebuild::{SparseDispatchPolicy,operators::CpuRebuildBackend}};
    for (n,cap) in [(16,16),(16,64),(64,16),(64,128)] {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(24000+n as u64+cap as u64)).unwrap();
        let mut cfg=packet_profile::config("packet",SparseDispatchPolicy::ForceGpu);cfg.transform_packet_max_len=Some(cap);
        let mut cpu_cfg=cfg.clone();cpu_cfg.sparse=SparseDispatchPolicy::CpuReference;
        let mut cpu=p.start_incremental_session(&i,&w,cpu_cfg,CpuRebuildBackend::default()).unwrap();
        let mut gpu=p.start_incremental_session(&i,&w,cfg,IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap()).unwrap();
        let mut old=p.start_incremental_session(&i,&w,packet_profile::config("row16",SparseDispatchPolicy::ForceGpu),IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap()).unwrap();
        let allocated=gpu.backend().stats().bytes_allocated;let allocations=gpu.backend().resident_diagnostics().unwrap().resident_allocations;
        for step in 0..3 {
            let u=packet_profile::edits(cpu.current_witness(),n,if step%2==0 {8} else {4},step%2==1,24001+step as u64);
            assert_eq!(cpu.update(&u).unwrap(),gpu.update(&u).unwrap());old.update(&u).unwrap();assert_eq!(cpu.proof(),old.proof());
            assert_eq!(cpu.last_receipt(),gpu.last_receipt());assert!(gpu.last_receipt().swapped);
            assert_eq!(gpu.last_receipt().transform_packet_calls>0,cap>=n);
            assert_eq!(gpu.backend().stats().bytes_allocated,allocated);
            assert_eq!(gpu.backend().resident_diagnostics().unwrap().resident_allocations,allocations);
            assert_eq!(gpu.proof().anchor,p.prove_anchor(&i,gpu.current_witness()).unwrap().anchor);
            assert!(p.verifying_key().verify_reference(&i,gpu.proof()).unwrap());
            eprintln!("TP_ADMISSION n={n} cap={cap} step={step} packet_calls={} packet_elements={}",gpu.last_receipt().transform_packet_calls,gpu.last_receipt().transform_packet_elements);
        }
    }
}

#[test]
#[ignore="explicit serial row16 versus packet, three repeats and complete paid work"]
fn local_transform_packet_gpu_scaling() {packet_profile::profile("gpu",false,||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());}

#[test]
#[ignore="explicit serial k2 pending rebuild latency diagnostic"]
fn local_transform_packet_gpu_pending() {packet_profile::pending_profile("gpu",||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());}

#[test]
#[ignore="explicit serial normal versus instrumented packet probe; clocks remain separate"]
fn local_transform_packet_micro_profile() {
    use dynaverse_protocol::incremental_rebuild::operators::FourStepPlan;
    use std::time::Instant;
    let mut gpu=IcicleBackend::with_resident_limits(512<<20,4<<20).unwrap();
    let input=gpu.alloc(BufferKind::G1,2048,false).unwrap();let scratch=gpu.alloc(BufferKind::G1,2048,true).unwrap();
    let output=gpu.alloc(BufferKind::G1,2048,true).unwrap();let mut rng=StdRng::seed_from_u64(24500);
    let values:Vec<_>=(0..2048).map(|i|if i%17==0 {G1Affine::identity()} else {(G1Projective::generator()*Fr::rand(&mut rng)).into_affine()}).collect();
    gpu.write_points(input,0,&values).unwrap();gpu.seal();let allocated=gpu.stats().bytes_allocated;
    for len in [128usize,512,2048] {for inverse in [false,true] {
        let domain=Radix2EvaluationDomain::<Fr>::new(len).unwrap();
        let mut expected:Vec<G1Projective>=values[..len].iter().map(|p|(*p).into()).collect();
        if inverse {domain.ifft_in_place(&mut expected)} else {domain.fft_in_place(&mut expected)}
        let expected=G1Projective::normalize_batch(&expected);
        for repeat in 0..4 {for offset in 0..3 {
            let route=["row16","packet","profile"][(repeat+offset)%3];
            let tile=1usize<<len.trailing_zeros().div_ceil(2);
            let mut plan=FourStepPlan::new(BufferKind::G1,len,input,scratch,output,tile,16,inverse).unwrap();
            gpu.begin_transaction().unwrap();let before=gpu.stats();let diag=gpu.resident_diagnostics().unwrap();let start=Instant::now();
            let trace=if route=="profile" {Some(gpu.profile_transform_four_step(BufferKind::G1,input,scratch,output,len,inverse).unwrap())} else {
                if route=="packet" {gpu.transform_four_step(BufferKind::G1,input,scratch,output,len,inverse).unwrap();}
                else {while !plan.next_step(&mut gpu).unwrap() {}}
                None
            };
            let wall=start.elapsed().as_nanos();let after=gpu.stats();let now=gpu.resident_diagnostics().unwrap();
            let mut actual=vec![G1Affine::identity();len];gpu.read_points(output,0,&mut actual).unwrap();assert_eq!(actual,expected);
            assert_eq!(after.g1_transform_elements-before.g1_transform_elements,2*len as u64);
            assert_eq!(after.bytes_allocated,allocated);assert_eq!(now.resident_allocations,diag.resident_allocations);
            let p=trace.unwrap_or_default();
            eprintln!("TPM_SAMPLE len={len} inverse={} repeat={repeat} route={route} warm={} wall_ns={wall} calls={} journal_bytes={} journal_ranges={} d2d_bytes={} h2d_bytes={} d2h_bytes={} native_wall_ns={} journal_host_ns={} operation_host_ns={} wait_host_ns={} event_total_ns={} event_journal_ns={} event_operation_ns={} event_mask={}",
                usize::from(inverse),usize::from(repeat==0),after.calls-before.calls,now.journal_saved_bytes-diag.journal_saved_bytes,now.journal_saved_ranges-diag.journal_saved_ranges,
                now.explicit_d2d_bytes-diag.explicit_d2d_bytes,now.physical_h2d_bytes-diag.physical_h2d_bytes,now.physical_d2h_bytes-diag.physical_d2h_bytes,
                p.native.native_wall_ns,p.native.journal_host_ns,p.native.operation_host_ns,p.native.wait_host_ns,p.native.event_total_ns,p.native.event_journal_ns,p.native.event_operation_ns,p.native.event_mask);
            gpu.rollback_transaction().unwrap();
        }}
    }}
}
