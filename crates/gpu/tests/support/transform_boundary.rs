use super::*;
use dynaverse_gpu::ResidentCallProfile;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn chain(gpu:&mut IcicleBackend,input:dynaverse_protocol::incremental_rebuild::operators::BufferId,
    scratch:dynaverse_protocol::incremental_rebuild::operators::BufferId,
    output:dynaverse_protocol::incremental_rebuild::operators::BufferId,
    len:usize,tile:usize,inverse:bool,profile:bool)->Vec<(usize,ResidentCallProfile)> {
    let a=1usize<<(len.trailing_zeros()/2);let b=len/a;
    let domain=Radix2EvaluationDomain::<Fr>::new(len).unwrap();
    let root=if inverse {domain.group_gen_inv} else {domain.group_gen};
    let mut records=Vec::new();
    for phase in 0..5 {
        let (size,step)=if phase==1 {(b,16)} else if phase==3 {(a,16)} else {(len,tile)};
        for start in (0..size).step_by(step) {
            let count=step.min(size-start);
            let result=if phase==1 || phase==3 {
                let row_len=if phase==1 {a} else {b};
                if profile {gpu.profile_transform_rows(BufferKind::G1,output,scratch,row_len,start,count,inverse).unwrap()}
                else {gpu.transform_rows(BufferKind::G1,output,scratch,row_len,start,count,inverse).unwrap();ResidentCallProfile::default()}
            } else {
                let (src,rows,cols,twiddle)=match phase {
                    0=>(input,a,b,None),2=>(scratch,b,a,Some(root)),4=>(scratch,a,b,None),_=>unreachable!()};
                if profile {gpu.profile_transpose(BufferKind::G1,src,output,rows,cols,start,count,twiddle).unwrap()}
                else {gpu.transpose(BufferKind::G1,src,output,rows,cols,start,count,twiddle).unwrap();ResidentCallProfile::default()}
            };
            records.push((phase,result));
        }
    }
    records
}

pub fn run() {
    let mut gpu=IcicleBackend::with_resident_limits(512<<20,8<<20).unwrap();
    let input=gpu.alloc(BufferKind::G1,2048,true).unwrap();let scratch=gpu.alloc(BufferKind::G1,2048,true).unwrap();
    let output=gpu.alloc(BufferKind::G1,2048,true).unwrap();gpu.seal();let allocated=gpu.stats().bytes_allocated;
    for n in [64usize,256,1024] {
        let len=2*n;let tile=(2*(1usize<<(n.trailing_zeros()/2))).max(16);
        let mut rng=StdRng::seed_from_u64(22100+n as u64);
        let values:Vec<_>=(0..len).map(|i|if i%17==0 {G1Affine::identity()} else {(G1Projective::generator()*Fr::rand(&mut rng)).into_affine()}).collect();
        let upload=gpu.profile_write_points(input,0,&values).unwrap();
        eprintln!("TP_IO n={n} len={len} kind=upload total_ns={} encoding_ns={} ffi_ns={} native_conversion_ns={} event_operation_ns={} bytes={}",
            upload.total_ns,upload.encoding_ns,upload.ffi_ns,upload.native.conversion_host_ns,upload.native.event_operation_ns,upload.native.h2d_bytes);
        for inverse in [false,true] {for repeat in 0..4 {
            let mut expected:Vec<G1Projective>=values.iter().map(|p|(*p).into()).collect();
            let domain=Radix2EvaluationDomain::<Fr>::new(len).unwrap();let started=Instant::now();
            if inverse {domain.ifft_in_place(&mut expected)} else {domain.fft_in_place(&mut expected)}
            let cpu_ns=started.elapsed().as_nanos();let expected=G1Projective::normalize_batch(&expected);
            for offset in 0..2 {
                let profiled=(repeat+offset)%2==1;gpu.begin_transaction().unwrap();
                let before=gpu.stats();let diag=gpu.resident_diagnostics().unwrap();let started=Instant::now();
                let records=chain(&mut gpu,input,scratch,output,len,tile,inverse,profiled);
                let wall=started.elapsed().as_nanos();let after=gpu.stats();let now=gpu.resident_diagnostics().unwrap();
                assert_eq!((after.h2d_bytes-before.h2d_bytes,after.d2h_bytes-before.d2h_bytes),(0,0));
                assert_eq!(after.g1_transform_elements-before.g1_transform_elements,2*len as u64);
                assert_eq!(now.explicit_d2d_bytes-diag.explicit_d2d_bytes,2*len as u64*96);
                let mut actual=vec![G1Affine::identity();len];let download=gpu.profile_read_points(output,0,&mut actual).unwrap();
                assert_eq!(actual,expected);assert_eq!(gpu.stats().bytes_allocated,allocated);
                eprintln!("TP_BOUNDARY n={n} len={len} inverse={} repeat={repeat} warm={} profiled={} wall_ns={wall} cpu_fft_ns={cpu_ns} calls={} event_total_ns={} h2d={} d2h={} journal_d2d={} download_total_ns={} download_native_conversion_ns={} download_event_ns={}",
                    usize::from(inverse),usize::from(repeat==0),usize::from(profiled),after.calls-before.calls,now.device_operation_ns-diag.device_operation_ns,
                    now.physical_h2d_bytes-diag.physical_h2d_bytes,now.physical_d2h_bytes-diag.physical_d2h_bytes,now.explicit_d2d_bytes-diag.explicit_d2d_bytes,
                    download.total_ns,download.native.conversion_host_ns,download.native.event_operation_ns);
                if profiled {for phase in 0..5 {
                    let rr:Vec<_>=records.iter().filter(|(p,_)|*p==phase).map(|(_,r)|r).collect();
                    eprintln!("TP_PHASE n={n} inverse={} repeat={repeat} phase={phase} calls={} host_total_ns={} native_ns={} native_journal_host_ns={} native_operation_host_ns={} native_wait_host_ns={} event_journal_ns={} event_operation_ns={} event_total_ns={}",
                        usize::from(inverse),rr.len(),rr.iter().map(|r|r.total_ns).sum::<u64>(),rr.iter().map(|r|r.native.native_wall_ns).sum::<u64>(),
                        rr.iter().map(|r|r.native.journal_host_ns).sum::<u64>(),rr.iter().map(|r|r.native.operation_host_ns).sum::<u64>(),rr.iter().map(|r|r.native.wait_host_ns).sum::<u64>(),
                        rr.iter().map(|r|r.native.event_journal_ns).sum::<u64>(),rr.iter().map(|r|r.native.event_operation_ns).sum::<u64>(),rr.iter().map(|r|r.native.event_total_ns).sum::<u64>());
                }}
                gpu.rollback_transaction().unwrap();
            }
        }}
    }
}
