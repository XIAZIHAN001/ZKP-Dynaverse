#![cfg(feature="icicle")]
use ark_bn254::{Fr,G1Affine,G1Projective};
use ark_ec::{CurveGroup,Group};
use ark_ff::{Field,Zero};
use ark_poly::{EvaluationDomain,Radix2EvaluationDomain};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::incremental_rebuild::operators::{BufferKind,FourStepPlan,RebuildBackend};

#[test]
fn resident_matrix_and_batch_can_exceed_single_row_domain() {
    let n=1usize<<21;
    let mut backend=IcicleBackend::with_resident_limits(4<<30,1<<20).unwrap();
    let a=backend.alloc(BufferKind::Fr,n,true).unwrap();
    let b=backend.alloc(BufferKind::Fr,n,true).unwrap();
    backend.write_fields(a,n-1,&[Fr::from(7u64)]).unwrap();backend.seal();
    backend.transpose(BufferKind::Fr,a,b,1024,2048,n-1,1,None).unwrap();
    let mut actual=[Fr::zero()];backend.read_fields(b,n-1,&mut actual).unwrap();
    assert_eq!(actual[0],Fr::from(7u64));
    // 513 rows exceed the old 2^20 aggregate batch guard. The last row
    // contains one impulse; every frequency is independently known.
    backend.transform_rows(BufferKind::Fr,a,b,2048,511,513,false).unwrap();
    let domain=Radix2EvaluationDomain::<Fr>::new(2048).unwrap();
    let mut row=vec![Fr::zero();2048];backend.read_fields(b,n-2048,&mut row).unwrap();
    for (i,value) in row.iter().enumerate() {
        assert_eq!(*value,Fr::from(7u64)*domain.element(i).pow([2047u64]));
    }
    assert!(backend.transpose(BufferKind::Fr,a,b,2048,2048,0,0,None).is_err());
    assert!(backend.transform_rows(BufferKind::Fr,a,b,1<<21,0,1,false).is_err());
}

#[test]
#[ignore="explicit 2^21 matrix gate for million-gate circuits; not a full circuit test"]
fn full_2p21_four_step_fr_and_g1() {
    let n=1usize<<21;
    let domain=Radix2EvaluationDomain::<Fr>::new(n).unwrap();
    let input:Vec<Fr>=(0..n).map(|i|Fr::from(((i as u64*17+3)%65521)+1)).collect();
    for kind in [BufferKind::Fr,BufferKind::G1] {
        let mut backend=IcicleBackend::with_resident_limits(4<<30,1<<20).unwrap();
        let a=backend.alloc(kind,n,true).unwrap();
        let b=backend.alloc(kind,n,true).unwrap();
        let c=backend.alloc(kind,n,true).unwrap();
        let mut expected=if kind==BufferKind::Fr {input.clone()} else {vec![Fr::zero();n]};
        if kind==BufferKind::Fr {backend.write_fields(a,0,&input).unwrap();}
        else {
            // Two separated impulses expose transpose/twiddle mistakes.
            expected[1]=Fr::from(5u64);expected[n-17]=Fr::from(13u64);
            let mut points=vec![G1Affine::identity();n];
            points[1]=(G1Projective::generator()*expected[1]).into_affine();
            points[n-17]=(G1Projective::generator()*expected[n-17]).into_affine();
            backend.write_points(a,0,&points).unwrap();
        }
        backend.seal();
        let mut plan=FourStepPlan::new(kind,n,a,b,c,65536,32,false).unwrap();
        while !plan.next_step(&mut backend).unwrap() {}
        domain.fft_in_place(&mut expected);
        if kind==BufferKind::Fr {
            let mut actual=vec![Fr::zero();n];backend.read_fields(c,0,&mut actual).unwrap();
            assert_eq!(actual,expected);
        } else {
            // Independent CPU field FFT lifted to G1 at 65 frequencies.
            for index in (0..n).step_by(n/64).chain(std::iter::once(n-1)) {
                let mut actual=[G1Affine::identity()];backend.read_points(c,index,&mut actual).unwrap();
                assert_eq!(actual[0],(G1Projective::generator()*expected[index]).into_affine());
            }
        }
        let mut inverse=FourStepPlan::new(kind,n,c,b,a,65536,32,true).unwrap();
        while !inverse.next_step(&mut backend).unwrap() {}
        if kind==BufferKind::Fr {
            let mut actual=vec![Fr::zero();n];backend.read_fields(a,0,&mut actual).unwrap();
            assert_eq!(actual,input);
        } else {
            let mut actual=vec![G1Affine::identity();n];backend.read_points(a,0,&mut actual).unwrap();
            for (index,point) in actual.iter().enumerate() {
                let scalar=if index==1 {5} else if index==n-17 {13} else {0};
                assert_eq!(*point,(G1Projective::generator()*Fr::from(scalar as u64)).into_affine());
            }
        }
        eprintln!("LARGE_FOUR_STEP_OK length={n} kind={kind:?} forward_cpu=true inverse_all=true");
    }
}
