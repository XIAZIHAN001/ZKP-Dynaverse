//! Bounded calibration, outside setup/update timing and on a disposable owner.
use super::*;
use dynaverse_protocol::incremental_rebuild::operators::BufferKind;

pub fn calibrate(n:usize,policy:strategy::Strategy)->Result<bool> {
    let count=policy.cache_roots;
    let mut owner=fullopt_backend(n,true,false)?;
    let mut rng=StdRng::seed_from_u64(2026102250+n as u64);
    let mut projective:Vec<_>=(0..count).map(|_|G1Projective::rand(&mut rng)).collect();
    projective[0]=G1Projective::zero();
    let points=G1Projective::normalize_batch(&projective);
    let scalars:Vec<_>=(0..count).map(|_|Fr::rand(&mut rng)).collect();
    let expected:Vec<_>=points.iter().zip(&scalars).map(|(p,s)|(*p * *s).into_affine()).collect();
    let input=owner.alloc(BufferKind::G1,count,true)?;
    let scalar=owner.alloc(BufferKind::Fr,count,true)?;
    let output=owner.alloc(BufferKind::G1,count,true)?;
    owner.write_points(input,0,&points)?;
    owner.write_fields(scalar,0,&scalars)?;
    owner.seal();
    owner.scale_points(input,0,scalar,0,output,0,count)?;
    let mut samples=[Vec::new(),Vec::new()];
    let mut physical=[0u64;2];
    for trial in 0..8 {
        for offset in 0..2 {
            let route=(trial+offset)%2;
            owner.set_device_affine_readback(route==1);
            let mut values=vec![G1Affine::default();count];
            let profile=owner.profile_read_points(output,0,&mut values)?;
            ensure!(values==expected,"calibration CPU/GPU readback mismatch");
            ensure!(profile.g1_cofactor_decode_points==count as u64
                && profile.g1_full_decode_points==0,"bounded decode was not executed");
            physical[route]=profile.native.d2h_bytes;
            if trial>=2 {samples[route].push(profile.total_ns);}
        }
    }
    for row in &mut samples {row.sort_unstable();}
    let median=|row:&Vec<u64>|(row[2]+row[3])/2;
    let host=median(&samples[0]);let device=median(&samples[1]);
    // Small workloads retain the locally verified host path. For larger
    // workloads accept device normalization only with a margin over noise.
    let selected=policy.calibrate_affine && (device as u128)*100<(host as u128)*97;
    emit(json!({"kind":"fullopt_readback_calibration","n":n,"points":count,
        "host_bounded_median_ns":host,"device_bounded_median_ns":device,
        "device_affine_readback_selected":selected,"bounded_g1_decode":true,
        "cofactor_decode_points_per_call":count,"full_decode_points_per_call":0,
        "host_d2h_bytes":physical[0],"device_d2h_bytes":physical[1],
        "samples_per_route":6,"selection_margin_percent":3,"correct":true}));
    Ok(selected)
}
