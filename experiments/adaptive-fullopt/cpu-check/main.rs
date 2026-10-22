//! Exercise the actual scheduling/cache implementation, without a GPU.
use anyhow::{ensure,Result};
use ark_bn254::Fr;
use dynaverse_protocol::{Update,dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::CpuRebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};

fn main()->Result<()> {
    let window:usize=std::env::args().nth(1).unwrap().parse()?;
    let n=64;
    let circuit=ToyCircuit::new(n)?;
    let(instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(1150))?;
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig{window:Some(window),foreground_linear_reuse:true,
            support_geometry_reuse:true,sparse:SparseDispatchPolicy::CpuReference,
            memory_limit_bytes:2<<30,..Default::default()},CpuRebuildBackend::new())?;
    let mut initial=Update{indices:Vec::new(),values:Vec::new()};
    for i in 1..=3 {
        let value=witness.wires[4*n+i]+Fr::from(1u64);
        initial.indices.extend([4*n+i,5*n+i]);
        initial.values.extend([value,witness.wires[3*n+i]*value]);
    }
    session.update(&initial)?;
    let mut hits=0;
    for step in 0..6 {
        let w=session.current_witness();
        let value=w.wires[4*n+1]+Fr::from(1u64);
        let proof=session.update(&Update{indices:vec![4*n+1,5*n+1],
            values:vec![value,w.wires[3*n+1]*value]})?;
        let receipt=session.last_receipt();
        hits+=usize::from(receipt.support_geometry_reused);
        ensure!(params.verifying_key().verify_reference(&instance,&proof)?,"proof rejected");
        println!("step={step} window={window} k={} swap={} hit={}",
            receipt.changed,receipt.swapped,receipt.support_geometry_reused);
    }
    ensure!(hits>=5,"geometry reuse expected at least5 hits, got {hits}");
    println!("GEOMETRY_OK hits={hits}");Ok(())
}
