//! External measurement harness. Product files and native ABI are unchanged.
use anyhow::{ensure,Result};
use ark_bn254::{Fr,G1Affine,G1Projective,G2Affine};
use ark_ec::CurveGroup;
use ark_ff::{UniformRand,Zero};
use dynaverse_gpu::{ArkworksBackend,G2ValidationMode,IcicleBackend,SetupMode};
use dynaverse_protocol::{OperatorBackend,BatchMsmRequest,SparseMsmRequest,GpuCapabilities,
    AnchorOpeningPolicy,BatchOpenMode,VerificationMode,Update,Witness,
    dynaverse::DynaverseParams,toy_circuit::ToyCircuit,trace::CpuTrace,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::RebuildBackend},workcount};
use rand::{rngs::StdRng,SeedableRng};
use serde_json::{json,Value};
use std::{cell::Cell,time::Instant};
mod strategy;
mod readback;

// Intentionally omit msm_sparse_batch: its real default gathers into the
// same checked dense batch API. All other capabilities forward unchanged.
struct Gather<'a>(&'a IcicleBackend);
impl OperatorBackend for Gather<'_> {
    fn msm(&self,p:&[G1Affine],s:&[Fr])->Result<G1Affine>{self.0.msm(p,s)}
    fn msm_batch(&self,r:&BatchMsmRequest<'_>)->Result<Vec<G1Affine>>{self.0.msm_batch(r)}
    fn ntt(&self,v:&mut[Fr],i:bool)->Result<()>{self.0.ntt(v,i)}
    fn fixed_base_g1(&self,s:&[Fr])->Result<Vec<G1Affine>>{self.0.fixed_base_g1(s)}
    fn fixed_base_g2(&self,s:&[Fr])->Result<Vec<G2Affine>>{self.0.fixed_base_g2(s)}
    fn msm_g2_host(&self,p:&[G2Affine],s:&[Fr])->Result<G2Affine>{self.0.msm_g2_host(p,s)}
    fn gpu_capabilities(&self)->GpuCapabilities{self.0.gpu_capabilities()}
}

// Count indexed dispatches only in the diagnostic Delta trial, equally for
// all routes. This establishes that timed protocol calls reach this interface.
struct Observed<'a>{inner:&'a dyn OperatorBackend,sparse:Cell<(u64,u64)>}
impl OperatorBackend for Observed<'_> {
    fn msm(&self,p:&[G1Affine],s:&[Fr])->Result<G1Affine>{self.inner.msm(p,s)}
    fn msm_batch(&self,r:&BatchMsmRequest<'_>)->Result<Vec<G1Affine>>{self.inner.msm_batch(r)}
    fn msm_sparse_batch(&self,r:&[SparseMsmRequest<'_>])->Result<Vec<G1Affine>>{
        let (calls,points)=self.sparse.get();
        self.sparse.set((calls+1,points+r.iter().map(|r|r.entries.len() as u64).sum::<u64>()));
        self.inner.msm_sparse_batch(r)
    }
    fn ntt(&self,v:&mut[Fr],i:bool)->Result<()>{self.inner.ntt(v,i)}
    fn fixed_base_g1(&self,s:&[Fr])->Result<Vec<G1Affine>>{self.inner.fixed_base_g1(s)}
    fn fixed_base_g2(&self,s:&[Fr])->Result<Vec<G2Affine>>{self.inner.fixed_base_g2(s)}
    fn msm_g2_host(&self,p:&[G2Affine],s:&[Fr])->Result<G2Affine>{self.inner.msm_g2_host(p,s)}
    fn gpu_capabilities(&self)->GpuCapabilities{self.inner.gpu_capabilities()}
}
const ROUTES:[&str;3]=["indexed","gather","cpu"];
fn emit(row:Value){println!("ROW {row}");}

fn msm()->Result<()> {
    let mut rng=StdRng::seed_from_u64(2026102207);
    let mut cases=0;
    for n in [4096usize,65536] {
        let projective:Vec<_>=(0..n).map(|_|G1Projective::rand(&mut rng)).collect();
        let bases=G1Projective::normalize_batch(&projective);
        let opposite:Vec<_>=bases.iter().map(|p|-*p).collect();
        for d in [16usize,256,4096] {for batch in [1usize,8] {
            for shared in [true,false] {for pattern in ["hotspot","scattered"] {
                let mut rows=Vec::new();
                for r in 0..batch {
                    let entries:Vec<_>=(0..d).map(|j|{
                        let index=if pattern=="hotspot" {j%(n/64)} else {(j*4099+r*17)%n};
                        (index,Fr::rand(&mut rng))
                    }).collect();rows.push(entries);
                }
                let requests:Vec<_>=(0..batch).map(|r|SparseMsmRequest {
                    bases:if r%2==0 {&bases} else {&opposite},entries:if shared {&rows[0]} else {&rows[r]}
                }).collect();
                let expected:Vec<_>=requests.iter().map(|r|{
                    let mut sum=G1Projective::zero();
                    for &(i,s) in r.entries {sum+=r.bases[i]*s;}sum.into_affine()
                }).collect();
                let started=Instant::now();let indexed=IcicleBackend::new()?;
                emit(json!({"kind":"owner_init","case":cases,"route":"indexed","ns":started.elapsed().as_nanos()}));
                let started=Instant::now();let dense_owner=IcicleBackend::new()?;
                emit(json!({"kind":"owner_init","case":cases,"route":"gather","ns":started.elapsed().as_nanos()}));
                let gather=Gather(&dense_owner);let cpu=ArkworksBackend;
                let routes:[&dyn OperatorBackend;3]=[&indexed,&gather,&cpu];
                for trial in 0..10 {
                    let phase=if trial==0 {"first_use"} else if trial<3 {"warmup"} else {"sample"};
                    for offset in 0..3 {
                        let r=(trial+offset)%3;
                        let before=if r==0 {Some(indexed.batch_stats()?)} else if r==1 {Some(dense_owner.batch_stats()?)} else {None};
                        let started=Instant::now();let actual=routes[r].msm_sparse_batch(&requests)?;
                        let ns=started.elapsed().as_nanos();ensure!(actual==expected,"MSM mismatch case={cases} route={r}");
                        let after=if r==0 {Some(indexed.batch_stats()?)} else if r==1 {Some(dense_owner.batch_stats()?)} else {None};
                        let calls=before.zip(after).map_or(0,|(b,a)|a.icicle_calls-b.icicle_calls);
                        if r!=2 {ensure!(calls==1,"unexpected native batch dispatch");}
                        emit(json!({"kind":"msm","case":cases,"n":n,"d":d,"batch":batch,
                            "shared":shared,"pattern":pattern,"route":ROUTES[r],"phase":phase,
                            "sample":trial.saturating_sub(3),"ns":ns,"correct":true,
                            "calls":calls,"buffer_grows":after.map_or(0,|s|s.buffer_grows),
                            "input_capacity":after.map_or(0,|s|s.input_capacity),
                            "work":format!("{n}/{d}/{batch}/{shared}/{pattern}")}));
                    }
                }
                cases+=1;
            }}
        }}
    }
    ensure!(cases==48,"wrong MSM matrix");println!("MSM_OK cases={cases} samples=7");Ok(())
}

fn delta()->Result<()> {
    let mut cases=0;
    for n in [64usize,256,1024] {
        let circuit=ToyCircuit::new(n)?;let(instance,witness)=circuit.valid_instance();
        let started=Instant::now();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(2026102208+n as u64))?;
        emit(json!({"kind":"setup","n":n,"ns":started.elapsed().as_nanos()}));
        let vk=params.verifying_key();let steps=(n as f64).sqrt().ceil() as usize/2;
        for pattern in ["hotspot","spreading"] {
            for trial in 0..4 {
                let phase=if trial==0 {"warmup"} else {"sample"};
                let indexed=IcicleBackend::new()?;let dense_owner=IcicleBackend::new()?;
                let gather=Gather(&dense_owner);let cpu=ArkworksBackend;
                let observed=[Observed{inner:&indexed,sparse:Cell::new((0,0))},
                    Observed{inner:&gather,sparse:Cell::new((0,0))},Observed{inner:&cpu,sparse:Cell::new((0,0))}];
                let mut sessions=Vec::new();let mut prove_ns=[0u128;3];let mut update_ns=[0u128;3];
                let mut verify_ns=[0u128;3];let mut rebuilds=[0usize;3];
                let mut aggregate_work=[String::new(),String::new(),String::new()];
                for r in 0..3 {
                    let started=Instant::now();
                    let session=params.start_update_session(&instance,&witness)?;
                    prove_ns[r]=started.elapsed().as_nanos();
                    let started=Instant::now();
                    ensure!(vk.verify_reference(&instance,session.proof())?,"initial public verification failed");
                    verify_ns[r]+=started.elapsed().as_nanos();sessions.push(session);
                }
                ensure!(sessions[0].proof()==sessions[1].proof()
                    && sessions[0].proof()==sessions[2].proof(),"initial proof mismatch");
                ensure!(sessions[0].current_witness()==sessions[1].current_witness()
                    && sessions[0].current_witness()==sessions[2].current_witness(),"initial witness mismatch");
                for step in 0..steps {
                    let gate=if pattern=="hotspot" {1} else {step+1};
                    let current=sessions[2].current_witness();let next=current.wires[4*n+gate]+Fr::from(1u64);
                    let u=Update{indices:vec![4*n+gate,5*n+gate],values:vec![next,current.wires[3*n+gate]*next]};
                    let public=Update{indices:Vec::new(),values:Vec::new()};
                    let mut work_rows=Vec::new();
                    for offset in 0..3 {
                        let r=(trial+step+offset)%3;
                        let old_generation=sessions[r].proof().generation;
                        let old_anchor=sessions[r].proof().anchor_generation;
                        observed[r].sparse.set((0,0));
                        let started=Instant::now();
                        let (result,work)=workcount::measure(||sessions[r].update_with_backend(
                            &u,&public,&observed[r]));
                        let ns=started.elapsed().as_nanos();let proof=result?;update_ns[r]+=ns;
                        ensure!(proof.generation==old_generation+1,"generation did not advance");
                        let rebuilt=proof.anchor_generation!=old_anchor;
                        rebuilds[r]+=usize::from(rebuilt);
                        let sparse=observed[r].sparse.get();
                        let expected_d=if rebuilt {0} else if pattern=="hotspot" {2} else {2*(step+1)};
                        let distance=sessions[r].anchor_distance();
                        ensure!(distance==expected_d,"session anchor distance differs");
                        if rebuilt {
                            ensure!(sparse==(0,0),"rebuild unexpectedly used sparse MSM");
                        } else {
                            ensure!(sparse.0>0 && sparse.1>0,"sparse session update did not reach indexed MSM");
                        }
                        aggregate_work[r].push_str(&format!("{work:?}/{sparse:?}/{distance}/{rebuilt};"));
                        work_rows.push((work,sparse,distance));
                        let started=Instant::now();ensure!(vk.verify_reference(&instance,&proof)?,"updated proof rejected");
                        verify_ns[r]+=started.elapsed().as_nanos();
                        emit(json!({"kind":"session_delta_update","case":cases,"n":n,"pattern":pattern,
                            "route":ROUTES[r],"phase":phase,"sample":trial.saturating_sub(1),"step":step,
                            "k":2,"d":expected_d,"rebuilt":rebuilt,"ns":ns,"correct":true,
                            "sparse_calls":sparse.0,"sparse_points":sparse.1,
                            "anchor_distance":distance,
                            "generation":proof.generation,"anchor_generation":proof.anchor_generation,
                            "work":format!("{work:?}/{sparse:?}/{distance}/{rebuilt}")}));
                    }
                    ensure!(work_rows[0]==work_rows[1] && work_rows[0]==work_rows[2],"unequal update work");
                    ensure!(sessions[0].proof()==sessions[1].proof()
                        && sessions[0].proof()==sessions[2].proof(),"proof differs after update");
                    ensure!(sessions[0].current_witness()==sessions[1].current_witness()
                        && sessions[0].current_witness()==sessions[2].current_witness(),"witness differs");
                }
                ensure!(rebuilds==[usize::from(pattern=="spreading");3],"rebuild endpoint differs");
                if pattern=="spreading" {ensure!(sessions.iter().all(|s|s.proof().delta.is_none()),"unfinished delta at aligned endpoint");}
                ensure!(aggregate_work[0]==aggregate_work[1] && aggregate_work[0]==aggregate_work[2],"sequence work differs");
                for r in 0..3 {
                    emit(json!({"kind":"session_delta","case":cases,"n":n,"pattern":pattern,
                        "route":ROUTES[r],"phase":phase,"sample":trial.saturating_sub(1),
                        "ns":update_ns[r],"prove_ns":prove_ns[r],"verify_ns":verify_ns[r],
                        "completed_component_ns":prove_ns[r]+update_ns[r]+verify_ns[r],
                        "steps":steps,"rebuilds":rebuilds[r],"correct":true,"work":aggregate_work[r]}));
                }
            }
            cases+=1;
        }
    }
    ensure!(cases==6,"wrong session delta matrix");println!("SESSION_DELTA_OK cases={cases} samples=3");Ok(())
}

fn scale()->Result<()> {
    let mut cases=0;
    for n in [2048usize,4096] {
        let circuit=ToyCircuit::new(n)?;let(instance,witness)=circuit.valid_instance();
        let started=Instant::now();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(2026102212+n as u64))?;
        emit(json!({"kind":"setup","n":n,"ns":started.elapsed().as_nanos()}));
        let vk=params.verifying_key();let steps=(n as f64).sqrt().ceil() as usize/2;
        let pattern="spreading";
        for trial in 0..3 {
            let phase=if trial==0 {"warmup"} else {"sample"};
            let indexed=IcicleBackend::new()?;let dense_owner=IcicleBackend::new()?;
            let gather=Gather(&dense_owner);let cpu=ArkworksBackend;
            let observed=[Observed{inner:&indexed,sparse:Cell::new((0,0))},
                Observed{inner:&gather,sparse:Cell::new((0,0))},Observed{inner:&cpu,sparse:Cell::new((0,0))}];
            let mut sessions=Vec::new();let mut session_prove_ns=[0u128;3];
            let mut session_update_ns=[0u128;3];let mut session_verify_ns=[0u128;3];
            let mut session_rebuilds=[0usize;3];
            let mut session_work=[String::new(),String::new(),String::new()];
            for r in 0..3 {
                let started=Instant::now();
                let session=params.start_update_session(&instance,&witness)?;
                session_prove_ns[r]=started.elapsed().as_nanos();
                let started=Instant::now();
                ensure!(vk.verify_reference(&instance,session.proof())?,"initial session verification failed");
                session_verify_ns[r]+=started.elapsed().as_nanos();sessions.push(session);
            }
            let started=Instant::now();
            let mut generic_state=params.prove_with_aux_traced_with_backend(
                &instance,&witness,AnchorOpeningPolicy::Precompute,&observed[0],&mut CpuTrace::new())?;
            let generic_prove_ns=started.elapsed().as_nanos();
            let started=Instant::now();
            ensure!(vk.verify_reference(&instance,&generic_state.0)?,"initial generic verification failed");
            let mut generic_verify_ns=started.elapsed().as_nanos();
            ensure!(sessions[0].proof()==sessions[1].proof()
                && sessions[0].proof()==sessions[2].proof()
                && sessions[0].proof()==&generic_state.0,"initial proof mismatch");
            ensure!(sessions[0].current_witness()==sessions[1].current_witness()
                && sessions[0].current_witness()==sessions[2].current_witness()
                && sessions[0].current_witness()==generic_state.1.current_witness(),"initial witness mismatch");
            let mut generic_update_ns=0u128;let mut generic_rebuilds=0usize;
            let mut generic_work=String::new();
            for step in 0..steps {
                let gate=step+1;
                let current=sessions[0].current_witness();
                let next=current.wires[4*n+gate]+Fr::from(1u64);
                let u=Update{indices:vec![4*n+gate,5*n+gate],
                    values:vec![next,current.wires[3*n+gate]*next]};
                let public=Update{indices:Vec::new(),values:Vec::new()};
                let mut session_rows=Vec::new();
                for slot in 0..4 {
                    let action=(trial+step+slot)%4;
                    if action<3 {
                        let r=action;
                        let old_generation=sessions[r].proof().generation;
                        let old_anchor=sessions[r].proof().anchor_generation;
                        observed[r].sparse.set((0,0));
                        let started=Instant::now();
                        let (result,work)=workcount::measure(||sessions[r].update_with_backend(
                            &u,&public,&observed[r]));
                        let ns=started.elapsed().as_nanos();let proof=result?;
                        session_update_ns[r]+=ns;
                        ensure!(proof.generation==old_generation+1,"session generation did not advance");
                        let rebuilt=proof.anchor_generation!=old_anchor;
                        session_rebuilds[r]+=usize::from(rebuilt);
                        let sparse=observed[r].sparse.get();
                        let expected_d=if rebuilt {0} else {2*(step+1)};
                        let distance=sessions[r].anchor_distance();
                        ensure!(distance==expected_d,"session anchor distance differs");
                        if rebuilt {
                            ensure!(sparse==(0,0),"session rebuild unexpectedly used sparse MSM");
                        } else {
                            ensure!(sparse.0>0 && sparse.1>0,"session sparse route was not observed");
                        }
                        session_work[r].push_str(&format!("{work:?}/{sparse:?}/{distance}/{rebuilt};"));
                        session_rows.push((work,sparse,distance));
                        let started=Instant::now();
                        ensure!(vk.verify_reference(&instance,&proof)?,"session proof rejected");
                        session_verify_ns[r]+=started.elapsed().as_nanos();
                        emit(json!({"kind":"scale_session_update","case":cases,"n":n,
                            "pattern":pattern,"route":ROUTES[r],"phase":phase,
                            "sample":trial.saturating_sub(1),"step":step,"k":2,
                            "d":expected_d,"rebuilt":rebuilt,"ns":ns,"correct":true,
                            "sparse_calls":sparse.0,"sparse_points":sparse.1,
                            "anchor_distance":distance,"generation":proof.generation,
                            "anchor_generation":proof.anchor_generation,
                            "work":format!("{work:?}/{sparse:?}/{distance}/{rebuilt}")}));
                    } else {
                        let old_generation=generic_state.0.generation;
                        let old_anchor=generic_state.0.anchor_generation;
                        let old_witness=generic_state.1.current_witness().clone();
                        let old_proof=generic_state.0.clone();
                        observed[0].sparse.set((0,0));
                        let mut trace=CpuTrace::new();let started=Instant::now();
                        let (result,work)=workcount::measure(||params.update_proof_traced_with_backend(
                            &instance,&instance,&old_witness,&u,&old_proof,&mut generic_state.1,
                            VerificationMode::Product,BatchOpenMode::Prepared,
                            &observed[0],&mut trace));
                        let ns=started.elapsed().as_nanos();let proof=result?;
                        generic_update_ns+=ns;
                        ensure!(proof.generation==old_generation+1,"generic generation did not advance");
                        let rebuilt=proof.anchor_generation!=old_anchor;
                        generic_rebuilds+=usize::from(rebuilt);
                        let sparse=observed[0].sparse.get();
                        ensure!(sparse==(0,0),"generic update unexpectedly used sparse MSM");
                        let expected_d=if rebuilt {0} else {2*(step+1)};
                        generic_work.push_str(&format!("{work:?}/{sparse:?}/{expected_d}/{rebuilt};"));
                        generic_state.0=proof;
                        let started=Instant::now();
                        ensure!(vk.verify_reference(&instance,&generic_state.0)?,"generic proof rejected");
                        generic_verify_ns+=started.elapsed().as_nanos();
                        emit(json!({"kind":"scale_generic_update","case":cases,"n":n,
                            "pattern":pattern,"route":"generic_indexed","phase":phase,
                            "sample":trial.saturating_sub(1),"step":step,"k":2,
                            "d":expected_d,"rebuilt":rebuilt,"ns":ns,"correct":true,
                            "sparse_calls":sparse.0,"sparse_points":sparse.1,
                            "generation":generic_state.0.generation,
                            "anchor_generation":generic_state.0.anchor_generation,
                            "work":format!("{work:?}/{sparse:?}/{expected_d}/{rebuilt}")}));
                    }
                }
                ensure!(session_rows[0]==session_rows[1] && session_rows[0]==session_rows[2],
                    "unequal session update work");
                ensure!(sessions[0].proof()==sessions[1].proof()
                    && sessions[0].proof()==sessions[2].proof()
                    && sessions[0].proof()==&generic_state.0,"proof differs after update");
                ensure!(sessions[0].current_witness()==sessions[1].current_witness()
                    && sessions[0].current_witness()==sessions[2].current_witness()
                    && sessions[0].current_witness()==generic_state.1.current_witness(),
                    "witness differs after update");
            }
            ensure!(session_rebuilds==[1;3] && generic_rebuilds==1,"rebuild endpoint differs");
            ensure!(sessions.iter().all(|s|s.proof().delta.is_none())
                && generic_state.0.delta.is_none(),"unfinished delta at aligned endpoint");
            ensure!(session_work[0]==session_work[1] && session_work[0]==session_work[2],
                "session sequence work differs");
            for r in 0..3 {
                emit(json!({"kind":"scale_session","case":cases,"n":n,"pattern":pattern,
                    "route":ROUTES[r],"phase":phase,"sample":trial.saturating_sub(1),
                    "ns":session_update_ns[r],"prove_ns":session_prove_ns[r],
                    "verify_ns":session_verify_ns[r],
                    "completed_component_ns":session_prove_ns[r]+session_update_ns[r]+session_verify_ns[r],
                    "steps":steps,"rebuilds":session_rebuilds[r],"correct":true,
                    "work":session_work[r]}));
            }
            emit(json!({"kind":"scale_generic","case":cases,"n":n,"pattern":pattern,
                "route":"generic_indexed","phase":phase,"sample":trial.saturating_sub(1),
                "ns":generic_update_ns,"prove_ns":generic_prove_ns,
                "verify_ns":generic_verify_ns,
                "completed_component_ns":generic_prove_ns+generic_update_ns+generic_verify_ns,
                "steps":steps,"rebuilds":generic_rebuilds,"correct":true,
                "work":generic_work}));
        }
        cases+=1;
    }
    ensure!(cases==2,"wrong scale matrix");
    println!("SESSION_SCALE_OK cases={cases} samples=2");Ok(())
}

fn pow16()->Result<()> {
    let mut cases=0;
    for n in [1024usize,65536] {
        let circuit=ToyCircuit::new(n)?;let(instance,witness)=circuit.valid_instance();
        let started=Instant::now();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(2026102216+n as u64))?;
        emit(json!({"kind":"pow16_setup","case":cases,"n":n,"ns":started.elapsed().as_nanos()}));
        let vk=params.verifying_key();
        let indexed=IcicleBackend::new()?;
        let observed=Observed{inner:&indexed,sparse:Cell::new((0,0))};
        let started=Instant::now();
        let mut session=params.start_update_session(&instance,&witness)?;
        let session_prove_ns=started.elapsed().as_nanos();
        let started=Instant::now();
        ensure!(vk.verify_reference(&instance,session.proof())?,"initial session proof rejected");
        let initial_session_verify_ns=started.elapsed().as_nanos();
        let started=Instant::now();
        let mut generic=params.prove_with_aux_traced_with_backend(
            &instance,&witness,AnchorOpeningPolicy::Precompute,&observed,&mut CpuTrace::new())?;
        let generic_prove_ns=started.elapsed().as_nanos();
        let started=Instant::now();
        ensure!(vk.verify_reference(&instance,&generic.0)?,"initial generic proof rejected");
        let initial_generic_verify_ns=started.elapsed().as_nanos();
        ensure!(session.proof()==&generic.0,"initial proof mismatch");
        ensure!(session.current_witness()==generic.1.current_witness(),"initial witness mismatch");

        let mut session_measured_ns=0u128;let mut generic_measured_ns=0u128;
        let mut session_verify_ns=0u128;let mut generic_verify_ns=0u128;
        let mut ratios=Vec::new();
        for step in 0..10usize {
            let gate=step+1;
            let current=session.current_witness();
            let next=current.wires[4*n+gate]+Fr::from(1u64);
            let u=Update{indices:vec![4*n+gate,5*n+gate],
                values:vec![next,current.wires[3*n+gate]*next]};
            let public=Update{indices:Vec::new(),values:Vec::new()};
            let mut run_session=||->Result<(u128,(u64,u64))>{
                observed.sparse.set((0,0));
                let started=Instant::now();
                let (result,_)=workcount::measure(||session.update_with_backend(&u,&public,&observed));
                let ns=started.elapsed().as_nanos();let proof=result?;
                let sparse=observed.sparse.get();
                ensure!(sparse.0>0 && sparse.1>0,"session sparse route not observed");
                ensure!(proof.generation==(step+1) as u64,"session generation differs");
                ensure!(session.anchor_distance()==2*(step+1),"session distance differs");
                Ok((ns,sparse))
            };
            let mut run_generic=||->Result<(u128,(u64,u64))>{
                let old_witness=generic.1.current_witness().clone();
                let old_proof=generic.0.clone();
                observed.sparse.set((0,0));
                let mut trace=CpuTrace::new();let started=Instant::now();
                let (result,_)=workcount::measure(||params.update_proof_traced_with_backend(
                    &instance,&instance,&old_witness,&u,&old_proof,&mut generic.1,
                    VerificationMode::Product,BatchOpenMode::Prepared,&observed,&mut trace));
                let ns=started.elapsed().as_nanos();let proof=result?;
                ensure!(observed.sparse.get()==(0,0),"generic route unexpectedly sparse");
                ensure!(proof.generation==(step+1) as u64,"generic generation differs");
                generic.0=proof;
                Ok((ns,observed.sparse.get()))
            };
            let ((session_ns,session_sparse),(generic_ns,generic_sparse))=if step%2==0 {
                let generic=run_generic()?;let session=run_session()?;
                (session,generic)
            } else {
                let session=run_session()?;let generic=run_generic()?;
                (session,generic)
            };
            ensure!(session.proof()==&generic.0,"proof differs after fixed update");
            ensure!(session.current_witness()==generic.1.current_witness(),"witness differs after fixed update");
            let started=Instant::now();
            ensure!(vk.verify_reference(&instance,session.proof())?,"updated session proof rejected");
            let session_check=started.elapsed().as_nanos();
            let started=Instant::now();
            ensure!(vk.verify_reference(&instance,&generic.0)?,"updated generic proof rejected");
            let generic_check=started.elapsed().as_nanos();
            let measured=step>=2;
            if measured {
                session_measured_ns+=session_ns;generic_measured_ns+=generic_ns;
                session_verify_ns+=session_check;generic_verify_ns+=generic_check;
                ratios.push(generic_ns as f64/session_ns as f64);
            }
            emit(json!({"kind":"pow16_step","case":cases,"n":n,"step":step,
                "measured":measured,"k":2,"d":2*(step+1),
                "session_ns":session_ns,"generic_ns":generic_ns,
                "generic_over_session":generic_ns as f64/session_ns as f64,
                "session_sparse_calls":session_sparse.0,
                "session_sparse_points":session_sparse.1,
                "generic_sparse_calls":generic_sparse.0,
                "generic_sparse_points":generic_sparse.1,
                "generation":step+1,"correct":true}));
        }
        ratios.sort_by(|a,b|a.partial_cmp(b).unwrap());
        let ratio_median=(ratios[3]+ratios[4])/2.0;
        emit(json!({"kind":"pow16_summary","case":cases,"n":n,
            "measured_steps":8,"d_end":20,"step_k":2,
            "session_update_ns":session_measured_ns,
            "generic_update_ns":generic_measured_ns,
            "generic_over_session":ratio_median,
            "ratio_min":ratios[0],"ratio_max":ratios[7],
            "session_prove_ns":session_prove_ns,
            "generic_prove_ns":generic_prove_ns,
            "session_initial_verify_ns":initial_session_verify_ns,
            "generic_initial_verify_ns":initial_generic_verify_ns,
            "session_verify_ns":session_verify_ns,
            "generic_verify_ns":generic_verify_ns,
            "session_completed_component_ns":session_prove_ns+session_measured_ns+session_verify_ns,
            "generic_completed_component_ns":generic_prove_ns+generic_measured_ns+generic_verify_ns,
            "correct":true}));
        cases+=1;
    }
    ensure!(cases==2,"wrong pow16 matrix");
    println!("POW16_OK cases={cases} measured_steps=8");Ok(())
}

fn fullopt_backend(n:usize,optimized:bool,affine:bool)->Result<IcicleBackend> {
    let policy=strategy::Strategy::for_size(n).map_err(anyhow::Error::msg)?;
    let mut backend=IcicleBackend::with_resident_limits(policy.limit_bytes,policy.undo_bytes)?;
    backend.set_bounded_g1_decode(optimized);
    backend.set_device_affine_readback(optimized && affine);
    backend.set_setup_mode(SetupMode::TableBatchNormalize)?;
    backend.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63)?;
    Ok(backend)
}

fn fullopt_edits(w:&Witness,n:usize,roots:usize,addition:bool,seed:u64)->Update {
    let mut rng=StdRng::seed_from_u64(seed);
    let mut update=Update{indices:Vec::new(),values:Vec::new()};
    for i in 1..=roots {
        let value=Fr::rand(&mut rng);
        if addition {
            let sum=w.wires[i]+value;
            update.indices.extend([n+i,2*n+i,3*n+i,5*n+i]);
            update.values.extend([value,sum,sum,sum*w.wires[4*n+i]]);
        } else {
            update.indices.extend([4*n+i,5*n+i]);
            update.values.extend([value,w.wires[3*n+i]*value]);
        }
    }
    update
}

fn fullopt_cases(sizes:&[usize])->Result<()> {
    let mut cases=0;
    for &n in sizes {
        let policy=strategy::Strategy::for_size(n).map_err(anyhow::Error::msg)?;
        let circuit=ToyCircuit::new(n)?;let(instance,witness)=circuit.valid_instance();
        let setup_backend=fullopt_backend(n,false,false)?;
        let mut setup_trace=CpuTrace::new();
        let setup_started=Instant::now();
        let params=DynaverseParams::setup_with_backend_traced(
            circuit,&mut StdRng::seed_from_u64(2026102220+n as u64),
            &setup_backend,&mut setup_trace)?;
        let setup_ns=setup_started.elapsed().as_nanos();
        drop(setup_backend);
        emit(json!({"kind":"fullopt_setup","case":cases,"n":n,"ns":setup_ns,
            "trace_recorded_ns":setup_trace.recorded_total()}));
        let vk=params.verifying_key();
        let affine=readback::calibrate(n,policy)?;
        let base_cfg=IncrementalConfig {window:Some(policy.window),trace_updates:true,
            memory_limit_bytes:2u64<<30,sparse:SparseDispatchPolicy::ForceGpu,
            ..IncrementalConfig::default()};
        let full_cfg=IncrementalConfig {window:Some(policy.window),trace_updates:true,
            memory_limit_bytes:2u64<<30,sparse:SparseDispatchPolicy::ForceGpu,
            linear_rebuild_reuse:true,unchanged_cache_reuse:true,
            foreground_linear_reuse:true,freeze_linear_capture_limit:Some(128),
            frozen_g2_reuse_limit:Some(128),support_geometry_reuse:true,
            transform_row_batch:16,transform_packet_max_len:policy.packet_cap,
            rebuild_cache_mode:VerificationMode::Product,
            rebuild_prepared_cache_roots:Some(policy.cache_roots),
            rebuild_cache_readback_roots:Some(policy.cache_roots),..IncrementalConfig::default()};
        let started=Instant::now();
        let baseline=params.start_incremental_session(
            &instance,&witness,base_cfg,fullopt_backend(n,false,false)?)?;
        let baseline_initial_ns=started.elapsed().as_nanos();
        let started=Instant::now();
        let optimized=params.start_incremental_session(
            &instance,&witness,full_cfg,fullopt_backend(n,true,affine)?)?;
        let optimized_initial_ns=started.elapsed().as_nanos();
        let mut sessions=vec![baseline,optimized];
        ensure!(sessions[0].proof()==sessions[1].proof(),"fullopt initial proof mismatch");
        ensure!(sessions[0].current_witness()==sessions[1].current_witness(),
            "fullopt initial witness mismatch");
        let baseline_prep=sessions[0].cache_preparation_stats();
        let optimized_prep=sessions[1].cache_preparation_stats();
        let baseline_plan=sessions[0].resource_plan();
        let optimized_plan=sessions[1].resource_plan();
        let mut update_ns=[0u128;2];
        let mut feature_counts={
            let mut counts=std::collections::BTreeMap::new();
            for key in ["linear_reused","frozen_linear_captured",
                "foreground_linear_reused","global_g2_reused",
                "support_geometry_reused","cache_reused","transform_packet_calls",
                "cache_reused_points","linear_skipped_g1_points",
                "frozen_linear_skipped_g1_points","global_g2_inputs",
                "transform_packet_elements","sparse_gpu_g1_points",
                "sparse_cpu_g1_points"] {
                counts.insert(key,0u64);
            }
            counts
        };
        for step in 0..4usize {
            let addition=step%2==1;let roots=if addition {4} else {8};
            let update=fullopt_edits(
                sessions[0].current_witness(),n,roots,addition,22100+n as u64*100+step as u64);
            for offset in 0..2 {
                let route=(step+offset)%2;
                let session=&mut sessions[route];
                let before=session.backend().stats();
                let sparse_before=session.sparse_dispatch_stats();
                let started=Instant::now();
                let proof=session.update(&update)?;
                let ns=started.elapsed().as_nanos();
                let receipt=*session.last_receipt();
                let after=session.backend().stats();
                let sparse=session.sparse_dispatch_stats();
                ensure!(receipt.rebuild_units<=receipt.allowance,"update exceeds work allowance");
                let verify_started=Instant::now();
                ensure!(vk.verify_reference(&instance,&proof)?,"fullopt proof rejected");
                let verify_ns=verify_started.elapsed().as_nanos();
                let route_name=if route==0 {"baseline"} else {"full_optimized"};
                let measured=step>=2;
                if measured {update_ns[route]+=ns;}
                if route==1 {
                    *feature_counts.get_mut("linear_reused").unwrap()
                        +=u64::from(receipt.linear_reused);
                    *feature_counts.get_mut("frozen_linear_captured").unwrap()
                        +=u64::from(receipt.frozen_linear_captured);
                    *feature_counts.get_mut("foreground_linear_reused").unwrap()
                        +=u64::from(receipt.foreground_linear_reused);
                    *feature_counts.get_mut("global_g2_reused").unwrap()
                        +=u64::from(receipt.global_g2_reused);
                    *feature_counts.get_mut("support_geometry_reused").unwrap()
                        +=u64::from(receipt.support_geometry_reused);
                    *feature_counts.get_mut("cache_reused").unwrap()
                        +=u64::from(receipt.cache_reused_points>0);
                    *feature_counts.get_mut("transform_packet_calls").unwrap()
                        +=receipt.transform_packet_calls;
                    *feature_counts.get_mut("cache_reused_points").unwrap()
                        +=receipt.cache_reused_points as u64;
                    *feature_counts.get_mut("linear_skipped_g1_points").unwrap()
                        +=receipt.linear_skipped_g1_points as u64;
                    *feature_counts.get_mut("frozen_linear_skipped_g1_points").unwrap()
                        +=receipt.frozen_linear_skipped_g1_points as u64;
                    *feature_counts.get_mut("global_g2_inputs").unwrap()
                        +=receipt.global_g2_inputs as u64;
                    *feature_counts.get_mut("transform_packet_elements").unwrap()
                        +=receipt.transform_packet_elements;
                    *feature_counts.get_mut("sparse_gpu_g1_points").unwrap()
                        +=u64::try_from(sparse.gpu.g1_points-sparse_before.gpu.g1_points).unwrap_or(0);
                    *feature_counts.get_mut("sparse_cpu_g1_points").unwrap()
                        +=u64::try_from(sparse.cpu.g1_points-sparse_before.cpu.g1_points).unwrap_or(0);
                }
                emit(json!({"kind":"fullopt_update","case":cases,"n":n,
                    "step":step,"measured":measured,"route":route_name,
                    "family":if addition {"addition"} else {"multiplier"},
                    "roots":roots,"k":receipt.changed,"ns":ns,
                    "verify_ns":verify_ns,"correct":true,
                    "generation":proof.generation,"anchor_generation":proof.anchor_generation,
                    "large_batch":receipt.large_batch,"swapped":receipt.swapped,
                    "restarted":receipt.restarted,"pending":session.rebuild_progress().is_some(),
                    "linear_reused":receipt.linear_reused,
                    "frozen_linear_captured":receipt.frozen_linear_captured,
                    "foreground_linear_reused":receipt.foreground_linear_reused,
                    "global_g2_reused":receipt.global_g2_reused,
                    "support_geometry_reused":receipt.support_geometry_reused,
                    "cache_reused_points":receipt.cache_reused_points,
                    "transform_packet_calls":receipt.transform_packet_calls,
                    "transform_packet_elements":receipt.transform_packet_elements,
                    "sparse_cpu_g1_points":sparse.cpu.g1_points-sparse_before.cpu.g1_points,
                    "sparse_gpu_g1_points":sparse.gpu.g1_points-sparse_before.gpu.g1_points,
                    "resident_bytes":after.bytes_allocated,
                    "resident_calls":after.calls-before.calls,
                    "rebuild_units":receipt.rebuild_units.to_string(),
                    "allowance":receipt.allowance.to_string()}));
            }
            ensure!(sessions[0].proof()==sessions[1].proof(),"fullopt proof differs");
            ensure!(sessions[0].current_witness()==sessions[1].current_witness(),
                "fullopt witness differs");
        }
        // The timed large-batch updates intentionally exceed the rebuild horizon,
        // so they cannot exercise the incremental linear reuse path. Repeat one
        // small hotspot edit after timing to force the same support geometry and
        // exercise the scheduled rebuild options before declaring full coverage.
        let mut feature_audit_steps=0usize;
        if n==256 {
            for step in 0..12usize {
                let i=1usize;
                let current=sessions[0].current_witness();
                let next=current.wires[4*n+i]+Fr::from(1u64);
                let update=Update {indices:vec![4*n+i,5*n+i],
                    values:vec![next,current.wires[3*n+i]*next]};
                for offset in 0..2 {
                    let route=(step+offset)%2;
                    let session=&mut sessions[route];
                    let before=session.backend().stats();
                    let sparse_before=session.sparse_dispatch_stats();
                    let started=Instant::now();
                    let proof=session.update(&update)?;
                    let ns=started.elapsed().as_nanos();
                    let receipt=*session.last_receipt();
                    let after=session.backend().stats();
                    let sparse=session.sparse_dispatch_stats();
                    ensure!(receipt.rebuild_units<=receipt.allowance,
                        "feature-audit update exceeds work allowance");
                    ensure!(vk.verify_reference(&instance,&proof)?,
                        "feature-audit proof rejected");
                    if route==1 {
                        *feature_counts.get_mut("linear_reused").unwrap()
                            +=u64::from(receipt.linear_reused);
                        *feature_counts.get_mut("frozen_linear_captured").unwrap()
                            +=u64::from(receipt.frozen_linear_captured);
                        *feature_counts.get_mut("foreground_linear_reused").unwrap()
                            +=u64::from(receipt.foreground_linear_reused);
                        *feature_counts.get_mut("global_g2_reused").unwrap()
                            +=u64::from(receipt.global_g2_reused);
                        *feature_counts.get_mut("support_geometry_reused").unwrap()
                            +=u64::from(receipt.support_geometry_reused);
                        *feature_counts.get_mut("cache_reused").unwrap()
                            +=u64::from(receipt.cache_reused_points>0);
                        *feature_counts.get_mut("transform_packet_calls").unwrap()
                            +=receipt.transform_packet_calls;
                        *feature_counts.get_mut("cache_reused_points").unwrap()
                            +=receipt.cache_reused_points as u64;
                        *feature_counts.get_mut("linear_skipped_g1_points").unwrap()
                            +=receipt.linear_skipped_g1_points as u64;
                        *feature_counts.get_mut("frozen_linear_skipped_g1_points").unwrap()
                            +=receipt.frozen_linear_skipped_g1_points as u64;
                        *feature_counts.get_mut("global_g2_inputs").unwrap()
                            +=receipt.global_g2_inputs as u64;
                        *feature_counts.get_mut("transform_packet_elements").unwrap()
                            +=receipt.transform_packet_elements;
                        *feature_counts.get_mut("sparse_gpu_g1_points").unwrap()
                            +=u64::try_from(sparse.gpu.g1_points-sparse_before.gpu.g1_points).unwrap_or(0);
                        *feature_counts.get_mut("sparse_cpu_g1_points").unwrap()
                            +=u64::try_from(sparse.cpu.g1_points-sparse_before.cpu.g1_points).unwrap_or(0);
                    }
                    emit(json!({"kind":"fullopt_feature_probe","case":cases,"n":n,
                        "step":step,"route":if route==0 {"baseline"} else {"full_optimized"},
                        "family":"hotspot_multiplier","roots":1,"k":receipt.changed,"ns":ns,
                        "correct":true,"generation":proof.generation,
                        "anchor_generation":proof.anchor_generation,
                        "large_batch":receipt.large_batch,"swapped":receipt.swapped,
                        "restarted":receipt.restarted,"pending":session.rebuild_progress().is_some(),
                        "linear_reused":receipt.linear_reused,
                        "frozen_linear_captured":receipt.frozen_linear_captured,
                        "foreground_linear_reused":receipt.foreground_linear_reused,
                        "global_g2_reused":receipt.global_g2_reused,
                        "support_geometry_reused":receipt.support_geometry_reused,
                        "cache_reused_points":receipt.cache_reused_points,
                        "cache_reused_wire_mask":receipt.cache_reused_wire_mask,
                        "linear_skipped_g1_points":receipt.linear_skipped_g1_points,
                        "frozen_linear_skipped_g1_points":receipt.frozen_linear_skipped_g1_points,
                        "transform_packet_calls":receipt.transform_packet_calls,
                        "transform_packet_elements":receipt.transform_packet_elements,
                        "sparse_cpu_g1_points":sparse.cpu.g1_points-sparse_before.cpu.g1_points,
                        "sparse_gpu_g1_points":sparse.gpu.g1_points-sparse_before.gpu.g1_points,
                        "resident_bytes":after.bytes_allocated,
                        "resident_calls":after.calls-before.calls,
                        "rebuild_units":receipt.rebuild_units.to_string(),
                        "allowance":receipt.allowance.to_string()}));
                }
                ensure!(sessions[0].proof()==sessions[1].proof(),
                    "feature-audit proof differs");
                ensure!(sessions[0].current_witness()==sessions[1].current_witness(),
                    "feature-audit witness differs");
                feature_audit_steps+=1;
            }
            for key in ["frozen_linear_captured","foreground_linear_reused",
                "global_g2_reused","cache_reused","transform_packet_calls",
                "sparse_gpu_g1_points"] {
                ensure!(feature_counts[key]>0,
                    format!("full optimization did not execute: {key}"));
            }
            ensure!(feature_counts["sparse_cpu_g1_points"]==0,
                "ForceGpu feature audit dispatched sparse work on CPU");
            let mut geometry=params.start_incremental_session(&instance,&witness,
                IncrementalConfig {window:Some(6),foreground_linear_reuse:true,
                    support_geometry_reuse:true,trace_updates:true,
                    memory_limit_bytes:2u64<<30,sparse:SparseDispatchPolicy::ForceGpu,
                    ..IncrementalConfig::default()},fullopt_backend(n,true,affine)?)?;
            let initial=fullopt_edits(&witness,n,3,false,23000+n as u64);
            geometry.update(&initial)?;
            let mut geometry_hits=0usize;
            for step in 0..6 {
                let update=fullopt_edits(geometry.current_witness(),n,1,false,
                    24000+n as u64+step as u64);
                let proof=geometry.update(&update)?;
                geometry_hits+=usize::from(geometry.last_receipt().support_geometry_reused);
                ensure!(vk.verify_reference(&instance,&proof)?,"geometry proof rejected");
            }
            ensure!(geometry_hits>0,"geometry-only route did not reuse support geometry");
            emit(json!({"kind":"fullopt_extra_audit","case":cases,"n":n,
                "audit":"geometry_only","hits":geometry_hits,
                "support_geometry_builds":geometry.last_receipt().support_geometry_builds,
                "sparse_cpu_g1_points":geometry.sparse_dispatch_stats().cpu.g1_points,
                "sparse_gpu_g1_points":geometry.sparse_dispatch_stats().gpu.g1_points,
                "correct":true}));
            let mut linear=params.start_incremental_session(&instance,&witness,
                IncrementalConfig {window:Some(4),linear_rebuild_reuse:true,
                    trace_updates:true,memory_limit_bytes:2u64<<30,
                    sparse:SparseDispatchPolicy::ForceGpu,..IncrementalConfig::default()},
                fullopt_backend(n,true,affine)?)?;
            let a=witness.wires[4*n]+Fr::from(1u64);
            let b=witness.wires[4*n+1]+Fr::from(1u64);
            let linear_proof=linear.update(&Update {indices:vec![4*n,4*n+1,5*n+1],
                values:vec![a,b,witness.wires[3*n+1]*b]})?;
            ensure!(linear.last_receipt().linear_reused,
                "linear-only route did not reuse frozen linear correction");
            ensure!(vk.verify_reference(&instance,&linear_proof)?,"linear proof rejected");
            emit(json!({"kind":"fullopt_extra_audit","case":cases,"n":n,
                "audit":"linear_only","linear_reused":linear.last_receipt().linear_reused,
                "linear_skipped_g1_points":linear.last_receipt().linear_skipped_g1_points,
                "sparse_cpu_g1_points":linear.sparse_dispatch_stats().cpu.g1_points,
                "sparse_gpu_g1_points":linear.sparse_dispatch_stats().gpu.g1_points,
                "correct":true}));
        }
        let prep_json=|value:Option<dynaverse_protocol::incremental_rebuild::CachePreparationStats>| {
            value.map(|v|json!({"roots":v.roots,"prepared_entries":v.prepared_entries,
                "retained_bytes":v.retained_bytes,"host_upper_bytes":v.host_upper_bytes,
                "precompute_ns":v.precompute_ns}))
        };
        emit(json!({"kind":"fullopt_summary","case":cases,"n":n,
            "baseline_initial_ns":baseline_initial_ns,
            "optimized_initial_ns":optimized_initial_ns,
            "baseline_update_ns":update_ns[0],"optimized_update_ns":update_ns[1],
            "measured_steps":2,"feature_counts":feature_counts,
            "feature_audit_steps":feature_audit_steps,
            "configuration":{"window":policy.window,"sparse":"ForceGpu",
                "bounded_g1_decode":true,"device_affine_readback":affine,
                "native_limit_bytes":policy.limit_bytes,"undo_bytes":policy.undo_bytes,
                "linear_rebuild_reuse":true,"unchanged_cache_reuse":true,
                "foreground_linear_reuse":true,"freeze_linear_capture_limit":128,
                "frozen_g2_reuse_limit":128,"support_geometry_reuse":true,
                "transform_row_batch":16,"transform_packet_max_len":policy.packet_cap,
                "rebuild_cache_mode":"Product","rebuild_prepared_cache_roots":policy.cache_roots,
                "rebuild_cache_readback_roots":policy.cache_roots},
            "baseline_cache_preparation":prep_json(baseline_prep),
            "optimized_cache_preparation":prep_json(optimized_prep),
            "baseline_resource":{"upper_bytes":baseline_plan.upper_bytes,
                "resident_bytes":baseline_plan.resident_bytes,
                "workspace_reserve":baseline_plan.workspace_reserve,
                "transaction_bytes":baseline_plan.transaction_bytes,
                "frozen_linear_bytes":baseline_plan.frozen_linear_bytes,
                "global_g2_host_bytes":baseline_plan.global_g2_host_bytes,
                "cache_preparation_host_bytes":baseline_plan.cache_preparation_host_bytes,
                "cache_readback_host_bytes":baseline_plan.cache_readback_host_bytes},
            "optimized_resource":{"upper_bytes":optimized_plan.upper_bytes,
                "resident_bytes":optimized_plan.resident_bytes,
                "workspace_reserve":optimized_plan.workspace_reserve,
                "transaction_bytes":optimized_plan.transaction_bytes,
                "frozen_linear_bytes":optimized_plan.frozen_linear_bytes,
                "global_g2_host_bytes":optimized_plan.global_g2_host_bytes,
                "cache_preparation_host_bytes":optimized_plan.cache_preparation_host_bytes,
                "cache_readback_host_bytes":optimized_plan.cache_readback_host_bytes},
            "correct":true}));
        cases+=1;
    }
    ensure!(cases==sizes.len(),"wrong fullopt matrix");
    println!("FULLOPT_OK cases={cases} measured_steps=2");Ok(())
}
fn main()->Result<()> {
    match std::env::args().nth(1).as_deref(){
        Some("msm")=>msm(),Some("session")=>delta(),Some("scale")=>scale(),
        Some("pow16")=>pow16(),Some("fullopt")=>fullopt_cases(&[256,65536]),
        Some("fullopt-small")=>fullopt_cases(&[256]),
        Some("fullopt-medium")=>fullopt_cases(&[1024]),
        Some("readback-check")=>{let n=1024;readback::calibrate(n,strategy::Strategy::for_size(n).unwrap())?;Ok(())},
        _=>anyhow::bail!("expected msm, session, scale, pow16 or fullopt")
    }
}
