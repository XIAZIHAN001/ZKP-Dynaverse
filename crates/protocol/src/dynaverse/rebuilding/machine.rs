//! Execution and exact candidate checks, all inside the caller's transaction.
use super::*;
use super::program::{Command, POINTS, PUBLIC_QUOTIENT, MULTIPLICATION_QUOTIENT};

impl Machine {
    /// Initialization owns unpublished buffers: failure drops them, so it needs
    /// neither a rollback journal nor a synthetic old anchor. Execute the same
    /// commands and exact cache checks used for later rebuilt anchors.
    pub fn build_initial<B:RebuildBackend>(params:&DynaverseParams,program:&Program,
        storage:&mut AnchorStorage,backend:&mut B,public_h:G1Projective)->Result<AnchorProof> {
        let mut machine=Self::default();
        let mut unused_undo=Vec::new();
        while machine.cursor<program.commands.len() {
            let (command,next)=initial_command(program,machine.cursor);
            if let Command::Transform {plan,..}=&command {
                if machine.transform.is_none() {machine.transform=Some(plan.initialization_plan()?);}
            }
            if machine.execute(params,program,storage,backend,&command,&mut unused_undo,Some(public_h),&mut Default::default())? {
                machine.cursor=next;
            }
        }
        let anchor=machine.anchor();
        validate_initial_caches(params,storage,&anchor,&mut VerificationTrace::disabled())?;
        Ok(anchor)
    }
    pub fn anchor(&self)->AnchorProof {
        let p: [G1Affine;POINTS]=std::array::from_fn(|i| self.points[i].into_affine());
        let wires=WireCommitments {s1:p[0],s2:p[1],s3:p[2],s4:p[3],s5:p[4],s6:p[5]};
        let commitments=crate::DynamoCommitments {alpha:p[6],beta:p[7],z:p[8],h:p[9],b:p[10]};
        let dynamo=crate::DynamoProof {commitments,q:G1Affine::identity(),
            instance_z:p[11],instance_h:p[12]};
        let slices=std::array::from_fn(|t| {
            let i=13+7*t;
            crate::SliceConsistencyProof {qac:p[i],qcd:p[i+1],qbd:p[i+2],d:p[i+3],
                a_degree:p[i+4],b_degree:p[i+5],d_degree:p[i+6]}
        });
        AnchorProof {wires,s5_g2:self.s5_g2.into_affine(),w_commitment:p[11],
            h_commitment:p[12],dynamo,consistency:ConsistencyProof {slices},
            public_zero_quotient:p[PUBLIC_QUOTIENT],
            multiplication_quotient:p[MULTIPLICATION_QUOTIENT]}
    }

    pub fn advance<B:RebuildBackend>(&mut self,params:&DynaverseParams,program:&Program,
        storage:&mut AnchorStorage,backend:&mut B,allowance:u128,undo:&mut Vec<HostUndo>,
        receipt:&mut UpdateReceipt,timings:&mut BlockTimings,trace:&mut UpdateTimings,
        sparse_stats:&Cell<SparseDispatchStats>)->Result<()> {
        while self.cursor<program.commands.len() {
            let planned=&program.commands[self.cursor];
            let mut cache_wire=None;
            if program.cache_reuse {
                let source=self.source.as_ref().unwrap();
                for wire in 0..3 {
                    if self.cursor==program.cache_spans[wire].start && source.wire_mask&(1<<(wire+3))==0 {
                        cache_wire=Some(wire);break;
                    }
                }
            }
            let copy_len=program.resource.tile_len.min(params.circuit.n()-self.cache_offset);
            let capture=self.cursor==program.linear_span.start
                && self.source.as_ref().is_some_and(|s|s.fixed.is_some());
            let reuse=!capture && program.linear_reuse && self.cursor==program.linear_span.start
                && self.source.as_ref().is_some_and(|s|s.delta.len()<=program.window.window);
            let upper=if cache_wire.is_some() {program::cache_copy_bound(copy_len)?}
                else if capture {program::capture_copy_bound()}
                else if reuse {program.linear_correction_upper} else {planned.upper};
            let cost=upper.units()?;
            if cost>allowance-receipt.rebuild_units {break;}
            let started=Instant::now();
            let mut g2_shortcut=false;
            let done=if let Some(wire)=cache_wire {
                let copy_started=trace.cache_detail.start();
                let source=self.source.as_ref().unwrap();
                let old_commitment=match wire {0=>source.anchor.proof.wires.s4,
                    1=>source.anchor.proof.wires.s5,_=>source.anchor.proof.wires.s6};
                ensure!(self.points[wire+3]==G1Projective::from(old_commitment),
                    "unchanged cache commitment differs from verified anchor");
                let old=match wire {0=>&source.anchor.storage.cache.q4,
                    1=>&source.anchor.storage.cache.q5,_=>&source.anchor.storage.cache.q6};
                let start=self.cache_offset;let end=start+copy_len;
                let candidate=cache_mut(&mut storage.cache,wire);
                // Copy, never alias: old anchor stays immutable while the
                // candidate is recycled at switches. Undo every host write.
                undo.push(HostUndo::Cache(wire,start,candidate[start..end].to_vec()));
                candidate[start..end].copy_from_slice(&old[start..end]);
                self.cache_offset=end;
                trace.cache_detail.finish(7,copy_started);
                receipt.cache_reused_wire_mask|=1<<wire;
                receipt.cache_reused_points+=copy_len;
                if end==params.circuit.n() {
                    self.cursor=program.cache_spans[wire].end;self.cache_offset=0;
                }
                true
            } else if capture {
                let fixed=self.source.as_ref().unwrap().fixed.as_ref().unwrap();
                // Snapshot ownership is private to this frozen target. Only
                // fixed G1 fields are copied: global G2 and quotient still run.
                self.points[..56].copy_from_slice(fixed.points());
                self.cursor=program.linear_g1_end;
                receipt.frozen_linear_captured=true;
                receipt.frozen_linear_skipped_g1_points=program.linear_g1_points;
                true
            } else if reuse {
                let source=self.source.as_ref().unwrap();
                let selected=SparseBackend {backend:&*backend,policy:program.sparse_policy,stats:sparse_stats};
                let corrected=linear::correct_traced(params,&source.anchor.proof,&source.delta,&selected,&mut trace.linear_detail)?;
                // Publish all fields/cursor only after the full correction succeeds.
                self.points=corrected.points;self.s5_g2=corrected.s5_g2;
                self.cursor=program.linear_span.end;
                receipt.linear_reused=true;
                receipt.linear_skipped_g1_points=program.linear_g1_points;
                true
            } else if self.cursor==program.linear_g1_end && program.g2_reuse_limit.is_some() {
                let source=self.source.as_ref().unwrap();
                let selected=SparseBackend {backend:&*backend,policy:program.sparse_policy,stats:sparse_stats};
                let corrected=global_g2::correct(params,source,program.g2_reuse_limit.unwrap(),&selected)?;
                if let Some((point,count))=corrected {
                    self.s5_g2=point;self.cursor=program.linear_span.end;g2_shortcut=true;
                    receipt.global_g2_reused=true;receipt.global_g2_inputs=count;
                    receipt.global_g2_skipped_points=params.circuit.n();
                    true
                } else {
                    self.execute(params,program,storage,backend,&planned.command,undo,None,&mut trace.cache_detail)?
                }
            } else {self.execute(params,program,storage,backend,&planned.command,undo,None,&mut trace.cache_detail)?};
            if done && cache_wire.is_none() {
                if let Command::Transform {packet:true,len,..}=&planned.command {
                    receipt.transform_packet_calls+=1;receipt.transform_packet_elements+=*len as u64;
                }
            }
            let elapsed=started.elapsed();
            if capture {trace.linear_detail.record(2,elapsed);}
            else if !reuse && cache_wire.is_none() {
                match &planned.command {
                    Command::Msm {..}=>trace.linear_detail.record(
                        if program.linear_span.contains(&self.cursor) {0} else {3},elapsed),
                    Command::G2 {..}=>trace.linear_detail.record(1,elapsed),
                    _=>{},
                }
            }
            timings.record(elapsed);
            let phase=if cache_wire.is_some() {RebuildPhase::Cache}
                else if reuse || capture {RebuildPhase::Linear} else {match &planned.command {
                Command::Materialize {..}=>RebuildPhase::Materialize,
                Command::Msm {..}|Command::G2 {..}=>RebuildPhase::Linear,
                Command::Transform {kind:BufferKind::Fr,..}=>RebuildPhase::FrTransform,
                Command::Transform {..}=>RebuildPhase::G1Transform,
                Command::Zip {..}|Command::Scale {..}|Command::CheckRemainder {..}=>RebuildPhase::Polynomial,
                Command::Cache {..}=>RebuildPhase::Cache,
                Command::Finish=>RebuildPhase::Finish,
                _=>RebuildPhase::Memory,
            }};
            trace.record(phase,elapsed);
            receipt.rebuild_units+=cost;
            receipt.work=receipt.work.checked_add(upper)?;
            receipt.blocks+=1;
            if done && !reuse && !capture && !g2_shortcut && cache_wire.is_none() {self.cursor+=1;}
        }
        Ok(())
    }

    fn execute<B:RebuildBackend>(&mut self,params:&DynaverseParams,program:&Program,
        storage:&mut AnchorStorage,backend:&mut B,command:&Command,
        undo:&mut Vec<HostUndo>,initial_public_h:Option<G1Projective>,
        trace:&mut crate::incremental_rebuild::CacheCheckTimings)->Result<bool> {
        let n=params.circuit.n();
        match command {
            Command::Materialize {start,len} => {
                let end=start+len;
                let values=if let Some(source)=&self.source {
                    let mut values=source.anchor.storage.witness.wires[*start..end].to_vec();
                    for (&i,&difference) in source.delta.range(*start..end) {values[i-start]+=difference;}
                    values
                } else {
                    ensure!(initial_public_h.is_some(),"missing frozen snapshot");
                    storage.witness.wires[*start..end].to_vec()
                };
                if initial_public_h.is_none() {
                    undo.push(HostUndo::Witness(*start,storage.witness.wires[*start..end].to_vec()));
                }
                backend.write_fields(program.witness,*start,&values)?;
                storage.witness.wires[*start..end].copy_from_slice(&values);
            }
            Command::Copy {src,si,dst,di,len} =>
                backend.copy_buffer(*src,*si,*dst,*di,*len)?,
            Command::Reverse {kind,src,si,dst,di,len} =>
                backend.reverse_copy(*kind,*src,*si,*dst,*di,*len)?,
            Command::Transform {plan,packet,..} => {
                if *packet && initial_public_h.is_none() {
                    ensure!(self.transform.is_none(),"packet cannot resume a tiled transform cursor");
                    plan.execute_packet(backend)?;return Ok(true);
                }
                if self.transform.is_none() {self.transform=Some(plan.clone());}
                let done=self.transform.as_mut().unwrap().next_step(backend)?;
                if done {self.transform=None;}
                return Ok(done);
            }
            Command::Zip {op,left,li,right,ri,dst,di,len} =>
                backend.field_zip(*op,*left,*li,*right,*ri,*dst,*di,*len)?,
            Command::Scale {points,scalars,dst,start,len} =>
                backend.scale_points(*points,*start,*scalars,*start,*dst,*start,*len)?,
            Command::Msm {points,pi,scalars,si,len,slot} => {
                self.points[*slot]+=backend.msm_g1(*points,*pi,*scalars,*si,*len)?;
            }
            Command::G2 {start,len} => {
                let bases=&params.srs.sparse.small_lagrange_g2[*start..start+len];
                self.s5_g2+=backend.msm_g2(bases,program.witness,4*n+start)?;
            }
            Command::CheckRemainder {buffer,start,len} => {
                let mut values=vec![Fr::zero();*len];
                backend.read_fields(*buffer,*start,&mut values)?;
                ensure!(values.iter().all(Zero::is_zero),
                    "incremental multiplication quotient has a nonzero remainder");
            }
            Command::Cache {wire,buffer,start,len} => {
                let started=trace.start();
                let mut points=vec![G1Affine::identity();*len];
                backend.read_points(*buffer,*start,&mut points)?;
                trace.finish(0,started);
                if trace.enabled {
                    trace.read_batches+=1;trace.read_points+=*len as u64;
                    trace.max_read_roots=trace.max_read_roots.max(*len);
                }
                let commitment=self.points[wire+3];
                // Validate every cache point, rather than hiding a full CPU
                // group FFT or an unbounded validation pass at installation.
                if initial_public_h.is_none() {
                    let mut pairing_trace=if trace.enabled {VerificationTrace::new()}
                        else {VerificationTrace::disabled()};
                    validate_cache_block_prepared(params,storage,commitment,*wire,*start,&points,
                        program.cache_mode,&mut pairing_trace,trace,program.cache_preparation.as_ref())?;
                }
                let started=trace.start();
                let cache=cache_mut(&mut storage.cache,*wire);
                if initial_public_h.is_none() {
                    undo.push(HostUndo::Cache(*wire,*start,cache[*start..start+len].to_vec()));
                }
                cache[*start..start+len].copy_from_slice(&points);
                trace.finish(6,started);
            }
            Command::Finish => {
                let anchor=self.anchor();
                let public_h=match initial_public_h {
                    Some(value)=>value,
                    None=>self.source.as_ref().ok_or_else(|| anyhow::anyhow!("missing source"))?.public_h,
                };
                ensure!(params.verifying.verify_anchor_core(&anchor)?
                    && G1Projective::from(anchor.h_commitment)+public_h==G1Projective::zero(),
                    "incremental candidate anchor failed verification");
            }
        }
        Ok(true)
    }
}

/// Keep every cache equation independent. The initial path reuses the existing
/// exact product checker: two Miller terms, one final exponentiation per point.
/// Reference mode preserves the original two-pairing rebuild validation.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn validate_cache_block(params:&DynaverseParams,storage:&AnchorStorage,
    commitment:G1Projective,wire:usize,start:usize,points:&[G1Affine],
    mode:VerificationMode,trace:&mut VerificationTrace)->Result<()> {
    validate_cache_block_traced(params,storage,commitment,wire,start,points,mode,trace,
        &mut Default::default())
}

/// A checker owns at most five prepared G2 identities (generator + four roots)
/// and dies at the sub-tile boundary. No SRS-dependent state escapes this call.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn validate_cache_block_traced(params:&DynaverseParams,storage:&AnchorStorage,
    commitment:G1Projective,wire:usize,start:usize,points:&[G1Affine],
    mode:VerificationMode,trace:&mut VerificationTrace,
    detail:&mut crate::incremental_rebuild::CacheCheckTimings)->Result<()> {
    validate_cache_block_prepared(params,storage,commitment,wire,start,points,mode,trace,detail,None)
}

#[allow(clippy::too_many_arguments)]
fn validate_cache_block_prepared(params:&DynaverseParams,storage:&AnchorStorage,
    commitment:G1Projective,wire:usize,start:usize,points:&[G1Affine],
    mode:VerificationMode,trace:&mut VerificationTrace,
    detail:&mut crate::incremental_rebuild::CacheCheckTimings,
    prepared:Option<&preparation::PublicPreparation>)->Result<()> {
    let n=params.circuit.n();
    ensure!(wire<3 && start<=n && points.len()<=n-start,
        "cache validation range outside wire");
    ensure!(storage.witness.wires.len()==6*n,"cache witness length differs from circuit");
    if let Some(prepared)=prepared {
        ensure!(mode==VerificationMode::Product,"prepared cache requires Product equations");
        prepared.validate_owner(params)?;
    }
    let g2=params.srs.kzg_x.rebuild_g2_powers();
    let g1=params.srs.kzg_x.rebuild_g1_powers()[0];
    for (chunk,openings) in points.chunks(4).enumerate() {
        let first=start+4*chunk;
        let divisors=if mode==VerificationMode::Product {
            let started=detail.start();
            let mut projective=Vec::with_capacity(openings.len());
            for j in 0..openings.len() {
                if prepared.is_some_and(|table|table.contains(first+j)) {continue;}
                projective.push(G2Projective::from(g2[1])
                    -G2Projective::from(g2[0])*params.srs.sparse.small_points[first+j]);
            }
            let divisors=G2Projective::normalize_batch(&projective);
            detail.finish(2,started);
            if detail.enabled {detail.divisors_formed+=projective.len() as u64;}
            Some(divisors)
        } else {None};
        let mut checks=PairingChecks::new(mode,trace);
        let mut uncached=0;
        for (j,&opening) in openings.iter().enumerate() {
            let i=first+j;
            let started=detail.start();
            let value=storage.witness.wires[(wire+3)*n+i];
            let difference=(commitment-G1Projective::from(g1)*value).into_affine();
            detail.finish(1,started);
            if let Some(table)=prepared.filter(|table|table.contains(i)) {
                let started=detail.start();
                let (_,generator,divisor)=table.pair(i,params.srs.sparse.small_points[i])?;
                detail.finish(2,started);
                ensure!(checks.equal_prepared(difference,opening,generator,divisor),
                    "incremental anchor opening cache check failed at wire {} index {i}",wire+4);
                continue;
            }
            let divisor=if let Some(divisors)=&divisors {
                let value=divisors[uncached];uncached+=1;value
            } else {
                let started=detail.start();
                let value=(G2Projective::from(g2[1])
                    -G2Projective::from(g2[0])*params.srs.sparse.small_points[i]).into_affine();
                detail.finish(2,started);
                if detail.enabled {detail.divisors_formed+=1;}
                value
            };
            ensure!(checks.equal((difference,g2[0]),&[(opening,divisor)]),
                "incremental anchor opening cache check failed at wire {} index {i}",wire+4);
        }
    }
    if detail.enabled {
        let m=trace.phase(crate::trace::VerificationPhase::Structure);
        detail.ns[3]+=m.g2_prepare_time.as_nanos();
        detail.ns[4]+=m.miller_time.as_nanos();
        detail.ns[5]+=m.final_exponentiation_time.as_nanos();
        detail.equations+=m.counts.equations as u64;
        detail.pairing_terms+=m.counts.pairing_terms as u64;
        detail.miller_calls+=m.counts.miller_loops as u64;
        detail.final_exponentiations+=m.counts.final_exponentiations as u64;
        detail.g2_prepare_builds+=m.g2_prepare_builds as u64;
        detail.g2_prepare_hits+=m.g2_prepare_hits as u64;
        let retained=prepared.map_or(0,|p|p.stats().prepared_entries);
        detail.prepared_peak_entries=detail.prepared_peak_entries.max(retained+m.prepared_peak_entries);
    }
    Ok(())
}

/// Unpublished initialization only. Four roots bound the prepared-G2 cache;
/// each divisor is formed once and reused by three separate KZG equations.
fn validate_initial_caches(params:&DynaverseParams,storage:&AnchorStorage,
    anchor:&AnchorProof,trace:&mut VerificationTrace)->Result<usize> {
    let n=params.circuit.n();
    let powers=params.srs.kzg_x.rebuild_g2_powers();
    let g1=params.srs.kzg_x.rebuild_g1_powers()[0];
    let commitments=[anchor.wires.s4,anchor.wires.s5,anchor.wires.s6];
    let caches=[&storage.cache.q4,&storage.cache.q5,&storage.cache.q6];
    let mut formed=0;
    for start in (0..n).step_by(4) {
        let len=4.min(n-start);
        let mut projective=Vec::with_capacity(len);
        for i in start..start+len {
            projective.push(G2Projective::from(powers[1])
                -G2Projective::from(powers[0])*params.srs.sparse.small_points[i]);
            formed+=1;
        }
        let divisors=G2Projective::normalize_batch(&projective);
        let mut checks=PairingChecks::new(VerificationMode::Product,trace);
        for wire in 0..3 {
            for (j,&divisor) in divisors.iter().enumerate() {
                let i=start+j;
                let value=storage.witness.wires[(wire+3)*n+i];
                let difference=(G1Projective::from(commitments[wire])-G1Projective::from(g1)*value).into_affine();
                ensure!(checks.equal((difference,powers[0]),&[(caches[wire][i],divisor)]),
                    "incremental anchor opening cache check failed at wire {} index {i}",wire+4);
            }
        }
    }
    Ok(formed)
}

/// Fuse only consecutive partial MSMs with identical destination and contiguous
/// input ranges. No algebra or operation ordering changes. The initializer is
/// caller-paid and capped at 65536 points per native submission.
fn initial_command(program:&Program,cursor:usize)->(Command,usize) {
    let mut command=program.commands[cursor].command.clone();
    let mut next=cursor+1;
    if let Command::Msm {points,pi,scalars,si,len,slot}=&mut command {
        while next<program.commands.len() {
            match &program.commands[next].command {
                Command::Msm {points:p,pi:i,scalars:s,si:j,len:l,slot:k}
                    if p==points && s==scalars && k==slot && *i==*pi+*len
                        && *j==*si+*len && *len+*l<=65536 => {*len+=*l;next+=1;}
                _=>break,
            }
        }
    }
    (command,next)
}

pub(super) fn cache_mut(cache:&mut KzgPrecomputation,wire:usize)->&mut Vec<G1Affine> {
    match wire {0=>&mut cache.q4,1=>&mut cache.q5,2=>&mut cache.q6,_=>unreachable!()}
}

pub(super) fn rollback_host(storage:&mut AnchorStorage,undo:Vec<HostUndo>) {
    for entry in undo.into_iter().rev() {
        match entry {
            HostUndo::Witness(start,values) =>
                storage.witness.wires[start..start+values.len()].copy_from_slice(&values),
            HostUndo::Cache(wire,start,points) =>
                cache_mut(&mut storage.cache,wire)[start..start+points.len()].copy_from_slice(&points),
        }
    }
}

#[cfg(test)]
mod initial_cache_check_tests {
    use super::*;
    use crate::trace::VerificationTrace;
    use rand::{rngs::StdRng,SeedableRng};

    #[test]
    fn prepared_cache_undo_restores_a_written_chunk_after_a_later_equation_fails() {
        use crate::incremental_rebuild::operators::CpuRebuildBackend;
        for readback in [None,Some(8)] {
        let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let params=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(26002)).unwrap();
        let (proof,aux)=params.prove_with_aux(&i,&w).unwrap();
        let mut backend=CpuRebuildBackend::default();
        let program=Program::new(&params,&IncrementalConfig {rebuild_cache_mode:VerificationMode::Product,
            rebuild_prepared_cache_roots:Some(n),rebuild_cache_readback_roots:readback,
            ..Default::default()},&mut backend).unwrap();
        let cache_commands:Vec<_>=program.commands.iter().filter_map(|step|match &step.command {
            Command::Cache {wire:0,..}=>Some(step.command.clone()),_=>None}).collect();
        let (buffer,first_len,second_start)=match (&cache_commands[0],&cache_commands[1]) {
            (Command::Cache {buffer,len,..},Command::Cache {start,..})=>(*buffer,*len,*start),
            _=>unreachable!(),
        };
        backend.write_points(buffer,0,&aux.kzg.q4).unwrap();
        let mut machine=Machine::default();machine.points[3]=proof.anchor.wires.s4.into();
        let mut storage=AnchorStorage {witness:w,cache:KzgPrecomputation {
            q4:vec![G1Affine::identity();n],q5:vec![G1Affine::identity();n],q6:vec![G1Affine::identity();n]}};
        let original=storage.cache.q4.clone();let mut undo=Vec::new();
        let mut detail=crate::incremental_rebuild::CacheCheckTimings::default();
        backend.begin_transaction().unwrap();
        machine.execute(&params,&program,&mut storage,&mut backend,&cache_commands[0],&mut undo,None,&mut detail).unwrap();
        assert_eq!(&storage.cache.q4[..first_len],&aux.kzg.q4[..first_len]);
        assert_ne!(storage.cache.q4,original,"first successful command must actually mutate candidate cache");
        assert_eq!(undo.len(),1);
        let bad=(G1Projective::from(aux.kzg.q4[second_start])+G1Projective::generator()).into_affine();
        backend.write_points(buffer,second_start,&[bad]).unwrap();
        let error=machine.execute(&params,&program,&mut storage,&mut backend,&cache_commands[1],
            &mut undo,None,&mut detail).unwrap_err();
        assert!(error.to_string().contains("opening cache check failed"));
        assert_eq!(undo.len(),1,"the rejected chunk must not be written");
        rollback_host(&mut storage,undo);backend.rollback_transaction().unwrap();
        assert_eq!(storage.cache.q4,original,"rollback must restore an already accepted cache write");
        let mut retry_undo=Vec::new();backend.begin_transaction().unwrap();
        for command in &cache_commands {
            machine.execute(&params,&program,&mut storage,&mut backend,command,&mut retry_undo,None,&mut detail).unwrap();
        }
        assert_eq!(storage.cache.q4,aux.kzg.q4);backend.commit_transaction();
        }
    }

    #[test]
    fn cache_readback_requires_the_full_certificate_before_any_read_or_write() {
        use crate::incremental_rebuild::operators::CpuRebuildBackend;
        let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let params=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(26140)).unwrap();
        let (proof,aux)=params.prove_with_aux(&i,&w).unwrap();
        let mut backend=CpuRebuildBackend::default();
        let program=Program::new(&params,&IncrementalConfig {rebuild_cache_mode:VerificationMode::Product,
            rebuild_prepared_cache_roots:Some(3),rebuild_cache_readback_roots:Some(8),
            ..Default::default()},&mut backend).unwrap();
        let cursor=program.commands.iter().position(|s|matches!(s.command,Command::Cache {wire:0,..})).unwrap();
        let buffer=match program.commands[cursor].command {Command::Cache {buffer,..}=>buffer,_=>unreachable!()};
        backend.write_points(buffer,0,&aux.kzg.q4).unwrap();
        let mut machine=Machine {cursor,..Default::default()};machine.points[3]=proof.anchor.wires.s4.into();
        let mut storage=AnchorStorage {witness:w,cache:KzgPrecomputation {
            q4:vec![G1Affine::identity();n],q5:vec![G1Affine::identity();n],q6:vec![G1Affine::identity();n]}};
        let cost=program.commands[cursor].upper.units().unwrap();
        let before=backend.stats();let mut undo=Vec::new();let mut receipt=UpdateReceipt::default();
        let mut timing=BlockTimings::default();let mut trace=UpdateTimings::new(true);
        let sparse=Cell::new(SparseDispatchStats::default());
        for allowance in [0,cost-1] {
            machine.advance(&params,&program,&mut storage,&mut backend,allowance,&mut undo,
                &mut receipt,&mut timing,&mut trace,&sparse).unwrap();
            assert_eq!(backend.stats(),before);assert_eq!(machine.cursor,cursor);
            assert!(undo.is_empty());assert_eq!(receipt.rebuild_units,0);
            assert_eq!(trace.cache_detail.read_batches,0);assert_eq!(trace.cache_detail.equations,0);
            assert!(storage.cache.q4.iter().all(|p|*p==G1Affine::identity()));
        }
        backend.begin_transaction().unwrap();
        machine.advance(&params,&program,&mut storage,&mut backend,cost,&mut undo,
            &mut receipt,&mut timing,&mut trace,&sparse).unwrap();
        assert_eq!(machine.cursor,cursor+1);assert_eq!(receipt.rebuild_units,cost);
        assert_eq!(trace.cache_detail.read_batches,1);assert_eq!(trace.cache_detail.read_points,8);
        assert_eq!(trace.cache_detail.equations,8);assert_eq!(undo.len(),1);
        assert_eq!(&storage.cache.q4[..8],&aux.kzg.q4[..8]);backend.commit_transaction();
    }

    #[test]
    fn resident_preparations_reject_changed_equations_and_cancelling_errors() {
        use ark_ff::UniformRand;
        let n=8;let c=ToyCircuit::new(n).unwrap();let(_,mut witness)=c.valid_instance();
        let params=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(25040)).unwrap();
        let mut rng=StdRng::seed_from_u64(25041);
        for value in &mut witness.wires[3*n..4*n] {*value=Fr::rand(&mut rng);}
        let domain=radix2_domain(n).unwrap();let coeff=domain.ifft(&witness.wires[3*n..4*n]);
        let commitment=G1Projective::from(params.srs.kzg_x.commit(&coeff).unwrap().point);
        let points=params.srs.kzg_x.precompute_single_openings(&coeff,&domain).unwrap();
        let mut storage=AnchorStorage {witness,cache:KzgPrecomputation {q4:vec![],q5:vec![],q6:vec![]}};
        for cap in [3,8] {
            let bound=preparation::PublicPreparation::host_bound(n,Some(cap),VerificationMode::Product).unwrap();
            let table=preparation::PublicPreparation::new(&params,cap,bound).unwrap();
            let check=|storage:&AnchorStorage,commitment,start,points:&[G1Affine]| {
                validate_cache_block_prepared(&params,storage,commitment,0,start,points,VerificationMode::Product,
                    &mut VerificationTrace::disabled(),&mut Default::default(),Some(&table))
            };
            for (start,len) in [(0,8),(2,3),(7,1),(8,0)] {
                check(&storage,commitment,start,&points[start..start+len]).unwrap();
            }
            assert!(check(&storage,commitment+G1Projective::generator(),0,&points).is_err());
            let old=storage.witness.wires[3*n+1];storage.witness.wires[3*n+1]+=Fr::from(1u64);
            assert!(check(&storage,commitment,0,&points).is_err());storage.witness.wires[3*n+1]=old;
            let mut corrupt=points.clone();corrupt.swap(0,1);
            assert!(check(&storage,commitment,0,&corrupt).is_err());
            let powers=params.srs.kzg_x.rebuild_g1_powers();
            let mut corrupt=points.clone();
            corrupt[0]=(G1Projective::from(corrupt[0])+G1Projective::from(powers[1])
                -G1Projective::from(powers[0])*params.srs.sparse.small_points[1]).into_affine();
            corrupt[1]=(G1Projective::from(corrupt[1])-G1Projective::from(powers[1])
                +G1Projective::from(powers[0])*params.srs.sparse.small_points[0]).into_affine();
            assert!(check(&storage,commitment,0,&corrupt).is_err());
            assert!(check(&storage,commitment,8,&points[..1]).is_err());
            let zero=G1Affine::identity();
            let mut constant=storage.witness.clone();constant.wires[3*n..4*n].fill(Fr::from(7u64));
            let constant=AnchorStorage {witness:constant,cache:KzgPrecomputation {q4:vec![],q5:vec![],q6:vec![]}};
            check(&constant,G1Projective::from(powers[0])*Fr::from(7u64),0,&[zero;8]).unwrap();
        }
    }

    #[test]
    fn packet_requires_the_complete_transform_certificate_before_any_backend_work() {
        use crate::incremental_rebuild::operators::CpuRebuildBackend;
        let n=8;let c=ToyCircuit::new(n).unwrap();let(_,witness)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(23100)).unwrap();
        let mut backend=CpuRebuildBackend::default();
        let program=Program::new(&p,&IncrementalConfig {transform_packet_max_len:Some(16),..Default::default()},&mut backend).unwrap();
        let cursor=program.commands.iter().position(|c|matches!(c.command,Command::Transform {packet:true,..})).unwrap();
        let len=match &program.commands[cursor].command {Command::Transform {len,..}=>*len,_=>unreachable!()};
        assert_eq!(program.commands[cursor].repeats,1);
        let cost=program.commands[cursor].upper.units().unwrap();
        let mut machine=Machine {cursor,..Default::default()};
        let mut storage=AnchorStorage {witness,cache:KzgPrecomputation {q4:vec![G1Affine::identity();n],q5:vec![G1Affine::identity();n],q6:vec![G1Affine::identity();n]}};
        let mut receipt=UpdateReceipt::default();let mut undo=Vec::new();let mut timings=BlockTimings::default();
        let mut trace=UpdateTimings::new(true);let stats=Cell::new(SparseDispatchStats::default());let before=backend.stats();
        for allowance in [0,cost-1] {
            machine.advance(&p,&program,&mut storage,&mut backend,allowance,&mut undo,&mut receipt,&mut timings,&mut trace,&stats).unwrap();
            assert_eq!(backend.stats(),before);assert_eq!(machine.cursor,cursor);
        }
        backend.begin_transaction().unwrap();
        machine.advance(&p,&program,&mut storage,&mut backend,cost,&mut undo,&mut receipt,&mut timings,&mut trace,&stats).unwrap();
        assert_eq!(receipt.rebuild_units,cost);assert_eq!(receipt.transform_packet_calls,1);
        assert_eq!(receipt.transform_packet_elements,len as u64);assert_eq!(machine.cursor,cursor+1);
        assert!(machine.transform.is_none());assert_eq!(backend.stats().fr_transform_elements-before.fr_transform_elements,2*len as u64);
        backend.commit_transaction();
    }

    #[test]
    fn global_g2_probe_requires_its_certified_chunk_allowance() {
        use crate::incremental_rebuild::operators::CpuRebuildBackend;
        let n=8;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(20070)).unwrap();
        let (proof,aux)=params.prove_with_aux(&instance,&witness).unwrap();
        let anchor=Arc::new(AnchorEpoch {storage:AnchorStorage {witness:witness.clone(),cache:aux.kzg},proof:proof.anchor.clone(),generation:0});
        let mut backend=CpuRebuildBackend::default();
        let program=Program::new(&params,&IncrementalConfig {frozen_g2_reuse_limit:Some(3),..Default::default()},&mut backend).unwrap();
        let mut machine=Machine {cursor:program.linear_g1_end,source:Some(Frozen {anchor,
            delta:BTreeMap::from([(4*n+1,Fr::from(1u64)),(5*n+1,witness.wires[3*n+1])]),
            generation:1,public_h:-G1Projective::from(proof.anchor.h_commitment),wire_mask:0b11_0000,fixed:None}),..Default::default()};
        let mut storage=AnchorStorage {witness,cache:KzgPrecomputation {q4:vec![G1Affine::identity();n],q5:vec![G1Affine::identity();n],q6:vec![G1Affine::identity();n]}};
        let cost=program.commands[program.linear_g1_end].upper.units().unwrap();
        let mut receipt=UpdateReceipt::default();let mut undo=Vec::new();let mut timings=BlockTimings::default();
        let mut trace=UpdateTimings::new(true);let stats=Cell::new(SparseDispatchStats::default());
        for allowance in [0,cost-1] {
            let (result,count)=crate::workcount::measure(||machine.advance(&params,&program,&mut storage,
                &mut backend,allowance,&mut undo,&mut receipt,&mut timings,&mut trace,&stats));
            result.unwrap();assert_eq!(count.scan_elements,0);assert_eq!(count.g2_msm_points,0);
            assert_eq!(machine.cursor,program.linear_g1_end);assert_eq!(receipt.blocks,0);
        }
        backend.begin_transaction().unwrap();
        machine.advance(&params,&program,&mut storage,&mut backend,cost,&mut undo,&mut receipt,&mut timings,&mut trace,&stats).unwrap();
        assert_eq!(machine.cursor,program.linear_span.end);assert!(receipt.global_g2_reused);
        assert_eq!(receipt.global_g2_inputs,1);assert_eq!(receipt.rebuild_units,cost);
        backend.commit_transaction();
    }

    #[test]
    fn frozen_g1_capture_is_immutable_while_overlay_changes_before_global_g2() {
        use crate::incremental_rebuild::operators::CpuRebuildBackend;
        let n=16;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(19440)).unwrap();
        let (proof,aux)=params.prove_with_aux(&instance,&witness).unwrap();
        let anchor=Arc::new(AnchorEpoch {storage:AnchorStorage {witness:witness.clone(),cache:aux.kzg},proof:proof.anchor.clone(),generation:0});
        let mut target=witness.clone();target.wires[4*n+1]+=Fr::from(1u64);
        target.wires[5*n+1]=target.wires[3*n+1]*target.wires[4*n+1];
        let expected=params.prove_anchor(&instance,&target).unwrap().anchor;
        let delta=BTreeMap::from([(4*n+1,target.wires[4*n+1]-witness.wires[4*n+1]),
            (5*n+1,target.wires[5*n+1]-witness.wires[5*n+1])]);
        let fixed=FixedLinear::from_anchor(&expected).unwrap();
        let mut backend=CpuRebuildBackend::default();
        let program=Program::new(&params,&IncrementalConfig {foreground_linear_reuse:true,
            freeze_linear_capture_limit:Some(32),..Default::default()},&mut backend).unwrap();
        let mut machine=Machine {source:Some(Frozen {anchor,delta,generation:1,
            public_h:-G1Projective::from(proof.anchor.h_commitment),wire_mask:0b11_0000,fixed:Some(fixed)}),..Default::default()};
        let mut storage=AnchorStorage {witness:Witness {wires:vec![Fr::zero();6*n]},
            cache:KzgPrecomputation {q4:vec![G1Affine::identity();n],q5:vec![G1Affine::identity();n],q6:vec![G1Affine::identity();n]}};
        let mut allowance=program::capture_copy_bound().units().unwrap();
        for step in &program.commands[..program.linear_span.start] {allowance+=step.upper.units().unwrap();}
        let mut receipt=UpdateReceipt::default();let mut undo=Vec::new();let mut timings=BlockTimings::default();
        let mut trace=UpdateTimings::new(true);let stats=Cell::new(SparseDispatchStats::default());
        backend.begin_transaction().unwrap();
        machine.advance(&params,&program,&mut storage,&mut backend,allowance,&mut undo,&mut receipt,&mut timings,&mut trace,&stats).unwrap();
        assert!(receipt.frozen_linear_captured);assert_eq!(machine.cursor,program.linear_g1_end);
        assert_eq!(trace.linear_detail.ns[1],0);
        // Later live edits, including a dirty cache wire and a reversion of a
        // frozen entry, belong only to overlay and cannot change captured T(S).
        machine.overlay.insert(3*n+2,Fr::from(9u64));
        machine.overlay.insert(4*n+1,-machine.source.as_ref().unwrap().delta[&(4*n+1)]);
        let mut receipt=UpdateReceipt::default();
        machine.advance(&params,&program,&mut storage,&mut backend,program.budget.total,&mut undo,&mut receipt,&mut timings,&mut trace,&stats).unwrap();
        assert_eq!(machine.cursor,program.commands.len());assert_eq!(machine.anchor(),expected);
        assert_eq!(storage.witness,target);assert!(trace.linear_detail.ns[1]>0);
        backend.commit_transaction();
    }

    #[test]
    fn frozen_cache_copy_reads_verified_source_despite_later_dirty_overlay() {
        use crate::incremental_rebuild::operators::CpuRebuildBackend;
        let n=8;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18600)).unwrap();
        let (proof,aux)=params.prove_with_aux(&instance,&witness).unwrap();
        let old_q4=aux.kzg.q4.clone();
        let anchor=Arc::new(AnchorEpoch {storage:AnchorStorage {witness:witness.clone(),cache:aux.kzg},
            proof:proof.anchor.clone(),generation:0});
        let delta=BTreeMap::from([(4*n+1,Fr::from(1u64)),(5*n+1,witness.wires[3*n+1])]);
        let mut frozen_witness=witness.clone();
        for (&i,&v) in &delta {frozen_witness.wires[i]+=v;}
        let corrected=linear::correct(&params,&proof.anchor,&delta,&CpuRebuildBackend::default()).unwrap();
        let mut backend=CpuRebuildBackend::default();
        let program=Program::new(&params,&IncrementalConfig {unchanged_cache_reuse:true,..Default::default()},&mut backend).unwrap();
        let mut machine=Machine {source:Some(Frozen {anchor,delta,generation:1,
                public_h:-G1Projective::from(proof.anchor.h_commitment),wire_mask:0b11_0000,fixed:None}),
            overlay:BTreeMap::from([(n+2,Fr::from(1u64)),(2*n+2,Fr::from(1u64)),
                (3*n+2,Fr::from(1u64)),(5*n+2,witness.wires[4*n+2])]),
            cursor:program.cache_spans[0].start,points:corrected.points,s5_g2:corrected.s5_g2,..Default::default()};
        let mut storage=AnchorStorage {witness:frozen_witness,cache:KzgPrecomputation {
            q4:vec![G1Affine::identity();n],q5:vec![G1Affine::identity();n],q6:vec![G1Affine::identity();n]}};
        let len=program.resource.tile_len.min(n);let allowance=program::cache_copy_bound(len).unwrap().units().unwrap();
        let mut undo=Vec::new();let mut receipt=UpdateReceipt::default();let mut timings=BlockTimings::default();
        backend.begin_transaction().unwrap();
        machine.advance(&params,&program,&mut storage,&mut backend,allowance,&mut undo,&mut receipt,
            &mut timings,&mut UpdateTimings::default(),&Cell::new(SparseDispatchStats::default())).unwrap();
        assert_eq!(receipt.cache_reused_points,len);assert_eq!(receipt.cache_reused_wire_mask,1);
        assert_eq!(&storage.cache.q4[..len],&old_q4[..len]);assert_eq!(machine.cache_offset,len);
        backend.commit_transaction();undo.clear();
        // Start at a genuinely partial, committed checkpoint. Fail the next
        // caller transaction after it copies the rest and jumps the span.
        let checkpoint=machine.clone();let partial_cache=storage.cache.q4.clone();
        backend.begin_transaction().unwrap();
        let failed:Result<()>=(|| {
            machine.advance(&params,&program,&mut storage,&mut backend,allowance,&mut undo,
                &mut UpdateReceipt::default(),&mut timings,&mut UpdateTimings::default(),
                &Cell::new(SparseDispatchStats::default()))?;
            ensure!(machine.cursor==program.cache_spans[0].end && machine.cache_offset==0,"second tile must finish before injected failure");
            anyhow::bail!("injected caller failure after partial-checkpoint continuation")
        })();
        assert!(failed.unwrap_err().to_string().contains("injected caller failure"));
        rollback_host(&mut storage,undo);backend.rollback_transaction().unwrap();machine=checkpoint;
        assert_eq!(machine.cache_offset,len);assert_eq!(machine.cursor,program.cache_spans[0].start);
        assert_eq!(storage.cache.q4,partial_cache);
        let mut retry_undo=Vec::new();backend.begin_transaction().unwrap();
        machine.advance(&params,&program,&mut storage,&mut backend,allowance,&mut retry_undo,
            &mut UpdateReceipt::default(),&mut timings,&mut UpdateTimings::default(),
            &Cell::new(SparseDispatchStats::default())).unwrap();
        assert_eq!(storage.cache.q4,old_q4);assert_eq!(machine.cache_offset,0);
        assert_eq!(machine.cursor,program.cache_spans[0].end);backend.commit_transaction();
    }

    #[test]
    fn initial_three_wire_validation_reuses_divisors_without_merging_equations() {
        let circuit=ToyCircuit::new(8).unwrap();let (instance,witness)=circuit.valid_instance();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(7401)).unwrap();
        let (proof,aux)=params.prove_with_aux(&instance,&witness).unwrap();
        let mut storage=AnchorStorage {witness,cache:aux.kzg};
        let mut trace=VerificationTrace::new();
        assert_eq!(validate_initial_caches(&params,&storage,&proof.anchor,&mut trace).unwrap(),8);
        assert_eq!(trace.counts().equations,24);assert_eq!(trace.counts().final_exponentiations,24);
        // Same-root opposite errors in two wires cancel if equations are merged.
        let g=G1Projective::generator();
        storage.cache.q4[1]=(G1Projective::from(storage.cache.q4[1])+g).into_affine();
        storage.cache.q5[1]=(G1Projective::from(storage.cache.q5[1])-g).into_affine();
        assert!(validate_initial_caches(&params,&storage,&proof.anchor,&mut VerificationTrace::disabled()).is_err());
    }

    #[test]
    fn exact_initial_cache_checks_use_one_final_exponentiation_per_point() {
        let circuit=ToyCircuit::new(8).unwrap();
        let (_,witness)=circuit.valid_instance();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(6701)).unwrap();
        let domain=radix2_domain(8).unwrap();
        let coefficients=domain.ifft(&witness.wires[24..32]);
        let commitment=params.srs.kzg_x.commit(&coefficients).unwrap().point.into();
        let points=params.srs.kzg_x.precompute_single_openings(&coefficients,&domain).unwrap();
        let storage=AnchorStorage {witness,cache:KzgPrecomputation {q4:vec![],q5:vec![],q6:vec![]}};
        for mode in [VerificationMode::Product,VerificationMode::Reference] {
            let mut trace=VerificationTrace::new();
            validate_cache_block(&params,&storage,commitment,0,0,&points,mode,&mut trace).unwrap();
            assert_eq!(trace.counts().equations,8);
            assert_eq!(trace.counts().pairing_terms,16);
            let expected=if mode==VerificationMode::Product {8} else {16};
            assert_eq!(trace.counts().final_exponentiations,expected);
            assert!(trace.phase(crate::trace::VerificationPhase::Structure).prepared_peak_entries<=5);
            // These two errors cancel in a single product over both equations:
            // e(tau-x1,tau-x0) / e(tau-x0,tau-x1)=1. Each equation must fail.
            let powers=params.srs.kzg_x.rebuild_g1_powers();
            let mut corrupt=points.clone();
            corrupt[0]=(G1Projective::from(corrupt[0])+G1Projective::from(powers[1])
                -G1Projective::from(powers[0])*params.srs.sparse.small_points[1]).into_affine();
            corrupt[1]=(G1Projective::from(corrupt[1])-G1Projective::from(powers[1])
                +G1Projective::from(powers[0])*params.srs.sparse.small_points[0]).into_affine();
            assert!(validate_cache_block(&params,&storage,commitment,0,0,&corrupt,mode,
                &mut VerificationTrace::disabled()).is_err());
        }
    }

    #[test]
    fn cache_validation_handles_identity_tail_empty_and_bounds_in_both_modes() {
        let n=8;let c=ToyCircuit::new(n).unwrap();let (_,mut witness)=c.valid_instance();
        let params=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19006)).unwrap();
        witness.wires[3*n..4*n].fill(Fr::from(7u64));
        let commitment=G1Projective::from(params.srs.kzg_x.rebuild_g1_powers()[0])*Fr::from(7u64);
        let storage=AnchorStorage {witness,cache:KzgPrecomputation {q4:vec![],q5:vec![],q6:vec![]}};
        for mode in [VerificationMode::Reference,VerificationMode::Product] {
            for (start,len) in [(0,8),(7,1),(8,0)] {
                let mut t=VerificationTrace::new();
                validate_cache_block(&params,&storage,commitment,0,start,&vec![G1Affine::identity();len],mode,&mut t).unwrap();
                assert_eq!(t.counts().equations,len);
            }
            for (wire,start,len) in [(3,0,1),(0,8,1),(0,usize::MAX,0)] {
                assert!(validate_cache_block(&params,&storage,commitment,wire,start,&vec![G1Affine::identity();len],mode,
                    &mut VerificationTrace::disabled()).is_err());
            }
        }
    }

    #[test]
    fn rebuild_rejects_opposite_wire_errors_in_each_mode() {
        let n=8;let c=ToyCircuit::new(n).unwrap();let (instance,witness)=c.valid_instance();
        let params=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19007)).unwrap();
        let (proof,aux)=params.prove_with_aux(&instance,&witness).unwrap();
        let storage=AnchorStorage {witness,cache:aux.kzg};
        let g=G1Projective::generator();
        for mode in [VerificationMode::Reference,VerificationMode::Product] {
            for (wire,commitment,cache,sign) in [(0,proof.anchor.wires.s4,&storage.cache.q4,g),
                (1,proof.anchor.wires.s5,&storage.cache.q5,-g)] {
                let corrupt=(G1Projective::from(cache[1])+sign).into_affine();
                assert!(validate_cache_block(&params,&storage,commitment.into(),wire,1,&[corrupt],mode,
                    &mut VerificationTrace::disabled()).is_err());
            }
        }
    }
}
