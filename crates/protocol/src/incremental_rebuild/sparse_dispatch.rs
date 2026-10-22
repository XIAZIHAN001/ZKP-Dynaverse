//! Explicit small-operator policy. An execution error is never a reason to
//! silently retry a GPU operation on the CPU.
use crate::{BatchMsmRequest,GpuCapabilities,OperatorBackend,SparseMsmRequest};
use crate::dynaverse::ArkReferenceBackend;
use anyhow::{ensure,Result};
use ark_bn254::{Fr,G1Affine,G2Affine};
use ark_ec::CurveConfig;
use std::cell::Cell;

// The selected-point preflight below is sufficient only for BN254 G1's
// cofactor-one group. Keep that premise explicit in CPU-only builds too.
const _: () = {
    assert!(ark_bn254::g1::Config::COFACTOR.len()==1);
    assert!(ark_bn254::g1::Config::COFACTOR[0]==1);
};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum SparseDispatchPolicy {
    CpuReference,
    /// User/calibration supplied crossover, frozen at session initialization.
    Auto {cpu_below:usize},
    ForceGpu,
}
impl Default for SparseDispatchPolicy {
    fn default()->Self {Self::Auto {cpu_below:0}}
}

#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct OperatorCounts {
    pub g1_calls:u64,pub g1_points:u64,
    pub g2_calls:u64,pub g2_points:u64,
    pub ntt_calls:u64,pub ntt_elements:u64,
}
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct SparseDispatchStats {pub cpu:OperatorCounts,pub gpu:OperatorCounts}

pub(crate) struct SparseBackend<'a> {
    pub backend:&'a dyn OperatorBackend,
    pub policy:SparseDispatchPolicy,
    pub stats:&'a Cell<SparseDispatchStats>,
}
impl SparseBackend<'_> {
    fn reference(&self,len:usize)->bool {
        match self.policy {
            SparseDispatchPolicy::CpuReference=>true,
            SparseDispatchPolicy::Auto {cpu_below}=>len<cpu_below,
            SparseDispatchPolicy::ForceGpu=>false,
        }
    }
    fn record(&self,kind:u8,len:usize,gpu:bool) {
        let mut stats=self.stats.get();
        let selected=if gpu {&mut stats.gpu} else {&mut stats.cpu};
        match kind {
            0=>{selected.g1_calls=selected.g1_calls.saturating_add(1);
                selected.g1_points=selected.g1_points.saturating_add(len as u64);}
            1=>{selected.g2_calls=selected.g2_calls.saturating_add(1);
                selected.g2_points=selected.g2_points.saturating_add(len as u64);}
            _=>{selected.ntt_calls=selected.ntt_calls.saturating_add(1);
                selected.ntt_elements=selected.ntt_elements.saturating_add(len as u64);}
        }
        self.stats.set(stats);
    }
}
impl OperatorBackend for SparseBackend<'_> {
    fn msm(&self,points:&[G1Affine],scalars:&[Fr])->Result<G1Affine> {
        let reference=self.reference(scalars.len());
        self.record(0,scalars.len(),!reference && self.backend.gpu_capabilities().g1_msm);
        if reference {ArkReferenceBackend.msm(points,scalars)}
        else {self.backend.msm(points,scalars)}
    }
    fn msm_batch(&self,request:&BatchMsmRequest<'_>)->Result<Vec<G1Affine>> {
        ensure!(request.point_sets.len()==request.offsets.len()
            && request.offsets.len()==request.lengths.len(),"batch shape mismatch");
        let all_forward=request.lengths.iter().all(|&len|!self.reference(len));
        if all_forward {
            for &len in request.lengths {
                self.record(0,len,self.backend.gpu_capabilities().g1_msm);
            }
            return self.backend.msm_batch(request);
        }
        let mut result=Vec::with_capacity(request.lengths.len());
        for i in 0..request.lengths.len() {
            let len=request.lengths[i];
            let start=request.offsets[i];
            let end=start.checked_add(len).ok_or_else(||anyhow::anyhow!("batch range overflow"))?;
            ensure!(len<=request.point_sets[i].len() && end<=request.shared_scalars.len(),
                "batch range out of bounds");
            result.push(self.msm(&request.point_sets[i][..len],&request.shared_scalars[start..end])?);
        }
        Ok(result)
    }
    fn msm_sparse_batch(&self,requests:&[SparseMsmRequest<'_>])->Result<Vec<G1Affine>> {
        for request in requests {
            request.validate()?;
            // CPU reference assumes valid points. Validate its selections too,
            // before forwarding any GPU part of a mixed-policy batch.
            for &(index,_) in request.entries {
                ensure!(request.bases[index].is_on_curve(),"invalid G1 point");
            }
        }
        if requests.iter().all(|request|!self.reference(request.entries.len())) {
            for request in requests {
                self.record(0,request.entries.len(),self.backend.gpu_capabilities().g1_msm);
            }
            return self.backend.msm_sparse_batch(requests);
        }
        let mut result=vec![G1Affine::identity();requests.len()];
        let mut forwarded=Vec::new();
        let mut indices=Vec::new();
        forwarded.try_reserve_exact(requests.len())?;
        indices.try_reserve_exact(requests.len())?;
        for (index,request) in requests.iter().enumerate() {
            let reference=self.reference(request.entries.len());
            self.record(0,request.entries.len(),!reference && self.backend.gpu_capabilities().g1_msm);
            if reference {
                result[index]=ArkReferenceBackend.msm_sparse_batch(std::slice::from_ref(request))?[0];
            } else {
                // Metadata only: retain caller bases/entries through the
                // synchronous call, without gathering selected input vectors.
                forwarded.push(*request);
                indices.push(index);
            }
        }
        if !forwarded.is_empty() {
            let points=self.backend.msm_sparse_batch(&forwarded)?;
            ensure!(points.len()==forwarded.len(),"backend returned wrong sparse commitment count");
            for (slot,&index) in indices.iter().enumerate() {result[index]=points[slot];}
        }
        Ok(result)
    }
    fn ntt(&self,values:&mut[Fr],inverse:bool)->Result<()> {
        let reference=self.reference(values.len());
        self.record(2,values.len(),!reference && self.backend.gpu_capabilities().fr_ntt);
        if reference {ArkReferenceBackend.ntt(values,inverse)}
        else {self.backend.ntt(values,inverse)}
    }
    fn msm_g2_host(&self,points:&[G2Affine],scalars:&[Fr])->Result<G2Affine> {
        let reference=self.reference(scalars.len());
        self.record(1,scalars.len(),!reference && self.backend.gpu_capabilities().g2_msm);
        if reference {ArkReferenceBackend.msm_g2_host(points,scalars)}
        else {self.backend.msm_g2_host(points,scalars)}
    }
    fn gpu_capabilities(&self)->GpuCapabilities {self.backend.gpu_capabilities()}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{G1Projective,G2Projective};
    use ark_ec::{CurveGroup,Group};

    #[derive(Default)]
    struct Routed {calls:Cell<usize>,fail:Cell<bool>,indexed:Cell<usize>}
    impl Routed {
        fn enter(&self)->Result<()> {
            self.calls.set(self.calls.get()+1);
            ensure!(!self.fail.get(),"selected backend failure"); Ok(())
        }
    }
    impl OperatorBackend for Routed {
        fn msm(&self,p:&[G1Affine],s:&[Fr])->Result<G1Affine> {
            self.enter()?; ArkReferenceBackend.msm(p,s)
        }
        fn msm_batch(&self,r:&BatchMsmRequest<'_>)->Result<Vec<G1Affine>> {
            self.enter()?; ArkReferenceBackend.msm_batch(r)
        }
        fn msm_sparse_batch(&self,r:&[crate::SparseMsmRequest<'_>])->Result<Vec<G1Affine>> {
            self.indexed.set(self.indexed.get()+1);
            self.enter()?; ArkReferenceBackend.msm_sparse_batch(r)
        }
        fn ntt(&self,v:&mut[Fr],inverse:bool)->Result<()> {
            self.enter()?; ArkReferenceBackend.ntt(v,inverse)
        }
        fn msm_g2_host(&self,p:&[G2Affine],s:&[Fr])->Result<G2Affine> {
            self.enter()?; ArkReferenceBackend.msm_g2_host(p,s)
        }
        fn gpu_capabilities(&self)->GpuCapabilities {
            // A routing spy, not a hardware implementation. Hardware equality
            // is tested separately by the icicle feature integration tests.
            GpuCapabilities {g1_msm:true,g2_msm:true,fr_ntt:true}
        }
    }

    #[test]
    fn explicit_reference_and_threshold_route_each_operator() {
        let backend=Routed::default();
        let stats=Cell::new(SparseDispatchStats::default());
        let p=vec![G1Projective::generator().into_affine();4];
        let q=vec![G2Projective::generator().into_affine();4];
        let s=vec![Fr::from(3u64);4];
        for policy in [SparseDispatchPolicy::CpuReference,SparseDispatchPolicy::Auto {cpu_below:4},SparseDispatchPolicy::ForceGpu] {
            stats.set(Default::default()); backend.calls.set(0);
            let selected=SparseBackend {backend:&backend,policy,stats:&stats};
            for len in [2,4] {
                assert_eq!(selected.msm(&p[..len],&s[..len]).unwrap(),ArkReferenceBackend.msm(&p[..len],&s[..len]).unwrap());
                assert_eq!(selected.msm_g2_host(&q[..len],&s[..len]).unwrap(),ArkReferenceBackend.msm_g2_host(&q[..len],&s[..len]).unwrap());
                let mut values=s[..len].to_vec();
                selected.ntt(&mut values,false).unwrap();
                selected.ntt(&mut values,true).unwrap(); assert_eq!(values,s[..len]);
            }
            let st=stats.get();
            let gpu_calls=match policy {SparseDispatchPolicy::CpuReference=>0,SparseDispatchPolicy::Auto {..}=>4,SparseDispatchPolicy::ForceGpu=>8};
            assert_eq!(backend.calls.get(),gpu_calls);
            assert_eq!(st.gpu.g1_calls+st.gpu.g2_calls+st.gpu.ntt_calls,gpu_calls as u64);
            assert_eq!(st.cpu.g1_calls+st.cpu.g2_calls+st.cpu.ntt_calls,(8-gpu_calls) as u64);
        }
    }

    #[test]
    fn mixed_batch_routes_individually_and_errors_do_not_retry() {
        let backend=Routed::default();
        let stats=Cell::new(SparseDispatchStats::default());
        let p=vec![G1Projective::generator().into_affine();4];
        let q=vec![G2Projective::generator().into_affine();4];
        let s=vec![Fr::from(3u64);4];
        let sets=[p.as_slice(),p.as_slice()]; let offsets=[0,0]; let lengths=[2,4];
        let request=BatchMsmRequest {point_sets:&sets,shared_scalars:&s,offsets:&offsets,lengths:&lengths};
        let selected=SparseBackend {backend:&backend,policy:SparseDispatchPolicy::Auto {cpu_below:4},stats:&stats};
        assert_eq!(selected.msm_batch(&request).unwrap(),ArkReferenceBackend.msm_batch(&request).unwrap());
        assert_eq!(backend.calls.get(),1); assert_eq!(stats.get().cpu.g1_calls,1); assert_eq!(stats.get().gpu.g1_calls,1);
        backend.fail.set(true);
        assert!(selected.msm(&p,&s).unwrap_err().to_string().contains("selected backend failure"));
        assert!(selected.msm_g2_host(&q,&s).unwrap_err().to_string().contains("selected backend failure"));
        assert!(selected.ntt(&mut s.clone(),false).unwrap_err().to_string().contains("selected backend failure"));
        let full=SparseBackend {policy:SparseDispatchPolicy::ForceGpu,..selected};
        assert!(full.msm_batch(&request).unwrap_err().to_string().contains("selected backend failure"));
        assert_eq!(backend.calls.get(),5);
    }

    #[test]
    fn indexed_policy_preserves_thresholds_batches_gpu_and_does_not_retry_errors() {
        let backend=Routed::default();
        let stats=Cell::new(SparseDispatchStats::default());
        let points=vec![G1Projective::generator().into_affine();4];
        let small=[(3,Fr::from(3u64));2];
        let large=[(1,Fr::from(5u64));4];
        let requests=[crate::SparseMsmRequest {bases:&points,entries:&small},
            crate::SparseMsmRequest {bases:&points,entries:&large},
            crate::SparseMsmRequest {bases:&points,entries:&large}];
        let selected=SparseBackend {backend:&backend,
            policy:SparseDispatchPolicy::Auto {cpu_below:4},stats:&stats};
        let expected=vec![(G1Projective::generator()*Fr::from(6u64)).into_affine(),
            (G1Projective::generator()*Fr::from(20u64)).into_affine(),
            (G1Projective::generator()*Fr::from(20u64)).into_affine()];
        assert_eq!(selected.msm_sparse_batch(&requests).unwrap(),expected);
        assert_eq!(backend.calls.get(),1);
        assert_eq!(backend.indexed.get(),1,"selected GPU batch must remain indexed");
        assert_eq!(stats.get().cpu.g1_calls,1);
        assert_eq!(stats.get().cpu.g1_points,2);
        assert_eq!(stats.get().gpu.g1_calls,2);
        assert_eq!(stats.get().gpu.g1_points,8);
        let bad=[(usize::MAX,Fr::from(0u64))];
        assert!(selected.msm_sparse_batch(&[requests[0],crate::SparseMsmRequest {
            bases:&points,entries:&bad}]).is_err());
        assert_eq!(backend.calls.get(),1);
        let reference=SparseBackend {policy:SparseDispatchPolicy::CpuReference,..selected};
        assert_eq!(reference.msm_sparse_batch(&requests).unwrap(),expected);
        assert_eq!(backend.calls.get(),1);
        backend.fail.set(true);
        let forced=SparseBackend {policy:SparseDispatchPolicy::ForceGpu,..reference};
        assert!(forced.msm_sparse_batch(&requests).unwrap_err().to_string()
            .contains("selected backend failure"));
        assert_eq!(backend.calls.get(),2);
        assert_eq!(backend.indexed.get(),2);
    }

    #[test]
    fn indexed_policy_rejects_bad_cpu_point_before_forwarding_any_gpu_request() {
        let backend=Routed::default();
        let stats=Cell::new(SparseDispatchStats::default());
        let points=[G1Projective::generator().into_affine();4];
        let invalid=[G1Affine::new_unchecked(ark_bn254::Fq::from(1u64),ark_bn254::Fq::from(1u64))];
        let large=[(0,Fr::from(5u64));4];
        let zero=[(0,Fr::from(0u64))];
        let requests=[crate::SparseMsmRequest {bases:&points,entries:&large},
            crate::SparseMsmRequest {bases:&invalid,entries:&zero}];
        for policy in [SparseDispatchPolicy::Auto {cpu_below:4},
            SparseDispatchPolicy::CpuReference,SparseDispatchPolicy::ForceGpu] {
            let selected=SparseBackend {backend:&backend,policy,stats:&stats};
            assert!(selected.msm_sparse_batch(&requests).is_err());
            assert_eq!(backend.calls.get(),0,"late invalid point must reject before GPU forwarding");
            assert_eq!(backend.indexed.get(),0);
            assert_eq!(stats.get(),SparseDispatchStats::default());
        }
    }
}
