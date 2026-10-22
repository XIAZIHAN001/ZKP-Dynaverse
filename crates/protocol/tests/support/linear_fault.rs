use anyhow::{bail,Result};
use ark_bn254::{Fr,G1Affine,G1Projective,G2Affine};
use ark_ec::{CurveGroup,Group};
use dynaverse_protocol::{BatchMsmRequest,OperatorBackend};
use dynaverse_protocol::incremental_rebuild::operators::{
    BufferId,BufferKind,FieldBinaryOp,RebuildBackend,RebuildStats,CpuRebuildBackend};
use std::{cell::Cell,rc::Rc};

#[derive(Default)]
pub struct Control {calls:Cell<usize>,fail:Cell<Option<usize>>,fail_g2:Cell<bool>,fail_cache_read:Cell<bool>,
    fail_materialize:Cell<bool>,corrupt_cache:Cell<bool>,fail_resident_g2:Cell<bool>,
    corrupt_after:Cell<Option<usize>>,cache_reads:Cell<usize>}
impl Control {
    pub fn arm(&self,call:usize) {self.calls.set(0);self.fail.set(Some(call));}
    pub fn arm_g2(&self) {self.fail_g2.set(true);}
    pub fn arm_resident_g2(&self) {self.fail_resident_g2.set(true);}
    pub fn arm_cache_read(&self) {self.fail_cache_read.set(true);}
    pub fn arm_materialize(&self) {self.fail_materialize.set(true);}
    pub fn arm_corrupt_cache(&self) {self.corrupt_cache.set(true);}
    /// Corrupt a later read after this many preceding reads returned valid data.
    pub fn arm_corrupt_cache_after(&self,successful_reads:usize) {
        self.cache_reads.set(0);self.corrupt_after.set(Some(successful_reads));
    }
    pub fn cache_reads(&self)->usize {self.cache_reads.get()}
    fn enter(&self)->Result<()> {
        let call=self.calls.get();self.calls.set(call+1);
        if self.fail.get()==Some(call) {
            self.fail.set(None);bail!("injected linear batch failure {call}");
        }
        Ok(())
    }
}
pub struct FaultBackend {cpu:CpuRebuildBackend,control:Rc<Control>}
impl FaultBackend {
    pub fn new()->(Self,Rc<Control>) {
        let control=Rc::new(Control::default());
        (Self {cpu:CpuRebuildBackend::default(),control:control.clone()},control)
    }
}
impl OperatorBackend for FaultBackend {
    fn msm(&self,p:&[G1Affine],s:&[Fr])->Result<G1Affine> {self.cpu.msm(p,s)}
    fn msm_batch(&self,r:&BatchMsmRequest<'_>)->Result<Vec<G1Affine>> {
        self.control.enter()?;self.cpu.msm_batch(r)
    }
    fn msm_g2_host(&self,p:&[G2Affine],s:&[Fr])->Result<G2Affine> {
        if self.control.fail_g2.replace(false) {bail!("injected linear G2 failure");}
        self.cpu.msm_g2_host(p,s)
    }
    fn ntt(&self,v:&mut[Fr],inverse:bool)->Result<()> {self.cpu.ntt(v,inverse)}
}
macro_rules! forward {
    ($(fn $name:ident($($arg:ident:$ty:ty),*)->$result:ty;)*)=>{
        $(fn $name(&mut self,$($arg:$ty),*)->$result {self.cpu.$name($($arg),*)})*
    }
}
impl RebuildBackend for FaultBackend {
    fn msm_g2(&mut self,points:&[G2Affine],scalars:BufferId,offset:usize)->Result<G2Affine> {
        if self.control.fail_resident_g2.replace(false) {bail!("injected frozen global G2 failure");}
        self.cpu.msm_g2(points,scalars,offset)
    }
    fn seal(&mut self) {self.cpu.seal();}
    fn buffer_len(&self,id:BufferId)->Result<usize> {self.cpu.buffer_len(id)}
    fn buffer_kind(&self,id:BufferId)->Result<BufferKind> {self.cpu.buffer_kind(id)}
    fn stats(&self)->RebuildStats {self.cpu.stats()}
    fn commit_transaction(&mut self) {self.cpu.commit_transaction();}
    fn read_points(&mut self,id:BufferId,offset:usize,values:&mut[G1Affine])->Result<()> {
        if self.control.fail_cache_read.replace(false) {bail!("injected cache read failure after q4 reuse");}
        self.cpu.read_points(id,offset,values)?;
        let read=self.control.cache_reads.get();self.control.cache_reads.set(read+1);
        let delayed=self.control.corrupt_after.get()==Some(read);
        if delayed {self.control.corrupt_after.set(None);}
        if self.control.corrupt_cache.replace(false) || delayed {
            values[0]=(G1Projective::from(values[0])+G1Projective::generator()).into_affine();
        }
        Ok(())
    }
    fn write_fields(&mut self,id:BufferId,offset:usize,values:&[Fr])->Result<()> {
        if self.control.fail_materialize.replace(false) {bail!("injected materialize after complete sparse proof");}
        self.cpu.write_fields(id,offset,values)
    }
    forward! {
        fn alloc(kind:BufferKind,len:usize,writable:bool)->Result<BufferId>;
        fn begin_transaction()->Result<()>;
        fn rollback_transaction()->Result<()>;
        fn read_fields(id:BufferId,offset:usize,values:&mut[Fr])->Result<()>;
        fn write_points(id:BufferId,offset:usize,values:&[G1Affine])->Result<()>;
        fn copy_buffer(src:BufferId,si:usize,dst:BufferId,di:usize,len:usize)->Result<()>;
        fn reverse_copy(kind:BufferKind,src:BufferId,si:usize,dst:BufferId,di:usize,len:usize)->Result<()>;
        fn transpose(kind:BufferKind,src:BufferId,dst:BufferId,rows:usize,cols:usize,start:usize,len:usize,twiddle:Option<Fr>)->Result<()>;
        fn transform_rows(kind:BufferKind,src:BufferId,dst:BufferId,row_len:usize,first_row:usize,row_count:usize,inverse:bool)->Result<()>;
        fn field_zip(op:FieldBinaryOp,left:BufferId,li:usize,right:BufferId,ri:usize,dst:BufferId,di:usize,len:usize)->Result<()>;
        fn scale_points(points:BufferId,po:usize,scalars:BufferId,so:usize,dst:BufferId,di:usize,len:usize)->Result<()>;
        fn msm_g1(points:BufferId,po:usize,scalars:BufferId,so:usize,len:usize)->Result<G1Affine>;
    }
}
