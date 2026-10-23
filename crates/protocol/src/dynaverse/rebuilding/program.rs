//! Immutable, initialization-time command program. Epochs reuse these handles
//! and commands; starting or switching never constructs an n-element plan.
use super::*;
use crate::incremental_rebuild::operators::MAX_TRANSFORM_PACKET_LEN;

pub(super) const POINTS: usize = 57;
pub(super) const PUBLIC_QUOTIENT: usize = 55;
pub(super) const MULTIPLICATION_QUOTIENT: usize = 56;

#[derive(Clone)]
pub(super) enum Command {
    Materialize { start: usize, len: usize },
    Copy { src: BufferId, si: usize, dst: BufferId, di: usize, len: usize },
    Reverse { kind: BufferKind, src: BufferId, si: usize, dst: BufferId, di: usize, len: usize },
    Transform {plan:FourStepPlan,kind:BufferKind,len:usize,packet:bool},
    Zip { op: FieldBinaryOp, left: BufferId, li: usize, right: BufferId, ri: usize,
        dst: BufferId, di: usize, len: usize },
    Scale { points: BufferId, scalars: BufferId, dst: BufferId, start: usize, len: usize },
    Msm { points: BufferId, pi: usize, scalars: BufferId, si: usize, len: usize, slot: usize },
    G2 { start: usize, len: usize },
    CheckRemainder { buffer: BufferId, start: usize, len: usize },
    Cache { wire: usize, buffer: BufferId, start: usize, len: usize },
    Finish,
}

pub(super) struct Planned {
    pub command: Command,
    pub upper: WorkBound,
    pub repeats: usize,
}

pub(super) struct Program {
    pub commands: Vec<Planned>,
    pub witness: BufferId,
    pub budget: RebuildBudget,
    pub window: CachedWindow,
    pub resource: ResourcePlan,
    pub linear_reuse:bool,
    pub linear_span:std::ops::Range<usize>,
    pub linear_g1_end:usize,
    pub capture_limit:Option<usize>,
    pub g2_reuse_limit:Option<usize>,
    pub linear_g1_points:usize,
    pub linear_correction_upper:WorkBound,
    pub sparse_policy:SparseDispatchPolicy,
    pub cache_reuse:bool,
    pub cache_spans:[std::ops::Range<usize>;3],
    pub row_batch:usize,
    pub packet_max_len:Option<usize>,
    pub cache_mode:VerificationMode,
    pub cache_preparation:Option<preparation::PublicPreparation>,
    pub cache_divisors:Option<divisors::PublicDivisors>,
}

impl Program {
    fn tiles(len: usize, tile: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..len).step_by(tile).map(move |i| (i, tile.min(len-i)))
    }

    fn add(&mut self, command: Command, upper: WorkBound, repeats: usize) {
        self.commands.push(Planned { command, upper, repeats });
    }

    #[allow(clippy::too_many_arguments)]
    fn copies(&mut self, src: BufferId, si: usize, dst: BufferId, di: usize,
        len: usize, tile: usize, kind: BufferKind) {
        for (i,count) in Self::tiles(len,tile) {
            self.add(Command::Copy {src,si:si+i,dst,di:di+i,len:count},
                linear_bound(count,kind,0),1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reverses(&mut self, src: BufferId, si: usize, dst: BufferId, di: usize,
        len: usize, tile: usize, kind: BufferKind) {
        for (i,count) in Self::tiles(len,tile) {
            self.add(Command::Reverse {kind,src,si:si+len-i-count,dst,di:di+i,len:count},
                linear_bound(count,kind,0),1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fft(&mut self,kind:BufferKind,len:usize,input:BufferId,scratch:BufferId,
        output:BufferId,tile:usize,inverse:bool)->Result<()> {
        let plan=FourStepPlan::new(kind,len,input,scratch,output,tile,self.row_batch,inverse)?;
        let mut repeats=plan.total_steps();
        let row=1usize << ((len.trailing_zeros() as usize+1)/2);
        let batched=row.checked_mul(self.row_batch).ok_or_else(||anyhow::anyhow!("row batch bound overflow"))?.min(len);
        let elements=tile.max(batched);
        let log=row.trailing_zeros() as usize+1;
        let mut upper=linear_bound(elements,kind,128);
        upper.field_ops=(elements as u64).checked_mul((256*log) as u64)
            .ok_or_else(|| anyhow::anyhow!("transform cost overflow"))?;
        if kind==BufferKind::G1 {
            upper.g1_adds=(elements*log) as u64;
            upper.g1_muls=(elements*log) as u64;
        }
        let packet=self.packet_max_len.is_some_and(|cap|len<=cap);
        if packet {
            // One submission must pay for every stage that its old cursor
            // would have executed, plus whole-output journal preflight.
            upper=upper.checked_mul(repeats as u64)?.checked_add(packet_preflight_bound(len)?)?;
            repeats=1;
        }
        self.add(Command::Transform {plan,kind,len,packet},upper,repeats);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn msms(&mut self,points:BufferId,pi:usize,scalars:BufferId,si:usize,
        len:usize,slot:usize,tile:usize) {
        for (i,count) in Self::tiles(len,tile) {
            let mut upper=linear_bound(count,BufferKind::G1,512);
            // Envelope for fixed 254-bit scalars, partial buckets/reduction,
            // normalization, staging and local-result accumulation.
            upper.g1_muls=(4*count+64) as u64;
            upper.g1_adds=1024*((count as u64)+64);
            self.add(Command::Msm {points,pi:pi+i,scalars,si:si+i,len:count,slot},upper,1);
        }
    }

    pub fn new<B:RebuildBackend>(params:&DynaverseParams,config:&IncrementalConfig,
        backend:&mut B)->Result<Self> {
        let n=params.circuit.n();
        let cache_divisor_host_bytes=divisors::PublicDivisors::host_bound(n,
            config.rebuild_divisor_cache,config.rebuild_cache_mode)?;
        let cache_preparation_host_bytes=preparation::PublicPreparation::host_bound(n,
            config.rebuild_prepared_cache_roots,config.rebuild_cache_mode)?;
        let cache_readback_host_bytes=if let Some(cap)=config.rebuild_cache_readback_roots {
            ensure!((4..=64).contains(&cap),"cache readback roots must be in 4..=64");
            // Affine read Vec, wrapper encoding/native normalization and bounded
            // undo staging coexist. The four-root pairing scratch is unchanged.
            n.min(cap) as u64*512+4096
        } else {0};
        ensure!((1..=16).contains(&config.transform_row_batch),"transform row batch must be in 1..=16");
        if let Some(cap)=config.transform_packet_max_len {
            ensure!((1..=MAX_TRANSFORM_PACKET_LEN).contains(&cap),"transform packet cap must be in1..=4096");
        }
        if let Some(limit)=config.freeze_linear_capture_limit {
            ensure!(config.foreground_linear_reuse,"frozen linear capture requires foreground linear reuse");
            ensure!((1..=4096).contains(&limit),"frozen linear capture limit must be in 1..=4096");
        }
        if let Some(limit)=config.frozen_g2_reuse_limit {
            ensure!((1..=4096).contains(&limit),"frozen global G2 limit must be in 1..=4096");
        }
        let window=CachedWindow::new(n,config.window,config.optimized_window)?;
        let minimum_tile={
            let bits=(2*n).trailing_zeros() as usize;
            1 << ((bits+1)/2)
        };
        let tile=config.tile_len.unwrap_or(minimum_tile);
        ensure!(tile>=minimum_tile && tile<=(2*minimum_tile).min(2*n),
            "session tile must be between one and two maximum FFT rows");
        let reserve=256u64 << 20;
        // Conservative device envelope for shared affine bases, dynamic
        // projective arrays, unique-write shadows and native scratch reserve.
        let frozen_linear_bytes=if config.freeze_linear_capture_limit.is_some() {
            3*std::mem::size_of::<FixedLinear>() as u64
        } else {0};
        let g2_entries=config.frozen_g2_reuse_limit.map(|limit|limit.min(n));
        let global_g2_host_bytes=g2_entries.map_or(0,|count|count as u64*512+4096);
        // dvgpu_bn254_g2_msm already preflights 64MiB + 1024*d temporary
        // bytes; synchronous calls do not overlap other operator scratch.
        let global_g2_workspace_bytes=g2_entries.map_or(0,|count|(64u64<<20)+count as u64*1024);
        ensure!(global_g2_workspace_bytes<=reserve,"global G2 workspace exceeds reserve");
        let upper_bytes=(n as u64).checked_mul(16_384)
            .and_then(|bytes| bytes.checked_add(reserve))
            .and_then(|bytes| bytes.checked_add(frozen_linear_bytes))
            .and_then(|bytes| bytes.checked_add(global_g2_host_bytes))
            .and_then(|bytes| bytes.checked_add(cache_preparation_host_bytes))
            .and_then(|bytes| bytes.checked_add(cache_readback_host_bytes))
            .and_then(|bytes| bytes.checked_add(cache_divisor_host_bytes))
            .ok_or_else(|| anyhow::anyhow!("resource estimate overflow"))?;
        ensure!(upper_bytes<=config.memory_limit_bytes,
            "rebuild resource estimate {upper_bytes} exceeds limit {}",config.memory_limit_bytes);
        let cache_divisors=if config.rebuild_divisor_cache {
            Some(divisors::PublicDivisors::new(params,cache_divisor_host_bytes)?)
        } else {None};
        let cache_preparation=match config.rebuild_prepared_cache_roots {
            Some(cap)=>Some(preparation::PublicPreparation::new(params,cap,cache_preparation_host_bytes)?),
            None=>None,
        };
        let witness=backend.alloc(BufferKind::Fr,6*n,true)?;
        let zeros_f=backend.alloc(BufferKind::Fr,2*n,false)?;
        backend.write_fields(zeros_f,0,&vec![Fr::zero();2*n])?;
        let zeros_g=backend.alloc(BufferKind::G1,2*n,false)?;
        backend.write_points(zeros_g,0,&vec![G1Affine::identity();2*n])?;
        let mut registered=std::collections::HashMap::new();
        let small=register_points(backend,&params.srs.sparse.small_lagrange_g1,&mut registered)?;
        let mut dyn_bases=Vec::with_capacity(7);
        for bases in params.srs.dynamo.rebuild_bases() {
            dyn_bases.push(register_points(backend,&bases[..6*n],&mut registered)?);
        }
        let h=params.srs.dynamo.rebuild_h_bases(params.circuit.permutation_ref())?;
        dyn_bases.push(register_points(backend,&h[..6*n],&mut registered)?);
        let mut consistency=Vec::with_capacity(6);
        for key in &params.srs.consistency {
            let mut ids=Vec::with_capacity(7);
            for (i,bases) in key.rebuild_bases().into_iter().enumerate() {
                let len=if i==0 || i==4 {6*n} else {n};
                ids.push(register_points(backend,&bases[..len],&mut registered)?);
            }
            consistency.push(ids);
        }
        let public=register_points(backend,&params.srs.sparse.public_quotient_g1,&mut registered)?;
        let powers=register_points(backend,&params.srs.kzg_x.rebuild_g1_powers()[..n],&mut registered)?;
        let input=backend.alloc(BufferKind::Fr,2*n,true)?;
        let scratch=backend.alloc(BufferKind::Fr,2*n,true)?;
        let coeff=[
            backend.alloc(BufferKind::Fr,n,true)?,
            backend.alloc(BufferKind::Fr,n,true)?,
            backend.alloc(BufferKind::Fr,n,true)?];
        let f0=backend.alloc(BufferKind::Fr,2*n,true)?;
        let f1=backend.alloc(BufferKind::Fr,2*n,true)?;
        let product=backend.alloc(BufferKind::Fr,2*n,true)?;
        let remainder=backend.alloc(BufferKind::Fr,n,true)?;
        let scalar_fft=backend.alloc(BufferKind::Fr,2*n,true)?;
        let g_input=backend.alloc(BufferKind::G1,2*n,true)?;
        let g_scratch=backend.alloc(BufferKind::G1,2*n,true)?;
        let g_output=backend.alloc(BufferKind::G1,2*n,true)?;
        let opening_coeff=backend.alloc(BufferKind::G1,n,true)?;
        let opening_eval=backend.alloc(BufferKind::G1,n,true)?;
        // Polynomial-independent basis FFT is prepared at initialization.
        // These are initialization-only temporaries; later cache generation
        // reuses the same writable arrays after basis_fft has been prepared.
        let basis_input=g_input;
        let basis_scratch=g_scratch;
        let basis_fft=backend.alloc(BufferKind::G1,2*n,false)?;
        backend.write_points(basis_input,0,&vec![G1Affine::identity();2*n])?;
        if n>1 {
            backend.write_points(basis_input,0,&params.srs.kzg_x.rebuild_g1_powers()[..n-1])?;
        }
        let mut prepared=FourStepPlan::new(BufferKind::G1,2*n,basis_input,
            basis_scratch,basis_fft,tile,1,false)?;
        while !prepared.next_step(backend)? {}
        let dummy=RebuildBudget::new(WorkBound::default(),WorkBound::default(),
            WorkBound::default(),window.horizon)?;
        let resource=ResourcePlan {upper_bytes,resident_bytes:0,workspace_reserve:reserve,
            transaction_bytes:0,tile_len:tile,frozen_linear_bytes,global_g2_host_bytes,global_g2_workspace_bytes,
            cache_preparation_host_bytes,cache_readback_host_bytes,cache_divisor_host_bytes};
        let mut p=Self {commands:Vec::new(),witness,budget:dummy,window,resource,
            linear_reuse:config.linear_rebuild_reuse,linear_span:0..0,linear_g1_points:0,
            linear_g1_end:0,capture_limit:config.freeze_linear_capture_limit,
            g2_reuse_limit:config.frozen_g2_reuse_limit,
            linear_correction_upper:WorkBound::default(),sparse_policy:config.sparse,
            cache_reuse:config.unchanged_cache_reuse,cache_spans:std::array::from_fn(|_|0..0),row_batch:config.transform_row_batch,
            packet_max_len:config.transform_packet_max_len,cache_mode:config.rebuild_cache_mode,
            cache_preparation,cache_divisors};
        for (start,len) in Self::tiles(6*n,tile) {
            p.add(Command::Materialize {start,len},linear_bound(len,BufferKind::Fr,0),1);
        }
        p.linear_span.start=p.commands.len();
        for t in 0..6 {p.msms(small,0,witness,t*n,n,t,tile);}
        for (i,&points) in dyn_bases.iter().enumerate() {
            p.msms(points,0,witness,0,6*n,6+i,tile);
        }
        for (t,ids) in consistency.iter().enumerate() {
            for (i,&points) in ids.iter().enumerate() {
                let full=i==0 || i==4;
                p.msms(points,0,witness,if full {0} else {t*n},
                    if full {6*n} else {n},13+7*t+i,tile);
            }
        }
        p.msms(public,0,witness,0,6*n,PUBLIC_QUOTIENT,tile);
        p.linear_g1_end=p.commands.len();
        for (start,len) in Self::tiles(n,tile) {
            let mut upper=linear_bound(len,BufferKind::Fr,512);
            upper.g2_muls=(4*len+64) as u64;
            upper.g2_adds=1024*((len as u64)+64);
            p.add(Command::G2 {start,len},upper,1);
        }
        p.linear_span.end=p.commands.len();
        if let Some(limit)=p.g2_reuse_limit {
            // Charge the full capped alternative AND retain the first dense
            // tile in this one chunk. This covers probing even on fallback,
            // before W/Cmax/Q are certified and before any runtime scan.
            let first=&mut p.commands[p.linear_g1_end];
            first.upper=first.upper.checked_add(global_g2::work_bound(limit)?)?;
        }
        for step in &p.commands[p.linear_span.clone()] {
            if let Command::Msm {len,..}=&step.command {
                p.linear_g1_points=p.linear_g1_points.checked_add(*len)
                    .ok_or_else(||anyhow::anyhow!("linear point inventory overflow"))?;
            }
        }
        for t in 0..3 {
            p.copies(witness,(t+3)*n,input,0,n,tile,BufferKind::Fr);
            p.fft(BufferKind::Fr,n,input,scratch,coeff[t],tile,true)?;
        }
        p.copies(zeros_f,0,input,0,2*n,tile,BufferKind::Fr);
        p.copies(coeff[0],0,input,0,n,tile,BufferKind::Fr);
        p.fft(BufferKind::Fr,2*n,input,scratch,f0,tile,false)?;
        p.copies(zeros_f,0,input,0,2*n,tile,BufferKind::Fr);
        p.copies(coeff[1],0,input,0,n,tile,BufferKind::Fr);
        p.fft(BufferKind::Fr,2*n,input,scratch,f1,tile,false)?;
        for (i,len) in Self::tiles(2*n,tile) {
            p.add(Command::Zip {op:FieldBinaryOp::Mul,left:f0,li:i,right:f1,ri:i,
                dst:product,di:i,len},linear_bound(len,BufferKind::Fr,1),1);
        }
        p.fft(BufferKind::Fr,2*n,product,scratch,input,tile,true)?;
        for (i,len) in Self::tiles(n,tile) {
            p.add(Command::Zip {op:FieldBinaryOp::Add,left:input,li:i,right:input,ri:n+i,
                dst:remainder,di:i,len},linear_bound(len,BufferKind::Fr,1),1);
            p.add(Command::Zip {op:FieldBinaryOp::Sub,left:remainder,li:i,right:coeff[2],ri:i,
                dst:remainder,di:i,len},linear_bound(len,BufferKind::Fr,1),1);
            p.add(Command::CheckRemainder {buffer:remainder,start:i,len},
                linear_bound(len,BufferKind::Fr,0),1);
        }
        p.msms(powers,0,input,n,n.saturating_sub(1),MULTIPLICATION_QUOTIENT,tile);
        for t in 0..3 {
            p.cache_spans[t].start=p.commands.len();
            p.copies(zeros_f,0,input,0,2*n,tile,BufferKind::Fr);
            p.reverses(coeff[t],1,input,0,n.saturating_sub(1),tile,BufferKind::Fr);
            p.fft(BufferKind::Fr,2*n,input,scratch,scalar_fft,tile,false)?;
            for (start,len) in Self::tiles(2*n,tile) {
                let mut upper=linear_bound(len,BufferKind::G1,1);
                upper.g1_muls=len as u64;
                p.add(Command::Scale {points:basis_fft,scalars:scalar_fft,dst:g_input,start,len},upper,1);
            }
            p.fft(BufferKind::G1,2*n,g_input,g_scratch,g_output,tile,true)?;
            p.copies(zeros_g,0,opening_coeff,0,n,tile,BufferKind::G1);
            p.reverses(g_output,0,opening_coeff,0,n.saturating_sub(1),tile,BufferKind::G1);
            p.fft(BufferKind::G1,n,opening_coeff,g_scratch,opening_eval,tile,false)?;
            // Read coalescing never enlarges the inner four-root checker.
            let old_tile=tile.min(4);
            let read_tile=config.rebuild_cache_readback_roots.unwrap_or(old_tile);
            for (start,len) in Self::tiles(n,read_tile) {
                let mut upper=WorkBound::default();
                for (_,count) in Self::tiles(len,old_tile) {
                    upper=upper.checked_add(cache_check_bound(count,p.cache_preparation.is_some())?)?;
                }
                if config.rebuild_cache_readback_roots.is_some() {
                    // Charge the complete old certificate before the first
                    // read/write, plus conservative merged-read bookkeeping.
                    upper=upper.checked_add(WorkBound {field_ops:1024,host_elements:64,
                        copy_bytes:256*len as u64,launches:16,..Default::default()})?;
                }
                p.add(Command::Cache {wire:t,buffer:opening_eval,start,len},upper,1);
            }
            p.cache_spans[t].end=p.commands.len();
        }
        p.add(Command::Finish,WorkBound {field_ops:128_000_000,
            g1_adds:128,g1_muls:128,g2_adds:128,g2_muls:128,
            host_elements:128,..WorkBound::default()},1);
        let mut total=WorkBound::default();
        let mut maximum=WorkBound::default();
        for step in &p.commands {
            total=total.checked_add(step.upper.checked_mul(step.repeats as u64)?)?;
            if step.upper.units()?>maximum.units()? {maximum=step.upper;}
        }
        if p.capture_limit.is_some() {
            let copy=capture_copy_bound();
            total=total.checked_add(copy)?;
            if copy.units()?>maximum.units()? {maximum=copy;}
        }
        if p.linear_reuse {
            p.linear_correction_upper=linear::work_bound(window.window)?;
            // Retain the dense total as a conservative fallback and additionally
            // charge the entire sparse alternative. Recheck the deadline with
            // its new maximum chunk instead of silently reusing old kQ values.
            total=total.checked_add(p.linear_correction_upper)?;
            if p.linear_correction_upper.units()?>maximum.units()? {maximum=p.linear_correction_upper;}
        }
        if p.cache_reuse {
            // Keep the dense fallback in W and additionally cover each possible
            // old-cache copy, including its host undo journal and bookkeeping.
            for (_,len) in Self::tiles(n,tile) {
                let upper=cache_copy_bound(len)?;
                total=total.checked_add(upper.checked_mul(3)?)?;
                if upper.units()?>maximum.units()? {maximum=upper;}
            }
        }
        let mut transaction_bytes=0u64;
        for step in &p.commands {
            let written=match &step.command {
                Command::Materialize {len,..}|Command::Zip {len,..} => (*len as u64)*32,
                Command::Copy {dst,len,..} => (*len as u64)*buffer_width(backend.buffer_kind(*dst)?),
                Command::Reverse {kind,len,..} => (*len as u64)*buffer_width(*kind),
                Command::Scale {len,..} => (*len as u64)*96,
                Command::Transform {kind,len,..} => (*len as u64)*5*buffer_width(*kind),
                _=>0,
            };
            transaction_bytes=transaction_bytes.checked_add(written)
                .ok_or_else(||anyhow::anyhow!("transaction resource estimate overflow"))?;
        }
        let writable=[witness,input,scratch,coeff[0],coeff[1],coeff[2],f0,f1,product,
            remainder,scalar_fft,g_input,g_scratch,g_output,opening_coeff,opening_eval];
        let mut unique=0u64;
        for id in writable {
            let bytes=(backend.buffer_len(id)? as u64).checked_mul(buffer_width(backend.buffer_kind(id)?))
                .ok_or_else(||anyhow::anyhow!("unique write bound overflow"))?;
            unique=unique.checked_add(bytes).ok_or_else(||anyhow::anyhow!("unique write bound overflow"))?;
        }
        p.resource.transaction_bytes=backend.check_write_capacity(unique,transaction_bytes)?;
        p.budget=RebuildBudget::new(total,maximum,total.checked_mul(2)?,window.horizon)?;
        p.resource.resident_bytes=backend.stats().bytes_allocated;
        ensure!(p.resource.resident_bytes<=upper_bytes,"resident allocation exceeds resource envelope");
        backend.reserve_sparse_msm((2*window.window).max(p.capture_limit.unwrap_or(0)),n)?;
        p.resource.resident_bytes=backend.stats().bytes_allocated;
        ensure!(p.resource.resident_bytes<=upper_bytes,"foreground reservation exceeds resource envelope");
        backend.seal();
        Ok(p)
    }
}

/// Constant snapshot/rollback envelope (eight complete 56-point copies).
/// No scalar multiplication is credited to this G1-only copy alternative.
pub(super) fn capture_copy_bound()->WorkBound {
    WorkBound {field_ops:512,host_elements:512,copy_bytes:56*96*8,..Default::default()}
}

pub(super) fn cache_copy_bound(len:usize)->Result<WorkBound> {
    let count=u64::try_from(len).map_err(|_|anyhow::anyhow!("cache copy length overflow"))?;
    Ok(WorkBound {
        field_ops:256,
        copy_bytes:count.checked_mul(256).ok_or_else(||anyhow::anyhow!("cache copy byte bound overflow"))?,
        host_elements:count.checked_mul(8).and_then(|v|v.checked_add(32))
            .ok_or_else(||anyhow::anyhow!("cache copy host bound overflow"))?,
        ..WorkBound::default()
    })
}

fn buffer_width(kind:BufferKind)->u64 {if kind==BufferKind::Fr {32} else {96}}

fn packet_preflight_bound(len:usize)->Result<WorkBound> {
    let n=u64::try_from(len).map_err(|_|anyhow::anyhow!("packet bound overflow"))?;
    Ok(WorkBound {field_ops:n.checked_mul(1024).and_then(|v|v.checked_add(8192)).ok_or_else(||anyhow::anyhow!("packet field bound overflow"))?,
        copy_bytes:n.checked_mul(96*16).ok_or_else(||anyhow::anyhow!("packet copy bound overflow"))?,
        host_elements:n.checked_mul(32).and_then(|v|v.checked_add(4096)).ok_or_else(||anyhow::anyhow!("packet host bound overflow"))?,
        launches:128,..Default::default()})
}

fn linear_bound(len:usize,kind:BufferKind,launches:u64)->WorkBound {
    let count=len as u64;
    WorkBound {field_ops:count*256,copy_bytes:count*2048,host_elements:count*256,
        launches:launches+16,g1_adds:if kind==BufferKind::G1 {count} else {0},
        ..WorkBound::default()}
}

/// Original per-block arithmetic/transport certificate. Kept intact when
/// several blocks are paid and submitted as one bounded read command.
pub(super) fn cache_check_bound(len:usize,prepared:bool)->Result<WorkBound> {
    let mut upper=linear_bound(len,BufferKind::G1,0);
    upper.field_ops=len as u64*4_000_000;
    upper.g1_muls=len as u64;upper.g2_muls=len as u64;upper.g2_adds=len as u64;
    if prepared {
        upper=upper.checked_add(WorkBound {field_ops:4096*len as u64,
            copy_bytes:2*65_536*len as u64,host_elements:512*len as u64,
            ..Default::default()})?;
    }
    Ok(upper)
}

fn register_points<B:RebuildBackend>(backend:&mut B,points:&[G1Affine],
    registered:&mut std::collections::HashMap<(usize,usize),BufferId>)->Result<BufferId> {
    // Initialization-only identity cache. All source slices remain owned by
    // params (or the live h temporary) until this function has finished.
    let key=(points.as_ptr() as usize,points.len());
    if let Some(&id)=registered.get(&key) {return Ok(id);}
    let id=backend.register_msm_bases(points)?;
    registered.insert(key,id);Ok(id)
}

#[cfg(test)]
mod readback_tests {
    use super::*;
    use crate::incremental_rebuild::operators::CpuRebuildBackend;
    use rand::{rngs::StdRng,SeedableRng};

    #[test]
    fn readback_config_and_memory_reject_before_backend_allocation() {
        let p=DynaverseParams::setup(ToyCircuit::new(16).unwrap(),&mut StdRng::seed_from_u64(26150)).unwrap();
        for cap in [0,1,3,65,usize::MAX] {
            let mut backend=CpuRebuildBackend::default();let before=backend.stats();
            let cfg=IncrementalConfig {rebuild_cache_readback_roots:Some(cap),..Default::default()};
            let error=Program::new(&p,&cfg,&mut backend).err().unwrap();
            assert!(error.to_string().contains("readback roots"));
            assert_eq!(backend.stats(),before);
        }
        let mut backend=CpuRebuildBackend::default();let before=backend.stats();
        let cfg=IncrementalConfig {rebuild_cache_readback_roots:Some(64),
            memory_limit_bytes:16*16_384+(256u64<<20)+16*512+4096-1,..Default::default()};
        let error=Program::new(&p,&cfg,&mut backend).err().unwrap();
        assert!(error.to_string().contains("resource estimate"));assert_eq!(backend.stats(),before);
    }

    #[test]
    fn coalesced_cache_certificate_dominates_all_original_blocks() {
        let p=DynaverseParams::setup(ToyCircuit::new(16).unwrap(),&mut StdRng::seed_from_u64(26151)).unwrap();
        let cfg=IncrementalConfig {rebuild_cache_mode:VerificationMode::Product,
            rebuild_prepared_cache_roots:Some(3),..Default::default()};
        let old=Program::new(&p,&cfg,&mut CpuRebuildBackend::default()).unwrap();
        let new=Program::new(&p,&IncrementalConfig {rebuild_cache_readback_roots:Some(16),..cfg},
            &mut CpuRebuildBackend::default()).unwrap();
        let old_blocks:Vec<_>=old.commands.iter().filter(|p|matches!(p.command,Command::Cache {wire:0,..})).collect();
        let merged=new.commands.iter().find(|p|matches!(p.command,Command::Cache {wire:0,..})).unwrap();
        let mut sum=WorkBound::default();for p in old_blocks {sum=sum.checked_add(p.upper).unwrap();}
        assert!(merged.upper.units().unwrap()>=sum.units().unwrap());
        assert!(merged.upper.field_ops>=sum.field_ops && merged.upper.copy_bytes>=sum.copy_bytes
            && merged.upper.host_elements>=sum.host_elements && merged.upper.launches>=sum.launches
            && merged.upper.g1_muls>=sum.g1_muls && merged.upper.g1_adds>=sum.g1_adds
            && merged.upper.g2_muls>=sum.g2_muls && merged.upper.g2_adds>=sum.g2_adds);
        assert!(new.budget.total>=old.budget.total && new.budget.q>=old.budget.q);
        assert!(new.budget.q>new.budget.max_chunk);
    }
}
