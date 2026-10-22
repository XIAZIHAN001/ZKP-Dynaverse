use super::*;
use crate::{FixedBaseCpuBackend,IcicleBackend,SetupMode};
use ark_ec::CurveGroup;
use ark_ff::{One,UniformRand};
use dynaverse_protocol::OperatorBackend;
use rand::{rngs::StdRng,SeedableRng};
use std::io::Write;
use std::rc::Rc;
#[path="../../tests/support/g2_artifacts.rs"]
mod g2_artifacts;

#[derive(Clone,Copy,PartialEq)]
enum Route {CpuWire,GpuWire,Setup}
struct Case {
    label:String,route:Route,mode:G2ValidationMode,cpu_threads:usize,config:G2KernelConfig,
    gpu:Option<Rc<IcicleBackend>>,resources:Option<G2KernelResources>,
}
impl Case {
    fn new(label:String,route:Route,mode:G2ValidationMode,cpu_threads:usize,config:G2KernelConfig,shared:&Rc<IcicleBackend>)->Self {
        let gpu=if route==Route::CpuWire {None} else {Some(Rc::clone(shared))};
        Self{label,route,mode,cpu_threads,config,gpu,resources:None}
    }
    fn sample(&mut self,n:usize,scalars:&[Fr],bytes:&[u8],expected:&[G2Affine],sample:i32)->String {
        // Cold operator uses a fresh owner; steady cases share one SERIAL owner.
        // Configuration/context creation are outside the operator wall, and Rc
        // shares ownership without allowing concurrent access to this backend.
        let owner=if sample == -1 && self.gpu.is_some() {Some(Rc::new(IcicleBackend::new().unwrap()))} else {self.gpu.clone()};
        if let Some(gpu)=&owner {gpu.set_setup_mode(SetupMode::TableBatchNormalize).unwrap();
            gpu.set_g2_validation_mode(self.mode).unwrap();gpu.set_g2_kernel_config(self.config).unwrap();}
        let before=owner.as_ref().map(|gpu|(gpu.g2_validation_stats().unwrap(),gpu.g2_experiment_stats().unwrap(),
            gpu.g2_validation_host_stats(),gpu.g2_validation_traffic().unwrap(),gpu.setup_stats().unwrap()));
        let start=Instant::now();let wall;
        match self.route {
            Route::CpuWire=>{
                let masks=inspect_g2_cpu(bytes,self.mode,self.cpu_threads).unwrap();wall=start.elapsed().as_nanos();
                assert!(masks.iter().all(|&m|m==0));
            },
            Route::GpuWire=>{
                let (masks,report)=owner.as_ref().unwrap().inspect_g2_validation_batched(bytes,self.mode.gpu_algorithm().unwrap()).unwrap();
                wall=start.elapsed().as_nanos();assert_eq!(report.invalid_points,0);assert!(masks.iter().all(|&m|m==0));
            },
            Route::Setup=>{
                let actual=owner.as_ref().unwrap().fixed_base_g2(scalars).unwrap();wall=start.elapsed().as_nanos();
                assert_eq!(actual,expected);
            },
        }
        if self.resources.is_none() {if let Some(algorithm)=self.mode.gpu_algorithm() {
            self.resources=Some(owner.as_ref().unwrap().g2_kernel_resources(algorithm).unwrap());
        }}
        let scope=if self.route==Route::Setup {"integrated"} else {"standalone"};
        let mut row=vec![self.label.clone(),scope.into(),n.to_string(),sample.to_string(),self.cpu_threads.to_string(),
            self.config.threads.to_string(),self.config.schedule.to_string(),self.config.layout.to_string(),self.config.fused.to_string(),wall.to_string()];
        if let Some(gpu)=&owner {
            let (a,e,h,t,setup)=before.unwrap();let b=gpu.g2_validation_stats().unwrap();let f=gpu.g2_experiment_stats().unwrap();
            let host=gpu.g2_validation_host_stats();let traffic=gpu.g2_validation_traffic().unwrap();
            let native=if self.route==Route::Setup {gpu.setup_stats().unwrap().native_wall_ns-setup.native_wall_ns} else {b.native_wall_ns-a.native_wall_ns};
            for value in [b.parse_curve_device_ns-a.parse_curve_device_ns,b.subgroup_device_ns-a.subgroup_device_ns,
                f.repack_device_ns-e.repack_device_ns,f.subgroup_kernel_ns-e.subgroup_kernel_ns,b.reduce_device_ns-a.reduce_device_ns,
                b.h2d_device_ns-a.h2d_device_ns,b.d2h_device_ns-a.d2h_device_ns,native,
                host.host_parse_ns-h.host_parse_ns,host.host_curve_ns-h.host_curve_ns,host.host_subgroup_ns-h.host_subgroup_ns,
                host.host_publish_decode_ns-h.host_publish_decode_ns,traffic.scalar_h2d_bytes-t.scalar_h2d_bytes,
                traffic.point_h2d_bytes-t.point_h2d_bytes,traffic.point_d2h_bytes-t.point_d2h_bytes,
                traffic.status_d2h_bytes-t.status_d2h_bytes,f.logical_repack_bytes-e.logical_repack_bytes] {row.push(value.to_string());}
        } else {row.extend(vec![String::new();17]);}
        if let Some(r)=self.resources {
            for value in [u64::from(r.registers),r.local_bytes,r.shared_bytes,u64::from(r.active_blocks),u64::from(r.max_threads_per_sm)] {row.push(value.to_string());}
            if self.route==Route::Setup {
                for value in [u64::from(r.normalize_registers),r.normalize_local_bytes,r.normalize_shared_bytes,u64::from(r.normalize_active_blocks)] {row.push(value.to_string());}
            } else {row.extend(vec![String::new();4]);}
        } else {row.extend(vec![String::new();9]);}
        row.join(",")+"\n"
    }
}

#[test]
#[ignore="G2-5 full matrix, CPU strong baselines may take tens of minutes"]
fn g2_validation_profile() {
    let dir=g2_artifacts::directory().unwrap();
    let mut rng=StdRng::seed_from_u64(1750001);let mut scalars=Vec::with_capacity(65536);
    for _ in 0..65536 {scalars.push(Fr::rand(&mut rng));}scalars[0]=Fr::zero();scalars[1]=Fr::one();scalars[2]=-Fr::one();
    let expected=FixedBaseCpuBackend::default().fixed_base_g2(&scalars).unwrap();
    for i in [0,1,2,65535] {assert_eq!(expected[i],(G2Projective::generator()*scalars[i]).into_affine());}
    let mut bytes=Vec::with_capacity(65536*128);for point in &expected {bytes.extend_from_slice(&super::tests::wire(*point));}
    std::fs::write(dir.join("G2-5-public-corpus.bin"),&bytes).unwrap();
    let mut output=std::io::BufWriter::new(std::fs::File::create(dir.join("G2-5-profile.csv")).unwrap());
    writeln!(output,"case,scope,points,sample,cpu_threads,block_threads,schedule,layout,fused,wall_ns,parse_curve_ns,subgroup_stage_ns,repack_ns,subgroup_kernel_event_ns,reduce_ns,h2d_ns,d2h_ns,native_wall_ns,host_parse_ns,host_curve_ns,host_subgroup_ns,host_publish_ns,scalar_h2d_bytes,point_h2d_bytes,point_d2h_bytes,status_d2h_bytes,logical_repack_bytes,registers,local_bytes,shared_bytes,active_blocks,max_threads_per_sm,normalize_registers,normalize_local_bytes,normalize_shared_bytes,normalize_active_blocks").unwrap();
    let modes=[G2ValidationMode::GpuFullOrder,G2ValidationMode::GpuFrobenius127,G2ValidationMode::GpuFrobenius63];
    for (size_index,n) in [128,1024,4096,16384,65536].into_iter().enumerate() {
        let mut cases=Vec::new();let base=G2KernelConfig::default();let shared=Rc::new(IcicleBackend::new().unwrap());
        for (index,mode) in [G2ValidationMode::CpuReference,G2ValidationMode::CpuFrobenius127,G2ValidationMode::CpuFrobenius63].into_iter().enumerate() {
            for threads in [1,8] {cases.push(Case::new(format!("cpu{index}_t{threads}"),Route::CpuWire,mode,threads,base,&shared));}
            cases.push(Case::new(format!("cpu{index}_production_t1"),Route::Setup,mode,1,base,&shared));
        }
        for (index,mode) in modes.into_iter().enumerate() {
            let mut configs=Vec::new();
            let max_schedule=if index==2 {1} else {0};
            for threads in [64,128,256] {for schedule in 0..=max_schedule {for layout in [0,1] {for fused in [0,1] {
                configs.push(G2KernelConfig{threads,schedule,layout,fused});
            }}}}
            for config in configs {
                let name=format!("gpu{}_b{}_s{}_l{}_f{}",index+1,config.threads,config.schedule,config.layout,config.fused);
                if config.fused==0 {cases.push(Case::new(name.clone(),Route::GpuWire,mode,0,config,&shared));}
                cases.push(Case::new(name,Route::Setup,mode,0,config,&shared));
            }
        }
        for case in &mut cases {
            let row=case.sample(n,&scalars[..n],&bytes[..n*128],&expected[..n],-1);output.write_all(row.as_bytes()).unwrap();output.flush().unwrap();
        }
        for round in 0..3 {let count=cases.len();for k in 0..count {
            let index=(k+round+size_index)%count;cases[index].sample(n,&scalars[..n],&bytes[..n*128],&expected[..n],-2);
        }}
        for round in 0..20 {
            let count=cases.len();for k in 0..count {
                let index=(k+round*7+size_index)%count;
                let row=cases[index].sample(n,&scalars[..n],&bytes[..n*128],&expected[..n],round as i32);
                output.write_all(row.as_bytes()).unwrap();output.flush().unwrap();
            }
            eprintln!("G2_PROFILE_PROGRESS points={n} sample={round} cases={count}");
        }
    }
}

// Separate from the timing matrix: profiler replay/counter collection changes
// execution costs, so these measurements must never replace operator wall data.
#[test]
#[ignore="explicit Nsight kernel counter probe, not a latency benchmark"]
fn g2_nsight_resource_probe() {
    let dir=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/g2-validation-20261017");
    let bytes=std::fs::read(dir.join("public-corpus.bin")).unwrap();
    let gpu=IcicleBackend::new().unwrap();
    for schedule in [0,1] {
        gpu.set_g2_kernel_config(G2KernelConfig{threads:128,schedule,layout:0,fused:0}).unwrap();
        let (_,report)=gpu.inspect_g2_validation(&bytes[..4096*128],3).unwrap();
        assert_eq!(report.invalid_points,0);
        eprintln!("G2_NSIGHT_RESOURCES schedule={schedule} {:?}",gpu.g2_kernel_resources(3).unwrap());
    }
}
