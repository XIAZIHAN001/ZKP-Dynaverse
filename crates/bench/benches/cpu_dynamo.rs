use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ark_bn254::Fr;
use dynaverse_gpu::CpuBackend;
use dynaverse_protocol::dynamo::{
    derive_h_sparse, dynamo_prove_sparse, DynamoParams, DynamoPermutation, SparseVector,
};
use rand::{rngs::StdRng, SeedableRng};

fn bench_dynamo_sparse_prove(c: &mut Criterion) {
    let m = 256usize;
    let mut setup_rng = StdRng::seed_from_u64(7);
    let sigma = DynamoPermutation::try_new((0..m).rev().collect()).unwrap();
    let params = DynamoParams::setup(m, sigma.clone(), &mut setup_rng).unwrap();
    let backend = CpuBackend;
    let mut group = c.benchmark_group("cpu_dynamo_sparse_prove_m256");
    for k in [1usize, 16, 256] {
        let entries = (0..k)
            .map(|i| (i, Fr::from((i + 1) as u64)))
            .collect();
        let z = SparseVector::try_new(m, entries).unwrap();
        let h = derive_h_sparse(&z, &sigma).unwrap();
        let mut rng = StdRng::seed_from_u64(17 + k as u64);
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, _| {
            b.iter(|| dynamo_prove_sparse(&params, &z, &h, &backend, &mut rng).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dynamo_sparse_prove);
criterion_main!(benches);
