use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_als(c: &mut Criterion) {
    // Basic setup to call ALS, mimicking PyO3 calls
}

criterion_group!(benches, bench_als);
criterion_main!(benches);
