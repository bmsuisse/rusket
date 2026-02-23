use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_als(c: &mut Criterion) {
}

criterion_group!(benches, bench_als);
criterion_main!(benches);
