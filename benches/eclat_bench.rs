use criterion::{black_box, criterion_group, criterion_main, Criterion};

// A simple stand-in for bitsets to test
fn bench_declat(c: &mut Criterion) {
}

criterion_group!(benches, bench_declat);
criterion_main!(benches);
