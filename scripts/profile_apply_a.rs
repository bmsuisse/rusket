use std::time::Instant;

macro_rules! dot {
    ($a:expr, $b:expr) => {{
        let a: &[f32] = &$a;
        let b: &[f32] = &$b;
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }};
}

macro_rules! axpy {
    ($alpha:expr, $x:expr, $y:expr) => {{
        let alpha: f32 = $alpha;
        let x: &[f32] = &$x;
        let y: &mut [f32] = &mut *$y;
        y.iter_mut().zip(x.iter()).for_each(|(yi, &xi)| {
            *yi += alpha * xi;
        });
    }};
}

// SIMD optimized axpy with explicit chunks
#[inline(always)]
fn axpy_unrolled(alpha: f32, x: &[f32], y: &mut [f32]) {
    let mut i = 0;
    while i + 8 <= x.len() {
        y[i] += alpha * x[i];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
        y[i + 4] += alpha * x[i + 4];
        y[i + 5] += alpha * x[i + 5];
        y[i + 6] += alpha * x[i + 6];
        y[i + 7] += alpha * x[i + 7];
        i += 8;
    }
    while i < x.len() {
        y[i] += alpha * x[i];
        i += 1;
    }
}

// SIMD optimized dot with explicit chunks
#[inline(always)]
fn dot_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;
    while i + 8 <= a.len() {
        sum += a[i] * b[i] 
             + a[i + 1] * b[i + 1] 
             + a[i + 2] * b[i + 2] 
             + a[i + 3] * b[i + 3] 
             + a[i + 4] * b[i + 4] 
             + a[i + 5] * b[i + 5] 
             + a[i + 6] * b[i + 6] 
             + a[i + 7] * b[i + 7];
        i += 8;
    }
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

fn test_apply_a(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    other: &[f32],
    gram: &[f32],
    n: usize,
    k: usize,
    lambda: f32,
    alpha: f32,
) {
    let mut p = vec![0.5f32; k];
    let mut ap = vec![0.0f32; k];
    
    // Test base
    let s0 = Instant::now();
    for _ in 0..10 {
        for u in 0..n {
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            
            for a in 0..k {
                let mut s = 0.0f32;
                for bb in 0..k {
                    s += gram[a * k + bb] * p[bb];
                }
                ap[a] = s + lambda * p[a];
            }
            for idx in start..end {
                let i = indices[idx] as usize;
                let c = 1.0 + alpha * data[idx];
                let yi = &other[i * k..(i + 1) * k];
                let w = (c - 1.0) * dot!(yi, &p);
                axpy!(w, yi, &mut ap);
            }
        }
    }
    println!("Base Apply A (10 iters): {:.3}s", s0.elapsed().as_secs_f64());

    // Test unrolled
    let s1 = Instant::now();
    for _ in 0..10 {
        for u in 0..n {
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            
            for a in 0..k {
                ap[a] = dot_unrolled(&gram[a * k .. (a+1) * k], &p) + lambda * p[a];
            }
            for idx in start..end {
                let i = indices[idx] as usize;
                let c = 1.0 + alpha * data[idx];
                let yi = &other[i * k..(i + 1) * k];
                let w = (c - 1.0) * dot_unrolled(yi, &p);
                axpy_unrolled(w, yi, &mut ap);
            }
        }
    }
    println!("Unrolled Apply A (10 iters): {:.3}s", s1.elapsed().as_secs_f64());
}

fn main() {
    let n_users = 10_000;
    let n_items = 10_000;
    let k = 64;
    let nnz_per_user = 100;

    let mut indptr = vec![0i64; n_users + 1];
    let mut indices = Vec::with_capacity(n_users * nnz_per_user);
    let mut data = Vec::with_capacity(n_users * nnz_per_user);

    let mut pos = 0;
    for u in 0..n_users {
        indptr[u] = pos;
        for j in 0..nnz_per_user {
            indices.push(((u + j * 97) % n_items) as i32);
            data.push(1.0);
            pos += 1;
        }
    }
    indptr[n_users] = pos;
    
    let mut other = vec![0.5f32; n_items * k];
    let mut gram = vec![0.2f32; k * k];

    test_apply_a(&indptr, &indices, &data, &other, &gram, n_users, k, 0.01, 40.0);
}
