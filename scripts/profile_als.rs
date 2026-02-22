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

fn gramian_basic(factors: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; k * k];
    for col in 0..k {
        for row in 0..k {
            let mut sum = 0.0;
            for i in 0..n {
                sum += factors[i * k + row] * factors[i * k + col];
            }
            g[row * k + col] = sum;
        }
    }
    g
}

fn solve_one_side_cg(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    other: &[f32],
    gram: &[f32],
    n: usize,
    k: usize,
    lambda: f32,
    alpha: f32,
    cg_iters: usize,
) -> (Vec<f32>, f64, f64, f64) {
    let eff_lambda = lambda.max(1e-6);
    let mut out = vec![0.0f32; n * k];

    let mut t_setup = 0.0;
    let mut t_apply_a = 0.0;
    let mut t_cg_update = 0.0;

    for u in 0..n {
        let xu = &mut out[u * k..(u + 1) * k];
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;

        let s0 = Instant::now();
        let mut b = vec![0.0; k];
        for idx in start..end {
            let i = indices[idx] as usize;
            let c = 1.0 + alpha * data[idx];
            let yi = &other[i * k..(i + 1) * k];
            axpy!(c, yi, &mut b);
        }
        
        xu.fill(0.0);
        let mut r = b.clone();
        let mut p = b.clone();
        let mut ap = vec![0.0; k];
        let mut rsold = dot!(r, r);
        t_setup += s0.elapsed().as_secs_f64();

        if rsold < 1e-20 {
            continue;
        }

        for _ in 0..cg_iters {
            let s1 = Instant::now();
            // Apply A
            for a in 0..k {
                let mut s = 0.0f32;
                for bb in 0..k {
                    s += gram[a * k + bb] * p[bb];
                }
                ap[a] = s + eff_lambda * p[a];
            }
            for idx in start..end {
                let i = indices[idx] as usize;
                let c = 1.0 + alpha * data[idx];
                let yi = &other[i * k..(i + 1) * k];
                let w = (c - 1.0) * dot!(yi, &p);
                axpy!(w, yi, &mut ap);
            }
            t_apply_a += s1.elapsed().as_secs_f64();

            let s2 = Instant::now();
            let pap = dot!(&p, &ap);
            if pap <= 0.0 {
                t_cg_update += s2.elapsed().as_secs_f64();
                break;
            }
            let ak = rsold / pap;

            axpy!(ak, &p, xu);
            axpy!(-ak, &ap, &mut r);

            let rsnew = dot!(&r, &r);
            if rsnew < 1e-20 {
                t_cg_update += s2.elapsed().as_secs_f64();
                break;
            }
            let beta = rsnew / rsold;
            for j in 0..k {
                p[j] = r[j] + beta * p[j];
            }
            rsold = rsnew;
            t_cg_update += s2.elapsed().as_secs_f64();
        }
    }

    (out, t_setup, t_apply_a, t_cg_update)
}

fn random_factors(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut out = vec![0.0f32; n * k];
    let mut s = seed;
    let scale = 1.0 / (k as f32).sqrt();
    for v in out.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = ((s & 0xFFFF) as f32) / (0xFFFF as f32) * scale;
    }
    out
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

    let mut user_factors = random_factors(n_users, k, 42);
    let item_factors = random_factors(n_items, k, 43);

    let mut total_setup = 0.0;
    let mut total_apply_a = 0.0;
    let mut total_cg_update = 0.0;

    let g_item = gramian_basic(&item_factors, n_items, k);
    let s0 = Instant::now();
    for _ in 0..3 {
        let (uf, setup, apply, update) = solve_one_side_cg(&indptr, &indices, &data, &item_factors, &g_item, n_users, k, 0.01, 40.0, 5);
        user_factors = uf;
        total_setup += setup;
        total_apply_a += apply;
        total_cg_update += update;
    }
    println!("Total solve_one_side_cg time: {:.3}s", s0.elapsed().as_secs_f64());
    println!("  -> Setup (RHS b generation): {:.3}s ({:.1}%)", total_setup, total_setup / s0.elapsed().as_secs_f64() * 100.0);
    println!("  -> Apply A (Matrix-vector mult): {:.3}s ({:.1}%)", total_apply_a, total_apply_a / s0.elapsed().as_secs_f64() * 100.0);
    println!("  -> CG Vec Update (axpy): {:.3}s ({:.1}%)", total_cg_update, total_cg_update / s0.elapsed().as_secs_f64() * 100.0);
}
