use faer::{linalg::matmul::matmul, Accum, MatMut, MatRef, Par};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;

// ── SIMD‑friendly primitives ───────────────────────────────────────
// 8‑wide manual unroll – LLVM maps these to NEON/AVX without needing
// architecture‑specific intrinsics, and it's faster than the plain
// iterator chain because we guarantee no loop‑carried dependency.
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let n8 = n / 8 * 8;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = 0;
    while i < n8 {
        unsafe {
            s0 += *a.get_unchecked(i)     * *b.get_unchecked(i);
            s1 += *a.get_unchecked(i + 1) * *b.get_unchecked(i + 1);
            s2 += *a.get_unchecked(i + 2) * *b.get_unchecked(i + 2);
            s3 += *a.get_unchecked(i + 3) * *b.get_unchecked(i + 3);
            s4 += *a.get_unchecked(i + 4) * *b.get_unchecked(i + 4);
            s5 += *a.get_unchecked(i + 5) * *b.get_unchecked(i + 5);
            s6 += *a.get_unchecked(i + 6) * *b.get_unchecked(i + 6);
            s7 += *a.get_unchecked(i + 7) * *b.get_unchecked(i + 7);
        }
        i += 8;
    }
    while i < n {
        unsafe { s0 += *a.get_unchecked(i) * *b.get_unchecked(i); }
        i += 1;
    }
    (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}

#[inline(always)]
fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let n8 = n / 8 * 8;
    let mut i = 0;
    while i < n8 {
        unsafe {
            *y.get_unchecked_mut(i)     += alpha * *x.get_unchecked(i);
            *y.get_unchecked_mut(i + 1) += alpha * *x.get_unchecked(i + 1);
            *y.get_unchecked_mut(i + 2) += alpha * *x.get_unchecked(i + 2);
            *y.get_unchecked_mut(i + 3) += alpha * *x.get_unchecked(i + 3);
            *y.get_unchecked_mut(i + 4) += alpha * *x.get_unchecked(i + 4);
            *y.get_unchecked_mut(i + 5) += alpha * *x.get_unchecked(i + 5);
            *y.get_unchecked_mut(i + 6) += alpha * *x.get_unchecked(i + 6);
            *y.get_unchecked_mut(i + 7) += alpha * *x.get_unchecked(i + 7);
        }
        i += 8;
    }
    while i < n {
        unsafe { *y.get_unchecked_mut(i) += alpha * *x.get_unchecked(i); }
        i += 1;
    }
}

fn gramian(factors: &[f32], n: usize, k: usize) -> Vec<f32> {
    let y = MatRef::from_row_major_slice(factors, n, k);
    let yt = y.transpose();

    let mut g = faer::Mat::<f32>::zeros(k, k);
    matmul(g.as_mut(), Accum::Replace, yt, y, 1.0f32, Par::rayon(0));

    let mut r = vec![0.0f32; k * k];
    for a in 0..k {
        for b in 0..k {
            r[a * k + b] = g[(a, b)];
        }
    }
    r
}

thread_local! {
    // CG scratch: b, r, p, ap, yi_dense (item matrix), w_vec (weights), tmp
    static SCRATCH: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>,
                             Vec<f32>, Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new(),
                              Vec::new(), Vec::new(), Vec::new())) };
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
) -> Vec<f32> {
    let eff_lambda = lambda.max(1e-6);
    let mut out = vec![0.0f32; n * k];

    out.par_chunks_mut(k).enumerate().for_each(|(u, xu)| {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let nnz_u = end - start;

        SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut b, ref mut r, ref mut p, ref mut ap,
                 ref mut yi_dense, ref mut w_vec, ref mut tmp) = *borrow;
            b.clear(); b.resize(k, 0.0);
            r.clear(); r.resize(k, 0.0);
            p.clear(); p.resize(k, 0.0);
            ap.clear(); ap.resize(k, 0.0);

            // Pre-collect interacted item vectors + weights
            yi_dense.clear();
            yi_dense.resize(nnz_u * k, 0.0);
            w_vec.clear();
            w_vec.resize(nnz_u, 0.0);
            tmp.clear();
            tmp.resize(nnz_u, 0.0);

            for (local, idx) in (start..end).enumerate() {
                let i = indices[idx] as usize;
                let c = 1.0 + alpha * data[idx];
                let yi = &other[i * k..(i + 1) * k];
                axpy_f32(c, yi, b);

                // Copy yi into dense matrix and store weight
                yi_dense[local * k..(local + 1) * k].copy_from_slice(yi);
                w_vec[local] = alpha * data[idx]; // = c - 1
            }

            // apply_a: computes out = (Gram + lambda*I + Y^T diag(w) Y) * v
            // using batch BLAS for the Y^T diag(w) Y * v part
            let mut apply_a = |v: &[f32], out: &mut [f32]| {
                // Part 1: Gram × v (8-wide unrolled)
                let k8 = k / 8 * 8;
                for a in 0..k {
                    let gram_row = &gram[a * k..];
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    let mut s4 = 0.0f32;
                    let mut s5 = 0.0f32;
                    let mut s6 = 0.0f32;
                    let mut s7 = 0.0f32;
                    let mut bb = 0;
                    while bb < k8 {
                        unsafe {
                            s0 += *gram_row.get_unchecked(bb)   * *v.get_unchecked(bb);
                            s1 += *gram_row.get_unchecked(bb+1) * *v.get_unchecked(bb+1);
                            s2 += *gram_row.get_unchecked(bb+2) * *v.get_unchecked(bb+2);
                            s3 += *gram_row.get_unchecked(bb+3) * *v.get_unchecked(bb+3);
                            s4 += *gram_row.get_unchecked(bb+4) * *v.get_unchecked(bb+4);
                            s5 += *gram_row.get_unchecked(bb+5) * *v.get_unchecked(bb+5);
                            s6 += *gram_row.get_unchecked(bb+6) * *v.get_unchecked(bb+6);
                            s7 += *gram_row.get_unchecked(bb+7) * *v.get_unchecked(bb+7);
                        }
                        bb += 8;
                    }
                    while bb < k {
                        unsafe { s0 += *gram_row.get_unchecked(bb) * *v.get_unchecked(bb); }
                        bb += 1;
                    }
                    out[a] = (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7) + eff_lambda * v[a];
                }

                // Part 2: Y^T diag(w) Y * v  via two BLAS passes
                // tmp[i] = w[i] * dot(yi, v) — scalar pass
                if nnz_u > 0 {
                    for i in 0..nnz_u {
                        let yi = &yi_dense[i * k..(i + 1) * k];
                        tmp[i] = w_vec[i] * dot_f32(yi, v);
                    }
                    // out += Y^T * tmp  — single gemv (k × nnz_u) × (nnz_u × 1)
                    let y_mat = MatRef::from_row_major_slice(
                        &yi_dense[..nnz_u * k], nnz_u, k,
                    );
                    let t_mat = MatRef::from_column_major_slice(
                        &tmp[..nnz_u], nnz_u, 1,
                    );
                    let mut o_mat = MatMut::from_column_major_slice_mut(out, k, 1);
                    matmul(o_mat.as_mut(), Accum::Add, y_mat.transpose(), t_mat, 1.0f32, Par::Seq);
                }
            };

            xu.fill(0.0);
            r.copy_from_slice(b);
            p.copy_from_slice(b);
            let mut rsold = dot_f32(r, r);

            if rsold < 1e-20 {
                return;
            }

            for _ in 0..cg_iters {
                apply_a(p, ap);
                let pap = dot_f32(p, ap);
                if pap <= 0.0 {
                    break;
                }
                let ak = rsold / pap;

                axpy_f32(ak, p, xu);
                axpy_f32(-ak, ap, r);

                let rsnew = dot_f32(r, r);
                if rsnew < 1e-20 {
                    break;
                }
                let beta = rsnew / rsold;
                for j in 0..k {
                    p[j] = r[j] + beta * p[j];
                }
                rsold = rsnew;
            }
        });
    });

    out
}

thread_local! {
    static SCRATCH_EALS: RefCell<(Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new())) };
}

fn solve_one_side_eals(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    other: &[f32],
    gram: &[f32],
    out: &mut [f32],
    k: usize,
    lambda: f32,
    alpha: f32,
    eals_iters: usize,
) {
    let eff_lambda = lambda.max(1e-6);

    out.par_chunks_mut(k).enumerate().for_each(|(u, xu)| {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let nnz_u = end - start;

        if nnz_u == 0 {
            xu.fill(0.0);
            return;
        }

        SCRATCH_EALS.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut r_hat, ref mut s_u) = *borrow;

            r_hat.clear();
            r_hat.resize(nnz_u, 0.0);
            s_u.clear();
            s_u.resize(k, 0.0);

            // r_hat = y_i * xu (prediction for interacted items)
            for (local, idx) in (start..end).enumerate() {
                let i = indices[idx] as usize;
                let yi = &other[i * k..(i + 1) * k];
                r_hat[local] = dot_f32(xu, yi);
            }

            for _pass in 0..eals_iters {
                // Precompute s_u = Gram * xu
                // Use the SIMD dot_f32 for fast matrix-vector multiply
                for f in 0..k {
                    s_u[f] = dot_f32(&gram[f * k..(f + 1) * k], xu);
                }

                for f in 0..k {
                    let mut numer = -(s_u[f] - xu[f] * gram[f * k + f]);
                    let mut denom = eff_lambda + gram[f * k + f];

                    // Process all interacted items for user u
                    for (local, idx) in (start..end).enumerate() {
                        let i = indices[idx] as usize;
                        let w = alpha * data[idx];
                        let y_if = other[i * k + f];
                        let r_hat_minus_f = r_hat[local] - xu[f] * y_if;
                        
                        numer += (w + 1.0) * y_if - w * r_hat_minus_f * y_if;
                        denom += w * y_if * y_if;
                    }

                    let new_u_f = numer / denom;
                    let diff = new_u_f - xu[f];

                    if diff.abs() > 1e-9 {
                        // Update r_hat and s_u using fast unrolled loops
                        for (local, idx) in (start..end).enumerate() {
                            let i = indices[idx] as usize;
                            let y_if = other[i * k + f];
                            r_hat[local] += diff * y_if;
                        }
                        // Use AXPY for updating the s_u vector efficiently
                        let gram_row = &gram[f * k..(f + 1) * k];
                        axpy_f32(diff, gram_row, s_u);
                        
                        xu[f] = new_u_f;
                    }
                }
            }
        });
    });
}

use faer::linalg::solvers::Solve;

fn cholesky_solve_inplace(a: &mut [f32], b: &mut [f32], k: usize) {
    let a_mat = faer::MatMut::from_row_major_slice_mut(a, k, k);
    let mut b_mat = faer::MatMut::from_column_major_slice_mut(b, k, 1);

    if let Ok(llt) = a_mat.as_ref().llt(faer::Side::Lower) {
        llt.solve_in_place(b_mat.as_mut());
    }
}

thread_local! {
    static SCRATCH_CHOL: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

fn solve_one_side_cholesky(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    other: &[f32],
    gram: &[f32],
    n: usize,
    k: usize,
    lambda: f32,
    alpha: f32,
) -> Vec<f32> {
    let eff_lambda = lambda.max(1e-6);
    let mut out = vec![0.0f32; n * k];

    out.par_chunks_mut(k).enumerate().for_each(|(u, xu)| {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let nnz_u = end - start;

        if nnz_u == 0 {
            return;
        }

        let (mut a_buf, mut b_buf, mut yi_buf, mut w_buf) = SCRATCH_CHOL.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut a, ref mut b, ref mut yi, ref mut w) = *borrow;
            // take the vecs out of the RefCell temporarily so we don't hold the borrow
            (
                std::mem::take(a),
                std::mem::take(b),
                std::mem::take(yi),
                std::mem::take(w),
            )
        });

        // --- Build A = Gram + lambda*I + sum_i w_i * y_i * y_i^T ---
        a_buf.clear();
        a_buf.extend_from_slice(gram);
        b_buf.clear();
        b_buf.resize(k, 0.0);

        for j in 0..k {
            a_buf[j * k + j] += eff_lambda;
        }

        // Collect item vectors and weights for batch rank-1 update
        yi_buf.clear();
        yi_buf.resize(nnz_u * k, 0.0);
        w_buf.clear();
        w_buf.resize(nnz_u, 0.0);

        for (local, idx) in (start..end).enumerate() {
            let i = indices[idx] as usize;
            let ci = 1.0 + alpha * data[idx];
            let yi = &other[i * k..(i + 1) * k];

            // b += ci * yi
            axpy_f32(ci, yi, &mut b_buf);

            // Store sqrt(w) * yi for batch syrk
            let w = ci - 1.0; // = alpha * data[idx]
            if w > 0.0 {
                let sw = w.sqrt();
                let dest = &mut yi_buf[local * k..(local + 1) * k];
                for f in 0..k {
                    dest[f] = sw * yi[f];
                }
                w_buf[local] = 1.0; // marker: has weight
            } else {
                w_buf[local] = 0.0;
            }
        }

        if b_buf.iter().all(|&v| v == 0.0) {
            // Put vecs back
            SCRATCH_CHOL.with(|cell| {
                *cell.borrow_mut() = (a_buf, b_buf, yi_buf, w_buf);
            });
            return;
        }

        // Batch rank-1 updates via syrk:  A += W^T * W  where W rows = sqrt(w_i)*y_i
        // This is a single BLAS call instead of nnz_u individual rank-1 updates.
        // Build W matrix with only non-zero weight rows
        let w_mat = MatRef::from_row_major_slice(&yi_buf[..nnz_u * k], nnz_u, k);
        let w_mat_t = w_mat.transpose();
        let mut a_mat = faer::MatMut::from_row_major_slice_mut(&mut a_buf, k, k);
        matmul(a_mat.as_mut(), Accum::Add, w_mat_t, w_mat, 1.0f32, Par::Seq);

        cholesky_solve_inplace(&mut a_buf, &mut b_buf, k);
        xu.copy_from_slice(&b_buf);

        // Put vecs back into TLS
        SCRATCH_CHOL.with(|cell| {
            *cell.borrow_mut() = (a_buf, b_buf, yi_buf, w_buf);
        });
    });

    out
}


fn csr_transpose(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let nnz = indices.len();
    let mut cc = vec![0i64; n_cols];
    for &c in indices {
        cc[c as usize] += 1;
    }
    let mut ti = vec![0i64; n_cols + 1];
    for i in 0..n_cols {
        ti[i + 1] = ti[i] + cc[i];
    }
    let mut tv = vec![0i32; nnz];
    let mut td = vec![0.0f32; nnz];
    let mut pos = ti[..n_cols].to_vec();
    for row in 0..n_rows {
        let s = indptr[row] as usize;
        let e = indptr[row + 1] as usize;
        for idx in s..e {
            let col = indices[idx] as usize;
            let p = pos[col] as usize;
            tv[p] = row as i32;
            td[p] = data[idx];
            pos[col] += 1;
        }
    }
    (ti, tv, td)
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

struct AndersonAccel {
    m: usize,
    x_hist: std::collections::VecDeque<Vec<f32>>,
    f_hist: std::collections::VecDeque<Vec<f32>>,
}

impl AndersonAccel {
    fn new(m: usize) -> Self {
        Self {
            m,
            x_hist: std::collections::VecDeque::new(),
            f_hist: std::collections::VecDeque::new(),
        }
    }

    /// Push x_old (before the ALS step) and f = x_new - x_old.
    /// Returns the Anderson-mixed iterate (replacing x_new in-place).
    fn push_and_mix(&mut self, x_old: Vec<f32>, f: Vec<f32>) -> Vec<f32> {
        let dim = x_old.len();

        if self.x_hist.len() == self.m {
            self.x_hist.pop_front();
            self.f_hist.pop_front();
        }
        self.x_hist.push_back(x_old);
        self.f_hist.push_back(f.clone());

        let h = self.f_hist.len();

        if h == 1 {
            let mut out = self.x_hist[0].clone();
            for (o, &fi) in out.iter_mut().zip(&f) {
                *o += fi;
            }
            return out;
        }

        let mut g = vec![0.0f32; h * h];
        for i in 0..h {
            for j in i..h {
                let d: f32 = self.f_hist[i]
                    .iter()
                    .zip(&self.f_hist[j])
                    .map(|(a, b)| a * b)
                    .sum();
                g[i * h + j] = d;
                g[j * h + i] = d;
            }
        }

        let n = h + 1;
        let mut mat = vec![0.0f32; n * n];
        for i in 0..h {
            for j in 0..h {
                mat[i * n + j] = g[i * h + j];
            }
            mat[i * n + h] = 1.0;
            mat[h * n + i] = 1.0;
        }

        let mut rhs = vec![0.0f32; n];
        rhs[h] = 1.0;

        if !gauss_solve_inplace(&mut mat, &mut rhs, n) {
            let mut out = self.x_hist[h - 1].clone();
            for (o, &fi) in out.iter_mut().zip(&self.f_hist[h - 1]) {
                *o += fi;
            }
            return out;
        }

        let mut out = vec![0.0f32; dim];
        for i in 0..h {
            let theta = rhs[i];
            for d in 0..dim {
                out[d] += theta * (self.x_hist[i][d] + self.f_hist[i][d]);
            }
        }
        out
    }
}

fn gauss_solve_inplace(mat: &mut Vec<f32>, rhs: &mut Vec<f32>, n: usize) -> bool {
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return false;
        }
        if max_row != col {
            for j in 0..n {
                mat.swap(col * n + j, max_row * n + j);
            }
            rhs.swap(col, max_row);
        }
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for j in col..n {
                let v = factor * mat[col * n + j];
                mat[row * n + j] -= v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for col in (0..n).rev() {
        rhs[col] /= mat[col * n + col];
        for row in 0..col {
            let v = mat[row * n + col] * rhs[col];
            rhs[row] -= v;
        }
    }
    true
}

fn als_train(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    indptr_t: &[i64],
    indices_t: &[i32],
    data_t: &[f32],
    n_users: usize,
    n_items: usize,
    k: usize,
    lambda: f32,
    alpha: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
    cg_iters: usize,
    use_cholesky: bool,
    anderson_m: usize,
    use_eals: bool,
    eals_iters: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut user_factors = random_factors(n_users, k, seed);
    let mut item_factors = random_factors(n_items, k, seed.wrapping_add(1));

    if verbose {
        let solver_name = if use_eals {
            format!("eALS(iters={})", eals_iters)
        } else if use_cholesky {
            "Cholesky".to_string()
        } else {
            format!("CG(iters={})", cg_iters)
        };
        println!("  Solver: {}  factors={}", solver_name, k);
        println!("  ITER | USER FACTORS | ITEM FACTORS | TOTAL TIME ");
        println!("  ------------------------------------------------");
    }

    let mut total_time = std::time::Duration::new(0, 0);
    let use_aa = anderson_m > 0;
    let mut accel = AndersonAccel::new(anderson_m);

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();

        let solve = |ip: &[i64], ix: &[i32], d: &[f32], other: &[f32], gram: &[f32], out: &mut [f32]| {
            if use_eals {
                solve_one_side_eals(ip, ix, d, other, gram, out, k, lambda, alpha, eals_iters);
            } else if use_cholesky {
                let res = solve_one_side_cholesky(ip, ix, d, other, gram, out.len() / k, k, lambda, alpha);
                out.copy_from_slice(&res);
            } else {
                let res = solve_one_side_cg(ip, ix, d, other, gram, out.len() / k, k, lambda, alpha, cg_iters);
                out.copy_from_slice(&res);
            }
        };

        let x_old: Vec<f32> = if use_aa {
            let mut v = user_factors.clone();
            v.extend_from_slice(&item_factors);
            v
        } else {
            Vec::new()
        };

        let start_u = std::time::Instant::now();
        let g_item = gramian(&item_factors, n_items, k);
        solve(indptr, indices, data, &item_factors, &g_item, &mut user_factors);
        let u_time = start_u.elapsed();

        let start_i = std::time::Instant::now();
        let g_user = gramian(&user_factors, n_users, k);
        solve(indptr_t, indices_t, data_t, &user_factors, &g_user, &mut item_factors);
        let i_time = start_i.elapsed();

        if use_aa {
            let mut x_new: Vec<f32> = user_factors.clone();
            x_new.extend_from_slice(&item_factors);
            let f: Vec<f32> = x_new.iter().zip(&x_old).map(|(a, b)| a - b).collect();
            let x_mixed = accel.push_and_mix(x_old, f);
            let uf_len = n_users * k;
            user_factors.copy_from_slice(&x_mixed[..uf_len]);
            item_factors.copy_from_slice(&x_mixed[uf_len..]);
        }

        let iter_time = iter_start.elapsed();
        total_time += iter_time;

        if verbose {
            let aa_tag = if use_aa { " AA" } else { "   " };
            println!(
                "  {:>4}{} | {:>10.1}s | {:>10.1}s | {:>9.1}s ",
                iter + 1,
                aa_tag,
                u_time.as_secs_f64(),
                i_time.as_secs_f64(),
                iter_time.as_secs_f64()
            );
        }
    }

    if verbose {
        println!("  ------------------------------------------------");
        println!("  Done in {:.1}s", total_time.as_secs_f64());
    }

    (user_factors, item_factors)
}

fn top_n_items(
    uf: &[f32],
    itf: &[f32],
    uid: usize,
    n_items: usize,
    k: usize,
    n: usize,
    exc: &[i32],
    exc_start: usize,
    exc_end: usize,
) -> (Vec<i32>, Vec<f32>) {
    use ahash::AHashSet;
    let u = &uf[uid * k..(uid + 1) * k];
    let excluded: AHashSet<i32> = exc[exc_start..exc_end].iter().copied().collect();

    let mut scores = vec![0.0f32; n_items];
    faer::linalg::matmul::matmul(
        faer::MatMut::from_column_major_slice_mut(&mut scores, n_items, 1).as_mut(),
        faer::Accum::Replace,
        MatRef::from_row_major_slice(itf, n_items, k),
        MatRef::from_column_major_slice(u, k, 1),
        1.0f32,
        faer::Par::Seq,
    );

    let mut scored: Vec<(f32, i32)> = scores
        .into_iter()
        .enumerate()
        .filter_map(|(i, sc)| {
            let item_id = i as i32;
            if excluded.contains(&item_id) { None } else { Some((sc, item_id)) }
        })
        .collect();
    let take = n.min(scored.len());
    if take == 0 {
        return (vec![], vec![]);
    }
    scored.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(take);
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (
        scored.iter().map(|(_, i)| *i).collect(),
        scored.iter().map(|(s, _)| *s).collect(),
    )
}

fn top_n_users(
    uf: &[f32],
    itf: &[f32],
    iid: usize,
    n_users: usize,
    k: usize,
    n: usize,
) -> (Vec<i32>, Vec<f32>) {
    let y = &itf[iid * k..(iid + 1) * k];

    let mut scores = vec![0.0f32; n_users];
    faer::linalg::matmul::matmul(
        faer::MatMut::from_column_major_slice_mut(&mut scores, n_users, 1).as_mut(),
        faer::Accum::Replace,
        MatRef::from_row_major_slice(uf, n_users, k),
        MatRef::from_column_major_slice(y, k, 1),
        1.0f32,
        faer::Par::Seq,
    );

    let mut scored: Vec<(f32, i32)> = scores
        .into_iter()
        .enumerate()
        .map(|(u, sc)| (sc, u as i32))
        .collect();
    let take = n.min(scored.len());
    if take == 0 {
        return (vec![], vec![]);
    }
    scored.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(take);
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (
        scored.iter().map(|(_, i)| *i).collect(),
        scored.iter().map(|(s, _)| *s).collect(),
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_users, n_items, factors, regularization, alpha, iterations, seed, verbose, cg_iters=10, use_cholesky=false, anderson_m=0, use_eals=false, eals_iters=1))]
pub fn als_fit_implicit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f32>,
    n_users: usize,
    n_items: usize,
    factors: usize,
    regularization: f32,
    alpha: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
    cg_iters: usize,
    use_cholesky: bool,
    anderson_m: usize,
    use_eals: bool,
    eals_iters: usize,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let id = data.as_slice()?;

    let (ti, tx, td) = py.detach(|| csr_transpose(ip, ix, id, n_users, n_items));

    let (uf, itf) = py.detach(|| {
        als_train(
            ip,
            ix,
            id,
            &ti,
            &tx,
            &td,
            n_users,
            n_items,
            factors,
            regularization,
            alpha,
            iterations,
            seed,
            verbose,
            cg_iters,
            use_cholesky,
            anderson_m,
            use_eals,
            eals_iters,
        )
    });

    let ua = PyArray1::from_vec(py, uf);
    let ia = PyArray1::from_vec(py, itf);

    Ok((
        ua.reshape([n_users, factors])?.into(),
        ia.reshape([n_items, factors])?.into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, user_id, n, exclude_indptr, exclude_indices))]
pub fn als_recommend_items<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray2<f32>,
    item_factors: PyReadonlyArray2<f32>,
    user_id: usize,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let k = user_factors.shape()[1];
    let n_items = item_factors.shape()[0];
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;
    let es = ep[user_id] as usize;
    let ee = ep[user_id + 1] as usize;
    let (ids, scores) = top_n_items(uf, itf, user_id, n_items, k, n, ex, es, ee);
    Ok((ids.into_pyarray(py), scores.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, item_id, n))]
pub fn als_recommend_users<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray2<f32>,
    item_factors: PyReadonlyArray2<f32>,
    item_id: usize,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let k = user_factors.shape()[1];
    let n_users = user_factors.shape()[0];
    let (ids, scores) = top_n_users(uf, itf, item_id, n_users, k, n);
    Ok((ids.into_pyarray(py), scores.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (item_factors, indices, data, regularization, alpha, cg_iters=10, use_cholesky=false, use_eals=false, eals_iters=1))]
pub fn als_recalculate_user<'py>(
    py: Python<'py>,
    item_factors: PyReadonlyArray2<f32>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f32>,
    regularization: f32,
    alpha: f32,
    cg_iters: usize,
    use_cholesky: bool,
    use_eals: bool,
    eals_iters: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let itf = item_factors.as_slice()?;
    let ix = indices.as_slice()?;
    let id = data.as_slice()?;

    let k = item_factors.shape()[1];
    let n_items = item_factors.shape()[0];
    let mut out = vec![0.0f32; k];
    let ip = vec![0, ix.len() as i64];
    
    let g_item = gramian(itf, n_items, k);
    
    if use_eals {
        solve_one_side_eals(&ip, ix, id, itf, &g_item, &mut out, k, regularization, alpha, eals_iters);
    } else if use_cholesky {
        let res = solve_one_side_cholesky(&ip, ix, id, itf, &g_item, 1, k, regularization, alpha);
        out.copy_from_slice(&res);
    } else {
        let res = solve_one_side_cg(&ip, ix, id, itf, &g_item, 1, k, regularization, alpha, cg_iters);
        out.copy_from_slice(&res);
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, n, exclude_indptr, exclude_indices))]
pub fn als_recommend_all<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray2<f32>,
    item_factors: PyReadonlyArray2<f32>,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let k = user_factors.shape()[1];
    let n_users = user_factors.shape()[0];
    let n_items = item_factors.shape()[0];
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;

    // Parallel processing across all users
    let results: Vec<(Vec<i32>, Vec<i32>, Vec<f32>)> = (0..n_users)
        .into_par_iter()
        .map(|user_id| {
            let es = ep[user_id] as usize;
            let ee = ep[user_id + 1] as usize;
            let (ids, scores) = top_n_items(uf, itf, user_id, n_items, k, n, ex, es, ee);
            
            let user_ids = vec![user_id as i32; ids.len()];
            (user_ids, ids, scores)
        })
        .collect();

    // Flatten results
    let mut all_user_ids = Vec::with_capacity(n_users * n);
    let mut all_item_ids = Vec::with_capacity(n_users * n);
    let mut all_scores = Vec::with_capacity(n_users * n);

    for (u_ids, i_ids, sc) in results {
        all_user_ids.extend(u_ids);
        all_item_ids.extend(i_ids);
        all_scores.extend(sc);
    }

    Ok((
        all_user_ids.into_pyarray(py),
        all_item_ids.into_pyarray(py),
        all_scores.into_pyarray(py),
    ))
}
