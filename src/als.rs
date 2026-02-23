use faer::linalg::solvers::Solve;
use faer::{linalg::matmul::matmul, Accum, MatRef, Par, Side};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;

// SIMD optimized dot with explicit chunks for 64-bit vectors
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;
    while i + 16 <= a.len() {
        sum += a[i] * b[i] 
             + a[i + 1] * b[i + 1] 
             + a[i + 2] * b[i + 2] 
             + a[i + 3] * b[i + 3] 
             + a[i + 4] * b[i + 4] 
             + a[i + 5] * b[i + 5] 
             + a[i + 6] * b[i + 6] 
             + a[i + 7] * b[i + 7]
             + a[i + 8] * b[i + 8]
             + a[i + 9] * b[i + 9]
             + a[i + 10] * b[i + 10]
             + a[i + 11] * b[i + 11]
             + a[i + 12] * b[i + 12]
             + a[i + 13] * b[i + 13]
             + a[i + 14] * b[i + 14]
             + a[i + 15] * b[i + 15];
        i += 16;
    }
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

// SIMD optimized axpy with explicit chunks
#[inline(always)]
fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    let mut i = 0;
    while i + 16 <= x.len() {
        y[i] += alpha * x[i];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
        y[i + 4] += alpha * x[i + 4];
        y[i + 5] += alpha * x[i + 5];
        y[i + 6] += alpha * x[i + 6];
        y[i + 7] += alpha * x[i + 7];
        y[i + 8] += alpha * x[i + 8];
        y[i + 9] += alpha * x[i + 9];
        y[i + 10] += alpha * x[i + 10];
        y[i + 11] += alpha * x[i + 11];
        y[i + 12] += alpha * x[i + 12];
        y[i + 13] += alpha * x[i + 13];
        y[i + 14] += alpha * x[i + 14];
        y[i + 15] += alpha * x[i + 15];
        i += 16;
    }
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
    static SCRATCH: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
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

        SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut b, ref mut r, ref mut p, ref mut ap) = *borrow;
            b.clear();
            b.resize(k, 0.0);
            r.clear();
            r.resize(k, 0.0);
            p.clear();
            p.resize(k, 0.0);
            ap.clear();
            ap.resize(k, 0.0);

            for idx in start..end {
                let i = indices[idx] as usize;
                let c = 1.0 + alpha * data[idx];
                let yi = &other[i * k..(i + 1) * k];
                axpy_f32(c, yi, b);
            }

            let apply_a = |v: &[f32], out: &mut [f32]| {
                for a in 0..k {
                    let mut s = 0.0f32;
                    for bb in 0..k {
                        s += gram[a * k + bb] * v[bb];
                    }
                    out[a] = s + eff_lambda * v[a];
                }
                for idx in start..end {
                    let i = indices[idx] as usize;
                    let c = 1.0 + alpha * data[idx];
                    let yi = &other[i * k..(i + 1) * k];
                    let w = (c - 1.0) * dot_f32(yi, v);
                    axpy_f32(w, yi, out);
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

fn cholesky_solve_inplace(a: &mut [f32], b: &mut [f32], k: usize) {
    let a_mat = faer::MatMut::from_row_major_slice_mut(a, k, k);
    let mut b_mat = faer::MatMut::from_column_major_slice_mut(b, k, 1);

    if let Ok(llt) = a_mat.as_ref().llt(Side::Lower) {
        let x = llt.solve(b_mat.as_ref());
        b_mat.copy_from(x.as_ref());
    }
}

thread_local! {
    static SCRATCH_CHOL: RefCell<(Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new())) };
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

        SCRATCH_CHOL.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut a_buf, ref mut b_buf) = *borrow;
            a_buf.clear();
            a_buf.extend_from_slice(gram);
            b_buf.clear();
            b_buf.resize(k, 0.0);

            for j in 0..k {
                a_buf[j * k + j] += eff_lambda;
            }

            for idx in start..end {
                let i = indices[idx] as usize;
                let ci = 1.0 + alpha * data[idx];
                let yi = &other[i * k..(i + 1) * k];

                axpy_f32(ci, yi, b_buf);

                let w = ci - 1.0;
                for r in 0..k {
                    axpy_f32(w * yi[r], yi, &mut a_buf[r * k..(r + 1) * k]);
                }
            }

            if b_buf.iter().all(|&v| v == 0.0) {
                return;
            }

            cholesky_solve_inplace(a_buf, b_buf, k);
            xu.copy_from_slice(b_buf);
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
) -> (Vec<f32>, Vec<f32>) {
    let mut user_factors = random_factors(n_users, k, seed);
    let mut item_factors = random_factors(n_items, k, seed.wrapping_add(1));

    if verbose {
        let solver_name = if use_cholesky {
            "Cholesky"
        } else {
            &format!("CG(iters={})", cg_iters)
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

        let solve = |ip: &[i64], ix: &[i32], d: &[f32], other: &[f32], gram: &[f32], n: usize| {
            if use_cholesky {
                solve_one_side_cholesky(ip, ix, d, other, gram, n, k, lambda, alpha)
            } else {
                solve_one_side_cg(ip, ix, d, other, gram, n, k, lambda, alpha, cg_iters)
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
        user_factors = solve(indptr, indices, data, &item_factors, &g_item, n_users);
        let u_time = start_u.elapsed();

        let start_i = std::time::Instant::now();
        let g_user = gramian(&user_factors, n_users, k);
        item_factors = solve(indptr_t, indices_t, data_t, &user_factors, &g_user, n_items);
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
    let mut scored: Vec<(f32, i32)> = (0..n_items as i32)
        .filter(|i| !excluded.contains(i))
        .map(|i| {
            let y = &itf[(i as usize) * k..(i as usize + 1) * k];
            (dot_f32(u, y), i)
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
    let mut scored: Vec<(f32, i32)> = (0..n_users as i32)
        .map(|u| {
            let x = &uf[(u as usize) * k..(u as usize + 1) * k];
            (dot_f32(x, y), u)
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

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_users, n_items, factors, regularization, alpha, iterations, seed, verbose, cg_iters=10, use_cholesky=false, anderson_m=0))]
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
