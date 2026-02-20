use faer::{MatRef, linalg::matmul::matmul, Accum, Par};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;

// ─── Dot / axpy helpers ──────────────────────────────────────────────────────

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline(always)]
fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    // 8-wide manual unroll — compiler can auto-vectorize with -C opt-level=3
    let n = x.len();
    let chunks = n / 8 * 8;
    let mut i = 0;
    while i < chunks {
        y[i]   += alpha * x[i];
        y[i+1] += alpha * x[i+1];
        y[i+2] += alpha * x[i+2];
        y[i+3] += alpha * x[i+3];
        y[i+4] += alpha * x[i+4];
        y[i+5] += alpha * x[i+5];
        y[i+6] += alpha * x[i+6];
        y[i+7] += alpha * x[i+7];
        i += 8;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

// ─── Gramian YᵀY (parallel, upper-triangle then symmetrise) ──────────────────

fn gramian(factors: &[f32], n: usize, k: usize) -> Vec<f32> {
    // faer's matmul computes YᵀY with SIMD + rayon parallelism.
    let y = faer::Mat::from_fn(n, k, |r, c| factors[r * k + c]);
    let yt = y.transpose();

    let mut g = faer::Mat::<f32>::zeros(k, k);
    matmul(
        g.as_mut(),
        Accum::Replace,
        yt,
        y.as_ref(),
        1.0f32,
        Par::rayon(0),
    );

    let mut r = vec![0.0f32; k * k];
    for a in 0..k {
        for b in 0..k {
            r[a * k + b] = g[(a, b)];
        }
    }
    r
}

// ─── CG solver for implicit ALS ──────────────────────────────────────────────
//
// Solves: (YᵀY + λI + Σᵢ(cᵢ-1)yᵢyᵢᵀ) x = Σᵢ cᵢyᵢ
//
// Instead of forming the full A matrix (k×k) and doing Cholesky O(k³),
// we do CG with implicit matrix-vector products:
//   A·p = (YᵀY)·p + λ·p + Σᵢ(cᵢ-1)(yᵢᵀp)yᵢ
//
// This is O(k × nnz_per_row) per CG iteration, and we need only ~3 iterations.

// Thread-local scratch buffers — allocated once per thread, reused across users.
thread_local! {
    static SCRATCH: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
        RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
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

        // Pre-collect (item, confidence) pairs
        let mut sparse: Vec<(usize, f32)> = Vec::with_capacity(nnz_u);
        for idx in start..end {
            let i = indices[idx] as usize;
            let c = 1.0 + alpha * data[idx];
            sparse.push((i, c));
        }

        // Use thread-local scratch to avoid heap alloc per user
        SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut b, ref mut r, ref mut p, ref mut ap) = *borrow;
            b.clear(); b.resize(k, 0.0);
            r.clear(); r.resize(k, 0.0);
            p.clear(); p.resize(k, 0.0);
            ap.clear(); ap.resize(k, 0.0);

            // Compute RHS: b = Σᵢ cᵢ yᵢ
            for &(i, c) in &sparse {
                let yi = &other[i * k..(i + 1) * k];
                axpy(c, yi, b);
            }

            // A·v helper: A·v = (YᵀY + λI)·v + Σᵢ(cᵢ-1)(yᵢᵀv)yᵢ
            let apply_a = |v: &[f32], out: &mut [f32]| {
                for a in 0..k {
                    let mut s = 0.0f32;
                    for bb in 0..k { s += gram[a * k + bb] * v[bb]; }
                    out[a] = s + eff_lambda * v[a];
                }
                for &(i, c) in &sparse {
                    let yi = &other[i * k..(i + 1) * k];
                    let w = (c - 1.0) * dot(yi, v);
                    axpy(w, yi, out);
                }
            };

            xu.fill(0.0);
            r.copy_from_slice(b);
            p.copy_from_slice(b);
            let mut rsold = dot(r, r);

            if rsold < 1e-20 { return; }

            for _ in 0..cg_iters {
                apply_a(p, ap);
                let pap = dot(p, ap);
                if pap <= 0.0 { break; }
                let ak = rsold / pap;

                axpy(ak, p, xu);
                axpy(-ak, ap, r);

                let rsnew = dot(r, r);
                if rsnew < 1e-20 { break; }
                let beta = rsnew / rsold;
                for j in 0..k { p[j] = r[j] + beta * p[j]; }
                rsold = rsnew;
            }
        }); // end SCRATCH.with
    });

    out
}

// ─── Cholesky solver for implicit ALS ────────────────────────────────────────
//
// Direct approach: form A = (YᵀY + λI) + Σᵢ(cᵢ-1)yᵢyᵢᵀ  (k×k)
//                  b = Σᵢ cᵢ yᵢ                             (k)
// Then solve Ax = b via Cholesky. O(k³ + nnz_u·k) per user.
// Exact solution — no iterations. Wins when avg nnz_u >> k.
//
// Cholesky: we do the LL' factorisation in-place (lower-triangular).

fn cholesky_solve_inplace(a: &mut [f32], b: &mut [f32], k: usize) {
    // LL' factorisation of a (column-major k×k, lower triangle overwritten)
    for j in 0..k {
        // Compute diagonal
        let mut sum = a[j * k + j];
        for p in 0..j {
            sum -= a[j * k + p] * a[j * k + p];
        }
        if sum <= 0.0 {
            sum = 1e-8; // numerical guard
        }
        let ljj = sum.sqrt();
        a[j * k + j] = ljj;
        let inv_ljj = 1.0 / ljj;

        // Fill column j below diagonal
        for i in (j + 1)..k {
            let mut s = a[i * k + j];
            for p in 0..j {
                s -= a[i * k + p] * a[j * k + p];
            }
            a[i * k + j] = s * inv_ljj;
        }
    }

    // Forward substitution: L·y = b
    for i in 0..k {
        let mut s = b[i];
        for j in 0..i {
            s -= a[i * k + j] * b[j];
        }
        b[i] = s / a[i * k + i];
    }

    // Backward substitution: L'·x = y
    for i in (0..k).rev() {
        let mut s = b[i];
        for j in (i + 1)..k {
            s -= a[j * k + i] * b[j];
        }
        b[i] = s / a[i * k + i];
    }
}

thread_local! {
    static SCRATCH_CHOL: RefCell<(Vec<f32>, Vec<f32>)> =
        RefCell::new((Vec::new(), Vec::new()));
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
        let end   = indptr[u + 1] as usize;

        SCRATCH_CHOL.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut a_buf, ref mut b_buf) = *borrow;
            // a_buf = k×k matrix A (row-major), b_buf = rhs of length k
            a_buf.clear(); a_buf.extend_from_slice(gram); // start with YᵀY
            b_buf.clear(); b_buf.resize(k, 0.0);

            // Add λI to diagonal
            for j in 0..k { a_buf[j * k + j] += eff_lambda; }

            // Add sparse part: Σᵢ (cᵢ-1) yᵢ yᵢᵀ  and accumulate rhs
            for idx in start..end {
                let i = indices[idx] as usize;
                let ci = 1.0 + alpha * data[idx];
                let yi = &other[i * k..(i + 1) * k];

                // rhs: b += ci * yi
                axpy(ci, yi, b_buf);

                // matrix: A += (ci-1) * yi * yi'
                let w = ci - 1.0;
                for r in 0..k {
                    axpy(w * yi[r], yi, &mut a_buf[r * k..(r + 1) * k]);
                }
            }

            if b_buf.iter().all(|&v| v == 0.0) { return; }

            cholesky_solve_inplace(a_buf, b_buf, k);
            xu.copy_from_slice(b_buf);
        });
    });

    out
}

// ─── CSR transpose ───────────────────────────────────────────────────────────

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

// ─── ALS train ───────────────────────────────────────────────────────────────

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
) -> (Vec<f32>, Vec<f32>) {
    let mut user_factors = random_factors(n_users, k, seed);
    let mut item_factors = random_factors(n_items, k, seed.wrapping_add(1));

    if verbose {
        let solver_name = if use_cholesky { "Cholesky" } else { &format!("CG(iters={})", cg_iters) };
        println!("  Solver: {}  factors={}", solver_name, k);
        println!("  ITER | USER FACTORS | ITEM FACTORS | TOTAL TIME ");
        println!("  ------------------------------------------------");
    }

    let mut total_time = std::time::Duration::new(0, 0);

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();
        
        let start_u = std::time::Instant::now();
        // 2) user_factors = solve(item_factors, data)
        let solve = |ip: &[i64], ix: &[i32], d: &[f32], other: &[f32], gram: &[f32], n: usize| {
            if use_cholesky {
                solve_one_side_cholesky(ip, ix, d, other, gram, n, k, lambda, alpha)
            } else {
                solve_one_side_cg(ip, ix, d, other, gram, n, k, lambda, alpha, cg_iters)
            }
        };

        let g_item = gramian(&item_factors, n_items, k);
        user_factors = solve(indptr, indices, data, &item_factors, &g_item, n_users);
        let u_time = start_u.elapsed();

        let start_i = std::time::Instant::now();
        let g_user = gramian(&user_factors, n_users, k);
        item_factors = solve(indptr_t, indices_t, data_t, &user_factors, &g_user, n_items);
        let i_time = start_i.elapsed();
        let iter_time = iter_start.elapsed();
        total_time += iter_time;

        if verbose {
            println!(
                "  {:>4} | {:>10.1}s | {:>10.1}s | {:>9.1}s ",
                iter + 1,
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

// ─── Top-N helpers ───────────────────────────────────────────────────────────

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
            (dot(u, y), i)
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
    scored.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
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
            (dot(x, y), u)
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
    scored.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    (
        scored.iter().map(|(_, i)| *i).collect(),
        scored.iter().map(|(s, _)| *s).collect(),
    )
}

// ─── PyO3 exports ────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_users, n_items, factors, regularization, alpha, iterations, seed, verbose, cg_iters=10, use_cholesky=false))]
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
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let ip = indptr.as_slice()?.to_vec();
    let ix = indices.as_slice()?.to_vec();
    let id = data.as_slice()?.to_vec();

    let (ti, tx, td) = csr_transpose(&ip, &ix, &id, n_users, n_items);

    let (uf, itf) = als_train(
        &ip, &ix, &id, &ti, &tx, &td, n_users, n_items, factors, regularization,
        alpha, iterations, seed, verbose, cg_iters, use_cholesky,
    );

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
