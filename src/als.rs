use faer::{Mat, linalg::matmul::matmul, Accum, Par};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

// ─── Dot / axpy helpers ──────────────────────────────────────────────────────

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline(always)]
fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

// ─── Gramian YᵀY (parallel, upper-triangle then symmetrise) ──────────────────

fn gramian(factors: &[f32], n: usize, k: usize) -> Vec<f32> {
    // View factors as a faer matrix (n × k), column-major by building a Mat.
    // faer's matmul computes Yᵀ · Y with SIMD blocking and Rayon parallelism.
    let y: Mat<f32> = Mat::from_fn(n, k, |r, c| factors[r * k + c]);
    let yt = y.transpose();

    let mut g: Mat<f32> = Mat::zeros(k, k);
    // g = Yᵀ · Y   (alpha=1, beta=0, parallel)
    matmul(
        g.as_mut(),
        Accum::Replace,
        yt,
        y.as_ref(),
        1.0f32,
        Par::rayon(0), // use all available threads via Rayon
    );

    // Flatten to Vec<f32> in row-major order (matching our existing layout)
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

const CG_ITERS: usize = 10;

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
) -> Vec<f32> {
    let eff_lambda = lambda.max(1e-6);
    let mut out = vec![0.0f32; n * k];

    out.par_chunks_mut(k).enumerate().for_each(|(u, xu)| {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let nnz_u = end - start;

        // Pre-collect sparse data for this user
        // (item_index, confidence) pairs
        let mut sparse: Vec<(usize, f32)> = Vec::with_capacity(nnz_u);
        for idx in start..end {
            let i = indices[idx] as usize;
            let c = 1.0 + alpha * data[idx];
            sparse.push((i, c));
        }

        // Compute RHS: b = Σᵢ cᵢ yᵢ
        let mut b = vec![0.0f32; k];
        for &(i, c) in &sparse {
            let yi = &other[i * k..(i + 1) * k];
            axpy(c, yi, &mut b);
        }

        // ── CG iteration ─────────────────────────────────────────
        // x₀ = 0, r₀ = b - A·x₀ = b, p₀ = r₀

        // A·v helper (inline closure): computes A·v = (YᵀY + λI)·v + Σᵢ(cᵢ-1)(yᵢᵀv)yᵢ
        let apply_a = |v: &[f32], out: &mut [f32]| {
            // out = (YᵀY) · v
            for a in 0..k {
                let mut s = 0.0f32;
                for b in 0..k {
                    s += gram[a * k + b] * v[b];
                }
                out[a] = s + eff_lambda * v[a];
            }
            // Add sparse part: Σᵢ(cᵢ-1)(yᵢᵀv)yᵢ
            for &(i, c) in &sparse {
                let yi = &other[i * k..(i + 1) * k];
                let w = (c - 1.0) * dot(yi, v);
                axpy(w, yi, out);
            }
        };

        // x = 0
        xu.fill(0.0);
        let mut r = b.clone(); // r = b (since x=0)
        let mut p = r.clone(); // p = r
        let mut rsold = dot(&r, &r);

        if rsold < 1e-20 {
            return; // zero RHS means zero solution
        }

        let mut ap = vec![0.0f32; k];

        for _ in 0..CG_ITERS {
            apply_a(&p, &mut ap);
            let pap = dot(&p, &ap);
            if pap <= 0.0 {
                break;
            }
            let ak = rsold / pap;

            // x += ak * p
            axpy(ak, &p, xu);
            // r -= ak * Ap
            axpy(-ak, &ap, &mut r);

            let rsnew = dot(&r, &r);
            if rsnew < 1e-20 {
                break;
            }
            let beta = rsnew / rsold;
            // p = r + beta * p
            for j in 0..k {
                p[j] = r[j] + beta * p[j];
            }
            rsold = rsnew;
        }
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
) -> (Vec<f32>, Vec<f32>) {
    let mut user_factors = random_factors(n_users, k, seed);
    let mut item_factors = random_factors(n_items, k, seed.wrapping_add(1));

    if verbose {
        println!("  Optimizing {} latent factors using Conjugate Gradient.", k);
        println!("  ITER | USER FACTORS | ITEM FACTORS | TOTAL TIME ");
        println!("  ------------------------------------------------");
    }

    let mut total_time = std::time::Duration::new(0, 0);

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();
        
        let start_u = std::time::Instant::now();
        // 2) user_factors = solve(item_factors, data)
        let g_item = gramian(&item_factors, n_items, k);
        user_factors = solve_one_side_cg(
            indptr,
            indices,
            data,
            &item_factors,
            &g_item,
            n_users,
            k,
            lambda,
            alpha,
        );
        let u_time = start_u.elapsed();

        let start_i = std::time::Instant::now();
        // 3) item_factors = solve(user_factors, data_t)
        let g_user = gramian(&user_factors, n_users, k);
        item_factors = solve_one_side_cg(
            indptr_t,
            indices_t,
            data_t,
            &user_factors,
            &g_user,
            n_items,
            k,
            lambda,
            alpha,
        );
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
#[pyo3(signature = (indptr, indices, data, n_users, n_items, factors, regularization, alpha, iterations, seed, verbose))]
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
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let ip = indptr.as_slice()?.to_vec();
    let ix = indices.as_slice()?.to_vec();
    let id = data.as_slice()?.to_vec();

    // 1) Build transpose graph
    let (ti, tx, td) = csr_transpose(&ip, &ix, &id, n_users, n_items);

    let (uf, itf) = als_train(
        &ip, &ix, &id, &ti, &tx, &td, n_users, n_items, factors, regularization,
        alpha, iterations, seed, verbose,
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
