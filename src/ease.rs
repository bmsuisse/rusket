use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline(always)]
fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

fn ease_top_n_items(
    weights: &[f32],
    user_indptr: &[i64],
    user_indices: &[i32],
    user_data: &[f32],
    uid: usize,
    n_items: usize,
    n: usize,
    exc: &[i32],
    exc_start: usize,
    exc_end: usize,
) -> (Vec<i32>, Vec<f32>) {
    let u_start = user_indptr[uid] as usize;
    let u_end = user_indptr[uid + 1] as usize;
    let u_indices = &user_indices[u_start..u_end];
    let u_data = &user_data[u_start..u_end];

    // Scores the paper's way: s = u . B, where B[j,i] = -P_ji / P_ii is
    // normalised by the target item's diagonal (see ease_compute_weights
    // step 4). Streaming row j of B for each of the user's nonzero items j
    // is both the correct orientation and contiguous (B is row-major).
    //
    // History: v0.1.96 flipped this from (B . u) to (u . B) to match the
    // Python and CUDA paths, but at that time step 4 stored B^T, so `u . B`
    // was the WRONG orientation and EASE rankings regressed. Step 4 now
    // stores the real B, which makes this form correct here and in the
    // Python/CUDA paths that already used it.
    let mut scores = vec![0.0f32; n_items];
    for (&j, &u_j) in u_indices.iter().zip(u_data.iter()) {
        let row = &weights[(j as usize) * n_items..(j as usize + 1) * n_items];
        axpy(u_j, row, &mut scores);
    }

    // ponytail: mask exclusions directly instead of an AHashSet — |exc| is
    // typically tiny next to n_items.
    for &item_id in &exc[exc_start..exc_end] {
        if let Some(sc) = scores.get_mut(item_id as usize) {
            *sc = f32::NEG_INFINITY;
        }
    }

    let mut scored: Vec<(f32, i32)> = scores
        .into_iter()
        .enumerate()
        .filter_map(|(i, sc)| if sc.is_finite() { Some((sc, i as i32)) } else { None })
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
#[pyo3(signature = (weights, user_indptr, user_indices, user_data, user_id, n, exclude_indptr, exclude_indices))]
pub fn ease_recommend_items<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<f32>,
    user_indptr: PyReadonlyArray1<i64>,
    user_indices: PyReadonlyArray1<i32>,
    user_data: PyReadonlyArray1<f32>,
    user_id: usize,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let w = weights.as_slice()?;
    let u_ip = user_indptr.as_slice()?;
    let u_ix = user_indices.as_slice()?;
    let u_data = user_data.as_slice()?;
    
    let n_items = weights.shape()[0];
    
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;
    let es = ep[user_id] as usize;
    let ee = ep[user_id + 1] as usize;
    let (ids, scores) = ease_top_n_items(
        w, u_ip, u_ix, u_data, user_id, n_items, n, ex, es, ee,
    );
    Ok((ids.into_pyarray(py), scores.into_pyarray(py)))
}

/// Compute EASE item weight matrix B entirely in Rust.
///
/// Steps:
/// 1. Build dense Gram matrix G = X^T X from CSR input
/// 2. Add regularization to diagonal: G += λI
/// 3. Invert G via Cholesky decomposition (using faer)
/// 4. Compute B[i,j] = -P_ij / P_jj (normalised by the target item's
///    diagonal, per Steck 2019 eq. 8), zero the diagonal
fn ease_compute_weights(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_items: usize,
    regularization: f32,
) -> Vec<f32> {
    let n_users = indptr.len() - 1;

    // Step 1: Build dense Gram matrix G = X^T X  (n_items × n_items)
    //
    // PERF FIX: the previous approach gave every rayon thread its own dense
    // `n_items * n_items * 8` byte f64 buffer and reduced them serially at the
    // end. At 20k items that is ~3.2 GB *per thread*, which was the practical
    // ceiling on how many items EASE could handle. It also computed both
    // (a, b) and (b, a) for every co-occurring pair even though G = X^T X is
    // symmetric, doing 2x the necessary work.
    //
    // Fix: parallelize over item ROWS of a single shared `gram` buffer
    // instead of over users with per-thread buffers. Row `a`'s slice
    // `gram[a*n_items .. (a+1)*n_items]` is written by exactly one rayon task
    // (via `par_chunks_mut(n_items)`), so no two tasks ever touch the same
    // memory — no locks, no per-thread copies, no reduction pass. For row
    // `a` we only need: which users rated item `a`, and for each such user,
    // the rest of their item vector (to accumulate cross terms). That is a
    // CSC (item -> users) lookup joined against the existing CSR (user ->
    // items) structure. We only write `b >= a` (upper triangle, diagonal
    // included), which halves the outer-product work; the lower triangle is
    // filled by mirroring afterwards in one O(n^2) pass.
    //
    // CORRECTNESS: gram[a,b] = sum over users u of val_a(u) * val_b(u) for
    // all u that rated both a and b. Fixing row a and iterating exactly the
    // users who rated a (from the CSC), then for each such user iterating
    // their *entire* item row (from the CSR) and keeping only b >= a,
    // enumerates precisely those (u, b) pairs — nothing added or dropped
    // relative to the original full double loop, just reordered and half of
    // it deferred to the mirror step. Diagonal entries are naturally
    // included because b >= a allows b == a.
    let mut item_indptr = vec![0i64; n_items + 1];
    for &it in indices {
        item_indptr[it as usize + 1] += 1;
    }
    for i in 0..n_items {
        item_indptr[i + 1] += item_indptr[i];
    }
    let nnz = indices.len();
    let mut item_indices = vec![0i32; nnz];
    let mut item_data = vec![0.0f32; nnz];
    {
        let mut cursor = item_indptr.clone();
        for u in 0..n_users {
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            for k in start..end {
                let it = indices[k] as usize;
                let pos = cursor[it] as usize;
                item_indices[pos] = u as i32;
                item_data[pos] = data[k];
                cursor[it] += 1;
            }
        }
    }

    let mut gram = vec![0.0f64; n_items * n_items];
    gram.par_chunks_mut(n_items).enumerate().for_each(|(a, row)| {
        let istart = item_indptr[a] as usize;
        let iend = item_indptr[a + 1] as usize;
        for k in istart..iend {
            let u = item_indices[k] as usize;
            let val_a = item_data[k] as f64;
            let ustart = indptr[u] as usize;
            let uend = indptr[u + 1] as usize;
            for m in ustart..uend {
                let b = indices[m] as usize;
                if b >= a {
                    row[b] += val_a * (data[m] as f64);
                }
            }
        }
    });

    // Mirror the upper triangle (b >= a, just computed) into the lower
    // triangle. Single O(n^2) serial pass — cheap relative to the Gram
    // build/Cholesky steps, and simple enough to be obviously correct.
    for a in 0..n_items {
        for b in (a + 1)..n_items {
            let v = gram[a * n_items + b];
            gram[b * n_items + a] = v;
        }
    }

    // Step 2: Add regularization to diagonal
    for i in 0..n_items {
        gram[i * n_items + i] += regularization as f64;
    }

    // Step 3: Invert via faer Cholesky
    //
    // PEAK-MEMORY FIX: `gram` (n^2 f64), `gram_mat` (another n^2 f64 copy),
    // `llt` (which owns its own internal n^2 f64 factor `L`, per faer's
    // `Llt<T> { L: Mat<T> }`), and `p_mat` (another n^2 f64) used to all stay
    // alive simultaneously until the function returned, because Rust only
    // drops locals at end-of-scope, not at last-use. That is up to four
    // n_items^2 * 8-byte buffers alive at once. `gram` and `gram_mat` hold
    // identical data at the point `gram_mat` is built (same for `gram_mat`
    // vs. `llt`'s internal factor once `llt` is computed, and `llt` vs.
    // `p_mat` once the solve is done) so each predecessor is dead weight the
    // instant its successor exists. Explicit `drop()` calls below free each
    // one as soon as it is superseded, capping the peak at two coexisting
    // n_items^2 f64 buffers (16 bytes/item^2) instead of four (32
    // bytes/item^2) -- see ease.py's `_estimate_peak_bytes` for the same
    // arithmetic used in the pre-flight memory guard.
    use faer::linalg::solvers::Solve;
    let gram_mat = faer::Mat::<f64>::from_fn(n_items, n_items, |r, c| gram[r * n_items + c]);
    drop(gram);
    let llt = gram_mat.as_ref().llt(faer::Side::Lower).expect("Cholesky decomposition failed");
    drop(gram_mat);

    // Solve G * P = I to get P = G^-1
    let mut p_mat = faer::Mat::<f64>::identity(n_items, n_items);
    llt.solve_in_place(p_mat.as_mut());
    drop(llt);

    // Step 4: Compute B = P / (-diag(P)), zero diagonal.
    //
    // PERF FIX: this was a serial loop over `i` (rows of the row-major output
    // `b`) with `j` innermost reading `p_mat[(i, j)]`. faer's `Mat` is
    // column-major, so fixing `i` and varying `j` strides through memory by a
    // full column (`n_items` elements) on every read — the opposite of what
    // the storage layout wants, and it was serial to boot.
    //
    // Fix: compute into a column-major scratch buffer `b_col` (same layout
    // as `p_mat`) in parallel via `par_chunks_mut(n_items)` over its columns
    // — column `j` is `b_col[j*n_items .. (j+1)*n_items]`, one task per
    // column, disjoint memory, and `p_mat[(i, j)]` for fixed `j` varying `i`
    // is now a contiguous read. Then transpose `b_col` into the final
    // row-major `b` (the layout `ease_fit` reshapes into a numpy array) in a
    // single O(n^2) pass, mirroring the same pattern used for the Gram
    // mirror above.
    let neg_diag: Vec<f64> = (0..n_items).map(|i| -p_mat[(i, i)]).collect();

    let mut b_col = vec![0.0f32; n_items * n_items];
    b_col.par_chunks_mut(n_items).enumerate().for_each(|(j, col)| {
        let nd_j = neg_diag[j];
        for (i, slot) in col.iter_mut().enumerate() {
            *slot = if i == j {
                0.0
            } else {
                // B[i,j] = -P_ij / P_jj -- normalised by the TARGET item's
                // diagonal (column j), per Steck 2019 eq. 8. Dividing by
                // neg_diag[i] instead stores B^T, which makes the `u @ B`
                // scoring used by the Rust, Python and CUDA paths compute
                // `u @ B^T` and rank wrongly whenever diag(P) is not constant.
                (p_mat[(i, j)] / nd_j) as f32
            };
        }
    });
    drop(p_mat);

    let mut b = vec![0.0f32; n_items * n_items];
    for j in 0..n_items {
        for i in 0..n_items {
            b[i * n_items + j] = b_col[j * n_items + i];
        }
    }
    drop(b_col);

    b
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_items, regularization))]
pub fn ease_fit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f32>,
    n_items: usize,
    regularization: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let dt = data.as_slice()?;

    let weights = py.detach(|| {
        ease_compute_weights(ip, ix, dt, n_items, regularization)
    });

    let arr = PyArray1::from_vec(py, weights);
    Ok(arr.reshape([n_items, n_items])?.into())
}

