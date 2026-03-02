use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;

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
    use ahash::AHashSet;
    let excluded: AHashSet<i32> = exc[exc_start..exc_end].iter().copied().collect();
    
    let u_start = user_indptr[uid] as usize;
    let u_end = user_indptr[uid + 1] as usize;
    let u_indices = &user_indices[u_start..u_end];
    let u_data = &user_data[u_start..u_end];

    let mut scored: Vec<(f32, i32)> = (0..n_items as i32)
        .into_par_iter()
        .filter(|&i| !excluded.contains(&i))
        .map(|i| {
            let i = i as usize;
            let yw = &weights[i * n_items..(i + 1) * n_items];
            
            // Sparse dot product
            let mut score = 0.0f32;
            for (idx_u, &val_u) in u_indices.iter().zip(u_data.iter()) {
                score += yw[*idx_u as usize] * val_u;
            }
            
            (score, i as i32)
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
/// 4. Compute B = P / (-diag(P)), zero the diagonal
fn ease_compute_weights(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_items: usize,
    regularization: f32,
) -> Vec<f32> {
    let n_users = indptr.len() - 1;

    // Step 1: Build dense Gram matrix G = X^T X  (n_items × n_items)
    // For each user u, accumulate outer products of their item vectors
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n_users + num_threads - 1) / num_threads;

    let partials: Vec<Vec<f64>> = (0..num_threads)
        .into_par_iter()
        .map(|t| {
            let mut local = vec![0.0f64; n_items * n_items];
            let row_start = t * chunk_size;
            let row_end = (row_start + chunk_size).min(n_users);
            for u in row_start..row_end {
                let start = indptr[u] as usize;
                let end = indptr[u + 1] as usize;
                let u_ix = &indices[start..end];
                let u_data = &data[start..end];
                // Outer product contribution
                for (idx_a, &val_a) in u_ix.iter().zip(u_data.iter()) {
                    let a = *idx_a as usize;
                    for (idx_b, &val_b) in u_ix.iter().zip(u_data.iter()) {
                        let b = *idx_b as usize;
                        local[a * n_items + b] += (val_a as f64) * (val_b as f64);
                    }
                }
            }
            local
        })
        .collect();

    let mut gram = vec![0.0f64; n_items * n_items];
    for partial in &partials {
        for (g, p) in gram.iter_mut().zip(partial.iter()) {
            *g += *p;
        }
    }

    // Step 2: Add regularization to diagonal
    for i in 0..n_items {
        gram[i * n_items + i] += regularization as f64;
    }

    // Step 3: Invert via faer Cholesky
    use faer::linalg::solvers::Solve;
    let gram_mat = faer::Mat::<f64>::from_fn(n_items, n_items, |r, c| gram[r * n_items + c]);
    let llt = gram_mat.as_ref().llt(faer::Side::Lower).expect("Cholesky decomposition failed");
    
    // Solve G * P = I to get P = G^-1
    let mut p_mat = faer::Mat::<f64>::identity(n_items, n_items);
    llt.solve_in_place(p_mat.as_mut());

    // Step 4: Compute B = P / (-diag(P)), zero diagonal
    let mut b = vec![0.0f32; n_items * n_items];
    for i in 0..n_items {
        let diag_val = p_mat[(i, i)];
        let neg_diag = -diag_val;
        for j in 0..n_items {
            if i == j {
                b[i * n_items + j] = 0.0;
            } else {
                b[i * n_items + j] = (p_mat[(i, j)] / neg_diag) as f32;
            }
        }
    }

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

