use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;

fn prune_top_k(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    k: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let n_rows = indptr.len() - 1;
    
    // We compute row-wise top K in parallel
    let row_results: Vec<(Vec<i32>, Vec<f32>)> = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            
            let mut row_data: Vec<(f32, i32)> = indices[start..end]
                .iter()
                .zip(data[start..end].iter())
                .filter_map(|(&idx, &val)| {
                    if idx as usize != row { // ignore self-similarity
                        Some((val, idx))
                    } else {
                        None
                    }
                })
                .collect();
                
            let take = k.min(row_data.len());
            if take == 0 {
                return (vec![], vec![]);
            }
            
            row_data.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            row_data.truncate(take);
            // Sort by index so it's a valid CSR matrix
            row_data.sort_unstable_by_key(|&(_, idx)| idx);
            
            let out_idx: Vec<i32> = row_data.iter().map(|&(_, idx)| idx).collect();
            let out_val: Vec<f32> = row_data.iter().map(|&(val, _)| val).collect();
            
            (out_idx, out_val)
        })
        .collect();
        
    let mut new_indptr = Vec::with_capacity(n_rows + 1);
    new_indptr.push(0);
    
    let mut total_nnz = 0;
    for (idx, _) in &row_results {
        total_nnz += idx.len() as i64;
        new_indptr.push(total_nnz);
    }
    
    let mut new_indices = Vec::with_capacity(total_nnz as usize);
    let mut new_data = Vec::with_capacity(total_nnz as usize);
    
    for (idx, val) in row_results {
        new_indices.extend(idx);
        new_data.extend(val);
    }
    
    (new_indptr, new_indices, new_data)
}


fn knn_top_n_items(
    w_indptr: &[i64],
    w_indices: &[i32],
    w_data: &[f32],
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

    // Compute W^T * u
    let mut scores = vec![0.0f32; n_items];
    
    // W is symmetric item-item similarity. W^T is the same as W if it is symmetric.
    // However, top-K pruning makes it non-symmetric. We really want to compute score sum_j W_{ij} u_j
    // For a user row u, score_i = \sum_{j \in items(u)} W_ji * u_j.
    // Assuming W is item-item where W_ji is similarity from j to i
    for (idx_j_user, &val_u) in u_indices.iter().zip(u_data.iter()) {
        let j = *idx_j_user as usize;
        let w_start = w_indptr[j] as usize;
        let w_end = w_indptr[j + 1] as usize;
        
        let w_idx = &w_indices[w_start..w_end];
        let w_val = &w_data[w_start..w_end];
        
        for (&i, &val_w) in w_idx.iter().zip(w_val.iter()) {
            scores[i as usize] += val_w * val_u;
        }
    }
    
    let mut scored: Vec<(f32, i32)> = scores
        .into_iter()
        .enumerate()
        .filter_map(|(i, score)| {
            if score > 0.0 && !excluded.contains(&(i as i32)) {
                Some((score, i as i32))
            } else {
                None
            }
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
#[pyo3(signature = (indptr, indices, data, k))]
pub fn itemknn_top_k<'a>(
    py: Python<'a>,
    indptr: PyReadonlyArray1<'a, i64>,
    indices: PyReadonlyArray1<'a, i32>,
    data: PyReadonlyArray1<'a, f32>,
    k: usize,
) -> PyResult<(Bound<'a, PyArray1<i64>>, Bound<'a, PyArray1<i32>>, Bound<'a, PyArray1<f32>>)> {
    let (ip, ix, dt) = prune_top_k(indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, k);
    Ok((ip.into_pyarray(py), ix.into_pyarray(py), dt.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (w_indptr, w_indices, w_data, user_indptr, user_indices, user_data, user_id, n, exclude_indptr, exclude_indices, n_items))]
pub fn itemknn_recommend_items<'py>(
    py: Python<'py>,
    w_indptr: PyReadonlyArray1<i64>,
    w_indices: PyReadonlyArray1<i32>,
    w_data: PyReadonlyArray1<f32>,
    user_indptr: PyReadonlyArray1<i64>,
    user_indices: PyReadonlyArray1<i32>,
    user_data: PyReadonlyArray1<f32>,
    user_id: usize,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
    n_items: usize,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let w_ip = w_indptr.as_slice()?;
    let w_ix = w_indices.as_slice()?;
    let w_dt = w_data.as_slice()?;
    
    let u_ip = user_indptr.as_slice()?;
    let u_ix = user_indices.as_slice()?;
    let u_data = user_data.as_slice()?;
    
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;
    let es = ep[user_id] as usize;
    let ee = ep[user_id + 1] as usize;
    
    let (ids, scores) = knn_top_n_items(
        w_ip, w_ix, w_dt, u_ip, u_ix, u_data, user_id, n_items, n, ex, es, ee,
    );
    Ok((ids.into_pyarray(py), scores.into_pyarray(py)))
}
