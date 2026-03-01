use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
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
