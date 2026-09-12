use ahash::AHashSet;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

// ── Pure-Rust metric functions (no PyO3, take slices) ──────────────
pub(crate) fn precision_raw(actual: &[i32], predicted: &[i32], k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    let k_actual = k.min(predicted.len());
    if k_actual == 0 {
        return 0.0;
    }
    let actual_set: AHashSet<i32> = actual.iter().copied().collect();
    let hits = predicted[..k_actual].iter().filter(|i| actual_set.contains(i)).count();
    hits as f32 / k as f32
}

pub(crate) fn recall_raw(actual: &[i32], predicted: &[i32], k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    let k_actual = k.min(predicted.len());
    if k_actual == 0 {
        return 0.0;
    }
    let actual_set: AHashSet<i32> = actual.iter().copied().collect();
    let hits = predicted[..k_actual].iter().filter(|i| actual_set.contains(i)).count();
    hits as f32 / actual_set.len() as f32
}

pub(crate) fn hit_rate_raw(actual: &[i32], predicted: &[i32], k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    let k_actual = k.min(predicted.len());
    if k_actual == 0 {
        return 0.0;
    }
    let actual_set: AHashSet<i32> = actual.iter().copied().collect();
    if predicted[..k_actual].iter().any(|i| actual_set.contains(i)) { 1.0 } else { 0.0 }
}

pub(crate) fn ndcg_raw(actual: &[i32], predicted: &[i32], k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    let k_actual = k.min(predicted.len());
    if k_actual == 0 {
        return 0.0;
    }
    let actual_set: AHashSet<i32> = actual.iter().copied().collect();
    let mut dcg = 0.0f32;
    for i in 0..k_actual {
        if actual_set.contains(&predicted[i]) {
            dcg += 1.0 / (2.0 + i as f32).log2();
        }
    }
    let idcg_len = k_actual.min(actual_set.len());
    let mut idcg = 0.0f32;
    for i in 0..idcg_len {
        idcg += 1.0 / (2.0 + i as f32).log2();
    }
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

// ── PyO3 wrappers ──────────────────────────────────────────────────

#[pyfunction]
pub fn precision_at_k(py: Python<'_>, actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    py.detach(|| precision_raw(&actual, &predicted, k))
}

#[pyfunction]
pub fn recall_at_k(py: Python<'_>, actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    py.detach(|| recall_raw(&actual, &predicted, k))
}

#[pyfunction]
pub fn hit_rate_at_k(py: Python<'_>, actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    py.detach(|| hit_rate_raw(&actual, &predicted, k))
}

#[pyfunction]
pub fn ndcg_at_k(py: Python<'_>, actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    py.detach(|| ndcg_raw(&actual, &predicted, k))
}

// ── Batched entry point ─────────────────────────────────────────────
//
// `evaluate()` in rusket/evaluation/metrics.py used to call `recommend_items`
// once per user and then one of the four `*_at_k` functions above once per
// user *per metric* -- 4 * n_users extra FFI round trips converting a Python
// list into a `Vec<i32>` each time. `metrics_batch` replaces all of that with
// a single call: ground truth is passed as a CSR-style (indptr, flat) pair
// (each user has a different number of relevant items) and predictions as a
// dense `(n_users, k)` i32 matrix padded with `-1` (predictions are internal
// item indices, always >= 0, so -1 is an unambiguous "no prediction here"
// sentinel for users with fewer than k recommendations).
//
// Returns per-user arrays rather than means: the raw per-user values are
// cheap to average in Python (`arr.mean()`) and keep this function useful
// for callers that may later want the distribution (e.g. percentiles),
// not just the mean `evaluate()` currently needs.
#[pyfunction]
pub fn metrics_batch<'py>(
    py: Python<'py>,
    actual_indptr: PyReadonlyArray1<i64>,
    actual_flat: PyReadonlyArray1<i32>,
    pred: PyReadonlyArray2<i32>,
    k: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let indptr = actual_indptr.as_slice()?.to_vec();
    let flat = actual_flat.as_slice()?.to_vec();
    let pred_shape = pred.shape().to_vec();
    let n_users = pred_shape[0];
    let row_width = pred_shape[1];
    // ponytail: copy the pred matrix into an owned Vec so the parallel loop
    // below can run under `py.detach` without holding a borrow tied to the
    // GIL (same idiom as als_recommend_all).
    let pred_owned: Vec<i32> = pred.as_array().iter().copied().collect();

    let (ndcg, hit_rate, precision, recall): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) = py.detach(|| {
        let per_user: Vec<(f32, f32, f32, f32)> = (0..n_users)
            .into_par_iter()
            .map(|u| {
                let a_start = indptr[u] as usize;
                let a_end = indptr[u + 1] as usize;
                let actual = &flat[a_start..a_end];

                let p_start = u * row_width;
                let p_row = &pred_owned[p_start..p_start + row_width];
                let valid_len = p_row.iter().position(|&x| x < 0).unwrap_or(row_width);
                let predicted = &p_row[..valid_len];

                (
                    ndcg_raw(actual, predicted, k),
                    hit_rate_raw(actual, predicted, k),
                    precision_raw(actual, predicted, k),
                    recall_raw(actual, predicted, k),
                )
            })
            .collect();

        let mut ndcg = Vec::with_capacity(n_users);
        let mut hit_rate = Vec::with_capacity(n_users);
        let mut precision = Vec::with_capacity(n_users);
        let mut recall = Vec::with_capacity(n_users);
        for (n_, h_, p_, r_) in per_user {
            ndcg.push(n_);
            hit_rate.push(h_);
            precision.push(p_);
            recall.push(r_);
        }
        (ndcg, hit_rate, precision, recall)
    });

    Ok((
        ndcg.into_pyarray(py),
        hit_rate.into_pyarray(py),
        precision.into_pyarray(py),
        recall.into_pyarray(py),
    ))
}

