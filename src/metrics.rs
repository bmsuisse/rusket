use pyo3::prelude::*;
use std::collections::HashSet;

// ── Pure-Rust metric functions (no PyO3, take slices) ──────────────
pub(crate) fn precision_raw(actual: &[i32], predicted: &[i32], k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    let k_actual = k.min(predicted.len());
    if k_actual == 0 {
        return 0.0;
    }
    let actual_set: HashSet<i32> = actual.iter().copied().collect();
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
    let actual_set: HashSet<i32> = actual.iter().copied().collect();
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
    let actual_set: HashSet<i32> = actual.iter().copied().collect();
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
    let actual_set: HashSet<i32> = actual.iter().copied().collect();
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
pub fn precision_at_k(actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    precision_raw(&actual, &predicted, k)
}

#[pyfunction]
pub fn recall_at_k(actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    recall_raw(&actual, &predicted, k)
}

#[pyfunction]
pub fn hit_rate_at_k(actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    hit_rate_raw(&actual, &predicted, k)
}

#[pyfunction]
pub fn ndcg_at_k(actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    ndcg_raw(&actual, &predicted, k)
}

