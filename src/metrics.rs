use pyo3::prelude::*;
use std::collections::HashSet;

#[pyfunction]
pub fn precision_at_k(actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    
    let k_actual = std::cmp::min(k, predicted.len());
    if k_actual == 0 {
        return 0.0;
    }

    let actual_set: HashSet<i32> = actual.into_iter().collect();
    let mut hits = 0;

    for i in 0..k_actual {
        if actual_set.contains(&predicted[i]) {
            hits += 1;
        }
    }

    hits as f32 / k as f32
}

#[pyfunction]
pub fn recall_at_k(actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    
    let k_actual = std::cmp::min(k, predicted.len());
    if k_actual == 0 {
        return 0.0;
    }

    let actual_set: HashSet<i32> = actual.clone().into_iter().collect();
    let mut hits = 0;

    for i in 0..k_actual {
        if actual_set.contains(&predicted[i]) {
            hits += 1;
        }
    }

    hits as f32 / actual_set.len() as f32
}

#[pyfunction]
pub fn hit_rate_at_k(actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    
    let k_actual = std::cmp::min(k, predicted.len());
    if k_actual == 0 {
        return 0.0;
    }

    let actual_set: HashSet<i32> = actual.into_iter().collect();

    for i in 0..k_actual {
        if actual_set.contains(&predicted[i]) {
            return 1.0;
        }
    }

    0.0
}

#[pyfunction]
pub fn ndcg_at_k(actual: Vec<i32>, predicted: Vec<i32>, k: usize) -> f32 {
    if actual.is_empty() || k == 0 {
        return 0.0;
    }
    
    let k_actual = std::cmp::min(k, predicted.len());
    if k_actual == 0 {
        return 0.0;
    }

    let actual_set: HashSet<i32> = actual.clone().into_iter().collect();
    let mut dcg = 0.0;

    for i in 0..k_actual {
        if actual_set.contains(&predicted[i]) {
            dcg += 1.0 / (2.0 + i as f32).log2();
        }
    }

    // IDCG calculation
    let idcg_len = std::cmp::min(k_actual, actual_set.len());
    let mut idcg = 0.0;
    for i in 0..idcg_len {
        idcg += 1.0 / (2.0 + i as f32).log2();
    }

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}
