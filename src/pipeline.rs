// Pipeline batch recommendation: multi-model merge + rerank for all users in parallel.
//
// This is the Rust hot path for `Pipeline.recommend_batch()`.
// It takes factor matrices from multiple retrievers and an optional reranker,
// then scores all users in parallel using rayon.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Score a single user against a single model's item factors.
/// Returns top-k (item_id, score) pairs.
fn score_user(
    user_vec: &[f32],
    item_factors: &[f32],
    n_items: usize,
    k_factors: usize,
    top_k: usize,
    exclude: &ahash::AHashSet<i32>,
) -> Vec<(i32, f32)> {
    let mut scores: Vec<(f32, i32)> = Vec::with_capacity(n_items);

    for i in 0..n_items {
        let item_id = i as i32;
        if exclude.contains(&item_id) {
            continue;
        }
        let item_vec = &item_factors[i * k_factors..(i + 1) * k_factors];
        let mut dot = 0.0f32;

        // 4-wide unrolled dot product
        let k4 = k_factors / 4 * 4;
        let mut j = 0;
        while j < k4 {
            unsafe {
                dot += user_vec.get_unchecked(j) * item_vec.get_unchecked(j)
                    + user_vec.get_unchecked(j + 1) * item_vec.get_unchecked(j + 1)
                    + user_vec.get_unchecked(j + 2) * item_vec.get_unchecked(j + 2)
                    + user_vec.get_unchecked(j + 3) * item_vec.get_unchecked(j + 3);
            }
            j += 4;
        }
        while j < k_factors {
            unsafe {
                dot += user_vec.get_unchecked(j) * item_vec.get_unchecked(j);
            }
            j += 1;
        }

        scores.push((dot, item_id));
    }

    let take = top_k.min(scores.len());
    if take == 0 {
        return vec![];
    }
    scores.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    scores.truncate(take);
    scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scores.into_iter().map(|(sc, id)| (id, sc)).collect()
}

/// Merge candidates from multiple retrievers using max/mean/sum strategy.
/// merge_strategy: 0 = max, 1 = sum, 2 = mean
fn merge_candidates(
    candidates_per_model: &[Vec<(i32, f32)>],
    merge_strategy: u8,
) -> Vec<(i32, f32)> {
    use ahash::AHashMap;

    let mut merged: AHashMap<i32, Vec<f32>> = AHashMap::new();
    for candidates in candidates_per_model {
        for &(item_id, score) in candidates {
            merged.entry(item_id).or_default().push(score);
        }
    }

    let mut result: Vec<(i32, f32)> = merged
        .into_iter()
        .map(|(id, scores)| {
            let merged_score = match merge_strategy {
                1 => scores.iter().sum::<f32>(),
                2 => scores.iter().sum::<f32>() / scores.len() as f32,
                _ => scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            };
            (id, merged_score)
        })
        .collect();

    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Rerank candidates using user-item dot product from the reranker model.
fn rerank_candidates(
    candidates: &[(i32, f32)],
    user_vec: &[f32],
    reranker_item_factors: &[f32],
    k_factors: usize,
    n_reranker_items: usize,
) -> Vec<(i32, f32)> {
    let mut result: Vec<(i32, f32)> = candidates
        .iter()
        .map(|&(item_id, _old_score)| {
            let idx = item_id as usize;
            if idx >= n_reranker_items {
                return (item_id, f32::NEG_INFINITY);
            }
            let item_vec = &reranker_item_factors[idx * k_factors..(idx + 1) * k_factors];
            let mut dot = 0.0f32;
            for j in 0..k_factors {
                unsafe {
                    dot += user_vec.get_unchecked(j) * item_vec.get_unchecked(j);
                }
            }
            (item_id, dot)
        })
        .collect();

    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Batch pipeline: retrieve + merge + rerank for ALL users in parallel.
///
/// Arguments:
/// - retriever_user_factors: list of (n_users × k_factors) flat arrays
/// - retriever_item_factors: list of (n_items × k_factors) flat arrays
/// - reranker_user_factors: optional (n_users × k_rerank) flat array
/// - reranker_item_factors: optional (n_items × k_rerank) flat array
/// - exclude_indptr: CSR indptr for per-user exclusion sets
/// - exclude_indices: CSR indices (item IDs to exclude per user)
/// - n_users, n_items, k_factors (per retriever), k_rerank, top_k, retrieve_k
/// - merge_strategy: 0=max, 1=sum, 2=mean
///
/// Returns: (user_ids, item_ids, scores) flat arrays.
#[pyfunction]
#[pyo3(signature = (
    retriever_user_factors_list,
    retriever_item_factors_list,
    n_users, n_items_list, k_factors_list,
    top_k, retrieve_k, merge_strategy,
    exclude_indptr, exclude_indices,
    reranker_user_factors=None,
    reranker_item_factors=None,
    n_reranker_items=0,
    k_rerank=0
))]
pub fn pipeline_batch_recommend<'py>(
    py: Python<'py>,
    retriever_user_factors_list: Vec<PyReadonlyArray2<'py, f32>>,
    retriever_item_factors_list: Vec<PyReadonlyArray2<'py, f32>>,
    n_users: usize,
    n_items_list: Vec<usize>,
    k_factors_list: Vec<usize>,
    top_k: usize,
    retrieve_k: usize,
    merge_strategy: u8,
    exclude_indptr: PyReadonlyArray1<'py, i64>,
    exclude_indices: PyReadonlyArray1<'py, i32>,
    reranker_user_factors: Option<PyReadonlyArray2<'py, f32>>,
    reranker_item_factors: Option<PyReadonlyArray2<'py, f32>>,
    n_reranker_items: usize,
    k_rerank: usize,
) -> PyResult<(
    pyo3::Bound<'py, PyArray1<i32>>,
    pyo3::Bound<'py, PyArray1<i32>>,
    pyo3::Bound<'py, PyArray1<f32>>,
)> {
    let n_retrievers = retriever_user_factors_list.len();

    // Collect slices from numpy arrays
    let uf_slices: Vec<&[f32]> = retriever_user_factors_list
        .iter()
        .map(|a| a.as_slice().unwrap())
        .collect();
    let if_slices: Vec<&[f32]> = retriever_item_factors_list
        .iter()
        .map(|a| a.as_slice().unwrap())
        .collect();

    let reranker_uf: Option<&[f32]> = reranker_user_factors.as_ref().map(|a| a.as_slice().unwrap());
    let reranker_if: Option<&[f32]> = reranker_item_factors.as_ref().map(|a| a.as_slice().unwrap());

    // Exclusion arrays — we accept them as 2D with shape (n_users+1,) and (nnz,) reshaped
    let ep_flat = exclude_indptr.as_slice()?;
    let ex_flat = exclude_indices.as_slice()?;

    // Process all users in parallel
    let results: Vec<Vec<(i32, i32, f32)>> = (0..n_users)
        .into_par_iter()
        .map(|user_id| {
            // Build exclusion set
            let es = ep_flat[user_id] as usize;
            let ee = ep_flat[user_id + 1] as usize;
            let excluded: ahash::AHashSet<i32> = ex_flat[es..ee].iter().copied().collect();

            // Stage 1: Retrieve from each model
            let candidates_per_model: Vec<Vec<(i32, f32)>> = (0..n_retrievers)
                .map(|m| {
                    let k = k_factors_list[m];
                    let n_items = n_items_list[m];
                    let user_vec = &uf_slices[m][user_id * k..(user_id + 1) * k];
                    score_user(user_vec, if_slices[m], n_items, k, retrieve_k, &excluded)
                })
                .collect();

            // Stage 2: Merge
            let mut candidates = merge_candidates(&candidates_per_model, merge_strategy);

            // Stage 3: Rerank (if reranker provided)
            if let (Some(ruf), Some(rif)) = (reranker_uf, reranker_if) {
                if k_rerank > 0 && user_id * k_rerank + k_rerank <= ruf.len() {
                    let reranker_user_vec = &ruf[user_id * k_rerank..(user_id + 1) * k_rerank];
                    candidates = rerank_candidates(
                        &candidates,
                        reranker_user_vec,
                        rif,
                        k_rerank,
                        n_reranker_items,
                    );
                }
            }

            // Top-k final results
            candidates.truncate(top_k);

            candidates
                .into_iter()
                .map(|(item_id, score)| (user_id as i32, item_id, score))
                .collect()
        })
        .collect();

    // Flatten
    let total: usize = results.iter().map(|v| v.len()).sum();
    let mut all_user_ids = Vec::with_capacity(total);
    let mut all_item_ids = Vec::with_capacity(total);
    let mut all_scores = Vec::with_capacity(total);

    for triples in results {
        for (uid, iid, sc) in triples {
            all_user_ids.push(uid);
            all_item_ids.push(iid);
            all_scores.push(sc);
        }
    }

    Ok((
        all_user_ids.into_pyarray(py),
        all_item_ids.into_pyarray(py),
        all_scores.into_pyarray(py),
    ))
}
