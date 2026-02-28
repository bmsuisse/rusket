// Pipeline batch recommendation — BLAS-accelerated.
//
// V2: Uses faer matmul to compute the full score matrix (U × I.T) in one
// BLAS call per retriever, then does top-k selection + merge + rerank
// in parallel via rayon.  This is dramatically faster than per-user
// dot product loops.

use faer::{linalg::matmul::matmul, Accum, MatMut, MatRef, Par};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// For a single model, compute the full score matrix (n_users × n_items)
/// using BLAS, then extract top-k per user in parallel.
fn bulk_score_and_topk(
    user_factors: &[f32],
    item_factors: &[f32],
    n_users: usize,
    n_items: usize,
    k_factors: usize,
    top_k: usize,
    ep: &[i64],
    ex: &[i32],
) -> Vec<Vec<(i32, f32)>> {
    // Compute full score matrix: scores = U × I.T   (n_users × n_items)
    let mut scores = vec![0.0f32; n_users * n_items];
    matmul(
        MatMut::from_row_major_slice_mut(&mut scores, n_users, n_items).as_mut(),
        Accum::Replace,
        MatRef::from_row_major_slice(user_factors, n_users, k_factors),
        MatRef::from_row_major_slice(item_factors, n_items, k_factors).transpose(),
        1.0f32,
        Par::rayon(0),
    );

    // Extract top-k per user in parallel
    (0..n_users)
        .into_par_iter()
        .map(|uid| {
            let row = &scores[uid * n_items..(uid + 1) * n_items];

            // Build exclusion set for this user
            let es = ep[uid] as usize;
            let ee = ep[uid + 1] as usize;
            let excluded: ahash::AHashSet<i32> = ex[es..ee].iter().copied().collect();

            let mut scored: Vec<(f32, i32)> = row
                .iter()
                .enumerate()
                .filter_map(|(i, &sc)| {
                    let item_id = i as i32;
                    if excluded.contains(&item_id) {
                        None
                    } else {
                        Some((sc, item_id))
                    }
                })
                .collect();

            let take = top_k.min(scored.len());
            if take == 0 {
                return vec![];
            }
            scored.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(take);
            scored
                .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            scored.into_iter().map(|(sc, id)| (id, sc)).collect()
        })
        .collect()
}

/// Merge candidates from multiple retrievers using max/mean/sum strategy.
/// merge_strategy: 0 = max, 1 = sum, 2 = mean
fn merge_candidates(
    candidates_per_model: &[Vec<(i32, f32)>],
    merge_strategy: u8,
) -> Vec<(i32, f32)> {
    use ahash::AHashMap;

    let mut merged: AHashMap<i32, (f32, f32)> = AHashMap::new(); // (sum, count) or (max, _)
    for candidates in candidates_per_model {
        for &(item_id, score) in candidates {
            let entry = merged.entry(item_id).or_insert((
                if merge_strategy == 0 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                },
                0.0,
            ));
            match merge_strategy {
                0 => entry.0 = entry.0.max(score), // max
                _ => entry.0 += score,              // sum or mean (accumulate)
            }
            entry.1 += 1.0;
        }
    }

    let mut result: Vec<(i32, f32)> = merged
        .into_iter()
        .map(|(id, (agg, count))| {
            let final_score = if merge_strategy == 2 {
                agg / count
            } else {
                agg
            };
            (id, final_score)
        })
        .collect();

    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Rerank candidates using dot product from the reranker's factors.
/// Uses faer matmul for the batch dot product on candidate subset.
fn rerank_user(
    candidates: &[(i32, f32)],
    user_vec: &[f32],
    reranker_item_factors: &[f32],
    k_factors: usize,
    n_reranker_items: usize,
) -> Vec<(i32, f32)> {
    let mut result: Vec<(i32, f32)> = Vec::with_capacity(candidates.len());
    for &(item_id, _) in candidates {
        let idx = item_id as usize;
        if idx >= n_reranker_items {
            result.push((item_id, f32::NEG_INFINITY));
            continue;
        }
        let item_vec = &reranker_item_factors[idx * k_factors..(idx + 1) * k_factors];
        let mut dot = 0.0f32;
        // Unrolled dot product for the (small) rerank dimension
        let k8 = k_factors / 8 * 8;
        let mut j = 0;
        while j < k8 {
            unsafe {
                dot += user_vec.get_unchecked(j) * item_vec.get_unchecked(j)
                    + user_vec.get_unchecked(j + 1) * item_vec.get_unchecked(j + 1)
                    + user_vec.get_unchecked(j + 2) * item_vec.get_unchecked(j + 2)
                    + user_vec.get_unchecked(j + 3) * item_vec.get_unchecked(j + 3)
                    + user_vec.get_unchecked(j + 4) * item_vec.get_unchecked(j + 4)
                    + user_vec.get_unchecked(j + 5) * item_vec.get_unchecked(j + 5)
                    + user_vec.get_unchecked(j + 6) * item_vec.get_unchecked(j + 6)
                    + user_vec.get_unchecked(j + 7) * item_vec.get_unchecked(j + 7);
            }
            j += 8;
        }
        while j < k_factors {
            unsafe {
                dot += user_vec.get_unchecked(j) * item_vec.get_unchecked(j);
            }
            j += 1;
        }
        result.push((item_id, dot));
    }
    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// BLAS-accelerated batch pipeline: retrieve + merge + rerank for ALL users.
///
/// Key optimisation: uses faer matmul to compute the full (n_users × n_items)
/// score matrix per retriever in a single BLAS call, then does top-k selection
/// and merge in parallel via rayon.  This replaces per-user dot product loops.
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

    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;

    // Stage 1: BLAS bulk scoring per retriever (the big speedup!)
    let all_topk: Vec<Vec<Vec<(i32, f32)>>> = (0..n_retrievers)
        .map(|m| {
            bulk_score_and_topk(
                uf_slices[m],
                if_slices[m],
                n_users,
                n_items_list[m],
                k_factors_list[m],
                retrieve_k,
                ep,
                ex,
            )
        })
        .collect();

    // Stage 2+3: Merge + Rerank per user (parallel)
    let results: Vec<Vec<(i32, i32, f32)>> = (0..n_users)
        .into_par_iter()
        .map(|user_id| {
            // Collect candidates from all retrievers for this user
            let candidates_per_model: Vec<&Vec<(i32, f32)>> =
                (0..n_retrievers).map(|m| &all_topk[m][user_id]).collect();

            // Single retriever fast path (skip merge overhead)
            let mut candidates = if n_retrievers == 1 {
                all_topk[0][user_id].clone()
            } else {
                let refs: Vec<Vec<(i32, f32)>> =
                    candidates_per_model.into_iter().cloned().collect();
                merge_candidates(&refs, merge_strategy)
            };

            // Rerank
            if let (Some(ruf), Some(rif)) = (reranker_uf, reranker_if) {
                if k_rerank > 0 && user_id * k_rerank + k_rerank <= ruf.len() {
                    let user_vec = &ruf[user_id * k_rerank..(user_id + 1) * k_rerank];
                    candidates = rerank_user(&candidates, user_vec, rif, k_rerank, n_reranker_items);
                }
            }

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
