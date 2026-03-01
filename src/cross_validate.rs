use ahash::AHashMap;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::als::{als_train, csr_transpose, top_n_items};
use crate::metrics::{hit_rate_raw, ndcg_raw, precision_raw, recall_raw};

// ── Helpers ────────────────────────────────────────────────────────

/// Build a CSR matrix from COO (users, items, values) arrays.
fn build_csr(
    users: &[i32],
    items: &[i32],
    values: &[f32],
    n_users: usize,
    n_items: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    // Count nnz per user
    let mut counts = vec![0i64; n_users];
    for &u in users {
        counts[u as usize] += 1;
    }
    let mut indptr = vec![0i64; n_users + 1];
    for i in 0..n_users {
        indptr[i + 1] = indptr[i] + counts[i];
    }
    let nnz = users.len();
    let mut indices = vec![0i32; nnz];
    let mut data = vec![0.0f32; nnz];
    let mut pos = indptr[..n_users].to_vec();
    for idx in 0..nnz {
        let u = users[idx] as usize;
        let p = pos[u] as usize;
        indices[p] = items[idx];
        data[p] = values[idx];
        pos[u] += 1;
    }
    // Deduplicate: sort each row by item and sum duplicate values
    for u in 0..n_users {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        if end - start <= 1 {
            continue;
        }
        // Simple insertion sort (rows are usually short)
        let row_indices = &mut indices[start..end];
        let row_data = &mut data[start..end];
        for i in 1..row_indices.len() {
            let mut j = i;
            while j > 0 && row_indices[j - 1] > row_indices[j] {
                row_indices.swap(j - 1, j);
                row_data.swap(j - 1, j);
                j -= 1;
            }
        }
    }
    let _ = n_items; // used for validation only
    (indptr, indices, data)
}

/// Evaluate one trained model on a test set: returns (precision, recall, ndcg, hr).
fn evaluate_model(
    user_factors: &[f32],
    item_factors: &[f32],
    n_items: usize,
    k_latent: usize,
    test_users: &[i32],
    test_items: &[i32],
    // CSR indptr/indices for exclude-seen
    train_indptr: &[i64],
    train_indices: &[i32],
    k: usize,
) -> [f32; 4] {
    // Group test items by user
    let mut user_test_items: AHashMap<i32, Vec<i32>> = AHashMap::new();
    for (&u, &i) in test_users.iter().zip(test_items.iter()) {
        user_test_items.entry(u).or_default().push(i);
    }

    let unique_users: Vec<i32> = user_test_items.keys().copied().collect();
    let n_eval = unique_users.len();
    if n_eval == 0 {
        return [0.0; 4];
    }

    let (sum_p, sum_r, sum_n, sum_h) = unique_users
        .iter()
        .map(|&uid| {
            let u = uid as usize;
            let es = train_indptr[u] as usize;
            let ee = train_indptr[u + 1] as usize;
            let (pred_ids, _scores) =
                top_n_items(user_factors, item_factors, u, n_items, k_latent, k, train_indices, es, ee);
            let actual = &user_test_items[&uid];
            (
                precision_raw(actual, &pred_ids, k),
                recall_raw(actual, &pred_ids, k),
                ndcg_raw(actual, &pred_ids, k),
                hit_rate_raw(actual, &pred_ids, k),
            )
        })
        .fold((0.0f32, 0.0f32, 0.0f32, 0.0f32), |acc, (p, r, n, h)| {
            (acc.0 + p, acc.1 + r, acc.2 + n, acc.3 + h)
        });

    let n = n_eval as f32;
    [sum_p / n, sum_r / n, sum_n / n, sum_h / n]
}

// ── Config struct ──────────────────────────────────────────────────

struct AlsConfig {
    factors: usize,
    regularization: f32,
    alpha: f32,
    iterations: usize,
    use_eals: bool,
    eals_iters: usize,
    cg_iters: usize,
    use_cholesky: bool,
    seed: u64,
}

/// Run cross-validation for a single config, returning per-fold metric arrays.
fn cv_one_config(
    config: &AlsConfig,
    folds: &[(Vec<i32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<i32>, Vec<f32>)],
    n_users: usize,
    n_items: usize,
    k: usize,
    verbose: bool,
    config_idx: usize,
    n_configs: usize,
) -> Vec<[f32; 4]> {
    let n_folds = folds.len();
    let mut fold_metrics = Vec::with_capacity(n_folds);

    for (fi, fold) in folds.iter().enumerate() {
        let (tr_users, tr_items, tr_values, te_users, te_items, _te_values) = fold;

        // Build CSR from train data
        let (indptr, indices, data) = build_csr(tr_users, tr_items, tr_values, n_users, n_items);
        let (ti, tx, td) = csr_transpose(&indptr, &indices, &data, n_users, n_items);

        // Train
        let (uf, itf) = als_train(
            &indptr,
            &indices,
            &data,
            &ti,
            &tx,
            &td,
            n_users,
            n_items,
            config.factors,
            config.regularization,
            config.alpha,
            config.iterations,
            config.seed,
            false, // verbose for individual ALS training off
            config.cg_iters,
            config.use_cholesky,
            0, // anderson_m
            config.use_eals,
            config.eals_iters,
        );

        // Evaluate
        let metrics = evaluate_model(
            &uf,
            &itf,
            n_items,
            config.factors,
            te_users,
            te_items,
            &indptr,
            &indices,
            k,
        );

        if verbose {
            let metric_names = ["precision", "recall", "ndcg", "hr"];
            let params = format!(
                "factors={} alpha={} reg={} iter={} eals={}",
                config.factors, config.alpha, config.regularization,
                config.iterations, config.use_eals
            );
            println!(
                "  [{}/{}] {}  fold {}/{}  {}@{}={:.4}  {}@{}={:.4}  {}@{}={:.4}  {}@{}={:.4}",
                config_idx + 1,
                n_configs,
                params,
                fi + 1,
                n_folds,
                metric_names[0], k, metrics[0],
                metric_names[1], k, metrics[1],
                metric_names[2], k, metrics[2],
                metric_names[3], k, metrics[3],
            );
        }

        fold_metrics.push(metrics);
    }

    fold_metrics
}

// ── Main PyO3 function ─────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    users, items, values,
    n_users, n_items,
    factors_list, regularization_list, alpha_list, iterations_list,
    use_eals_list, eals_iters_list, cg_iters_list, use_cholesky_list,
    seed_list,
    n_folds, k, metric, fold_seed, verbose
))]
#[allow(clippy::too_many_arguments)]
pub fn cross_validate_als(
    py: Python<'_>,
    users: Vec<i32>,
    items: Vec<i32>,
    values: Vec<f32>,
    n_users: usize,
    n_items: usize,
    // Param grid — one entry per config
    factors_list: Vec<usize>,
    regularization_list: Vec<f32>,
    alpha_list: Vec<f32>,
    iterations_list: Vec<usize>,
    use_eals_list: Vec<bool>,
    eals_iters_list: Vec<usize>,
    cg_iters_list: Vec<usize>,
    use_cholesky_list: Vec<bool>,
    seed_list: Vec<u64>,
    // CV settings
    n_folds: usize,
    k: usize,
    metric: String,
    fold_seed: u64,
    verbose: bool,
) -> PyResult<(
    usize,                // best_config_idx
    f32,                  // best_mean_score (for primary metric)
    Vec<Vec<f32>>,        // per_config_means: [n_configs][4] — precision, recall, ndcg, hr
    Vec<Vec<f32>>,        // per_config_stds:  [n_configs][4]
    Vec<Vec<Vec<f32>>>,   // per_config_fold_scores: [n_configs][n_folds][4]
)> {
    let n_configs = factors_list.len();
    let n = users.len();

    // Figure out which metric index to optimize (0=precision, 1=recall, 2=ndcg, 3=hr)
    let metric_idx = match metric.as_str() {
        "precision" => 0,
        "recall" => 1,
        "ndcg" => 2,
        "hr" => 3,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown metric: {}. Must be one of: precision, recall, ndcg, hr.",
                metric
            )))
        }
    };

    // ── Create fold splits (shuffled indices) ──────────────────────
    // Use a simple xorshift to shuffle indices, seeded by fold_seed
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_state = fold_seed;
    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let j = (rng_state as usize) % (i + 1);
        indices.swap(i, j);
    }

    // Split into folds
    let fold_size = n / n_folds;
    let mut fold_indices: Vec<Vec<usize>> = Vec::with_capacity(n_folds);
    for fi in 0..n_folds {
        let start = fi * fold_size;
        let end = if fi == n_folds - 1 { n } else { (fi + 1) * fold_size };
        fold_indices.push(indices[start..end].to_vec());
    }

    // Pre-build fold data: (train_users, train_items, train_values, test_users, test_items, test_values)
    let folds: Vec<(Vec<i32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<i32>, Vec<f32>)> = (0..n_folds)
        .map(|fi| {
            let test_idx = &fold_indices[fi];
            let train_count: usize = fold_indices.iter().enumerate()
                .filter(|(j, _)| *j != fi)
                .map(|(_, v)| v.len())
                .sum();

            let mut tr_u = Vec::with_capacity(train_count);
            let mut tr_i = Vec::with_capacity(train_count);
            let mut tr_v = Vec::with_capacity(train_count);
            let mut te_u = Vec::with_capacity(test_idx.len());
            let mut te_i = Vec::with_capacity(test_idx.len());
            let mut te_v = Vec::with_capacity(test_idx.len());

            for (j, fold_idx) in fold_indices.iter().enumerate() {
                if j == fi {
                    for &idx in fold_idx {
                        te_u.push(users[idx]);
                        te_i.push(items[idx]);
                        te_v.push(values[idx]);
                    }
                } else {
                    for &idx in fold_idx {
                        tr_u.push(users[idx]);
                        tr_i.push(items[idx]);
                        tr_v.push(values[idx]);
                    }
                }
            }
            (tr_u, tr_i, tr_v, te_u, te_i, te_v)
        })
        .collect();

    // ── Build configs ──────────────────────────────────────────────
    let configs: Vec<AlsConfig> = (0..n_configs)
        .map(|i| AlsConfig {
            factors: factors_list[i],
            regularization: regularization_list[i],
            alpha: alpha_list[i],
            iterations: iterations_list[i],
            use_eals: use_eals_list[i],
            eals_iters: eals_iters_list[i],
            cg_iters: cg_iters_list[i],
            use_cholesky: use_cholesky_list[i],
            seed: seed_list[i],
        })
        .collect();

    // ── Run CV in parallel across configs ──────────────────────────
    // Release the GIL so Python threads aren't blocked
    let results: Vec<Vec<[f32; 4]>> = py.detach(|| {
        configs
            .par_iter()
            .enumerate()
            .map(|(ci, config)| {
                cv_one_config(config, &folds, n_users, n_items, k, verbose, ci, n_configs)
            })
            .collect()
    });

    // ── Aggregate results ──────────────────────────────────────────
    let mut per_config_means: Vec<Vec<f32>> = Vec::with_capacity(n_configs);
    let mut per_config_stds: Vec<Vec<f32>> = Vec::with_capacity(n_configs);
    let mut per_config_fold_scores: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_configs);
    let mut best_idx = 0usize;
    let mut best_mean = -1.0f32;

    for (ci, fold_metrics) in results.iter().enumerate() {
        let fold_metrics: &Vec<[f32; 4]> = fold_metrics;
        let nf = fold_metrics.len() as f32;
        let mut means = [0.0f32; 4];
        for fm in fold_metrics {
            for m in 0..4 {
                means[m] += fm[m];
            }
        }
        for m in 0..4 {
            means[m] /= nf;
        }

        let mut stds = [0.0f32; 4];
        for fm in fold_metrics {
            for m in 0..4 {
                let diff = fm[m] - means[m];
                stds[m] += diff * diff;
            }
        }
        for m in 0..4 {
            stds[m] = (stds[m] / nf).sqrt();
        }

        if means[metric_idx] > best_mean {
            best_mean = means[metric_idx];
            best_idx = ci;
        }

        per_config_means.push(means.to_vec());
        per_config_stds.push(stds.to_vec());

        let fold_vecs: Vec<Vec<f32>> = fold_metrics.iter().map(|fm: &[f32; 4]| fm.to_vec()).collect();
        per_config_fold_scores.push(fold_vecs);
    }

    if verbose {
        let metric_names = ["precision", "recall", "ndcg", "hr"];
        println!(
            "\n  Best: {}@{}={:.4}  config #{}",
            metric_names[metric_idx],
            k,
            best_mean,
            best_idx + 1
        );
    }

    Ok((best_idx, best_mean, per_config_means, per_config_stds, per_config_fold_scores))
}

// ══════════════════════════════════════════════════════════════════════════
// Generic cross-validation for all factor-based models
// ══════════════════════════════════════════════════════════════════════════

/// Model-specific training configuration.
struct GenericConfig {
    // Common
    factors: usize,
    regularization: f32,
    iterations: usize,
    seed: u64,
    // ALS-specific
    alpha: f32,
    use_eals: bool,
    eals_iters: usize,
    cg_iters: usize,
    use_cholesky: bool,
    // SGD-based (BPR, SVD, SVD++, LightGCN)
    learning_rate: f32,
    // LightGCN
    k_layers: usize,
}

/// Dispatch training to the right model and return (user_factors, item_factors).
/// For SVD/SVD++, we discard biases since ranking metrics only need ordering.
fn train_model(
    kind: &str,
    config: &GenericConfig,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_users: usize,
    n_items: usize,
) -> (Vec<f32>, Vec<f32>) {
    match kind {
        "als" => {
            let (ti, tx, td) = csr_transpose(indptr, indices, data, n_users, n_items);
            als_train(
                indptr, indices, data, &ti, &tx, &td,
                n_users, n_items, config.factors, config.regularization,
                config.alpha, config.iterations, config.seed, false,
                config.cg_iters, config.use_cholesky, 0,
                config.use_eals, config.eals_iters,
            )
        }
        "bpr" => {
            crate::bpr::bpr_train(
                indptr, indices, n_users, n_items,
                config.factors, config.learning_rate, config.regularization,
                config.iterations, config.seed, false,
            )
        }
        "svd" => {
            let (uf, itf, _ub, _ib, _gm) = crate::svd::svd_train(
                indptr, indices, data, n_users, n_items,
                config.factors, config.learning_rate, config.regularization,
                config.iterations, config.seed, false,
            );
            (uf, itf)
        }
        "svdpp" => {
            let (uf, itf, _y, _ub, _ib, _gm) = crate::svd::svdpp_train(
                indptr, indices, data, n_users, n_items,
                config.factors, config.learning_rate, config.regularization,
                config.iterations, config.seed, false,
            );
            (uf, itf)
        }
        "lightgcn" => {
            let (ti, tx, _td) = csr_transpose(indptr, indices, data, n_users, n_items);
            // LightGCN needs item→user CSR (transpose of user→item)
            // Build item indptr from transpose
            crate::lightgcn::lightgcn_train(
                indptr, indices, &ti, &tx,
                n_users, n_items,
                config.factors, config.k_layers, config.learning_rate,
                config.regularization, config.iterations, config.seed, false,
            )
        }
        _ => panic!("Unknown model kind: {}", kind),
    }
}

/// Run cross-validation for one config using the generic train_model dispatch.
fn cv_one_config_generic(
    kind: &str,
    config: &GenericConfig,
    folds: &[(Vec<i32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<i32>, Vec<f32>)],
    n_users: usize,
    n_items: usize,
    k: usize,
    verbose: bool,
    config_idx: usize,
    n_configs: usize,
) -> Vec<[f32; 4]> {
    let n_folds = folds.len();
    let mut fold_metrics = Vec::with_capacity(n_folds);

    for (fi, fold) in folds.iter().enumerate() {
        let (tr_users, tr_items, tr_values, te_users, te_items, _te_values) = fold;

        let (indptr, indices, data) = build_csr(tr_users, tr_items, tr_values, n_users, n_items);

        let (uf, itf) = train_model(kind, config, &indptr, &indices, &data, n_users, n_items);

        let metrics = evaluate_model(
            &uf, &itf, n_items, config.factors,
            te_users, te_items, &indptr, &indices, k,
        );

        if verbose {
            println!(
                "  [{}/{}] kind={} factors={} reg={} iter={} lr={}  fold {}/{}  p@{}={:.4}  r@{}={:.4}  n@{}={:.4}  h@{}={:.4}",
                config_idx + 1, n_configs, kind,
                config.factors, config.regularization, config.iterations, config.learning_rate,
                fi + 1, n_folds,
                k, metrics[0], k, metrics[1], k, metrics[2], k, metrics[3],
            );
        }

        fold_metrics.push(metrics);
    }

    fold_metrics
}

#[pyfunction]
#[pyo3(signature = (
    kind,
    users, items, values,
    n_users, n_items,
    factors_list, regularization_list, iterations_list, seed_list,
    alpha_list, use_eals_list, eals_iters_list, cg_iters_list, use_cholesky_list,
    learning_rate_list, k_layers_list,
    n_folds, k, metric, fold_seed, verbose
))]
#[allow(clippy::too_many_arguments)]
pub fn cross_validate_generic(
    py: Python<'_>,
    kind: String,
    users: Vec<i32>,
    items: Vec<i32>,
    values: Vec<f32>,
    n_users: usize,
    n_items: usize,
    // Common params — one entry per config
    factors_list: Vec<usize>,
    regularization_list: Vec<f32>,
    iterations_list: Vec<usize>,
    seed_list: Vec<u64>,
    // ALS-specific
    alpha_list: Vec<f32>,
    use_eals_list: Vec<bool>,
    eals_iters_list: Vec<usize>,
    cg_iters_list: Vec<usize>,
    use_cholesky_list: Vec<bool>,
    // SGD-based (BPR, SVD, SVD++, LightGCN)
    learning_rate_list: Vec<f32>,
    // LightGCN
    k_layers_list: Vec<usize>,
    // CV settings
    n_folds: usize,
    k: usize,
    metric: String,
    fold_seed: u64,
    verbose: bool,
) -> PyResult<(
    usize,                // best_config_idx
    f32,                  // best_mean_score
    Vec<Vec<f32>>,        // per_config_means: [n_configs][4]
    Vec<Vec<f32>>,        // per_config_stds:  [n_configs][4]
    Vec<Vec<Vec<f32>>>,   // per_config_fold_scores: [n_configs][n_folds][4]
)> {
    let n_configs = factors_list.len();
    let n = users.len();

    let metric_idx = match metric.as_str() {
        "precision" => 0,
        "recall" => 1,
        "ndcg" => 2,
        "hr" => 3,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown metric: {}. Must be one of: precision, recall, ndcg, hr.",
                metric
            )))
        }
    };

    // Validate model kind
    match kind.as_str() {
        "als" | "bpr" | "svd" | "svdpp" | "lightgcn" => {}
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown model kind: {}. Must be one of: als, bpr, svd, svdpp, lightgcn.",
                kind
            )))
        }
    }

    // ── Create fold splits ─────────────────────────────────────────
    let mut indices_arr: Vec<usize> = (0..n).collect();
    let mut rng_state = fold_seed;
    for i in (1..n).rev() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let j = (rng_state as usize) % (i + 1);
        indices_arr.swap(i, j);
    }

    let fold_size = n / n_folds;
    let mut fold_indices: Vec<Vec<usize>> = Vec::with_capacity(n_folds);
    for fi in 0..n_folds {
        let start = fi * fold_size;
        let end = if fi == n_folds - 1 { n } else { (fi + 1) * fold_size };
        fold_indices.push(indices_arr[start..end].to_vec());
    }

    let folds: Vec<(Vec<i32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<i32>, Vec<f32>)> = (0..n_folds)
        .map(|fi| {
            let test_idx = &fold_indices[fi];
            let train_count: usize = fold_indices.iter().enumerate()
                .filter(|(j, _)| *j != fi)
                .map(|(_, v)| v.len())
                .sum();

            let mut tr_u = Vec::with_capacity(train_count);
            let mut tr_i = Vec::with_capacity(train_count);
            let mut tr_v = Vec::with_capacity(train_count);
            let mut te_u = Vec::with_capacity(test_idx.len());
            let mut te_i = Vec::with_capacity(test_idx.len());
            let mut te_v = Vec::with_capacity(test_idx.len());

            for (j, fold_idx) in fold_indices.iter().enumerate() {
                if j == fi {
                    for &idx in fold_idx {
                        te_u.push(users[idx]);
                        te_i.push(items[idx]);
                        te_v.push(values[idx]);
                    }
                } else {
                    for &idx in fold_idx {
                        tr_u.push(users[idx]);
                        tr_i.push(items[idx]);
                        tr_v.push(values[idx]);
                    }
                }
            }
            (tr_u, tr_i, tr_v, te_u, te_i, te_v)
        })
        .collect();

    // ── Build configs ──────────────────────────────────────────────
    let configs: Vec<GenericConfig> = (0..n_configs)
        .map(|i| GenericConfig {
            factors: factors_list[i],
            regularization: regularization_list[i],
            iterations: iterations_list[i],
            seed: seed_list[i],
            alpha: alpha_list[i],
            use_eals: use_eals_list[i],
            eals_iters: eals_iters_list[i],
            cg_iters: cg_iters_list[i],
            use_cholesky: use_cholesky_list[i],
            learning_rate: learning_rate_list[i],
            k_layers: k_layers_list[i],
        })
        .collect();

    // ── Run CV in parallel across configs ──────────────────────────
    let results: Vec<Vec<[f32; 4]>> = py.detach(|| {
        configs
            .par_iter()
            .enumerate()
            .map(|(ci, config)| {
                cv_one_config_generic(&kind, config, &folds, n_users, n_items, k, verbose, ci, n_configs)
            })
            .collect()
    });

    // ── Aggregate results ──────────────────────────────────────────
    let mut per_config_means: Vec<Vec<f32>> = Vec::with_capacity(n_configs);
    let mut per_config_stds: Vec<Vec<f32>> = Vec::with_capacity(n_configs);
    let mut per_config_fold_scores: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_configs);
    let mut best_idx = 0usize;
    let mut best_mean = -1.0f32;

    for (ci, fold_metrics) in results.iter().enumerate() {
        let nf = fold_metrics.len() as f32;
        let mut means = [0.0f32; 4];
        for fm in fold_metrics {
            for m in 0..4 {
                means[m] += fm[m];
            }
        }
        for m in 0..4 {
            means[m] /= nf;
        }

        let mut stds = [0.0f32; 4];
        for fm in fold_metrics {
            for m in 0..4 {
                let diff = fm[m] - means[m];
                stds[m] += diff * diff;
            }
        }
        for m in 0..4 {
            stds[m] = (stds[m] / nf).sqrt();
        }

        if means[metric_idx] > best_mean {
            best_mean = means[metric_idx];
            best_idx = ci;
        }

        per_config_means.push(means.to_vec());
        per_config_stds.push(stds.to_vec());

        let fold_vecs: Vec<Vec<f32>> = fold_metrics.iter().map(|fm: &[f32; 4]| fm.to_vec()).collect();
        per_config_fold_scores.push(fold_vecs);
    }

    if verbose {
        let metric_names = ["precision", "recall", "ndcg", "hr"];
        println!(
            "\n  Best: {}@{}={:.4}  config #{}",
            metric_names[metric_idx], k, best_mean, best_idx + 1
        );
    }

    Ok((best_idx, best_mean, per_config_means, per_config_stds, per_config_fold_scores))
}
