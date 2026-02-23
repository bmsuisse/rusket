use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

// ── SIMD-optimised primitives ─────────────────────────────────────────────
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ── Fast XorShift64 RNG ───────────────────────────────────────────────────
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xbad5eed } else { seed },
        }
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    #[inline(always)]
    fn next_float(&mut self) -> f32 {
        let v = self.next() & 0xFFFFFF;
        v as f32 / 0xFFFFFF as f32
    }
}

fn random_factors(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let scale = 1.0 / (k as f32).sqrt();
    let mut out = vec![0.0f32; n * k];
    for v in out.iter_mut() {
        *v = (rng.next_float() * 2.0 - 1.0) * scale;
    }
    out
}

fn svd_train(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_users: usize,
    n_items: usize,
    k: usize,
    learning_rate: f32,
    regularization: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let total_ratings: f64 = data.iter().map(|&v| v as f64).sum();
    let n_ratings = data.len();
    let global_mean = if n_ratings > 0 {
        (total_ratings / n_ratings as f64) as f32
    } else {
        0.0
    };

    let mut user_factors = random_factors(n_users, k, seed);
    let mut item_factors = random_factors(n_items, k, seed.wrapping_add(1));
    let mut user_biases = vec![0.0f32; n_users];
    let mut item_biases = vec![0.0f32; n_items];

    let num_threads = rayon::current_num_threads();

    let uf_ptr_raw = user_factors.as_mut_ptr() as usize;
    let if_ptr_raw = item_factors.as_mut_ptr() as usize;
    let ub_ptr_raw = user_biases.as_mut_ptr() as usize;
    let ib_ptr_raw = item_biases.as_mut_ptr() as usize;

    if verbose {
        println!("  SVD (Funk SVD / Biased SGD Matrix Factorization)");
        println!(
            "  Users: {}, Items: {}, Ratings: {}",
            n_users, n_items, n_ratings
        );
        println!(
            "  Factors: {}, lr={}, reg={}, μ={:.4}",
            k, learning_rate, regularization, global_mean
        );
        println!("  ITER |     RMSE     | SAMPLES/s  | TIME");
        println!("  ------------------------------------------------");
    }

    let start_time = std::time::Instant::now();

    // Build a flat (user, item, rating) triplet array for shuffled SGD
    let mut triplets: Vec<(u32, u32, f32)> = Vec::with_capacity(n_ratings);
    for u in 0..n_users {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        for idx in start..end {
            triplets.push((u as u32, indices[idx] as u32, data[idx]));
        }
    }

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();

        let mut shuffler = XorShift64::new(seed.wrapping_add(iter as u64).wrapping_add(999));
        let len = triplets.len();
        for i in (1..len).rev() {
            let j = (shuffler.next() as usize) % (i + 1);
            triplets.swap(i, j);
        }

        let chunk_size = (n_ratings + num_threads - 1) / num_threads;

        // Parallel SGD with per-thread RMSE accumulation
        let rmse_sum: f64 = triplets
            .par_chunks(chunk_size)
            .map(|chunk| {
                let uf_ptr = uf_ptr_raw as *mut f32;
                let if_ptr = if_ptr_raw as *mut f32;
                let ub_ptr = ub_ptr_raw as *mut f32;
                let ib_ptr = ib_ptr_raw as *mut f32;

                let mut local_sse = 0.0f64;

                for &(u, i, r) in chunk {
                    let u = u as usize;
                    let i = i as usize;

                    unsafe {
                        let pu = std::slice::from_raw_parts_mut(uf_ptr.add(u * k), k);
                        let qi = std::slice::from_raw_parts_mut(if_ptr.add(i * k), k);
                        let bu = &mut *ub_ptr.add(u);
                        let bi = &mut *ib_ptr.add(i);

                        let pred = global_mean + *bu + *bi + dot(pu, qi);
                        let err = r - pred;
                        local_sse += (err as f64) * (err as f64);

                        // Update biases
                        *bu += learning_rate * (err - regularization * *bu);
                        *bi += learning_rate * (err - regularization * *bi);

                        // Update factors
                        for f in 0..k {
                            let pu_f = pu[f];
                            let qi_f = qi[f];
                            pu[f] += learning_rate * (err * qi_f - regularization * pu_f);
                            qi[f] += learning_rate * (err * pu_f - regularization * qi_f);
                        }
                    }
                }

                local_sse
            })
            .sum();

        let rmse = (rmse_sum / n_ratings as f64).sqrt();
        let iter_time = iter_start.elapsed().as_secs_f64();

        if verbose {
            let samples_per_sec = (n_ratings as f64) / iter_time;
            println!(
                "  {:>4} | {:>12.6} | {:>10.0} | {:>6.2}s",
                iter + 1,
                rmse,
                samples_per_sec,
                iter_time
            );
        }
    }

    if verbose {
        println!("  ------------------------------------------------");
        println!("  Total time: {:.1}s", start_time.elapsed().as_secs_f64());
    }

    (
        user_factors,
        item_factors,
        user_biases,
        item_biases,
        global_mean,
    )
}

fn top_n_items(
    uf: &[f32],
    itf: &[f32],
    ub: &[f32],
    ib: &[f32],
    global_mean: f32,
    uid: usize,
    n_items: usize,
    k: usize,
    n: usize,
    exc: &[i32],
    exc_start: usize,
    exc_end: usize,
) -> (Vec<i32>, Vec<f32>) {
    use ahash::AHashSet;
    let u = &uf[uid * k..(uid + 1) * k];
    let bu = ub[uid];
    let excluded: AHashSet<i32> = exc[exc_start..exc_end].iter().copied().collect();
    let mut scored: Vec<(f32, i32)> = (0..n_items as i32)
        .filter(|i| !excluded.contains(i))
        .map(|i| {
            let y = &itf[(i as usize) * k..(i as usize + 1) * k];
            let score = global_mean + bu + ib[i as usize] + dot(u, y);
            (score, i)
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

fn top_n_users(
    uf: &[f32],
    itf: &[f32],
    ub: &[f32],
    ib: &[f32],
    global_mean: f32,
    iid: usize,
    n_users: usize,
    k: usize,
    n: usize,
) -> (Vec<i32>, Vec<f32>) {
    let y = &itf[iid * k..(iid + 1) * k];
    let bi = ib[iid];
    let mut scored: Vec<(f32, i32)> = (0..n_users as i32)
        .map(|u| {
            let x = &uf[(u as usize) * k..(u as usize + 1) * k];
            let score = global_mean + ub[u as usize] + bi + dot(x, y);
            (score, u)
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
#[pyo3(signature = (indptr, indices, data, n_users, n_items, factors, learning_rate, regularization, iterations, seed, verbose))]
pub fn svd_fit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f32>,
    n_users: usize,
    n_items: usize,
    factors: usize,
    learning_rate: f32,
    regularization: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<(
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
    Py<PyArray1<f32>>,
    Py<PyArray1<f32>>,
    f32,
)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let d = data.as_slice()?;

    let (uf, itf, ub, ib, gm) = py.detach(|| {
        svd_train(
            ip,
            ix,
            d,
            n_users,
            n_items,
            factors,
            learning_rate,
            regularization,
            iterations,
            seed,
            verbose,
        )
    });

    let ua = PyArray1::from_vec(py, uf);
    let ia = PyArray1::from_vec(py, itf);

    Ok((
        ua.reshape([n_users, factors])?.into(),
        ia.reshape([n_items, factors])?.into(),
        PyArray1::from_vec(py, ub).into(),
        PyArray1::from_vec(py, ib).into(),
        gm,
    ))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, user_biases, item_biases, global_mean, user_id, n, exclude_indptr, exclude_indices))]
pub fn svd_recommend_items<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray1<f32>,
    item_factors: PyReadonlyArray1<f32>,
    user_biases: PyReadonlyArray1<f32>,
    item_biases: PyReadonlyArray1<f32>,
    global_mean: f32,
    user_id: usize,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let ub = user_biases.as_slice()?;
    let ib = item_biases.as_slice()?;
    let k = uf.len() / ub.len(); // factors = total_user_factor_elems / n_users
    let n_items = ib.len();
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;
    let es = ep[user_id] as usize;
    let ee = ep[user_id + 1] as usize;
    let (ids, scores) = top_n_items(uf, itf, ub, ib, global_mean, user_id, n_items, k, n, ex, es, ee);
    Ok((
        PyArray1::from_vec(py, ids).into(),
        PyArray1::from_vec(py, scores).into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, user_biases, item_biases, global_mean, item_id, n))]
pub fn svd_recommend_users<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray1<f32>,
    item_factors: PyReadonlyArray1<f32>,
    user_biases: PyReadonlyArray1<f32>,
    item_biases: PyReadonlyArray1<f32>,
    global_mean: f32,
    item_id: usize,
    n: usize,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let ub = user_biases.as_slice()?;
    let ib = item_biases.as_slice()?;
    let k = uf.len() / ub.len();
    let n_users = ub.len();
    let (ids, scores) = top_n_users(uf, itf, ub, ib, global_mean, item_id, n_users, k, n);
    Ok((
        PyArray1::from_vec(py, ids).into(),
        PyArray1::from_vec(py, scores).into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, user_biases, item_biases, global_mean, n, exclude_indptr, exclude_indices))]
pub fn svd_recommend_all<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray1<f32>,
    item_factors: PyReadonlyArray1<f32>,
    user_biases: PyReadonlyArray1<f32>,
    item_biases: PyReadonlyArray1<f32>,
    global_mean: f32,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let ub = user_biases.as_slice()?;
    let ib = item_biases.as_slice()?;
    let k = uf.len() / ub.len();
    let n_users = ub.len();
    let n_items = ib.len();
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;

    let results: Vec<(Vec<i32>, Vec<i32>, Vec<f32>)> = (0..n_users)
        .into_par_iter()
        .map(|user_id| {
            let es = ep[user_id] as usize;
            let ee = ep[user_id + 1] as usize;
            let (ids, scores) =
                top_n_items(uf, itf, ub, ib, global_mean, user_id, n_items, k, n, ex, es, ee);
            let user_ids = vec![user_id as i32; ids.len()];
            (user_ids, ids, scores)
        })
        .collect();

    let mut all_user_ids = Vec::with_capacity(n_users * n);
    let mut all_item_ids = Vec::with_capacity(n_users * n);
    let mut all_scores = Vec::with_capacity(n_users * n);

    for (u_ids, i_ids, sc) in results {
        all_user_ids.extend(u_ids);
        all_item_ids.extend(i_ids);
        all_scores.extend(sc);
    }

    Ok((
        PyArray1::from_vec(py, all_user_ids).into(),
        PyArray1::from_vec(py, all_item_ids).into(),
        PyArray1::from_vec(py, all_scores).into(),
    ))
}

// ══════════════════════════════════════════════════════════════════════════
// SVD++ (Koren 2008)
// r̂_ui = μ + b_u + b_i + q_i · (p_u + |N(u)|^{-0.5} Σ_{j∈N(u)} y_j)
// ══════════════════════════════════════════════════════════════════════════

fn svdpp_train(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_users: usize,
    n_items: usize,
    k: usize,
    learning_rate: f32,
    regularization: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let total_ratings: f64 = data.iter().map(|&v| v as f64).sum();
    let n_ratings = data.len();
    let global_mean = if n_ratings > 0 {
        (total_ratings / n_ratings as f64) as f32
    } else {
        0.0
    };

    let mut user_factors = random_factors(n_users, k, seed);
    let mut item_factors = random_factors(n_items, k, seed.wrapping_add(1));
    let mut y_factors = random_factors(n_items, k, seed.wrapping_add(2));
    let mut user_biases = vec![0.0f32; n_users];
    let mut item_biases = vec![0.0f32; n_items];

    // Precompute per-user item sets from CSR
    let user_items: Vec<Vec<usize>> = (0..n_users)
        .map(|u| {
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            indices[start..end].iter().map(|&j| j as usize).collect()
        })
        .collect();

    let user_norm: Vec<f32> = user_items
        .iter()
        .map(|items| {
            if items.is_empty() {
                0.0
            } else {
                1.0 / (items.len() as f32).sqrt()
            }
        })
        .collect();

    if verbose {
        println!("  SVD++ (Koren 2008)");
        println!(
            "  Users: {}, Items: {}, Ratings: {}",
            n_users, n_items, n_ratings
        );
        println!(
            "  Factors: {}, lr={}, reg={}, μ={:.4}",
            k, learning_rate, regularization, global_mean
        );
        println!("  ITER |     RMSE     | SAMPLES/s  | TIME");
        println!("  ------------------------------------------------");
    }

    let start_time = std::time::Instant::now();

    // Build triplets for shuffled SGD
    let mut triplets: Vec<(u32, u32, f32)> = Vec::with_capacity(n_ratings);
    for u in 0..n_users {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        for idx in start..end {
            triplets.push((u as u32, indices[idx] as u32, data[idx]));
        }
    }

    // Temporary buffer for implicit sum (reused per user)
    let mut sum_y = vec![0.0f32; k];

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();

        // Shuffle
        let mut shuffler = XorShift64::new(seed.wrapping_add(iter as u64).wrapping_add(999));
        let len = triplets.len();
        for i in (1..len).rev() {
            let j = (shuffler.next() as usize) % (i + 1);
            triplets.swap(i, j);
        }

        // Sequential SGD (SVD++ requires per-user implicit sums → no race-free parallelism)
        let mut sse = 0.0f64;
        let mut last_user: u32 = u32::MAX;

        for &(u, i, r) in &triplets {
            let uu = u as usize;
            let ii = i as usize;

            // Recompute sum_y when we switch users
            if u != last_user {
                for f in 0..k {
                    sum_y[f] = 0.0;
                }
                for &j in &user_items[uu] {
                    for f in 0..k {
                        sum_y[f] += y_factors[j * k + f];
                    }
                }
                let norm = user_norm[uu];
                for f in 0..k {
                    sum_y[f] *= norm;
                }
                last_user = u;
            }

            let pu = &user_factors[uu * k..(uu + 1) * k];
            let qi = &item_factors[ii * k..(ii + 1) * k];

            // p_u_hat = p_u + sum_y
            let pred = global_mean
                + user_biases[uu]
                + item_biases[ii]
                + (0..k).map(|f| qi[f] * (pu[f] + sum_y[f])).sum::<f32>();
            let err = r - pred;
            sse += (err as f64) * (err as f64);

            // Update biases
            user_biases[uu] += learning_rate * (err - regularization * user_biases[uu]);
            item_biases[ii] += learning_rate * (err - regularization * item_biases[ii]);

            // Update q_i and p_u
            for f in 0..k {
                let pu_f = user_factors[uu * k + f];
                let qi_f = item_factors[ii * k + f];
                let sum_y_f = sum_y[f];
                item_factors[ii * k + f] +=
                    learning_rate * (err * (pu_f + sum_y_f) - regularization * qi_f);
                user_factors[uu * k + f] +=
                    learning_rate * (err * qi_f - regularization * pu_f);
            }

            // Update y_j for all j in N(u)
            let norm = user_norm[uu];
            if norm > 0.0 {
                for &j in &user_items[uu] {
                    for f in 0..k {
                        let qi_f = item_factors[ii * k + f];
                        let y_jf = y_factors[j * k + f];
                        let delta = learning_rate * (err * qi_f * norm - regularization * y_jf);
                        y_factors[j * k + f] += delta;
                        // Update running sum_y
                        sum_y[f] += delta * norm;
                    }
                }
            }
        }

        let rmse = (sse / n_ratings as f64).sqrt();
        let iter_time = iter_start.elapsed().as_secs_f64();

        if verbose {
            let samples_per_sec = (n_ratings as f64) / iter_time;
            println!(
                "  {:>4} | {:>12.6} | {:>10.0} | {:>6.2}s",
                iter + 1,
                rmse,
                samples_per_sec,
                iter_time
            );
        }
    }

    if verbose {
        println!("  ------------------------------------------------");
        println!("  Total time: {:.1}s", start_time.elapsed().as_secs_f64());
    }

    (
        user_factors,
        item_factors,
        y_factors,
        user_biases,
        item_biases,
        global_mean,
    )
}

/// Compute the effective user vector: p_u + |N(u)|^{-0.5} Σ y_j
#[inline]
fn svdpp_user_vec(
    pu: &[f32],
    y: &[f32],
    items: &[usize],
    k: usize,
) -> Vec<f32> {
    let mut vec = pu.to_vec();
    if !items.is_empty() {
        let norm = 1.0 / (items.len() as f32).sqrt();
        for &j in items {
            for f in 0..k {
                vec[f] += y[j * k + f] * norm;
            }
        }
    }
    vec
}

fn svdpp_top_n_items(
    uf: &[f32],
    itf: &[f32],
    y: &[f32],
    ub: &[f32],
    ib: &[f32],
    global_mean: f32,
    uid: usize,
    n_items: usize,
    k: usize,
    n: usize,
    user_rated: &[usize],
    exc: &[i32],
    exc_start: usize,
    exc_end: usize,
) -> (Vec<i32>, Vec<f32>) {
    use ahash::AHashSet;
    let pu = &uf[uid * k..(uid + 1) * k];
    let bu = ub[uid];
    let u_vec = svdpp_user_vec(pu, y, user_rated, k);
    let excluded: AHashSet<i32> = exc[exc_start..exc_end].iter().copied().collect();
    let mut scored: Vec<(f32, i32)> = (0..n_items as i32)
        .filter(|i| !excluded.contains(i))
        .map(|i| {
            let qi = &itf[(i as usize) * k..(i as usize + 1) * k];
            let score = global_mean + bu + ib[i as usize] + dot(&u_vec, qi);
            (score, i)
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

fn svdpp_top_n_users(
    uf: &[f32],
    itf: &[f32],
    y: &[f32],
    ub: &[f32],
    ib: &[f32],
    global_mean: f32,
    iid: usize,
    n_users: usize,
    k: usize,
    n: usize,
    indptr: &[i64],
    indices: &[i32],
) -> (Vec<i32>, Vec<f32>) {
    let qi = &itf[iid * k..(iid + 1) * k];
    let bi = ib[iid];
    let mut scored: Vec<(f32, i32)> = (0..n_users as i32)
        .map(|u| {
            let uu = u as usize;
            let pu = &uf[uu * k..(uu + 1) * k];
            let start = indptr[uu] as usize;
            let end = indptr[uu + 1] as usize;
            let items: Vec<usize> = indices[start..end].iter().map(|&j| j as usize).collect();
            let u_vec = svdpp_user_vec(pu, y, &items, k);
            let score = global_mean + ub[uu] + bi + dot(&u_vec, qi);
            (score, u)
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

// ── PyO3 exports ──────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_users, n_items, factors, learning_rate, regularization, iterations, seed, verbose))]
pub fn svdpp_fit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f32>,
    n_users: usize,
    n_items: usize,
    factors: usize,
    learning_rate: f32,
    regularization: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<(
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
    Py<PyArray1<f32>>,
    Py<PyArray1<f32>>,
    f32,
)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let d = data.as_slice()?;

    let (uf, itf, yf, ub, ib, gm) = py.detach(|| {
        svdpp_train(
            ip,
            ix,
            d,
            n_users,
            n_items,
            factors,
            learning_rate,
            regularization,
            iterations,
            seed,
            verbose,
        )
    });

    let ua = PyArray1::from_vec(py, uf);
    let ia = PyArray1::from_vec(py, itf);
    let ya = PyArray1::from_vec(py, yf);

    Ok((
        ua.reshape([n_users, factors])?.into(),
        ia.reshape([n_items, factors])?.into(),
        ya.reshape([n_items, factors])?.into(),
        PyArray1::from_vec(py, ub).into(),
        PyArray1::from_vec(py, ib).into(),
        gm,
    ))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, y_factors, user_biases, item_biases, global_mean, user_id, n, exclude_indptr, exclude_indices, interact_indptr, interact_indices))]
pub fn svdpp_recommend_items<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray1<f32>,
    item_factors: PyReadonlyArray1<f32>,
    y_factors: PyReadonlyArray1<f32>,
    user_biases: PyReadonlyArray1<f32>,
    item_biases: PyReadonlyArray1<f32>,
    global_mean: f32,
    user_id: usize,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
    interact_indptr: PyReadonlyArray1<i64>,
    interact_indices: PyReadonlyArray1<i32>,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let y = y_factors.as_slice()?;
    let ub = user_biases.as_slice()?;
    let ib = item_biases.as_slice()?;
    let k = uf.len() / ub.len();
    let n_items = ib.len();
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;
    let ip = interact_indptr.as_slice()?;
    let ix = interact_indices.as_slice()?;
    let es = ep[user_id] as usize;
    let ee = ep[user_id + 1] as usize;
    let is_ = ip[user_id] as usize;
    let ie = ip[user_id + 1] as usize;
    let user_rated: Vec<usize> = ix[is_..ie].iter().map(|&j| j as usize).collect();
    let (ids, scores) = svdpp_top_n_items(
        uf, itf, y, ub, ib, global_mean, user_id, n_items, k, n, &user_rated, ex, es, ee,
    );
    Ok((
        PyArray1::from_vec(py, ids).into(),
        PyArray1::from_vec(py, scores).into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, y_factors, user_biases, item_biases, global_mean, item_id, n, interact_indptr, interact_indices))]
pub fn svdpp_recommend_users<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray1<f32>,
    item_factors: PyReadonlyArray1<f32>,
    y_factors: PyReadonlyArray1<f32>,
    user_biases: PyReadonlyArray1<f32>,
    item_biases: PyReadonlyArray1<f32>,
    global_mean: f32,
    item_id: usize,
    n: usize,
    interact_indptr: PyReadonlyArray1<i64>,
    interact_indices: PyReadonlyArray1<i32>,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let y = y_factors.as_slice()?;
    let ub = user_biases.as_slice()?;
    let ib = item_biases.as_slice()?;
    let k = uf.len() / ub.len();
    let n_users = ub.len();
    let ip = interact_indptr.as_slice()?;
    let ix = interact_indices.as_slice()?;
    let (ids, scores) = svdpp_top_n_users(
        uf, itf, y, ub, ib, global_mean, item_id, n_users, k, n, ip, ix,
    );
    Ok((
        PyArray1::from_vec(py, ids).into(),
        PyArray1::from_vec(py, scores).into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, y_factors, user_biases, item_biases, global_mean, n, exclude_indptr, exclude_indices, interact_indptr, interact_indices))]
pub fn svdpp_recommend_all<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray1<f32>,
    item_factors: PyReadonlyArray1<f32>,
    y_factors: PyReadonlyArray1<f32>,
    user_biases: PyReadonlyArray1<f32>,
    item_biases: PyReadonlyArray1<f32>,
    global_mean: f32,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
    interact_indptr: PyReadonlyArray1<i64>,
    interact_indices: PyReadonlyArray1<i32>,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let y = y_factors.as_slice()?;
    let ub = user_biases.as_slice()?;
    let ib = item_biases.as_slice()?;
    let k = uf.len() / ub.len();
    let n_users = ub.len();
    let n_items = ib.len();
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;
    let ip = interact_indptr.as_slice()?;
    let ix = interact_indices.as_slice()?;

    let results: Vec<(Vec<i32>, Vec<i32>, Vec<f32>)> = (0..n_users)
        .into_par_iter()
        .map(|user_id| {
            let es = ep[user_id] as usize;
            let ee = ep[user_id + 1] as usize;
            let is_ = ip[user_id] as usize;
            let ie = ip[user_id + 1] as usize;
            let user_rated: Vec<usize> = ix[is_..ie].iter().map(|&j| j as usize).collect();
            let (ids, scores) = svdpp_top_n_items(
                uf, itf, y, ub, ib, global_mean, user_id, n_items, k, n, &user_rated, ex, es, ee,
            );
            let user_ids = vec![user_id as i32; ids.len()];
            (user_ids, ids, scores)
        })
        .collect();

    let mut all_user_ids = Vec::with_capacity(n_users * n);
    let mut all_item_ids = Vec::with_capacity(n_users * n);
    let mut all_scores = Vec::with_capacity(n_users * n);

    for (u_ids, i_ids, sc) in results {
        all_user_ids.extend(u_ids);
        all_item_ids.extend(i_ids);
        all_scores.extend(sc);
    }

    Ok((
        PyArray1::from_vec(py, all_user_ids).into(),
        PyArray1::from_vec(py, all_item_ids).into(),
        PyArray1::from_vec(py, all_scores).into(),
    ))
}
