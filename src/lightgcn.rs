use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::rng::XorShift64;
use crate::simd::dot;

fn random_factors(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let scale = (1.0_f32 / k as f32).sqrt() * 0.1;
    let mut out = vec![0.0f32; n * k];
    for v in out.iter_mut() {
        *v = (rng.next_float() * 2.0 - 1.0) * scale;
    }
    out
}

// ── Graph Augmentation (Edge Dropout for SGL) ──────────────────────────────
fn edge_dropout(
    u_indptr: &[i64], 
    u_indices: &[i32], 
    n_users: usize, 
    n_items: usize, 
    p: f32, 
    seed: u64
) -> (Vec<i64>, Vec<i32>, Vec<i64>, Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut rng = XorShift64::new(seed);
    let mut new_u_indptr = vec![0i64; n_users + 1];
    let mut new_u_indices = Vec::with_capacity((u_indices.len() as f32 * (1.0 - p) * 1.1) as usize);

    for u in 0..n_users {
        let start = u_indptr[u] as usize;
        let end = u_indptr[u + 1] as usize;
        for j in start..end {
            if rng.next_float() >= p {
                new_u_indices.push(u_indices[j]);
            }
        }
        new_u_indptr[u + 1] = new_u_indices.len() as i64;
    }

    let mut d_i = vec![0.0f32; n_items];
    let mut i_counts = vec![0i64; n_items];
    for &i in new_u_indices.iter() {
        d_i[i as usize] += 1.0;
        i_counts[i as usize] += 1;
    }

    let mut new_i_indptr = vec![0i64; n_items + 1];
    let mut current = 0;
    for i in 0..n_items {
        new_i_indptr[i] = current;
        current += i_counts[i];
    }
    new_i_indptr[n_items] = current;

    let mut i_insert_pos = new_i_indptr.clone();
    let mut new_i_indices = vec![0i32; current as usize];
    
    for u in 0..n_users {
        let start = new_u_indptr[u] as usize;
        let end = new_u_indptr[u + 1] as usize;
        for j in start..end {
            let item = new_u_indices[j] as usize;
            let pos = i_insert_pos[item] as usize;
            new_i_indices[pos] = u as i32;
            i_insert_pos[item] += 1;
        }
    }

    let d_u = (0..n_users).map(|u| (new_u_indptr[u + 1] - new_u_indptr[u]) as f32).collect();
    (new_u_indptr, new_u_indices, new_i_indptr, new_i_indices, d_u, d_i)
}

// ── InfoNCE Loss for SSL ────────────────────────────────────────────────────
fn compute_infonce_loss(
    v1: &[f32],
    v2: &[f32],
    k: usize,
    n_entities: usize,
    temp: f32,
    chunk_size: usize,
) -> (f32, Vec<f32>, Vec<f32>) {
    let mut g1 = vec![0.0f32; n_entities * k];
    let mut g2 = vec![0.0f32; n_entities * k];
    let g1_ptr = g1.as_mut_ptr() as usize;
    let g2_ptr = g2.as_mut_ptr() as usize;

    let loss: f32 = (0..n_entities)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| {
            let mut loss_sum = 0.0;
            for &i in &chunk {
                let s_pos = dot(&v1[i * k..(i + 1) * k], &v2[i * k..(i + 1) * k]) / temp;
                let mut sum_exp = 0.0;
                let mut exp_scores = Vec::with_capacity(chunk.len());
                for &j in &chunk {
                    let s = dot(&v1[i * k..(i + 1) * k], &v2[j * k..(j + 1) * k]) / temp;
                    let es = s.exp();
                    exp_scores.push(es);
                    sum_exp += es;
                }
                
                loss_sum += -(s_pos.exp() / sum_exp).ln(); 
                
                let mut grad_v1_i = vec![0.0; k];
                let g1_p = g1_ptr as *mut f32;
                let g2_p = g2_ptr as *mut f32;
                
                for (idx, &j) in chunk.iter().enumerate() {
                    let es = exp_scores[idx];
                    let weight = es / sum_exp;
                    let grad_score = if i == j { weight - 1.0 } else { weight };
                    let g_factor = grad_score / temp;
                    
                    for f in 0..k {
                        let val_v1 = v1[i * k + f];
                        let val_v2 = v2[j * k + f];
                        grad_v1_i[f] += g_factor * val_v2;
                        unsafe {
                            *g2_p.add(j * k + f) += g_factor * val_v1;
                        }
                    }
                }
                
                for f in 0..k {
                    unsafe {
                        *g1_p.add(i * k + f) += grad_v1_i[f];
                    }
                }
            }
            loss_sum
        })
        .sum();

    (loss, g1, g2)
}



// ---- Graph propagation (LightGCN forward pass) ----------------------------
// Propagate embeddings through the bipartite graph for k_layers steps.
// A_tilde = D_u^{-0.5} A D_i^{-0.5}   (normalised adjacency)
// E^{l+1}_u = A_tilde * E^l_i, E^{l+1}_i = A_tilde^T * E^l_u
// Final = mean of all layers including E^0
//
// OPTIMISATION: uses par fold/reduce with thread-local buffers instead of
// atomic CAS loops — eliminates all contention.

fn propagate(
    e_u: &[f32],      // n_users * k
    e_i: &[f32],      // n_items * k
    k: usize,
    n_users: usize,
    n_items: usize,
    u_indptr: &[i64],  // CSR for user → items
    u_indices: &[i32],
    i_indptr: &[i64], // CSR for item → users (transpose)
    i_indices: &[i32],
    d_u: &[f32],      // degree of each user
    d_i: &[f32],      // degree of each item
    k_layers: usize,
) -> (Vec<f32>, Vec<f32>) {
    // Pre-compute inverse square roots (avoid redundant sqrt per layer)
    let inv_sqrt_u: Vec<f32> = d_u.iter().map(|d| if *d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();
    let inv_sqrt_i: Vec<f32> = d_i.iter().map(|d| if *d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();

    // Accumulate final_u / final_i = sum of all layer E^l
    let mut final_u = e_u.to_vec();
    let mut final_i = e_i.to_vec();

    let mut curr_u = e_u.to_vec();
    let mut curr_i = e_i.to_vec();

    for _ in 0..k_layers {
        // Next user embeddings: aggregate from items using thread-local fold/reduce
        // Each user row is independent, so we scatter directly with no contention
        let mut next_u = vec![0.0f32; n_users * k];
        next_u
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(u, out)| {
                let start = u_indptr[u] as usize;
                let end = u_indptr[u + 1] as usize;
                if start == end {
                    return;
                }
                let nu = inv_sqrt_u[u];
                for &i_raw in &u_indices[start..end] {
                    let i = i_raw as usize;
                    let alpha = nu * inv_sqrt_i[i];
                    let src = &curr_i[i * k..(i + 1) * k];
                    for f in 0..k {
                        unsafe { *out.get_unchecked_mut(f) += alpha * *src.get_unchecked(f); }
                    }
                }
            });

        // Next item embeddings: aggregate from users
        let mut next_i = vec![0.0f32; n_items * k];
        next_i
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, out)| {
                let start = i_indptr[i] as usize;
                let end = i_indptr[i + 1] as usize;
                if start == end {
                    return;
                }
                let ni = inv_sqrt_i[i];
                for &u_raw in &i_indices[start..end] {
                    let u = u_raw as usize;
                    let alpha = ni * inv_sqrt_u[u];
                    let src = &curr_u[u * k..(u + 1) * k];
                    for f in 0..k {
                        unsafe { *out.get_unchecked_mut(f) += alpha * *src.get_unchecked(f); }
                    }
                }
            });

        // Accumulate
        for idx in 0..final_u.len() {
            final_u[idx] += next_u[idx];
        }
        for idx in 0..final_i.len() {
            final_i[idx] += next_i[idx];
        }

        curr_u = next_u;
        curr_i = next_i;
    }

    // Average
    let scale = 1.0_f32 / (k_layers as f32 + 1.0);
    for v in final_u.iter_mut() {
        *v *= scale;
    }
    for v in final_i.iter_mut() {
        *v *= scale;
    }

    (final_u, final_i)
}

// ---- BPR loss + Adam update ------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn lightgcn_train(
    u_indptr: &[i64],
    u_indices: &[i32],
    i_indptr: &[i64],
    i_indices: &[i32],
    n_users: usize,
    n_items: usize,
    k: usize,
    k_layers: usize,
    learning_rate: f32,
    lambda: f32,
    ssl_ratio: f32,
    ssl_temp: f32,
    ssl_weight: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> (Vec<f32>, Vec<f32>) {
    // Build degree vectors
    let d_u: Vec<f32> = (0..n_users)
        .map(|u| (u_indptr[u + 1] - u_indptr[u]) as f32)
        .collect();
    let d_i: Vec<f32> = (0..n_items)
        .map(|i| (i_indptr[i + 1] - i_indptr[i]) as f32)
        .collect();

    // Learnable base embeddings
    let mut e_u = random_factors(n_users, k, seed);
    let mut e_i = random_factors(n_items, k, seed.wrapping_add(1));

    // Adam state
    let mut m_u = vec![0.0_f32; n_users * k];
    let mut v_u = vec![0.0_f32; n_users * k];
    let mut m_i = vec![0.0_f32; n_items * k];
    let mut v_i = vec![0.0_f32; n_items * k];
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1e-8_f32;

    let num_threads = rayon::current_num_threads();
    let n_interactions = *u_indptr.last().unwrap() as usize;

    if verbose {
        println!("LightGCN | users={n_users} items={n_items} k={k} layers={k_layers}");
        println!("ITER |    LOSS | TIME");
    }

    for iter in 0..iterations {
        let t_iter = std::time::Instant::now();

        // Forward propagation to get final embeddings
        let (final_u, final_i) = propagate(
            &e_u, &e_i, k, n_users, n_items,
            u_indptr, u_indices, i_indptr, i_indices,
            &d_u, &d_i, k_layers,
        );

        // Compute BPR gradients: user grads are race-free (disjoint user ranges),
        // item grads use fold/reduce to avoid CAS contention
        let mut g_fu = vec![0.0f32; n_users * k];
        let g_fu_ptr = g_fu.as_mut_ptr() as usize;

        let chunk = (n_users / num_threads).max(1);
        let (total_loss, g_fi): (f32, Vec<f32>) = (0..num_threads)
            .into_par_iter()
            .map(|tid| {
                let mut rng = XorShift64::new(
                    seed.wrapping_add(iter as u64).wrapping_add(tid as u64 * 997),
                );
                let mut loss = 0.0_f32;
                let mut local_gi = vec![0.0f32; n_items * k];
                let u_start = tid * chunk;
                let u_end = ((tid + 1) * chunk).min(n_users);
                for u in u_start..u_end {
                    let s = u_indptr[u] as usize;
                    let e = u_indptr[u + 1] as usize;
                    if s == e {
                        continue;
                    }
                    let pos_off = rng.next_usize(e - s);
                    let i = u_indices[s + pos_off] as usize;

                    let mut j = rng.next_usize(n_items);
                    for _ in 0..10 {
                        if u_indices[s..e].binary_search(&(j as i32)).is_err() {
                            break;
                        }
                        j = rng.next_usize(n_items);
                    }

                    // SIMD dot product for scores
                    let u_slice = &final_u[u * k..(u + 1) * k];
                    let y_i = dot(u_slice, &final_i[i * k..(i + 1) * k]);
                    let y_j = dot(u_slice, &final_i[j * k..(j + 1) * k]);

                    let diff = y_i - y_j;
                    let log_sig = if diff >= 0.0 {
                        let e = (-diff).exp();
                        -(1.0 + e).ln()
                    } else {
                        diff - (1.0 + diff.exp()).ln()
                    };
                    loss -= log_sig;

                    let sig = if diff >= 0.0 {
                        let e = (-diff).exp();
                        1.0 / (1.0 + e)
                    } else {
                        let e = diff.exp();
                        e / (1.0 + e)
                    };
                    let g = sig - 1.0;

                    // User grads: no contention (disjoint user ranges)
                    let gu_ptr = g_fu_ptr as *mut f32;
                    for f in 0..k {
                        let ei = final_i[i * k + f];
                        let ej = final_i[j * k + f];
                        unsafe { *gu_ptr.add(u * k + f) += g * (ei - ej); }
                    }
                    // Item grads: thread-local accumulation
                    for f in 0..k {
                        let eu = final_u[u * k + f];
                        local_gi[i * k + f] += g * eu;
                        local_gi[j * k + f] -= g * eu;
                    }
                }
                (loss, local_gi)
            })
            .reduce(
                || (0.0f32, vec![0.0f32; n_items * k]),
                |(l1, mut g1), (l2, g2)| {
                    for (a, b) in g1.iter_mut().zip(g2.iter()) {
                        *a += b;
                    }
                    (l1 + l2, g1)
                },
            );

        // SSL Contrastive Loss and Gradients
        let mut g_eu_ssl = vec![0.0f32; n_users * k];
        let mut g_ei_ssl = vec![0.0f32; n_items * k];
        let mut iter_ssl_loss = 0.0f32;

        if ssl_ratio > 0.0 && ssl_weight > 0.0 {
            // View 1
            let (v1_u_indptr, v1_u_indices, v1_i_indptr, v1_i_indices, v1_du, v1_di) = 
                edge_dropout(u_indptr, u_indices, n_users, n_items, ssl_ratio, seed.wrapping_add(iter as u64 * 3));
            let (v1_u, v1_i) = propagate(
                &e_u, &e_i, k, n_users, n_items,
                &v1_u_indptr, &v1_u_indices, &v1_i_indptr, &v1_i_indices,
                &v1_du, &v1_di, k_layers,
            );

            // View 2
            let (v2_u_indptr, v2_u_indices, v2_i_indptr, v2_i_indices, v2_du, v2_di) = 
                edge_dropout(u_indptr, u_indices, n_users, n_items, ssl_ratio, seed.wrapping_add(iter as u64 * 7));
            let (v2_u, v2_i) = propagate(
                &e_u, &e_i, k, n_users, n_items,
                &v2_u_indptr, &v2_u_indices, &v2_i_indptr, &v2_i_indices,
                &v2_du, &v2_di, k_layers,
            );

            let chunk_size = 512;
            let (lu, g1_u, g2_u) = compute_infonce_loss(&v1_u, &v2_u, k, n_users, ssl_temp, chunk_size);
            let (li, g1_i, g2_i) = compute_infonce_loss(&v1_i, &v2_i, k, n_items, ssl_temp, chunk_size);
            
            iter_ssl_loss = (lu + li) * ssl_weight;

            let (g_eu_v1, g_ei_v1) = propagate(
                &g1_u, &g1_i, k, n_users, n_items,
                &v1_u_indptr, &v1_u_indices, &v1_i_indptr, &v1_i_indices,
                &v1_du, &v1_di, k_layers,
            );
            
            let (g_eu_v2, g_ei_v2) = propagate(
                &g2_u, &g2_i, k, n_users, n_items,
                &v2_u_indptr, &v2_u_indices, &v2_i_indptr, &v2_i_indices,
                &v2_du, &v2_di, k_layers,
            );

            for idx in 0..g_eu_ssl.len() {
                g_eu_ssl[idx] = (g_eu_v1[idx] + g_eu_v2[idx]) * ssl_weight;
            }
            for idx in 0..g_ei_ssl.len() {
                g_ei_ssl[idx] = (g_ei_v1[idx] + g_ei_v2[idx]) * ssl_weight;
            }
        }

        // Back-propagate through graph layers to get BPR gradient w.r.t. e_u, e_i
        let (g_eu, g_ei) = propagate(
            &g_fu, &g_fi, k, n_users, n_items,
            u_indptr, u_indices, i_indptr, i_indices,
            &d_u, &d_i, k_layers,
        );

        // Adam update
        let t = (iter + 1) as f32;
        let alpha = learning_rate * (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));

        for idx in 0..e_u.len() {
            let g = g_eu[idx] + g_eu_ssl[idx] + lambda * e_u[idx];
            m_u[idx] = beta1 * m_u[idx] + (1.0 - beta1) * g;
            v_u[idx] = beta2 * v_u[idx] + (1.0 - beta2) * g * g;
            e_u[idx] -= alpha * m_u[idx] / (v_u[idx].sqrt() + eps);
        }
        for idx in 0..e_i.len() {
            let g = g_ei[idx] + g_ei_ssl[idx] + lambda * e_i[idx];
            m_i[idx] = beta1 * m_i[idx] + (1.0 - beta1) * g;
            v_i[idx] = beta2 * v_i[idx] + (1.0 - beta2) * g * g;
            e_i[idx] -= alpha * m_i[idx] / (v_i[idx].sqrt() + eps);
        }

        if verbose {
            let bpr_loss_avg = total_loss / n_interactions as f32;
            if ssl_ratio > 0.0 {
                println!(
                    "{:>4} | BPR: {:>6.4} SSL: {:>6.4} | {:.2}s",
                    iter + 1,
                    bpr_loss_avg,
                    iter_ssl_loss / (n_users + n_items) as f32,
                    t_iter.elapsed().as_secs_f64()
                );
            } else {
                println!(
                    "{:>4} | {:>7.4} | {:.2}s",
                    iter + 1,
                    bpr_loss_avg,
                    t_iter.elapsed().as_secs_f64()
                );
            }
        }
    }

    (e_u, e_i)
}

// ---- PyO3 entry point ------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    u_indptr, u_indices, i_indptr, i_indices,
    n_users, n_items, factors, k_layers,
    learning_rate, lambda_, ssl_ratio, ssl_temp, ssl_weight, iterations, seed, verbose
))]
#[allow(clippy::too_many_arguments)]
pub fn lightgcn_fit<'py>(
    py: Python<'py>,
    u_indptr: PyReadonlyArray1<'py, i64>,
    u_indices: PyReadonlyArray1<'py, i32>,
    i_indptr: PyReadonlyArray1<'py, i64>,
    i_indices: PyReadonlyArray1<'py, i32>,
    n_users: usize,
    n_items: usize,
    factors: usize,
    k_layers: usize,
    learning_rate: f32,
    lambda_: f32,
    ssl_ratio: f32,
    ssl_temp: f32,
    ssl_weight: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let up = u_indptr.as_slice()?;
    let ui = u_indices.as_slice()?;
    let ip = i_indptr.as_slice()?;
    let ii = i_indices.as_slice()?;

    let (eu, ei) = py.detach(|| {
        lightgcn_train(
            up, ui, ip, ii,
            n_users, n_items, factors, k_layers,
            learning_rate, lambda_, ssl_ratio, ssl_temp, ssl_weight, iterations, seed, verbose,
        )
    });

    let ua = PyArray1::from_vec(py, eu);
    let ia = PyArray1::from_vec(py, ei);

    Ok((
        ua.reshape([n_users, factors])?.into(),
        ia.reshape([n_items, factors])?.into(),
    ))
}
