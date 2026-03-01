use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

// ---- Fast RNG (same pattern as bpr.rs) ------------------------------------

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
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }

    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        let v = self.next() & 0xFFFFFF;
        v as f32 / 0xFFFFFF as f32
    }
}

fn random_factors(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let scale = (1.0_f32 / k as f32).sqrt() * 0.1;
    let mut out = vec![0.0f32; n * k];
    for v in out.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * scale;
    }
    out
}
// ── SIMD-optimised dot product (8-wide unrolling for NEON / AVX2) ──────────
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    let chunks = k / 8;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut idx = 0;
    for _ in 0..chunks {
        unsafe {
            s0 += *a.get_unchecked(idx) * *b.get_unchecked(idx);
            s1 += *a.get_unchecked(idx + 1) * *b.get_unchecked(idx + 1);
            s2 += *a.get_unchecked(idx + 2) * *b.get_unchecked(idx + 2);
            s3 += *a.get_unchecked(idx + 3) * *b.get_unchecked(idx + 3);
            s4 += *a.get_unchecked(idx + 4) * *b.get_unchecked(idx + 4);
            s5 += *a.get_unchecked(idx + 5) * *b.get_unchecked(idx + 5);
            s6 += *a.get_unchecked(idx + 6) * *b.get_unchecked(idx + 6);
            s7 += *a.get_unchecked(idx + 7) * *b.get_unchecked(idx + 7);
        }
        idx += 8;
    }
    while idx < k {
        unsafe { s0 += *a.get_unchecked(idx) * *b.get_unchecked(idx); }
        idx += 1;
    }
    (s0 + s1 + s2 + s3) + (s4 + s5 + s6 + s7)
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

        // Back-propagate through graph layers to get gradient w.r.t. e_u, e_i
        let (g_eu, g_ei) = propagate(
            &g_fu, &g_fi, k, n_users, n_items,
            u_indptr, u_indices, i_indptr, i_indices,
            &d_u, &d_i, k_layers,
        );

        // Adam update
        let t = (iter + 1) as f32;
        let alpha = learning_rate * (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));

        for idx in 0..e_u.len() {
            let g = g_eu[idx] + lambda * e_u[idx];
            m_u[idx] = beta1 * m_u[idx] + (1.0 - beta1) * g;
            v_u[idx] = beta2 * v_u[idx] + (1.0 - beta2) * g * g;
            e_u[idx] -= alpha * m_u[idx] / (v_u[idx].sqrt() + eps);
        }
        for idx in 0..e_i.len() {
            let g = g_ei[idx] + lambda * e_i[idx];
            m_i[idx] = beta1 * m_i[idx] + (1.0 - beta1) * g;
            v_i[idx] = beta2 * v_i[idx] + (1.0 - beta2) * g * g;
            e_i[idx] -= alpha * m_i[idx] / (v_i[idx].sqrt() + eps);
        }

        if verbose {
            println!(
                "{:>4} | {:>7.4} | {:.2}s",
                iter + 1,
                total_loss / n_interactions as f32,
                t_iter.elapsed().as_secs_f64()
            );
        }
    }

    (e_u, e_i)
}

// ---- PyO3 entry point ------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    u_indptr, u_indices, i_indptr, i_indices,
    n_users, n_items, factors, k_layers,
    learning_rate, lambda_, iterations, seed, verbose
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
            learning_rate, lambda_, iterations, seed, verbose,
        )
    });

    let ua = PyArray1::from_vec(py, eu);
    let ia = PyArray1::from_vec(py, ei);

    Ok((
        ua.reshape([n_users, factors])?.into(),
        ia.reshape([n_items, factors])?.into(),
    ))
}
