use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

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

// ---- Atomic f32 accumulation for parallel gradient reduction ---------------

struct AtomicF32Buf {
    data: Vec<AtomicU32>,
}

impl AtomicF32Buf {
    fn zeros(n: usize) -> Self {
        let data = (0..n).map(|_| AtomicU32::new(0_f32.to_bits())).collect();
        Self { data }
    }

    #[inline(always)]
    fn add(&self, idx: usize, val: f32) {
        let atomic = &self.data[idx];
        let mut cur = atomic.load(Ordering::Relaxed);
        loop {
            let new = (f32::from_bits(cur) + val).to_bits();
            match atomic.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => cur = x,
            }
        }
    }

    fn as_slice_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|a| f32::from_bits(a.load(Ordering::Relaxed)))
            .collect()
    }
}

// ---- Graph propagation (LightGCN forward pass) ----------------------------
// Propagate embeddings through the bipartite graph for k_layers steps.
// A_tilde = D_u^{-0.5} A D_i^{-0.5}   (normalised adjacency)
// E^{l+1}_u = A_tilde * E^l_i, E^{l+1}_i = A_tilde^T * E^l_u
// Final = mean of all layers including E^0

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
    // Accumulate final_u / final_i = sum of all layer E^l
    let mut final_u = e_u.to_vec();
    let mut final_i = e_i.to_vec();

    let mut curr_u = e_u.to_vec();
    let mut curr_i = e_i.to_vec();

    for _ in 0..k_layers {
        // Next user embeddings: aggregate from items
        let next_u_atomic = AtomicF32Buf::zeros(n_users * k);
        u_indptr.par_windows(2).enumerate().for_each(|(u, w)| {
            let start = w[0] as usize;
            let end = w[1] as usize;
            if start == end {
                return;
            }
            let norm_u = d_u[u].sqrt();
            for &i_raw in &u_indices[start..end] {
                let i = i_raw as usize;
                let alpha = 1.0_f32 / (norm_u * d_i[i].sqrt());
                for f in 0..k {
                    next_u_atomic.add(u * k + f, alpha * curr_i[i * k + f]);
                }
            }
        });
        let next_u = next_u_atomic.as_slice_f32();

        // Next item embeddings: aggregate from users
        let next_i_atomic = AtomicF32Buf::zeros(n_items * k);
        i_indptr.par_windows(2).enumerate().for_each(|(i, w)| {
            let start = w[0] as usize;
            let end = w[1] as usize;
            if start == end {
                return;
            }
            let norm_i = d_i[i].sqrt();
            for &u_raw in &i_indices[start..end] {
                let u = u_raw as usize;
                let alpha = 1.0_f32 / (norm_i * d_u[u].sqrt());
                for f in 0..k {
                    next_i_atomic.add(i * k + f, alpha * curr_u[u * k + f]);
                }
            }
        });
        let next_i = next_i_atomic.as_slice_f32();

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
fn lightgcn_train(
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

        // Compute BPR gradients in parallel
        let g_final_u_atomic = AtomicF32Buf::zeros(n_users * k);
        let g_final_i_atomic = AtomicF32Buf::zeros(n_items * k);

        let total_loss: f32 = {
            let chunk = (n_users / num_threads).max(1);
            (0..num_threads)
                .into_par_iter()
                .map(|tid| {
                    let mut rng = XorShift64::new(
                        seed.wrapping_add(iter as u64).wrapping_add(tid as u64 * 997),
                    );
                    let mut loss = 0.0_f32;
                    let u_start = tid * chunk;
                    let u_end = ((tid + 1) * chunk).min(n_users);
                    for u in u_start..u_end {
                        let s = u_indptr[u] as usize;
                        let e = u_indptr[u + 1] as usize;
                        if s == e {
                            continue;
                        }
                        // Pick a random positive item
                        let pos_off = rng.next_usize(e - s);
                        let i = u_indices[s + pos_off] as usize;

                        // Sample a negative item
                        let mut j = rng.next_usize(n_items);
                        for _ in 0..10 {
                            if u_indices[s..e].binary_search(&(j as i32)).is_err() {
                                break;
                            }
                            j = rng.next_usize(n_items);
                        }

                        // Scores
                        let mut y_i = 0.0_f32;
                        let mut y_j = 0.0_f32;
                        for f in 0..k {
                            y_i += final_u[u * k + f] * final_i[i * k + f];
                            y_j += final_u[u * k + f] * final_i[j * k + f];
                        }

                        let diff = y_i - y_j;
                        // log sigmoid loss
                        let log_sig = if diff >= 0.0 {
                            let e = (-diff).exp();
                            -(1.0 + e).ln()
                        } else {
                            diff - (1.0 + diff.exp()).ln()
                        };
                        loss -= log_sig;

                        // Gradient of log(sigmoid(diff)) w.r.t. embeddings
                        let sig = if diff >= 0.0 {
                            let e = (-diff).exp();
                            1.0 / (1.0 + e)
                        } else {
                            let e = diff.exp();
                            e / (1.0 + e)
                        };
                        let g = sig - 1.0; // derivative of -log(sigmoid)

                        for f in 0..k {
                            let eu = final_u[u * k + f];
                            let ei = final_i[i * k + f];
                            let ej = final_i[j * k + f];
                            g_final_u_atomic.add(u * k + f, g * (ei - ej));
                            g_final_i_atomic.add(i * k + f, g * eu);
                            g_final_i_atomic.add(j * k + f, -g * eu);
                        }
                    }
                    loss
                })
                .sum()
        };

        let g_fu = g_final_u_atomic.as_slice_f32();
        let g_fi = g_final_i_atomic.as_slice_f32();

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
