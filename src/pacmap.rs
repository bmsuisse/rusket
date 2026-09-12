//! PaCMAP: Pairwise Controlled Manifold Approximation Projection.
//!
//! State-of-the-art non-linear dimensionality reduction that preserves
//! both local and global structure via a tri-component loss with dynamic
//! phase-based weighting. Uses Adam optimizer for fast convergence.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::nn_descent;

// ── XorShift RNG ────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xCAFE_BABE_DEAD_BEEF } else { seed })
    }
    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }
    #[inline(always)]
    fn usize(&mut self, max: usize) -> usize {
        (self.next() % max as u64) as usize
    }
    #[inline(always)]
    fn f32(&mut self) -> f32 {
        (self.next() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }
}

// ── Distance computation ────────────────────────────────────────────

/// Squared Euclidean distance in the LOW-dimensional embedding (2D/3D).
#[inline(always)]
fn embed_dist_sq(y: &[f32], i: usize, j: usize, nc: usize) -> f32 {
    let a = i * nc;
    let b = j * nc;
    let mut sum = 0.0f32;
    for c in 0..nc {
        let d = y[a + c] - y[b + c];
        sum += d * d;
    }
    sum
}

/// Squared Euclidean distance in high-dimensional space (SIMD-friendly 8-wide).
#[inline(always)]
fn hd_dist_sq(data: &[f32], i: usize, j: usize, d: usize) -> f32 {
    let a = &data[i * d..(i + 1) * d];
    let b = &data[j * d..(j + 1) * d];
    let mut sum = 0.0f32;
    // Fixed-size (8) array chunks: bounds are known at compile time, so LLVM
    // can auto-vectorize without per-element bounds checks. Same 8-accumulator
    // summation order as before, so results stay bit-identical.
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ca, cb) in a_chunks.zip(b_chunks) {
        let ca: [f32; 8] = ca.try_into().unwrap();
        let cb: [f32; 8] = cb.try_into().unwrap();
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        let d4 = ca[4] - cb[4];
        let d5 = ca[5] - cb[5];
        let d6 = ca[6] - cb[6];
        let d7 = ca[7] - cb[7];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
             + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
    }

    for (x, y) in a_rem.iter().zip(b_rem.iter()) {
        let d_val = x - y;
        sum += d_val * d_val;
    }
    sum
}

// ── Pair sampling ───────────────────────────────────────────────────

/// Near pairs: the k-nearest neighbors (from NN-Descent graph).
fn sample_near_pairs(knn_indices: &[u32], n: usize, k: usize) -> Vec<(u32, u32)> {
    let mut pairs = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            let neighbor = knn_indices[i * k + j];
            if neighbor != u32::MAX {
                pairs.push((i as u32, neighbor));
            }
        }
    }
    pairs
}

/// Mid-near pairs: for each point, sample `n_mn_per_point` random points
/// from the 6th-to-kth-nearest distance range (PaCMAP paper strategy).
fn sample_mid_near_pairs(
    data: &[f32],
    n: usize,
    d: usize,
    knn_distances: &[f32],
    k: usize,
    mn_per_point: usize,
    seed: u64,
) -> Vec<(u32, u32)> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(seed.wrapping_add(i as u64 * 2654435761));
            let mut local_pairs = Vec::with_capacity(mn_per_point);

            // Use the k-th nearest neighbor distance as reference scale
            let far_nn_dist = knn_distances[i * k + k - 1].max(1e-10);

            // Sample random points, keep those that are "mid-near":
            // distance roughly between 2× and 20× the k-th neighbor
            let lo = 2.0 * far_nn_dist;
            let hi = 20.0 * far_nn_dist;

            let max_attempts = mn_per_point * 30;
            let mut attempts = 0;
            while local_pairs.len() < mn_per_point && attempts < max_attempts {
                let j = rng.usize(n);
                if j != i {
                    let dist = hd_dist_sq(data, i, j, d);
                    if dist >= lo && dist <= hi {
                        local_pairs.push((i as u32, j as u32));
                    }
                }
                attempts += 1;
            }

            // Fall back: if not enough mid-near found, pick random non-neighbors
            while local_pairs.len() < mn_per_point {
                let j = rng.usize(n);
                if j != i {
                    local_pairs.push((i as u32, j as u32));
                }
            }

            local_pairs
        })
        .flatten()
        .collect()
}

/// Further pairs: random pairs for repulsion (global structure).
fn sample_further_pairs(n: usize, fp_per_point: usize, seed: u64) -> Vec<(u32, u32)> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(seed.wrapping_add(i as u64 * 6364136223846793005 + 1));
            let mut local_pairs = Vec::with_capacity(fp_per_point);
            for _ in 0..fp_per_point {
                let j = rng.usize(n);
                if j != i {
                    local_pairs.push((i as u32, j as u32));
                }
            }
            local_pairs
        })
        .flatten()
        .collect()
}

// ── PCA initialisation ──────────────────────────────────────────────

/// Initialize embedding with PCA projection.
///
/// The PaCMAP paper initializes with PCA and uses a scale of ~0.01×std.
/// We scale so the initial embedding has std ≈ 1e-4 per component, matching
/// the original PaCMAP reference implementation.
fn pca_init(
    data: &[f32],
    n: usize,
    d: usize,
    n_components: usize,
) -> Vec<f32> {
    let mean: Vec<f32> = {
        let mut m = vec![0.0f64; d];
        for row in data.chunks_exact(d) {
            for j in 0..d {
                m[j] += row[j] as f64;
            }
        }
        let inv_n = 1.0 / n as f64;
        m.into_iter().map(|v| (v * inv_n) as f32).collect()
    };

    let k = n_components.min(n).min(d);

    if d <= 500 {
        // Covariance eigendecomposition
        let mut cov = vec![0.0f32; d * d];
        for row in data.chunks_exact(d) {
            for i in 0..d {
                let diff_i = row[i] - mean[i];
                if diff_i == 0.0 { continue; }
                for j in i..d {
                    cov[i * d + j] += diff_i * (row[j] - mean[j]);
                }
            }
        }
        for i in 0..d {
            for j in 0..i {
                cov[i * d + j] = cov[j * d + i];
            }
        }

        let cov_mat = faer::Mat::<f32>::from_fn(d, d, |r, c| cov[r * d + c]);
        let svd = cov_mat.thin_svd().expect("PCA init SVD failed");
        let v = svd.V();

        // Project
        let mut embedding = vec![0.0f32; n * k];
        embedding
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, row)| {
                for c in 0..k {
                    let mut val = 0.0f32;
                    for j in 0..d {
                        val += (data[i * d + j] - mean[j]) * v[(j, c)];
                    }
                    row[c] = val;
                }
            });

        // Scale: PaCMAP reference uses 0.0001 * data_range as init scale
        // We normalize to have std ≈ 1e-4 × data_scale
        for c in 0..k {
            let mut var_sum = 0.0f64;
            for i in 0..n {
                var_sum += (embedding[i * k + c] as f64).powi(2);
            }
            let std = ((var_sum / n as f64).sqrt()) as f32;
            if std > 1e-10 {
                let scale = 1e-4 / std;
                for i in 0..n {
                    embedding[i * k + c] *= scale;
                }
            }
        }

        embedding
    } else {
        let mut rng = Rng::new(42);
        (0..n * k).map(|_| (rng.f32() - 0.5) * 2e-4).collect()
    }
}

// ── Adam optimizer state ────────────────────────────────────────────

struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: u32,
}

impl AdamState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-7,
            t: 0,
        }
    }

    fn step(&mut self, grad: &[f32], lr: f32, params: &mut [f32]) {
        self.t += 1;
        let t = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        let lr_t = lr * bc2.sqrt() / bc1;
        let (beta1, beta2, eps) = (self.beta1, self.beta2, self.eps);

        // ponytail: parallel Adam update — each index is independent, so a
        // plain par_iter_mut over the zipped (m, v, params, grad) removes the
        // serial pass with no reduction needed.
        self.m
            .par_iter_mut()
            .zip(self.v.par_iter_mut())
            .zip(params.par_iter_mut())
            .zip(grad.par_iter())
            .for_each(|(((m, v), p), g)| {
                *m = beta1 * *m + (1.0 - beta1) * g;
                *v = beta2 * *v + (1.0 - beta2) * g * g;
                *p -= lr_t * *m / (v.sqrt() + eps);
            });
    }
}

// ── Per-point pair adjacency (CSR-style, both directions) ──────────

/// Pair type tag used to select the loss formula in the gradient kernel.
const PAIR_NEAR: u8 = 0;
const PAIR_MN: u8 = 1;
const PAIR_FP: u8 = 2;

/// Build a per-point adjacency list from the three pair lists, with BOTH
/// directions of every pair represented (point `i` gets an entry for `j`
/// and point `j` gets an entry for `i`). This lets the gradient be computed
/// with each point owning exclusively its own output chunk — no write
/// contention, no reduction buffers — while reproducing exactly the same
/// per-term contributions the serial loop computed for `i` and for `j`.
///
/// Why this is equivalent to the serial pass: for a pair (i, j) the serial
/// code computes `coeff` from `d2(i, j)` (symmetric in i/j) and does
/// `grad[i] += coeff * (y_i - y_j)` and `grad[j] -= coeff * (y_i - y_j)`
/// i.e. `grad[j] += coeff * (y_j - y_i)`. Both updates have the same shape:
/// "the owning point's contribution is `coeff * (y_self - y_other)`". So
/// storing `(other, type)` in each point's own adjacency list and evaluating
/// that same formula from each point's own perspective reproduces the exact
/// per-pair terms — only the summation order changes (grouped by point
/// instead of pair-array order), which is the ~1e-6 float noise the task
/// description calls out.
fn build_adjacency(
    n: usize,
    near_pairs: &[(u32, u32)],
    mn_pairs: &[(u32, u32)],
    fp_pairs: &[(u32, u32)],
) -> (Vec<u32>, Vec<(u32, u8)>) {
    let groups: [(&[(u32, u32)], u8); 3] =
        [(near_pairs, PAIR_NEAR), (mn_pairs, PAIR_MN), (fp_pairs, PAIR_FP)];

    let mut degree = vec![0u32; n];
    for (pairs, _) in groups.iter() {
        for &(i, j) in *pairs {
            degree[i as usize] += 1;
            degree[j as usize] += 1;
        }
    }

    let mut offsets = vec![0u32; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + degree[i];
    }

    let mut adj: Vec<(u32, u8)> = vec![(0u32, 0u8); offsets[n] as usize];
    let mut cursor = offsets.clone();
    for (pairs, tag) in groups.iter() {
        for &(i, j) in *pairs {
            let (ii, jj) = (i as usize, j as usize);
            adj[cursor[ii] as usize] = (j, *tag);
            cursor[ii] += 1;
            adj[cursor[jj] as usize] = (i, *tag);
            cursor[jj] += 1;
        }
    }

    (offsets, adj)
}

// ── PaCMAP optimization ─────────────────────────────────────────────

/// Core PaCMAP fitting routine.
///
/// Three-phase optimization with dynamic attraction/repulsion balance:
/// Phase 1: Mid-near dominant → build global skeleton
/// Phase 2: Near dominant → tighten local clusters
/// Phase 3: Fine-tuning both local and global
fn pacmap_optimize(
    embedding: &mut [f32],
    n: usize,
    nc: usize,
    near_pairs: &[(u32, u32)],
    mn_pairs: &[(u32, u32)],
    fp_pairs: &[(u32, u32)],
    n_iters: usize,
    lr: f32,
) {
    let total = n * nc;
    let mut adam = AdamState::new(total);

    // Phase boundaries per PaCMAP paper
    let phase1_end = (n_iters as f32 * 0.22) as usize; // ~100/450
    let phase2_end = (n_iters as f32 * 0.50) as usize; // ~225/450

    let inv_near = if !near_pairs.is_empty() { 1.0 / near_pairs.len() as f32 } else { 0.0 };
    let inv_mn = if !mn_pairs.is_empty() { 1.0 / mn_pairs.len() as f32 } else { 0.0 };
    let inv_fp = if !fp_pairs.is_empty() { 1.0 / fp_pairs.len() as f32 } else { 0.0 };

    // ponytail: build the per-point (both-directions) adjacency once, outside
    // the iteration loop, instead of re-deriving anything from the pair lists
    // every iteration. See build_adjacency's doc comment for why this
    // reproduces the serial per-pair contributions exactly (just reordered).
    let (offsets, adj) = build_adjacency(n, near_pairs, mn_pairs, fp_pairs);

    // ponytail: hoist the gradient buffer above the loop; reused via fill(0.0)
    // instead of reallocating `total` floats every iteration.
    let mut grad = vec![0.0f32; total];

    for iter in 0..n_iters {
        // Dynamic per-phase weights (from PaCMAP paper, Table 2)
        let (w_near, w_mn, w_fp) = if iter < phase1_end {
            // Phase 1: Mid-near dominant
            let t = iter as f32 / phase1_end.max(1) as f32;
            (2.0, 1000.0 * (1.0 - t) + 3.0 * t, 1.0)
        } else if iter < phase2_end {
            // Phase 2: near dominant
            (3.0, 3.0, 1.0)
        } else {
            // Phase 3: fine-tuning
            (1.0, 3.0, 1.0)
        };

        grad.fill(0.0);

        // Each point sums its own attractive (near/mid-near) and repulsive
        // (further) terms into its own chunk — disjoint chunks, so no write
        // contention and no per-thread reduction buffers are needed.
        grad.par_chunks_mut(nc).enumerate().for_each(|(p, chunk)| {
            let start = offsets[p] as usize;
            let end = offsets[p + 1] as usize;
            for &(q, tag) in &adj[start..end] {
                let q = q as usize;
                let d2 = embed_dist_sq(embedding, p, q, nc);
                let coeff = match tag {
                    PAIR_NEAR => {
                        // L_near = d²/(10 + d²)
                        let denom = 10.0 + d2;
                        w_near * inv_near * 20.0 / (denom * denom)
                    }
                    PAIR_MN => {
                        // L_mn = d²/(10000 + d²)
                        let denom = 10000.0 + d2;
                        w_mn * inv_mn * 20000.0 / (denom * denom)
                    }
                    _ => {
                        // L_fp = 1/(1 + d²)
                        let denom = 1.0 + d2;
                        w_fp * inv_fp * -2.0 / (denom * denom)
                    }
                };
                for c in 0..nc {
                    chunk[c] += coeff * (embedding[p * nc + c] - embedding[q * nc + c]);
                }
            }
        });

        // Adam update (no aggressive clipping — let Adam handle it)
        adam.step(&grad, lr, embedding);
    }
}

// ── Python binding ──────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (data, n_components=2, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, num_iters=450, lr=1.0, seed=42))]
pub fn pacmap_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    n_components: usize,
    n_neighbors: usize,
    mn_ratio: f32,
    fp_ratio: f32,
    num_iters: usize,
    lr: f32,
    seed: u64,
) -> PyResult<Py<PyArray2<f32>>> {
    let shape = data.shape();
    let n = shape[0];
    let d = shape[1];

    if n < 3 {
        return Err(PyValueError::new_err(
            "PaCMAP requires at least 3 data points.",
        ));
    }
    if n_components == 0 {
        return Err(PyValueError::new_err(
            "n_components must be >= 1.",
        ));
    }

    let data_slice = data.as_slice().map_err(|_| {
        PyValueError::new_err("Input array must be C-contiguous.")
    })?;

    let k = n_neighbors.min(n - 1);

    // 1. Build k-NN graph using NN-Descent
    let (knn_indices, knn_distances) =
        nn_descent::nn_descent_inner(data_slice, n, d, k, 12, 0.001, seed);

    // 2. Sample three types of pairs (per-point counts from PaCMAP paper)
    let mn_per_point = ((k as f32 * mn_ratio) as usize).max(1);
    let fp_per_point = ((k as f32 * fp_ratio) as usize).max(1);

    let near_pairs = sample_near_pairs(&knn_indices, n, k);
    let mn_pairs = sample_mid_near_pairs(
        data_slice, n, d, &knn_distances, k, mn_per_point, seed + 1,
    );
    let fp_pairs = sample_further_pairs(n, fp_per_point, seed + 2);

    // 3. Initialize embedding with PCA
    let mut embedding = pca_init(data_slice, n, d, n_components);

    // 4. Optimize with Adam
    pacmap_optimize(
        &mut embedding,
        n,
        n_components,
        &near_pairs,
        &mn_pairs,
        &fp_pairs,
        num_iters,
        lr,
    );

    // 5. Return
    let result = PyArray1::from_vec(py, embedding)
        .reshape([n, n_components])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .into_pyobject(py)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .unbind();

    Ok(result)
}
