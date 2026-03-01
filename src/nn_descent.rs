//! NN-Descent: blazing-fast approximate k-NN graph construction.
//!
//! Implements the Dong et al. (2011) algorithm: "a neighbor of a neighbor
//! is likely also a neighbor". Converges in O(N^1.14) empirically.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

// ── Distance primitives ─────────────────────────────────────────────────

/// Squared Euclidean distance — kept branchless and auto-vectorisable.
#[inline(always)]
fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    // Process in chunks of 8 for SIMD-friendly auto-vectorization
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        let d4 = a[base + 4] - b[base + 4];
        let d5 = a[base + 5] - b[base + 5];
        let d6 = a[base + 6] - b[base + 6];
        let d7 = a[base + 7] - b[base + 7];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
             + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }
    sum
}

// ── Heap for top-k tracking ─────────────────────────────────────────────

/// A max-heap entry: (distance, index, is_new).
/// We keep a max-heap so the *worst* neighbor is at the top for easy eviction.
#[derive(Clone, Copy)]
struct HeapItem {
    dist: f32,
    idx: u32,
    is_new: bool,
}

/// Fixed-capacity max-heap for maintaining k nearest neighbors for one point.
#[derive(Clone)]
struct KnnHeap {
    items: Vec<HeapItem>,
    k: usize,
}

impl KnnHeap {
    fn new(k: usize) -> Self {
        Self {
            items: Vec::with_capacity(k),
            k,
        }
    }

    /// Current worst (largest) distance in the heap.
    #[inline]
    fn worst_dist(&self) -> f32 {
        if self.items.is_empty() {
            f32::INFINITY
        } else {
            self.items[0].dist
        }
    }

    /// Try to insert a candidate. Returns true if it was accepted.
    #[inline]
    fn insert(&mut self, dist: f32, idx: u32) -> bool {
        if dist >= self.worst_dist() && self.items.len() == self.k {
            return false;
        }
        // Check for duplicate
        for item in &self.items {
            if item.idx == idx {
                return false;
            }
        }

        if self.items.len() < self.k {
            self.items.push(HeapItem {
                dist,
                idx,
                is_new: true,
            });
            self.sift_up(self.items.len() - 1);
        } else {
            // Replace the root (worst)
            self.items[0] = HeapItem {
                dist,
                idx,
                is_new: true,
            };
            self.sift_down(0);
        }
        true
    }

    /// Extract sorted (ascending distance) list of neighbor indices.
    fn sorted_indices(&self) -> Vec<u32> {
        let mut pairs: Vec<(f32, u32)> = self.items.iter().map(|h| (h.dist, h.idx)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        pairs.into_iter().map(|(_, i)| i).collect()
    }

    /// Extract sorted distances.
    fn sorted_distances(&self) -> Vec<f32> {
        let mut pairs: Vec<(f32, u32)> = self.items.iter().map(|h| (h.dist, h.idx)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        pairs.into_iter().map(|(d, _)| d).collect()
    }

    /// Drain "new" flags and return (new_items, old_items).
    fn drain_new(&mut self) -> (Vec<u32>, Vec<u32>) {
        let mut new = Vec::new();
        let mut old = Vec::new();
        for item in &mut self.items {
            if item.is_new {
                new.push(item.idx);
                item.is_new = false;
            } else {
                old.push(item.idx);
            }
        }
        (new, old)
    }

    // ── standard binary max-heap operations ──

    #[inline]
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.items[idx].dist > self.items[parent].dist {
                self.items.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn sift_down(&mut self, mut idx: usize) {
        let len = self.items.len();
        loop {
            let mut largest = idx;
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            if left < len && self.items[left].dist > self.items[largest].dist {
                largest = left;
            }
            if right < len && self.items[right].dist > self.items[largest].dist {
                largest = right;
            }
            if largest == idx {
                break;
            }
            self.items.swap(idx, largest);
            idx = largest;
        }
    }
}

// ── XorShift RNG (fast, thread-safe when cloned per-thread) ─────────

struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed })
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
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next() % max as u64) as usize
    }
}

// ── NN-Descent core ─────────────────────────────────────────────────

/// Build an approximate k-NN graph using NN-Descent.
///
/// Returns (indices, distances) as flat row-major arrays of shape (n, k).
pub(crate) fn nn_descent_inner(
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    max_iters: usize,
    delta: f32,
    seed: u64,
) -> (Vec<u32>, Vec<f32>) {
    // 1. Random initialization: each point gets k random neighbors
    let mut heaps: Vec<KnnHeap> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = XorShift::new(seed.wrapping_add(i as u64 * 6364136223846793005));
            let mut heap = KnnHeap::new(k);
            let mut attempts = 0;
            while heap.items.len() < k && attempts < k * 10 {
                let j = rng.next_usize(n);
                if j != i {
                    let dist = dist_sq(
                        &data[i * d..(i + 1) * d],
                        &data[j * d..(j + 1) * d],
                    );
                    heap.insert(dist, j as u32);
                }
                attempts += 1;
            }
            heap
        })
        .collect();

    // 2. NN-Descent iterations
    let convergence_threshold = (delta * n as f32 * k as f32) as usize;

    for _iter in 0..max_iters {
        // 2a. Build reverse lists and separate new/old
        let mut new_fwd: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut old_fwd: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut new_rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut old_rev: Vec<Vec<u32>> = vec![Vec::new(); n];

        for i in 0..n {
            let (new_items, old_items) = heaps[i].drain_new();
            for &j in &new_items {
                new_fwd[i].push(j);
                new_rev[j as usize].push(i as u32);
            }
            for &j in &old_items {
                old_fwd[i].push(j);
                old_rev[j as usize].push(i as u32);
            }
        }

        // 2b. Local joins — the core of NN-Descent
        // For each point, consider all (new_union) × (new_union ∪ old_union) pairs
        // Collect updates in thread-local buffers, then merge
        let updates: Vec<Vec<(u32, f32, u32)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut local_updates = Vec::new();

                // Build union lists
                let mut new_union: Vec<u32> = Vec::new();
                new_union.extend_from_slice(&new_fwd[i]);
                new_union.extend_from_slice(&new_rev[i]);
                new_union.sort_unstable();
                new_union.dedup();

                let mut all_union: Vec<u32> = Vec::new();
                all_union.extend_from_slice(&new_union);
                all_union.extend_from_slice(&old_fwd[i]);
                all_union.extend_from_slice(&old_rev[i]);
                all_union.sort_unstable();
                all_union.dedup();

                // For each pair where at least one is "new"
                for ni in 0..new_union.len() {
                    let u = new_union[ni] as usize;
                    if u == i { continue; }
                    for ai in 0..all_union.len() {
                        let v = all_union[ai] as usize;
                        if v == i || v == u || u >= v { continue; }

                        let dist = dist_sq(
                            &data[u * d..(u + 1) * d],
                            &data[v * d..(v + 1) * d],
                        );

                        local_updates.push((u as u32, dist, v as u32));
                        local_updates.push((v as u32, dist, u as u32));
                    }
                }
                local_updates
            })
            .collect();

        // 2c. Apply updates
        let mut n_updates = 0usize;
        for batch in &updates {
            for &(point, dist, neighbor) in batch {
                if heaps[point as usize].insert(dist, neighbor) {
                    n_updates += 1;
                }
            }
        }

        // 2d. Check convergence
        if n_updates <= convergence_threshold {
            break;
        }
    }

    // 3. Extract results
    let mut indices = vec![0u32; n * k];
    let mut distances = vec![0.0f32; n * k];

    for i in 0..n {
        let sorted_idx = heaps[i].sorted_indices();
        let sorted_dist = heaps[i].sorted_distances();
        let actual_k = sorted_idx.len().min(k);
        for j in 0..actual_k {
            indices[i * k + j] = sorted_idx[j];
            distances[i * k + j] = sorted_dist[j];
        }
        // Pad with max values if we have fewer than k neighbors
        for j in actual_k..k {
            indices[i * k + j] = u32::MAX;
            distances[i * k + j] = f32::INFINITY;
        }
    }

    (indices, distances)
}

// ── Python binding ──────────────────────────────────────────────────

/// Build an approximate k-NN graph using NN-Descent.
///
/// Returns (indices, distances) as numpy arrays of shape (n, k).
#[pyfunction]
#[pyo3(signature = (data, k, max_iters=12, delta=0.001, seed=42))]
pub fn nn_descent_build<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    k: usize,
    max_iters: usize,
    delta: f32,
    seed: u64,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray2<f32>>)> {
    let shape = data.shape();
    let n = shape[0];
    let d = shape[1];

    if n < 2 {
        return Err(PyValueError::new_err("Need at least 2 data points."));
    }
    if k == 0 || k >= n {
        return Err(PyValueError::new_err(
            format!("k must be in [1, {}), got {}", n, k),
        ));
    }

    let data_slice = data.as_slice().map_err(|_| {
        PyValueError::new_err("Input array must be C-contiguous.")
    })?;

    let (indices, distances) = nn_descent_inner(data_slice, n, d, k, max_iters, delta, seed);

    let idx_np = PyArray1::from_vec(py, indices)
        .reshape([n, k])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .into_pyobject(py)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .unbind();

    let dist_np = PyArray1::from_vec(py, distances)
        .reshape([n, k])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .into_pyobject(py)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .unbind();

    Ok((idx_np, dist_np))
}
