use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Sparse CSR × dense matrix multiply: result = A (n×m sparse) × B (m×k dense)
/// Output: row-major flat Vec of length n*k
fn spmm_csr_dense(
    indptr: &[i64],
    indices: &[i32],
    data: &[f64],
    n_rows: usize,
    b: &[f64],
    b_cols: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; n_rows * b_cols];
    out.par_chunks_mut(b_cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for idx in start..end {
                let col = indices[idx] as usize;
                let val = data[idx];
                let b_row = &b[col * b_cols..(col + 1) * b_cols];
                for j in 0..b_cols {
                    out_row[j] += val * b_row[j];
                }
            }
        });
    out
}

/// Sparse CSR^T × dense matrix multiply: result = A^T (m×n sparse.T → n×m) × B (n×k dense)
/// i.e. columns of A dot rows of B.
/// Output: row-major flat Vec of length m*k
fn spmm_csrt_dense(
    indptr: &[i64],
    indices: &[i32],
    data: &[f64],
    n_cols: usize,
    b: &[f64],
    b_cols: usize,
) -> Vec<f64> {
    // Parallel accumulation: each thread gets a partial buffer
    let n_rows = indptr.len() - 1;
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n_rows + num_threads - 1) / num_threads;

    let partials: Vec<Vec<f64>> = (0..num_threads)
        .into_par_iter()
        .map(|t| {
            let mut local = vec![0.0f64; n_cols * b_cols];
            let row_start = t * chunk_size;
            let row_end = (row_start + chunk_size).min(n_rows);
            for row in row_start..row_end {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                let b_row = &b[row * b_cols..(row + 1) * b_cols];
                for idx in start..end {
                    let col = indices[idx] as usize;
                    let val = data[idx];
                    let local_row = &mut local[col * b_cols..(col + 1) * b_cols];
                    for j in 0..b_cols {
                        local_row[j] += val * b_row[j];
                    }
                }
            }
            local
        })
        .collect();

    // Sum partials
    let mut out = vec![0.0f64; n_cols * b_cols];
    for partial in &partials {
        for (o, p) in out.iter_mut().zip(partial.iter()) {
            *o += *p;
        }
    }
    out
}

/// Dense matrix multiply: C = A^T (k×n) × B (k×m) = out(n×m)
/// A is row-major (n_rows×n), B is row-major (n_rows×m)
/// Result: row-major (n×m)
fn dense_ata(a: &[f64], n_rows: usize, n: usize) -> Vec<f64> {
    // A^T A where A is (n_rows × n) → result is (n × n)
    let mut out = vec![0.0f64; n * n];
    // Use parallel rows
    out.par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, out_row)| {
            for j in 0..n {
                let mut s = 0.0f64;
                for r in 0..n_rows {
                    s += a[r * n + i] * a[r * n + j];
                }
                out_row[j] = s;
            }
        });
    out
}

/// Dense matmul: C(n×p) = A(n×m) × B(m×p), all row-major flat
fn dense_mm(a: &[f64], b: &[f64], n: usize, m: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * p];
    out.par_chunks_mut(p)
        .enumerate()
        .for_each(|(i, out_row)| {
            let a_row = &a[i * m..(i + 1) * m];
            for k in 0..m {
                let b_row = &b[k * p..(k + 1) * p];
                let a_val = a_row[k];
                for j in 0..p {
                    out_row[j] += a_val * b_row[j];
                }
            }
        });
    out
}

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

    fn next_float(&mut self) -> f64 {
        let v = self.next() & 0xFFFFFF;
        v as f64 / 0xFFFFFF as f64
    }
}

/// NMF via Multiplicative Update rules.
///
/// Decomposes R ≈ W × H where W(n_users×k), H(k×n_items).
/// Uses the standard Lee & Seung (2001) update rules with L2 regularization.
///
/// Returns (W as f32 user_factors, H^T as f32 item_factors)
pub(crate) fn nmf_train(
    indptr: &[i64],
    indices: &[i32],
    data: &[f64],
    n_users: usize,
    n_items: usize,
    k: usize,
    iterations: usize,
    regularization: f64,
    seed: u64,
    verbose: bool,
) -> (Vec<f32>, Vec<f32>) {
    let eps = 1e-12f64;

    let mut rng = XorShift64::new(seed);

    // Initialise W (n_users × k) and H (k × n_items) with small positive values
    let mut w = vec![0.0f64; n_users * k];
    for v in w.iter_mut() {
        *v = (rng.next_float() * 0.01 + eps).abs();
    }
    let mut h = vec![0.0f64; k * n_items];
    for v in h.iter_mut() {
        *v = (rng.next_float() * 0.01 + eps).abs();
    }

    let start_time = std::time::Instant::now();

    if verbose {
        println!("  NMF (Multiplicative Update)");
        println!("  Users: {}, Items: {}", n_users, n_items);
        println!("  Factors: {}, reg={}", k, regularization);
        println!("  ITER |  TIME");
        println!("  -----------------------");
    }

    for it in 0..iterations {
        let iter_start = std::time::Instant::now();

        // --- Update H ---
        // numerator = W^T × V  (k × n_items)
        // W is (n_users × k), V is sparse CSR (n_users × n_items)
        // W^T V = transpose of W (k × n_users) × V (n_users × n_items, sparse)
        // Equivalent to V^T W → transpose → but easier to do directly
        // W^T is (k × n_users), we need (k × n_items)
        // For each column j of V (each item), sum W[u,:] * V[u,j] for all u that rated j
        // Rather: iterate CSR rows and accumulate.

        // W^T V: for each user u, for each item j in u's row: h_num[:, j] += W[u, :] * val
        let wt_v = spmm_csrt_dense(indptr, indices, data, n_items, &w, k);
        // wt_v is (n_items × k) but we need (k × n_items)
        // Actually spmm_csrt_dense returns (n_cols × b_cols) = (n_items × k)
        // We need the transpose of that → (k × n_items)
        // Let's transpose it
        let mut numerator_h = vec![0.0f64; k * n_items];
        for item in 0..n_items {
            for f in 0..k {
                numerator_h[f * n_items + item] = wt_v[item * k + f];
            }
        }

        // denominator = W^T W × H + reg * H
        // W^T W is (k × k)
        let wtw = dense_ata(&w, n_users, k);
        // W^T W × H: (k×k) × (k×n_items) = (k×n_items)
        let wtw_h = dense_mm(&wtw, &h, k, k, n_items);

        // Apply update: H *= numerator / (denominator + eps)
        h.par_iter_mut()
            .enumerate()
            .for_each(|(idx, h_val)| {
                let denom = wtw_h[idx] + regularization * (*h_val) + eps;
                *h_val *= numerator_h[idx] / denom;
            });

        // --- Update W ---
        // numerator = V × H^T  (n_users × k)
        // V is sparse CSR (n_users × n_items), H is (k × n_items), H^T is (n_items × k)
        // We need V × H^T where H^T(item, f) = H(f, item) = h[f * n_items + item]
        // Build H^T explicitly
        let mut ht = vec![0.0f64; n_items * k];
        for f in 0..k {
            for item in 0..n_items {
                ht[item * k + f] = h[f * n_items + item];
            }
        }

        // V × H^T: sparse (n_users × n_items) × dense (n_items × k) = (n_users × k)
        let numerator_w = spmm_csr_dense(indptr, indices, data, n_users, &ht, k);

        // denominator = W × H × H^T + reg * W = W × (H H^T) + reg * W
        // H H^T: (k × n_items) × (n_items × k) = (k × k)
        let hht = dense_mm(&h, &ht, k, n_items, k);
        // W × HH^T: (n_users × k) × (k × k) = (n_users × k)
        let w_hht = dense_mm(&w, &hht, n_users, k, k);

        // Apply update: W *= numerator / (denominator + eps)
        w.par_iter_mut()
            .enumerate()
            .for_each(|(idx, w_val)| {
                let denom = w_hht[idx] + regularization * (*w_val) + eps;
                *w_val *= numerator_w[idx] / denom;
            });

        if verbose {
            let print_interval = (iterations / 10).max(1);
            if (it + 1) % print_interval == 0 || it + 1 == iterations {
                println!(
                    "  {:>4} | {:>6.2}s",
                    it + 1,
                    iter_start.elapsed().as_secs_f64()
                );
            }
        }
    }

    if verbose {
        println!("  -----------------------");
        println!("  Total time: {:.1}s", start_time.elapsed().as_secs_f64());
    }

    // Convert to f32
    let user_factors: Vec<f32> = w.iter().map(|&v| v as f32).collect();
    // H is (k × n_items), we want item_factors as (n_items × k)
    let mut item_factors = vec![0.0f32; n_items * k];
    for f in 0..k {
        for item in 0..n_items {
            item_factors[item * k + f] = h[f * n_items + item] as f32;
        }
    }

    (user_factors, item_factors)
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_users, n_items, factors, iterations, regularization, seed, verbose))]
pub fn nmf_fit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f64>,
    n_users: usize,
    n_items: usize,
    factors: usize,
    iterations: usize,
    regularization: f64,
    seed: u64,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let dt = data.as_slice()?;

    let (uf, itf) = py.detach(|| {
        nmf_train(ip, ix, dt, n_users, n_items, factors, iterations, regularization, seed, verbose)
    });

    let ua = PyArray1::from_vec(py, uf);
    let ia = PyArray1::from_vec(py, itf);

    Ok((
        ua.reshape([n_users, factors])?.into(),
        ia.reshape([n_items, factors])?.into(),
    ))
}
