use faer::{linalg::matmul::matmul, Accum, MatRef, Par};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Sparse CSR × dense matrix multiply: result = A (n×m sparse) × B (m×k dense)
/// Output: row-major flat Vec of length n*k
fn spmm_csr_dense(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_rows: usize,
    b: &[f32],
    b_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * b_cols];
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

/// Gram matrix: A^T A for row-major A (n_rows × k) → (k × k) row-major.
fn gram(a: &[f32], n_rows: usize, k: usize) -> Vec<f32> {
    let y = MatRef::from_row_major_slice(a, n_rows, k);
    let yt = y.transpose();
    let mut g = faer::Mat::<f32>::zeros(k, k);
    matmul(g.as_mut(), Accum::Replace, yt, y, 1.0f32, Par::rayon(0));
    let mut out = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = g[(i, j)];
        }
    }
    out
}

/// Dense matmul: C(n×p) = A(n×m) × B(m×p), all row-major flat.
fn mm(a: &[f32], n: usize, m: usize, b: &[f32], p: usize) -> Vec<f32> {
    let am = MatRef::from_row_major_slice(a, n, m);
    let bm = MatRef::from_row_major_slice(b, m, p);
    let mut out = vec![0.0f32; n * p];
    {
        let mut cm = faer::MatMut::from_row_major_slice_mut(&mut out, n, p);
        matmul(cm.as_mut(), Accum::Replace, am, bm, 1.0f32, Par::rayon(0));
    }
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

    fn next_float(&mut self) -> f32 {
        let v = self.next() & 0xFFFFFF;
        v as f32 / 0xFFFFFF as f32
    }
}

/// NMF via Multiplicative Update rules.
///
/// Decomposes R ≈ W × H where W(n_users×k), H(k×n_items).
/// Uses the standard Lee & Seung (2001) update rules with L2 regularization.
///
/// `h_t` is stored as H^T (n_items × k, row-major) throughout — this is the
/// canonical layout every consumer (numerator_h, numerator_w, output) wants,
/// so no transposes are needed per iteration.
///
/// Returns (W as f32 user_factors, H^T as f32 item_factors)
pub(crate) fn nmf_train(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_users: usize,
    n_items: usize,
    k: usize,
    iterations: usize,
    regularization: f32,
    seed: u64,
    verbose: bool,
) -> (Vec<f32>, Vec<f32>) {
    let eps = 1e-7f32;

    let mut rng = XorShift64::new(seed);

    // Initialise W (n_users × k) and H^T (n_items × k) with small positive values
    let mut w = vec![0.0f32; n_users * k];
    for v in w.iter_mut() {
        *v = (rng.next_float() * 0.01 + eps).abs();
    }
    let mut h_t = vec![0.0f32; n_items * k];
    for v in h_t.iter_mut() {
        *v = (rng.next_float() * 0.01 + eps).abs();
    }

    // Transpose the CSR matrix once, up front, instead of re-deriving V^T
    // (via per-thread partial buffers) on every iteration.
    let (t_indptr, t_indices, t_data) = crate::als::csr_transpose(indptr, indices, data, n_users, n_items);

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

        // --- Update H (stored as h_t = H^T, n_items × k) ---
        // numerator_h = V^T × W  (n_items × k) = W^T V transposed, computed
        // directly from the precomputed CSR^T so no per-iteration transpose
        // (of V or of the result) is needed.
        let numerator_h = spmm_csr_dense(&t_indptr, &t_indices, &t_data, n_items, &w, k);

        // denominator = (W^T W × H)^T = H^T × (W^T W), since W^T W is symmetric.
        let wtw = gram(&w, n_users, k);
        let denom_h = mm(&h_t, n_items, k, &wtw, k);

        // Apply update: H^T *= numerator / (denominator + eps)
        h_t.par_iter_mut()
            .zip(numerator_h.par_iter())
            .zip(denom_h.par_iter())
            .for_each(|((h_val, &num), &den)| {
                let denom = den + regularization * (*h_val) + eps;
                *h_val *= num / denom;
            });

        // --- Update W ---
        // numerator_w = V × H^T  (n_users × k); h_t IS H^T already, so this
        // is a direct sparse × dense product with no transpose built first.
        let numerator_w = spmm_csr_dense(indptr, indices, data, n_users, &h_t, k);

        // denominator = W × (H H^T) + reg * W; H H^T = h_t^T h_t (a gramian).
        let hht = gram(&h_t, n_items, k);
        let w_hht = mm(&w, n_users, k, &hht, k);

        // Apply update: W *= numerator / (denominator + eps)
        w.par_iter_mut()
            .zip(numerator_w.par_iter())
            .zip(w_hht.par_iter())
            .for_each(|((w_val, &num), &den)| {
                let denom = den + regularization * (*w_val) + eps;
                *w_val *= num / denom;
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

    (w, h_t)
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
    // ponytail: cast f64 -> f32 once up front rather than threading a dtype
    // generic through the whole module; the Python-side API keeps passing f64.
    let dt_f32: Vec<f32> = dt.iter().map(|&v| v as f32).collect();

    let (uf, itf) = py.detach(|| {
        nmf_train(
            ip,
            ix,
            &dt_f32,
            n_users,
            n_items,
            factors,
            iterations,
            regularization as f32,
            seed,
            verbose,
        )
    });

    let ua = PyArray1::from_vec(py, uf);
    let ia = PyArray1::from_vec(py, itf);

    Ok((
        ua.reshape([n_users, factors])?.into(),
        ia.reshape([n_items, factors])?.into(),
    ))
}
