use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

#[cfg(target_os = "macos")]
extern crate accelerate_src;

/// Cross-platform GEMM: C = alpha * op(A) * op(B) + beta * C (row-major).
/// Uses Apple Accelerate (AMX) on macOS, faer::matmul elsewhere.
#[inline]
fn gemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(target_os = "macos")]
    {
        let ta = if trans_a { cblas::Transpose::Ordinary } else { cblas::Transpose::None };
        let tb = if trans_b { cblas::Transpose::Ordinary } else { cblas::Transpose::None };
        unsafe {
            cblas::sgemm(
                cblas::Layout::RowMajor,
                ta, tb,
                m as i32, n as i32, k as i32,
                alpha, a, lda as i32, b, ldb as i32,
                beta, c, ldc as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Dimensions for faer: A is (rows_a, cols_a), B is (rows_b, cols_b)
        let (rows_a, cols_a) = if trans_a { (k, m) } else { (m, k) };
        let (rows_b, cols_b) = if trans_b { (n, k) } else { (k, n) };
        let a_mat = faer::mat::MatRef::from_row_major_slice(a, rows_a, cols_a);
        let b_mat = faer::mat::MatRef::from_row_major_slice(b, rows_b, cols_b);
        let a_ref = if trans_a { a_mat.transpose() } else { a_mat };
        let b_ref = if trans_b { b_mat.transpose() } else { b_mat };

        // Apply beta scaling
        if beta == 0.0 {
            c.iter_mut().for_each(|v| *v = 0.0);
        } else if beta != 1.0 {
            c.iter_mut().for_each(|v| *v *= beta);
        }

        let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c, m, n);
        faer::linalg::matmul::matmul(
            c_mat.as_mut(),
            faer::Accum::Add,
            a_ref,
            b_ref,
            alpha,
            faer::Par::rayon(0),
        );
    }
}

/// Deterministic sign flip for SVD components (matches scikit-learn / Spark).
///
/// For each component (row), finds the element with the largest absolute value.
/// If that element is negative, flips the entire row.
#[inline]
fn svd_flip(components: &mut [f32], n_features: usize) {
    let k = components.len() / n_features;
    for comp in 0..k {
        let row = &components[comp * n_features..(comp + 1) * n_features];
        let max_idx = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if row[max_idx] < 0.0 {
            let row_mut = &mut components[comp * n_features..(comp + 1) * n_features];
            for v in row_mut.iter_mut() {
                *v = -*v;
            }
        }
    }
}

/// Fit PCA on the input data matrix.
///
/// 1. Computes per-feature means and centers the data (parallel).
/// 2. Selects SVD solver: covariance trick, randomized SVD, or full SVD.
/// 3. Returns the top `n_components` principal components plus diagnostics.
#[pyfunction]
pub fn pca_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    n_components: usize,
    svd_solver: &str,
) -> PyResult<(
    Py<PyArray2<f32>>,           // components  (n_components, n_features)
    Py<PyArray1<f32>>,           // explained_variance  (n_components,)
    Py<PyArray1<f32>>,           // explained_variance_ratio  (n_components,)
    Py<PyArray1<f32>>,           // singular_values  (n_components,)
    Py<PyArray1<f32>>,           // mean  (n_features,)
)> {
    let shape = data.shape();
    let n_samples = shape[0];
    let n_features = shape[1];

    if n_samples == 0 || n_features == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input data must have at least one sample and one feature.",
        ));
    }

    let k = n_components.min(n_samples).min(n_features);

    // Read input into a contiguous Vec<f32>
    let data_slice = data.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("Input array must be C-contiguous.")
    })?;

    // ── 1. Compute column means (Parallel) ──────────────────────────────
    let mean_sum = data_slice.par_chunks(2048 * n_features).fold(
        || vec![0.0f64; n_features],
        |mut acc, chunk| {
            for row in chunk.chunks_exact(n_features) {
                for i in 0..n_features {
                    acc[i] += row[i] as f64;
                }
            }
            acc
        }
    ).reduce(
        || vec![0.0f64; n_features],
        |mut a, b| {
            for i in 0..n_features {
                a[i] += b[i];
            }
            a
        }
    );

    let inv_n = 1.0 / n_samples as f64;
    let mean: Vec<f32> = mean_sum.into_iter().map(|s| (s * inv_n) as f32).collect();

    let use_covariance = (svd_solver == "auto" && n_samples > n_features * 2 && n_features <= 2048) || svd_solver == "exact";
    let use_randomized = svd_solver == "randomized" || (svd_solver == "auto" && !use_covariance && k < n_samples.min(n_features) && n_samples.max(n_features) > 500);

    let min_dim = n_samples.min(n_features);

    let (mut components, mut singular_values) = if use_covariance {
        // Parallel X^T X computation
        let mut cov = data_slice.par_chunks(2048 * n_features).fold(
            || vec![0.0f32; n_features * n_features],
            |mut local_cov, chunk| {
                let mut c_row = vec![0.0f32; n_features];
                for row in chunk.chunks_exact(n_features) {
                    for i in 0..n_features {
                        c_row[i] = unsafe { *row.get_unchecked(i) - *mean.get_unchecked(i) };
                    }
                    for i in 0..n_features {
                        let diff_i = c_row[i];
                        if diff_i == 0.0 { continue; }
                        let row_out = i * n_features;
                        
                        let (local_slice, crow_slice) = unsafe {
                            (
                                std::slice::from_raw_parts_mut(local_cov.as_mut_ptr().add(row_out + i), n_features - i),
                                std::slice::from_raw_parts(c_row.as_ptr().add(i), n_features - i)
                            )
                        };
                        for j in 0..n_features - i {
                            local_slice[j] += diff_i * crow_slice[j];
                        }
                    }
                }
                local_cov
            }
        ).reduce(
            || vec![0.0f32; n_features * n_features],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            }
        );

        // Fill lower triangular
        for i in 0..n_features {
            for j in 0..i {
                cov[i * n_features + j] = cov[j * n_features + i];
            }
        }

        let cov_mat = faer::Mat::<f32>::from_fn(n_features, n_features, |r, c| cov[r * n_features + c]);
        let svd = cov_mat.thin_svd().expect("SVD failed on covariance matrix");

        let s_cov = svd.S();
        let v_cov = svd.V();

        let mut comps = vec![0.0f32; k * n_features];
        for comp in 0..k {
            for feat in 0..n_features {
                comps[comp * n_features + feat] = v_cov[(feat, comp)];
            }
        }
        svd_flip(&mut comps, n_features);

        let mut svs = Vec::with_capacity(min_dim);
        for i in 0..min_dim {
            let mut val = s_cov.column_vector()[i];
            if val < 0.0 { val = 0.0; }
            svs.push(val.sqrt());
        }

        (comps, svs)
    } else if use_randomized {
        let k_extra = (k + 10).min(n_features).min(n_samples);
        let mut omega = vec![0.0f32; n_features * k_extra];
        for i in (0..omega.len()).step_by(2) {
            let mut u1: f32 = rand::random();
            while u1 == 0.0 { u1 = rand::random(); }
            let u2: f32 = rand::random();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            omega[i] = r * theta.cos();
            if i + 1 < omega.len() {
                omega[i + 1] = r * theta.sin();
            }
        }
        
        // Y = (X - mu) * Omega, computed as X*Omega - mu*Omega to avoid centering copy
        
        let mut y_mat_data = vec![0.0f32; n_samples * k_extra];
        
        // Y = X * Omega via BLAS
        gemm(
            false, false,
            n_samples, k_extra, n_features,
            1.0, data_slice, n_features,
            &omega, k_extra,
            0.0, &mut y_mat_data, k_extra,
        );
        
        // Subtract mu * Omega from every row of Y
        let mut mu_omega = vec![0.0f32; k_extra];
        for j in 0..k_extra {
            for i in 0..n_features {
                mu_omega[j] += mean[i] * omega[i * k_extra + j];
            }
        }
        
        y_mat_data.par_chunks_mut(k_extra).for_each(|y_row| {
            for j in 0..k_extra {
                y_row[j] -= mu_omega[j];
            }
        });

        // Power iteration: improves subspace quality
        // Y = (X - mu) * ((X - mu)^T * Y)
        let n_iter = 1;
        for _power_iter in 0..n_iter {
            // Step A: Z = X^T * Y  (n_features x k_extra)
            let mut z_data = vec![0.0f32; n_features * k_extra];
            gemm(
                true, false,
                n_features, k_extra, n_samples,
                1.0, data_slice, n_features,
                &y_mat_data, k_extra,
                0.0, &mut z_data, k_extra,
            );
            // Subtract mu^T * Y from Z: each z[f, j] -= mean[f] * sum_r(y[r, j])
            let mut y_col_sums = vec![0.0f32; k_extra];
            for r in 0..n_samples {
                for j in 0..k_extra {
                    y_col_sums[j] += y_mat_data[r * k_extra + j];
                }
            }
            for f in 0..n_features {
                for j in 0..k_extra {
                    z_data[f * k_extra + j] -= mean[f] * y_col_sums[j];
                }
            }

            // Step B: Y = X * Z  (n_samples x k_extra)
            gemm(
                false, false,
                n_samples, k_extra, n_features,
                1.0, data_slice, n_features,
                &z_data, k_extra,
                0.0, &mut y_mat_data, k_extra,
            );
            // Subtract mu * Z from Y: each y[r, j] -= sum_f(mean[f] * z[f, j])
            let mut mu_z = vec![0.0f32; k_extra];
            for j in 0..k_extra {
                for f in 0..n_features {
                    mu_z[j] += mean[f] * z_data[f * k_extra + j];
                }
            }
            y_mat_data.par_chunks_mut(k_extra).for_each(|y_row| {
                for j in 0..k_extra {
                    y_row[j] -= mu_z[j];
                }
            });
        }

        // After power iteration, Y spans the principal subspace.
        // We need Q from QR(Y) to compute B = Q^T (X-mu), then SVD(B).
        // Instead: compute B = Y^T (X-mu) directly (k_extra x n_features).
        // Then do SVD on Y^T Y to orthogonalize:
        //   Y^T Y = V S^2 V^T,  Q = Y V S^{-1}
        //   B = Q^T (X-mu) = S^{-1} V^T Y^T (X-mu)
        // So: M = Y^T (X-mu) is (k_extra x n_features), then B = S^{-1} V^T M
        // And Y^T Y is tiny (k_extra x k_extra), so its SVD is essentially free.
        
        // Step 1: M = Y^T * X  (k_extra x n_features)
        let mut m_data = vec![0.0f32; k_extra * n_features];
        gemm(
            true, false,
            k_extra, n_features, n_samples,
            1.0, &y_mat_data, k_extra,
            data_slice, n_features,
            0.0, &mut m_data, n_features,
        );
        
        // Subtract Y^T mu: m[i,j] -= y_col_sums[i] * mean[j]
        let mut y_col_sums = vec![0.0f32; k_extra];
        for r in 0..n_samples {
            for j in 0..k_extra {
                y_col_sums[j] += y_mat_data[r * k_extra + j];
            }
        }
        for i in 0..k_extra {
            for j in 0..n_features {
                m_data[i * n_features + j] -= y_col_sums[i] * mean[j];
            }
        }
        
        // Step 2: G = Y^T * Y  (k_extra x k_extra)
        let mut g_data = vec![0.0f32; k_extra * k_extra];
        gemm(
            true, false,
            k_extra, k_extra, n_samples,
            1.0, &y_mat_data, k_extra,
            &y_mat_data, k_extra,
            0.0, &mut g_data, k_extra,
        );
        
        // Step 3: SVD of G to get V, S^2 for orthogonalization
        let g_mat = faer::Mat::<f32>::from_fn(k_extra, k_extra, |r, c| g_data[r * k_extra + c]);
        let g_svd = g_mat.thin_svd().expect("SVD failed on Y^T Y");
        let g_v = g_svd.V();
        let g_s = g_svd.S();
        
        // Step 4: B = S^{-1} * V^T * M  (k_extra x n_features)
        // First compute V^T * M
        let mut vtm_data = vec![0.0f32; k_extra * n_features];
        for i in 0..k_extra {
            for j in 0..n_features {
                for p in 0..k_extra {
                    vtm_data[i * n_features + j] += g_v[(p, i)] * m_data[p * n_features + j];
                }
            }
        }
        
        // Scale by S^{-1} (S contains eigenvalues of G = Y^T Y)
        let mut b_mat_data = vec![0.0f32; k_extra * n_features];
        for i in 0..k_extra {
            let s_val = g_s.column_vector()[i];
            let s_inv = if s_val > 1e-10 { 1.0 / s_val.sqrt() } else { 0.0 };
            for j in 0..n_features {
                b_mat_data[i * n_features + j] = s_inv * vtm_data[i * n_features + j];
            }
        }
        
        let b = faer::Mat::<f32>::from_fn(k_extra, n_features, |r, c| b_mat_data[r * n_features + c]);
        let b_svd = b.thin_svd().expect("SVD failed on randomized B matrix");
        
        let v = b_svd.V();
        let mut comps = vec![0.0f32; k * n_features];
        for comp in 0..k {
            for feat in 0..n_features {
                comps[comp * n_features + feat] = v[(feat, comp)];
            }
        }
        svd_flip(&mut comps, n_features);
        
        let mut svs = Vec::with_capacity(min_dim);
        for i in 0..k_extra.min(min_dim) {
            svs.push(b_svd.S().column_vector()[i]);
        }
        for _ in k_extra.min(min_dim)..min_dim {
            svs.push(0.0);
        }
        (comps, svs)
    } else {
        // Standard centered matrix SVD (less efficient for large n_samples, good for others)
        let centered = faer::Mat::<f32>::from_fn(n_samples, n_features, |r, c| {
            let val = unsafe { *data_slice.get_unchecked(r * n_features + c) };
            val - mean[c]
        });

        let svd = centered.thin_svd().expect("SVD failed (data may contain NaNs or Infs).");
        
        let s_diag = svd.S();
        let v = svd.V();

        let mut comps = vec![0.0f32; k * n_features];
        for comp in 0..k {
            for feat in 0..n_features {
                comps[comp * n_features + feat] = v[(feat, comp)];
            }
        }
        svd_flip(&mut comps, n_features);

        let mut svs = Vec::with_capacity(min_dim);
        for i in 0..min_dim {
            svs.push(s_diag.column_vector()[i]);
        }

        (comps, svs)
    };

    // ── 5. Explained variance: S² / (n - 1) ────────────────────────────
    let dof = if n_samples > 1 { (n_samples - 1) as f32 } else { 1.0 };

    let mut explained_variance = Vec::with_capacity(k);
    let mut total_variance = 0.0f32;
    for i in 0..min_dim {
        let sv = singular_values[i];
        total_variance += sv * sv / dof;
    }
    for i in 0..k {
        let sv = singular_values[i];
        explained_variance.push(sv * sv / dof);
    }

    // ── 6. Explained variance ratio ─────────────────────────────────────
    let mut explained_variance_ratio = Vec::with_capacity(k);
    let inv_total = if total_variance > 0.0 {
        1.0 / total_variance
    } else {
        0.0
    };
    for ev in &explained_variance {
        explained_variance_ratio.push(*ev * inv_total);
    }

    // Truncate singular values to top-k only for python return
    singular_values.truncate(k);

    // ── 8. Convert to numpy arrays ──────────────────────────────────────
    let components_np = numpy::PyArray1::from_vec(py, components)
        .reshape([k, n_features])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .into_pyobject(py)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .unbind();
    let ev_np = explained_variance.into_pyarray(py).into();
    let evr_np = explained_variance_ratio.into_pyarray(py).into();
    let sv_np = singular_values.into_pyarray(py).into();
    let mean_np = mean.into_pyarray(py).into();

    Ok((components_np, ev_np, evr_np, sv_np, mean_np))
}

/// Transform data using pre-computed PCA components and mean.
///
/// Returns the projected data: (X - mean) @ components.T
#[pyfunction]
pub fn pca_transform<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    mean: PyReadonlyArray1<'py, f32>,
    components: PyReadonlyArray2<'py, f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let data_shape = data.shape();
    let n_samples = data_shape[0];
    let n_features = data_shape[1];
    let comp_shape = components.shape();
    let n_components = comp_shape[0];

    if comp_shape[1] != n_features {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Components have {} features but data has {}.",
            comp_shape[1], n_features
        )));
    }

    let data_s = data.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("Data array must be C-contiguous.")
    })?;
    let mean_s = mean.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("Mean array must be C-contiguous.")
    })?;
    let comp_s = components.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("Components array must be C-contiguous.")
    })?;

    // Result = X * components^T
    let mut result = vec![0.0f32; n_samples * n_components];
    gemm(
        false, true,
        n_samples, n_components, n_features,
        1.0, data_s, n_features,
        comp_s, n_features,
        0.0, &mut result, n_components,
    );

    // Subtract mu * W^T from every row
    let mut mu_proj = vec![0.0f32; n_components];
    for k in 0..n_components {
        for p in 0..n_features {
            mu_proj[k] += mean_s[p] * comp_s[k * n_features + p];
        }
    }

    result.par_chunks_mut(n_components).for_each(|res_row| {
        for k in 0..n_components {
            res_row[k] -= mu_proj[k];
        }
    });

    let result_np = numpy::PyArray1::from_vec(py, result)
        .reshape([n_samples, n_components])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .into_pyobject(py)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .unbind();

    Ok(result_np)
}
