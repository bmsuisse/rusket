use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Fit PCA on the input data matrix.
///
/// 1. Computes per-feature means and centers the data.
/// 2. Runs thin SVD via faer.
/// 3. Returns the top `n_components` principal components plus diagnostics.
#[pyfunction]
pub fn pca_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    n_components: usize,
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

    // ── 1. Compute column means ─────────────────────────────────────────
    let mut mean = vec![0.0f32; n_features];
    for row in 0..n_samples {
        let base = row * n_features;
        for col in 0..n_features {
            mean[col] += data_slice[base + col];
        }
    }
    let inv_n = 1.0 / n_samples as f32;
    for m in mean.iter_mut() {
        *m *= inv_n;
    }

    // ── 2. Build centered faer matrix ───────────────────────────────────
    let centered = faer::Mat::<f32>::from_fn(n_samples, n_features, |r, c| {
        data_slice[r * n_features + c] - mean[c]
    });

    // ── 3. Thin SVD via faer ────────────────────────────────────────────
    let svd = centered.thin_svd();
    // svd.s_diagonal() returns a DiagRef with singular values in descending order.
    // svd.v() returns the right singular vectors (n_features × min(n,p)).

    let s_diag = svd.s_diagonal();
    let v = svd.v(); // n_features × min(n_samples, n_features)

    let min_dim = n_samples.min(n_features);

    // ── 4. Extract top-k components (rows of Vt) ────────────────────────
    let mut components = vec![0.0f32; k * n_features];
    for comp in 0..k {
        for feat in 0..n_features {
            // V is (n_features, min_dim), column-major
            // Vt[comp, feat] = V[feat, comp]
            components[comp * n_features + feat] = v[(feat, comp)];
        }
    }

    // ── 5. Explained variance: S² / (n - 1) ────────────────────────────
    let dof = if n_samples > 1 { (n_samples - 1) as f32 } else { 1.0 };

    let mut explained_variance = Vec::with_capacity(k);
    let mut total_variance = 0.0f32;
    for i in 0..min_dim {
        let sv = s_diag.column_vector()[(i, 0)];
        total_variance += sv * sv / dof;
    }
    for i in 0..k {
        let sv = s_diag.column_vector()[(i, 0)];
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

    // ── 7. Singular values (top-k) ──────────────────────────────────────
    let mut singular_values = Vec::with_capacity(k);
    for i in 0..k {
        singular_values.push(s_diag.column_vector()[(i, 0)]);
    }

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

    // result[i, j] = sum_f (data[i, f] - mean[f]) * components[j, f]
    let mut result = vec![0.0f32; n_samples * n_components];
    for i in 0..n_samples {
        let row_base = i * n_features;
        let out_base = i * n_components;
        for j in 0..n_components {
            let comp_base = j * n_features;
            let mut acc = 0.0f32;
            for f in 0..n_features {
                acc += (data_s[row_base + f] - mean_s[f]) * comp_s[comp_base + f];
            }
            result[out_base + j] = acc;
        }
    }

    let result_np = numpy::PyArray1::from_vec(py, result)
        .reshape([n_samples, n_components])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .into_pyobject(py)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .unbind();

    Ok(result_np)
}

use numpy::PyArray1;
