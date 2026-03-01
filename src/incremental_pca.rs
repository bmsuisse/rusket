use faer::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

#[pyclass]
pub struct RustIncrementalPCA {
    #[pyo3(get)]
    n_components: usize,
    #[pyo3(get)]
    n_features: usize,
    #[pyo3(get)]
    n_samples_seen: usize,
    mean: Vec<f32>,
    components: faer::Mat<f32>, // (K, D)
    singular_values: Vec<f32>,  // (K,)
}

#[pymethods]
impl RustIncrementalPCA {
    #[new]
    fn new(n_components: usize, n_features: usize) -> Self {
        Self {
            n_components,
            n_features,
            n_samples_seen: 0,
            mean: vec![0.0; n_features],
            components: faer::Mat::zeros(n_components, n_features),
            singular_values: vec![0.0; n_components],
        }
    }

    fn partial_fit<'py>(&mut self, _py: Python<'py>, data: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = data.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        if n_features != self.n_features {
            return Err(PyValueError::new_err("Feature dimension mismatch."));
        }
        let data_slice = data.as_slice().unwrap();

        // 1. Compute column means of batch
        let batch_mean = data_slice.par_chunks(2048 * n_features).fold(
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
        let mu_batch: Vec<f32> = batch_mean.iter().map(|s| (s * inv_n) as f32).collect();

        // 2. Updated mean and mean_correction
        let mut mean_correction = vec![0.0f32; n_features];
        if self.n_samples_seen == 0 {
            self.mean = mu_batch.clone();
            self.n_samples_seen = n_samples;
        } else {
            let n_total_old = self.n_samples_seen as f64;
            let n_new = n_samples as f64;
            let n_total_new = n_total_old + n_new;
            let weight_old = n_total_old / n_total_new;
            let weight_new = n_new / n_total_new;

            let correction_scale = ((n_total_old * n_new) / n_total_new).sqrt() as f32;

            for i in 0..n_features {
                let m_old = self.mean[i];
                let m_new = mu_batch[i];
                self.mean[i] = (m_old as f64 * weight_old + m_new as f64 * weight_new) as f32;
                mean_correction[i] = correction_scale * (m_old - m_new);
            }
            self.n_samples_seen += n_samples;
        }

        // 3. Construct M
        let rows_m = self.n_components + n_samples + 1;
        let mut m_mat = faer::Mat::<f32>::zeros(rows_m, n_features);

        let mut row_idx = 0;
        // Previous components scaled by singular values
        if self.n_samples_seen > n_samples {
            for k in 0..self.n_components {
                let sv = self.singular_values[k];
                for f in 0..n_features {
                    m_mat[(row_idx, f)] = sv * self.components[(k, f)];
                }
                row_idx += 1;
            }
        }

        // Centered batch data
        for i in 0..n_samples {
            let row_offset = i * n_features;
            for f in 0..n_features {
                m_mat[(row_idx, f)] = data_slice[row_offset + f] - mu_batch[f];
            }
            row_idx += 1;
        }

        // Mean correction
        if self.n_samples_seen > n_samples {
            for f in 0..n_features {
                m_mat[(row_idx, f)] = mean_correction[f];
            }
            row_idx += 1;
        }

        // 4. SVD on M
        let actual_rows = row_idx;
        let m_slice = m_mat.submatrix(0, 0, actual_rows, n_features);
        
        let svd = m_slice.thin_svd();
        let v_mat = svd.V();
        let s_vec = svd.S().column_vector();

        let k_actual = self.n_components.min(actual_rows).min(n_features);
        for k in 0..k_actual {
            self.singular_values[k] = s_vec[k];
            for f in 0..n_features {
                self.components[(k, f)] = v_mat[(f, k)];
            }
        }
        for k in k_actual..self.n_components {
            self.singular_values[k] = 0.0;
            for f in 0..n_features {
                self.components[(k, f)] = 0.0;
            }
        }

        // 5. Deterministic sign flip
        for comp in 0..k_actual {
            let mut max_val = 0.0f32;
            let mut max_idx = 0;
            for j in 0..n_features {
                let val = self.components[(comp, j)];
                if val.abs() > max_val {
                    max_val = val.abs();
                    max_idx = j;
                }
            }
            if self.components[(comp, max_idx)] < 0.0 {
                for j in 0..n_features {
                    self.components[(comp, j)] = -self.components[(comp, j)];
                }
            }
        }

        Ok(())
    }

    #[getter]
    fn get_components<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f32>>> {
        let mut comps = vec![0.0f32; self.n_components * self.n_features];
        for k in 0..self.n_components {
            for f in 0..self.n_features {
                comps[k * self.n_features + f] = self.components[(k, f)];
            }
        }
        let arr = PyArray1::from_vec(py, comps).reshape((self.n_components, self.n_features)).unwrap();
        Ok(arr.into_pyobject(py).unwrap().unbind())
    }

    #[getter]
    fn get_singular_values<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        let arr = PyArray1::from_vec(py, self.singular_values.clone());
        Ok(arr.into_pyobject(py).unwrap().unbind())
    }

    #[getter]
    fn get_mean<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        let arr = PyArray1::from_vec(py, self.mean.clone());
        Ok(arr.into_pyobject(py).unwrap().unbind())
    }
}
