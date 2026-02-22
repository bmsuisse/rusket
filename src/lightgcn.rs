use ndarray::{Array2, Axis, Zip};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

struct AtomicF32Array {
    data: Vec<AtomicU32>,
    cols: usize,
}

impl AtomicF32Array {
    fn zeros(rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        let mut data = Vec::with_capacity(len);
        data.resize_with(len, || AtomicU32::new(0_f32.to_bits()));
        Self { data, cols }
    }

    #[inline(always)]
    fn add(&self, row: usize, col: usize, val: f32) {
        let idx = row * self.cols + col;
        let atomic = &self.data[idx];
        let mut current = atomic.load(Ordering::Relaxed);
        loop {
            let current_f32 = f32::from_bits(current);
            let new_val = current_f32 + val;
            match atomic.compare_exchange_weak(
                current,
                new_val.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }

    fn to_array2(&self, rows: usize) -> Array2<f32> {
        let mut arr = Array2::zeros((rows, self.cols));
        for r in 0..rows {
            for c in 0..self.cols {
                arr[[r, c]] = f32::from_bits(self.data[r * self.cols + c].load(Ordering::Relaxed));
            }
        }
        arr
    }
}

fn propagate(
    e_u: &Array2<f32>,
    e_i: &Array2<f32>,
    user_indptr: &[i32],
    user_indices: &[i32],
    item_indptr: &[i32],
    item_indices: &[i32],
    d_u: &[f32],
    d_i: &[f32],
    k_layers: usize,
) -> (Array2<f32>, Array2<f32>) {
    let users = e_u.nrows();
    let items = e_i.nrows();
    let factors = e_u.ncols();

    let mut final_u = e_u.clone();
    let mut final_i = e_i.clone();

    let mut curr_u = e_u.clone();
    let mut curr_i = e_i.clone();

    for _ in 0..k_layers {
        let mut next_u = Array2::<f32>::zeros((users, factors));
        let mut next_i = Array2::<f32>::zeros((items, factors));

        next_u
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(u, mut row)| {
                let start = user_indptr[u] as usize;
                let end = user_indptr[u + 1] as usize;
                if start == end {
                    return;
                }
                let norm_u = d_u[u].sqrt();
                for &i in &user_indices[start..end] {
                    let i = i as usize;
                    let norm_i = d_i[i].sqrt();
                    let alpha = 1.0 / (norm_u * norm_i);
                    let i_row = curr_i.row(i);
                    for f in 0..factors {
                        row[f] += alpha * i_row[f];
                    }
                }
            });

        next_i
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let start = item_indptr[i] as usize;
                let end = item_indptr[i + 1] as usize;
                if start == end {
                    return;
                }
                let norm_i = d_i[i].sqrt();
                for &u in &item_indices[start..end] {
                    let u = u as usize;
                    let norm_u = d_u[u].sqrt();
                    let alpha = 1.0 / (norm_u * norm_i);
                    let u_row = curr_u.row(u);
                    for f in 0..factors {
                        row[f] += alpha * u_row[f];
                    }
                }
            });

        curr_u = next_u;
        curr_i = next_i;

        Zip::from(&mut final_u).and(&curr_u).par_for_each(|f, &c| *f += c);
        Zip::from(&mut final_i).and(&curr_i).par_for_each(|f, &c| *f += c);
    }

    let scale = 1.0 / (k_layers as f32 + 1.0);
    final_u.mapv_inplace(|x| x * scale);
    final_i.mapv_inplace(|x| x * scale);

    (final_u, final_i)
}

#[pyclass]
pub struct LightGCNCore {
    users: usize,
    items: usize,
    factors: usize,
    k_layers: usize,
    learning_rate: f32,
    lambda: f32,
    e_u: Array2<f32>,
    e_i: Array2<f32>,
    adam_m_u: Array2<f32>,
    adam_v_u: Array2<f32>,
    adam_m_i: Array2<f32>,
    adam_v_i: Array2<f32>,
    beta1: f32,
    beta2: f32,
    t: usize,
}

#[pymethods]
impl LightGCNCore {
    #[new]
    fn new(
        users: usize,
        items: usize,
        factors: usize,
        k_layers: usize,
        learning_rate: f32,
        lambda: f32,
    ) -> Self {
        let e_u = Array2::random((users, factors), Uniform::new(-0.1, 0.1));
        let e_i = Array2::random((items, factors), Uniform::new(-0.1, 0.1));
        Self {
            users,
            items,
            factors,
            k_layers,
            learning_rate,
            lambda,
            e_u,
            e_i,
            adam_m_u: Array2::zeros((users, factors)),
            adam_v_u: Array2::zeros((users, factors)),
            adam_m_i: Array2::zeros((items, factors)),
            adam_v_i: Array2::zeros((items, factors)),
            beta1: 0.9,
            beta2: 0.999,
            t: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_epoch<'py>(
        &mut self,
        _py: Python<'py>,
        user_indptr: PyReadonlyArray1<'py, i32>,
        user_indices: PyReadonlyArray1<'py, i32>,
        item_indptr: PyReadonlyArray1<'py, i32>,
        item_indices: PyReadonlyArray1<'py, i32>,
        d_u: PyReadonlyArray1<'py, f32>,
        d_i: PyReadonlyArray1<'py, f32>,
    ) -> f32 {
        let u_ptr = user_indptr.as_slice().unwrap();
        let u_idx = user_indices.as_slice().unwrap();
        let i_ptr = item_indptr.as_slice().unwrap();
        let i_idx = item_indices.as_slice().unwrap();
        let du = d_u.as_slice().unwrap();
        let di = d_i.as_slice().unwrap();

        let (final_u, final_i) = propagate(
            &self.e_u,
            &self.e_i,
            u_ptr,
            u_idx,
            i_ptr,
            i_idx,
            du,
            di,
            self.k_layers,
        );

        let g_final_i_atomic = AtomicF32Array::zeros(self.items, self.factors);
        let mut g_final_u = Array2::<f32>::zeros((self.users, self.factors));

        let loss: f32 = g_final_u
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .map(|(u, mut g_u)| {
                let start = u_ptr[u] as usize;
                let end = u_ptr[u + 1] as usize;
                if start == end {
                    return 0.0;
                }

                let mut rng = rand::thread_rng();
                let mut local_loss = 0.0;

                let e_u_row = final_u.row(u);

                for &i in &u_idx[start..end] {
                    let i = i as usize;

                    let mut j = rng.gen_range(0..self.items);
                    while u_idx[start..end].contains(&(j as i32)) {
                        j = rng.gen_range(0..self.items);
                    }

                    let e_i_row = final_i.row(i);
                    let e_j_row = final_i.row(j);

                    let mut y_ui = 0.0;
                    let mut y_uj = 0.0;
                    for f in 0..self.factors {
                        y_ui += e_u_row[f] * e_i_row[f];
                        y_uj += e_u_row[f] * e_j_row[f];
                    }

                    let x_uij = y_ui - y_uj;
                    let exp_nx = (-x_uij).exp();
                    let sigmoid = 1.0 / (1.0 + exp_nx);

                    local_loss += (1.0 + exp_nx).ln();
                    let grad = sigmoid - 1.0;

                    for f in 0..self.factors {
                        let e_uf = e_u_row[f];
                        let e_if = e_i_row[f];
                        let e_jf = e_j_row[f];

                        g_u[f] += grad * (e_if - e_jf);
                        g_final_i_atomic.add(i, f, grad * e_uf);
                        g_final_i_atomic.add(j, f, grad * (-e_uf));
                    }
                }
                local_loss
            })
            .sum();

        let g_final_i = g_final_i_atomic.to_array2(self.items);

        let (g_0_u, g_0_i) = propagate(
            &g_final_u,
            &g_final_i,
            u_ptr,
            u_idx,
            i_ptr,
            i_idx,
            du,
            di,
            self.k_layers,
        );

        self.t += 1;
        let t_f32 = self.t as f32;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let lr = self.learning_rate;
        let lambda = self.lambda;
        let bias_correction1 = 1.0 - beta1.powf(t_f32);
        let bias_correction2 = 1.0 - beta2.powf(t_f32);
        let step_size = lr * (bias_correction2.sqrt()) / bias_correction1;

        Zip::from(&mut self.e_u)
            .and(&mut self.adam_m_u)
            .and(&mut self.adam_v_u)
            .and(&g_0_u)
            .par_for_each(|e, m, v, &g| {
                let grad = g + lambda * (*e);
                *m = beta1 * (*m) + (1.0 - beta1) * grad;
                *v = beta2 * (*v) + (1.0 - beta2) * grad.powi(2);
                *e -= step_size * (*m) / (v.sqrt() + 1e-8);
            });

        Zip::from(&mut self.e_i)
            .and(&mut self.adam_m_i)
            .and(&mut self.adam_v_i)
            .and(&g_0_i)
            .par_for_each(|e, m, v, &g| {
                let grad = g + lambda * (*e);
                *m = beta1 * (*m) + (1.0 - beta1) * grad;
                *v = beta2 * (*v) + (1.0 - beta2) * grad.powi(2);
                *e -= step_size * (*m) / (v.sqrt() + 1e-8);
            });

        loss / (u_idx.len() as f32)
    }

    #[allow(clippy::too_many_arguments)]
    fn get_final_embeddings<'py>(
        &self,
        py: Python<'py>,
        user_indptr: PyReadonlyArray1<'py, i32>,
        user_indices: PyReadonlyArray1<'py, i32>,
        item_indptr: PyReadonlyArray1<'py, i32>,
        item_indices: PyReadonlyArray1<'py, i32>,
        d_u: PyReadonlyArray1<'py, f32>,
        d_i: PyReadonlyArray1<'py, f32>,
    ) -> (PyObject, PyObject) {
        let u_ptr = user_indptr.as_slice().unwrap();
        let u_idx = user_indices.as_slice().unwrap();
        let i_ptr = item_indptr.as_slice().unwrap();
        let i_idx = item_indices.as_slice().unwrap();
        let du = d_u.as_slice().unwrap();
        let di = d_i.as_slice().unwrap();

        let (final_u, final_i) = propagate(
            &self.e_u,
            &self.e_i,
            u_ptr,
            u_idx,
            i_ptr,
            i_idx,
            du,
            di,
            self.k_layers,
        );

        (
            final_u.into_pyarray(py).into(),
            final_i.into_pyarray(py).into(),
        )
    }
}
