use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

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
    fn next_float(&mut self) -> f32 {
        let v = self.next() & 0xFFFFFF;
        v as f32 / 0xFFFFFF as f32
    }
}

fn random_factors(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let scale = 0.01;
    let mut out = vec![0.0f32; size];
    for v in out.iter_mut() {
        *v = (rng.next_float() * 2.0 - 1.0) * scale;
    }
    out
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn log_loss(y: f32, pred: f32) -> f32 {
    let p = pred.clamp(1e-7, 1.0 - 1e-7);
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

// Factorization Machines (FM) for Binary Classification
// Input is binary CSR matrix (x_i = 1 for present features)
fn fm_train(
    indptr: &[i64],
    indices: &[i32],
    y: &[f32],
    n_samples: usize,
    n_features: usize,
    k: usize,
    learning_rate: f32,
    regularization: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> (f32, Vec<f32>, Vec<f32>) {
    let mut w0 = 0.0f32;
    let mut w = vec![0.0f32; n_features];
    let mut v = random_factors(n_features * k, seed);

    let num_threads = rayon::current_num_threads();
    let w0_ptr_raw = &mut w0 as *mut f32 as usize;
    let w_ptr_raw = w.as_mut_ptr() as usize;
    let v_ptr_raw = v.as_mut_ptr() as usize;

    if verbose {
        println!("  FM (Factorization Machines - Binary Classification)");
        println!("  Samples: {}, Features: {}", n_samples, n_features);
        println!("  Factors: {}, lr={}, reg={}", k, learning_rate, regularization);
        println!("  ITER |    LOSS    |  SAMPLES/s | TIME");
        println!("  ---------------------------------------------");
    }

    let start_time = std::time::Instant::now();

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();
        let chunk_size = n_samples / num_threads + 1;

        // Perform parallel SGD (Hogwild!)
        let total_loss: f32 = (0..num_threads).into_par_iter().map(|thread_idx| {
            let mut rng = XorShift64::new(seed.wrapping_add(iter as u64).wrapping_add(thread_idx as u64 * 100));
            
            let w0_ptr = w0_ptr_raw as *mut f32;
            let w_ptr = w_ptr_raw as *mut f32;
            let v_ptr = v_ptr_raw as *mut f32;
            
            let mut sum_v = vec![0.0f32; k];
            let mut local_loss = 0.0f32;

            for _ in 0..chunk_size {
                let s = (rng.next() as usize) % n_samples;
                let target = y[s];
                
                let start = indptr[s] as usize;
                let end = indptr[s + 1] as usize;
                let active_features = &indices[start..end];
                
                sum_v.fill(0.0);
                let mut sum_sq_v = vec![0.0f32; k];
                
                unsafe {
                    let mut y_hat = *w0_ptr;
                    for &i in active_features {
                        let i_idx = i as usize;
                        y_hat += *w_ptr.add(i_idx);
                        
                        let v_i = std::slice::from_raw_parts_mut(v_ptr.add(i_idx * k), k);
                        for f in 0..k {
                            let val = v_i[f];
                            sum_v[f] += val;
                            sum_sq_v[f] += val * val;
                        }
                    }

                    let mut cross_term = 0.0f32;
                    for f in 0..k {
                        cross_term += sum_v[f] * sum_v[f] - sum_sq_v[f];
                    }
                    y_hat += 0.5 * cross_term;
                    
                    let pred = sigmoid(y_hat);
                    local_loss += log_loss(target, pred);
                    
                    let error = pred - target; // gradient of Log Loss with Sigmoid = (pred - y)
                    
                    // Update w0
                    let lr = learning_rate;
                    let reg = regularization;
                    
                    *w0_ptr -= lr * error;
                    
                    // Update w and V
                    for &i in active_features {
                        let i_idx = i as usize;
                        let w_i = w_ptr.add(i_idx);
                        *w_i -= lr * (error + reg * (*w_i));
                        
                        let v_i = std::slice::from_raw_parts_mut(v_ptr.add(i_idx * k), k);
                        for f in 0..k {
                            let grad_v_if = error * (sum_v[f] - v_i[f]) + reg * v_i[f];
                            v_i[f] -= lr * grad_v_if;
                        }
                    }
                }
            }
            local_loss
        }).sum();

        let iter_time = iter_start.elapsed().as_secs_f64();
        if verbose {
            let avg_loss = total_loss / (chunk_size * num_threads) as f32;
            let samples_per_sec = (n_samples as f64) / iter_time;
            println!("  {:>4} | {:>10.4} | {:>10.0} | {:>6.2}s", iter + 1, avg_loss, samples_per_sec, iter_time);
        }
    }

    if verbose {
        println!("  ---------------------------------------------");
        println!("  Total time: {:.1}s", start_time.elapsed().as_secs_f64());
    }

    (w0, w, v)
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, y, n_samples, n_features, factors, learning_rate, regularization, iterations, seed, verbose))]
pub fn fm_fit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    y: PyReadonlyArray1<f32>,
    n_samples: usize,
    n_features: usize,
    factors: usize,
    learning_rate: f32,
    regularization: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<(f32, Py<PyArray1<f32>>, Py<PyArray2<f32>>)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let target = y.as_slice()?;

    let (w0, w, v) = py.detach(|| {
        fm_train(ip, ix, target, n_samples, n_features, factors, learning_rate, regularization, iterations, seed, verbose)
    });

    let a_w = PyArray1::from_vec(py, w);
    let a_v = PyArray1::from_vec(py, v).reshape([n_features, factors])?;

    Ok((w0, a_w.into(), a_v.into()))
}

// FM Inference function for fast predictions
#[pyfunction]
#[pyo3(signature = (indptr, indices, w0, w, v, factors, n_samples))]
pub fn fm_predict<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    w0: f32,
    w: PyReadonlyArray1<f32>,
    v: PyReadonlyArray2<f32>,
    factors: usize,
    n_samples: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let w_slice = w.as_slice()?;
    let v_slice = v.as_slice()?;

    let mut predictions = vec![0.0f32; n_samples];

    py.detach(|| {
        predictions.par_iter_mut().enumerate().for_each(|(s, p)| {
            let start = ip[s] as usize;
            let end = ip[s + 1] as usize;
            let active_features = &ix[start..end];

            let mut y_hat = w0;
            let mut sum_v = vec![0.0f32; factors];
            let mut sum_sq_v = vec![0.0f32; factors];

            for &i in active_features {
                let i_idx = i as usize;
                y_hat += w_slice[i_idx];
                
                for f in 0..factors {
                    let val = v_slice[i_idx * factors + f];
                    sum_v[f] += val;
                    sum_sq_v[f] += val * val;
                }
            }

            let mut cross_term = 0.0f32;
            for f in 0..factors {
                cross_term += sum_v[f] * sum_v[f] - sum_sq_v[f];
            }
            y_hat += 0.5 * cross_term;
            
            *p = sigmoid(y_hat); // return probability
        });
    });

    Ok(PyArray1::from_vec(py, predictions).into())
}
