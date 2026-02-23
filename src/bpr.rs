use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

// ── SIMD-optimised dot product (8-wide unrolling for NEON / AVX2) ──────────
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    let chunks = k / 8;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut idx = 0;
    for _ in 0..chunks {
        unsafe {
            s0 += *a.get_unchecked(idx) * *b.get_unchecked(idx);
            s1 += *a.get_unchecked(idx + 1) * *b.get_unchecked(idx + 1);
            s2 += *a.get_unchecked(idx + 2) * *b.get_unchecked(idx + 2);
            s3 += *a.get_unchecked(idx + 3) * *b.get_unchecked(idx + 3);
            s4 += *a.get_unchecked(idx + 4) * *b.get_unchecked(idx + 4);
            s5 += *a.get_unchecked(idx + 5) * *b.get_unchecked(idx + 5);
            s6 += *a.get_unchecked(idx + 6) * *b.get_unchecked(idx + 6);
            s7 += *a.get_unchecked(idx + 7) * *b.get_unchecked(idx + 7);
        }
        idx += 8;
    }
    while idx < k {
        unsafe { s0 += *a.get_unchecked(idx) * *b.get_unchecked(idx); }
        idx += 1;
    }
    (s0 + s1 + s2 + s3) + (s4 + s5 + s6 + s7)
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

    #[inline(always)]
    fn next_float(&mut self) -> f32 {
        let v = self.next() & 0xFFFFFF;
        v as f32 / 0xFFFFFF as f32
    }
}

fn random_factors(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let scale = 1.0 / (k as f32).sqrt();
    let mut out = vec![0.0f32; n * k];
    for v in out.iter_mut() {
        *v = (rng.next_float() * 2.0 - 1.0) * scale;
    }
    out
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    0.5 + 0.5 * x / (1.0 + x.abs())
}

struct CSRLookup<'a> {
    indptr: &'a [i64],
    indices: &'a [i32],
}

impl<'a> CSRLookup<'a> {
    fn has_interaction(&self, u: usize, i: usize) -> bool {
        let start = self.indptr[u] as usize;
        let end = self.indptr[u + 1] as usize;
        self.indices[start..end].binary_search(&(i as i32)).is_ok()
    }

    fn get_positive_item(&self, u: usize, rng: &mut XorShift64) -> Option<usize> {
        let start = self.indptr[u] as usize;
        let end = self.indptr[u + 1] as usize;
        if start == end {
            return None;
        }
        let count = end - start;
        let offset = (rng.next() as usize) % count;
        Some(self.indices[start + offset] as usize)
    }
}

fn bpr_train(
    indptr: &[i64],
    indices: &[i32],
    n_users: usize,
    n_items: usize,
    k: usize,
    learning_rate: f32,
    regularization: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut user_factors = random_factors(n_users, k, seed);
    let mut item_factors = random_factors(n_items, k, seed.wrapping_add(1));
    let num_threads = rayon::current_num_threads();

    let uf_ptr_raw = user_factors.as_mut_ptr() as usize;
    let if_ptr_raw = item_factors.as_mut_ptr() as usize;

    let lookup = CSRLookup { indptr, indices };

    let n_samples = *indptr.last().unwrap() as usize;

    if verbose {
        println!("  BPR (Bayesian Personalized Ranking)");
        println!(
            "  Users: {}, Items: {}, Interactions: {}",
            n_users, n_items, n_samples
        );
        println!(
            "  Factors: {}, lr={}, reg={}",
            k, learning_rate, regularization
        );
        println!("  ITER |  SAMPLES/s | TIME");
        println!("  --------------------------------------");
    }

    let start_time = std::time::Instant::now();

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();

        let chunk_size = n_samples / num_threads;

        (0..num_threads).into_par_iter().for_each(|thread_idx| {
            let mut rng = XorShift64::new(
                seed.wrapping_add(iter as u64)
                    .wrapping_add(thread_idx as u64 * 100),
            );

            let uf_ptr = uf_ptr_raw as *mut f32;
            let if_ptr = if_ptr_raw as *mut f32;

            for _ in 0..chunk_size {
                let u = (rng.next() as usize) % n_users;

                let i = match lookup.get_positive_item(u, &mut rng) {
                    Some(val) => val,
                    None => continue,
                };

                let mut j = (rng.next() as usize) % n_items;
                for _ in 0..10 {
                    if !lookup.has_interaction(u, j) {
                        break;
                    }
                    j = (rng.next() as usize) % n_items;
                }

                unsafe {
                    let u_ptr = uf_ptr.add(u * k);
                    let i_ptr = if_ptr.add(i * k);
                    let j_ptr = if_ptr.add(j * k);

                    let xu = std::slice::from_raw_parts_mut(u_ptr, k);
                    let xi = std::slice::from_raw_parts_mut(i_ptr, k);
                    let xj = std::slice::from_raw_parts_mut(j_ptr, k);

                    let p_i = dot(xu, xi);
                    let p_j = dot(xu, xj);
                    let diff = p_i - p_j;

                    let sig = sigmoid(diff);
                    let deriv = 1.0 - sig;

                    for f in 0..k {
                        let w_u = xu[f];
                        let w_i = xi[f];
                        let w_j = xj[f];

                        xu[f] += learning_rate * (deriv * (w_i - w_j) - regularization * w_u);
                        xi[f] += learning_rate * (deriv * w_u - regularization * w_i);
                        xj[f] += learning_rate * (-deriv * w_u - regularization * w_j);
                    }
                }
            }
        });

        let iter_time = iter_start.elapsed().as_secs_f64();
        if verbose {
            let samples_per_sec = (n_samples as f64) / iter_time;
            println!(
                "  {:>4} | {:>10.0} | {:>6.2}s",
                iter + 1,
                samples_per_sec,
                iter_time
            );
        }
    }

    if verbose {
        println!("  --------------------------------------");
        println!("  Total time: {:.1}s", start_time.elapsed().as_secs_f64());
    }

    (user_factors, item_factors)
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_users, n_items, factors, learning_rate, regularization, iterations, seed, verbose))]
pub fn bpr_fit_implicit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    n_users: usize,
    n_items: usize,
    factors: usize,
    learning_rate: f32,
    regularization: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;

    let (uf, itf) = py.detach(|| {
        bpr_train(
            ip,
            ix,
            n_users,
            n_items,
            factors,
            learning_rate,
            regularization,
            iterations,
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
