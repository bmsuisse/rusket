use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

fn random_factors(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let scale = 0.01; // small random initialization
    let mut out = vec![0.0f32; size];
    for v in out.iter_mut() {
        *v = (rng.next_float() * 2.0 - 1.0) * scale;
    }
    out
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    0.5 + 0.5 * x / (1.0 + x.abs())
}

struct SequenceLookup<'a> {
    indptr: &'a [i64],
    indices: &'a [i32],
}

impl<'a> SequenceLookup<'a> {
    fn has_item(&self, u: usize, i: usize) -> bool {
        let start = self.indptr[u] as usize;
        let end = self.indptr[u + 1] as usize;
        self.indices[start..end].iter().any(|&item| item == i as i32)
    }

    fn get_sequence(&self, u: usize) -> &'a [i32] {
        let start = self.indptr[u] as usize;
        let end = self.indptr[u + 1] as usize;
        &self.indices[start..end]
    }
}

fn fpmc_train(
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
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    // FPMC Models:
    // V_{U,I} -> User factors for Matrix Factorization
    // V_{I,U} -> Item factors for Matrix Factorization
    // V_{I,L} -> Item next-factors for Markov Chain
    // V_{L,I} -> Item prev-factors for Markov Chain
    
    let mut vu = random_factors(n_users * k, seed);
    let mut viu = random_factors(n_items * k, seed.wrapping_add(1));
    let mut vil = random_factors(n_items * k, seed.wrapping_add(2));
    let mut vli = random_factors(n_items * k, seed.wrapping_add(3));

    let num_threads = rayon::current_num_threads();

    let vu_ptr_raw = vu.as_mut_ptr() as usize;
    let viu_ptr_raw = viu.as_mut_ptr() as usize;
    let vil_ptr_raw = vil.as_mut_ptr() as usize;
    let vli_ptr_raw = vli.as_mut_ptr() as usize;

    let lookup = SequenceLookup { indptr, indices };

    // Count transitions
    let mut total_transitions = 0;
    for u in 0..n_users {
        let seq = lookup.get_sequence(u);
        if seq.len() > 1 {
            total_transitions += seq.len() - 1;
        }
    }

    if verbose {
        println!("  FPMC (Factorized Personalized Markov Chains)");
        println!("  Users: {}, Items: {}, Transitions: {}", n_users, n_items, total_transitions);
        println!("  Factors: {}, lr={}, reg={}", k, learning_rate, regularization);
        println!("  ITER |  SAMPLES/s | TIME");
        println!("  --------------------------------------");
    }

    let start_time = std::time::Instant::now();
    let samples_per_epoch = total_transitions;

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();
        let chunk_size = samples_per_epoch / num_threads + 1;

        (0..num_threads).into_par_iter().for_each(|thread_idx| {
            let mut rng = XorShift64::new(seed.wrapping_add(iter as u64).wrapping_add(thread_idx as u64 * 100));

            let vu_ptr = vu_ptr_raw as *mut f32;
            let viu_ptr = viu_ptr_raw as *mut f32;
            let vil_ptr = vil_ptr_raw as *mut f32;
            let vli_ptr = vli_ptr_raw as *mut f32;

            for _ in 0..chunk_size {
                let u = (rng.next() as usize) % n_users;
                let seq = lookup.get_sequence(u);
                
                if seq.len() < 2 {
                    continue;
                }

                // Randomly pick a target position in sequence (> 0)
                let pos = (rng.next() as usize % (seq.len() - 1)) + 1;
                let l = seq[pos - 1] as usize;
                let i = seq[pos] as usize;

                let mut j = (rng.next() as usize) % n_items;
                for _ in 0..10 {
                    if !lookup.has_item(u, j) {
                        break;
                    }
                    j = (rng.next() as usize) % n_items;
                }

                unsafe {
                    let u_ptr = vu_ptr.add(u * k);
                    let iu_ptr = viu_ptr.add(i * k);
                    let ju_ptr = viu_ptr.add(j * k);
                    
                    let il_ptr = vil_ptr.add(i * k);
                    let jl_ptr = vil_ptr.add(j * k);
                    let li_ptr = vli_ptr.add(l * k);

                    let v_u = std::slice::from_raw_parts_mut(u_ptr, k);
                    
                    let v_iu = std::slice::from_raw_parts_mut(iu_ptr, k);
                    let v_ju = std::slice::from_raw_parts_mut(ju_ptr, k);
                    
                    let v_il = std::slice::from_raw_parts_mut(il_ptr, k);
                    let v_jl = std::slice::from_raw_parts_mut(jl_ptr, k);
                    let v_li = std::slice::from_raw_parts_mut(li_ptr, k);

                    // Score calculation:
                    // \hat{x}_{u,l,i} = dot(v_u, v_iu) + dot(v_il, v_li)
                    // \hat{x}_{u,l,j} = dot(v_u, v_ju) + dot(v_jl, v_li)
                    
                    let p_i = dot(v_u, v_iu) + dot(v_il, v_li);
                    let p_j = dot(v_u, v_ju) + dot(v_jl, v_li);
                    let diff = p_i - p_j;

                    let sig = sigmoid(diff);
                    let deriv = 1.0 - sig;

                    for f in 0..k {
                        let w_u = v_u[f];
                        let w_iu = v_iu[f];
                        let w_ju = v_ju[f];
                        let w_il = v_il[f];
                        let w_jl = v_jl[f];
                        let w_li = v_li[f];

                        v_u[f] += learning_rate * (deriv * (w_iu - w_ju) - regularization * w_u);
                        v_iu[f] += learning_rate * (deriv * w_u - regularization * w_iu);
                        v_ju[f] += learning_rate * (deriv * (-w_u) - regularization * w_ju);
                        
                        v_li[f] += learning_rate * (deriv * (w_il - w_jl) - regularization * w_li);
                        v_il[f] += learning_rate * (deriv * w_li - regularization * w_il);
                        v_jl[f] += learning_rate * (deriv * (-w_li) - regularization * w_jl);
                    }
                }
            }
        });

        let iter_time = iter_start.elapsed().as_secs_f64();
        if verbose {
            let samples_per_sec = (samples_per_epoch as f64) / iter_time;
            println!("  {:>4} | {:>10.0} | {:>6.2}s", iter + 1, samples_per_sec, iter_time);
        }
    }

    if verbose {
        println!("  --------------------------------------");
        println!("  Total time: {:.1}s", start_time.elapsed().as_secs_f64());
    }

    (vu, viu, vil, vli)
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_users, n_items, factors, learning_rate, regularization, iterations, seed, verbose))]
pub fn fpmc_fit<'py>(
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
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>, Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;

    let (vu, viu, vil, vli) = py.detach(|| {
        fpmc_train(
            ip, ix, n_users, n_items, factors, learning_rate, regularization, iterations, seed, verbose,
        )
    });

    let a_vu = PyArray1::from_vec(py, vu).reshape([n_users, factors])?;
    let a_viu = PyArray1::from_vec(py, viu).reshape([n_items, factors])?;
    let a_vil = PyArray1::from_vec(py, vil).reshape([n_items, factors])?;
    let a_vli = PyArray1::from_vec(py, vli).reshape([n_items, factors])?;

    Ok((a_vu.into(), a_viu.into(), a_vil.into(), a_vli.into()))
}
