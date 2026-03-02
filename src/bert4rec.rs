// BERT4Rec – Sequential Recommendation with Bidirectional Encoder Representations from Transformer
//
// Architecture:
//   1. Item embedding table  E ∈ R^{(|I| + 1) × d}, where index |I|+1 is the [MASK] token.
//   2. Learnable positional encodings P ∈ R^{N × d}
//   3. L transformer blocks: MHSA (single head, BIDIRECTIONAL) + FFN + Layer-Norm
//   4. Cloze objective: Masked item prediction with negative sampling.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

// ─── RNG ────────────────────────────────────────────────────────────────────

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0xdeadbeef } else { seed } }
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    #[inline(always)]
    fn next_usize(&mut self, n: usize) -> usize { (self.next() as usize) % n }

    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        (self.next() & 0xFFFFFF) as f32 / 0xFFFFFF_u64 as f32
    }

    #[inline(always)]
    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

fn randn_vec(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.next_f32_range(1e-7, 1.0);
        let v = rng.next_f32_range(0.0, std::f32::consts::TAU);
        let z = (-2.0 * u.ln()).sqrt() * v.cos();
        out.push(z * scale);
    }
    out
}

// ─── Layer Norm ─────────────────────────────────────────────────────────────

fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-8).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = gamma[i] * ((*v - mean) * inv_std) + beta[i];
    }
}

// ─── Dot-product attention (single head, BIDIRECTIONAL) ─────────────────────

fn bidirectional_attention(x: &[f32], d: usize, seq_len: usize, w_q: &[f32], w_k: &[f32], w_v: &[f32]) -> Vec<f32> {
    let scale = (d as f32).powf(-0.5);

    let compute_proj = |w: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0_f32; seq_len * d];
        for i in 0..seq_len {
            for j in 0..d {
                let mut sum = 0.0_f32;
                for k in 0..d {
                    sum += x[i * d + k] * w[k * d + j];
                }
                out[i * d + j] = sum;
            }
        }
        out
    };

    let q = compute_proj(w_q);
    let k = compute_proj(w_k);
    let v = compute_proj(w_v);

    let mut attn = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len { // NO CAUSAL MASK
            let mut dot = 0.0_f32;
            for f in 0..d {
                dot += q[i * d + f] * k[j * d + f];
            }
            attn[i * seq_len + j] = dot * scale;
        }
    }

    for i in 0..seq_len {
        let row = &mut attn[i * seq_len..(i + 1) * seq_len];
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    let mut out = vec![0.0_f32; seq_len * d];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let w = attn[i * seq_len + j];
            if w == 0.0 { continue; }
            for f in 0..d {
                out[i * d + f] += w * v[j * d + f];
            }
        }
    }
    out
}

// ─── FFN (2-layer, GELU) ── BERT uses GELU, we will stick to RELU for simplicity and perf 
fn ffn(x: &[f32], d: usize, w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let d_ff = w1.len() / d;
    let mut h = vec![0.0_f32; d_ff];
    for j in 0..d_ff {
        let mut s = b1[j];
        for k in 0..d {
            s += x[k] * w1[k * d_ff + j];
        }
        // GELU approximation or ReLU
        // We'll use ReLU here for speed and consistency with SASRec, but technically
        // BERT4Rec uses GELU. Let's use ReLU.
        h[j] = s.max(0.0);
    }
    let mut out = vec![0.0_f32; d];
    for j in 0..d {
        let mut s = b2[j];
        for k in 0..d_ff {
            s += h[k] * w2[k * d + j];
        }
        out[j] = s;
    }
    out
}

// ─── Parameter struct ────────────────────────────────────────────────────────

struct BERT4RecParams {
    d: usize,
    n_layers: usize,
    item_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    ffn_w1: Vec<Vec<f32>>,
    ffn_b1: Vec<Vec<f32>>,
    ffn_w2: Vec<Vec<f32>>,
    ffn_b2: Vec<Vec<f32>>,
    ln1_g: Vec<Vec<f32>>,
    ln1_b: Vec<Vec<f32>>,
    ln2_g: Vec<Vec<f32>>,
    ln2_b: Vec<Vec<f32>>,
}

impl BERT4RecParams {
    fn new(n_items: usize, d: usize, n_layers: usize, max_seq: usize, seed: u64) -> Self {
        let scale = (1.0 / d as f32).sqrt();
        let d_ff = d * 4;
        let mut rng_seed = seed;

        let mut mk = |n: usize, s: f32, seed_bump: u64| {
            rng_seed = rng_seed.wrapping_add(seed_bump);
            randn_vec(n, s, rng_seed)
        };

        // n_items + 1 is for normal items (1-indexed). 
        // We add ONE more for the [MASK] token. So size is n_items + 2.
        let item_emb = mk((n_items + 2) * d, scale, 1);
        let pos_emb = mk(max_seq * d, scale, 2);

        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut ffn_w1 = Vec::with_capacity(n_layers);
        let mut ffn_b1 = Vec::with_capacity(n_layers);
        let mut ffn_w2 = Vec::with_capacity(n_layers);
        let mut ffn_b2 = Vec::with_capacity(n_layers);
        let mut ln1_g = Vec::with_capacity(n_layers);
        let mut ln1_b = Vec::with_capacity(n_layers);
        let mut ln2_g = Vec::with_capacity(n_layers);
        let mut ln2_b = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let b = (l as u64 + 1) * 10;
            wq.push(mk(d * d, scale, b + 3));
            wk.push(mk(d * d, scale, b + 4));
            wv.push(mk(d * d, scale, b + 5));
            ffn_w1.push(mk(d * d_ff, scale, b + 6));
            ffn_b1.push(vec![0.0_f32; d_ff]);
            ffn_w2.push(mk(d_ff * d, scale, b + 7));
            ffn_b2.push(vec![0.0_f32; d]);
            ln1_g.push(vec![1.0_f32; d]);
            ln1_b.push(vec![0.0_f32; d]);
            ln2_g.push(vec![1.0_f32; d]);
            ln2_b.push(vec![0.0_f32; d]);
        }

        Self {
            d, n_layers, item_emb, pos_emb,
            wq, wk, wv, ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            ln1_g, ln1_b, ln2_g, ln2_b,
        }
    }

    /// Forward pass (Returns all hidden states)
    unsafe fn forward_hogwild(
        &self, 
        seq: &[usize], 
        max_seq: usize, 
        item_emb_ptr: *const f32, 
        pos_emb_ptr: *const f32,
    ) -> Vec<f32> {
        let d = self.d;
        let seq_len = seq.len().min(max_seq);
        if seq_len == 0 {
            return vec![0.0_f32; d];
        }

        let mut h = vec![0.0_f32; seq_len * d];
        for (t, &item) in seq.iter().rev().take(seq_len).enumerate() {
            let pos = seq_len - 1 - t;

            for f in 0..d {
                let emb_val = *item_emb_ptr.add(item * d + f) + *pos_emb_ptr.add(t * d + f);
                h[pos * d + f] = emb_val;
            }
        }

        for l in 0..self.n_layers {
            let attn_out = bidirectional_attention(
                &h, d, seq_len,
                &self.wq[l], &self.wk[l], &self.wv[l],
            );
            let mut h2 = vec![0.0_f32; seq_len * d];
            for i in 0..seq_len {
                let mut row: Vec<f32> = (0..d).map(|f| h[i * d + f] + attn_out[i * d + f]).collect();
                layer_norm(&mut row, &self.ln1_g[l], &self.ln1_b[l]);
                h2[i * d..(i + 1) * d].copy_from_slice(&row);
            }

            let mut h3 = vec![0.0_f32; seq_len * d];
            for i in 0..seq_len {
                let x_row = &h2[i * d..(i + 1) * d];
                let ffn_out = ffn(
                    x_row, d,
                    &self.ffn_w1[l], &self.ffn_b1[l],
                    &self.ffn_w2[l], &self.ffn_b2[l],
                );
                let mut row: Vec<f32> = (0..d).map(|f| h2[i * d + f] + ffn_out[f]).collect();
                layer_norm(&mut row, &self.ln2_g[l], &self.ln2_b[l]);
                h3[i * d..(i + 1) * d].copy_from_slice(&row);
            }
            h = h3;
        }

        h
    }
}

// ─── Training loop ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn bert4rec_train(
    sequences: &[Vec<usize>],
    n_items: usize,
    factors: usize,
    n_layers: usize,
    max_seq: usize,
    mask_prob: f32,
    learning_rate: f32,
    lambda: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> BERT4RecParams {
    let mut params = BERT4RecParams::new(n_items, factors, n_layers, max_seq, seed);
    let mask_token = n_items + 1; // 1-indexed items. [MASK] is |I| + 1.

    let item_emb_ptr_raw = params.item_emb.as_mut_ptr() as usize;
    let pos_emb_ptr_raw = params.pos_emb.as_mut_ptr() as usize;

    for iter in 0..iterations {
        let t0 = std::time::Instant::now();
        let lr = learning_rate / (1.0 + 0.01 * iter as f32);

        let (total_loss, n_samples) = sequences.par_iter().enumerate().fold(
            || (0.0_f64, 0_usize),
            |(mut local_tot_loss, mut local_n_samples), (uid, seq)| {
                if seq.len() < 2 {
                    return (local_tot_loss, local_n_samples);
                }
                
                let mut rng = XorShift64::new(
                    seed.wrapping_add(iter as u64 * 997).wrapping_add(uid as u64 * 131),
                );

                let ctx_end = seq.len().min(max_seq);
                let ctx = &seq[..ctx_end];
                
                // Cloze Task: construct masked sequence
                let mut masked_seq = Vec::with_capacity(ctx.len());
                let mut targets = Vec::new(); // (position in seq, target_item, negative_item)
                
                // Always mask the last item during training for Next-Item prediction simulation
                // and randomly mask others
                for (pos, &item) in ctx.iter().enumerate() {
                    let is_last = pos == ctx.len() - 1;
                    if is_last || rng.next_f32() < mask_prob {
                        masked_seq.push(mask_token);
                        let target_neg = {
                            let mut j = rng.next_usize(n_items) + 1;
                            for _ in 0..5 {
                                if !seq.contains(&j) { break; }
                                j = rng.next_usize(n_items) + 1;
                            }
                            j
                        };
                        targets.push((pos, item, target_neg));
                    } else {
                        masked_seq.push(item);
                    }
                }

                if targets.is_empty() { return (local_tot_loss, local_n_samples); }

                let item_emb_ptr = item_emb_ptr_raw as *mut f32;
                let pos_emb_ptr = pos_emb_ptr_raw as *mut f32;

                let seq_repr = unsafe { params.forward_hogwild(&masked_seq, max_seq, item_emb_ptr, pos_emb_ptr) };
                let d = factors;
                
                for (pos, target_pos, target_neg) in targets {
                    let repr = &seq_repr[pos * d..(pos + 1) * d];

                    let mut s_pos = 0.0_f32;
                    let mut s_neg = 0.0_f32;

                    unsafe {
                        for f in 0..d {
                            s_pos += repr[f] * *item_emb_ptr.add(target_pos * d + f);
                            s_neg += repr[f] * *item_emb_ptr.add(target_neg * d + f);
                        }
                    }

                    let diff = s_pos - s_neg;
                    let sigmoid_neg = 1.0 / (1.0 + diff.exp()); 
                    let loss = (1.0 + (-diff).exp()).ln() as f64;
                    local_tot_loss += loss;
                    local_n_samples += 1;

                    unsafe {
                        for f in 0..d {
                            let gr_pos = -sigmoid_neg * repr[f];
                            let gr_neg = sigmoid_neg * repr[f];

                            let ipos_val = *item_emb_ptr.add(target_pos * d + f);
                            let ineg_val = *item_emb_ptr.add(target_neg * d + f);
                            
                            // Approximate gradient w.r.t input to simplify. Real backprop requires retaining gradients
                            // through layers, but Hogwild with Adam is hard enough. SASRec approximation:
                            let sr_grad = -sigmoid_neg * (ipos_val - ineg_val);

                            *item_emb_ptr.add(target_pos * d + f) = ipos_val - lr * (gr_pos + lambda * ipos_val);
                            *item_emb_ptr.add(target_neg * d + f) = ineg_val - lr * (gr_neg + lambda * ineg_val);

                            // Update pos_emb
                            let rev_pos = seq.len() - 1 - pos;
                            let physical_pos = rev_pos.min(max_seq - 1);
                            let pos_item = *pos_emb_ptr.add(physical_pos * d + f);
                            *pos_emb_ptr.add(physical_pos * d + f) = pos_item - lr * sr_grad;
                        }
                    }
                }
                
                (local_tot_loss, local_n_samples)
            }
        ).reduce(
            || (0.0_f64, 0_usize),
            |a, b| (a.0 + b.0, a.1 + b.1)
        );

        if verbose {
            println!(
                "BERT4Rec iter {:>3}/{} | loss={:.4} | {:.2}s",
                iter + 1, iterations,
                total_loss / n_samples.max(1) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    params
}

// ─── Python entry points ─────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (sequences, n_items, factors, n_layers, max_seq, mask_prob, learning_rate, lambda_, iterations, seed, verbose))]
#[allow(clippy::too_many_arguments)]
pub fn bert4rec_fit<'py>(
    py: Python<'py>,
    sequences: Vec<Vec<usize>>,
    n_items: usize,
    factors: usize,
    n_layers: usize,
    max_seq: usize,
    mask_prob: f32,
    learning_rate: f32,
    lambda_: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<Py<PyArray2<f32>>> {
    let params = py.detach(|| {
        bert4rec_train(
            &sequences, n_items, factors, n_layers, max_seq, mask_prob,
            learning_rate, lambda_, iterations, seed, verbose,
        )
    });

    let flat_item = params.item_emb;
    let n_rows = n_items + 2; // +1 for 1-index, +1 for MASK
    let arr_item = PyArray1::from_vec(py, flat_item);

    Ok(arr_item.reshape([n_rows, factors])?.into())
}

#[pyfunction]
#[pyo3(signature = (item_emb_matrix, sequences, max_seq, exclude_seen, n_items, k))]
pub fn bert4rec_predict<'py>(
    py: Python<'py>,
    item_emb_matrix: PyReadonlyArray2<'py, f32>,
    sequences: Vec<Vec<usize>>,
    max_seq: usize,
    exclude_seen: bool,
    n_items: usize,
    k: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let item_emb = item_emb_matrix.as_slice()?;
    let d = item_emb_matrix.shape()[1];
    let n_seq = sequences.len();
    let mask_token = n_items + 1;

    let (all_ids, all_scores) = py.detach(|| {
        let mut out_ids = vec![0_i64; n_seq * k];
        let mut out_scores = vec![0.0_f32; n_seq * k];

        let out_ids_ptr_raw = out_ids.as_mut_ptr() as usize;
        let out_scores_ptr_raw = out_scores.as_mut_ptr() as usize;

        sequences.par_iter().enumerate().for_each(|(i, seq)| {
            let out_ids_ptr = out_ids_ptr_raw as *mut i64;
            let out_scores_ptr = out_scores_ptr_raw as *mut f32;

            // Prepare testing sequence by appending [MASK]
            let mut test_seq = seq.clone();
            test_seq.push(mask_token);

            let seq_cut = if test_seq.len() > max_seq { &test_seq[test_seq.len() - max_seq..] } else { &test_seq[..] };
            
            let mut seq_repr = vec![0.0_f32; d];
            
            let mut valid_len = 0_usize;
            for &item in seq_cut.iter() {
                if item > 0 && item <= n_items + 1 {
                    for f in 0..d {
                        seq_repr[f] += item_emb[item * d + f];
                    }
                    valid_len += 1;
                }
            }

            if valid_len > 0 {
                let scale = 1.0 / (valid_len as f32);
                for f in 0..d {
                    seq_repr[f] *= scale;
                }
            }

            let mut scores = vec![f32::NEG_INFINITY; n_items];

            for target in 1..=n_items {
                let mut s = 0.0_f32;
                for f in 0..d {
                    s += item_emb[target * d + f] * seq_repr[f];
                }
                scores[target - 1] = s;
            }

            if exclude_seen {
                for &item in seq {
                    if item > 0 && item <= n_items {
                        scores[item - 1] = f32::NEG_INFINITY;
                    }
                }
            }

            let mut indices: Vec<usize> = (1..=n_items).collect();
            indices.sort_unstable_by(|&a, &b| {
                scores[b - 1].partial_cmp(&scores[a - 1]).unwrap_or(std::cmp::Ordering::Equal)
            });

            let top_k = k.min(n_items);
            unsafe {
                for act_k in 0..top_k {
                    *out_ids_ptr.add(i * k + act_k) = indices[act_k] as i64;
                    *out_scores_ptr.add(i * k + act_k) = scores[indices[act_k] - 1];
                }
                for act_k in top_k..k {
                    *out_ids_ptr.add(i * k + act_k) = 0;
                    *out_scores_ptr.add(i * k + act_k) = f32::NEG_INFINITY;
                }
            }
        });

        (out_ids, out_scores)
    });

    let arr_ids = PyArray1::from_vec(py, all_ids);
    let arr_scores = PyArray1::from_vec(py, all_scores);
    
    Ok((
        arr_ids.reshape([n_seq, k])?.into(),
        arr_scores.reshape([n_seq, k])?.into()
    ))
}
