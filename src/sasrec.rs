// SASRec – Self-Attentive Sequential Recommendation (simplified Rust implementation)
// 
// Architecture:
//   1. Item embedding table  E ∈ R^{|I| × d}
//   2. Learnable positional encodings P ∈ R^{N × d}
//   3. L transformer blocks: MHSA (single head, causal) + FFN + Layer-Norm
//   4. Next-item binary-cross-entropy loss over positive / sampled-negative
//
// All math is done with raw Vec<f32> to avoid external crate dependencies.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

// ─── RNG (same as bpr.rs) ───────────────────────────────────────────────────

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
        // Box-Muller approx: map two uniforms via a lookup table approximation
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

// ─── Dot-product attention (single head, causal) ────────────────────────────

/// Compute attention over a single sequence of length `seq_len`.
/// x:   [seq_len × d] row-major
/// Returns y: [seq_len × d]
fn causal_attention(x: &[f32], d: usize, seq_len: usize, w_q: &[f32], w_k: &[f32], w_v: &[f32]) -> Vec<f32> {
    let scale = (d as f32).powf(-0.5);

    // Compute Q, K, V  [seq × d]
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

    // Attention weights  [seq × seq], causal mask
    let mut attn = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            let mut dot = 0.0_f32;
            for f in 0..d {
                dot += q[i * d + f] * k[j * d + f];
            }
            attn[i * seq_len + j] = dot * scale;
        }
    }

    // Row-wise softmax
    for i in 0..seq_len {
        let row = &mut attn[i * seq_len..(i + 1) * seq_len];
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }

    // Output = attn @ V
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

// ─── FFN (2-layer, ReLU) ─────────────────────────────────────────────────────

fn ffn(x: &[f32], d: usize, w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let d_ff = w1.len() / d;  // hidden dim
    let mut h = vec![0.0_f32; d_ff];
    for j in 0..d_ff {
        let mut s = b1[j];
        for k in 0..d {
            s += x[k] * w1[k * d_ff + j];
        }
        h[j] = s.max(0.0); // ReLU
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

struct SASRecParams {
    d: usize,
    n_layers: usize,
    item_emb: Vec<f32>,   // (n_items+1) × d — index 0 is padding
    pos_emb: Vec<f32>,    // max_seq × d
    // per layer: attn Q/K/V weights [d×d], FFN weights, layer-norm params
    wq: Vec<Vec<f32>>,    // n_layers × [d×d]
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    ffn_w1: Vec<Vec<f32>>, // n_layers × [d × 4d]
    ffn_b1: Vec<Vec<f32>>,
    ffn_w2: Vec<Vec<f32>>, // n_layers × [4d × d]
    ffn_b2: Vec<Vec<f32>>,
    ln1_g: Vec<Vec<f32>>, // LayerNorm after attn
    ln1_b: Vec<Vec<f32>>,
    ln2_g: Vec<Vec<f32>>, // LayerNorm after FFN
    ln2_b: Vec<Vec<f32>>,
}

impl SASRecParams {
    fn new(n_items: usize, d: usize, n_layers: usize, max_seq: usize, seed: u64) -> Self {
        let scale = (1.0 / d as f32).sqrt();
        let d_ff = d * 4;
        let mut rng_seed = seed;

        let mut mk = |n: usize, s: f32, seed_bump: u64| {
            rng_seed = rng_seed.wrapping_add(seed_bump);
            randn_vec(n, s, rng_seed)
        };

        let item_emb = mk((n_items + 1) * d, scale, 1);
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
            d,
            n_layers,
            item_emb,
            pos_emb,
            wq, wk, wv,
            ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            ln1_g, ln1_b,
            ln2_g, ln2_b,
        }
    }

    /// Forward pass: encode a sequence of item indices → output vector at last position
    fn forward(&self, seq: &[usize], max_seq: usize) -> Vec<f32> {
        let d = self.d;
        let seq_len = seq.len().min(max_seq);
        if seq_len == 0 {
            return vec![0.0_f32; d];
        }

        // Build input: item_emb + pos_emb
        let mut h = vec![0.0_f32; seq_len * d];
        for (t, &item) in seq.iter().rev().take(seq_len).enumerate() {
            let pos = seq_len - 1 - t; // most recent → position 0
            for f in 0..d {
                h[pos * d + f] = self.item_emb[item * d + f] + self.pos_emb[t * d + f];
            }
        }

        // Transformer layers
        for l in 0..self.n_layers {
            // Attn sublayer
            let attn_out = causal_attention(
                &h, d, seq_len,
                &self.wq[l], &self.wk[l], &self.wv[l],
            );
            // Residual + LayerNorm
            let mut h2 = vec![0.0_f32; seq_len * d];
            for i in 0..seq_len {
                let mut row: Vec<f32> = (0..d).map(|f| h[i * d + f] + attn_out[i * d + f]).collect();
                layer_norm(&mut row, &self.ln1_g[l], &self.ln1_b[l]);
                h2[i * d..(i + 1) * d].copy_from_slice(&row);
            }

            // FFN sublayer
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

        // Return the last (most recent) position vector
        h[(seq_len - 1) * d..seq_len * d].to_vec()
    }

    /// Score a user sequence against a target item
    fn score(&self, seq_repr: &[f32], item: usize) -> f32 {
        let d = self.d;
        let emb = &self.item_emb[item * d..(item + 1) * d];
        seq_repr.iter().zip(emb).map(|(a, b)| a * b).sum()
    }
}

// ─── Training loop ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn sasrec_train(
    sequences: &[Vec<usize>],
    n_items: usize,
    factors: usize,
    n_layers: usize,
    max_seq: usize,
    learning_rate: f32,
    lambda: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> SASRecParams {
    let mut params = SASRecParams::new(n_items, factors, n_layers, max_seq, seed);
    let _n_users = sequences.len();

    // Simplified SGD: for each user, take all consecutive (seq, next_item) pairs
    // Use stochastic gradient descent on binary cross-entropy loss (positive vs random negative)
    // This is a single-pass approximation per epoch.

    for iter in 0..iterations {
        let t0 = std::time::Instant::now();
        let mut total_loss = 0.0_f64;
        let mut n_samples = 0usize;

        let lr = learning_rate / (1.0 + 0.01 * iter as f32);

        for (uid, seq) in sequences.iter().enumerate() {
            if seq.len() < 2 {
                continue;
            }
            let mut rng = XorShift64::new(
                seed.wrapping_add(iter as u64 * 997).wrapping_add(uid as u64 * 131),
            );

            // Take the full history as context, predict the last item
            let ctx_end = seq.len().min(max_seq + 1);
            let ctx = &seq[..ctx_end.saturating_sub(1)];
            let target_pos = seq[ctx_end - 1];
            let target_neg = {
                let mut j = rng.next_usize(n_items) + 1; // 1-indexed
                for _ in 0..5 {
                    if !seq.contains(&j) { break; }
                    j = rng.next_usize(n_items) + 1;
                }
                j
            };

            if ctx.is_empty() { continue; }

            let seq_repr = params.forward(ctx, max_seq);
            let s_pos = params.score(&seq_repr, target_pos);
            let s_neg = params.score(&seq_repr, target_neg);

            let diff = s_pos - s_neg;
            // BCE loss gradient
            let sigmoid_neg = 1.0 / (1.0 + diff.exp()); // d(-log(sigmoid)) / d(diff)
            let loss = (1.0 + (-diff).exp()).ln() as f64;
            total_loss += loss;
            n_samples += 1;

            // Gradient w.r.t. item embeddings (simplified: only update final layer output path)
            let d = factors;
            for f in 0..d {
                let gr_pos = -sigmoid_neg * seq_repr[f];
                let gr_neg = sigmoid_neg * seq_repr[f];
                let sr_grad = -sigmoid_neg * (params.item_emb[target_pos * d + f] - params.item_emb[target_neg * d + f]);

                params.item_emb[target_pos * d + f] -= lr * (gr_pos + lambda * params.item_emb[target_pos * d + f]);
                params.item_emb[target_neg * d + f] -= lr * (gr_neg + lambda * params.item_emb[target_neg * d + f]);

                // Update positional embedding at the last position too
                if ctx.len() > 0 {
                    let last_pos = (ctx.len() - 1).min(max_seq - 1);
                    params.pos_emb[last_pos * d + f] -= lr * sr_grad;
                }
            }
        }

        if verbose {
            println!(
                "SASRec iter {:>3}/{} | loss={:.4} | {:.2}s",
                iter + 1, iterations,
                total_loss / n_samples.max(1) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    params
}

// ─── Python entry points ─────────────────────────────────────────────────────

/// Train SASRec and return item embedding matrix (n_items+1) × d.
/// Sequences are 1-indexed item IDs (0 = padding token).
#[pyfunction]
#[pyo3(signature = (sequences, n_items, factors, n_layers, max_seq, learning_rate, lambda_, iterations, seed, verbose))]
#[allow(clippy::too_many_arguments)]
pub fn sasrec_fit<'py>(
    py: Python<'py>,
    sequences: Vec<Vec<usize>>,
    n_items: usize,
    factors: usize,
    n_layers: usize,
    max_seq: usize,
    learning_rate: f32,
    lambda_: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<Py<PyArray2<f32>>> {
    let params = py.detach(|| {
        sasrec_train(
            &sequences, n_items, factors, n_layers, max_seq,
            learning_rate, lambda_, iterations, seed, verbose,
        )
    });

    let flat = params.item_emb;
    let n_rows = n_items + 1;
    let arr = PyArray1::from_vec(py, flat);
    Ok(arr.reshape([n_rows, factors])?.into())
}

/// Run forward pass for a batch of sequences and return last-position vectors.
#[pyfunction]
#[pyo3(signature = (item_emb_matrix, pos_emb_matrix, sequences, max_seq))]
pub fn sasrec_encode<'py>(
    py: Python<'py>,
    item_emb_matrix: PyReadonlyArray2<'py, f32>,
    pos_emb_matrix: PyReadonlyArray2<'py, f32>,
    sequences: Vec<Vec<usize>>,
    max_seq: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let item_emb = item_emb_matrix.as_slice()?;
    let pos_emb = pos_emb_matrix.as_slice()?;
    let d = item_emb_matrix.shape()[1];
    let n_seq = sequences.len();

    let out: Vec<f32> = sequences
        .par_iter()
        .flat_map(|seq| {
            let seq_len = seq.len().min(max_seq);
            if seq_len == 0 {
                return vec![0.0_f32; d];
            }
            let mut h = vec![0.0_f32; seq_len * d];
            for (t, &item) in seq.iter().rev().take(seq_len).enumerate() {
                let pos = seq_len - 1 - t;
                for f in 0..d {
                    h[pos * d + f] = item_emb[item * d + f] + pos_emb[t * d + f];
                }
            }
            // Return the last position embedding (simplified, no attention layers)
            h[(seq_len - 1) * d..seq_len * d].to_vec()
        })
        .collect();

    let arr = PyArray1::from_vec(py, out);
    Ok(arr.reshape([n_seq, d])?.into())
}
