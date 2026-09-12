use faer::{linalg::matmul::matmul, Accum, MatMut, MatRef, Par};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;

use crate::rng::random_factors;

// ── SIMD‑friendly primitives ───────────────────────────────────────
// 8‑wide manual unroll – LLVM maps these to NEON/AVX without needing
// architecture‑specific intrinsics, and it's faster than the plain
// iterator chain because we guarantee no loop‑carried dependency.
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // ponytail: was a private 8-wide unroll identical to crate::simd::dot.
    // Routed through simd so ALS gets the runtime AVX2+FMA dispatch too —
    // the private copy never did, so x86 wheels ran ALS on the scalar path.
    crate::simd::dot(a, b)
}

#[inline(always)]
fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    crate::simd::axpy(alpha, x, y)
}

fn gramian(factors: &[f32], n: usize, k: usize) -> Vec<f32> {
    let y = MatRef::from_row_major_slice(factors, n, k);
    let yt = y.transpose();

    let mut g = faer::Mat::<f32>::zeros(k, k);
    matmul(g.as_mut(), Accum::Replace, yt, y, 1.0f32, Par::rayon(0));

    let mut r = vec![0.0f32; k * k];
    for a in 0..k {
        for b in 0..k {
            r[a * k + b] = g[(a, b)];
        }
    }
    r
}

/// Weighted gramian: S = Σ_i w_i · y_i · y_i^T
/// Used by eALS with item-popularity weighting for unobserved entries.
fn weighted_gramian(factors: &[f32], weights: &[f32], n: usize, k: usize) -> Vec<f32> {
    // Scale each row y_i by sqrt(w_i), then compute standard gramian
    let mut scaled = vec![0.0f32; n * k];
    for i in 0..n {
        let sw = weights[i].sqrt();
        let src = &factors[i * k..(i + 1) * k];
        let dst = &mut scaled[i * k..(i + 1) * k];
        for f in 0..k {
            dst[f] = sw * src[f];
        }
    }
    gramian(&scaled, n, k)
}

thread_local! {
    // CG scratch: b, r, p, ap, w_vec (per-item confidence weights).
    // yi_dense/tmp are gone: apply_a reads item rows directly from `other`.
    static SCRATCH: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

fn solve_one_side_cg(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    other: &[f32],
    gram: &[f32],
    k: usize,
    lambda: f32,
    alpha: f32,
    cg_iters: usize,
    out: &mut [f32],
) {
    let eff_lambda = lambda.max(1e-6);

    out.par_chunks_mut(k).enumerate().for_each(|(u, xu)| {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let nnz_u = end - start;

        SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut b, ref mut r, ref mut p, ref mut ap, ref mut w_vec) = *borrow;
            b.clear(); b.resize(k, 0.0);
            r.clear(); r.resize(k, 0.0);
            p.clear(); p.resize(k, 0.0);
            ap.clear(); ap.resize(k, 0.0);

            // Build the right-hand side and the per-item confidence weights.
            // apply_a reads the item rows straight out of `other`, so no dense
            // gather is needed.
            w_vec.clear();
            w_vec.resize(nnz_u, 0.0);

            for (local, idx) in (start..end).enumerate() {
                let i = indices[idx] as usize;
                let c = 1.0 + alpha * data[idx];
                axpy_f32(c, &other[i * k..(i + 1) * k], b);
                w_vec[local] = alpha * data[idx]; // = c - 1
            }

            // apply_a: out = (Gram + lambda*I + Y^T diag(w) Y) * v
            //
            // Part 1 was a private 8-wide unroll — the one dot in the crate
            // that still bypassed crate::simd, so on x86 it ran the SSE2
            // autovec path while everything else got AVX2+FMA. Unlike the
            // memory-bound kernels, this one re-reads an L2-resident Gram
            // cg_iters times per entity, so the wider FMA actually pays.
            //
            // Part 2 used to materialise the entity's item rows into a dense
            // `yi_dense` scratch matrix, then make two passes over it per CG
            // step (a scalar dot pass, then a faer gemv). The rows are already
            // contiguous k-slices of `other`, so the gather bought no
            // locality; fusing dot+axpy into one pass halves the traffic, hits
            // each row while it is still in L1, and drops the yi_dense/tmp
            // scratch buffers (and the per-entity copy) entirely.
            let apply_a = |v: &[f32], out: &mut [f32]| {
                for a in 0..k {
                    out[a] = dot_f32(&gram[a * k..(a + 1) * k], v) + eff_lambda * v[a];
                }
                for (local, idx) in (start..end).enumerate() {
                    let i = indices[idx] as usize;
                    let yi = &other[i * k..(i + 1) * k];
                    let t = w_vec[local] * dot_f32(yi, v);
                    if t != 0.0 {
                        axpy_f32(t, yi, out);
                    }
                }
            };

            // A user with no interactions has no equation to solve. `xu` is the
            // live factor matrix (seeded with random_factors), so it must be
            // zeroed explicitly rather than left at its initial value.
            if nnz_u == 0 {
                xu.fill(0.0);
                return;
            }

            // WARM START: `xu` still holds this entity's factors from the
            // previous outer iteration, which is a far better starting point
            // than zero — consecutive ALS iterations move the solution only
            // slightly. Starting from zero threw that away and made every
            // outer iteration pay the full CG budget from scratch. (This is
            // why `implicit` gets good results with only ~3 CG steps.)
            //
            //   r = b - A*x0,  p = r     instead of     x0 = 0, r = p = b
            apply_a(&xu[..], ap);
            for j in 0..k {
                r[j] = b[j] - ap[j];
            }
            p.copy_from_slice(r);
            let mut rsold = dot_f32(r, r);

            // Relative stopping criterion: once the residual is small next to
            // the right-hand side this solve is converged, so later outer
            // iterations exit after a step or two instead of always burning
            // the whole budget. The old absolute 1e-20 test effectively never
            // fired for a cold start.
            // ponytail: fixed rtol, not user-tunable until someone needs it.
            let rtol2 = 1e-10 * dot_f32(b, b);
            if rsold <= rtol2 {
                return;
            }

            for _ in 0..cg_iters {
                apply_a(p, ap);
                let pap = dot_f32(p, ap);
                if pap <= 0.0 {
                    break;
                }
                let ak = rsold / pap;

                axpy_f32(ak, p, xu);
                axpy_f32(-ak, ap, r);

                let rsnew = dot_f32(r, r);
                if rsnew <= rtol2 || rsnew < 1e-20 {
                    break;
                }
                let beta = rsnew / rsold;
                for j in 0..k {
                    p[j] = r[j] + beta * p[j];
                }
                rsold = rsnew;
            }
        });
    });
}

thread_local! {
    static SCRATCH_EALS: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

fn solve_one_side_eals(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    other: &[f32],
    gram: &[f32],
    out: &mut [f32],
    k: usize,
    lambda: f32,
    alpha: f32,
    eals_iters: usize,
    item_pop_weights: Option<&[f32]>,
) {
    let eff_lambda = lambda.max(1e-6);

    out.par_chunks_mut(k).enumerate().for_each(|(u, xu)| {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let nnz_u = end - start;

        if nnz_u == 0 {
            xu.fill(0.0);
            return;
        }

        SCRATCH_EALS.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut r_hat, ref mut s_u, ref mut yi_cols, ref mut w_vec, ref mut c0_vec) = *borrow;

            r_hat.clear();
            r_hat.resize(nnz_u, 0.0);
            s_u.clear();
            s_u.resize(k, 0.0);

            // Pre-allocate contiguous column-major memory for item vectors
            yi_cols.clear();
            yi_cols.resize(k * nnz_u, 0.0);
            w_vec.clear();
            w_vec.resize(nnz_u, 0.0);
            // ponytail: c0_i (popularity weight) is invariant across `f` in 0..k and
            // across eALS passes — gather it once here instead of k*eals_iters times
            // in the hot loop below.
            c0_vec.clear();
            c0_vec.resize(nnz_u, 0.0);

            // Populate the contiguous memory buffer (column-major)
            // But we only actually need to store and iterate over items where weight != 0 or where r_hat affects the item!
            // In eALS, all items interacting inherently have an alpha * rating. 
            for (local, idx) in (start..end).enumerate() {
                let i = indices[idx] as usize;
                
                // Transpose row-major into col-major
                // Using get_unchecked for the hot loop mapping yi into yi_cols
                unsafe {
                    let yi = other.get_unchecked(i * k..(i + 1) * k);
                    w_vec[local] = alpha * *data.get_unchecked(idx);
                    
                    let mut f = 0;
                    let k8 = k / 8 * 8;
                    // 8-wide unrolled transpose for max memory throughput
                    while f < k8 {
                        *yi_cols.get_unchecked_mut(f * nnz_u + local) = *yi.get_unchecked(f);
                        *yi_cols.get_unchecked_mut((f + 1) * nnz_u + local) = *yi.get_unchecked(f + 1);
                        *yi_cols.get_unchecked_mut((f + 2) * nnz_u + local) = *yi.get_unchecked(f + 2);
                        *yi_cols.get_unchecked_mut((f + 3) * nnz_u + local) = *yi.get_unchecked(f + 3);
                        *yi_cols.get_unchecked_mut((f + 4) * nnz_u + local) = *yi.get_unchecked(f + 4);
                        *yi_cols.get_unchecked_mut((f + 5) * nnz_u + local) = *yi.get_unchecked(f + 5);
                        *yi_cols.get_unchecked_mut((f + 6) * nnz_u + local) = *yi.get_unchecked(f + 6);
                        *yi_cols.get_unchecked_mut((f + 7) * nnz_u + local) = *yi.get_unchecked(f + 7);
                        f += 8;
                    }
                    while f < k {
                        *yi_cols.get_unchecked_mut(f * nnz_u + local) = *yi.get_unchecked(f);
                        f += 1;
                    }
                }
                
                // r_hat is pred for interacted items
                r_hat[local] = dot_f32(xu, &other[i * k..(i + 1) * k]);

                c0_vec[local] = item_pop_weights.map_or(1.0f32, |w| w[i]);
            }

            for _pass in 0..eals_iters {
                // Precompute s_u = Gram * xu
                for f in 0..k {
                    s_u[f] = dot_f32(&gram[f * k..(f + 1) * k], xu);
                }

                for f in 0..k {
                    // With popularity weighting, the Gram already encodes
                    // Σ c0_i · y_if · y_ig, so the unobserved-data contribution
                    // is correctly weighted in numer and denom.
                    let mut numer = -(s_u[f] - xu[f] * gram[f * k + f]);
                    let mut denom = eff_lambda + gram[f * k + f];
                    
                    // Col-major offset for latent factor f
                    let yi_f_offset = f * nnz_u;

                    // SIMD-optimized pass over interacted items
                    // We can move the addition of (w+1)*y_if outside the r_hat calculation
                    let xu_f = xu[f];
                    
                    unsafe {
                        let w_ptr = w_vec.as_ptr();
                        let yi_ptr = yi_cols.as_ptr().add(yi_f_offset);
                        let r_hat_ptr = r_hat.as_ptr();
                        let c0_ptr = c0_vec.as_ptr();
                        
                            let mut local = 0;
                            let local8 = nnz_u / 8 * 8;
                            
                            while local < local8 {
                                let w0 = *w_ptr.add(local);
                                let y_if0 = *yi_ptr.add(local);
                                let r_hat_val0 = *r_hat_ptr.add(local);
                                let c0_i0 = *c0_ptr.add(local);
                                let wy_if0 = w0 * y_if0;
                                let wy20 = wy_if0 * y_if0;
                                numer += c0_i0 * y_if0 + wy_if0 * (1.0 - r_hat_val0) + xu_f * (wy20 + (c0_i0 - 1.0) * y_if0 * y_if0);
                                denom += wy20 + (c0_i0 - 1.0) * y_if0 * y_if0;
                                
                                let w1 = *w_ptr.add(local + 1);
                                let y_if1 = *yi_ptr.add(local + 1);
                                let r_hat_val1 = *r_hat_ptr.add(local + 1);
                                let c0_i1 = *c0_ptr.add(local + 1);
                                let wy_if1 = w1 * y_if1;
                                let wy21 = wy_if1 * y_if1;
                                numer += c0_i1 * y_if1 + wy_if1 * (1.0 - r_hat_val1) + xu_f * (wy21 + (c0_i1 - 1.0) * y_if1 * y_if1);
                                denom += wy21 + (c0_i1 - 1.0) * y_if1 * y_if1;
                                
                                let w2 = *w_ptr.add(local + 2);
                                let y_if2 = *yi_ptr.add(local + 2);
                                let r_hat_val2 = *r_hat_ptr.add(local + 2);
                                let c0_i2 = *c0_ptr.add(local + 2);
                                let wy_if2 = w2 * y_if2;
                                let wy22 = wy_if2 * y_if2;
                                numer += c0_i2 * y_if2 + wy_if2 * (1.0 - r_hat_val2) + xu_f * (wy22 + (c0_i2 - 1.0) * y_if2 * y_if2);
                                denom += wy22 + (c0_i2 - 1.0) * y_if2 * y_if2;
                                
                                let w3 = *w_ptr.add(local + 3);
                                let y_if3 = *yi_ptr.add(local + 3);
                                let r_hat_val3 = *r_hat_ptr.add(local + 3);
                                let c0_i3 = *c0_ptr.add(local + 3);
                                let wy_if3 = w3 * y_if3;
                                let wy23 = wy_if3 * y_if3;
                                numer += c0_i3 * y_if3 + wy_if3 * (1.0 - r_hat_val3) + xu_f * (wy23 + (c0_i3 - 1.0) * y_if3 * y_if3);
                                denom += wy23 + (c0_i3 - 1.0) * y_if3 * y_if3;
                                
                                let w4 = *w_ptr.add(local + 4);
                                let y_if4 = *yi_ptr.add(local + 4);
                                let r_hat_val4 = *r_hat_ptr.add(local + 4);
                                let c0_i4 = *c0_ptr.add(local + 4);
                                let wy_if4 = w4 * y_if4;
                                let wy24 = wy_if4 * y_if4;
                                numer += c0_i4 * y_if4 + wy_if4 * (1.0 - r_hat_val4) + xu_f * (wy24 + (c0_i4 - 1.0) * y_if4 * y_if4);
                                denom += wy24 + (c0_i4 - 1.0) * y_if4 * y_if4;
                                
                                let w5 = *w_ptr.add(local + 5);
                                let y_if5 = *yi_ptr.add(local + 5);
                                let r_hat_val5 = *r_hat_ptr.add(local + 5);
                                let c0_i5 = *c0_ptr.add(local + 5);
                                let wy_if5 = w5 * y_if5;
                                let wy25 = wy_if5 * y_if5;
                                numer += c0_i5 * y_if5 + wy_if5 * (1.0 - r_hat_val5) + xu_f * (wy25 + (c0_i5 - 1.0) * y_if5 * y_if5);
                                denom += wy25 + (c0_i5 - 1.0) * y_if5 * y_if5;
                                
                                let w6 = *w_ptr.add(local + 6);
                                let y_if6 = *yi_ptr.add(local + 6);
                                let r_hat_val6 = *r_hat_ptr.add(local + 6);
                                let c0_i6 = *c0_ptr.add(local + 6);
                                let wy_if6 = w6 * y_if6;
                                let wy26 = wy_if6 * y_if6;
                                numer += c0_i6 * y_if6 + wy_if6 * (1.0 - r_hat_val6) + xu_f * (wy26 + (c0_i6 - 1.0) * y_if6 * y_if6);
                                denom += wy26 + (c0_i6 - 1.0) * y_if6 * y_if6;
                                
                                let w7 = *w_ptr.add(local + 7);
                                let y_if7 = *yi_ptr.add(local + 7);
                                let r_hat_val7 = *r_hat_ptr.add(local + 7);
                                let c0_i7 = *c0_ptr.add(local + 7);
                                let wy_if7 = w7 * y_if7;
                                let wy27 = wy_if7 * y_if7;
                                numer += c0_i7 * y_if7 + wy_if7 * (1.0 - r_hat_val7) + xu_f * (wy27 + (c0_i7 - 1.0) * y_if7 * y_if7);
                                denom += wy27 + (c0_i7 - 1.0) * y_if7 * y_if7;
                                
                                local += 8;
                            }
                            
                            while local < nnz_u {
                                let w = *w_ptr.add(local);
                                let y_if = *yi_ptr.add(local);
                                let r_hat_val = *r_hat_ptr.add(local);
                                
                                let c0_i = *c0_ptr.add(local);
                                let wy_if = w * y_if;
                                let wy2 = wy_if * y_if;
                                numer += c0_i * y_if + wy_if * (1.0 - r_hat_val) + xu_f * (wy2 + (c0_i - 1.0) * y_if * y_if);
                                denom += wy2 + (c0_i - 1.0) * y_if * y_if;
                                local += 1;
                            }
                    }

                    let new_u_f = numer / denom;
                    let diff = new_u_f - xu_f;

                    if diff.abs() > 1e-9 {
                        // Update r_hat and s_u using contiguous SIMD loops
                        unsafe {
                            let r_hat_mut = r_hat.as_mut_ptr();
                            let yi_ptr = yi_cols.as_ptr().add(yi_f_offset);
                            
                            for local in 0..nnz_u {
                                *r_hat_mut.add(local) += diff * *yi_ptr.add(local);
                            }
                        }
                        
                        // Use AXPY for updating the s_u vector efficiently
                        let gram_row = &gram[f * k..(f + 1) * k];
                        axpy_f32(diff, gram_row, s_u);
                        
                        xu[f] = new_u_f;
                    }
                }
            }
        });
    });
}

thread_local! {
    // Reused MemBuffer for the low-level (allocation-free) Cholesky factor +
    // solve calls below. Grown on demand, then kept for the life of the
    // thread instead of allocating a fresh Mat + MemBuffer per call (as the
    // old `.llt()` / `.solve_in_place()` high-level API did).
    static CHOL_SCRATCH_MEM: RefCell<faer::dyn_stack::MemBuffer> =
        RefCell::new(faer::dyn_stack::MemBuffer::new(faer::dyn_stack::StackReq::EMPTY));
}

fn cholesky_solve_inplace(a: &mut [f32], b: &mut [f32], k: usize) {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::cholesky::llt::{
        factor::{cholesky_in_place, cholesky_in_place_scratch},
        solve::{solve_in_place_scratch, solve_in_place_with_conj},
    };
    use faer::{Conj, Par};

    let par = Par::Seq;
    // Only the lower triangle of `a` is read/written by either call below
    // (both the recursive factorization and the triangular solves operate
    // exclusively on the lower triangle / its transpose), so `a`'s upper
    // triangle never needs to be filled or synced.
    let req = cholesky_in_place_scratch::<f32>(k, par, Default::default())
        .or(solve_in_place_scratch::<f32>(k, 1, par));

    CHOL_SCRATCH_MEM.with(|cell| {
        let mut mem = cell.borrow_mut();
        if mem.len() < req.size_bytes() {
            *mem = MemBuffer::new(req);
        }
        let stack = MemStack::new(&mut mem);

        let mut a_mat = faer::MatMut::from_row_major_slice_mut(a, k, k);
        let mut b_mat = faer::MatMut::from_column_major_slice_mut(b, k, 1);

        if cholesky_in_place(a_mat.as_mut(), Default::default(), par, stack, Default::default()).is_ok() {
            solve_in_place_with_conj(a_mat.as_ref(), Conj::No, b_mat.as_mut(), par, stack);
        }
    });
}

thread_local! {
    static SCRATCH_CHOL: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

fn solve_one_side_cholesky(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    other: &[f32],
    gram: &[f32],
    k: usize,
    lambda: f32,
    alpha: f32,
    out: &mut [f32],
) {
    let eff_lambda = lambda.max(1e-6);

    out.par_chunks_mut(k).enumerate().for_each(|(u, xu)| {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let nnz_u = end - start;

        if nnz_u == 0 {
            // `out` is the live factor matrix (seeded with random_factors), not
            // a fresh zeroed buffer as before the &mut out refactor — so a cold
            // entity must be explicitly zeroed here, exactly as CG and eALS do.
            xu.fill(0.0);
            return;
        }

        let (mut a_buf, mut b_buf, mut yi_buf, mut w_buf) = SCRATCH_CHOL.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut a, ref mut b, ref mut yi, ref mut w) = *borrow;
            // take the vecs out of the RefCell temporarily so we don't hold the borrow
            (
                std::mem::take(a),
                std::mem::take(b),
                std::mem::take(yi),
                std::mem::take(w),
            )
        });

        // --- Build A = Gram + lambda*I + sum_i w_i * y_i * y_i^T ---
        a_buf.clear();
        a_buf.extend_from_slice(gram);
        b_buf.clear();
        b_buf.resize(k, 0.0);

        for j in 0..k {
            a_buf[j * k + j] += eff_lambda;
        }

        // Collect item vectors and weights for batch rank-1 update
        yi_buf.clear();
        yi_buf.resize(nnz_u * k, 0.0);
        w_buf.clear();
        w_buf.resize(nnz_u, 0.0);

        for (local, idx) in (start..end).enumerate() {
            let i = indices[idx] as usize;
            let ci = 1.0 + alpha * data[idx];
            let yi = &other[i * k..(i + 1) * k];

            // b += ci * yi
            axpy_f32(ci, yi, &mut b_buf);

            // Store sqrt(w) * yi for batch syrk
            let w = ci - 1.0; // = alpha * data[idx]
            if w > 0.0 {
                let sw = w.sqrt();
                let dest = &mut yi_buf[local * k..(local + 1) * k];
                for f in 0..k {
                    dest[f] = sw * yi[f];
                }
                w_buf[local] = 1.0; // marker: has weight
            } else {
                w_buf[local] = 0.0;
            }
        }

        if b_buf.iter().all(|&v| v == 0.0) {
            // Same as the nnz_u == 0 case: zero rather than leaving the
            // random initialisation in place.
            xu.fill(0.0);
            // Put vecs back
            SCRATCH_CHOL.with(|cell| {
                *cell.borrow_mut() = (a_buf, b_buf, yi_buf, w_buf);
            });
            return;
        }

        // Batch rank-1 updates via syrk:  A += W^T * W  where W rows = sqrt(w_i)*y_i
        // This is a single BLAS call instead of nnz_u individual rank-1 updates.
        // Build W matrix with only non-zero weight rows
        let w_mat = MatRef::from_row_major_slice(&yi_buf[..nnz_u * k], nnz_u, k);
        let w_mat_t = w_mat.transpose();
        let mut a_mat = faer::MatMut::from_row_major_slice_mut(&mut a_buf, k, k);
        // Only the lower triangle of A is ever read by cholesky_solve_inplace,
        // so compute only that half of W^T * W instead of the full k x k
        // product (halves the flops of this step).
        {
            use faer::linalg::matmul::triangular::{matmul as triangular_matmul, BlockStructure};
            triangular_matmul(
                a_mat.as_mut(),
                BlockStructure::TriangularLower,
                Accum::Add,
                w_mat_t,
                BlockStructure::Rectangular,
                w_mat,
                BlockStructure::Rectangular,
                1.0f32,
                Par::Seq,
            );
        }

        cholesky_solve_inplace(&mut a_buf, &mut b_buf, k);
        xu.copy_from_slice(&b_buf);

        // Put vecs back into TLS
        SCRATCH_CHOL.with(|cell| {
            *cell.borrow_mut() = (a_buf, b_buf, yi_buf, w_buf);
        });
    });
}


pub(crate) fn csr_transpose(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let nnz = indices.len();
    let mut cc = vec![0i64; n_cols];
    for &c in indices {
        cc[c as usize] += 1;
    }
    let mut ti = vec![0i64; n_cols + 1];
    for i in 0..n_cols {
        ti[i + 1] = ti[i] + cc[i];
    }
    let mut tv = vec![0i32; nnz];
    let mut td = vec![0.0f32; nnz];
    let mut pos = ti[..n_cols].to_vec();
    for row in 0..n_rows {
        let s = indptr[row] as usize;
        let e = indptr[row + 1] as usize;
        for idx in s..e {
            let col = indices[idx] as usize;
            let p = pos[col] as usize;
            tv[p] = row as i32;
            td[p] = data[idx];
            pos[col] += 1;
        }
    }
    (ti, tv, td)
}



struct AndersonAccel {
    m: usize,
    x_hist: std::collections::VecDeque<Vec<f32>>,
    f_hist: std::collections::VecDeque<Vec<f32>>,
}

impl AndersonAccel {
    fn new(m: usize) -> Self {
        Self {
            m,
            x_hist: std::collections::VecDeque::new(),
            f_hist: std::collections::VecDeque::new(),
        }
    }

    /// Push x_old (before the ALS step) and f = x_new - x_old.
    /// Returns the Anderson-mixed iterate (replacing x_new in-place).
    fn push_and_mix(&mut self, x_old: Vec<f32>, f: Vec<f32>) -> Vec<f32> {
        let dim = x_old.len();

        if self.x_hist.len() == self.m {
            self.x_hist.pop_front();
            self.f_hist.pop_front();
        }
        self.x_hist.push_back(x_old);
        self.f_hist.push_back(f.clone());

        let h = self.f_hist.len();

        if h == 1 {
            let mut out = self.x_hist[0].clone();
            for (o, &fi) in out.iter_mut().zip(&f) {
                *o += fi;
            }
            return out;
        }

        let mut g = vec![0.0f32; h * h];
        for i in 0..h {
            for j in i..h {
                let d: f32 = self.f_hist[i]
                    .iter()
                    .zip(&self.f_hist[j])
                    .map(|(a, b)| a * b)
                    .sum();
                g[i * h + j] = d;
                g[j * h + i] = d;
            }
        }

        let n = h + 1;
        let mut mat = vec![0.0f32; n * n];
        for i in 0..h {
            for j in 0..h {
                mat[i * n + j] = g[i * h + j];
            }
            mat[i * n + h] = 1.0;
            mat[h * n + i] = 1.0;
        }

        let mut rhs = vec![0.0f32; n];
        rhs[h] = 1.0;

        if !gauss_solve_inplace(&mut mat, &mut rhs, n) {
            let mut out = self.x_hist[h - 1].clone();
            for (o, &fi) in out.iter_mut().zip(&self.f_hist[h - 1]) {
                *o += fi;
            }
            return out;
        }

        let mut out = vec![0.0f32; dim];
        for i in 0..h {
            let theta = rhs[i];
            for d in 0..dim {
                out[d] += theta * (self.x_hist[i][d] + self.f_hist[i][d]);
            }
        }
        out
    }
}

fn gauss_solve_inplace(mat: &mut Vec<f32>, rhs: &mut Vec<f32>, n: usize) -> bool {
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return false;
        }
        if max_row != col {
            for j in 0..n {
                mat.swap(col * n + j, max_row * n + j);
            }
            rhs.swap(col, max_row);
        }
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for j in col..n {
                let v = factor * mat[col * n + j];
                mat[row * n + j] -= v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for col in (0..n).rev() {
        rhs[col] /= mat[col * n + col];
        for row in 0..col {
            let v = mat[row * n + col] * rhs[col];
            rhs[row] -= v;
        }
    }
    true
}

pub(crate) fn als_train(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    indptr_t: &[i64],
    indices_t: &[i32],
    data_t: &[f32],
    n_users: usize,
    n_items: usize,
    k: usize,
    lambda: f32,
    alpha: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
    cg_iters: usize,
    use_cholesky: bool,
    anderson_m: usize,
    use_eals: bool,
    eals_iters: usize,
    item_pop_weights: Option<&[f32]>,
    use_biases: bool,
) -> (Vec<f32>, Vec<f32>, f32, Vec<f32>, Vec<f32>) {
    let mut user_factors = random_factors(n_users, k, seed);
    let mut item_factors = random_factors(n_items, k, seed.wrapping_add(1));

    // Bias terms: μ + b_u + b_i + w_u · h_i
    let mut global_bias: f32 = 0.0;
    let mut user_biases = vec![0.0f32; n_users];
    let mut item_biases = vec![0.0f32; n_items];

    if use_biases {
        // Compute global bias = average observed value
        let total_vals: f64 = data.iter().map(|&v| v as f64).sum();
        let total_count = data.len() as f64;
        if total_count > 0.0 {
            global_bias = (total_vals / total_count) as f32;
        }
    }

    if verbose {
        let solver_name = if use_eals {
            format!("eALS(iters={})", eals_iters)
        } else if use_cholesky {
            "Cholesky".to_string()
        } else {
            format!("CG(iters={})", cg_iters)
        };
        println!("  Solver: {}  factors={}", solver_name, k);
        println!("  ITER | USER FACTORS | ITEM FACTORS | TOTAL TIME ");
        println!("  ------------------------------------------------");
    }

    let mut total_time = std::time::Duration::new(0, 0);
    let use_aa = anderson_m > 0;
    let mut accel = AndersonAccel::new(anderson_m);

    for iter in 0..iterations {
        let iter_start = std::time::Instant::now();

        let solve = |ip: &[i64], ix: &[i32], d: &[f32], other: &[f32], gram: &[f32], out: &mut [f32], ipw: Option<&[f32]>| {
            if use_eals {
                solve_one_side_eals(ip, ix, d, other, gram, out, k, lambda, alpha, eals_iters, ipw);
            } else if use_cholesky {
                solve_one_side_cholesky(ip, ix, d, other, gram, k, lambda, alpha, out);
            } else {
                solve_one_side_cg(ip, ix, d, other, gram, k, lambda, alpha, cg_iters, out);
            }
        };

        let x_old: Vec<f32> = if use_aa {
            let mut v = user_factors.clone();
            v.extend_from_slice(&item_factors);
            v
        } else {
            Vec::new()
        };

        let start_u = std::time::Instant::now();
        // When eALS + popularity weights: use weighted gramian for user-side solve
        let g_item = if use_eals && item_pop_weights.is_some() {
            weighted_gramian(&item_factors, item_pop_weights.unwrap(), n_items, k)
        } else {
            gramian(&item_factors, n_items, k)
        };
        solve(indptr, indices, data, &item_factors, &g_item, &mut user_factors, item_pop_weights);
        let u_time = start_u.elapsed();

        let start_i = std::time::Instant::now();
        let g_user = gramian(&user_factors, n_users, k);
        // Item-side solve: no popularity weighting (items are the "other" side)
        solve(indptr_t, indices_t, data_t, &user_factors, &g_user, &mut item_factors, None);
        let i_time = start_i.elapsed();

        if use_aa {
            let mut x_new: Vec<f32> = user_factors.clone();
            x_new.extend_from_slice(&item_factors);
            let f: Vec<f32> = x_new.iter().zip(&x_old).map(|(a, b)| a - b).collect();
            let x_mixed = accel.push_and_mix(x_old, f);
            let uf_len = n_users * k;
            user_factors.copy_from_slice(&x_mixed[..uf_len]);
            item_factors.copy_from_slice(&x_mixed[uf_len..]);
        }

        let iter_time = iter_start.elapsed();
        total_time += iter_time;

        if verbose {
            let aa_tag = if use_aa { " AA" } else { "   " };
            println!(
                "  {:>4}{} | {:>10.1}s | {:>10.1}s | {:>9.1}s ",
                iter + 1,
                aa_tag,
                u_time.as_secs_f64(),
                i_time.as_secs_f64(),
                iter_time.as_secs_f64()
            );
        }
    }

    if verbose {
        println!("  ------------------------------------------------");
        println!("  Done in {:.1}s", total_time.as_secs_f64());
    }

    if use_biases {
        // Final bias update after all iterations
        // b_u = Σ_i c_ui (r_ui - μ - b_i - x_u·y_i) / (Σ_i c_ui + λ)
        // ponytail: user pass is independent per-row (only reads item_biases, which
        // is not touched here) so it parallelizes trivially; the item pass below
        // must stay serialized after this one since it reads the finished user_biases.
        user_biases.par_iter_mut().enumerate().for_each(|(u, bu)| {
            let s = indptr[u] as usize;
            let e = indptr[u + 1] as usize;
            let xu = &user_factors[u * k..(u + 1) * k];
            let mut num = 0.0f32;
            let mut den = lambda;
            for idx in s..e {
                let i = indices[idx] as usize;
                let r = data[idx];
                let c = 1.0 + alpha * r;
                let pred = dot_f32(xu, &item_factors[i * k..(i + 1) * k]) + item_biases[i] + global_bias;
                num += c * (r - pred);
                den += c;
            }
            *bu = num / den;
        });
        // b_i = Σ_u c_ui (r_ui - μ - b_u - x_u·y_i) / (Σ_u c_ui + λ)
        item_biases.par_iter_mut().enumerate().for_each(|(i, bi)| {
            let s = indptr_t[i] as usize;
            let e = indptr_t[i + 1] as usize;
            let yi = &item_factors[i * k..(i + 1) * k];
            let mut num = 0.0f32;
            let mut den = lambda;
            for idx in s..e {
                let u = indices_t[idx] as usize;
                let r = data_t[idx];
                let c = 1.0 + alpha * r;
                let pred = dot_f32(&user_factors[u * k..(u + 1) * k], yi) + user_biases[u] + global_bias;
                num += c * (r - pred);
                den += c;
            }
            *bi = num / den;
        });
    }

    (user_factors, item_factors, global_bias, user_biases, item_biases)
}

// Applies bias terms + exclusion masking to a raw (unbiased) score row in
// place, then extracts the top-`n` (id, score) pairs. Shared by the
// single-user gemv path (`top_n_items`) and the blocked gemm path
// (`als_recommend_all`) so both stay bit-for-bit identical in tie ordering,
// bias handling and exclusion semantics.
fn top_n_from_scores(
    scores: &mut [f32],
    n: usize,
    exc: &[i32],
    exc_start: usize,
    exc_end: usize,
    bias_offset: f32,
    item_biases: Option<&[f32]>,
) -> (Vec<i32>, Vec<f32>) {
    if let Some(ib) = item_biases {
        for (i, sc) in scores.iter_mut().enumerate() {
            *sc += bias_offset + ib[i];
        }
    } else if bias_offset != 0.0 {
        for sc in scores.iter_mut() {
            *sc += bias_offset;
        }
    }

    // ponytail: write NEG_INFINITY straight from the exclusion slice instead of
    // building an AHashSet and doing n_items lookups — |exc| is typically tiny
    // next to n_items.
    for &item_id in &exc[exc_start..exc_end] {
        if let Some(sc) = scores.get_mut(item_id as usize) {
            *sc = f32::NEG_INFINITY;
        }
    }

    let mut scored: Vec<(f32, i32)> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, &sc)| if sc.is_finite() { Some((sc, i as i32)) } else { None })
        .collect();
    let take = n.min(scored.len());
    if take == 0 {
        return (vec![], vec![]);
    }
    scored.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(take);
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (
        scored.iter().map(|(_, i)| *i).collect(),
        scored.iter().map(|(s, _)| *s).collect(),
    )
}

pub(crate) fn top_n_items(
    uf: &[f32],
    itf: &[f32],
    uid: usize,
    n_items: usize,
    k: usize,
    n: usize,
    exc: &[i32],
    exc_start: usize,
    exc_end: usize,
    global_bias: f32,
    user_biases: Option<&[f32]>,
    item_biases: Option<&[f32]>,
) -> (Vec<i32>, Vec<f32>) {
    let u = &uf[uid * k..(uid + 1) * k];

    let mut scores = vec![0.0f32; n_items];
    faer::linalg::matmul::matmul(
        faer::MatMut::from_column_major_slice_mut(&mut scores, n_items, 1).as_mut(),
        faer::Accum::Replace,
        MatRef::from_row_major_slice(itf, n_items, k),
        MatRef::from_column_major_slice(u, k, 1),
        1.0f32,
        faer::Par::Seq,
    );

    let bu = user_biases.map_or(0.0, |b| b[uid]);
    let bias_offset = global_bias + bu;
    top_n_from_scores(&mut scores, n, exc, exc_start, exc_end, bias_offset, item_biases)
}

fn top_n_users(
    uf: &[f32],
    itf: &[f32],
    iid: usize,
    n_users: usize,
    k: usize,
    n: usize,
    global_bias: f32,
    user_biases: Option<&[f32]>,
    item_biases: Option<&[f32]>,
) -> (Vec<i32>, Vec<f32>) {
    let y = &itf[iid * k..(iid + 1) * k];

    let mut scores = vec![0.0f32; n_users];
    faer::linalg::matmul::matmul(
        faer::MatMut::from_column_major_slice_mut(&mut scores, n_users, 1).as_mut(),
        faer::Accum::Replace,
        MatRef::from_row_major_slice(uf, n_users, k),
        MatRef::from_column_major_slice(y, k, 1),
        1.0f32,
        faer::Par::Seq,
    );

    // Add bias terms if present
    let bi = item_biases.map_or(0.0, |b| b[iid]);
    let bias_offset = global_bias + bi;
    if let Some(ub) = user_biases {
        for (u, sc) in scores.iter_mut().enumerate() {
            *sc += bias_offset + ub[u];
        }
    } else if bias_offset != 0.0 {
        for sc in scores.iter_mut() {
            *sc += bias_offset;
        }
    }

    let mut scored: Vec<(f32, i32)> = scores
        .into_iter()
        .enumerate()
        .map(|(u, sc)| (sc, u as i32))
        .collect();
    let take = n.min(scored.len());
    if take == 0 {
        return (vec![], vec![]);
    }
    scored.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(take);
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (
        scored.iter().map(|(_, i)| *i).collect(),
        scored.iter().map(|(s, _)| *s).collect(),
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_users, n_items, factors, regularization, alpha, iterations, seed, verbose, cg_iters=10, use_cholesky=false, anderson_m=0, use_eals=false, eals_iters=1, item_pop_weights=None, use_biases=false))]
pub fn als_fit_implicit<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f32>,
    n_users: usize,
    n_items: usize,
    factors: usize,
    regularization: f32,
    alpha: f32,
    iterations: usize,
    seed: u64,
    verbose: bool,
    cg_iters: usize,
    use_cholesky: bool,
    anderson_m: usize,
    use_eals: bool,
    eals_iters: usize,
    item_pop_weights: Option<PyReadonlyArray1<f32>>,
    use_biases: bool,
) -> PyResult<(
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
    f32,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let id = data.as_slice()?;

    let ipw_vec: Option<Vec<f32>> = item_pop_weights.map(|w| w.as_slice().unwrap().to_vec());

    let (ti, tx, td) = py.detach(|| csr_transpose(ip, ix, id, n_users, n_items));

    let (uf, itf, gb, ub, ib) = py.detach(|| {
        als_train(
            ip,
            ix,
            id,
            &ti,
            &tx,
            &td,
            n_users,
            n_items,
            factors,
            regularization,
            alpha,
            iterations,
            seed,
            verbose,
            cg_iters,
            use_cholesky,
            anderson_m,
            use_eals,
            eals_iters,
            ipw_vec.as_deref(),
            use_biases,
        )
    });

    let ua = PyArray1::from_vec(py, uf);
    let ia = PyArray1::from_vec(py, itf);

    Ok((
        ua.reshape([n_users, factors])?.into(),
        ia.reshape([n_items, factors])?.into(),
        gb,
        ub.into_pyarray(py),
        ib.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, user_id, n, exclude_indptr, exclude_indices, global_bias=0.0, user_biases=None, item_biases=None))]
pub fn als_recommend_items<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray2<f32>,
    item_factors: PyReadonlyArray2<f32>,
    user_id: usize,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
    global_bias: f32,
    user_biases: Option<PyReadonlyArray1<f32>>,
    item_biases: Option<PyReadonlyArray1<f32>>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let k = user_factors.shape()[1];
    let n_items = item_factors.shape()[0];
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;
    let es = ep[user_id] as usize;
    let ee = ep[user_id + 1] as usize;
    let ub: Option<&[f32]> = user_biases.as_ref().and_then(|b| b.as_slice().ok());
    let ib: Option<&[f32]> = item_biases.as_ref().and_then(|b| b.as_slice().ok());
    let (ids, scores) = top_n_items(uf, itf, user_id, n_items, k, n, ex, es, ee, global_bias, ub, ib);
    Ok((ids.into_pyarray(py), scores.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, item_id, n, global_bias=0.0, user_biases=None, item_biases=None))]
pub fn als_recommend_users<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray2<f32>,
    item_factors: PyReadonlyArray2<f32>,
    item_id: usize,
    n: usize,
    global_bias: f32,
    user_biases: Option<PyReadonlyArray1<f32>>,
    item_biases: Option<PyReadonlyArray1<f32>>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let k = user_factors.shape()[1];
    let n_users = user_factors.shape()[0];
    let ub: Option<&[f32]> = user_biases.as_ref().and_then(|b| b.as_slice().ok());
    let ib: Option<&[f32]> = item_biases.as_ref().and_then(|b| b.as_slice().ok());
    let (ids, scores) = top_n_users(uf, itf, item_id, n_users, k, n, global_bias, ub, ib);
    Ok((ids.into_pyarray(py), scores.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (item_factors, indices, data, regularization, alpha, cg_iters=10, use_cholesky=false, use_eals=false, eals_iters=1, item_pop_weights=None))]
pub fn als_recalculate_user<'py>(
    py: Python<'py>,
    item_factors: PyReadonlyArray2<f32>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f32>,
    regularization: f32,
    alpha: f32,
    cg_iters: usize,
    use_cholesky: bool,
    use_eals: bool,
    eals_iters: usize,
    item_pop_weights: Option<PyReadonlyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let itf = item_factors.as_slice()?;
    let ix = indices.as_slice()?;
    let id = data.as_slice()?;

    let k = item_factors.shape()[1];
    let n_items = item_factors.shape()[0];
    let mut out = vec![0.0f32; k];
    let ip = vec![0, ix.len() as i64];
    
    let ipw: Option<&[f32]> = item_pop_weights.as_ref().and_then(|w| w.as_slice().ok());
    
    let g_item = if use_eals && ipw.is_some() {
        weighted_gramian(itf, ipw.unwrap(), n_items, k)
    } else {
        gramian(itf, n_items, k)
    };
    
    if use_eals {
        solve_one_side_eals(&ip, ix, id, itf, &g_item, &mut out, k, regularization, alpha, eals_iters, ipw);
    } else if use_cholesky {
        solve_one_side_cholesky(&ip, ix, id, itf, &g_item, k, regularization, alpha, &mut out);
    } else {
        solve_one_side_cg(&ip, ix, id, itf, &g_item, k, regularization, alpha, cg_iters, &mut out);
    }

    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (user_factors, item_factors, n, exclude_indptr, exclude_indices, global_bias=0.0, user_biases=None, item_biases=None))]
pub fn als_recommend_all<'py>(
    py: Python<'py>,
    user_factors: PyReadonlyArray2<f32>,
    item_factors: PyReadonlyArray2<f32>,
    n: usize,
    exclude_indptr: PyReadonlyArray1<i64>,
    exclude_indices: PyReadonlyArray1<i32>,
    global_bias: f32,
    user_biases: Option<PyReadonlyArray1<f32>>,
    item_biases: Option<PyReadonlyArray1<f32>>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let uf = user_factors.as_slice()?;
    let itf = item_factors.as_slice()?;
    let k = user_factors.shape()[1];
    let n_users = user_factors.shape()[0];
    let n_items = item_factors.shape()[0];
    let ep = exclude_indptr.as_slice()?;
    let ex = exclude_indices.as_slice()?;
    let ub: Option<&[f32]> = user_biases.as_ref().and_then(|b| b.as_slice().ok());
    let ib: Option<&[f32]> = item_biases.as_ref().and_then(|b| b.as_slice().ok());

    // ponytail: fixed block size, not tuned per-machine — just big enough that
    // the gemm below is compute-bound instead of memory-bound (see
    // top_n_from_scores / the module doc for why per-user gemv was slow).
    const RECOMMEND_BLOCK: usize = 256;

    let item_mat = MatRef::from_row_major_slice(itf, n_items, k);

    // Flatten results
    let mut all_user_ids = Vec::with_capacity(n_users * n);
    let mut all_item_ids = Vec::with_capacity(n_users * n);
    let mut all_scores = Vec::with_capacity(n_users * n);

    let mut block_start = 0usize;
    while block_start < n_users {
        let bsize = RECOMMEND_BLOCK.min(n_users - block_start);
        let users_block = &uf[block_start * k..(block_start + bsize) * k];

        // One blocked gemm instead of `bsize` memory-bound gemvs: the item
        // matrix (n_items x k) is streamed once per block rather than once
        // per user, so it stays cache-resident across the block.
        let mut scores_block = vec![0.0f32; bsize * n_items];
        matmul(
            MatMut::from_row_major_slice_mut(&mut scores_block, bsize, n_items).as_mut(),
            Accum::Replace,
            MatRef::from_row_major_slice(users_block, bsize, k),
            item_mat.transpose(),
            1.0f32,
            Par::rayon(0),
        );

        let results: Vec<(Vec<i32>, Vec<f32>)> = scores_block
            .par_chunks_mut(n_items)
            .enumerate()
            .map(|(local, row)| {
                let user_id = block_start + local;
                let es = ep[user_id] as usize;
                let ee = ep[user_id + 1] as usize;
                let bu = ub.map_or(0.0, |b| b[user_id]);
                let bias_offset = global_bias + bu;
                top_n_from_scores(row, n, ex, es, ee, bias_offset, ib)
            })
            .collect();

        for (local, (ids, sc)) in results.into_iter().enumerate() {
            let user_id = block_start + local;
            all_user_ids.extend(std::iter::repeat(user_id as i32).take(ids.len()));
            all_item_ids.extend(ids);
            all_scores.extend(sc);
        }

        block_start += bsize;
    }

    Ok((
        all_user_ids.into_pyarray(py),
        all_item_ids.into_pyarray(py),
        all_scores.into_pyarray(py),
    ))
}
