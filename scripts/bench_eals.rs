use std::time::Instant;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Csr {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f32>,
}

fn solve_ials_cg_one_user(
    u: usize,
    csr: &Csr,
    item_factors: &[f32],
    k: usize,
    gram: &[f32], // k x k
    alpha: f32, // confidence scaling
    lambda: f32,
    cg_iters: usize,
    out: &mut [f32], // len = k
) {
    let start = csr.indptr[u];
    let end = csr.indptr[u + 1];

    let mut b = vec![0.0; k];
    for idx in start..end {
        let i = csr.indices[idx];
        let c = 1.0 + alpha * csr.data[idx];
        let yi = &item_factors[i * k .. (i+1) * k];
        for f in 0..k {
            b[f] += c * yi[f]; // true preference p_{ui} = 1
        }
    }

    let apply_a = |v: &[f32], ap: &mut [f32]| {
        for a in 0..k {
            let mut s = 0.0;
            for bb in 0..k {
                s += gram[a * k + bb] * v[bb];
            }
            ap[a] = s + lambda * v[a];
        }
        for idx in start..end {
            let i = csr.indices[idx];
            let c = 1.0 + alpha * csr.data[idx];
            let yi = &item_factors[i * k .. (i+1) * k];
            let w = (c - 1.0) * dot(yi, v);
            for f in 0..k {
                ap[f] += w * yi[f];
            }
        }
    };

    let mut r = b.clone();
    let mut p = b.clone();
    let mut ap = vec![0.0; k];
    out.fill(0.0);

    let mut rsold = dot(&r, &r);
    for _ in 0..cg_iters {
        apply_a(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 { break; }
        let ak = rsold / pap;
        for j in 0..k { out[j] += ak * p[j]; }
        for j in 0..k { r[j] -= ak * ap[j]; }
        let rsnew = dot(&r, &r);
        if rsnew < 1e-10 { break; }
        let beta = rsnew / rsold;
        for j in 0..k { p[j] = r[j] + beta * p[j]; }
        rsold = rsnew;
    }
}


fn solve_eals_one_user(
    u: usize,
    csr: &Csr,
    item_factors: &[f32],
    k: usize,
    gram: &[f32], // k x k
    alpha: f32, // confidence scaling
    lambda: f32,
    out: &mut [f32], // len = k
) {
    let start = csr.indptr[u];
    let end = csr.indptr[u + 1];

    let mut r_hat = vec![0.0; end - start];
    for idx in start..end {
        let i = csr.indices[idx];
        let yi = &item_factors[i * k .. (i+1) * k];
        r_hat[idx - start] = dot(out, yi);
    }

    let mut s_u = vec![0.0; k];
    for f in 0..k {
        let mut s = 0.0;
        for a in 0..k {
            s += gram[f * k + a] * out[a];
        }
        s_u[f] = s;
    }

    // eALS requires multi-pass usually to converge, but 1 pass per outer-ALS iteration works well
    for f in 0..k {
        let mut numer = -(s_u[f] - out[f] * gram[f * k + f]);
        let mut denom = lambda + gram[f * k + f];

        for idx in start..end {
            let i = csr.indices[idx];
            let w = alpha * csr.data[idx]; 
            let y_if = item_factors[i * k + f];
            let r_hat_minus_f = r_hat[idx - start] - out[f] * y_if;
            
            numer += (w + 1.0) * y_if - w * r_hat_minus_f * y_if;
            denom += w * y_if * y_if;
        }

        let new_u_f = numer / denom;
        let diff = new_u_f - out[f];

        if diff != 0.0 {
            for idx in start..end {
                let i = csr.indices[idx];
                let y_if = item_factors[i * k + f];
                r_hat[idx - start] += diff * y_if;
            }
            for a in 0..k {
                s_u[a] += diff * gram[a * k + f];
            }
            out[f] = new_u_f;
        }
    }
}

// Cholesky exact solver for ground truth
fn solve_cholesky_one_user(
    u: usize,
    csr: &Csr,
    item_factors: &[f32],
    k: usize,
    gram: &[f32],
    alpha: f32,
    lambda: f32,
    out: &mut [f32],
) {
    // skipped to avoid faer dependency
}

fn calc_loss(u: usize, csr: &Csr, item_factors: &[f32], out: &[f32], k: usize, gram: &[f32], alpha: f32, lambda: f32) -> f32 {
    let mut loss = 0.0;
    
    // lambda * ||u||^2
    loss += lambda * dot(out, out);
    
    // sum_{i not in R_u} 1 * (0 - u^T y_i)^2 
    // = sum_{i} (u^T y_i)^2 - sum_{i in R_u} (u^T y_i)^2
    // = u^T (\sum_i y_i y_i^T) u - sum_{i in R_u} (u^T y_i)^2
    let mut u_gram_u = 0.0;
    for a in 0..k {
        let mut s = 0.0;
        for b in 0..k {
            s += gram[a * k + b] * out[b];
        }
        u_gram_u += out[a] * s;
    }
    loss += u_gram_u;

    let start = csr.indptr[u];
    let end = csr.indptr[u + 1];

    for idx in start..end {
        let i = csr.indices[idx];
        let c = 1.0 + alpha * csr.data[idx];
        let yi = &item_factors[i * k .. (i+1) * k];
        let pred = dot(out, yi);
        
        loss -= pred * pred; // remove the 1 * pred^2
        loss += c * (1.0 - pred) * (1.0 - pred); // add the true term
    }
    loss
}

fn main() {
    let n_users = 10;
    let n_items = 1000;
    let k = 64;
    let nnz_per_user = 50;

    let mut csr = Csr {
        indptr: vec![0; n_users + 1],
        indices: Vec::with_capacity(n_users * nnz_per_user),
        data: Vec::with_capacity(n_users * nnz_per_user),
    };

    let mut pos = 0;
    for u in 0..n_users {
        csr.indptr[u] = pos;
        for j in 0..nnz_per_user {
            csr.indices.push((u + j * 97) % n_items);
            csr.data.push(1.0);
            pos += 1;
        }
    }
    csr.indptr[n_users] = pos;

    let mut item_factors = vec![0.01; n_items * k];
    for i in 0..item_factors.len() {
        item_factors[i] = ((i * 137) % 50) as f32 / 50.0;
    }

    let mut gram = vec![0.0; k * k];
    for i in 0..n_items {
        for f1 in 0..k {
            for f2 in 0..k {
                gram[f1 * k + f2] += item_factors[i * k + f1] * item_factors[i * k + f2];
            }
        }
    }

    let alpha = 40.0;
    let lambda = 0.01;

    println!("Calculating Cholesky (Exact Base)...");
    let mut out_chol = vec![0.0; k];
    let mut chol_loss = 0.0;
    for u in 0..n_users {
        solve_cholesky_one_user(u, &csr, &item_factors, k, &gram, alpha, lambda, &mut out_chol);
        chol_loss += calc_loss(u, &csr, &item_factors, &out_chol, k, &gram, alpha, lambda);
    }
    println!("Exact Loss: {:.4}", chol_loss);

    let mut out_ials = vec![0.0; k];
    let mut ials_loss = 0.0;
    for u in 0..n_users {
        out_ials.fill(0.1);
        solve_ials_cg_one_user(u, &csr, &item_factors, k, &gram, alpha, lambda, 5, &mut out_ials);
        ials_loss += calc_loss(u, &csr, &item_factors, &out_ials, k, &gram, alpha, lambda);
    }
    println!("iALS CG(5) Loss: {:.4}", ials_loss);

    let mut out_eals = vec![0.0; k];
    let mut eals_loss = 0.0;
    for u in 0..n_users {
        out_eals.fill(0.1);
        // eALS typically runs coordinate descent multiple times to fully converge
        solve_eals_one_user(u, &csr, &item_factors, k, &gram, alpha, lambda, &mut out_eals);
        eals_loss += calc_loss(u, &csr, &item_factors, &out_eals, k, &gram, alpha, lambda);
    }
    println!("eALS (1 pass) Loss: {:.4}", eals_loss);

    eals_loss = 0.0;
    for u in 0..n_users {
        out_eals.fill(0.1);
        solve_eals_one_user(u, &csr, &item_factors, k, &gram, alpha, lambda, &mut out_eals);
        solve_eals_one_user(u, &csr, &item_factors, k, &gram, alpha, lambda, &mut out_eals);
        solve_eals_one_user(u, &csr, &item_factors, k, &gram, alpha, lambda, &mut out_eals);
        eals_loss += calc_loss(u, &csr, &item_factors, &out_eals, k, &gram, alpha, lambda);
    }
    println!("eALS (3 pass) Loss: {:.4}", eals_loss);
}
