//! Shared XorShift64 PRNG and random factor initialization.
//!
//! Used across ALS, BPR, SVD, LightGCN, FPMC, FM, BERT4Rec, SASRec, NMF,
//! NN-Descent, and ANN modules to avoid code duplication.

/// Fast XorShift64 PRNG — deterministic, no allocation, no external dep.
pub(crate) struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xbad5eed } else { seed },
        }
    }

    #[inline(always)]
    pub fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    #[inline(always)]
    pub fn next_float(&mut self) -> f32 {
        let v = self.next() & 0xFFFFFF;
        v as f32 / 0xFFFFFF as f32
    }

    /// Return a random index in `[0, max)`.
    #[inline(always)]
    pub fn next_usize(&mut self, max: usize) -> usize {
        (self.next() % max as u64) as usize
    }
}

/// Generate random latent factors of shape `(n, k)` in `[-scale, scale]`
/// where `scale = 1 / sqrt(k)`.
pub(crate) fn random_factors(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let scale = 1.0 / (k as f32).sqrt();
    let mut out = vec![0.0f32; n * k];
    for v in out.iter_mut() {
        *v = (rng.next_float() * 2.0 - 1.0) * scale;
    }
    out
}

/// Generate random factors of a given total size (for models like FPMC/FM that
/// use a flat size rather than `(n, k)`).
pub(crate) fn random_factors_flat(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let scale = 0.01;
    let mut out = vec![0.0f32; size];
    for v in out.iter_mut() {
        *v = (rng.next_float() * 2.0 - 1.0) * scale;
    }
    out
}

/// Generate normally-distributed random vector (Box-Muller transform).
/// Shared by SASRec and BERT4Rec for weight initialization.
pub(crate) fn randn_vec(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let u = 1e-7 + rng.next_float() * (1.0 - 1e-7);
        let v = rng.next_float() * std::f32::consts::TAU;
        let z = (-2.0 * u.ln()).sqrt() * v.cos();
        out.push(z * scale);
    }
    out
}
