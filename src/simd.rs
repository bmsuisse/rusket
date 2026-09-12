//! Shared SIMD-friendly dot product — 8-wide manual unroll, with runtime
//! AVX2+FMA dispatch on x86/x86_64.
//!
//! Used across BPR, SVD, FPMC, LightGCN, and other modules.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::sync::OnceLock;

/// Returns true if the running CPU supports AVX2 + FMA. The result is
/// detected once (via `is_x86_feature_detected!`, which itself queries
/// CPUID) and cached, so this is cheap to call from a hot loop.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn has_avx2_fma() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

/// SIMD-optimised dot product using 8-wide unrolling for NEON / AVX2.
///
/// On x86/x86_64, dispatches at runtime to an AVX2+FMA kernel when the CPU
/// supports it, falling back to the portable scalar unroll otherwise (e.g.
/// baseline SSE2-only x86-64 wheels). On all other architectures (notably
/// aarch64/Apple Silicon, where NEON is part of the baseline ISA) this
/// compiles straight down to the portable path with no dispatch overhead.
#[inline(always)]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if has_avx2_fma() {
            // SAFETY: `has_avx2_fma()` returned true, so the running CPU
            // supports both AVX2 and FMA, which is the sole precondition of
            // `dot_avx2`.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_portable(a, b)
}

/// Portable scalar implementation (relies on autovectorization). Used as
/// the fallback on x86/x86_64 when AVX2+FMA isn't available, and as the
/// only implementation on every other architecture.
#[inline(always)]
fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
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

/// AVX2+FMA dot product kernel.
///
/// # Safety
///
/// Callers must ensure:
/// - The running CPU supports both the `avx2` and `fma` target features
///   (checked once by `has_avx2_fma` and cached — this function must never
///   be called without that check having returned `true`).
/// - `b` is at least as long as `a`. Like `dot_portable`, this function
///   iterates using `a.len()` as the trip count and indexes into `b` at the
///   same offsets via unchecked/unaligned loads, so a shorter `b` is
///   undefined behavior. This matches the pre-existing contract of `dot`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k = a.len();
    let chunks = k / 8;
    let mut idx = 0usize;
    let mut acc = _mm256_setzero_ps();
    for _ in 0..chunks {
        // SAFETY: `idx + 8 <= k == a.len()` and `b.len() >= a.len()` per the
        // function's safety contract, so both loads read in-bounds memory.
        // `_mm256_loadu_ps` does not require alignment.
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        acc = _mm256_fmadd_ps(va, vb, acc);
        idx += 8;
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum: f32 = lanes.iter().sum();

    // Scalar tail for the remainder (k % 8 elements).
    while idx < k {
        sum += *a.get_unchecked(idx) * *b.get_unchecked(idx);
        idx += 1;
    }
    sum
}

/// `y += alpha * x`, with the same runtime AVX2+FMA dispatch as [`dot`].
///
/// `y` must be at least as long as `x`.
#[inline(always)]
pub(crate) fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if has_avx2_fma() {
            // SAFETY: `has_avx2_fma()` returned true, so the CPU supports
            // AVX2+FMA — the sole precondition of `axpy_avx2` beyond the
            // length contract documented above.
            unsafe { axpy_avx2(alpha, x, y) };
            return;
        }
    }
    axpy_portable(alpha, x, y);
}

/// Portable `y += alpha * x` (relies on autovectorization).
#[inline(always)]
fn axpy_portable(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let chunks = n / 8;
    let mut idx = 0;
    for _ in 0..chunks {
        unsafe {
            *y.get_unchecked_mut(idx) += alpha * *x.get_unchecked(idx);
            *y.get_unchecked_mut(idx + 1) += alpha * *x.get_unchecked(idx + 1);
            *y.get_unchecked_mut(idx + 2) += alpha * *x.get_unchecked(idx + 2);
            *y.get_unchecked_mut(idx + 3) += alpha * *x.get_unchecked(idx + 3);
            *y.get_unchecked_mut(idx + 4) += alpha * *x.get_unchecked(idx + 4);
            *y.get_unchecked_mut(idx + 5) += alpha * *x.get_unchecked(idx + 5);
            *y.get_unchecked_mut(idx + 6) += alpha * *x.get_unchecked(idx + 6);
            *y.get_unchecked_mut(idx + 7) += alpha * *x.get_unchecked(idx + 7);
        }
        idx += 8;
    }
    while idx < n {
        unsafe { *y.get_unchecked_mut(idx) += alpha * *x.get_unchecked(idx); }
        idx += 1;
    }
}

/// AVX2+FMA `y += alpha * x`.
///
/// # Safety
///
/// Same contract as [`dot_avx2`]: the CPU must support `avx2` and `fma`, and
/// `y` must be at least as long as `x`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    let va = _mm256_set1_ps(alpha);
    let mut idx = 0usize;
    for _ in 0..chunks {
        // SAFETY: idx + 8 <= n == x.len() <= y.len() per the contract.
        let vx = _mm256_loadu_ps(x.as_ptr().add(idx));
        let vy = _mm256_loadu_ps(y.as_ptr().add(idx));
        _mm256_storeu_ps(y.as_mut_ptr().add(idx), _mm256_fmadd_ps(va, vx, vy));
        idx += 8;
    }
    while idx < n {
        *y.get_unchecked_mut(idx) += alpha * *x.get_unchecked(idx);
        idx += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32) * 0.5 + seed - (i as f32 % 3.0) * 0.25)
            .collect()
    }

    const LENGTHS: [usize; 9] = [0, 1, 7, 8, 9, 31, 32, 33, 1000];

    /// The portable path must give the same answer regardless of architecture
    /// — this test runs everywhere (x86 and aarch64 alike).
    #[test]
    fn portable_matches_naive_reference() {
        for &len in &LENGTHS {
            let a = make(len, 1.0);
            let b = make(len, -2.0);
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let got = dot_portable(&a, &b);
            assert!(
                (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "len={len}: portable={got} expected={expected}"
            );
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_matches_portable_when_available() {
        if !has_avx2_fma() {
            eprintln!("skipping avx2_matches_portable_when_available: CPU lacks AVX2+FMA");
            return;
        }
        for &len in &LENGTHS {
            let a = make(len, 3.25);
            let b = make(len, 0.75);
            let portable = dot_portable(&a, &b);
            // SAFETY: guarded by `has_avx2_fma()` above; `b.len() == a.len()`.
            let avx2 = unsafe { dot_avx2(&a, &b) };
            assert!(
                (portable - avx2).abs() <= 1e-3 * portable.abs().max(1.0),
                "len={len}: portable={portable} avx2={avx2}"
            );
        }
    }

    /// `dot()` itself (the public dispatch entry point) must agree with the
    /// portable reference on every length, on any architecture. Note: when
    /// the AVX2+FMA path is selected, exact bit-for-bit equality isn't
    /// guaranteed (FMA fuses the multiply-add with no intermediate
    /// rounding, and the accumulation order differs from the 8-way scalar
    /// unroll), so this compares within a tight tolerance rather than with
    /// `assert_eq!`.
    #[test]
    fn dot_matches_portable() {
        for &len in &LENGTHS {
            let a = make(len, 5.0);
            let b = make(len, -1.5);
            let expected = dot_portable(&a, &b);
            let got = dot(&a, &b);
            assert!(
                (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "len={len}: dot={got} portable={expected}"
            );
        }
    }
}
