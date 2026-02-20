//! FP-TDA: Frequent-Pattern Two-Dimensional Array
//!
//! An alternative frequent-itemset miner based on the 2-D array approach
//! described in IJISRT25NOV1256.
//!
//! Key idea: represent filtered, frequency-ordered transactions as a 2-D
//! matrix (TDA).  Mine frequent itemsets by right-to-left column projection
//! (without constructing conditional FP-trees).  Each column `c` is projected
//! to the subset of rows that contain item `c`; those rows are then
//! recursively projected to columns 0..c.  This yields every frequent itemset
//! with the same support values as FP-Growth.
//!
//! Output: flat `(supports, offsets, items)` triple — identical layout to the
//! FP-Growth output so Python callers are fully interchangeable.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::fpgrowth::{flatten_results, process_item_counts};

// ---------------------------------------------------------------------------
// Mining
// ---------------------------------------------------------------------------

/// A projected database: each entry is a sorted list of local column ids.
type Projection = Vec<Vec<u32>>;

/// Recursively mine frequent itemsets from a projected transaction database.
///
/// `rows`      – transactions as sorted lists of local col ids (0 = most freq)
/// `max_col`   – only columns 0..max_col are in scope (exclusive upper bound)
/// `suffix`    – original item ids accumulated so far (right side of itemset)
/// `min_count` – absolute support threshold
/// `max_len`   – maximum itemset length (None = unlimited)
/// `original`  – local_id → original item id mapping
/// `results`   – output accumulator
fn mine_projection(
    rows: &Projection,
    max_col: usize,
    suffix: &[u32],
    min_count: u64,
    max_len: Option<usize>,
    original: &[u32],
    results: &mut Vec<(u64, Vec<u32>)>,
) {
    if max_col == 0 { return; }

    // Count occurrences of each column in this projection.
    let mut col_counts = vec![0u64; max_col];
    for row in rows {
        for &c in row {
            // rows are already restricted to < max_col from the caller.
            col_counts[c as usize] += 1;
        }
    }

    // Scan columns right-to-left: each frequent column produces an itemset
    // and a recursive sub-projection.
    for c in (0..max_col).rev() {
        let count = col_counts[c];
        if count < min_count { continue; }

        let item_orig = original[c];
        let new_len = suffix.len() + 1;

        // Emit {item_c} ∪ suffix.
        if max_len.map_or(true, |ml| new_len <= ml) {
            let mut iset = vec![item_orig];
            iset.extend_from_slice(suffix);
            results.push((count, iset));
        }

        // Recurse only if we can still grow the itemset.
        if max_len.map_or(true, |ml| new_len < ml) && c > 0 {
            // Project: keep only rows that contain col c, restricted to cols < c.
            let projected: Projection = rows
                .iter()
                .filter(|row| row.binary_search(&(c as u32)).is_ok())
                .map(|row| row.iter().filter(|&&x| x < c as u32).copied().collect())
                .collect();

            if !projected.is_empty() {
                let mut new_suffix = vec![item_orig];
                new_suffix.extend_from_slice(suffix);
                mine_projection(
                    &projected, c, &new_suffix, min_count, max_len, original, results,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal entry points (dense / CSR)
// ---------------------------------------------------------------------------

fn _mine_fptda_dense(
    flat: &[u8],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Vec<u64>, Vec<u32>, Vec<u32>)> {
    // 1. Count item frequencies.
    let item_count: Vec<u64> = flat
        .par_chunks(n_cols)
        .fold(
            || vec![0u64; n_cols],
            |mut acc, row| {
                for (col, &val) in row.iter().enumerate() {
                    if val != 0 {
                        unsafe { *acc.get_unchecked_mut(col) += 1; }
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0u64; n_cols],
            |mut a, b| { for (x, y) in a.iter_mut().zip(b.iter()) { *x += y; } a },
        );

    let (global_to_local, original_items, frequent_cols, frequent_len) =
        match process_item_counts(item_count, min_count, n_cols) {
            Some(v) => v,
            None => return Ok((vec![], vec![], vec![])),
        };

    // 2. Build per-transaction sorted lists of local col ids.
    let itemsets: Projection = flat
        .par_chunks(n_cols)
        .filter_map(|row| {
            let mut items: Vec<u32> = frequent_cols
                .iter()
                .filter(|&&col| unsafe { *row.get_unchecked(col) != 0 })
                .map(|&col| global_to_local[col])
                .collect();
            if items.is_empty() { return None; }
            items.sort_unstable();
            Some(items)
        })
        .collect();

    // 3. Mine.
    let mut results = Vec::new();
    mine_projection(&itemsets, frequent_len, &[], min_count, max_len, &original_items, &mut results);

    Ok(flatten_results(results))
}

fn _mine_fptda_csr(
    indptr: &[i32],
    indices: &[i32],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Vec<u64>, Vec<u32>, Vec<u32>)> {
    let n_rows = indptr.len() - 1;

    // 1. Count item frequencies.
    let item_count: Vec<u64> = (0..n_rows)
        .into_par_iter()
        .fold(
            || vec![0u64; n_cols],
            |mut acc, row| {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                for &col in &indices[start..end] {
                    let c = col as usize;
                    if c < n_cols { unsafe { *acc.get_unchecked_mut(c) += 1; } }
                }
                acc
            },
        )
        .reduce(
            || vec![0u64; n_cols],
            |mut a, b| { for (x, y) in a.iter_mut().zip(b.iter()) { *x += y; } a },
        );

    let (global_to_local, original_items, _, frequent_len) =
        match process_item_counts(item_count, min_count, n_cols) {
            Some(v) => v,
            None => return Ok((vec![], vec![], vec![])),
        };

    // 2. Build per-transaction sorted lists of local col ids.
    let itemsets: Projection = (0..n_rows)
        .into_par_iter()
        .filter_map(|row| {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let mut items: Vec<u32> = indices[start..end]
                .iter()
                .filter_map(|&col| {
                    if (col as usize) < n_cols {
                        let l = global_to_local[col as usize];
                        if l != u32::MAX { Some(l) } else { None }
                    } else { None }
                })
                .collect();
            if items.is_empty() { return None; }
            items.sort_unstable();
            Some(items)
        })
        .collect();

    // 3. Mine.
    let mut results = Vec::new();
    mine_projection(&itemsets, frequent_len, &[], min_count, max_len, &original_items, &mut results);

    Ok(flatten_results(results))
}

// ---------------------------------------------------------------------------
// PyO3 entry points
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, min_count, max_len=None))]
pub fn fptda_from_dense<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<u8>,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let arr = data.as_array();
    let n_rows = arr.nrows();
    let n_cols = arr.ncols();

    if n_cols == 0 || n_rows == 0 {
        return Ok((
            Vec::<u64>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
        ));
    }

    let flat: &[u8] = arr.as_slice().unwrap();
    let (supports, offsets, items) = _mine_fptda_dense(flat, n_cols, min_count, max_len)?;

    Ok((
        supports.into_pyarray(py),
        offsets.into_pyarray(py),
        items.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_cols, min_count, max_len=None))]
pub fn fptda_from_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i32>,
    indices: PyReadonlyArray1<i32>,
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;

    if ip.len() < 2 || n_cols == 0 {
        return Ok((
            Vec::<u64>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
        ));
    }

    let (supports, offsets, items) = _mine_fptda_csr(ip, ix, n_cols, min_count, max_len)?;

    Ok((
        supports.into_pyarray(py),
        offsets.into_pyarray(py),
        items.into_pyarray(py),
    ))
}
