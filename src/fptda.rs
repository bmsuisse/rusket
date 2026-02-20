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
// Mining  (iterative zero-copy slices)
// ---------------------------------------------------------------------------

/// A projected database: each entry is a slice of a transaction.
/// Since items are sorted by frequency (0 = most freq), the subset
/// of items `< c` is always a contiguous prefix of the slice.
type Projection<'a> = Vec<&'a [u32]>;

/// Mine all frequent itemsets from `root_rows` using an explicit worklist.
///
/// Each worklist item is `(rows, max_col, suffix)`:
///   - `rows`    – transactions represented as slices `&[u32]` restricted to `0..max_col`
///   - `max_col` – upper bound on column ids present in `rows`
///   - `suffix`  – original item ids already chosen (right side of the itemset)
///
/// By using `&[u32]`, we avoid all allocation overhead for row data during
/// projection. We only allocate a vector of pointers.
fn mine_tda_iterative<'a>(
    root_rows: Projection<'a>,
    original: &[u32],
    min_count: u64,
    max_len: Option<usize>,
    num_cols: usize,
) -> Vec<(u64, Vec<u32>)> {
    let mut results: Vec<(u64, Vec<u32>)> = Vec::new();

    // Worklist: (rows, max_col, suffix_items)
    let mut stack: Vec<(Projection<'a>, usize, Vec<u32>)> = vec![(root_rows, num_cols, vec![])];

    while let Some((rows, max_col, suffix)) = stack.pop() {
        if max_col == 0 || rows.is_empty() { continue; }

        // Count how many rows contain each column 0..max_col.
        let mut col_counts = vec![0u64; max_col];
        for row in &rows {
            for &c in *row {
                col_counts[c as usize] += 1;
            }
        }

        // Scan right-to-left; push sub-projections for frequent columns.
        for c in (0..max_col).rev() {
            let count = col_counts[c];
            if count < min_count { continue; }

            let item_orig = original[c];
            let new_len = suffix.len() + 1;

            // Emit {item_c} ∪ suffix.
            if max_len.map_or(true, |ml| new_len <= ml) {
                let mut iset = vec![item_orig];
                iset.extend_from_slice(&suffix);
                results.push((count, iset));
            }

            // Push sub-projection if we can still grow the itemset.
            if c > 0 && max_len.map_or(true, |ml| new_len < ml) {
                let mut projected = Vec::with_capacity(count as usize);

                for row in &rows {
                    // Since the row is sorted, `binary_search` is fast.
                    // If we find `c`, all elements before the found index are `< c`.
                    if let Ok(idx) = row.binary_search(&(c as u32)) {
                        let prefix = &row[0..idx];
                        if !prefix.is_empty() {
                            projected.push(prefix);
                        }
                    }
                }

                if !projected.is_empty() {
                    let mut new_suffix = vec![item_orig];
                    new_suffix.extend_from_slice(&suffix);
                    stack.push((projected, c, new_suffix));
                }
            }
        }
    }

    results
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
    let owned_itemsets: Vec<Vec<u32>> = flat
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
    let root_rows: Projection = owned_itemsets.iter().map(|v| v.as_slice()).collect();
    let results = mine_tda_iterative(root_rows, &original_items, min_count, max_len, frequent_len);

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
    let owned_itemsets: Vec<Vec<u32>> = (0..n_rows)
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
    let root_rows: Projection = owned_itemsets.iter().map(|v| v.as_slice()).collect();
    let results = mine_tda_iterative(root_rows, &original_items, min_count, max_len, frequent_len);

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
