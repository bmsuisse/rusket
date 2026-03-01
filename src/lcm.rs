use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

use crate::eclat::BitSet;
use crate::fpgrowth::{flatten_results, process_item_counts};

impl BitSet {
    #[inline]
    fn is_subset_of(&self, other: &BitSet) -> bool {
        for (a, b) in self.blocks.iter().zip(other.blocks.iter()) {
            if (a & !b) != 0 {
                return false;
            }
        }
        true
    }
}

pub(crate) fn lcm_mine(
    prefix: &[u32],
    prefix_bs: &BitSet,
    tail_idx: usize,
    active_items: &Arc<Vec<(u32, BitSet)>>,
    min_count: u64,
    max_len: Option<usize>,
    global_to_local: &[u32],
) -> Vec<(u64, Vec<u32>)> {
    let mut results = Vec::new();
    let n_blocks = prefix_bs.blocks.len();

    // Fast membership check for prefix.
    // Instead of passing a full boolean array, since prefix is small, we just check if 
    // `global_to_local[item] <= tail_idx` (mostly) but wait, `prefix` contains items that could be anywhere.
    // Actually, `active_items` is sorted by `local_id`. So `active_items[j].0` has local_id `j`.
    // Wait, `process_item_counts` returns `original_items` where `original_items[j]` is the true item.
    // So `active_items[j]` is `(original_item, bitset)`.
    // If we just track a `prefix_mask` of size `active_items.len()`, it's very fast.
    let mut in_prefix = vec![false; active_items.len()];
    for &num in prefix {
        let local = global_to_local[num as usize];
        if local != u32::MAX {
            in_prefix[local as usize] = true;
        }
    }

    // Parallelize at the top level or current level if the branching factor is large enough.
    // Given the recursion tree can be unbalanced, PyO3/Rayon parallelism needs careful chunking.
    // For simplicity, we just use a flat loop and parallelize at the root in the main function.
    
    for i in (tail_idx + 1)..active_items.len() {
        let (item_i, bs_i) = &active_items[i];
        let mut t_new = BitSet {
            blocks: vec![0u128; n_blocks],
        };
        let c = prefix_bs.intersect_count_into(bs_i, &mut t_new, min_count);
        
        if c >= min_count {
            // 1. Check PPC (Prefix-Preserving Closure) Pruning.
            let mut is_ppc = true;
            for j in 0..i {
                if !in_prefix[j] {
                    // if T_new is a subset of T_j, prune!
                    if t_new.is_subset_of(&active_items[j].1) {
                        is_ppc = false;
                        break;
                    }
                }
            }
            if !is_ppc {
                continue;
            }

            // 2. Compute closure (items > i)
            let mut closure = Vec::with_capacity(prefix.len() + 1);
            closure.extend_from_slice(prefix);
            closure.push(*item_i);

            for k in (i + 1)..active_items.len() {
                if t_new.is_subset_of(&active_items[k].1) {
                    closure.push(active_items[k].0);
                }
            }

            // 3. Record closed itemset
            if max_len.is_none_or(|ml| closure.len() <= ml) {
                results.push((c, closure.clone()));

                // 4. Recurse
                if max_len.is_none_or(|ml| closure.len() < ml) {
                    results.extend(lcm_mine(&closure, &t_new, i, active_items, min_count, max_len, global_to_local));
                }
            }
        }
    }

    results
}

fn mine_itemsets_lcm(
    active_items: Vec<(u32, BitSet)>,
    global_to_local: Vec<u32>,
    min_count: u64,
    max_len: Option<usize>,
    _n_rows: usize,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    let active_items_arc = Arc::new(active_items);

    // Root transaction set contains all transactions (all 1s)
    // BitSet now uses u128 blocks (128 transactions per block)
    // _root_bs is not used currently

    // Parallelize the first level of the tree
    let results: Vec<(u64, Vec<u32>)> = (0..active_items_arc.len())
        .into_par_iter()
        .flat_map(|i| {
            let mut sub_results = Vec::new();
            let (item_i, bs_i) = &active_items_arc[i];
            let c = bs_i.count_ones();
            
            if c >= min_count {
                let mut is_ppc = true;
                for j in 0..i {
                    if bs_i.is_subset_of(&active_items_arc[j].1) {
                        is_ppc = false;
                        break;
                    }
                }

                if is_ppc {
                    let mut closure = Vec::with_capacity(1);
                    closure.push(*item_i);

                    for k in (i + 1)..active_items_arc.len() {
                        if bs_i.is_subset_of(&active_items_arc[k].1) {
                            closure.push(active_items_arc[k].0);
                        }
                    }

                    if max_len.is_none_or(|ml| closure.len() <= ml) {
                        sub_results.push((c, closure.clone()));

                        if max_len.is_none_or(|ml| closure.len() < ml) {
                            sub_results.extend(lcm_mine(
                                &closure,
                                bs_i,
                                i,
                                &active_items_arc,
                                min_count,
                                max_len,
                                &global_to_local,
                            ));
                        }
                    }
                }
            }
            sub_results
        })
        .collect();

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (data, min_count, max_len=None))]
pub fn lcm_from_dense<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<u8>,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let array = data.as_array();
    let (n_rows, n_cols) = (array.shape()[0], array.shape()[1]);

    if n_rows == 0 || n_cols == 0 {
        return Ok((
            Vec::<u64>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
        ));
    }

    let flat = array.as_slice().unwrap();
    let item_count = (0..n_cols)
        .into_par_iter()
        .map(|c| {
            let mut count = 0u64;
            for r in 0..n_rows {
                if unsafe { *flat.get_unchecked(r * n_cols + c) } != 0 {
                    count += 1;
                }
            }
            count
        })
        .collect::<Vec<u64>>();

    let (global_to_local, original_items, frequent_cols, frequent_len) =
        match process_item_counts(item_count, min_count, n_cols) {
            Some(v) => v,
            None => {
                return Ok((
                    Vec::<u64>::new().into_pyarray(py),
                    Vec::<u32>::new().into_pyarray(py),
                    Vec::<u32>::new().into_pyarray(py),
                ))
            }
        };

    let mut bitsets = vec![BitSet::new(n_rows); frequent_len];
    for (r, row) in flat.chunks(n_cols).enumerate() {
        for &c in &frequent_cols {
            if unsafe { *row.get_unchecked(c) } != 0 {
                let local_id = global_to_local[c];
                bitsets[local_id as usize].set(r);
            }
        }
    }

    let active_items: Vec<(u32, BitSet)> = original_items.into_iter().zip(bitsets).collect();

    let results = mine_itemsets_lcm(active_items, global_to_local, min_count, max_len, n_rows)?;
    let (flat_supports, flat_offsets, flat_items) = flatten_results(results);

    Ok((
        flat_supports.into_pyarray(py),
        flat_offsets.into_pyarray(py),
        flat_items.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_cols, min_count, max_len=None))]
pub fn lcm_from_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i32>,
    indices: PyReadonlyArray1<i32>,
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let indptr = indptr.as_slice().unwrap();
    let indices = indices.as_slice().unwrap();
    let n_rows = indptr.len().saturating_sub(1);

    if n_rows == 0 || n_cols == 0 {
        return Ok((
            Vec::<u64>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
        ));
    }
    
    let item_count: Vec<u64> = (0..n_rows)
        .into_par_iter()
        .fold(
            || vec![0u64; n_cols],
            |mut acc, row| {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                for &col in &indices[start..end] {
                    let c = col as usize;
                    if c < n_cols {
                        unsafe { *acc.get_unchecked_mut(c) += 1; }
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0u64; n_cols],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );

    let (global_to_local, original_items, _frequent_cols, frequent_len) =
        match process_item_counts(item_count, min_count, n_cols) {
            Some(v) => v,
            None => {
                return Ok((
                    Vec::<u64>::new().into_pyarray(py),
                    Vec::<u32>::new().into_pyarray(py),
                    Vec::<u32>::new().into_pyarray(py),
                ))
            }
        };

    let mut bitsets = vec![BitSet::new(n_rows); frequent_len];
    for r in 0..n_rows {
        let start = indptr[r] as usize;
        let end = indptr[r + 1] as usize;
        for &col in &indices[start..end] {
            if (col as usize) < n_cols {
                let local_id = global_to_local[col as usize];
                if local_id != u32::MAX {
                    bitsets[local_id as usize].set(r);
                }
            }
        }
    }

    let active_items: Vec<(u32, BitSet)> = original_items.into_iter().zip(bitsets).collect();

    let results = mine_itemsets_lcm(active_items, global_to_local, min_count, max_len, n_rows)?;
    let (flat_supports, flat_offsets, flat_items) = flatten_results(results);

    Ok((
        flat_supports.into_pyarray(py),
        flat_offsets.into_pyarray(py),
        flat_items.into_pyarray(py),
    ))
}
