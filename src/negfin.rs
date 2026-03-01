use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::eclat::BitSet;
use crate::fpgrowth::{flatten_results, process_item_counts};

pub(crate) fn negfin_mine(
    prefix: &[u32],
    active_items: &[(u32, BitSet)],
    min_count: u64,
    max_len: Option<usize>,
) -> Vec<(u64, Vec<u32>)> {
    let mut results = Vec::new();
    let new_len = prefix.len() + 1;
    let n_blocks = active_items.first().map_or(0, |(_, bs)| bs.blocks.len());
    let mut scratch = BitSet {
        blocks: vec![0u128; n_blocks],
    };

    for (i, (item_a, bs_a)) in active_items.iter().enumerate() {
        let count_a = bs_a.count_ones();
        if count_a < min_count {
            continue;
        }

        let mut promoted = Vec::new();
        let mut next_active = Vec::with_capacity(active_items.len() - i - 1);

        // Evaluate all extensions for PEP and next recursion
        for (item_b, bs_b) in &active_items[i + 1..] {
            let count_ab = bs_a.intersect_count_into(bs_b, &mut scratch, min_count);
            if count_ab >= min_count {
                if count_ab == count_a {
                    // Parent Equivalence Pruning (PEP)
                    // item_b appears in EVERY transaction where item_a appears
                    promoted.push(*item_b);
                } else {
                    let mut fresh = BitSet {
                        blocks: vec![0u128; n_blocks],
                    };
                    std::mem::swap(&mut scratch, &mut fresh);
                    next_active.push((*item_b, fresh));
                }
            }
        }

        let mut base_iset = Vec::with_capacity(prefix.len() + 1);
        base_iset.extend_from_slice(prefix);
        base_iset.push(*item_a);

        let n_promoted = promoted.len();
        // Capping combinations to prevent extreme OOM on pathological data sets
        // (If an item has 60 promoted items, 2^60 is impossible to store).
        // 1 << 63 is the max for u64, but anything over 1 << 20 is dangerous for RAM.
        // We'll trust the caller's min_support/max_len. 
        // If max_len limits us, we only generate up to max_len.
        
        let n_combinations = 1u64.checked_shl(n_promoted as u32).unwrap_or(u64::MAX);

        for mask in 0..n_combinations {
            let mut combo = base_iset.clone();
            let mut valid = true;
            for (p_idx, &p_item) in promoted.iter().enumerate() {
                if (mask & (1u64 << p_idx)) != 0 {
                    combo.push(p_item);
                }
            }
            if let Some(ml) = max_len {
                if combo.len() > ml {
                    valid = false;
                }
            }
            if valid {
                results.push((count_a, combo));
            } else if max_len.is_some() && mask == 0 {
                // if even the empty mask is invalid (shouldn't happen here since new_len <= max_len via external check)
            }
        }

        if max_len.is_none() || new_len < max_len.unwrap() {
            if !next_active.is_empty() {
                let sub_results = negfin_mine(&base_iset, &next_active, min_count, max_len);
                if n_promoted == 0 {
                    results.extend(sub_results);
                } else {
                    for (c, s_iset) in sub_results {
                        for mask in 0..n_combinations {
                            let mut combo = s_iset.clone();
                            let mut valid = true;
                            for (p_idx, &p_item) in promoted.iter().enumerate() {
                                if (mask & (1u64 << p_idx)) != 0 {
                                    combo.push(p_item);
                                }
                            }
                            if let Some(ml) = max_len {
                                if combo.len() > ml {
                                    valid = false;
                                }
                            }
                            if valid {
                                results.push((c, combo));
                            }
                        }
                    }
                }
            }
        }
    }

    results
}

#[pyfunction]
#[pyo3(signature = (data, min_count, max_len=None))]
pub fn negfin_from_dense(
    py: Python<'_>,
    data: PyReadonlyArray2<u8>,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
    let array = data.as_array();
    let (n_rows, n_cols) = (array.shape()[0], array.shape()[1]);

    if n_rows == 0 || n_cols == 0 {
        return Ok((
            vec![].into_pyarray(py).into(),
            vec![].into_pyarray(py).into(),
            vec![].into_pyarray(py).into(),
        ));
    }

    let flat = array.as_slice().unwrap();
    let item_count = (0..n_cols)
        .into_par_iter()
        .map(|c| {
            let mut count = 0u64;
            for r in 0..n_rows {
                if flat[r * n_cols + c] != 0 {
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
                    vec![].into_pyarray(py).into(),
                    vec![].into_pyarray(py).into(),
                    vec![].into_pyarray(py).into(),
                ))
            }
        };

    let mut bitsets = vec![BitSet::new(n_rows); frequent_len];
    for (r, row) in flat.chunks(n_cols).enumerate() {
        for &c in &frequent_cols {
            if row[c] != 0 {
                let local_id = global_to_local[c];
                bitsets[local_id as usize].set(r);
            }
        }
    }

    let active_items: Vec<(u32, BitSet)> = original_items.into_iter().zip(bitsets).collect();

    let results: Vec<(u64, Vec<u32>)> = active_items
        .par_iter()
        .enumerate()
        .flat_map(|(i, (item_a, bs_a))| {
            let mut sub_results = Vec::new();
            let count = bs_a.count_ones();
            if count >= min_count {
                let mut promoted = Vec::new();
                let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
                let n_blocks = bs_a.blocks.len();
                let mut scratch = BitSet {
                    blocks: vec![0u128; n_blocks],
                };

                for (item_b, bs_b) in &active_items[i + 1..] {
                    let c = bs_a.intersect_count_into(bs_b, &mut scratch, min_count);
                    if c >= min_count {
                        if c == count {
                            promoted.push(*item_b);
                        } else {
                            let mut fresh = BitSet {
                                blocks: vec![0u128; n_blocks],
                            };
                            std::mem::swap(&mut scratch, &mut fresh);
                            next_active.push((*item_b, fresh));
                        }
                    }
                }

                let base_iset = vec![*item_a];
                let n_promoted = promoted.len();
                let n_combinations = 1u64.checked_shl(n_promoted as u32).unwrap_or(u64::MAX);

                for mask in 0..n_combinations {
                    let mut combo = base_iset.clone();
                    let mut valid = true;
                    for (p_idx, &p_item) in promoted.iter().enumerate() {
                        if (mask & (1u64 << p_idx)) != 0 {
                            combo.push(p_item);
                        }
                    }
                    if let Some(ml) = max_len {
                        if combo.len() > ml {
                            valid = false;
                        }
                    }
                    if valid {
                        sub_results.push((count, combo));
                    }
                }

                if max_len.is_none() || 1 < max_len.unwrap() {
                    if !next_active.is_empty() {
                        let rec_results = negfin_mine(&base_iset, &next_active, min_count, max_len);
                        if n_promoted == 0 {
                            sub_results.extend(rec_results);
                        } else {
                            for (rc, s_iset) in rec_results {
                                for mask in 0..n_combinations {
                                    let mut combo = s_iset.clone();
                                    let mut valid = true;
                                    for (p_idx, &p_item) in promoted.iter().enumerate() {
                                        if (mask & (1u64 << p_idx)) != 0 {
                                            combo.push(p_item);
                                        }
                                    }
                                    if let Some(ml) = max_len {
                                        if combo.len() > ml {
                                            valid = false;
                                        }
                                    }
                                    if valid {
                                        sub_results.push((rc, combo));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            sub_results
        })
        .collect();

    let (flat_supports, flat_offsets, flat_items) = flatten_results(results);

    Ok((
        flat_supports.into_pyarray(py).into(),
        flat_offsets.into_pyarray(py).into(),
        flat_items.into_pyarray(py).into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_cols, min_count, max_len=None))]
pub fn negfin_from_csr(
    py: Python<'_>,
    indptr: PyReadonlyArray1<i32>,
    indices: PyReadonlyArray1<i32>,
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
    let indptr = indptr.as_slice().unwrap();
    let indices = indices.as_slice().unwrap();
    let n_rows = indptr.len().saturating_sub(1);

    if n_rows == 0 || n_cols == 0 {
        return Ok((
            vec![].into_pyarray(py).into(),
            vec![].into_pyarray(py).into(),
            vec![].into_pyarray(py).into(),
        ));
    }
    let mut item_count = vec![0u64; n_cols];
    for &col in indices {
        if (col as usize) < n_cols {
            item_count[col as usize] += 1;
        }
    }

    let (global_to_local, original_items, _frequent_cols, frequent_len) =
        match process_item_counts(item_count, min_count, n_cols) {
            Some(v) => v,
            None => {
                return Ok((
                    vec![].into_pyarray(py).into(),
                    vec![].into_pyarray(py).into(),
                    vec![].into_pyarray(py).into(),
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

    let results: Vec<(u64, Vec<u32>)> = active_items
        .par_iter()
        .enumerate()
        .flat_map(|(i, (item_a, bs_a))| {
            let mut sub_results = Vec::new();
            let count = bs_a.count_ones();
            if count >= min_count {
                let mut promoted = Vec::new();
                let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
                let n_blocks = bs_a.blocks.len();
                let mut scratch = BitSet {
                    blocks: vec![0u128; n_blocks],
                };

                for (item_b, bs_b) in &active_items[i + 1..] {
                    let c = bs_a.intersect_count_into(bs_b, &mut scratch, min_count);
                    if c >= min_count {
                        if c == count {
                            promoted.push(*item_b);
                        } else {
                            let mut fresh = BitSet {
                                blocks: vec![0u128; n_blocks],
                            };
                            std::mem::swap(&mut scratch, &mut fresh);
                            next_active.push((*item_b, fresh));
                        }
                    }
                }

                let base_iset = vec![*item_a];
                let n_promoted = promoted.len();
                let n_combinations = 1u64.checked_shl(n_promoted as u32).unwrap_or(u64::MAX);

                for mask in 0..n_combinations {
                    let mut combo = base_iset.clone();
                    let mut valid = true;
                    for (p_idx, &p_item) in promoted.iter().enumerate() {
                        if (mask & (1u64 << p_idx)) != 0 {
                            combo.push(p_item);
                        }
                    }
                    if let Some(ml) = max_len {
                        if combo.len() > ml {
                            valid = false;
                        }
                    }
                    if valid {
                        sub_results.push((count, combo));
                    }
                }

                if max_len.is_none() || 1 < max_len.unwrap() {
                    if !next_active.is_empty() {
                        let rec_results = negfin_mine(&base_iset, &next_active, min_count, max_len);
                        if n_promoted == 0 {
                            sub_results.extend(rec_results);
                        } else {
                            for (rc, s_iset) in rec_results {
                                for mask in 0..n_combinations {
                                    let mut combo = s_iset.clone();
                                    let mut valid = true;
                                    for (p_idx, &p_item) in promoted.iter().enumerate() {
                                        if (mask & (1u64 << p_idx)) != 0 {
                                            combo.push(p_item);
                                        }
                                    }
                                    if let Some(ml) = max_len {
                                        if combo.len() > ml {
                                            valid = false;
                                        }
                                    }
                                    if valid {
                                        sub_results.push((rc, combo));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            sub_results
        })
        .collect();

    let (flat_supports, flat_offsets, flat_items) = flatten_results(results);

    Ok((
        flat_supports.into_pyarray(py).into(),
        flat_offsets.into_pyarray(py).into(),
        flat_items.into_pyarray(py).into(),
    ))
}

pub(crate) fn _negfin_mine_csr(
    indptr: &[i32],
    indices: &[i32],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Vec<u64>, Vec<u32>, Vec<u32>)> {
    let n_rows = indptr.len().saturating_sub(1);
    if n_rows == 0 || n_cols == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    let mut item_count = vec![0u64; n_cols];
    for &col in indices {
        if (col as usize) < n_cols {
            item_count[col as usize] += 1;
        }
    }

    let (global_to_local, original_items, _frequent_cols, frequent_len) =
        match process_item_counts(item_count, min_count, n_cols) {
            Some(v) => v,
            None => return Ok((vec![], vec![], vec![])),
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

    let results: Vec<(u64, Vec<u32>)> = active_items
        .par_iter()
        .enumerate()
        .flat_map(|(i, (item_a, bs_a))| {
            let mut sub_results = Vec::new();
            let count = bs_a.count_ones();
            if count >= min_count {
                let mut promoted = Vec::new();
                let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
                let n_blocks = bs_a.blocks.len();
                let mut scratch = BitSet {
                    blocks: vec![0u128; n_blocks],
                };

                for (item_b, bs_b) in &active_items[i + 1..] {
                    let c = bs_a.intersect_count_into(bs_b, &mut scratch, min_count);
                    if c >= min_count {
                        if c == count {
                            promoted.push(*item_b);
                        } else {
                            let mut fresh = BitSet {
                                blocks: vec![0u128; n_blocks],
                            };
                            std::mem::swap(&mut scratch, &mut fresh);
                            next_active.push((*item_b, fresh));
                        }
                    }
                }

                let base_iset = vec![*item_a];
                let n_promoted = promoted.len();
                let n_combinations = 1u64.checked_shl(n_promoted as u32).unwrap_or(u64::MAX);

                for mask in 0..n_combinations {
                    let mut combo = base_iset.clone();
                    let mut valid = true;
                    for (p_idx, &p_item) in promoted.iter().enumerate() {
                        if (mask & (1u64 << p_idx)) != 0 {
                            combo.push(p_item);
                        }
                    }
                    if let Some(ml) = max_len {
                        if combo.len() > ml {
                            valid = false;
                        }
                    }
                    if valid {
                        sub_results.push((count, combo));
                    }
                }

                if max_len.is_none() || 1 < max_len.unwrap() {
                    if !next_active.is_empty() {
                        let rec_results = negfin_mine(&base_iset, &next_active, min_count, max_len);
                        if n_promoted == 0 {
                            sub_results.extend(rec_results);
                        } else {
                            for (rc, s_iset) in rec_results {
                                for mask in 0..n_combinations {
                                    let mut combo = s_iset.clone();
                                    let mut valid = true;
                                    for (p_idx, &p_item) in promoted.iter().enumerate() {
                                        if (mask & (1u64 << p_idx)) != 0 {
                                            combo.push(p_item);
                                        }
                                    }
                                    if let Some(ml) = max_len {
                                        if combo.len() > ml {
                                            valid = false;
                                        }
                                    }
                                    if valid {
                                        sub_results.push((rc, combo));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            sub_results
        })
        .collect();

    Ok(flatten_results(results))
}
