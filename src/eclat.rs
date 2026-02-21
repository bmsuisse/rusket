use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;


use crate::fpgrowth::{process_item_counts, flatten_results};

/// Simple BitSet backed by `Vec<u64>`
#[derive(Clone)]
struct BitSet {
    blocks: Vec<u64>,
}

impl BitSet {
    fn new(num_bits: usize) -> Self {
        let num_blocks = (num_bits + 63) / 64;
        BitSet {
            blocks: vec![0; num_blocks],
        }
    }

    #[inline]
    fn set(&mut self, bit: usize) {
        self.blocks[bit / 64] |= 1 << (bit % 64);
    }

    #[inline]
    fn count_ones(&self) -> u64 {
        self.blocks.iter().map(|b| b.count_ones() as u64).sum()
    }

    #[inline]
    fn intersect(&self, other: &BitSet) -> BitSet {
        let blocks = self.blocks
            .iter()
            .zip(other.blocks.iter())
            .map(|(a, b)| a & b)
            .collect();
        BitSet { blocks }
    }
}

pub(crate) fn eclat_mine(
    prefix: &[u32],
    active_items: &[(u32, BitSet)],
    min_count: u64,
    max_len: Option<usize>,
) -> Vec<(u64, Vec<u32>)> {
    let mut results = Vec::new();
    let new_len = prefix.len() + 1;

    for (i, (item_a, bs_a)) in active_items.iter().enumerate() {
        let count = bs_a.count_ones();
        if count < min_count { continue; }

        if max_len.map_or(true, |ml| new_len <= ml) {
            let mut iset = Vec::with_capacity(new_len);
            iset.extend_from_slice(prefix);
            iset.push(*item_a);
            results.push((count, iset.clone()));

            if max_len.map_or(true, |ml| new_len < ml) {
                let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
                for (item_b, bs_b) in &active_items[i + 1..] {
                    let new_bs = bs_a.intersect(bs_b);
                    if new_bs.count_ones() >= min_count {
                        next_active.push((*item_b, new_bs));
                    }
                }
                
                if !next_active.is_empty() {
                    results.extend(eclat_mine(&iset, &next_active, min_count, max_len));
                }
            }
        }
    }

    results
}

#[pyfunction]
#[pyo3(signature = (data, min_count, max_len=None))]
pub fn eclat_from_dense(
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

    let active_items: Vec<(u32, BitSet)> = original_items
        .into_iter()
        .zip(bitsets.into_iter())
        .collect();

    let results: Vec<(u64, Vec<u32>)> = active_items
        .par_iter()
        .enumerate()
        .flat_map(|(i, (item_a, bs_a))| {
            let mut sub_results = Vec::new();
            let count = bs_a.count_ones();
            if count >= min_count {
                let iset = vec![*item_a];
                sub_results.push((count, iset.clone()));

                if max_len.map_or(true, |ml| ml > 1) {
                    let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
                    for (item_b, bs_b) in &active_items[i + 1..] {
                        let new_bs = bs_a.intersect(bs_b);
                        if new_bs.count_ones() >= min_count {
                            next_active.push((*item_b, new_bs));
                        }
                    }
                    if !next_active.is_empty() {
                        sub_results.extend(eclat_mine(&iset, &next_active, min_count, max_len));
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
pub fn eclat_from_csr(
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

    // 3. Prepare initial active set
    let active_items: Vec<(u32, BitSet)> = original_items
        .into_iter()
        .zip(bitsets.into_iter())
        .collect();

    // 4. Mine (Parallel top level)
    let results: Vec<(u64, Vec<u32>)> = active_items
        .par_iter()
        .enumerate()
        .flat_map(|(i, (item_a, bs_a))| {
            let mut sub_results = Vec::new();
            let count = bs_a.count_ones();
            if count >= min_count {
                let iset = vec![*item_a];
                sub_results.push((count, iset.clone()));

                if max_len.map_or(true, |ml| ml > 1) {
                    let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
                    for (item_b, bs_b) in &active_items[i + 1..] {
                        let new_bs = bs_a.intersect(bs_b);
                        if new_bs.count_ones() >= min_count {
                            next_active.push((*item_b, new_bs));
                        }
                    }
                    if !next_active.is_empty() {
                        sub_results.extend(eclat_mine(&iset, &next_active, min_count, max_len));
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

/// Internal (non-pyfunction) Eclat implementation on raw CSR slices.
/// Used by FPMiner to avoid re-entering Python.
pub(crate) fn _eclat_mine_csr(
    indptr: &[i32],
    indices: &[i32],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Vec<u64>, Vec<u32>, Vec<u32>)> {
    use crate::fpgrowth::{process_item_counts, flatten_results};

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

    let active_items: Vec<(u32, BitSet)> = original_items
        .into_iter()
        .zip(bitsets.into_iter())
        .collect();

    use rayon::prelude::*;
    let results: Vec<(u64, Vec<u32>)> = active_items
        .par_iter()
        .enumerate()
        .flat_map(|(i, (item_a, bs_a))| {
            let mut sub_results = Vec::new();
            let count = bs_a.count_ones();
            if count >= min_count {
                let iset = vec![*item_a];
                sub_results.push((count, iset.clone()));
                if max_len.map_or(true, |ml| ml > 1) {
                    let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
                    for (item_b, bs_b) in &active_items[i + 1..] {
                        let new_bs = bs_a.intersect(bs_b);
                        if new_bs.count_ones() >= min_count {
                            next_active.push((*item_b, new_bs));
                        }
                    }
                    if !next_active.is_empty() {
                        sub_results.extend(eclat_mine(&iset, &next_active, min_count, max_len));
                    }
                }
            }
            sub_results
        })
        .collect();

    Ok(flatten_results(results))
}
