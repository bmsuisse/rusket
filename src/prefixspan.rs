use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

fn prefixspan_simple(
    sequences: &[Vec<u32>],
    min_support: usize,
    max_len: Option<usize>,
) -> Vec<(usize, Vec<u32>)> {
    let mut max_item_id = 0;
    for seq in sequences {
        for &item in seq {
            if item > max_item_id {
                max_item_id = item;
            }
        }
    }

    // Pseudo-projected database: holds (seq_id, start_index)
    let mut pdb = Vec::with_capacity(sequences.len());
    for i in 0..sequences.len() {
        if !sequences[i].is_empty() {
            pdb.push((i, 0));
        }
    }

    let mut results = Vec::new();
    let mut current_pattern = Vec::new();

    let mut item_counts = vec![0_usize; max_item_id as usize + 1];
    let mut last_seen_seq = vec![usize::MAX; max_item_id as usize + 1];

    prefixspan_mine(
        sequences,
        &pdb,
        min_support,
        max_len,
        &mut current_pattern,
        &mut results,
        &mut item_counts,
        &mut last_seen_seq,
    );
    results
}

fn prefixspan_mine(
    sequences: &[Vec<u32>],
    pdb: &[(usize, usize)],
    min_support: usize,
    max_len: Option<usize>,
    current_pattern: &mut Vec<u32>,
    results: &mut Vec<(usize, Vec<u32>)>,
    item_counts: &mut [usize],
    last_seen_seq: &mut [usize],
) {
    if let Some(ml) = max_len {
        if current_pattern.len() >= ml {
            return;
        }
    }

    let mut active_items = Vec::new();

    for &(seq_id, start_idx) in pdb {
        let seq = &sequences[seq_id];
        for i in start_idx..seq.len() {
            let item_idx = seq[i] as usize;
            if last_seen_seq[item_idx] != seq_id {
                last_seen_seq[item_idx] = seq_id;
                
                if item_counts[item_idx] == 0 {
                    active_items.push(item_idx as u32);
                }
                item_counts[item_idx] += 1;
            }
        }
    }

    let mut frequent_items = Vec::new();
    for &item in &active_items {
        let count = item_counts[item as usize];
        if count >= min_support {
            frequent_items.push((item, count));
        }
        // Reset the counters for the next recursive branch
        item_counts[item as usize] = 0;
        last_seen_seq[item as usize] = usize::MAX;
    }

    frequent_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    for (item, support) in frequent_items {
        current_pattern.push(item);
        results.push((support, current_pattern.clone()));

        let mut new_pdb = Vec::new();
        for &(seq_id, start_idx) in pdb.iter() {
            let seq = &sequences[seq_id];
            if let Some(pos) = seq[start_idx..].iter().position(|&x| x == item) {
                let abs_pos = start_idx + pos;
                if abs_pos + 1 < seq.len() {
                    new_pdb.push((seq_id, abs_pos + 1));
                }
            }
        }

        if !new_pdb.is_empty() {
            prefixspan_mine(
                sequences, 
                &new_pdb, 
                min_support, 
                max_len, 
                current_pattern, 
                results, 
                item_counts, 
                last_seen_seq
            );
        }

        current_pattern.pop();
    }
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, min_count, max_len=None))]
pub fn prefixspan_mine_py<'py>(
    indptr: PyReadonlyArray1<'py, usize>,
    indices: PyReadonlyArray1<'py, u32>,
    min_count: usize,
    max_len: Option<usize>,
) -> PyResult<(Vec<usize>, Vec<Vec<u32>>)> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;

    let mut sequences = Vec::with_capacity(indptr_slice.len().saturating_sub(1));
    for i in 0..indptr_slice.len().saturating_sub(1) {
        let start = indptr_slice[i];
        let end = indptr_slice[i + 1];
        let seq = indices_slice[start..end].to_vec();
        sequences.push(seq);
    }

    let raw_res = prefixspan_simple(&sequences, min_count, max_len);

    let mut supports = Vec::with_capacity(raw_res.len());
    let mut patterns = Vec::with_capacity(raw_res.len());

    for (sup, pat) in raw_res {
        supports.push(sup);
        patterns.push(pat);
    }

    Ok((supports, patterns))
}
