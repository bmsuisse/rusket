use pyo3::prelude::*;
use std::collections::HashMap;

fn prefixspan_simple(
    sequences: &[Vec<u32>],
    min_support: usize,
    max_len: Option<usize>,
) -> Vec<(usize, Vec<u32>)> {
    let mut pdb = Vec::with_capacity(sequences.len());
    for (i, seq) in sequences.iter().enumerate() {
        pdb.push((seq.as_slice(), i));
    }

    let mut results = Vec::new();
    let mut current_pattern = Vec::new();
    prefixspan_mine(
        &pdb,
        min_support,
        max_len,
        &mut current_pattern,
        &mut results,
    );
    results
}

fn prefixspan_mine(
    pdb: &[(&[u32], usize)],
    min_support: usize,
    max_len: Option<usize>,
    current_pattern: &mut Vec<u32>,
    results: &mut Vec<(usize, Vec<u32>)>,
) {
    if let Some(ml) = max_len {
        if current_pattern.len() >= ml {
            return;
        }
    }

    let mut item_counts: HashMap<u32, usize> = HashMap::new();
    let mut last_seen_seq: HashMap<u32, usize> = HashMap::new();

    for &(seq, seq_id) in pdb {
        for &item in seq {
            if last_seen_seq.get(&item) != Some(&seq_id) {
                last_seen_seq.insert(item, seq_id);
                *item_counts.entry(item).or_insert(0) += 1;
            }
        }
    }

    let mut frequent_items: Vec<(u32, usize)> = item_counts
        .into_iter()
        .filter(|&(_, count)| count >= min_support)
        .collect();

    frequent_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    for (item, support) in frequent_items {
        current_pattern.push(item);
        results.push((support, current_pattern.clone()));

        let mut new_pdb = Vec::new();
        for &(seq, seq_id) in pdb.iter() {
            if let Some(pos) = seq.iter().position(|&x| x == item) {
                if pos + 1 < seq.len() {
                    new_pdb.push((&seq[pos + 1..], seq_id));
                }
            }
        }

        if !new_pdb.is_empty() {
            prefixspan_mine(&new_pdb, min_support, max_len, current_pattern, results);
        }

        current_pattern.pop();
    }
}

#[pyfunction]
#[pyo3(signature = (sequences, min_count, max_len=None))]
pub fn prefixspan_mine_py(
    sequences: Vec<Vec<u32>>,
    min_count: usize,
    max_len: Option<usize>,
) -> PyResult<(Vec<usize>, Vec<Vec<u32>>)> {
    let raw_res = prefixspan_simple(&sequences, min_count, max_len);

    let mut supports = Vec::with_capacity(raw_res.len());
    let mut patterns = Vec::with_capacity(raw_res.len());

    for (sup, pat) in raw_res {
        supports.push(sup);
        patterns.push(pat);
    }

    Ok((supports, patterns))
}
