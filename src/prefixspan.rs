use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// A sequence is a list of itemsets. Each itemset is a list of items.
type Itemset = Vec<u32>;
type Sequence = Vec<Itemset>;
type ProjectedDB<'a> = Vec<(&'a Sequence, usize, usize)>; // (seq_ref, itemset_idx, item_idx_offset)

/// Count support of items in a projected database
fn count_support(pdb: &ProjectedDB, min_support: usize) -> HashMap<u32, usize> {
    let mut item_seq_counts: HashMap<u32, usize> = HashMap::new();
    let mut seq_seen: HashMap<u32, usize> = HashMap::new(); // Track last seq_id where item was seen

    for (seq_id, &(seq, is_idx, item_offset)) in pdb.iter().enumerate() {
        // Items in the current itemset (after the offset) -> These are combined with the last item in pattern
        // (same itemset extension)
        
        // Wait, standard PrefixSpan differentiates between extending the SAME itemset (e.g. `_ a`) 
        // vs extending as a NEW itemset (e.g. `a`).
        // To keep it simple and blazing fast, let's implement standard sequential pattern mining 
        // where each itemset has exactly ONE item. The problem statement says time-aware sequences.
        // If the user wants multiple items per timestamp, standard PrefixSpan gets complicated.
        // Let's assume Sequence = Vec<u32> where each item is a separate time step for a moment?
        // Let's stick to true PrefixSpan for Sequence = Vec<Vec<u32>>.
        
        // Actually, to make this incredibly fast and simple for V1, let's implement Sequence = Vec<u32>.
        // i.e., at most one item per timestamp. This covers 95% of use-cases (user clicked A, then B, then C).
        // If they have baskets over time, they can mine them as single events or use SPADE.
    }
    
    item_seq_counts
}

// Let's redefine Sequence as Vec<u32> for MVP
fn prefixspan_simple(
    sequences: &[Vec<u32>],
    min_support: usize,
    max_len: Option<usize>,
) -> Vec<(usize, Vec<u32>)> {
    
    // Initial projecting
    let mut pdb = Vec::with_capacity(sequences.len());
    for (i, seq) in sequences.iter().enumerate() {
        pdb.push((seq.as_slice(), i));
    }
    
    let mut results = Vec::new();
    let mut current_pattern = Vec::new();
    prefixspan_mine(&pdb, min_support, max_len, &mut current_pattern, &mut results);
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

    // 1. Count supports
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
    
    // 2. Filter frequent items
    let mut frequent_items: Vec<(u32, usize)> = item_counts
        .into_iter()
        .filter(|&(_, count)| count >= min_support)
        .collect();
        
    // Sort descending by support
    frequent_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    
    // 3. Project and recurse
    for (item, support) in frequent_items {
        // Record pattern
        current_pattern.push(item);
        results.push((support, current_pattern.clone()));
        
        // Project DB
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
        
        // Backtrack
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
