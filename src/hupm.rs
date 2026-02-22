use pyo3::prelude::*;
use std::collections::HashMap;

/// High-Utility Pattern Mining (HUPM) using depth-first projection.
/// Similar to EFIM, but simplified.
/// Transactions contain sorted items (by TWU or frequency globally) and their utilities.

type Transaction = (Vec<u32>, Vec<f32>);
// Projection: (transaction_idx, starting_position, prefix_utility)
type ProjectedDB = Vec<(usize, usize, f32)>;

fn hupm_mine_recursive(
    transactions: &[Transaction],
    pdb: &ProjectedDB,
    min_utility: f32,
    max_len: Option<usize>,
    current_pattern: &mut Vec<u32>,
    results: &mut Vec<(f32, Vec<u32>)>,
) {
    if let Some(ml) = max_len {
        if current_pattern.len() >= ml {
            return;
        }
    }

    let mut item_twu: HashMap<u32, f32> = HashMap::new();
    let mut item_exact_utility: HashMap<u32, f32> = HashMap::new();

    // 1. Scan projected DB to find TWU and Exact Utility for each item
    for &(tx_idx, start_pos, prefix_util) in pdb {
        let (items, utils) = &transactions[tx_idx];

        // Calculate remaining utility (RU) in this transaction from start_pos
        let mut remaining_util = 0.0;
        for j in start_pos..items.len() {
            remaining_util += utils[j];
        }

        let local_tu = prefix_util + remaining_util;

        for j in start_pos..items.len() {
            let item = items[j];
            let item_u = utils[j];

            *item_twu.entry(item).or_insert(0.0) += local_tu;
            *item_exact_utility.entry(item).or_insert(0.0) += prefix_util + item_u;
        }
    }

    // 2. Identify promising items (TWU >= min_util)
    let mut promising_items: Vec<u32> = item_twu
        .into_iter()
        .filter(|&(_, twu)| twu >= min_utility)
        .map(|(item, _)| item)
        .collect();

    // Optimization: Mine extensions in order
    promising_items.sort_unstable();

    // 3. Project and recurse
    for item in promising_items {
        current_pattern.push(item);

        let exact_u = *item_exact_utility.get(&item).unwrap_or(&0.0);
        if exact_u >= min_utility {
            results.push((exact_u, current_pattern.clone()));
        }

        let mut new_pdb: ProjectedDB = Vec::with_capacity(pdb.len());
        for &(tx_idx, start_pos, prefix_util) in pdb.iter() {
            let (items, utils) = &transactions[tx_idx];

            // Find position of `item`
            if let Some(offset) = items[start_pos..].iter().position(|&x| x == item) {
                let actual_pos = start_pos + offset;
                // If there are more items after this one, we can project
                if actual_pos + 1 < items.len() {
                    let new_prefix_util = prefix_util + utils[actual_pos];
                    new_pdb.push((tx_idx, actual_pos + 1, new_prefix_util));
                }
            }
        }

        if !new_pdb.is_empty() {
            hupm_mine_recursive(
                transactions,
                &new_pdb,
                min_utility,
                max_len,
                current_pattern,
                results,
            );
        }

        current_pattern.pop();
    }
}

pub fn hupm_simple(
    transactions: &[Transaction],
    min_utility: f32,
    max_len: Option<usize>,
) -> Vec<(f32, Vec<u32>)> {
    // Initial projecting
    let mut pdb = Vec::with_capacity(transactions.len());
    for i in 0..transactions.len() {
        if !transactions[i].0.is_empty() {
            pdb.push((i, 0, 0.0));
        }
    }

    let mut results = Vec::new();
    let mut current_pattern = Vec::new();

    hupm_mine_recursive(
        transactions,
        &pdb,
        min_utility,
        max_len,
        &mut current_pattern,
        &mut results,
    );

    results
}

#[pyfunction]
#[pyo3(signature = (items_list, utils_list, min_utility, max_len=None))]
pub fn hupm_mine_py(
    items_list: Vec<Vec<u32>>,
    utils_list: Vec<Vec<f32>>,
    min_utility: f32,
    max_len: Option<usize>,
) -> PyResult<(Vec<f32>, Vec<Vec<u32>>)> {
    // Combine lists into standard transaction rep
    if items_list.len() != utils_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "items_list and utils_list must have the same length",
        ));
    }

    let mut transactions = Vec::with_capacity(items_list.len());
    for (items, utils) in items_list.into_iter().zip(utils_list.into_iter()) {
        if items.len() != utils.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each inner list of items and utils must have the same length",
            ));
        }
        // Internally, items within a transaction MUST be sorted by some global order.
        // For simplicity, we just sort by item ID here natively to ensure combinations
        // are deterministic. EFIM sorts by TWU.

        let mut pairs: Vec<(u32, f32)> = items.into_iter().zip(utils.into_iter()).collect();
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let sorted_items: Vec<u32> = pairs.iter().map(|&(i, _)| i).collect();
        let sorted_utils: Vec<f32> = pairs.iter().map(|&(_, u)| u).collect();

        transactions.push((sorted_items, sorted_utils));
    }

    let raw_res = hupm_simple(&transactions, min_utility, max_len);

    let mut utilities = Vec::with_capacity(raw_res.len());
    let mut patterns = Vec::with_capacity(raw_res.len());

    for (util, pat) in raw_res {
        utilities.push(util);
        patterns.push(pat);
    }

    Ok((utilities, patterns))
}
