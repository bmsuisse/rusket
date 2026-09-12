use pyo3::prelude::*;

type Transaction = (Vec<u32>, Vec<f32>);
type ProjectedDB = Vec<(usize, usize, f32)>;

fn hupm_mine_recursive(
    transactions: &[Transaction],
    pdb: &ProjectedDB,
    min_utility: f32,
    max_len: Option<usize>,
    current_pattern: &mut Vec<u32>,
    results: &mut Vec<(f32, Vec<u32>)>,
    twu: &mut [f32],
    exact: &mut [f32],
    seen: &mut [bool],
    touched: &mut Vec<u32>,
) {
    if let Some(ml) = max_len {
        if current_pattern.len() >= ml {
            return;
        }
    }

    touched.clear();

    for &(tx_idx, start_pos, prefix_util) in pdb {
        let (items, utils) = &transactions[tx_idx];

        let mut remaining_util = 0.0;
        for j in start_pos..items.len() {
            remaining_util += utils[j];
        }

        let local_tu = prefix_util + remaining_util;

        for j in start_pos..items.len() {
            let item = items[j] as usize;
            let item_u = utils[j];

            if !seen[item] {
                seen[item] = true;
                touched.push(item as u32);
            }
            twu[item] += local_tu;
            exact[item] += prefix_util + item_u;
        }
    }

    // Capture (item, exact_utility) for promising items before resetting the
    // shared buffers, since children reuse them (same pattern as prefixspan).
    let mut promising_items: Vec<(u32, f32)> = touched
        .iter()
        .copied()
        .filter_map(|item| {
            let idx = item as usize;
            if twu[idx] >= min_utility {
                Some((item, exact[idx]))
            } else {
                None
            }
        })
        .collect();

    promising_items.sort_unstable_by_key(|&(item, _)| item);

    for &item in touched.iter() {
        let idx = item as usize;
        seen[idx] = false;
        twu[idx] = 0.0;
        exact[idx] = 0.0;
    }

    for (item, exact_u) in promising_items {
        current_pattern.push(item);

        if exact_u >= min_utility {
            results.push((exact_u, current_pattern.clone()));
        }

        let mut new_pdb: ProjectedDB = Vec::with_capacity(pdb.len());
        for &(tx_idx, start_pos, prefix_util) in pdb.iter() {
            let (items, utils) = &transactions[tx_idx];

            if let Some(offset) = items[start_pos..].iter().position(|&x| x == item) {
                let actual_pos = start_pos + offset;
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
                twu,
                exact,
                seen,
                touched,
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
    let mut pdb = Vec::with_capacity(transactions.len());
    let mut max_item: i64 = -1;
    let mut distinct: ahash::AHashSet<u32> = ahash::AHashSet::new();
    for i in 0..transactions.len() {
        if !transactions[i].0.is_empty() {
            pdb.push((i, 0, 0.0));
            for &item in &transactions[i].0 {
                if item as i64 > max_item {
                    max_item = item as i64;
                }
                distinct.insert(item);
            }
        }
    }

    let mut results = Vec::new();
    let mut current_pattern = Vec::new();

    if max_item < 0 {
        return results;
    }

    // Flat Vecs indexed by item id let the recursion reuse scratch buffers
    // instead of rebuilding a HashMap<u32, f32> at every node. That indexing
    // is only affordable when ids are dense: callers pass raw ids (SKUs,
    // hashed ids, EANs), so a single transaction holding item id 4e9 would
    // otherwise ask for ~36 GB. When the id space is sparse, remap to a dense
    // 0..n_distinct range for the duration of the mine and map back on the way
    // out.
    let n_distinct = distinct.len();
    let dense_span = max_item as usize + 1;
    // ponytail: 4x slack, not a tuned ratio — dense-enough ids skip the remap
    // entirely and keep the old zero-overhead path.
    let remap = dense_span > 4 * n_distinct.max(1);

    let (transactions_owned, id_of_dense);
    let transactions: &[Transaction] = if remap {
        let mut ids: Vec<u32> = distinct.into_iter().collect();
        ids.sort_unstable();
        let dense_of_id: ahash::AHashMap<u32, u32> =
            ids.iter().enumerate().map(|(d, &orig)| (orig, d as u32)).collect();
        transactions_owned = transactions
            .iter()
            .map(|t| (t.0.iter().map(|i| dense_of_id[i]).collect::<Vec<u32>>(), t.1.clone()))
            .collect::<Vec<Transaction>>();
        id_of_dense = ids;
        &transactions_owned
    } else {
        id_of_dense = Vec::new();
        transactions
    };

    let size = if remap { n_distinct } else { dense_span };
    let mut twu = vec![0.0f32; size];
    let mut exact = vec![0.0f32; size];
    let mut seen = vec![false; size];
    let mut touched: Vec<u32> = Vec::new();

    hupm_mine_recursive(
        transactions,
        &pdb,
        min_utility,
        max_len,
        &mut current_pattern,
        &mut results,
        &mut twu,
        &mut exact,
        &mut seen,
        &mut touched,
    );

    if remap {
        // Patterns came back in dense ids; translate them to the caller's ids.
        for (_, pattern) in results.iter_mut() {
            for item in pattern.iter_mut() {
                *item = id_of_dense[*item as usize];
            }
        }
    }

    results
}

#[pyfunction]
#[pyo3(signature = (items_list, utils_list, min_utility, max_len=None))]
pub fn hupm_mine_py(
    py: Python<'_>,
    items_list: Vec<Vec<u32>>,
    utils_list: Vec<Vec<f32>>,
    min_utility: f32,
    max_len: Option<usize>,
) -> PyResult<(Vec<f32>, Vec<Vec<u32>>)> {
    if items_list.len() != utils_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "items_list and utils_list must have the same length",
        ));
    }

    py.detach(|| -> PyResult<_> {
        let mut transactions = Vec::with_capacity(items_list.len());
        for (items, utils) in items_list.into_iter().zip(utils_list.into_iter()) {
            if items.len() != utils.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Each inner list of items and utils must have the same length",
                ));
            }

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
    })
}
