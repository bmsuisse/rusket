use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;

const PAR_ITEMS_CUTOFF: usize = 4;

#[derive(Debug, Clone)]
pub(crate) struct FPNode {
    pub item: u32,
    pub count: u64,
    pub parent: usize,
    pub children: SmallVec<[(u32, usize); 4]>,
}

impl FPNode {
    #[inline]
    fn new(item: u32, count: u64, parent: usize) -> Self {
        FPNode { item, count, parent, children: SmallVec::new() }
    }

    #[inline]
    fn find_child(&self, item: u32) -> Option<usize> {
        for &(k, v) in &self.children { if k == item { return Some(v); } }
        None
    }

    #[inline]
    fn add_child(&mut self, item: u32, idx: usize) { self.children.push((item, idx)); }
}

pub(crate) struct FPTree {
    pub nodes: Vec<FPNode>,
    pub item_nodes: Vec<Vec<usize>>,
    pub original_items: Vec<u32>,
    pub cond_items: Vec<u32>,
}

impl FPTree {
    pub fn new(num_items: usize, original_items: Vec<u32>) -> Self {
        let root = FPNode::new(u32::MAX, 0, 0);
        let mut nodes = Vec::with_capacity(256);
        nodes.push(root);
        FPTree {
            nodes,
            item_nodes: vec![Vec::new(); num_items],
            original_items,
            cond_items: Vec::new(),
        }
    }

    pub fn is_path(&self) -> bool {
        for node in &self.nodes {
            if node.children.len() > 1 { return false; }
        }
        true
    }

    pub fn insert_itemset(&mut self, itemset: &[u32], count: u64) {
        self.nodes[0].count += count;
        if itemset.is_empty() { return; }
        let mut node_idx = 0usize;
        for &item in itemset {
            if let Some(child_idx) = self.nodes[node_idx].find_child(item) {
                self.nodes[child_idx].count += count;
                node_idx = child_idx;
            } else {
                let new_idx = self.nodes.len();
                let new_node = FPNode::new(item, count, node_idx);
                self.nodes.push(new_node);
                self.nodes[node_idx].add_child(item, new_idx);
                self.item_nodes[item as usize].push(new_idx);
                node_idx = new_idx;
            }
        }
    }

    pub fn conditional_tree(&self, item: u32, minsup: u64) -> FPTree {
        let node_indices = &self.item_nodes[item as usize];
        let mut counts = vec![0u64; item as usize];

        let mut branches: Vec<(Vec<u32>, u64)> = Vec::with_capacity(node_indices.len());
        for &ni in node_indices {
            let mut branch = Vec::new();
            let mut idx = self.nodes[ni].parent;
            while self.nodes[idx].item != u32::MAX {
                branch.push(self.nodes[idx].item);
                idx = self.nodes[idx].parent;
            }
            branch.reverse();
            let node_count = self.nodes[ni].count;
            for &i in &branch { counts[i as usize] += node_count; }
            branches.push((branch, node_count));
        }

        let mut valid_items: Vec<(u32, u64)> = counts
            .into_iter()
            .enumerate()
            .filter(|&(_, c)| c >= minsup)
            .map(|(i, c)| (i as u32, c))
            .collect();
        valid_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let mut old_to_new = vec![u32::MAX; item as usize];
        let mut new_original_items = Vec::with_capacity(valid_items.len());
        for (new_id, &(old_id, _)) in valid_items.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
            new_original_items.push(self.original_items[old_id as usize]);
        }

        let mut cond_tree = FPTree::new(valid_items.len(), new_original_items);
        cond_tree.cond_items = self.cond_items.clone();
        cond_tree.cond_items.push(self.original_items[item as usize]);

        for (branch, branch_count) in branches {
            let mut filtered: Vec<u32> = branch
                .into_iter()
                .filter_map(|i| {
                    let new_id = old_to_new[i as usize];
                    if new_id != u32::MAX { Some(new_id) } else { None }
                })
                .collect();
            if filtered.is_empty() { continue; }
            filtered.sort_unstable();
            cond_tree.insert_itemset(&filtered, branch_count);
        }
        cond_tree
    }
}

fn combinations<T: Copy>(items: &[T], size: usize) -> impl Iterator<Item = Vec<T>> + '_ {
    let n = items.len();
    let mut indices: Vec<usize> = (0..size).collect();
    let mut first = size > 0 && n >= size;
    
    std::iter::from_fn(move || {
        if size == 0 { return None; }
        if n < size { return None; }
        if !first {
            let mut i = size as isize - 1;
            while i >= 0 {
                if indices[i as usize] < n - size + i as usize { break; }
                i -= 1;
            }
            if i < 0 { return None; }
            let idx = i as usize;
            indices[idx] += 1;
            for j in (idx + 1)..size { indices[j] = indices[j - 1] + 1; }
        } else {
            first = false;
        }
        Some(indices.iter().map(|&i| items[i]).collect())
    })
}

pub(crate) fn fpg_step(
    tree: &FPTree,
    minsup: u64,
    max_len: Option<usize>,
) -> Vec<(u64, Vec<u32>)> {
    let num_items = tree.original_items.len();
    let cond_len = tree.cond_items.len();
    let mut results: Vec<(u64, Vec<u32>)> = Vec::with_capacity(num_items.saturating_mul(2));

    if tree.is_path() {
        let max_size_from_len = max_len.map_or(num_items + 1, |ml| ml.saturating_sub(cond_len) + 1);
        let size_remain = std::cmp::min(num_items + 1, max_size_from_len);
        for size in 1..size_remain {
            let local_ids: Vec<u32> = (0..num_items as u32).collect();
            for combo in combinations(&local_ids, size) {
                let support = combo
                    .iter()
                    .map(|&local_id| tree.nodes[tree.item_nodes[local_id as usize][0]].count)
                    .min()
                    .unwrap_or(0);
                let mut iset = tree.cond_items.clone();
                iset.extend(combo.iter().map(|&id| tree.original_items[id as usize]));
                results.push((support, iset));
            }
        }
    } else if max_len.map_or(true, |ml| ml > cond_len) {
        let prefix = &tree.cond_items;

        let item_trees: Vec<(u32, u64, FPTree)> = (0..num_items as u32)
            .rev()
            .map(|local_id| {
                let support: u64 = tree.item_nodes[local_id as usize]
                    .iter()
                    .map(|&ni| tree.nodes[ni].count)
                    .sum();
                let cond_tree = tree.conditional_tree(local_id, minsup);
                (local_id, support, cond_tree)
            })
            .collect();

        let sub_results: Vec<Vec<(u64, Vec<u32>)>> = if item_trees.len() >= PAR_ITEMS_CUTOFF {
            item_trees
                .into_par_iter()
                .map(|(local_id, support, cond_tree)| {
                    let mut sub = Vec::new();
                    let mut iset = Vec::with_capacity(prefix.len() + 1);
                    iset.extend_from_slice(prefix);
                    iset.push(tree.original_items[local_id as usize]);
                    sub.push((support, iset));
                    sub.extend(fpg_step(&cond_tree, minsup, max_len));
                    sub
                })
                .collect()
        } else {
            item_trees
                .into_iter()
                .map(|(local_id, support, cond_tree)| {
                    let mut sub = Vec::new();
                    let mut iset = Vec::with_capacity(prefix.len() + 1);
                    iset.extend_from_slice(prefix);
                    iset.push(tree.original_items[local_id as usize]);
                    sub.push((support, iset));
                    sub.extend(fpg_step(&cond_tree, minsup, max_len));
                    sub
                })
                .collect()
        };

        for mut chunk in sub_results { results.append(&mut chunk); }
    }

    results
}

fn process_item_counts(
    item_count: Vec<u64>,
    min_count: u64,
    n_cols: usize,
) -> Option<(Vec<u32>, Vec<u32>, Vec<usize>, usize)> {
    let mut frequent: Vec<(u32, u64)> = item_count
        .iter().enumerate()
        .filter(|(_, &c)| c >= min_count)
        .map(|(col, &c)| (col as u32, c))
        .collect();
    if frequent.is_empty() { return None; }
    frequent.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    let mut global_to_local = vec![u32::MAX; n_cols];
    let mut original_items = Vec::with_capacity(frequent.len());
    let mut frequent_cols = Vec::with_capacity(frequent.len());
    for (local_id, &(col, _)) in frequent.iter().enumerate() {
        global_to_local[col as usize] = local_id as u32;
        original_items.push(col);
        frequent_cols.push(col as usize);
    }
    Some((global_to_local, original_items, frequent_cols, frequent.len()))
}

fn mine_itemsets(
    mut itemsets: Vec<Vec<u32>>,
    frequent_len: usize,
    original_items: Vec<u32>,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    itemsets.par_sort_unstable();
    let mut tree = FPTree::new(frequent_len, original_items);
    let total = itemsets.len();
    let mut i = 0;
    while i < total {
        let basket = &itemsets[i];
        let mut j = i + 1;
        while j < total && itemsets[j] == *basket { j += 1; }
        tree.insert_itemset(basket, (j - i) as u64);
        i = j;
    }
    Ok(fpg_step(&tree, min_count, max_len))
}

fn _mine_dense(
    flat: &[u8],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    let item_count: Vec<u64> = flat
        .par_chunks(n_cols)
        .fold(
            || vec![0u64; n_cols],
            |mut acc, row| {
                for (col, &val) in row.iter().enumerate() {
                    if val != 0 { unsafe { *acc.get_unchecked_mut(col) += 1; } }
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
            None => return Ok(vec![]),
        };

    let itemsets: Vec<Vec<u32>> = flat
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

    mine_itemsets(itemsets, frequent_len, original_items, min_count, max_len)
}

fn _mine_csr(
    indptr: &[i32],
    indices: &[i32],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    let n_rows = indptr.len() - 1;

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
            None => return Ok(vec![]),
        };

    let itemsets: Vec<Vec<u32>> = (0..n_rows)
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

    mine_itemsets(itemsets, frequent_len, original_items, min_count, max_len)
}

#[pyfunction]
#[pyo3(signature = (data, min_count, max_len=None))]
pub fn fpgrowth_from_dense(
    _py: Python<'_>,
    data: PyReadonlyArray2<u8>,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    let arr = data.as_array();
    let n_rows = arr.nrows();
    let n_cols = arr.ncols();
    if n_cols == 0 || n_rows == 0 { return Ok(vec![]); }
    let flat: &[u8] = arr.as_slice().unwrap();
    _mine_dense(flat, n_cols, min_count, max_len)
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_cols, min_count, max_len=None))]
pub fn fpgrowth_from_csr(
    _py: Python<'_>,
    indptr: PyReadonlyArray1<i32>,
    indices: PyReadonlyArray1<i32>,
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    if ip.len() < 2 || n_cols == 0 { return Ok(vec![]); }
    _mine_csr(ip, ix, n_cols, min_count, max_len)
}
