use ahash::AHashMap;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;

// ─── Parallelism cutoff ───────────────────────────────────────────────────────
// Below this many items, use serial recursion to avoid Rayon spawn overhead.
const PAR_ITEMS_CUTOFF: usize = 4;

// ─── FP-Tree node ─────────────────────────────────────────────────────────────
// SmallVec<[(u32, usize); 4]>: 4 inline child slots covers most nodes without
// heap allocation; spills to heap only for high-degree nodes.
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

// ─── FP-Tree ──────────────────────────────────────────────────────────────────
pub(crate) struct FPTree {
    pub nodes: Vec<FPNode>,
    pub item_nodes: AHashMap<u32, Vec<usize>>,
    pub rank: AHashMap<u32, usize>,
    pub cond_items: Vec<u32>,
}

impl FPTree {
    pub fn new(rank: AHashMap<u32, usize>) -> Self {
        let root = FPNode::new(u32::MAX, 0, 0);
        let n = rank.len();
        let mut nodes = Vec::with_capacity(n.saturating_mul(4) + 1);
        nodes.push(root);
        FPTree { nodes, item_nodes: AHashMap::default(), rank, cond_items: Vec::new() }
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
                self.item_nodes.entry(item).or_default().push(new_idx);
                node_idx = new_idx;
            }
        }
    }

    pub fn itempath_from_root(&self, node_idx: usize) -> Vec<u32> {
        let mut path = Vec::new();
        let mut idx = self.nodes[node_idx].parent;
        while self.nodes[idx].item != u32::MAX {
            path.push(self.nodes[idx].item);
            idx = self.nodes[idx].parent;
        }
        path.reverse();
        path
    }

    pub fn conditional_tree(&self, item: u32, minsup: u64) -> FPTree {
        let node_indices = &self.item_nodes[&item];
        let mut count: AHashMap<u32, u64> = AHashMap::default();

        let mut branches: Vec<(Vec<u32>, u64)> = Vec::with_capacity(node_indices.len());
        for &ni in node_indices {
            let branch = self.itempath_from_root(ni);
            let node_count = self.nodes[ni].count;
            for &i in &branch { *count.entry(i).or_insert(0) += node_count; }
            branches.push((branch, node_count));
        }

        let mut new_items: Vec<(u32, u64)> = count
            .iter()
            .filter(|(_, &c)| c >= minsup)
            .map(|(&i, &c)| (i, c))
            .collect();
        new_items.sort_unstable_by_key(|&(_, c)| c);

        let new_rank: AHashMap<u32, usize> = new_items
            .iter()
            .enumerate()
            .map(|(r, &(i, _))| (i, r))
            .collect();

        let mut cond_tree = FPTree::new(new_rank.clone());
        for (branch, branch_count) in branches {
            let mut filtered: Vec<u32> = branch
                .into_iter()
                .filter(|i| new_rank.contains_key(i))
                .collect();
            if filtered.is_empty() { continue; }
            filtered.sort_unstable_by(|a, b| new_rank[b].cmp(&new_rank[a]));
            cond_tree.insert_itemset(&filtered, branch_count);
        }
        cond_tree.cond_items = self.cond_items.clone();
        cond_tree.cond_items.push(item);
        cond_tree
    }
}

// ─── Combinations ─────────────────────────────────────────────────────────────
fn combinations(items: &[u32], size: usize) -> Vec<Vec<u32>> {
    if size == 0 { return vec![vec![]]; }
    if items.len() < size { return vec![]; }
    let n = items.len();
    let mut result = Vec::new();
    let mut indices: Vec<usize> = (0..size).collect();
    loop {
        result.push(indices.iter().map(|&i| items[i]).collect());
        let mut i = size as isize - 1;
        while i >= 0 {
            if indices[i as usize] < n - (size - i as usize) { break; }
            i -= 1;
        }
        if i < 0 { break; }
        let i = i as usize;
        indices[i] += 1;
        for j in (i + 1)..size { indices[j] = indices[j - 1] + 1; }
    }
    result
}

// ─── Core FP-Growth recursive step ───────────────────────────────────────────
pub(crate) fn fpg_step(
    tree: &FPTree,
    minsup: u64,
    max_len: Option<usize>,
) -> Vec<(u64, Vec<u32>)> {
    let items: Vec<u32> = tree.item_nodes.keys().copied().collect();
    let cond_len = tree.cond_items.len();
    let mut results: Vec<(u64, Vec<u32>)> = Vec::with_capacity(items.len().saturating_mul(2));

    if tree.is_path() {
        let size_remain = max_len.map_or(items.len() + 1, |ml| ml.saturating_sub(cond_len) + 1);
        for size in 1..size_remain {
            for combo in combinations(&items, size) {
                let support = combo
                    .iter()
                    .map(|&item| tree.nodes[tree.item_nodes[&item][0]].count)
                    .min()
                    .unwrap_or(0);
                let mut iset = tree.cond_items.clone();
                iset.extend_from_slice(&combo);
                results.push((support, iset));
            }
        }
    } else if max_len.map_or(true, |ml| ml > cond_len) {
        let prefix = &tree.cond_items;

        let item_trees: Vec<(u32, u64, FPTree)> = items
            .iter()
            .map(|&item| {
                let support: u64 = tree.item_nodes[&item]
                    .iter()
                    .map(|&ni| tree.nodes[ni].count)
                    .sum();
                let cond_tree = tree.conditional_tree(item, minsup);
                (item, support, cond_tree)
            })
            .collect();

        let sub_results: Vec<Vec<(u64, Vec<u32>)>> = if item_trees.len() >= PAR_ITEMS_CUTOFF {
            item_trees
                .into_par_iter()
                .map(|(item, support, cond_tree)| {
                    let mut sub = Vec::new();
                    let mut iset = Vec::with_capacity(prefix.len() + 1);
                    iset.extend_from_slice(prefix);
                    iset.push(item);
                    sub.push((support, iset));
                    sub.extend(fpg_step(&cond_tree, minsup, max_len));
                    sub
                })
                .collect()
        } else {
            item_trees
                .into_iter()
                .map(|(item, support, cond_tree)| {
                    let mut sub = Vec::new();
                    let mut iset = Vec::with_capacity(prefix.len() + 1);
                    iset.extend_from_slice(prefix);
                    iset.push(item);
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

// ─── Dense mining ─────────────────────────────────────────────────────────────
fn _mine_dense(
    flat: &[u8],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    // ── 1. Parallel frequency count ───────────────────────────────────────────
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

    // ── 2. Rank structures ───────────────────────────────────────────────────
    let mut frequent: Vec<(u32, u64)> = item_count
        .iter().enumerate()
        .filter(|(_, &c)| c >= min_count)
        .map(|(col, &c)| (col as u32, c))
        .collect();
    if frequent.is_empty() { return Ok(vec![]); }
    frequent.sort_unstable_by_key(|&(_, c)| c);

    let mut rank_arr = vec![u32::MAX; n_cols];
    for (rank, &(col, _)) in frequent.iter().enumerate() { rank_arr[col as usize] = rank as u32; }
    let frequent_cols: Vec<usize> = frequent.iter().map(|&(col, _)| col as usize).collect();

    let rank_map: AHashMap<u32, usize> = frequent.iter()
        .enumerate()
        .map(|(r, &(col, _))| (col, r))
        .collect();

    // ── 3. Parallel row extraction ────────────────────────────────────────────
    let mut itemsets: Vec<Vec<u32>> = flat
        .par_chunks(n_cols)
        .filter_map(|row| {
            let mut items: Vec<u32> = frequent_cols
                .iter()
                .filter(|&&col| unsafe { *row.get_unchecked(col) != 0 })
                .map(|&col| col as u32)
                .collect();
            if items.is_empty() { return None; }
            items.sort_unstable_by(|a, b| rank_map[b].cmp(&rank_map[a]));
            Some(items)
        })
        .collect();

    // ── 4. Deduplication ──────────────────────────────────────────────────────
    itemsets.par_sort_unstable();

    // ── 5. Build FP-Tree ──────────────────────────────────────────────────────
    let mut tree = FPTree::new(rank_map);
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

// ─── CSR mining ───────────────────────────────────────────────────────────────
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

    let mut frequent: Vec<(u32, u64)> = item_count
        .iter().enumerate()
        .filter(|(_, &c)| c >= min_count)
        .map(|(col, &c)| (col as u32, c))
        .collect();
    if frequent.is_empty() { return Ok(vec![]); }
    frequent.sort_unstable_by_key(|&(_, c)| c);

    let mut rank_arr = vec![u32::MAX; n_cols];
    for (rank, &(col, _)) in frequent.iter().enumerate() { rank_arr[col as usize] = rank as u32; }

    let rank_map: AHashMap<u32, usize> = frequent.iter()
        .enumerate()
        .map(|(r, &(col, _))| (col, r))
        .collect();

    let mut itemsets: Vec<Vec<u32>> = (0..n_rows)
        .into_par_iter()
        .filter_map(|row| {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let mut items: Vec<u32> = indices[start..end]
                .iter()
                .filter_map(|&col| {
                    let r = rank_arr.get(col as usize).copied().unwrap_or(u32::MAX);
                    (r != u32::MAX).then_some(col as u32)
                })
                .collect();
            if items.is_empty() { return None; }
            items.sort_unstable_by(|a, b| rank_map[b].cmp(&rank_map[a]));
            Some(items)
        })
        .collect();

    itemsets.par_sort_unstable();

    let mut tree = FPTree::new(rank_map);
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

// ─── PyO3 exports ─────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (rows, rank, min_count, max_len=None))]
pub fn fpgrowth_inner(
    _py: Python<'_>,
    rows: Vec<Vec<u32>>,
    rank: Vec<(u32, usize)>,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    let rank_map: AHashMap<u32, usize> = rank.into_iter().collect();
    let mut tree = FPTree::new(rank_map.clone());
    for row in &rows {
        let mut itemset: Vec<u32> = row
            .iter()
            .filter(|i| rank_map.contains_key(i))
            .copied()
            .collect();
        itemset.sort_unstable_by(|a, b| rank_map[b].cmp(&rank_map[a]));
        tree.insert_itemset(&itemset, 1);
    }
    Ok(fpg_step(&tree, min_count, max_len))
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
    let flat: &[u8] = arr.as_slice()
        .expect("array must be C-contiguous; call np.ascontiguousarray first");
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
