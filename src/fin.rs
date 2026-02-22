use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::fpgrowth::{flatten_results, process_item_counts};

// FIN algorithm
// 1. Build an FP-Tree (PPC-Tree)
// 2. Do a DFS to assign `pre` and `post` traversal orders to each node.
// 3. Extract NodeSets for each 1-itemset. A node is (pre, post, count).
// 4. Recursive intersection: N_AB = { b \in N_B | \exists a \in N_A, a.pre <= b.pre && b.post <= a.post }
// Because nodes in a Nodeset are sorted by `pre` natively (from tree traversal or we can sort them),
// the intersection can be done linearly.

#[derive(Debug, Clone, Copy)]
pub(crate) struct PPNode {
    pub item: u32,
    pub count: u64,
    pub parent: u32,
    pub children_start: u32,
    pub children_end: u32,
    pub pre: u32,
    pub post: u32,
}

impl PPNode {
    #[inline(always)]
    fn new(item: u32, count: u64, parent: u32) -> Self {
        PPNode {
            item,
            count,
            parent,
            children_start: 0,
            children_end: 0,
            pre: 0,
            post: 0,
        }
    }
}

pub(crate) struct PPCTree {
    pub nodes: Vec<PPNode>,
    children_arena: Vec<(u32, u32)>,
    pub item_nodes: Vec<Vec<u32>>,
    pub original_items: Vec<u32>,
    pub single_path: bool,
}

impl PPCTree {
    pub fn new(num_items: usize, original_items: Vec<u32>) -> Self {
        let root = PPNode::new(u32::MAX, 0, 0);
        let mut nodes = Vec::with_capacity(256);
        nodes.push(root);
        PPCTree {
            nodes,
            children_arena: Vec::with_capacity(256),
            item_nodes: vec![Vec::new(); num_items],
            original_items,
            single_path: true,
        }
    }

    #[inline]
    fn find_child(&self, node_idx: u32, item: u32) -> Option<u32> {
        let node = &self.nodes[node_idx as usize];
        let start = node.children_start as usize;
        let end = node.children_end as usize;
        for i in start..end {
            let (k, v) = self.children_arena[i];
            if k == item {
                return Some(v);
            }
        }
        None
    }

    #[inline]
    fn add_child(&mut self, parent_idx: u32, item: u32, child_idx: u32) {
        let parent = &self.nodes[parent_idx as usize];
        let n_children = parent.children_end - parent.children_start;

        if n_children == 0 {
            let pos = self.children_arena.len() as u32;
            self.children_arena.push((item, child_idx));
            let parent = &mut self.nodes[parent_idx as usize];
            parent.children_start = pos;
            parent.children_end = pos + 1;
        } else if parent.children_end as usize == self.children_arena.len() {
            self.children_arena.push((item, child_idx));
            self.nodes[parent_idx as usize].children_end += 1;
            if n_children >= 1 {
                self.single_path = false;
            }
        } else {
            let old_start = parent.children_start as usize;
            let old_end = parent.children_end as usize;
            let new_start = self.children_arena.len() as u32;
            for i in old_start..old_end {
                self.children_arena.push(self.children_arena[i]);
            }
            self.children_arena.push((item, child_idx));
            let parent = &mut self.nodes[parent_idx as usize];
            parent.children_start = new_start;
            parent.children_end = new_start + (old_end - old_start) as u32 + 1;
            self.single_path = false;
        }
    }

    pub fn insert_itemset(&mut self, itemset: &[u32], count: u64) {
        self.nodes[0].count += count;
        if itemset.is_empty() {
            return;
        }
        let mut node_idx = 0u32;
        for &item in itemset {
            if let Some(child_idx) = self.find_child(node_idx, item) {
                self.nodes[child_idx as usize].count += count;
                node_idx = child_idx;
            } else {
                let new_idx = self.nodes.len() as u32;
                let new_node = PPNode::new(item, count, node_idx);
                self.nodes.push(new_node);
                self.add_child(node_idx, item, new_idx);
                self.item_nodes[item as usize].push(new_idx);
                node_idx = new_idx;
            }
        }
    }

    pub fn compute_pre_post(&mut self) {
        let mut pre_counter = 0u32;
        let mut post_counter = 0u32;
        self.dfs_pre_post(0, &mut pre_counter, &mut post_counter);
    }

    fn dfs_pre_post(&mut self, current: u32, pre_counter: &mut u32, post_counter: &mut u32) {
        self.nodes[current as usize].pre = *pre_counter;
        *pre_counter += 1;

        let start = self.nodes[current as usize].children_start as usize;
        let end = self.nodes[current as usize].children_end as usize;
        let children: Vec<u32> = self.children_arena[start..end].iter().map(|&(_, v)| v).collect();

        for child in children {
            self.dfs_pre_post(child, pre_counter, post_counter);
        }

        self.nodes[current as usize].post = *post_counter;
        *post_counter += 1;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NodesetNode {
    pub pre: u32,
    pub post: u32,
    pub count: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Nodeset {
    pub nodes: Vec<NodesetNode>,
    pub support: u64,
}

impl Nodeset {
    // Intersect nodeset A (prefix) with nodeset B (extension candidate)
    // N_AB = { b \in B | \exists a \in A, a.pre <= b.pre && b.post <= a.post }
    // Since both A and B are ordered by `pre` (by virtue of the item_nodes creation order in DFS), we can do it efficiently.
    pub fn intersect(&self, other: &Nodeset) -> Option<Nodeset> {
        let mut result_nodes = Vec::with_capacity(std::cmp::min(self.nodes.len(), other.nodes.len()));
        let mut support = 0;

        let mut i = 0;
        let mut j = 0;
        let p_len = self.nodes.len();
        let q_len = other.nodes.len();

        let p_nodes = &self.nodes;
        let q_nodes = &other.nodes;

        while i < p_len && j < q_len {
            let a = &p_nodes[i];
            let b = &q_nodes[j];

            if a.pre <= b.pre {
                if b.post <= a.post {
                    // b is a descendant of a
                    result_nodes.push(b.clone());
                    support += b.count;
                    j += 1;
                } else {
                    // a.pre <= b.pre but b is entirely after a in post-order => a does not contain b.
                    // This means a and b are disjoint subtrees, and since b.pre >= a.pre, a must be finished.
                    i += 1;
                }
            } else {
                // b.pre < a.pre => b cannot be a descendant of a.
                // It means b is in a different branch before a.
                j += 1;
            }
        }

        if support > 0 {
            Some(Nodeset {
                nodes: result_nodes,
                support,
            })
        } else {
            None
        }
    }
}

pub(crate) fn fin_mine(
    prefix: &[u32],
    prefix_ns: &Nodeset,
    active_items: &[(u32, Nodeset)],
    min_count: u64,
    max_len: Option<usize>,
) -> Vec<(u64, Vec<u32>)> {
    let mut results = Vec::new();
    let new_len = prefix.len() + 1;

    for (i, (item_b, ns_b)) in active_items.iter().enumerate() {
        if let Some(ns_ab) = prefix_ns.intersect(ns_b) {
            if ns_ab.support >= min_count {
                if max_len.is_none_or(|ml| new_len <= ml) {
                    let mut iset = Vec::with_capacity(new_len);
                    iset.extend_from_slice(prefix);
                    iset.push(*item_b);
                    results.push((ns_ab.support, iset.clone()));

                    if max_len.is_none_or(|ml| new_len < ml) {
                        let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
                        for (item_c, ns_c) in &active_items[i + 1..] {
                            next_active.push((*item_c, ns_c.clone()));
                        }
                        if !next_active.is_empty() {
                            results.extend(fin_mine(&iset, &ns_ab, &next_active, min_count, max_len));
                        }
                    }
                }
            }
        }
    }

    results
}

fn build_initial_nodesets(tree: &PPCTree) -> Vec<(u32, Nodeset)> {
    let mut active = Vec::with_capacity(tree.item_nodes.len());
    // In FIN, the intersection works when candidate items are processed in a specific order.
    // FP-Growth uses reverse support order. We can provide active items in reverse support.
    
    for (local_id, node_indices) in tree.item_nodes.iter().enumerate() {
        let mut ns_nodes = Vec::with_capacity(node_indices.len());
        let mut support = 0;
        for &ni in node_indices {
            let node = &tree.nodes[ni as usize];
            ns_nodes.push(NodesetNode {
                pre: node.pre,
                post: node.post,
                count: node.count,
            });
            support += node.count;
        }
        // Natively, `item_nodes` lists node indices in order of insertion.
        // It's important for the intersect algorithm that `ns_nodes` are sorted by `pre`.
        ns_nodes.sort_unstable_by_key(|n| n.pre);

        active.push((local_id as u32, Nodeset {
            nodes: ns_nodes,
            support,
        }));
    }
    active
}

fn mine_itemsets_fin(
    itemsets: Vec<Vec<u32>>,
    frequent_len: usize,
    original_items: Vec<u32>,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<Vec<(u64, Vec<u32>)>> {
    use ahash::AHashMap;
    let mut basket_counts: AHashMap<Vec<u32>, u64> = AHashMap::with_capacity(itemsets.len());
    for basket in itemsets {
        *basket_counts.entry(basket).or_insert(0) += 1;
    }
    
    let mut tree = PPCTree::new(frequent_len, original_items.clone());
    for (basket, count) in basket_counts {
        tree.insert_itemset(&basket, count);
    }
    tree.compute_pre_post();

    let all_nodesets = build_initial_nodesets(&tree);

    let mut results = Vec::new();

    // Start FIN core
    let root_ns = Nodeset {
        nodes: vec![NodesetNode {
            pre: tree.nodes[0].pre,
            post: tree.nodes[0].post,
            count: tree.nodes[0].count,
        }],
        support: tree.nodes[0].count,
    };

    let sub_results: Vec<Vec<(u64, Vec<u32>)>> = all_nodesets
        .par_iter()
        .enumerate()
        .map(|(i, (local_id, ns_a))| {
            let mut sub = Vec::new();
            if ns_a.support >= min_count {
                let iset = vec![original_items[*local_id as usize]];
                sub.push((ns_a.support, iset.clone()));

                if max_len.is_none_or(|ml| ml > 1) {
                    let mut next_active = Vec::with_capacity(all_nodesets.len() - i - 1);
                    for (local_id_b, ns_b) in &all_nodesets[i + 1..] {
                        next_active.push((original_items[*local_id_b as usize], ns_b.clone()));
                    }
                    if !next_active.is_empty() {
                        sub.extend(fin_mine(&iset, ns_a, &next_active, min_count, max_len));
                    }
                }
            }
            sub
        })
        .collect();

    for mut chunk in sub_results {
        results.append(&mut chunk);
    }

    Ok(results)
}

fn _mine_dense_fin(
    flat: &[u8],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Vec<u64>, Vec<u32>, Vec<u32>)> {
    let item_count: Vec<u64> = flat
        .par_chunks(n_cols)
        .fold(
            || vec![0u64; n_cols],
            |mut acc, row| {
                for (col, &val) in row.iter().enumerate() {
                    if val != 0 {
                        unsafe {
                            *acc.get_unchecked_mut(col) += 1;
                        }
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

    let (_global_to_local, original_items, frequent_cols, frequent_len) =
        match process_item_counts(item_count, min_count, n_cols) {
            Some(v) => v,
            None => return Ok((vec![], vec![], vec![])),
        };

    let itemsets: Vec<Vec<u32>> = flat
        .par_chunks(n_cols)
        .filter_map(|row| {
            let items: Vec<u32> = frequent_cols
                .iter()
                .enumerate()
                .filter(|&(_, &col)| {
                    unsafe { *row.get_unchecked(col) != 0 }
                })
                .map(|(local_id, _)| local_id as u32)
                .collect();
            if items.is_empty() {
                return None;
            }
            Some(items)
        })
        .collect();

    let results = mine_itemsets_fin(itemsets, frequent_len, original_items, min_count, max_len)?;
    Ok(flatten_results(results))
}


pub(crate) fn _mine_csr_fin(
    indptr: &[i32],
    indices: &[i32],
    n_cols: usize,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(Vec<u64>, Vec<u32>, Vec<u32>)> {
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
                    if c < n_cols {
                        unsafe {
                            *acc.get_unchecked_mut(c) += 1;
                        }
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

    let (global_to_local, original_items, _, frequent_len) =
        match process_item_counts(item_count, min_count, n_cols) {
            Some(v) => v,
            None => return Ok((vec![], vec![], vec![])),
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
                        if l != u32::MAX {
                            Some(l)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if items.is_empty() {
                return None;
            }
            items.sort_unstable();
            Some(items)
        })
        .collect();

    let results = mine_itemsets_fin(itemsets, frequent_len, original_items, min_count, max_len)?;
    Ok(flatten_results(results))
}

#[pyfunction]
#[pyo3(signature = (data, min_count, max_len=None))]
pub fn fin_from_dense<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<u8>,
    min_count: u64,
    max_len: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let arr = data.as_array();
    let n_rows = arr.nrows();
    let n_cols = arr.ncols();

    if n_cols == 0 || n_rows == 0 {
        return Ok((
            Vec::<u64>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
        ));
    }

    let flat: &[u8] = arr.as_slice().unwrap();
    let (supports, offsets, items) = _mine_dense_fin(flat, n_cols, min_count, max_len)?;

    Ok((
        supports.into_pyarray(py),
        offsets.into_pyarray(py),
        items.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_cols, min_count, max_len=None))]
pub fn fin_from_csr<'py>(
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
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;

    if ip.len() < 2 || n_cols == 0 {
        return Ok((
            Vec::<u64>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
            Vec::<u32>::new().into_pyarray(py),
        ));
    }

    let (supports, offsets, items) = _mine_csr_fin(ip, ix, n_cols, min_count, max_len)?;

    Ok((
        supports.into_pyarray(py),
        offsets.into_pyarray(py),
        items.into_pyarray(py),
    ))
}
