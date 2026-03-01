use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_choice(&mut self, max: usize) -> usize {
        (self.next_u64() % (max as u64)) as usize
    }
}

#[derive(Clone)]
enum Node {
    Split {
        p1: u32,
        p2: u32,
        dist_p1_p2: f32,
        left: u32,
        right: u32,
    },
    Leaf {
        start: u32,
        end: u32,
    },
}

#[pyclass]
pub struct AnnIndex {
    n_features: usize,
    data: Vec<f32>,
    nodes: Vec<Node>,
    indices: Vec<u32>,
    roots: Vec<u32>,
}

fn fast_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum: f32 = 0.0;
    // We can unroll or let compiler optimize
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

fn build_tree_recursive(
    indices: &mut [u32],
    data: &[f32],
    n_features: usize,
    leaf_size: usize,
    rng: &mut XorShift,
    nodes: &mut Vec<Node>,
    leaf_indices: &mut Vec<u32>,
) -> u32 {
    if indices.len() <= leaf_size {
        let start = leaf_indices.len() as u32;
        leaf_indices.extend_from_slice(indices);
        let end = leaf_indices.len() as u32;
        let node_idx = nodes.len() as u32;
        nodes.push(Node::Leaf { start, end });
        return node_idx;
    }

    let p1_local = rng.next_choice(indices.len());
    let mut p2_local = rng.next_choice(indices.len());
    let mut attempts = 0;
    while p1_local == p2_local && attempts < 5 {
        p2_local = rng.next_choice(indices.len());
        attempts += 1;
    }

    let p1 = indices[p1_local];
    let p2 = indices[p2_local];

    let p1_vec = &data[p1 as usize * n_features..(p1 as usize + 1) * n_features];
    let p2_vec = &data[p2 as usize * n_features..(p2 as usize + 1) * n_features];

    let dist_p1_p2_sq = fast_dist_sq(p1_vec, p2_vec);
    if dist_p1_p2_sq == 0.0 {
        let start = leaf_indices.len() as u32;
        leaf_indices.extend_from_slice(indices);
        let end = leaf_indices.len() as u32;
        let node_idx = nodes.len() as u32;
        nodes.push(Node::Leaf { start, end });
        return node_idx;
    }

    let mut flip = false;
    let mut i = 0;
    let mut j = indices.len();
    while i < j {
        let x_idx = indices[i];
        let x_vec = &data[x_idx as usize * n_features..(x_idx as usize + 1) * n_features];
        let d1 = fast_dist_sq(x_vec, p1_vec);
        let d2 = fast_dist_sq(x_vec, p2_vec);

        let go_left = if d1 < d2 {
            true
        } else if d1 > d2 {
            false
        } else {
            flip = !flip;
            flip
        };

        if go_left {
            i += 1;
        } else {
            j -= 1;
            indices.swap(i, j);
        }
    }

    if i == 0 || i == indices.len() {
        i = indices.len() / 2;
    }

    let (left_indices, right_indices) = indices.split_at_mut(i);

    let node_idx = nodes.len() as u32;
    nodes.push(Node::Leaf { start: 0, end: 0 });

    let left_child =
        build_tree_recursive(left_indices, data, n_features, leaf_size, rng, nodes, leaf_indices);
    let right_child =
        build_tree_recursive(right_indices, data, n_features, leaf_size, rng, nodes, leaf_indices);

    nodes[node_idx as usize] = Node::Split {
        p1,
        p2,
        dist_p1_p2: dist_p1_p2_sq.sqrt(),
        left: left_child,
        right: right_child,
    };

    node_idx
}

#[derive(PartialEq)]
struct PrioritizedBranch {
    priority: f32,
    node_idx: u32,
}
impl Eq for PrioritizedBranch {}
impl PartialOrd for PrioritizedBranch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.priority.partial_cmp(&self.priority)
    }
}
impl Ord for PrioritizedBranch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(PartialEq)]
struct TopK {
    dist: f32,
    idx: u32,
}
impl Eq for TopK {}
impl PartialOrd for TopK {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for TopK {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[pymethods]
impl AnnIndex {
    #[new]
    #[pyo3(signature = (data, n_trees=10, leaf_size=30, seed=42))]
    fn new(data: PyReadonlyArray2<f32>, n_trees: usize, leaf_size: usize, seed: u64) -> PyResult<Self> {
        let shape = data.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        let data_slice = data.as_slice().unwrap();
        let mut data_vec = Vec::with_capacity(n_samples * n_features);
        data_vec.extend_from_slice(data_slice);

        let mut index = AnnIndex {
            n_features,
            data: data_vec,
            nodes: Vec::new(),
            indices: Vec::new(),
            roots: Vec::new(),
        };

        let trees: Vec<(Vec<Node>, Vec<u32>, u32)> = (0..n_trees)
            .into_par_iter()
            .map(|i| {
                let mut rng = XorShift::new(seed + i as u64);
                let mut point_indices: Vec<u32> = (0..n_samples as u32).collect();
                let mut local_nodes = Vec::new();
                let mut local_indices = Vec::new();
                let root = build_tree_recursive(
                    &mut point_indices,
                    &index.data,
                    n_features,
                    leaf_size,
                    &mut rng,
                    &mut local_nodes,
                    &mut local_indices,
                );
                (local_nodes, local_indices, root)
            })
            .collect();

        for (local_nodes, local_indices, root) in trees {
            let node_offset = index.nodes.len() as u32;
            let indices_offset = index.indices.len() as u32;

            for mut node in local_nodes {
                match &mut node {
                    Node::Split { left, right, .. } => {
                        *left += node_offset;
                        *right += node_offset;
                    }
                    Node::Leaf { start, end } => {
                        *start += indices_offset;
                        *end += indices_offset;
                    }
                }
                index.nodes.push(node);
            }
            index.indices.extend(local_indices);
            index.roots.push(root + node_offset);
        }

        Ok(index)
    }

    #[pyo3(signature = (queries, n_neighbors, search_k=None))]
    fn kneighbors<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        n_neighbors: usize,
        search_k: Option<usize>,
    ) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray2<f32>>)> {
        let query_shape = queries.shape();
        let n_queries = query_shape[0];
        if query_shape[1] != self.n_features {
            return Err(PyValueError::new_err("Queries features mismatch"));
        }
        let queries_slice = queries.as_slice().unwrap();
        let search_k = search_k.unwrap_or(n_neighbors * self.roots.len() * 2);

        let mut all_indices = vec![0u32; n_queries * n_neighbors];
        let mut all_distances = vec![0.0f32; n_queries * n_neighbors];

        all_indices
            .par_chunks_mut(n_neighbors)
            .zip(all_distances.par_chunks_mut(n_neighbors))
            .enumerate()
            .for_each(|(q_idx, (out_idx, out_dist))| {
                let q = &queries_slice[q_idx * self.n_features..(q_idx + 1) * self.n_features];
                
                let mut heap = BinaryHeap::new();
                for &root in &self.roots {
                    heap.push(PrioritizedBranch {
                        priority: 0.0,
                        node_idx: root,
                    });
                }

                let mut candidate_indices = Vec::new();
                let mut visited_nodes = 0;

                while let Some(branch) = heap.pop() {
                    if visited_nodes >= search_k {
                        break;
                    }

                    let mut curr = branch.node_idx;
                    loop {
                        visited_nodes += 1;
                        match &self.nodes[curr as usize] {
                            Node::Leaf { start, end } => {
                                for i in *start..*end {
                                    candidate_indices.push(self.indices[i as usize]);
                                }
                                break;
                            }
                            Node::Split {
                                p1,
                                p2,
                                dist_p1_p2,
                                left,
                                right,
                            } => {
                                let p1_vec = &self.data[*p1 as usize * self.n_features..(*p1 as usize + 1) * self.n_features];
                                let p2_vec = &self.data[*p2 as usize * self.n_features..(*p2 as usize + 1) * self.n_features];
                                let d1 = fast_dist_sq(q, p1_vec);
                                let d2 = fast_dist_sq(q, p2_vec);

                                let margin = (d2 - d1).abs() / (2.0 * *dist_p1_p2);

                                if d1 < d2 {
                                    heap.push(PrioritizedBranch {
                                        priority: branch.priority.max(margin),
                                        node_idx: *right,
                                    });
                                    curr = *left;
                                } else {
                                    heap.push(PrioritizedBranch {
                                        priority: branch.priority.max(margin),
                                        node_idx: *left,
                                    });
                                    curr = *right;
                                }
                            }
                        }
                    }
                }

                candidate_indices.sort_unstable();
                candidate_indices.dedup();

                let mut topk_heap = BinaryHeap::with_capacity(n_neighbors + 1);
                for &idx in &candidate_indices {
                    let x = &self.data[idx as usize * self.n_features..(idx as usize + 1) * self.n_features];
                    let d = fast_dist_sq(q, x);
                    if topk_heap.len() < n_neighbors {
                        topk_heap.push(TopK { dist: d, idx });
                    } else if d < topk_heap.peek().unwrap().dist {
                        topk_heap.push(TopK { dist: d, idx });
                        topk_heap.pop();
                    }
                }

                let limit = topk_heap.len().min(n_neighbors);
                for i in (0..limit).rev() {
                    let item = topk_heap.pop().unwrap();
                    out_idx[i] = item.idx;
                    out_dist[i] = item.dist.sqrt();
                }
                for i in limit..n_neighbors {
                    out_idx[i] = 0;
                    out_dist[i] = f32::MAX;
                }
            });

        let idx_arr = PyArray1::from_vec(py, all_indices)
            .reshape((n_queries, n_neighbors))
            .unwrap();
        let dist_arr = PyArray1::from_vec(py, all_distances)
            .reshape((n_queries, n_neighbors))
            .unwrap();
        Ok((
            idx_arr.into_pyobject(py).unwrap().unbind(),
            dist_arr.into_pyobject(py).unwrap().unbind(),
        ))
    }
}
