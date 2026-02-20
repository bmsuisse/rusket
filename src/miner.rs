/// FPMiner — streaming FP-Growth/Eclat accumulator.
///
/// Memory strategy:
///   • add_chunk(): sort each chunk in-place immediately (small, cheap).
///     Chunks are stored as sorted Vec<(i64, i32)>.
///   • mine(): k-way merge across all sorted chunks → CSR built on the fly.
///     Peak extra memory at mine() time = just the CSR output (small) +
///     a heap of k cursors (one per chunk).
///
/// For 1B rows in 100 × 10M chunks:
///   add_chunk: 10M × 16B = 160 MB per chunk, sorted in ~1s
///   mine():    merge-iterate, no extra 1B-row allocation → safe
use std::collections::BinaryHeap;
use std::cmp::Reverse;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::fpgrowth::_mine_csr;
use crate::eclat::_eclat_mine_csr;

#[pyclass]
pub struct FPMiner {
    /// Each element is one pre-sorted chunk of (txn_id, item_id) pairs.
    chunks: Vec<Vec<(i64, i32)>>,
    /// Total number of pairs across all chunks.
    n_rows: usize,
    n_items: usize,
}

#[pymethods]
impl FPMiner {
    #[new]
    pub fn new(n_items: usize) -> Self {
        FPMiner { chunks: Vec::new(), n_rows: 0, n_items }
    }

    /// Feed a chunk of (transaction_id, item_id) pairs.
    ///
    /// The chunk is sorted in-place immediately — O(k log k) where k = chunk size.
    /// Then stored as a sorted Vec.  Peak extra memory = one chunk.
    pub fn add_chunk(
        &mut self,
        txn_ids: PyReadonlyArray1<i64>,
        item_ids: PyReadonlyArray1<i32>,
    ) -> PyResult<()> {
        let txns  = txn_ids.as_slice()?;
        let items = item_ids.as_slice()?;
        if txns.len() != items.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "txn_ids and item_ids must have the same length",
            ));
        }
        let mut chunk: Vec<(i64, i32)> = txns.iter().zip(items.iter())
            .map(|(&t, &i)| (t, i))
            .collect();
        // Sort within the chunk — cheap, small allocation
        chunk.sort_unstable();
        self.n_rows += chunk.len();
        self.chunks.push(chunk);
        Ok(())
    }

    /// Total number of (txn_id, item_id) pairs accumulated.
    #[getter]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of items (columns).
    #[getter]
    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// Number of distinct transactions (0 until mine() called).
    #[getter]
    pub fn n_transactions(&self) -> usize {
        0
    }

    /// Mine frequent itemsets using FP-Growth.
    pub fn mine_fpgrowth(
        &mut self,
        min_support: f64,
        max_len: Option<usize>,
    ) -> PyResult<(usize, Vec<u64>, Vec<u32>, Vec<u32>)> {
        let (indptr, indices, n_txn) = self.build_csr()?;
        if n_txn == 0 {
            return Ok((0, vec![], vec![], vec![]));
        }
        let min_count = ((min_support * n_txn as f64).ceil() as u64).max(1);
        let (s, o, i) = _mine_csr(&indptr, &indices, self.n_items, min_count, max_len)?;
        Ok((n_txn, s, o, i))
    }

    /// Mine frequent itemsets using Eclat.
    pub fn mine_eclat(
        &mut self,
        min_support: f64,
        max_len: Option<usize>,
    ) -> PyResult<(usize, Vec<u64>, Vec<u32>, Vec<u32>)> {
        let (indptr, indices, n_txn) = self.build_csr()?;
        if n_txn == 0 {
            return Ok((0, vec![], vec![], vec![]));
        }
        let min_count = ((min_support * n_txn as f64).ceil() as u64).max(1);
        let (s, o, i) = _eclat_mine_csr(&indptr, &indices, self.n_items, min_count, max_len)?;
        Ok((n_txn, s, o, i))
    }

    /// Free all accumulated data.
    pub fn reset(&mut self) {
        self.chunks.clear();
        self.chunks.shrink_to_fit();
        self.n_rows = 0;
    }
}

impl FPMiner {
    /// K-way merge across all sorted chunks → build CSR in one streaming pass.
    ///
    /// Uses a min-heap (BinaryHeap<Reverse<...>>) with one cursor per chunk.
    /// Peak extra memory = heap of k entries + output indptr/indices.
    /// The chunks themselves are consumed (dropped) as exhausted.
    fn build_csr(&mut self) -> PyResult<(Vec<i32>, Vec<i32>, usize)> {
        if self.n_rows == 0 {
            return Ok((vec![0i32], vec![], 0));
        }
        let n_items = self.n_items;

        // Heap entry: Reverse((txn, item, chunk_idx, row_within_chunk))
        // min-heap → smallest (txn, item) pops first
        let mut heap: BinaryHeap<Reverse<(i64, i32, usize, usize)>> = BinaryHeap::new();

        // Seed with the first element of each chunk
        for (ci, chunk) in self.chunks.iter().enumerate() {
            if !chunk.is_empty() {
                let (t, i) = chunk[0];
                heap.push(Reverse((t, i, ci, 0)));
            }
        }

        let mut indptr: Vec<i32> = Vec::with_capacity(self.n_rows / 8 + 2);
        let mut indices: Vec<i32> = Vec::with_capacity(self.n_rows);
        indptr.push(0);

        let mut prev_txn  = i64::MIN;
        let mut prev_item = i32::MIN;

        while let Some(Reverse((txn, item, ci, ri))) = heap.pop() {
            // Advance cursor in this chunk
            let next_ri = ri + 1;
            if next_ri < self.chunks[ci].len() {
                let (t, i) = self.chunks[ci][next_ri];
                heap.push(Reverse((t, i, ci, next_ri)));
            }

            // Skip out-of-range items
            if item < 0 || (item as usize) >= n_items {
                continue;
            }

            // New transaction?
            if txn != prev_txn {
                if prev_txn != i64::MIN {
                    indptr.push(indices.len() as i32);
                }
                prev_txn  = txn;
                prev_item = i32::MIN;
            }

            // Deduplicate within transaction
            if item != prev_item {
                indices.push(item);
                prev_item = item;
            }
        }
        if prev_txn != i64::MIN {
            indptr.push(indices.len() as i32);
        }

        let n_txn = indptr.len() - 1;
        Ok((indptr, indices, n_txn))
    }
}
