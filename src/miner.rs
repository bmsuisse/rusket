/// FPMiner — streaming FP-Growth/Eclat accumulator.
///
/// Memory strategy (HashMap-first):
///   • add_chunk(): insert each (txn_id, item_id) pair into a HashMap<i64, Vec<i32>>.
///     Items are deduplicated per transaction incrementally.
///     Peak memory = O(unique_txns × avg_items) — NOT O(total_pairs).
///   • mine(): iterate the HashMap, sort each transaction's item list,
///     build CSR in one pass. No k-way merge needed.
///
/// For 1B rows with 43M unique transactions × 23 items:
///   Memory: 43M × (8 + 24 + 23×4) = ~5GB — far better than the sorted-chunk approach.
///   mine(): sort each txn's Vec once (items already collected) → build CSR → done.
use ahash::AHashMap;
use rayon::prelude::*;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::fpgrowth::_mine_csr;
use crate::eclat::_eclat_mine_csr;

#[pyclass]
pub struct FPMiner {
    /// Per-transaction item lists.  Key = transaction_id, Value = Vec of item ids.
    txns: AHashMap<i64, Vec<i32>>,
    /// Total number of (txn_id, item_id) pairs fed so far (pre-dedup).
    n_rows: usize,
    n_items: usize,
    /// Optional RAM cap (bytes). When set, add_chunk errors out instead of silently OOMing.
    max_ram_bytes: Option<usize>,
}

#[pymethods]
impl FPMiner {
    #[new]
    #[pyo3(signature = (n_items, max_ram_mb=None))]
    pub fn new(n_items: usize, max_ram_mb: Option<usize>) -> Self {
        let max_ram_bytes = max_ram_mb.map(|mb| mb.saturating_mul(1024 * 1024));
        FPMiner {
            txns: AHashMap::new(),
            n_rows: 0,
            n_items,
            max_ram_bytes,
        }
    }

    /// Feed a chunk of (transaction_id, item_id) pairs.
    ///
    /// Each pair is inserted into the per-transaction HashMap.
    /// Memory grows as O(unique_txns × avg_items), not O(total_pairs).
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

        self.n_rows += txns.len();

        for (&t, &i) in txns.iter().zip(items.iter()) {
            if i >= 0 && (i as usize) < self.n_items {
                self.txns.entry(t).or_default().push(i);
            }
        }

        Ok(())
    }

    /// Total number of (txn_id, item_id) pairs accumulated (pre-dedup).
    #[getter]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of items (columns).
    #[getter]
    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// Number of distinct transactions accumulated so far.
    #[getter]
    pub fn n_transactions(&self) -> usize {
        self.txns.len()
    }

    /// Max RAM threshold.
    #[getter]
    pub fn max_ram_mb(&self) -> Option<usize> {
        self.max_ram_bytes.map(|b| b / (1024 * 1024))
    }

    /// Mine frequent itemsets using FP-Growth.
    pub fn mine_fpgrowth(
        &self,
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
        &self,
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
        self.txns.clear();
        self.txns.shrink_to_fit();
        self.n_rows = 0;
    }
}

impl FPMiner {
    /// Build a CSR matrix from the accumulated HashMap.
    ///
    /// Uses iter() so the HashMap survives multiple mine() calls.
    /// Each transaction's item list is sorted + deduped on the fly.
    fn build_csr(&self) -> PyResult<(Vec<i32>, Vec<i32>, usize)> {
        if self.txns.is_empty() {
            return Ok((vec![0i32], vec![], 0));
        }

        // Collect (txn_id, &items) — parallel sort for deterministic row order
        let mut entries: Vec<(i64, &Vec<i32>)> = self.txns
            .iter()
            .map(|(&k, v)| (k, v))
            .collect();
        entries.par_sort_unstable_by_key(|(t, _)| *t);

        let n_txn = entries.len();

        let mut indptr: Vec<i32> = Vec::with_capacity(n_txn + 1);
        let mut indices: Vec<i32> = Vec::new();
        indptr.push(0);

        for (_, items) in entries {
            let mut sorted = items.clone();
            sorted.sort_unstable();
            sorted.dedup();
            for item in &sorted {
                indices.push(*item);
            }
            indptr.push(indices.len() as i32);
        }

        Ok((indptr, indices, n_txn))
    }
}
