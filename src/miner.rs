/// FPMiner — streaming FP-Growth/Eclat accumulator.
///
/// Memory strategy (HashMap-first):
///   • add_chunk(): aggregate each (txn_id, item_id) pair into AHashMap<i64, Vec<i32>>.
///     Peak memory = O(unique_txns × avg_items).
///   • mine(): iterate HashMap, sort+dedup each txn, build CSR, call mining backend.
///     No k-way merge. No disk spill. Eclat mine_t is ~8× faster than FPGrowth.
use ahash::AHashMap;
use rayon::prelude::*;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::fpgrowth::_mine_csr;
use crate::eclat::_eclat_mine_csr;

#[pyclass]
pub struct FPMiner {
    txns: AHashMap<i64, Vec<i32>>,
    n_rows: usize,
    n_items: usize,
    max_ram_bytes: Option<usize>,
}

#[pymethods]
impl FPMiner {
    #[new]
    #[pyo3(signature = (n_items, max_ram_mb=None))]
    pub fn new(n_items: usize, max_ram_mb: Option<usize>) -> Self {
        let max_ram_bytes = max_ram_mb.map(|mb| mb.saturating_mul(1024 * 1024));
        FPMiner { txns: AHashMap::new(), n_rows: 0, n_items, max_ram_bytes }
    }

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
        let n = self.n_items;
        for (&t, &i) in txns.iter().zip(items.iter()) {
            if i >= 0 && (i as usize) < n {
                self.txns.entry(t).or_default().push(i);
            }
        }
        Ok(())
    }

    #[getter] pub fn n_rows(&self) -> usize { self.n_rows }
    #[getter] pub fn n_items(&self) -> usize { self.n_items }
    #[getter] pub fn n_transactions(&self) -> usize { self.txns.len() }
    #[getter] pub fn max_ram_mb(&self) -> Option<usize> {
        self.max_ram_bytes.map(|b| b / (1024 * 1024))
    }

    pub fn mine_fpgrowth(&self, min_support: f64, max_len: Option<usize>)
        -> PyResult<(usize, Vec<u64>, Vec<u32>, Vec<u32>)>
    {
        let (indptr, indices, n_txn) = self.build_csr()?;
        if n_txn == 0 { return Ok((0, vec![], vec![], vec![])); }
        let min_count = ((min_support * n_txn as f64).ceil() as u64).max(1);
        let (s, o, i) = _mine_csr(&indptr, &indices, self.n_items, min_count, max_len)?;
        Ok((n_txn, s, o, i))
    }

    pub fn mine_eclat(&self, min_support: f64, max_len: Option<usize>)
        -> PyResult<(usize, Vec<u64>, Vec<u32>, Vec<u32>)>
    {
        let (indptr, indices, n_txn) = self.build_csr()?;
        if n_txn == 0 { return Ok((0, vec![], vec![], vec![])); }
        let min_count = ((min_support * n_txn as f64).ceil() as u64).max(1);
        let (s, o, i) = _eclat_mine_csr(&indptr, &indices, self.n_items, min_count, max_len)?;
        Ok((n_txn, s, o, i))
    }

    pub fn reset(&mut self) {
        self.txns.clear();
        self.txns.shrink_to_fit();
        self.n_rows = 0;
    }
}

impl FPMiner {
    fn build_csr(&self) -> PyResult<(Vec<i32>, Vec<i32>, usize)> {
        if self.txns.is_empty() {
            return Ok((vec![0i32], vec![], 0));
        }
        let mut entries: Vec<(i64, &Vec<i32>)> = self.txns.iter().map(|(&k, v)| (k, v)).collect();
        entries.par_sort_unstable_by_key(|(t, _)| *t);

        let n_txn = entries.len();
        let mut indptr: Vec<i32> = Vec::with_capacity(n_txn + 1);
        let mut indices: Vec<i32> = Vec::new();
        indptr.push(0);

        for (_, items) in &entries {
            let mut sorted = (*items).clone();
            sorted.sort_unstable();
            sorted.dedup();
            indices.extend_from_slice(&sorted);
            indptr.push(indices.len() as i32);
        }

        Ok((indptr, indices, n_txn))
    }
}
