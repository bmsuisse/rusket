use ahash::AHashMap;
use rayon::prelude::*;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::eclat::_eclat_mine_csr;
use crate::fpgrowth::_mine_csr;
use crate::negfin::_negfin_mine_csr;

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
    #[pyo3(signature = (n_items, max_ram_mb=None, hint_n_transactions=None))]
    pub fn new(
        n_items: usize,
        max_ram_mb: Option<usize>,
        hint_n_transactions: Option<usize>,
    ) -> Self {
        let max_ram_bytes = max_ram_mb.map(|mb| mb.saturating_mul(1024 * 1024));
        let txns = match hint_n_transactions {
            Some(n) => AHashMap::with_capacity(n),
            None => AHashMap::new(),
        };
        FPMiner {
            txns,
            n_rows: 0,
            n_items,
            max_ram_bytes,
        }
    }

    pub fn add_chunk(
        &mut self,
        txn_ids: PyReadonlyArray1<i64>,
        item_ids: PyReadonlyArray1<i32>,
    ) -> PyResult<()> {
        let txns = txn_ids.as_slice()?;
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

    #[getter]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    #[getter]
    pub fn n_items(&self) -> usize {
        self.n_items
    }
    #[getter]
    pub fn n_transactions(&self) -> usize {
        self.txns.len()
    }
    #[getter]
    pub fn max_ram_mb(&self) -> Option<usize> {
        self.max_ram_bytes.map(|b| b / (1024 * 1024))
    }

    pub fn mine_fpgrowth(
        &self,
        min_support: f64,
        max_len: Option<usize>,
    ) -> PyResult<(usize, Vec<u64>, Vec<u32>, Vec<u32>)> {
        let n_txn_est = self.txns.len();
        if n_txn_est == 0 {
            return Ok((0, vec![], vec![], vec![]));
        }
        let min_count = ((min_support * n_txn_est as f64).ceil() as u64).max(1);
        let (indptr, indices, n_txn, inv_remap) = self.build_csr_filtered(min_count)?;
        if n_txn == 0 {
            return Ok((0, vec![], vec![], vec![]));
        }
        let n_frequent = inv_remap.len();
        let (s, o, i) = _mine_csr(&indptr, &indices, n_frequent, min_count, max_len)?;
        let i = Self::unmap_items(i, &inv_remap);
        Ok((n_txn, s, o, i))
    }

    pub fn mine_eclat(
        &self,
        min_support: f64,
        max_len: Option<usize>,
    ) -> PyResult<(usize, Vec<u64>, Vec<u32>, Vec<u32>)> {
        let n_txn_est = self.txns.len();
        if n_txn_est == 0 {
            return Ok((0, vec![], vec![], vec![]));
        }
        let min_count = ((min_support * n_txn_est as f64).ceil() as u64).max(1);
        let (indptr, indices, n_txn, inv_remap) = self.build_csr_filtered(min_count)?;
        if n_txn == 0 {
            return Ok((0, vec![], vec![], vec![]));
        }
        let n_frequent = inv_remap.len();
        let (s, o, i) = _eclat_mine_csr(&indptr, &indices, n_frequent, min_count, max_len)?;
        let i = Self::unmap_items(i, &inv_remap);
        Ok((n_txn, s, o, i))
    }

    pub fn mine_negfin(
        &self,
        min_support: f64,
        max_len: Option<usize>,
    ) -> PyResult<(usize, Vec<u64>, Vec<u32>, Vec<u32>)> {
        let n_txn_est = self.txns.len();
        if n_txn_est == 0 {
            return Ok((0, vec![], vec![], vec![]));
        }
        let min_count = ((min_support * n_txn_est as f64).ceil() as u64).max(1);
        let (indptr, indices, n_txn, inv_remap) = self.build_csr_filtered(min_count)?;
        if n_txn == 0 {
            return Ok((0, vec![], vec![], vec![]));
        }
        let n_frequent = inv_remap.len();
        let (s, o, i) = _negfin_mine_csr(&indptr, &indices, n_frequent, min_count, max_len)?;
        let i = Self::unmap_items(i, &inv_remap);
        Ok((n_txn, s, o, i))
    }



    pub fn reset(&mut self) {
        self.txns.clear();
        self.txns.shrink_to_fit();
        self.n_rows = 0;
    }
}

impl FPMiner {
    fn unmap_items(items: Vec<u32>, inv_remap: &[u32]) -> Vec<u32> {
        items
            .into_iter()
            .map(|idx| inv_remap[idx as usize])
            .collect()
    }

    fn build_csr_filtered(
        &self,
        min_count: u64,
    ) -> PyResult<(Vec<i32>, Vec<i32>, usize, Vec<u32>)> {
        if self.txns.is_empty() {
            return Ok((vec![0i32], vec![], 0, vec![]));
        }

        let mut item_freq: Vec<u64> = vec![0u64; self.n_items];
        for items in self.txns.values() {
            for &item in items {
                if (item as usize) < self.n_items {
                    item_freq[item as usize] += 1;
                }
            }
        }

        let mut freq_items: Vec<(u64, usize)> = item_freq
            .iter()
            .enumerate()
            .filter(|(_, &f)| f >= min_count)
            .map(|(id, &f)| (f, id))
            .collect();
        freq_items.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut remap: Vec<u32> = vec![u32::MAX; self.n_items];
        let n_frequent = freq_items.len();
        for (new_id, &(_, orig_id)) in freq_items.iter().enumerate() {
            remap[orig_id] = new_id as u32;
        }
        let inv_remap: Vec<u32> = freq_items
            .iter()
            .map(|&(_, orig_id)| orig_id as u32)
            .collect();

        if n_frequent == 0 {
            return Ok((vec![0i32], vec![], 0, vec![]));
        }

        let mut entries: Vec<(i64, &Vec<i32>)> = self.txns.iter().map(|(&k, v)| (k, v)).collect();
        entries.par_sort_unstable_by_key(|(t, _)| *t);

        let n_txn = entries.len();
        let mut indptr: Vec<i32> = Vec::with_capacity(n_txn + 1);
        let mut indices: Vec<i32> = Vec::new();
        indptr.push(0);

        for (_, items) in &entries {
            let mut mapped: Vec<i32> = items
                .iter()
                .filter_map(|&item| {
                    let r = remap[item as usize];
                    if r != u32::MAX {
                        Some(r as i32)
                    } else {
                        None
                    }
                })
                .collect();
            mapped.sort_unstable();
            mapped.dedup();
            indices.extend_from_slice(&mapped);
            indptr.push(indices.len() as i32);
        }

        Ok((indptr, indices, n_txn, inv_remap))
    }
}
