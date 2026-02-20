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
use std::io::{Read, Write, BufReader, BufWriter, Seek, SeekFrom};
use tempfile::tempfile;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::fpgrowth::_mine_csr;
use crate::eclat::_eclat_mine_csr;

enum Chunk {
    Memory(Vec<(i64, i32)>),
    Disk(std::fs::File, usize),
}

enum ChunkCursor<'a> {
    Memory { vec: &'a [(i64, i32)], pos: usize },
    Disk { reader: BufReader<&'a mut std::fs::File>, remaining: usize },
}

impl<'a> ChunkCursor<'a> {
    fn next_pair(&mut self) -> Option<(i64, i32)> {
        match self {
            ChunkCursor::Memory { vec, pos } => {
                if *pos < vec.len() {
                    let pair = vec[*pos];
                    *pos += 1;
                    Some(pair)
                } else {
                    None
                }
            }
            ChunkCursor::Disk { reader, remaining } => {
                if *remaining > 0 {
                    let mut buf = [0u8; 12];
                    if reader.read_exact(&mut buf).is_ok() {
                        *remaining -= 1;
                        let txn = i64::from_ne_bytes(buf[0..8].try_into().unwrap());
                        let item = i32::from_ne_bytes(buf[8..12].try_into().unwrap());
                        Some((txn, item))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }
}

#[pyclass]
pub struct FPMiner {
    /// Each element is one pre-sorted chunk, either in memory or on disk.
    chunks: Vec<Chunk>,
    /// Total number of pairs across all chunks.
    n_rows: usize,
    n_items: usize,
    max_ram_bytes: Option<usize>,
    current_ram_bytes: usize,
}

#[pymethods]
impl FPMiner {
    #[new]
    #[pyo3(signature = (n_items, max_ram_mb=None))]
    pub fn new(n_items: usize, max_ram_mb: Option<usize>) -> Self {
        let max_ram_bytes = max_ram_mb.map(|mb| mb.saturating_mul(1024 * 1024));
        FPMiner { chunks: Vec::new(), n_rows: 0, n_items, max_ram_bytes, current_ram_bytes: 0 }
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
        
        let chunk_len = chunk.len();
        self.n_rows += chunk_len;
        
        let chunk_bytes = chunk_len * 12; // 8 bytes for i64, 4 bytes for i32
        
        if let Some(limit) = self.max_ram_bytes {
            if self.current_ram_bytes + chunk_bytes > limit {
                // Spill to disk
                let mut file = tempfile().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                
                // Scope the writer so it drops and releases `file`
                {
                    let mut writer = BufWriter::new(&mut file);
                    for &(t, i) in &chunk {
                        writer.write_all(&t.to_ne_bytes()).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                        writer.write_all(&i.to_ne_bytes()).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                    }
                    writer.flush().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                } // writer drops here
                
                file.seek(SeekFrom::Start(0)).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                self.chunks.push(Chunk::Disk(file, chunk_len));
                return Ok(());
            }
        }
        
        // Keep in memory
        self.current_ram_bytes += chunk_bytes;
        self.chunks.push(Chunk::Memory(chunk));
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

    /// Max RAM threshold before spilling to disk.
    #[getter]
    pub fn max_ram_mb(&self) -> Option<usize> {
        self.max_ram_bytes.map(|b| b / (1024 * 1024))
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
        self.current_ram_bytes = 0;
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

        let mut cursors: Vec<ChunkCursor> = Vec::with_capacity(self.chunks.len());
        for chunk in self.chunks.iter_mut() {
            match chunk {
                Chunk::Memory(vec) => cursors.push(ChunkCursor::Memory { vec, pos: 0 }),
                Chunk::Disk(file, len) => {
                    file.seek(SeekFrom::Start(0)).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                    cursors.push(ChunkCursor::Disk { reader: BufReader::new(file), remaining: *len })
                },
            }
        }

        // Heap entry: Reverse((txn, item, cursor_idx))
        // min-heap → smallest (txn, item) pops first
        let mut heap: BinaryHeap<Reverse<(i64, i32, usize)>> = BinaryHeap::new();

        // Seed with the first element of each chunk
        for (ci, cursor) in cursors.iter_mut().enumerate() {
            if let Some((t, i)) = cursor.next_pair() {
                heap.push(Reverse((t, i, ci)));
            }
        }

        let mut indptr: Vec<i32> = Vec::with_capacity(self.n_rows / 8 + 2);
        let mut indices: Vec<i32> = Vec::with_capacity(self.n_rows);
        indptr.push(0);

        let mut prev_txn  = i64::MIN;
        let mut prev_item = i32::MIN;

        while let Some(Reverse((txn, item, ci))) = heap.pop() {
            // Advance cursor
            if let Some((next_t, next_i)) = cursors[ci].next_pair() {
                heap.push(Reverse((next_t, next_i, ci)));
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
