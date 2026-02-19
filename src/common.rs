/// Compute per-item support counts from a binarised row dataset.
/// `rows` is a list of column index lists (non-zero columns per row).
/// Returns a Vec where `result[item] = count of rows containing item`.
pub fn count_item_support(rows: &[Vec<u32>], n_items: usize) -> Vec<u64> {
    let mut counts = vec![0u64; n_items];
    for row in rows {
        for &item in row {
            if (item as usize) < n_items {
                counts[item as usize] += 1;
            }
        }
    }
    counts
}
