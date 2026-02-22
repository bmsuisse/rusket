use std::time::Instant;

#[derive(Clone)]
struct BitSet {
    blocks: Vec<u64>,
}

impl BitSet {
    fn new(num_bits: usize) -> Self {
        BitSet {
            blocks: vec![0; num_bits.div_ceil(64)],
        }
    }
    fn set(&mut self, bit: usize) {
        self.blocks[bit / 64] |= 1 << (bit % 64);
    }
    fn count_ones(&self) -> u64 {
        self.blocks.iter().map(|b| b.count_ones() as u64).sum()
    }
    fn intersect_count_into(&self, other: &BitSet, out: &mut BitSet, min_count: u64) -> u64 {
        let n = self.blocks.len();
        let mut count = 0u64;
        for i in 0..n {
            let v = self.blocks[i] & other.blocks[i];
            out.blocks[i] = v;
            count += v.count_ones() as u64;
            let remaining_max = ((n - i - 1) * 64) as u64;
            if count + remaining_max < min_count {
                for j in (i + 1)..n {
                    out.blocks[j] = 0;
                }
                return 0;
            }
        }
        count
    }
}

// Eclat Recursive using BitSet
fn eclat_mine_bitset(
    active_items: &[(u32, BitSet)],
    min_count: u64,
) -> u64 {
    let mut total_itemsets = 0;
    let n_blocks = active_items.first().map_or(0, |(_, bs)| bs.blocks.len());
    let mut scratch = BitSet { blocks: vec![0u64; n_blocks] };

    for (i, (_item_a, bs_a)) in active_items.iter().enumerate() {
        let count = bs_a.count_ones();
        if count < min_count { continue; }
        total_itemsets += 1;

        let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
        for (item_b, bs_b) in &active_items[i + 1..] {
            let c = bs_a.intersect_count_into(bs_b, &mut scratch, min_count);
            if c >= min_count {
                next_active.push((*item_b, scratch.clone()));
            }
        }
        if !next_active.is_empty() {
            total_itemsets += eclat_mine_bitset(&next_active, min_count);
        }
    }
    total_itemsets
}

// Diffsets
fn diffset_intersect(diff_a: &[u32], diff_b: &[u32]) -> Vec<u32> {
    let mut res = Vec::new();
    let mut ia = 0;
    let mut ib = 0;
    while ia < diff_a.len() && ib < diff_b.len() {
        if diff_a[ia] < diff_b[ib] {
            ia += 1;
        } else if diff_a[ia] > diff_b[ib] {
            res.push(diff_b[ib]);
            ib += 1;
        } else {
            ia += 1;
            ib += 1;
        }
    }
    while ib < diff_b.len() {
        res.push(diff_b[ib]);
        ib += 1;
    }
    res
}

fn declat_recursive(
    active_items: &[(u32, Vec<u32>, u64)],
    min_count: u64,
) -> u64 {
    let mut total_itemsets = 0;
    for (i, (_item_a, d_a, sup_a)) in active_items.iter().enumerate() {
        if *sup_a < min_count { continue; }
        total_itemsets += 1;

        let mut next_active = Vec::with_capacity(active_items.len() - i - 1);
        for (item_b, d_b, sup_b) in &active_items[i + 1..] {
            if *sup_b < min_count { continue; }
            let d_ab = diffset_intersect(d_a, d_b);
            let sup_ab = sup_a - d_ab.len() as u64;
            if sup_ab >= min_count {
                next_active.push((*item_b, d_ab, sup_ab));
            }
        }
        if !next_active.is_empty() {
            total_itemsets += declat_recursive(&next_active, min_count);
        }
    }
    total_itemsets
}

// elements in tid_a not in tid_b
fn tidset_diff(tid_a: &[u32], tid_b: &[u32]) -> Vec<u32> {
    let mut res = Vec::new();
    let mut ia = 0;
    let mut ib = 0;
    while ia < tid_a.len() && ib < tid_b.len() {
        if tid_a[ia] < tid_b[ib] {
            res.push(tid_a[ia]);
            ia += 1;
        } else if tid_a[ia] > tid_b[ib] {
            ib += 1;
        } else {
            ia += 1;
            ib += 1;
        }
    }
    while ia < tid_a.len() {
        res.push(tid_a[ia]);
        ia += 1;
    }
    res
}

fn declat_mine(
    active_tids: &[(u32, Vec<u32>)], // item, tidset
    min_count: u64,
) -> u64 {
    let mut total_itemsets = 0;
    for (i, (_item_a, t_a)) in active_tids.iter().enumerate() {
        let sup_a = t_a.len() as u64;
        if sup_a < min_count { continue; }
        total_itemsets += 1;

        let mut next_active = Vec::with_capacity(active_tids.len() - i - 1);
        for (item_b, t_b) in &active_tids[i + 1..] {
            let sup_b = t_b.len() as u64;
            if sup_b < min_count { continue; }
            // d(ab) = t(a) \ t(b)
            let d_ab = tidset_diff(t_a, t_b);
            let sup_ab = sup_a - d_ab.len() as u64;
            if sup_ab >= min_count {
                next_active.push((*item_b, d_ab, sup_ab));
            }
        }
        if !next_active.is_empty() {
            total_itemsets += declat_recursive(&next_active, min_count);
        }
    }
    total_itemsets
}


fn hash(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn main() {
    let n_transactions = 100_000;
    let n_items = 200;
    let density = 0.05; // 5% sparse
    let min_count = 200; // 0.2%
    
    // Create deterministic realistic data
    // Add item frequencies: some popular, some rare
    let item_probs: Vec<f64> = (0..n_items).map(|i| {
        density * (2.0 * (n_items - i) as f64 / n_items as f64)
    }).collect();

    let mut bitsets = vec![BitSet::new(n_transactions); n_items];
    let mut tidsets = vec![Vec::new(); n_items];

    for i in 0..n_items {
        let p = item_probs[i];
        for t in 0..n_transactions {
            let h = hash(i as u64 * 100_000 + t as u64) % 100_000;
            if (h as f64) < (p * 100_000.0) {
                bitsets[i].set(t);
                tidsets[i].push(t as u32);
            }
        }
    }

    let active_bitsets: Vec<(u32, BitSet)> = bitsets.into_iter().enumerate().map(|(i, bs)| (i as u32, bs)).collect();
    let start = Instant::now();
    let count1 = eclat_mine_bitset(&active_bitsets, min_count);
    println!("dense Eclat took {:?} ({} itemsets)", start.elapsed(), count1);

    let active_tidsets: Vec<(u32, Vec<u32>)> = tidsets.into_iter().enumerate().map(|(i, tids)| (i as u32, tids)).collect();
    let start2 = Instant::now();
    let count2 = declat_mine(&active_tidsets, min_count);
    println!("dEclat took {:?} ({} itemsets)", start2.elapsed(), count2);
}
