use ahash::AHashMap;
use pyo3::prelude::*;

fn rule_combinations(itemset: &[u32]) -> Vec<(Vec<u32>, Vec<u32>)> {
    let n = itemset.len();
    let mut rules = Vec::new();
    for ant_size in 1..n {
        let mut indices: Vec<usize> = (0..ant_size).collect();
        loop {
            let ant: Vec<u32> = indices.iter().map(|&i| itemset[i]).collect();
            let ant_set: std::collections::HashSet<u32> = ant.iter().copied().collect();
            let con: Vec<u32> = itemset.iter().filter(|&&x| !ant_set.contains(&x)).copied().collect();
            rules.push((ant, con));

            let mut i = ant_size as isize - 1;
            while i >= 0 {
                if indices[i as usize] < n - (ant_size - i as usize) { break; }
                i -= 1;
            }
            if i < 0 { break; }
            let i = i as usize;
            indices[i] += 1;
            for j in (i + 1)..ant_size { indices[j] = indices[j - 1] + 1; }
        }
    }
    rules
}

const METRIC_NAMES: &[&str] = &[
    "antecedent support", "consequent support", "support", "confidence",
    "lift", "representativity", "leverage", "conviction",
    "zhangs_metric", "jaccard", "certainty", "kulczynski",
];

#[inline]
fn compute_metrics(
    s_ac: f64, s_a: f64, s_c: f64,
    dis_ac: f64, dis_a: f64, dis_c: f64, dis_int: f64, dis_int_: f64,
    num_itemsets: f64, return_metrics: &[usize],
) -> Vec<f64> {
    let conf_denom = s_a * (num_itemsets - dis_a) - dis_int;
    let confidence = if conf_denom == 0.0 { f64::INFINITY } else { s_ac * (num_itemsets - dis_ac) / conf_denom };
    let conf_ca_denom = s_c * (num_itemsets - dis_c) - dis_int_;
    let conf_ca = if conf_ca_denom == 0.0 { f64::INFINITY } else { s_ac * (num_itemsets - dis_ac) / conf_ca_denom };
    let support = s_ac;
    let lift = if s_c == 0.0 { f64::INFINITY } else { confidence / s_c };
    let representativity = (num_itemsets - dis_ac) / num_itemsets;
    let leverage = support - s_a * s_c;
    let conviction = if confidence >= 1.0 { f64::INFINITY } else { (1.0 - s_c) / (1.0 - confidence) };
    let zd = f64::max(s_ac * (1.0 - s_a), s_a * (s_c - s_ac));
    let zhangs_metric = if zd == 0.0 { 0.0 } else { leverage / zd };
    let jd = s_a + s_c - support;
    let jaccard = if jd == 0.0 { 0.0 } else { support / jd };
    let cd = 1.0 - s_c;
    let certainty = if cd == 0.0 { 0.0 } else { (confidence - s_c) / cd };
    let kulczynski = (confidence + conf_ca) / 2.0;
    let all = [s_a, s_c, support, confidence, lift, representativity, leverage, conviction, zhangs_metric, jaccard, certainty, kulczynski];
    return_metrics.iter().map(|&idx| all[idx]).collect()
}

#[inline]
fn make_key(items: &[u32]) -> Vec<u32> {
    let mut k = items.to_vec();
    k.sort_unstable();
    k
}

#[pyfunction]
#[pyo3(signature = (itemsets, supports, num_itemsets, metric, min_threshold, support_only, return_metrics))]
pub fn association_rules_inner(
    _py: Python<'_>,
    itemsets: Vec<Vec<u32>>,
    supports: Vec<f64>,
    num_itemsets: usize,
    metric: String,
    min_threshold: f64,
    support_only: bool,
    return_metrics: Vec<String>,
) -> PyResult<(Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<f64>>)> {
    let support_map: AHashMap<Vec<u32>, f64> = itemsets
        .iter()
        .zip(supports.iter())
        .map(|(iset, &sup)| (make_key(iset), sup))
        .collect();

    let filter_metric_idx = if support_only {
        2usize
    } else {
        METRIC_NAMES.iter().position(|&m| m == metric.as_str())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Unknown metric: '{}'", metric)))?
    };

    let return_indices: Vec<usize> = return_metrics
        .iter()
        .map(|m| METRIC_NAMES.iter().position(|&name| name == m.as_str())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Unknown metric: '{}'", m))))
        .collect::<PyResult<_>>()?;

    let n = num_itemsets as f64;
    let nan = f64::NAN;
    let n_ret = return_indices.len();

    type RuleEntry = (Vec<u32>, Vec<u32>, Vec<f64>);
    let mut ant_out: Vec<Vec<u32>> = Vec::new();
    let mut con_out: Vec<Vec<u32>> = Vec::new();
    let mut metric_cols: Vec<Vec<f64>> = vec![Vec::new(); n_ret];

    for (iset, &s_ac) in itemsets.iter().zip(supports.iter()) {
        if iset.len() < 2 { continue; }
        for (ant, con) in rule_combinations(iset) {
            let score = if support_only {
                s_ac
            } else {
                let ant_key = make_key(&ant);
                let con_key = make_key(&con);
                let s_a = *support_map.get(&ant_key)
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!(
                        "Missing support for antecedent {:?}. \
                         You are likely getting this error because the DataFrame is missing \
                         antecedent and/or consequent information. \
                         You can try using the `support_only=True` option", ant_key)))?;
                let s_c = *support_map.get(&con_key)
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!(
                        "Missing support for consequent {:?}. \
                         You are likely getting this error because the DataFrame is missing \
                         antecedent and/or consequent information. \
                         You can try using the `support_only=True` option", con_key)))?;
                compute_metrics(s_ac, s_a, s_c, 0.0, 0.0, 0.0, 0.0, 0.0, n, &[filter_metric_idx])[0]
            };

            if score >= min_threshold {
                let vals = if support_only {
                    let mut v = vec![nan; n_ret];
                    for (idx, &ri) in return_indices.iter().enumerate() {
                        if ri == 2 { v[idx] = s_ac; }
                    }
                    v
                } else {
                    let s_a = support_map[&make_key(&ant)];
                    let s_c = support_map[&make_key(&con)];
                    compute_metrics(s_ac, s_a, s_c, 0.0, 0.0, 0.0, 0.0, 0.0, n, &return_indices)
                };
                ant_out.push(ant);
                con_out.push(con);
                for (col, v) in metric_cols.iter_mut().zip(vals.into_iter()) {
                    col.push(v);
                }
            }
        }
    }

    Ok((ant_out, con_out, metric_cols))

}
