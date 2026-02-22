use mimalloc::MiMalloc;
use pyo3::prelude::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod als;
mod association_rules;
mod bpr;
mod eclat;
mod fpgrowth;
mod hupm;
mod miner;
mod prefixspan;
mod ease;
mod item_knn;

#[pymodule]
fn _rusket(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fpgrowth::fpgrowth_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fpgrowth::fpgrowth_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(eclat::eclat_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(eclat::eclat_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(
        association_rules::association_rules_inner,
        m
    )?)?;
    m.add_class::<miner::FPMiner>()?;
    m.add_function(wrap_pyfunction!(als::als_fit_implicit, m)?)?;
    m.add_function(wrap_pyfunction!(als::als_recommend_items, m)?)?;
    m.add_function(wrap_pyfunction!(als::als_recommend_users, m)?)?;
    m.add_function(wrap_pyfunction!(bpr::bpr_fit_implicit, m)?)?;
    m.add_function(wrap_pyfunction!(prefixspan::prefixspan_mine_py, m)?)?;
    m.add_function(wrap_pyfunction!(hupm::hupm_mine_py, m)?)?;
    m.add_function(wrap_pyfunction!(ease::ease_recommend_items, m)?)?;
    m.add_function(wrap_pyfunction!(item_knn::itemknn_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(item_knn::itemknn_recommend_items, m)?)?;
    Ok(())
}
