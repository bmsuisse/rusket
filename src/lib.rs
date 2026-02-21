use pyo3::prelude::*;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod fpgrowth;
mod eclat;
mod association_rules;
mod miner;
mod als;
mod bpr;
mod prefixspan;
mod hupm;

#[pymodule]
fn _rusket(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fpgrowth::fpgrowth_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fpgrowth::fpgrowth_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(eclat::eclat_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(eclat::eclat_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(association_rules::association_rules_inner, m)?)?;
    m.add_class::<miner::FPMiner>()?;
    m.add_function(wrap_pyfunction!(als::als_fit_implicit, m)?)?;
    m.add_function(wrap_pyfunction!(als::als_recommend_items, m)?)?;
    m.add_function(wrap_pyfunction!(als::als_recommend_users, m)?)?;
    m.add_function(wrap_pyfunction!(bpr::bpr_fit_implicit, m)?)?;
    m.add_function(wrap_pyfunction!(prefixspan::prefixspan_mine_py, m)?)?;
    m.add_function(wrap_pyfunction!(hupm::hupm_mine_py, m)?)?;
    Ok(())
}
