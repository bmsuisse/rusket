use pyo3::prelude::*;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod fpgrowth;
mod association_rules;

#[pymodule]
fn _rusket(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fpgrowth::fpgrowth_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fpgrowth::fpgrowth_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(association_rules::association_rules_inner, m)?)?;
    Ok(())
}
