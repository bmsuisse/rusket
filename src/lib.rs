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
mod lcm;
mod miner;
mod prefixspan;
mod fin;
mod negfin;
mod ease;
mod item_knn;
mod fpmc;
mod fm;
mod lightgcn;
mod metrics;
mod model_selection;
mod pca;
mod pipeline;
mod sasrec;
mod svd;
mod ann;
mod incremental_pca;
mod nn_descent;
mod pacmap;
mod cross_validate;

#[pymodule]
fn _rusket(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fpgrowth::fpgrowth_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fpgrowth::fpgrowth_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(eclat::eclat_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(eclat::eclat_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fin::fin_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fin::fin_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(negfin::negfin_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(negfin::negfin_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(lcm::lcm_from_dense, m)?)?;
    m.add_function(wrap_pyfunction!(lcm::lcm_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(
        association_rules::association_rules_inner,
        m
    )?)?;
    m.add_class::<miner::FPMiner>()?;
    m.add_function(wrap_pyfunction!(als::als_fit_implicit, m)?)?;
    m.add_function(wrap_pyfunction!(als::als_recommend_items, m)?)?;
    m.add_function(wrap_pyfunction!(als::als_recalculate_user, m)?)?;
    m.add_function(wrap_pyfunction!(als::als_recommend_all, m)?)?;
    m.add_function(wrap_pyfunction!(als::als_recommend_users, m)?)?;
    m.add_function(wrap_pyfunction!(bpr::bpr_fit_implicit, m)?)?;
    m.add_function(wrap_pyfunction!(prefixspan::prefixspan_mine_py, m)?)?;
    m.add_function(wrap_pyfunction!(hupm::hupm_mine_py, m)?)?;
    m.add_function(wrap_pyfunction!(ease::ease_recommend_items, m)?)?;
    m.add_function(wrap_pyfunction!(item_knn::itemknn_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(item_knn::itemknn_recommend_items, m)?)?;
    m.add_function(wrap_pyfunction!(fpmc::fpmc_fit, m)?)?;
    m.add_function(wrap_pyfunction!(fm::fm_fit, m)?)?;
    m.add_function(wrap_pyfunction!(fm::fm_predict, m)?)?;
    m.add_function(wrap_pyfunction!(lightgcn::lightgcn_fit, m)?)?;
    m.add_function(wrap_pyfunction!(sasrec::sasrec_fit, m)?)?;
    m.add_function(wrap_pyfunction!(sasrec::sasrec_encode, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::ndcg_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::precision_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::recall_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hit_rate_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(model_selection::leave_one_out, m)?)?;
    m.add_function(wrap_pyfunction!(model_selection::train_test_split, m)?)?;
    m.add_function(wrap_pyfunction!(svd::svd_fit, m)?)?;
    m.add_function(wrap_pyfunction!(svd::svd_recommend_items, m)?)?;
    m.add_function(wrap_pyfunction!(svd::svd_recommend_users, m)?)?;
    m.add_function(wrap_pyfunction!(svd::svd_recommend_all, m)?)?;
    m.add_function(wrap_pyfunction!(svd::svdpp_fit, m)?)?;
    m.add_function(wrap_pyfunction!(svd::svdpp_recommend_items, m)?)?;
    m.add_function(wrap_pyfunction!(svd::svdpp_recommend_users, m)?)?;
    m.add_function(wrap_pyfunction!(svd::svdpp_recommend_all, m)?)?;
    m.add_function(wrap_pyfunction!(pca::pca_fit, m)?)?;
    m.add_function(wrap_pyfunction!(pca::pca_transform, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::pipeline_batch_recommend, m)?)?;
    m.add_class::<ann::AnnIndex>()?;
    m.add_class::<incremental_pca::RustIncrementalPCA>()?;
    m.add_function(wrap_pyfunction!(nn_descent::nn_descent_build, m)?)?;
    m.add_function(wrap_pyfunction!(pacmap::pacmap_fit, m)?)?;
    m.add_function(wrap_pyfunction!(cross_validate::cross_validate_als, m)?)?;
    m.add_function(wrap_pyfunction!(cross_validate::cross_validate_generic, m)?)?;
    Ok(())
}
