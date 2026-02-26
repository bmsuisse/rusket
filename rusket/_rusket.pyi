import numpy as np
import numpy.typing as npt

def fpgrowth_from_dense(
    data: npt.NDArray[np.uint8], min_count: int, max_len: int | None = None
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def fpgrowth_from_csr(
    indptr: npt.NDArray[np.int32],
    indices: npt.NDArray[np.int32],
    n_cols: int,
    min_count: int,
    max_len: int | None = None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def eclat_from_dense(
    data: npt.NDArray[np.uint8], min_count: int, max_len: int | None = None
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def eclat_from_csr(
    indptr: npt.NDArray[np.int32],
    indices: npt.NDArray[np.int32],
    n_cols: int,
    min_count: int,
    max_len: int | None = None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def association_rules_inner(
    itemsets: list[list[int]],
    supports: list[float],
    num_itemsets: int,
    metric: str,
    min_threshold: float,
    support_only: bool,
    return_metrics: list[str],
) -> tuple[list[list[int]], list[list[int]], list[list[float]]]: ...
def als_fit_implicit(
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32],
    data: npt.NDArray[np.float32],
    n_users: int,
    n_items: int,
    factors: int,
    regularization: float,
    alpha: float,
    iterations: int,
    seed: int,
    verbose: bool,
    cg_iters: int = 3,
    use_cholesky: bool = False,
    anderson_m: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
def als_recommend_items(
    user_factors: npt.NDArray[np.float32],
    item_factors: npt.NDArray[np.float32],
    user_id: int,
    n: int,
    exclude_indptr: npt.NDArray[np.int64],
    exclude_indices: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]: ...
def als_recommend_all(
    user_factors: npt.NDArray[np.float32],
    item_factors: npt.NDArray[np.float32],
    n: int,
    exclude_indptr: npt.NDArray[np.int64],
    exclude_indices: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float32]]: ...
def als_recommend_users(
    user_factors: npt.NDArray[np.float32],
    item_factors: npt.NDArray[np.float32],
    item_id: int,
    n: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]: ...
def bpr_fit_implicit(
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32],
    n_users: int,
    n_items: int,
    factors: int,
    learning_rate: float,
    regularization: float,
    iterations: int,
    seed: int,
    verbose: bool,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
def hupm_mine_py(
    items_list: list[list[int]],
    utils_list: list[list[float]],
    min_utility: float,
    max_len: int | None = None,
) -> tuple[list[float], list[list[int]]]: ...
def prefixspan_mine_py(
    sequences: list[list[int]],
    min_count: int,
    max_len: int | None = None,
) -> tuple[list[int], list[list[int]]]: ...
def fin_from_dense(
    data: npt.NDArray[np.uint8], min_count: int, max_len: int | None = None
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def fin_from_csr(
    indptr: npt.NDArray[np.int32],
    indices: npt.NDArray[np.int32],
    n_cols: int,
    min_count: int,
    max_len: int | None = None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def lcm_from_dense(
    data: npt.NDArray[np.uint8], min_count: int, max_len: int | None = None
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def lcm_from_csr(
    indptr: npt.NDArray[np.int32],
    indices: npt.NDArray[np.int32],
    n_cols: int,
    min_count: int,
    max_len: int | None = None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...

class FPMiner:
    n_rows: int
    n_items: int
    n_transactions: int
    max_ram_mb: int | None

    def __init__(self, n_items: int, max_ram_mb: int | None = None, hint_n_transactions: int | None = None) -> None: ...
    def add_chunk(self, txn_ids: npt.NDArray[np.int64], item_ids: npt.NDArray[np.int32]) -> None: ...
    def mine_fpgrowth(self, min_support: float, max_len: int | None) -> tuple[int, list[int], list[int], list[int]]: ...
    def mine_eclat(self, min_support: float, max_len: int | None) -> tuple[int, list[int], list[int], list[int]]: ...
    def mine_auto(
        self, min_support: float, max_len: int | None
    ) -> tuple[int, list[int], list[int], list[int], str]: ...
    def reset(self) -> None: ...

def ease_recommend_items(
    item_weights: npt.NDArray[np.float32],
    fit_indptr: npt.NDArray[np.int64],
    fit_indices: npt.NDArray[np.int32],
    fit_data: npt.NDArray[np.float32],
    user_id: int,
    n: int,
    exc_indptr: npt.NDArray[np.int64],
    exc_indices: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]: ...
def itemknn_top_k(
    indptr: npt.NDArray[np.int64], indices: npt.NDArray[np.int32], data: npt.NDArray[np.float32], k: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.float32]]: ...
def itemknn_recommend_items(
    w_indptr: npt.NDArray[np.int64],
    w_indices: npt.NDArray[np.int32],
    w_data: npt.NDArray[np.float32],
    fit_indptr: npt.NDArray[np.int64],
    fit_indices: npt.NDArray[np.int32],
    user_data: npt.NDArray[np.float32],
    user_id: int,
    n: int,
    exc_indptr: npt.NDArray[np.int64],
    exc_indices: npt.NDArray[np.int32],
    n_items: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]: ...
def fpmc_fit(
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32],
    n_users: int,
    n_items: int,
    factors: int,
    learning_rate: float,
    regularization: float,
    iterations: int,
    seed: int,
    verbose: bool,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
def fm_fit(
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32],
    y: npt.NDArray[np.float32],
    n_samples: int,
    n_features: int,
    factors: int,
    learning_rate: float,
    regularization: float,
    iterations: int,
    seed: int,
    verbose: bool,
) -> tuple[float, npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
def fm_predict(
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32],
    w0: float,
    w: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
    factors: int,
    n_samples: int,
) -> npt.NDArray[np.float32]: ...
def lightgcn_fit(
    u_indptr: npt.NDArray[np.int64],
    u_indices: npt.NDArray[np.int32],
    i_indptr: npt.NDArray[np.int64],
    i_indices: npt.NDArray[np.int32],
    n_users: int,
    n_items: int,
    factors: int,
    k_layers: int,
    learning_rate: float,
    lambda_: float,
    iterations: int,
    seed: int,
    verbose: bool,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
def sasrec_fit(
    sequences: list[list[int]],
    n_items: int,
    factors: int,
    n_layers: int,
    max_seq: int,
    learning_rate: float,
    lambda_: float,
    iterations: int,
    seed: int,
    verbose: bool,
) -> npt.NDArray[np.float32]: ...
def ndcg_at_k(actual: list[int], pred: list[int], k: int) -> float: ...
def precision_at_k(actual: list[int], pred: list[int], k: int) -> float: ...
def recall_at_k(actual: list[int], pred: list[int], k: int) -> float: ...
def hit_rate_at_k(actual: list[int], pred: list[int], k: int) -> float: ...
def train_test_split(ids: list[int], test_size: float) -> tuple[list[int], list[int]]: ...
def leave_one_out(
    user_ids: list[int], item_ids: list[int], timestamps: list[float] | None
) -> tuple[list[int], list[int]]: ...
def pca_fit(
    data: npt.NDArray[np.float32],
    n_components: int,
) -> tuple[
    npt.NDArray[np.float32],  # components (n_components, n_features)
    npt.NDArray[np.float32],  # explained_variance (n_components,)
    npt.NDArray[np.float32],  # explained_variance_ratio (n_components,)
    npt.NDArray[np.float32],  # singular_values (n_components,)
    npt.NDArray[np.float32],  # mean (n_features,)
]: ...
def pca_transform(
    data: npt.NDArray[np.float32],
    mean: npt.NDArray[np.float32],
    components: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]: ...
