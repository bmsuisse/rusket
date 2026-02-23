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
