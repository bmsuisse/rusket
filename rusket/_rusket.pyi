from typing import List, Tuple, Optional
import numpy.typing as npt
import numpy as np

def fpgrowth_from_dense(
    data: npt.NDArray[np.uint8], min_count: int, max_len: Optional[int] = None
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def fpgrowth_from_csr(
    indptr: npt.NDArray[np.int32],
    indices: npt.NDArray[np.int32],
    n_cols: int,
    min_count: int,
    max_len: Optional[int] = None,
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def eclat_from_dense(
    data: npt.NDArray[np.uint8], min_count: int, max_len: Optional[int] = None
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def eclat_from_csr(
    indptr: npt.NDArray[np.int32],
    indices: npt.NDArray[np.int32],
    n_cols: int,
    min_count: int,
    max_len: Optional[int] = None,
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def association_rules_inner(
    itemsets: List[List[int]],
    supports: List[float],
    num_itemsets: int,
    metric: str,
    min_threshold: float,
    support_only: bool,
    return_metrics: List[str],
) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]: ...
