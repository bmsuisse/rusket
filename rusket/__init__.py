from typing import Any

from . import viz
from ._internal._config import (
    _auto_detect_cuda,
    disable_cuda,
    disable_gpu,
    enable_cuda,
    enable_gpu,
    is_cuda_enabled,
    is_gpu_enabled,
)
from ._internal.analytics import customer_saturation, find_substitutes
from ._internal.similarity import similar_items
from .evaluation.metrics import coverage_at_k, evaluate, novelty_at_k
from .evaluation.model_selection import (
    CrossValidationResult,
    OptunaSearchSpace,
    chronological_split,
    cross_validate,
    leave_one_out_split,
    optuna_optimize,
    train_test_split,
    user_stratified_split,
)
from .evaluation.pipeline import Pipeline
from .export import mlflow as mlflow
from .export.ann import ApproximateNearestNeighbors
from .export.factors import export_item_factors
from .export.hybrid_embedding import HybridEmbeddingIndex, fuse_embeddings
from .miners.association_rules import association_rules
from .miners.eclat import Eclat, eclat
from .miners.fin import FIN
from .miners.fpgrowth import FPGrowth, fpgrowth
from .miners.hupm import HUPM, hupm, mine_hupm
from .miners.lcm import LCM
from .miners.mine import mine
from .miners.negfin import NegFIN
from .miners.prefixspan import PrefixSpan, prefixspan, sequences_from_event_log
from .miners.streaming import FPMiner, mine_duckdb, mine_spark
from .miners.transactions import (
    from_arrow,
    from_pandas,
    from_polars,
    from_ratings,
    from_spark,
    from_transactions,
    from_transactions_csr,
)
from .model import BaseModel, load_model
from .recommenders.als import ALS, eALS
from .recommenders.bpr import BPR
from .recommenders.content_based import ContentBased
from .recommenders.ease import EASE
from .recommenders.fm import FM
from .recommenders.hybrid import HybridRecommender
from .recommenders.item_knn import ItemKNN
from .recommenders.lightgcn import LightGCN
from .recommenders.nmf import NMF
from .recommenders.popularity import PopularityRecommender
from .recommenders.recommend import NextBestAction, Recommender, score_potential
from .recommenders.rules import RuleBasedRecommender
from .recommenders.svd import SVD
from .recommenders.user_knn import UserKNN
from .sequential.bert4rec import BERT4Rec
from .sequential.fpmc import FPMC
from .sequential.sasrec import SASRec
from .viz.pacmap import PaCMAP, pacmap, pacmap2, pacmap3
from .viz.pca import PCA, pca, pca2, pca3
from .viz.plots import to_networkx, to_networkxr

# Auto-detect CUDA on import
_auto_detect_cuda()

__all__ = [
    "fpgrowth",
    "FPGrowth",
    "eclat",
    "Eclat",
    "FIN",
    "NegFIN",
    "LCM",
    "mine",
    "mine_duckdb",
    "mine_spark",
    "association_rules",
    "from_ratings",
    "from_transactions",
    "from_transactions_csr",
    "from_pandas",
    "from_polars",
    "from_spark",
    "from_arrow",
    "FPMiner",
    "BaseModel",
    "load_model",
    "ALS",
    "eALS",
    "BPR",
    "EASE",
    "ItemKNN",
    "UserKNN",
    "FPMC",
    "FM",
    "SVD",
    "LightGCN",
    "SASRec",
    "BERT4Rec",
    "prefixspan",
    "PrefixSpan",
    "sequences_from_event_log",
    "hupm",
    "HUPM",
    "mine_hupm",
    "similar_items",
    "Recommender",
    "NextBestAction",
    "score_potential",
    "find_substitutes",
    "customer_saturation",
    "export_item_factors",
    "viz",
    "evaluate",
    "coverage_at_k",
    "novelty_at_k",
    "train_test_split",
    "leave_one_out_split",
    "chronological_split",
    "user_stratified_split",
    "cross_validate",
    "optuna_optimize",
    "CrossValidationResult",
    "OptunaSearchSpace",
    "PCA",
    "pca",
    "pca2",
    "pca3",
    "PopularityRecommender",
    "ContentBased",
    "HybridRecommender",
    "HybridEmbeddingIndex",
    "fuse_embeddings",
    "NMF",
    "Pipeline",
    "to_networkx",
    "to_networkxr",
    "ApproximateNearestNeighbors",
    "PaCMAP",
    "pacmap",
    "pacmap2",
    "pacmap3",
    "mlflow",
    "RuleBasedRecommender",
    "FAISSIndex",
    "build_faiss_index",
    "export_vectors",
    "export_multi_vectors",
    # CUDA API (primary)
    "check_cuda_available",
    "enable_cuda",
    "disable_cuda",
    "is_cuda_enabled",
    # Backward compat (GPU aliases)
    "check_gpu_available",
    "enable_gpu",
    "disable_gpu",
    "is_gpu_enabled",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FAISSIndex": (".export.faiss_ann", "FAISSIndex"),
    "build_faiss_index": (".export.faiss_ann", "build_faiss_index"),
    "export_vectors": (".export.vector_export", "export_vectors"),
    "export_multi_vectors": (".export.vector_export", "export_multi_vectors"),
    "check_cuda_available": (".integrations.cuda", "check_cuda_available"),
    "check_gpu_available": (".integrations.cuda", "check_cuda_available"),
}


def __getattr__(name: str) -> Any:
    """Lazy imports for optional dependency modules."""
    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(mod_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module 'rusket' has no attribute {name!r}")
