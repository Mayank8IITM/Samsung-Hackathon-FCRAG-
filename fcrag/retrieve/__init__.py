"""FCRAG 2.0 — fcrag.retrieve package

Public API
----------
from fcrag.retrieve import HybridRetriever, RetrievedChunk
"""

from fcrag.retrieve.schemas import RetrievedChunk
from fcrag.retrieve.retriever import HybridRetriever, reciprocal_rank_fusion
from fcrag.retrieve.dense import DenseRetriever
from fcrag.retrieve.sparse import SparseRetriever
from fcrag.retrieve.reranker import CrossEncoderReranker

__all__ = [
    "RetrievedChunk",
    "HybridRetriever",
    "DenseRetriever",
    "SparseRetriever",
    "CrossEncoderReranker",
    "reciprocal_rank_fusion",
]
