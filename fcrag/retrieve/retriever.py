"""
fcrag/retrieve/retriever.py — FCRAG 2.0 Hybrid Retriever Orchestrator
======================================================================
Combines dense (Qdrant cosine) and sparse (BM25) retrieval via
Reciprocal Rank Fusion (RRF), then reranks the merged pool with a
cross-encoder.  This is the single entry-point used by LangGraph agents
and the FastAPI layer.

Pipeline:
  Query
    ├─► DenseRetriever   → top-20 per collection
    └─► SparseRetriever  → top-20 per collection
        │
        ▼
  Merge + Deduplicate  (by dedup_key)
        │
        ▼
  Reciprocal Rank Fusion  (RRF, k=60)
        │
        ▼
  CrossEncoderReranker  → top-5
        │
        ▼
  List[RetrievedChunk]
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fcrag.ingest.embedder import load_config
from fcrag.retrieve.dense import DenseRetriever
from fcrag.retrieve.reranker import CrossEncoderReranker
from fcrag.retrieve.schemas import RetrievedChunk
from fcrag.retrieve.sparse import SparseRetriever


# ---------------------------------------------------------------------------
# RRF helper
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: List[RetrievedChunk],
    k: int = 60,
) -> List[RetrievedChunk]:
    """
    Merge any number of ranked lists using Reciprocal Rank Fusion.

    RRF(d) = Σ  1 / (k + rank_i(d))
    where rank_i is 1-indexed position in list i.

    Chunks are identified by their dedup_key. The first occurrence of
    each key is used as the canonical RetrievedChunk object; subsequent
    occurrences only contribute their rank score.

    Parameters
    ----------
    *ranked_lists : Variable number of lists, each sorted best-first.
    k             : RRF constant (default: 60 per standard literature).

    Returns
    -------
    All unique chunks, sorted by descending rrf_score.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = chunk.dedup_key
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
            if key not in chunk_map:
                chunk_map[key] = chunk

    # Write scores back and return sorted list
    result: List[RetrievedChunk] = []
    for key, score in rrf_scores.items():
        chunk_map[key].rrf_score = score
        result.append(chunk_map[key])

    result.sort(key=lambda c: c.rrf_score, reverse=True)
    return result


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Public API for hybrid (dense + sparse + rerank) retrieval.

    Usage
    -----
    >>> hr = HybridRetriever()
    >>> chunks = hr.retrieve(
    ...     query="handover failure A3 offset TS 38.331",
    ...     collections=["3gpp_specs", "simu5g_narratives"],
    ...     top_k=5,
    ... )
    >>> for c in chunks:
    ...     print(f"[{c.rerank_score:.3f}] {c.clause_id}: {c.text[:80]}")
    """

    # Default collections to search when none are specified
    DEFAULT_COLLECTIONS = [
        "3gpp_specs",
        "simu5g_narratives",
        "alarm_history",
    ]

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
            
        self.config = load_config()
        self.retrieval_cfg = self.config["retrieval"]

        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.reranker = CrossEncoderReranker()

        self._rrf_k = self.retrieval_cfg.get("rrf_k_constant", 60)
        self._initialized = True

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        collections: List[str] | None = None,
        top_k: int | None = None,
        use_reranker: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Run the full hybrid retrieval pipeline.

        Parameters
        ----------
        query       : Natural language query (or fault-augmented string from TAAE).
        collections : Which Qdrant/BM25 collections to search. Defaults to
                      DEFAULT_COLLECTIONS.
        top_k       : Number of final results to return after reranking.
                      Defaults to retrieval.rerank_top_k from config (5).
        use_reranker: Set False to skip the cross-encoder step (faster,
                      useful for evaluation / bulk retrieval).

        Returns
        -------
        List[RetrievedChunk] sorted by rerank_score (or rrf_score if
        reranker is skipped), length ≤ top_k.
        """
        if collections is None:
            collections = self.DEFAULT_COLLECTIONS
        if top_k is None:
            top_k = self.retrieval_cfg.get("rerank_top_k", 5)

        dense_top_k = self.retrieval_cfg.get("dense_top_k", 20)
        sparse_top_k = self.retrieval_cfg.get("sparse_top_k", 20)

        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # 1. Fan-out dense + sparse retrieval in parallel
        # ------------------------------------------------------------------
        dense_results: Dict[str, List[RetrievedChunk]] = {}
        sparse_results: Dict[str, List[RetrievedChunk]] = {}

        def _dense(coll: str) -> tuple[str, List[RetrievedChunk]]:
            return coll, self.dense.retrieve(query, coll, top_k=dense_top_k)

        def _sparse(coll: str) -> tuple[str, List[RetrievedChunk]]:
            return coll, self.sparse.retrieve(query, coll, top_k=sparse_top_k)

        with ThreadPoolExecutor(max_workers=min(8, len(collections) * 2)) as executor:
            futures = (
                [executor.submit(_dense, c) for c in collections] +
                [executor.submit(_sparse, c) for c in collections]
            )
            for future in as_completed(futures):
                try:
                    coll, chunks = future.result()
                    # Distinguish dense vs sparse by checking which dict already has key
                    if coll not in dense_results:
                        dense_results[coll] = chunks
                    else:
                        sparse_results[coll] = chunks
                except Exception as exc:
                    print(f"[HybridRetriever] Retrieval error: {exc}")

        t_retrieved = time.perf_counter()

        # ------------------------------------------------------------------
        # 2. Flatten, then RRF across all collections
        # ------------------------------------------------------------------
        all_dense: List[RetrievedChunk] = []
        all_sparse: List[RetrievedChunk] = []
        for coll in collections:
            all_dense.extend(dense_results.get(coll, []))
            all_sparse.extend(sparse_results.get(coll, []))

        merged = reciprocal_rank_fusion(all_dense, all_sparse, k=self._rrf_k)

        t_fused = time.perf_counter()

        # ------------------------------------------------------------------
        # 3. Rerank
        # ------------------------------------------------------------------
        if use_reranker:
            final = self.reranker.rerank(query, merged, top_k=top_k)
        else:
            # Just truncate by RRF score
            for c in merged:
                c.rerank_score = c.rrf_score
            final = merged[:top_k]

        t_end = time.perf_counter()

        print(
            f"[HybridRetriever] Retrieved {len(all_dense)}d+{len(all_sparse)}s -> "
            f"merged {len(merged)} -> reranked {len(final)} | "
            f"retrieve={1000*(t_retrieved-t_start):.0f}ms "
            f"fuse={1000*(t_fused-t_retrieved):.0f}ms "
            f"rerank={1000*(t_end-t_fused):.0f}ms"
        )

        return final

    def search_text(
        self,
        query: str,
        collections: List[str] | None = None,
        top_k: int | None = None,
        use_reranker: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper — same as retrieve() but returns plain dicts.
        Useful for direct JSON serialisation or LangChain tool integration.
        """
        chunks = self.retrieve(query, collections, top_k, use_reranker)
        return [c.to_dict() for c in chunks]

    def retrieve_for_fault(
        self,
        fault_type: str,
        cell_id: str,
        kpi_summary: str,
        top_k: int | None = None,
    ) -> List[RetrievedChunk]:
        """
        Convenience method: builds a rich query from a structured fault event
        and searches across all default collections.

        Parameters
        ----------
        fault_type  : e.g. "HO_FAILURE", "PRB_CONGESTION"
        cell_id     : e.g. "Cell-42"
        kpi_summary : Short human-readable KPI deviation text.
        top_k       : Number of results.
        """
        query = (
            f"Fault: {fault_type} at {cell_id}. "
            f"KPI anomalies: {kpi_summary}. "
            f"Root cause and corrective action for {fault_type}."
        )
        return self.retrieve(query, collections=self.DEFAULT_COLLECTIONS, top_k=top_k)
