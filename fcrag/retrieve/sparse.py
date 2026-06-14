"""
fcrag/retrieve/sparse.py — FCRAG 2.0 Sparse (BM25) Retriever
=============================================================
Loads pre-built BM25 pickle files and runs keyword-based retrieval.

Key design decisions:
  - Re-uses `tokenize()` from bm25_builder so query tokens match index tokens.
  - BM25 indexes are loaded lazily per collection and cached in RAM.
  - Returns list[RetrievedChunk] sorted by descending BM25 score.
  - Fails gracefully (empty list + warning) if the index file is missing.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fcrag.ingest.bm25_builder import tokenize
from fcrag.ingest.embedder import load_config
from fcrag.retrieve.schemas import RetrievedChunk

# Where the BM25 pkl files live (mirrors bm25_builder.INDEX_OUTPUT)
INDEX_DIR = ROOT / "data" / "processed" / "indexes"


class SparseRetriever:
    """
    Performs keyword (BM25 Okapi) retrieval using pre-built index pickles.

    Usage
    -----
    >>> sr = SparseRetriever()
    >>> results = sr.retrieve("TS 38.331 handover A3 offset", "3gpp_specs", top_k=10)
    """

    def __init__(self):
        self.config = load_config()
        self.retrieval_cfg = self.config["retrieval"]
        # Lazy-loaded cache: collection_name → {"model": BM25Okapi, "payloads": list}
        self._index_cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_index(self, collection: str) -> dict | None:
        """Load the BM25 bundle from disk, caching in RAM for subsequent calls."""
        if collection in self._index_cache:
            return self._index_cache[collection]

        pkl_path = INDEX_DIR / f"bm25_{collection}.pkl"
        if not pkl_path.exists():
            print(f"[SparseRetriever] BM25 index not found: {pkl_path}. "
                  f"Run `uv run python scripts/ingest_all.py --step bm25` first.")
            return None

        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)

        self._index_cache[collection] = bundle
        return bundle

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
    ) -> List[RetrievedChunk]:
        """
        Tokenize *query* and return the top_k highest-scoring BM25 chunks.

        Parameters
        ----------
        query      : Natural language query (telecom-aware tokenizer applied).
        collection : Collection name whose pkl file to load.
        top_k      : Number of results (default: retrieval.sparse_top_k from config).

        Returns
        -------
        List[RetrievedChunk] sorted by descending bm25_score (zero-scored docs excluded).
        """
        if top_k is None:
            top_k = self.retrieval_cfg.get("sparse_top_k", 20)

        bundle = self._load_index(collection)
        if bundle is None:
            return []

        model = bundle["model"]
        payloads: List[dict] = bundle["payloads"]

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = model.get_scores(query_tokens)

        # Sort by descending score, keeping only non-zero entries
        ranked = sorted(
            ((score, idx) for idx, score in enumerate(scores) if score > 0.0),
            key=lambda x: x[0],
            reverse=True,
        )

        results: List[RetrievedChunk] = []
        for score, idx in ranked[:top_k]:
            payload = payloads[idx]
            # Build metadata from any extra keys beyond the standard ones
            std_keys = {"text", "source_file", "clause_id", "source_type", "id"}
            meta = {k: v for k, v in payload.items() if k not in std_keys}

            chunk = RetrievedChunk(
                text=payload.get("text", ""),
                collection=collection,
                source_file=payload.get("source_file", ""),
                clause_id=payload.get("clause_id", ""),
                source_type=payload.get("source_type", ""),
                bm25_score=float(score),
                metadata=meta,
            )
            results.append(chunk)

        return results
