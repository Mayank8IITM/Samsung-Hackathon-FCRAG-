"""
fcrag/retrieve/reranker.py — FCRAG 2.0 Cross-Encoder Reranker
=============================================================
Re-scores a merged candidate list using a cross-encoder model
(cross-encoder/ms-marco-MiniLM-L-6-v2) and returns the top_k
most relevant chunks.

Key design decisions:
  - Uses sentence_transformers.CrossEncoder directly (no ONNX dependency
    at this stage — pure PyTorch CPU is fast enough for ≤40 candidates).
  - Model is loaded lazily on first call; subsequent calls reuse it.
  - Falls back gracefully (returns input sorted by rrf_score) if the
    model cannot be loaded.
  - Pairs are batched and scored in a single forward pass for efficiency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fcrag.ingest.embedder import load_config
from fcrag.retrieve.schemas import RetrievedChunk


class CrossEncoderReranker:
    """
    Reranks a list of RetrievedChunk objects using a cross-encoder model.

    Usage
    -----
    >>> reranker = CrossEncoderReranker()
    >>> top5 = reranker.rerank("handover failure", candidates, top_k=5)
    """

    def __init__(self):
        self.config = load_config()
        reranker_cfg = self.config["models"]["reranker"]
        self.model_name = reranker_cfg.get("name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._model = None  # Lazy load

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Lazily load the CrossEncoder model (only on first rerank call)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            print(f"[Reranker] Loading CrossEncoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
            print(f"[Reranker] CrossEncoder ready.")
        except Exception as exc:
            print(f"[Reranker] WARNING — Could not load CrossEncoder: {exc}")
            print(f"[Reranker] Falling back to RRF score ordering.")
            self._model = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int | None = None,
    ) -> List[RetrievedChunk]:
        """
        Score each candidate against the query and return top_k.

        Parameters
        ----------
        query      : The original query string.
        candidates : Merged + deduplicated chunks from dense + sparse.
        top_k      : How many to return (default: retrieval.rerank_top_k from config).

        Returns
        -------
        List[RetrievedChunk] sorted by descending rerank_score.
        """
        cfg = self.config["retrieval"]
        if top_k is None:
            top_k = cfg.get("rerank_top_k", 5)

        if not candidates:
            return []

        self._load_model()

        if self._model is None:
            # Fallback: sort by RRF score and truncate
            for c in candidates:
                c.rerank_score = c.rrf_score
            return sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:top_k]

        # Build (query, passage) pairs for batch scoring
        pairs = [(query, chunk.text) for chunk in candidates]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            print(f"[Reranker] Prediction error: {exc}. Falling back to RRF scores.")
            for c in candidates:
                c.rerank_score = c.rrf_score
            return sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:top_k]

        # Assign scores back to chunks
        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)

        # Sort descending and take top_k
        reranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return reranked[:top_k]
