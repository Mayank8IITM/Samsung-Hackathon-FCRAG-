"""
fcrag/retrieve/schemas.py — FCRAG 2.0 Retrieval Data Schemas
=============================================================
Shared dataclasses used across all retrieval modules so every
component speaks the same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """
    A single document chunk returned by any retrieval pathway.

    Scores are filled in progressively:
      - dense_score  : cosine similarity from Qdrant  (0–1)
      - bm25_score   : raw BM25 score (unbounded, relative)
      - rrf_score    : Reciprocal Rank Fusion combined score
      - rerank_score : cross-encoder logit (higher = more relevant)
    """
    # Core content
    text: str
    collection: str
    source_file: str = ""
    clause_id: str = ""
    source_type: str = ""

    # Scores (filled in by the component that produced them)
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    # Passthrough metadata (fault_id, cell, etc. from simu5g)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def dedup_key(self) -> str:
        """
        Stable identity key for deduplication across dense/sparse results.
        Falls back to the first 120 chars of text if no structured id.
        """
        if self.source_file and self.clause_id:
            return f"{self.collection}::{self.source_file}::{self.clause_id}"
        if self.source_file:
            return f"{self.collection}::{self.source_file}::{self.text[:120]}"
        return f"{self.collection}::{self.text[:120]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (useful for LangChain / JSON output)."""
        return {
            "text": self.text,
            "collection": self.collection,
            "source_file": self.source_file,
            "clause_id": self.clause_id,
            "source_type": self.source_type,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], collection: str = "") -> "RetrievedChunk":
        """Reconstruct from a plain dict (e.g. from BM25 payload or Qdrant payload)."""
        known_keys = {
            "text", "collection", "source_file", "clause_id", "source_type",
            "dense_score", "bm25_score", "rrf_score", "rerank_score",
        }
        meta = {k: v for k, v in d.items() if k not in known_keys}
        return cls(
            text=d.get("text", ""),
            collection=d.get("collection", collection),
            source_file=d.get("source_file", ""),
            clause_id=d.get("clause_id", ""),
            source_type=d.get("source_type", ""),
            dense_score=float(d.get("dense_score", 0.0)),
            bm25_score=float(d.get("bm25_score", 0.0)),
            rrf_score=float(d.get("rrf_score", 0.0)),
            rerank_score=float(d.get("rerank_score", 0.0)),
            metadata=meta,
        )
