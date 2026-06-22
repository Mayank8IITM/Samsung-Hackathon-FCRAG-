"""
fcrag/retrieve/dense.py — FCRAG 2.0 Dense Retriever
====================================================
Wraps the Qdrant vector database for dense (embedding-based) retrieval.

Key design decisions:
  - Uses the same FCRAGEmbeddings from Phase 1 (hits disk cache instantly).
  - Qdrant runs in persistent disk mode (data/qdrant_db/).
  - If a collection has 0 vectors it auto-populates by calling Indexer.
  - Returns list[RetrievedChunk] sorted by descending cosine score.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from fcrag.ingest.embedder import FCRAGEmbeddings, load_config
from fcrag.retrieve.schemas import RetrievedChunk


class DenseRetriever:
    """
    Performs dense (cosine similarity) retrieval against Qdrant collections.

    Usage
    -----
    >>> dr = DenseRetriever()
    >>> results = dr.retrieve("handover failure A3 offset", "3gpp_specs", top_k=10)
    """

    def __init__(self):
        self.config = load_config()
        self.qdrant_cfg = self.config["qdrant"]
        self.retrieval_cfg = self.config["retrieval"]

        self.embedder = FCRAGEmbeddings()
        self.client = self._init_client()

        # Track which collections we have already confirmed are populated
        self._populated: set[str] = set()

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #

    def _init_client(self) -> QdrantClient:
        in_memory = self.qdrant_cfg.get("in_memory", False)
        if in_memory:
            return QdrantClient(location=":memory:")
        persist_dir = ROOT / self.qdrant_cfg.get("persist_directory", "data/qdrant_db")
        persist_dir.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(persist_dir))

    def _ensure_collection(self, collection_name: str) -> bool:
        """
        Make sure the collection exists and has vectors.
        If not, triggers Indexer to populate it (lazy indexing).
        Returns True if the collection is ready to query.
        """
        if collection_name in self._populated:
            return True

        # Check existence
        exists = self.client.collection_exists(collection_name=collection_name)
        if exists:
            info = self.client.get_collection(collection_name=collection_name)
            count = info.points_count or 0
            if count > 0:
                self._populated.add(collection_name)
                return True

        # Collection missing or empty → trigger ingest
        print(f"[DenseRetriever] Collection '{collection_name}' empty. Populating via Indexer...")
        try:
            from fcrag.ingest.indexer import Indexer
            indexer = Indexer()
            # Pass our already-initialised client so we don't create a second DB
            indexer.client = self.client
            indexer.init_collections()
            count = indexer.index_collection(collection_name)
            if count > 0:
                self._populated.add(collection_name)
                return True
        except Exception as exc:
            print(f"[DenseRetriever] Auto-indexing failed for '{collection_name}': {exc}")
        return False

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
        Embed *query* and return the top_k most similar chunks from *collection*.

        Parameters
        ----------
        query      : Natural language query string.
        collection : Qdrant collection name (e.g. "3gpp_specs").
        top_k      : Number of results (default: retrieval.dense_top_k from config).

        Returns
        -------
        List[RetrievedChunk] sorted by descending dense_score.
        """
        if top_k is None:
            top_k = self.retrieval_cfg.get("dense_top_k", 20)

        if not self._ensure_collection(collection):
            print(f"[DenseRetriever] Skipping '{collection}' — collection unavailable.")
            return []

        # Embed query (hits cache if seen before)
        query_vector = self.embedder.embed_query(query)

        # Query Qdrant
        query_response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=self.retrieval_cfg.get("min_score_threshold", 0.0),
        )
        hits = query_response.points

        results: List[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = RetrievedChunk(
                text=payload.get("text", ""),
                collection=collection,
                source_file=payload.get("source_file", ""),
                clause_id=payload.get("clause_id", ""),
                source_type=payload.get("source_type", ""),
                dense_score=float(hit.score),
                metadata={k: v for k, v in payload.items()
                           if k not in {"text", "source_file", "clause_id", "source_type"}},
            )
            results.append(chunk)

        return results
