"""
tests/test_retrieval.py — FCRAG 2.0 Phase 2 Retrieval Tests
============================================================
Tests for the hybrid retrieval pipeline:
  - RetrievedChunk schema helpers
  - RRF fusion math
  - SparseRetriever (BM25, using mock pkl)
  - CrossEncoderReranker fallback path
  - HybridRetriever end-to-end (real BM25 index, mocked Qdrant)
"""

import json
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fcrag.retrieve.schemas import RetrievedChunk
from fcrag.retrieve.retriever import reciprocal_rank_fusion


# ===========================================================================
# 1. RetrievedChunk schema
# ===========================================================================

class TestRetrievedChunk:
    def test_dedup_key_with_ids(self):
        c = RetrievedChunk(
            text="hello",
            collection="3gpp_specs",
            source_file="TS38331.txt",
            clause_id="5.5.4.4",
        )
        assert c.dedup_key == "3gpp_specs::TS38331.txt::5.5.4.4"

    def test_dedup_key_fallback_text(self):
        c = RetrievedChunk(text="some text", collection="col")
        assert "col::" in c.dedup_key
        assert "some text" in c.dedup_key

    def test_to_dict_roundtrip(self):
        c = RetrievedChunk(
            text="spec text",
            collection="3gpp_specs",
            source_file="doc.txt",
            clause_id="1.2.3",
            source_type="3gpp",
            dense_score=0.9,
            bm25_score=5.2,
            rrf_score=0.02,
            rerank_score=1.5,
            metadata={"page": 42},
        )
        d = c.to_dict()
        assert d["text"] == "spec text"
        assert d["dense_score"] == 0.9
        assert d["page"] == 42

    def test_from_dict(self):
        d = {
            "text": "test",
            "source_file": "f.txt",
            "clause_id": "7",
            "source_type": "oran",
            "extra_field": "extra_value",
        }
        c = RetrievedChunk.from_dict(d, collection="oran_specs")
        assert c.text == "test"
        assert c.collection == "oran_specs"
        assert c.metadata.get("extra_field") == "extra_value"


# ===========================================================================
# 2. Reciprocal Rank Fusion
# ===========================================================================

class TestRRF:
    def _make_chunks(self, texts, collection="col"):
        return [
            RetrievedChunk(text=t, collection=collection, clause_id=str(i))
            for i, t in enumerate(texts)
        ]

    def test_single_list_preserves_order(self):
        chunks = self._make_chunks(["a", "b", "c"])
        result = reciprocal_rank_fusion(chunks, k=60)
        # Best rank = rank 1 → highest score
        assert result[0].text == "a"

    def test_two_lists_boost_common_item(self):
        """A chunk appearing at rank 1 in both lists should outrank one only in one list."""
        dense = self._make_chunks(["X", "Y", "Z"])
        sparse = self._make_chunks(["X", "A", "B"])
        result = reciprocal_rank_fusion(dense, sparse, k=60)
        # "X" is rank-1 in both, should be first
        assert result[0].text == "X"

    def test_rrf_scores_positive(self):
        chunks = self._make_chunks(["p", "q"])
        result = reciprocal_rank_fusion(chunks, k=60)
        for c in result:
            assert c.rrf_score > 0

    def test_rrf_deduplication(self):
        """Same dedup_key appearing in two lists should produce one output chunk."""
        chunk_a = RetrievedChunk(text="shared", collection="c", source_file="f.txt", clause_id="1")
        chunk_b = RetrievedChunk(text="shared", collection="c", source_file="f.txt", clause_id="1")
        result = reciprocal_rank_fusion([chunk_a], [chunk_b], k=60)
        assert len(result) == 1
        # Score should be sum of both contributions
        assert result[0].rrf_score == pytest.approx(2 * (1.0 / (60 + 1)), rel=1e-5)

    def test_rrf_k_parameter(self):
        """Lower k → higher scores (RRF math: 1/(k+rank) increases as k decreases)."""
        # Verify the math directly without object mutation side-effects
        # rank=1: 1/(k+1) — smaller k gives bigger score
        score_k60 = 1.0 / (60 + 1)
        score_k1  = 1.0 / (1 + 1)
        assert score_k1 > score_k60

        # Verify via actual RRF function with fresh chunk lists each time
        # Use DISTINCT clause_ids so they are not deduped across the two calls
        chunks_a = [
            RetrievedChunk(text="a", collection="col", clause_id="k60_0"),
            RetrievedChunk(text="b", collection="col", clause_id="k60_1"),
        ]
        chunks_b = [
            RetrievedChunk(text="a", collection="col", clause_id="k1_0"),
            RetrievedChunk(text="b", collection="col", clause_id="k1_1"),
        ]
        result_k60 = reciprocal_rank_fusion(chunks_a, k=60)
        result_k1  = reciprocal_rank_fusion(chunks_b, k=1)
        assert result_k1[0].rrf_score > result_k60[0].rrf_score

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([], [], k=60)
        assert result == []


# ===========================================================================
# 3. SparseRetriever
# ===========================================================================

class TestSparseRetriever:
    def _make_bm25_pkl(self, tmp_path: Path, collection: str) -> Path:
        """Build a real (tiny) BM25 index and save it."""
        from rank_bm25 import BM25Okapi
        from fcrag.ingest.bm25_builder import tokenize

        corpus = [
            {"text": "Handover failure A3 offset TS 38.331", "source_file": "ts38331.txt", "clause_id": "5.5.4"},
            {"text": "PRB congestion Physical Resource Block utilization", "source_file": "ts38321.txt", "clause_id": "9.2.1"},
            {"text": "PRACH preamble collision rate increased", "source_file": "ts38321.txt", "clause_id": "9.3.5"},
        ]
        tokenized = [tokenize(d["text"]) for d in corpus]
        model = BM25Okapi(tokenized)
        payloads = [{"id": i, **d} for i, d in enumerate(corpus)]
        bundle = {"model": model, "payloads": payloads, "collection_name": collection}

        pkl_path = tmp_path / f"bm25_{collection}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(bundle, f)
        return pkl_path

    def test_sparse_retriever_keyword_match(self, tmp_path, monkeypatch):
        """BM25 should rank the handover document highest for an HO query."""
        import fcrag.retrieve.sparse as sparse_mod
        monkeypatch.setattr(sparse_mod, "INDEX_DIR", tmp_path)

        self._make_bm25_pkl(tmp_path, "3gpp_specs")

        from fcrag.retrieve.sparse import SparseRetriever
        sr = SparseRetriever()
        results = sr.retrieve("handover failure A3", "3gpp_specs", top_k=3)

        assert len(results) > 0
        assert results[0].clause_id == "5.5.4", "HO document should rank first"
        assert results[0].bm25_score > 0

    def test_sparse_retriever_missing_index_returns_empty(self, tmp_path, monkeypatch):
        """Missing pkl → empty list, no crash."""
        import fcrag.retrieve.sparse as sparse_mod
        monkeypatch.setattr(sparse_mod, "INDEX_DIR", tmp_path)

        from fcrag.retrieve.sparse import SparseRetriever
        sr = SparseRetriever()
        results = sr.retrieve("any query", "nonexistent_collection", top_k=5)
        assert results == []

    def test_sparse_retriever_empty_query(self, tmp_path, monkeypatch):
        """Empty / whitespace-only query → empty list, no crash."""
        import fcrag.retrieve.sparse as sparse_mod
        monkeypatch.setattr(sparse_mod, "INDEX_DIR", tmp_path)
        self._make_bm25_pkl(tmp_path, "3gpp_specs")

        from fcrag.retrieve.sparse import SparseRetriever
        sr = SparseRetriever()
        results = sr.retrieve("", "3gpp_specs", top_k=5)
        assert results == []


# ===========================================================================
# 4. CrossEncoderReranker — fallback path (no model download in CI)
# ===========================================================================

class TestCrossEncoderReranker:
    def _make_chunks(self, n=5):
        return [
            RetrievedChunk(
                text=f"Document about topic {i}",
                collection="col",
                clause_id=str(i),
                rrf_score=1.0 / (i + 1),  # simulates RRF ordering
            )
            for i in range(n)
        ]

    def test_fallback_no_model(self, monkeypatch):
        """When CrossEncoder cannot be imported, should fall back to RRF order."""
        from fcrag.retrieve.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        reranker._model = None  # Force fallback

        # Patch _load_model to be a no-op (keeps _model as None)
        monkeypatch.setattr(reranker, "_load_model", lambda: None)

        chunks = self._make_chunks(5)
        result = reranker.rerank("some query", chunks, top_k=3)

        assert len(result) == 3
        # Should be sorted by rrf_score descending
        for a, b in zip(result, result[1:]):
            assert a.rrf_score >= b.rrf_score

    def test_reranker_returns_top_k(self, monkeypatch):
        """Should never return more than top_k results."""
        from fcrag.retrieve.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        reranker._model = None
        monkeypatch.setattr(reranker, "_load_model", lambda: None)

        chunks = self._make_chunks(10)
        result = reranker.rerank("query", chunks, top_k=4)
        assert len(result) == 4

    def test_reranker_empty_input(self, monkeypatch):
        """Empty candidates → empty output."""
        from fcrag.retrieve.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


# ===========================================================================
# 5. HybridRetriever — end-to-end with mocked dense, real sparse
# ===========================================================================

class TestHybridRetriever:
    def _make_bm25_pkl(self, tmp_path: Path, collection: str):
        from rank_bm25 import BM25Okapi
        from fcrag.ingest.bm25_builder import tokenize
        # Need 5+ documents so BM25 IDF doesn't collapse to 0
        # (with only 2 docs, IDF = log((2-1+0.5)/(1+0.5)) = 0)
        corpus = [
            {"text": "HO_FAILURE handover A3 offset Cell-42", "source_file": "ts38331.txt", "clause_id": "5.5.4"},
            {"text": "PRB Physical Resource Block congestion", "source_file": "ts38321.txt", "clause_id": "9.2"},
            {"text": "PRACH preamble collision rate increased", "source_file": "ts38321.txt", "clause_id": "9.3"},
            {"text": "RLF radio link failure timer T310 expiry", "source_file": "ts38331.txt", "clause_id": "5.3.11"},
            {"text": "CQI channel quality indicator wideband report", "source_file": "ts38214.txt", "clause_id": "5.2.2"},
        ]
        tokenized = [tokenize(d["text"]) for d in corpus]
        model = BM25Okapi(tokenized)
        payloads = [{"id": i, **d} for i, d in enumerate(corpus)]
        bundle = {"model": model, "payloads": payloads, "collection_name": collection}
        (tmp_path / f"bm25_{collection}.pkl").write_bytes(pickle.dumps(bundle))

    def _make_retriever(self, tmp_path, monkeypatch):
        """
        Create a HybridRetriever with DenseRetriever fully mocked (no Qdrant lock)
        and reranker in fallback mode.
        """
        # Patch INDEX_DIR for SparseRetriever
        import fcrag.retrieve.sparse as sparse_mod
        monkeypatch.setattr(sparse_mod, "INDEX_DIR", tmp_path)

        # Mock DenseRetriever to avoid Qdrant persistent file lock.
        # Must patch in retriever module since it does `from ..dense import DenseRetriever`
        import fcrag.retrieve.retriever as retriever_mod
        mock_dense = MagicMock()
        mock_dense.retrieve = MagicMock(return_value=[])
        monkeypatch.setattr(retriever_mod, "DenseRetriever", lambda: mock_dense)

        from fcrag.retrieve.retriever import HybridRetriever
        hr = HybridRetriever()

        # Force reranker fallback (no model download)
        hr.reranker._model = None
        monkeypatch.setattr(hr.reranker, "_load_model", lambda: None)

        return hr

    def test_hybrid_retrieve_returns_chunks(self, tmp_path, monkeypatch):
        """End-to-end test: sparse finds chunks, dense is mocked, reranker uses fallback."""
        self._make_bm25_pkl(tmp_path, "3gpp_specs")
        hr = self._make_retriever(tmp_path, monkeypatch)

        results = hr.retrieve(
            "handover failure A3 offset",
            collections=["3gpp_specs"],
            top_k=2,
        )

        assert isinstance(results, list)
        assert all(isinstance(c, RetrievedChunk) for c in results)
        assert len(results) > 0
        # The HO-related chunk should be in results (BM25 keyword match)
        assert any(
            "A3" in c.text or "handover" in c.text.lower() or "HO_FAILURE" in c.text
            for c in results
        )

    def test_search_text_returns_dicts(self, tmp_path, monkeypatch):
        """search_text() should return plain dicts."""
        self._make_bm25_pkl(tmp_path, "3gpp_specs")
        hr = self._make_retriever(tmp_path, monkeypatch)

        results = hr.search_text("A3 offset", collections=["3gpp_specs"], top_k=2)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)
            assert "text" in r

    def test_retrieve_for_fault(self, tmp_path, monkeypatch):
        """retrieve_for_fault() builds a composite query and returns chunks."""
        for coll in ["3gpp_specs", "simu5g_narratives", "alarm_history"]:
            self._make_bm25_pkl(tmp_path, coll)
        hr = self._make_retriever(tmp_path, monkeypatch)

        results = hr.retrieve_for_fault(
            fault_type="HO_FAILURE",
            cell_id="Cell-42",
            kpi_summary="ho_sr_dev=-3.2sigma, rlf_dev=+2.1sigma",
            top_k=3,
        )
        assert isinstance(results, list)
