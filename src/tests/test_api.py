"""
tests/test_api.py -- Tests for FastAPI endpoints
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_fault_endpoint(monkeypatch):
    """
    Test the /analyze-fault endpoint end-to-end with the FCRAG LLM tier set to 3.
    Mocks the retriever to avoid needing Qdrant/BM25 setup during fast API tests.
    """
    monkeypatch.setenv("FCRAG_LLM_TIER", "3")
    
    # Mock HybridRetriever so we don't need real indexes
    from fcrag.retrieve.schemas import RetrievedChunk
    mock_chunks = [
        RetrievedChunk(
            text="Handover failure A3 offset threshold TS 38.331 section 5.5.4",
            collection="3gpp_specs",
            source_file="ts38331.txt",
            clause_id="5.5.4",
            rerank_score=0.9,
        )
    ]
    
    import fcrag.reason.agents.retriever_agent as ret_mod
    
    def patched_retrieve_context(state):
        import time
        latency = dict(state.get("latency_breakdown", {}))
        latency["retrieve_ms"] = 5
        return {
            "retrieved_contexts": [c.to_dict() for c in mock_chunks],
            "latency_breakdown": latency,
            "errors": list(state.get("errors", [])),
        }
        
    monkeypatch.setattr(ret_mod, "retrieve_context", patched_retrieve_context)
    
    import fcrag.reason.graph as graph_mod
    monkeypatch.setattr(graph_mod, "retrieve_context", patched_retrieve_context)

    payload = {
        "cell_id": "Cell-42",
        "severity": "HIGH",
        "kpi_snapshot": {
            "ho_success_rate_drop": 0.29,
            "latency_increase_ms": 10.0
        },
        "mode": "auto",
        "anomaly_score": 0.85
    }

    response = client.post("/analyze-fault", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] in ["RCA_COMPLETE", "LOW_CONFIDENCE", "INSUFFICIENT_CONTEXT"]
    assert "rca_summary" in data
    assert "causal_chain" in data
    assert "causal_graph" in data
    assert "nodes" in data["causal_graph"]  # NetworkX JSON format
    assert "links" in data["causal_graph"] or "edges" in data["causal_graph"]
    assert "latency_ms" in data
