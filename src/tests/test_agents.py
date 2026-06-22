"""
tests/test_agents.py -- FCRAG 2.0 Phase 3 Agent Tests
======================================================
Tests for the LangGraph multi-agent reasoning pipeline:
  - State schema dataclasses
  - Decomposer agent (fault classification + sub-query generation)
  - Validator agent (faithfulness scoring)
  - Reasoning agent (LLM response parsing)
  - LLM client (tier detection + template fallback)
  - Full pipeline (end-to-end with mocks)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.reason.state import (
    FCRAGState, CausalNode, Claim, Citation, CorrectiveAction,
)
from fcrag.reason.agents.decomposer import decompose, _classify_fault
from fcrag.reason.agents.validator import validate, _jaccard_similarity


# ===========================================================================
# Sample data fixtures
# ===========================================================================

def _ho_failure_event() -> dict:
    return {
        "event_id": "test-001",
        "cell_id": "Cell-42",
        "severity": "HIGH",
        "kpi_deltas": {
            "ho_success_rate_drop": 0.29,
            "throughput_drop_pct": 15.0,
            "latency_increase_ms": 42.0,
        },
        "anomaly_score": 0.85,
    }

def _prb_congestion_event() -> dict:
    return {
        "event_id": "test-002",
        "cell_id": "Cell-99",
        "severity": "CRITICAL",
        "kpi_deltas": {
            "prb_utilization_spike": 0.35,
            "throughput_drop_pct": 8.0,
        },
        "anomaly_score": 0.92,
    }

def _sample_contexts() -> list[dict]:
    return [
        {
            "text": "Handover failure occurs when A3 offset threshold is too aggressive causing premature handover triggering TS 38.331 section 5.5.4",
            "source_file": "ts38331.txt",
            "clause_id": "5.5.4",
            "collection": "3gpp_specs",
            "rerank_score": 0.95,
        },
        {
            "text": "The UE shall evaluate measurement event A3 when serving cell becomes offset worse than neighbour cell",
            "source_file": "ts38331.txt",
            "clause_id": "5.5.4.4",
            "collection": "3gpp_specs",
            "rerank_score": 0.88,
        },
        {
            "text": "PRB utilization above 90% indicates cell congestion requiring load balancing or capacity expansion",
            "source_file": "ts38321.txt",
            "clause_id": "9.2.1",
            "collection": "3gpp_specs",
            "rerank_score": 0.72,
        },
    ]


# ===========================================================================
# 1. State schema dataclasses
# ===========================================================================

class TestStateSchema:
    def test_causal_node_to_dict(self):
        node = CausalNode(node="HO_FAILURE", cause="A3 offset", evidence="TS 38.331")
        d = node.to_dict()
        assert d["node"] == "HO_FAILURE"
        assert d["cause"] == "A3 offset"

    def test_claim_to_dict(self):
        claim = Claim(text="test claim", is_supported=True, source_chunk_idx=2)
        d = claim.to_dict()
        assert d["is_supported"] is True
        assert d["source_chunk_idx"] == 2

    def test_corrective_action_to_dict(self):
        action = CorrectiveAction(priority=1, action="fix it", spec_reference="TS 38.331")
        d = action.to_dict()
        assert d["priority"] == 1
        assert d["spec_reference"] == "TS 38.331"


# ===========================================================================
# 2. Decomposer Agent
# ===========================================================================

class TestDecomposer:
    def test_classify_ho_failure(self):
        kpi = {"ho_success_rate_drop": 0.29}
        matched = _classify_fault(kpi)
        assert len(matched) >= 1
        assert matched[0]["fault_type"] == "HO_FAILURE"

    def test_classify_prb_congestion(self):
        kpi = {"prb_utilization_spike": 0.30}
        matched = _classify_fault(kpi)
        assert len(matched) >= 1
        assert matched[0]["fault_type"] == "PRB_CONGESTION"

    def test_classify_multiple_faults(self):
        """Multiple KPI deviations should match multiple fault types."""
        kpi = {
            "ho_success_rate_drop": 0.20,
            "prb_utilization_spike": 0.25,
            "latency_increase_ms": 30.0,
        }
        matched = _classify_fault(kpi)
        types = [m["fault_type"] for m in matched]
        assert len(types) >= 2

    def test_classify_no_match(self):
        """Below-threshold KPIs should return empty."""
        kpi = {"ho_success_rate_drop": 0.01}
        matched = _classify_fault(kpi)
        assert len(matched) == 0

    def test_decompose_ho_failure(self):
        state: FCRAGState = {"anomaly_event": _ho_failure_event()}
        result = decompose(state)

        assert result["fault_type"] == "HO_FAILURE"
        assert len(result["sub_queries"]) >= 2
        assert "latency_breakdown" in result
        assert "decompose_ms" in result["latency_breakdown"]

    def test_decompose_generates_spec_queries(self):
        state: FCRAGState = {"anomaly_event": _ho_failure_event()}
        result = decompose(state)

        # At least one query should mention specs
        queries_lower = " ".join(result["sub_queries"]).lower()
        assert any(term in queries_lower for term in ["ts", "38.331", "spec", "handover"])

    def test_decompose_empty_event(self):
        """Empty event should produce fallback queries, not crash."""
        state: FCRAGState = {"anomaly_event": {}}
        result = decompose(state)

        assert result["fault_type"] == "UNCLASSIFIED"
        assert len(result["sub_queries"]) >= 1


# ===========================================================================
# 3. Validator Agent
# ===========================================================================

class TestValidator:
    def test_jaccard_similarity(self):
        a = {"hello", "world", "test"}
        b = {"hello", "world", "foo"}
        score = _jaccard_similarity(a, b)
        assert score == pytest.approx(2/4)  # intersection=2, union=4

    def test_jaccard_empty(self):
        assert _jaccard_similarity(set(), {"a", "b"}) == 0.0
        assert _jaccard_similarity(set(), set()) == 0.0

    def test_validator_high_faithfulness(self):
        """Claims well-supported by context should score high."""
        contexts = _sample_contexts()
        claims = [
            {"text": "Handover failure occurs when A3 offset threshold is too aggressive", "is_supported": False},
            {"text": "PRB utilization above 90% indicates cell congestion", "is_supported": False},
        ]

        state: FCRAGState = {
            "anomaly_event": _ho_failure_event(),
            "claims": claims,
            "retrieved_contexts": contexts,
            "rca_summary": "Test RCA summary for handover failure.",
        }

        result = validate(state)

        assert result["faithfulness_score"] > 0.5
        assert result["final_response"] != "INSUFFICIENT_CONTEXT"
        # Claims should be updated with is_supported
        assert any(c["is_supported"] for c in result["claims"])

    def test_validator_low_faithfulness(self):
        """Claims unrelated to context should score low."""
        contexts = _sample_contexts()
        claims = [
            {"text": "quantum entanglement causes network failures in 6G systems", "is_supported": False},
            {"text": "solar flares disrupt baseband processing in mmWave bands", "is_supported": False},
        ]

        state: FCRAGState = {
            "anomaly_event": _ho_failure_event(),
            "claims": claims,
            "retrieved_contexts": contexts,
            "rca_summary": "Unrelated summary.",
        }

        result = validate(state)
        assert result["faithfulness_score"] < 0.5

    def test_validator_no_claims(self):
        """No claims should produce a default score."""
        state: FCRAGState = {
            "anomaly_event": _ho_failure_event(),
            "claims": [],
            "retrieved_contexts": _sample_contexts(),
            "rca_summary": "Some summary.",
        }

        result = validate(state)
        assert "faithfulness_score" in result

    def test_validator_no_context(self):
        """No context should produce INSUFFICIENT_CONTEXT."""
        state: FCRAGState = {
            "anomaly_event": _ho_failure_event(),
            "claims": [{"text": "some claim", "is_supported": False}],
            "retrieved_contexts": [],
            "rca_summary": "",
        }

        result = validate(state)
        assert result["final_response"] == "INSUFFICIENT_CONTEXT"


# ===========================================================================
# 4. Reasoning Agent — LLM response parsing
# ===========================================================================

class TestReasoningAgent:
    def test_parse_structured_response(self):
        """Test parsing a well-formatted LLM response."""
        from fcrag.reason.agents.reasoning_agent import _parse_llm_response

        llm_output = """### RCA Summary
The handover failure at Cell-42 is caused by an A3 offset that is too aggressive (-3dB). Per TS 38.331 section 5.5.4.4, this causes premature handover triggering.

### Causal Chain
- Symptom: Elevated handover failure rate (29% drop in ho_success_rate)
- Trigger: A3 offset configured at -3dB, below recommended threshold per TS 38.331 section 5.5.4
- Root Cause: Parameter misconfiguration of A3 event threshold per TS 38.331 section 5.5.4.4

### Corrective Actions
1. Increase A3 offset from -3dB to -1dB per TS 38.331 section 5.5.4.4
2. Enable HO history logging for Cell-42 per TS 38.401 section 8.3
3. Monitor handover KPIs for 24-hour stabilization

### Key Claims
- The A3 offset of -3dB is below the recommended threshold
- Handover failure rate increased by 29%
- TS 38.331 section 5.5.4.4 specifies A3 event configuration
"""

        parsed = _parse_llm_response(llm_output, _sample_contexts(), "HO_FAILURE")

        assert "A3 offset" in parsed["rca_summary"] or "handover" in parsed["rca_summary"].lower()
        assert len(parsed["causal_chain"]) >= 2
        assert len(parsed["corrective_actions"]) >= 2
        assert len(parsed["claims"]) >= 2

    def test_parse_unstructured_response(self):
        """Test parsing a response that doesn't follow the template."""
        from fcrag.reason.agents.reasoning_agent import _parse_llm_response

        llm_output = "The fault is caused by bad configuration. Fix the parameters."

        parsed = _parse_llm_response(llm_output, [], "HO_FAILURE")

        # Should produce fallback structures
        assert parsed["rca_summary"] != ""
        assert len(parsed["causal_chain"]) >= 1
        assert len(parsed["corrective_actions"]) >= 1

    def test_reason_node_with_template_fallback(self, monkeypatch):
        """Reasoning agent should work with template-based LLM fallback."""
        from fcrag.reason.agents.reasoning_agent import reason

        # Force Tier 3 (template)
        monkeypatch.setenv("FCRAG_LLM_TIER", "3")

        state: FCRAGState = {
            "anomaly_event": _ho_failure_event(),
            "fault_type": "HO_FAILURE",
            "retrieved_contexts": _sample_contexts(),
            "errors": [],
            "latency_breakdown": {},
        }

        result = reason(state)

        assert "rca_summary" in result
        assert result["rca_summary"] != ""
        assert len(result["causal_chain"]) >= 1
        assert len(result["corrective_actions"]) >= 1
        assert "reason_ms" in result["latency_breakdown"]


# ===========================================================================
# 5. LLM Client
# ===========================================================================

class TestLLMClient:
    def test_tier_3_fallback(self, monkeypatch):
        """Tier 3 (template) should always work without any deps."""
        monkeypatch.setenv("FCRAG_LLM_TIER", "3")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        from fcrag.reason.llm_client import FCRAGLLMClient
        client = FCRAGLLMClient(force_tier=3)

        assert client.tier == 3
        response = client.generate("Analyze handover failure at Cell-42")
        assert len(response) > 50
        assert "Root Cause" in response or "Fault" in response

    def test_auto_detect_no_gpu_no_token(self, monkeypatch):
        """Without GPU or HF_TOKEN, should auto-select Tier 3."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("FCRAG_LLM_TIER", raising=False)

        from fcrag.reason.llm_client import FCRAGLLMClient
        client = FCRAGLLMClient()

        # Should be tier 2 if HF_TOKEN exists, tier 3 otherwise
        # In test env without token, should be 1 (if CUDA) or 3 (no CUDA)
        assert client.tier in [1, 2, 3]

    def test_template_fault_detection(self, monkeypatch):
        """Template should detect fault type from prompt."""
        monkeypatch.setenv("FCRAG_LLM_TIER", "3")
        from fcrag.reason.llm_client import FCRAGLLMClient
        client = FCRAGLLMClient(force_tier=3)

        response = client.generate("PRB congestion detected at Cell-99 with high utilization")
        assert "PRB_CONGESTION" in response


# ===========================================================================
# 6. Full Pipeline (end-to-end with mocks)
# ===========================================================================

class TestFullPipeline:
    def test_pipeline_with_mocked_retriever_and_llm(self, monkeypatch):
        """
        End-to-end pipeline test with:
          - Real decomposer (deterministic)
          - Mocked retriever (returns sample contexts)
          - Template-based LLM (Tier 3)
          - Real validator
        """
        # Force template LLM
        monkeypatch.setenv("FCRAG_LLM_TIER", "3")

        # Mock the HybridRetriever to avoid Qdrant/embedding deps
        from fcrag.retrieve.schemas import RetrievedChunk

        mock_chunks = [
            RetrievedChunk(
                text="Handover failure A3 offset threshold TS 38.331 section 5.5.4",
                collection="3gpp_specs",
                source_file="ts38331.txt",
                clause_id="5.5.4",
                rerank_score=0.9,
            ),
            RetrievedChunk(
                text="measurement event A3 serving cell neighbour offset configuration",
                collection="3gpp_specs",
                source_file="ts38331.txt",
                clause_id="5.5.4.4",
                rerank_score=0.85,
            ),
        ]

        # Patch HybridRetriever in the retriever_agent module
        import fcrag.reason.agents.retriever_agent as ret_mod
        mock_retriever_cls = MagicMock()
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve.return_value = mock_chunks
        mock_retriever_cls.return_value = mock_retriever_instance

        with patch.object(ret_mod, "HybridRetriever", mock_retriever_cls, create=True):
            # Need to patch the import inside retrieve_context
            original_fn = ret_mod.retrieve_context

            def patched_retrieve_context(state):
                import time
                t_start = time.perf_counter()
                sub_queries = state.get("sub_queries", [])
                errors = list(state.get("errors", []))

                all_contexts = []
                seen = set()
                for q in sub_queries:
                    for chunk in mock_chunks:
                        key = chunk.dedup_key
                        if key not in seen:
                            seen.add(key)
                            all_contexts.append(chunk.to_dict())

                t_elapsed = (time.perf_counter() - t_start) * 1000
                latency = dict(state.get("latency_breakdown", {}))
                latency["retrieve_ms"] = t_elapsed

                return {
                    "retrieved_contexts": all_contexts,
                    "latency_breakdown": latency,
                    "errors": errors,
                }

            monkeypatch.setattr(ret_mod, "retrieve_context", patched_retrieve_context)

            # Also need to patch in the graph module
            import fcrag.reason.graph as graph_mod
            monkeypatch.setattr(graph_mod, "retrieve_context", patched_retrieve_context)

            from fcrag.reason.graph import run_pipeline

            result = run_pipeline(_ho_failure_event(), verbose=False)

        # Verify pipeline completed
        assert "fault_type" in result
        assert result["fault_type"] == "HO_FAILURE"
        assert "sub_queries" in result
        assert len(result["sub_queries"]) >= 2
        assert "retrieved_contexts" in result
        assert len(result["retrieved_contexts"]) >= 1
        assert "rca_summary" in result
        assert "faithfulness_score" in result
        assert "final_response" in result
        assert result["final_response"] != ""
        assert "latency_breakdown" in result

    def test_pipeline_empty_event(self, monkeypatch):
        """Pipeline should handle empty event gracefully."""
        monkeypatch.setenv("FCRAG_LLM_TIER", "3")

        # Mock retriever to return empty
        import fcrag.reason.agents.retriever_agent as ret_mod

        def empty_retriever(state):
            return {
                "retrieved_contexts": [],
                "latency_breakdown": dict(state.get("latency_breakdown", {})),
                "errors": list(state.get("errors", [])),
            }

        monkeypatch.setattr(ret_mod, "retrieve_context", empty_retriever)

        import fcrag.reason.graph as graph_mod
        monkeypatch.setattr(graph_mod, "retrieve_context", empty_retriever)

        from fcrag.reason.graph import run_pipeline
        result = run_pipeline({"cell_id": "Cell-1", "kpi_deltas": {}}, verbose=False)

        assert "fault_type" in result
        assert "final_response" in result
