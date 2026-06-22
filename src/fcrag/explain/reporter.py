"""
fcrag/explain/reporter.py -- FCRAG 2.0 Output Reporter
======================================================
Takes the final LangGraph state and packages it into the exact
JSON Output Package contract defined in the system design.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.reason.state import FCRAGState
from fcrag.explain.causal_graph import build_causal_graph


def build_output_package(state: FCRAGState) -> dict[str, Any]:
    """
    Build the final RCA Output Package from the pipeline state.

    Parameters
    ----------
    state : FCRAGState dictionary populated by the LangGraph pipeline

    Returns
    -------
    dict : The final output package matching the API contract
    """
    anomaly_event = state.get("anomaly_event", {})
    
    # Extract latency
    latency_breakdown = state.get("latency_breakdown", {})
    total_latency_ms = sum(latency_breakdown.values())

    # Build the Causal Graph data structure
    causal_chain = state.get("causal_chain", [])
    graph_data = build_causal_graph(causal_chain)

    # Format citations as simple strings for the final payload
    citations_raw = state.get("citations", [])
    formatted_citations = []
    for c in citations_raw:
        ref = c.get("spec_reference", "")
        if ref and ref not in formatted_citations:
            formatted_citations.append(ref)

    # Overall confidence (heuristic combining anomaly score and faithfulness)
    anomaly_score = anomaly_event.get("anomaly_score", 0.5)
    faithfulness = state.get("faithfulness_score", 0.0)
    confidence = (anomaly_score * 0.4) + (faithfulness * 0.6)

    # Final response status
    final_response = state.get("final_response", "")
    if final_response == "INSUFFICIENT_CONTEXT":
        status = "INSUFFICIENT_CONTEXT"
        rca_summary = "Could not generate a reliable RCA due to insufficient retrieved context."
    elif "PIPELINE_ERROR" in final_response:
        status = "ERROR"
        rca_summary = "An error occurred during pipeline execution."
    else:
        status = "RCA_COMPLETE"
        rca_summary = state.get("rca_summary", "")

    return {
        "status": status,
        "rca_summary": rca_summary,
        "causal_chain": causal_chain,
        "causal_graph": graph_data,
        "corrective_actions": state.get("corrective_actions", []),
        "citations": formatted_citations,
        "faithfulness_score": faithfulness,
        "confidence": min(confidence, 1.0),
        "latency_ms": int(total_latency_ms),
        "errors": state.get("errors", []),
    }
