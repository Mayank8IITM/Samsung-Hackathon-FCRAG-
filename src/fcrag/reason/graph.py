"""
fcrag/reason/graph.py -- FCRAG 2.0 LangGraph Pipeline
======================================================
Wires the 4 agent nodes into a LangGraph StateGraph:

  START -> decompose -> retrieve_context -> reason -> validate -> END

Provides `run_pipeline(anomaly_event)` as the single entry-point.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from langgraph.graph import StateGraph, END

from fcrag.reason.state import FCRAGState
from fcrag.reason.agents.decomposer import decompose
from fcrag.reason.agents.retriever_agent import retrieve_context
from fcrag.reason.agents.reasoning_agent import reason
from fcrag.reason.agents.validator import validate


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Build the FCRAG LangGraph StateGraph.

    Returns a compiled graph that can be invoked with:
        result = graph.invoke({"anomaly_event": {...}})
    """
    graph = StateGraph(FCRAGState)

    # Add nodes
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("reason", reason)
    graph.add_node("validate", validate)

    # Wire edges: linear pipeline
    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "retrieve_context")
    graph.add_edge("retrieve_context", "reason")
    graph.add_edge("reason", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    anomaly_event: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the full FCRAG reasoning pipeline on an anomaly event.

    Parameters
    ----------
    anomaly_event : Anomaly event dict with keys:
                    cell_id, severity, kpi_deltas, anomaly_score, etc.
    verbose       : If True, prints a summary after completion.

    Returns
    -------
    dict : Final FCRAGState with all fields populated:
           rca_summary, causal_chain, corrective_actions, citations,
           faithfulness_score, final_response, latency_breakdown, errors
    """
    t_start = time.perf_counter()

    graph = build_graph()

    # Initial state
    initial_state: FCRAGState = {
        "anomaly_event": anomaly_event,
        "errors": [],
        "latency_breakdown": {},
    }

    # Run the pipeline
    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        result = {
            **initial_state,
            "final_response": "PIPELINE_ERROR",
            "faithfulness_score": 0.0,
            "errors": [f"Pipeline error: {exc}"],
        }

    t_total = (time.perf_counter() - t_start) * 1000

    if verbose:
        latency = result.get("latency_breakdown", {})
        errors = result.get("errors", [])
        faith = result.get("faithfulness_score", 0.0)
        fault = result.get("fault_type", "?")

        print(f"\n{'='*60}")
        print(f"FCRAG Pipeline Complete")
        print(f"{'='*60}")
        print(f"  Fault Type:         {fault}")
        print(f"  Faithfulness:       {faith:.2f}")
        print(f"  Total Latency:      {t_total:.0f}ms")

        if latency:
            for key, val in latency.items():
                print(f"    {key}: {val:.0f}ms")

        if errors:
            print(f"  Errors ({len(errors)}):")
            for e in errors:
                print(f"    - {e}")

        response = result.get("final_response", "")
        if response and response != "INSUFFICIENT_CONTEXT":
            print(f"\n  RCA Summary:")
            for line in response.split("\n")[:5]:
                print(f"    {line}")

        print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo: run pipeline with a sample HO_FAILURE event
    sample_event = {
        "event_id": "demo-001",
        "timestamp": "2026-06-14T12:00:00Z",
        "cell_id": "Cell-42",
        "severity": "HIGH",
        "kpi_deltas": {
            "ho_success_rate_drop": 0.29,
            "throughput_drop_pct": 15.0,
            "latency_increase_ms": 42.0,
        },
        "anomaly_score": 0.85,
        "drift_detected": True,
    }

    result = run_pipeline(sample_event)
