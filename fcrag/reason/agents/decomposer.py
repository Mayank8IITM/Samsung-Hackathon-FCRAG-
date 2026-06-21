"""
fcrag/reason/agents/decomposer.py -- FCRAG 2.0 Decomposer Agent
================================================================
Classifies the fault type from an anomaly event's KPI deltas and
generates targeted sub-queries for hybrid retrieval.

This is a pure-logic node (no LLM needed) -- uses deterministic
rules based on telecom domain knowledge to map KPI signatures
to fault categories and generate appropriate search queries.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fcrag.reason.state import FCRAGState


# ---------------------------------------------------------------------------
# Fault classification rules
# ---------------------------------------------------------------------------

# Maps KPI delta keys to fault types and their query templates
FAULT_RULES = [
    {
        "fault_type": "HO_FAILURE",
        "trigger_kpis": ["ho_success_rate_drop"],
        "threshold": 0.1,  # >= 10% drop triggers this
        "spec_query": "handover failure measurement event A3 offset threshold configuration TS 38.331",
        "operational_query": "handover failure root cause corrective action parameter optimization",
        "simu5g_query": "handover failure scenario cell mobility",
    },
    {
        "fault_type": "PRB_CONGESTION",
        "trigger_kpis": ["prb_utilization_spike"],
        "threshold": 0.15,
        "spec_query": "physical resource block allocation scheduling PRB utilization TS 38.214",
        "operational_query": "PRB congestion mitigation load balancing cell capacity",
        "simu5g_query": "resource block congestion high load scenario",
    },
    {
        "fault_type": "THROUGHPUT_DEGRADATION",
        "trigger_kpis": ["throughput_drop_pct"],
        "threshold": 10.0,  # >= 10% throughput drop
        "spec_query": "throughput degradation CQI modulation coding scheme MCS TS 38.214",
        "operational_query": "throughput drop root cause interference signal quality degradation",
        "simu5g_query": "throughput degradation scenario DL throughput",
    },
    {
        "fault_type": "RRC_FAILURE",
        "trigger_kpis": ["rrc_retry_increase"],
        "threshold": 5.0,  # >= 5 extra retries
        "spec_query": "RRC connection setup failure reject retry TS 38.331",
        "operational_query": "RRC failure connection reject preamble RACH congestion",
        "simu5g_query": "RRC connection failure scenario",
    },
    {
        "fault_type": "LATENCY_SPIKE",
        "trigger_kpis": ["latency_increase_ms"],
        "threshold": 20.0,  # >= 20ms increase
        "spec_query": "user plane latency DRX scheduling request timing advance TS 38.321",
        "operational_query": "latency spike root cause backhaul scheduling delay",
        "simu5g_query": "latency increase delay scenario",
    },
]


def _classify_fault(kpi_deltas: dict[str, float]) -> list[dict]:
    """
    Match KPI deltas against fault rules.
    Returns all matching rules, sorted by normalized severity (value / threshold).
    """
    matches = []
    for rule in FAULT_RULES:
        for kpi_key in rule["trigger_kpis"]:
            value = abs(kpi_deltas.get(kpi_key, 0.0))
            if value >= rule["threshold"]:
                # Normalize severity so different units (ms vs %) are comparable
                normalized_severity = value / rule["threshold"]
                matches.append({**rule, "_severity": normalized_severity})
                break

    # Sort by normalized severity descending
    matches.sort(key=lambda m: m["_severity"], reverse=True)
    return matches


def _build_sub_queries(
    fault_type: str,
    matched_rules: list[dict],
    anomaly_event: dict,
    max_queries: int = 4,
) -> list[str]:
    """Build targeted sub-queries from matched fault rules."""
    queries = []
    cell_id = anomaly_event.get("cell_id", "")

    for rule in matched_rules[:2]:  # Top 2 fault types max
        # 1. Spec-focused query
        queries.append(rule["spec_query"])

        # 2. Operational query
        if cell_id:
            queries.append(f"{rule['operational_query']} {cell_id}")
        else:
            queries.append(rule["operational_query"])

        # 3. Simu5G query (if available)
        if len(queries) < max_queries:
            queries.append(rule["simu5g_query"])

    return queries[:max_queries]


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def decompose(state: FCRAGState) -> dict[str, Any]:
    """
    LangGraph node: Decompose an anomaly event into sub-queries.

    Reads:  state["anomaly_event"]
    Writes: fault_type, sub_queries, latency_breakdown
    """
    t_start = time.perf_counter()

    anomaly_event = state.get("anomaly_event", {})
    kpi_deltas = anomaly_event.get("kpi_deltas", {})
    errors = list(state.get("errors", []))

    # Classify fault
    matched = _classify_fault(kpi_deltas)

    if matched:
        fault_type = matched[0]["fault_type"]
    else:
        fault_type = "UNCLASSIFIED"
        errors.append("Decomposer: No fault rules matched KPI deltas")

    # Generate sub-queries
    max_queries = 4  # From config: agents.max_sub_queries
    sub_queries = _build_sub_queries(fault_type, matched, anomaly_event, max_queries)

    if not sub_queries:
        # Fallback: generate a generic query from the event
        cell_id = anomaly_event.get("cell_id", "unknown")
        sub_queries = [
            f"network fault {fault_type} root cause analysis {cell_id}",
            f"3GPP specification {fault_type} troubleshooting procedure",
        ]

    t_elapsed = (time.perf_counter() - t_start) * 1000
    latency = dict(state.get("latency_breakdown", {}))
    latency["decompose_ms"] = t_elapsed

    print(f"[Decomposer] Fault: {fault_type} | Queries: {len(sub_queries)} | {t_elapsed:.0f}ms")

    return {
        "fault_type": fault_type,
        "sub_queries": sub_queries,
        "latency_breakdown": latency,
        "errors": errors,
    }
