"""
fcrag/reason/state.py -- FCRAG 2.0 Shared State Schema
=======================================================
Defines the LangGraph shared state (TypedDict) that flows through
all agent nodes, plus supporting dataclasses for causal chains,
claims, and citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CausalNode:
    """One link in the causal chain: Symptom -> Trigger -> Root Cause."""
    node: str           # e.g. "HO_FAILURE"
    cause: str          # e.g. "A3_OFFSET_TOO_AGGRESSIVE"
    evidence: str       # e.g. "TS 38.331 section 5.5.4.4"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "cause": self.cause,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class Claim:
    """A single factual claim extracted from the reasoning agent's output."""
    text: str
    source_chunk_idx: int | None = None  # index into retrieved_contexts
    is_supported: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_chunk_idx": self.source_chunk_idx,
            "is_supported": self.is_supported,
        }


@dataclass
class Citation:
    """A reference to a specific source document / clause."""
    spec_reference: str   # e.g. "TS 38.331 section 5.5.4.4"
    chunk_text: str = ""  # the actual text snippet
    collection: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_reference": self.spec_reference,
            "chunk_text": self.chunk_text,
            "collection": self.collection,
        }


@dataclass
class CorrectiveAction:
    """A recommended corrective action with priority and spec reference."""
    priority: int
    action: str
    spec_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "action": self.action,
            "spec_reference": self.spec_reference,
        }


# ---------------------------------------------------------------------------
# LangGraph Shared State
# ---------------------------------------------------------------------------

class FCRAGState(TypedDict, total=False):
    """
    Shared state that flows through all LangGraph agent nodes.

    Fields are populated progressively:
      - decompose()       fills: fault_type, sub_queries
      - retrieve_context() fills: retrieved_contexts
      - reason()          fills: rca_summary, causal_chain, claims, citations,
                                 corrective_actions
      - validate()        fills: faithfulness_score, final_response
    """
    # Input (provided at pipeline start)
    anomaly_event: dict

    # Decomposer output
    fault_type: str                          # e.g. "HO_FAILURE"
    sub_queries: list[str]

    # Retriever output
    retrieved_contexts: list[dict]           # list of RetrievedChunk.to_dict()

    # Reasoning output
    rca_summary: str
    causal_chain: list[dict]                 # list of CausalNode.to_dict()
    claims: list[dict]                       # list of Claim.to_dict()
    citations: list[dict]                    # list of Citation.to_dict()
    corrective_actions: list[dict]           # list of CorrectiveAction.to_dict()

    # Validator output
    faithfulness_score: float
    final_response: str                      # full RCA or "INSUFFICIENT_CONTEXT"

    # Timing
    latency_breakdown: dict[str, float]      # node_name -> elapsed_ms

    # Error tracking
    errors: list[str]
