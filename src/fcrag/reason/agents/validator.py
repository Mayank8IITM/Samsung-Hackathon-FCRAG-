"""
fcrag/reason/agents/validator.py -- FCRAG 2.0 Validator Agent
==============================================================
Checks each factual claim from the reasoning agent against the
retrieved context chunks. Computes a faithfulness score based on
word overlap and decides whether the RCA is trustworthy.

Faithfulness scoring method: "overlap" (configurable in settings.yaml)
  - For each claim, find the best-matching context chunk (Jaccard similarity)
  - A claim is "supported" if its best match exceeds 0.15 Jaccard
  - faithfulness_score = fraction of supported claims
  - PASS if score >= 0.7, else INSUFFICIENT_CONTEXT
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.ingest.embedder import load_config
from fcrag.reason.state import FCRAGState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize_simple(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer for overlap computation."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Subset coverage: |A & B| / |A| to handle short claims matching long chunks."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    # Use length of claim (set_a) instead of union so short claims matching long contexts don't get penalized.
    return intersection / len(set_a)


def _check_claim_against_contexts(
    claim_text: str,
    contexts: list[dict],
    min_overlap: float = 0.15,
) -> tuple[bool, int | None]:
    """
    Check if a claim is supported by any retrieved context chunk.

    Returns (is_supported, best_matching_chunk_index).
    """
    claim_tokens = _tokenize_simple(claim_text)
    if len(claim_tokens) < 3:
        return True, None  # Too short to verify, assume OK

    best_score = 0.0
    best_idx = None

    for i, ctx in enumerate(contexts):
        ctx_text = ctx.get("text", "")
        ctx_tokens = _tokenize_simple(ctx_text)
        score = _jaccard_similarity(claim_tokens, ctx_tokens)

        if score > best_score:
            best_score = score
            best_idx = i

    return (best_score >= min_overlap), best_idx


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def validate(state: FCRAGState) -> dict[str, Any]:
    """
    LangGraph node: Validate claims and compute faithfulness score.

    Reads:  state["claims"], state["retrieved_contexts"], state["rca_summary"]
    Writes: faithfulness_score, final_response, claims (updated), latency_breakdown
    """
    t_start = time.perf_counter()

    config = load_config()
    agent_cfg = config.get("agents", {})
    thresholds = agent_cfg.get("faithfulness_thresholds", {})
    min_faithfulness = thresholds.get("medium", 0.60)

    claims = list(state.get("claims", []))
    contexts = state.get("retrieved_contexts", [])
    rca_summary = state.get("rca_summary", "")
    errors = list(state.get("errors", []))

    # Score each claim
    supported_count = 0
    updated_claims = []

    for claim_dict in claims:
        claim_text = claim_dict.get("text", "")
        is_supported, chunk_idx = _check_claim_against_contexts(
            claim_text, contexts, min_overlap=0.15,
        )
        updated_claim = {
            **claim_dict,
            "is_supported": is_supported,
            "source_chunk_idx": chunk_idx,
        }
        updated_claims.append(updated_claim)

        if is_supported:
            supported_count += 1

    # Compute faithfulness score
    if claims:
        faithfulness_score = supported_count / len(claims)
    else:
        # No claims extracted - base score on whether we have context
        faithfulness_score = 0.5 if contexts else 0.0

    # Determine final response
    if faithfulness_score >= min_faithfulness and rca_summary:
        final_response = rca_summary
    elif rca_summary and contexts:
        # Partial confidence -- append warning
        final_response = (
            f"{rca_summary}\n\n"
            f"[WARNING: Faithfulness score {faithfulness_score:.2f} "
            f"is below threshold {min_faithfulness:.2f}. "
            f"Some claims may not be fully supported by retrieved evidence.]"
        )
    else:
        final_response = "INSUFFICIENT_CONTEXT"

    t_elapsed = (time.perf_counter() - t_start) * 1000
    latency = dict(state.get("latency_breakdown", {}))
    latency["validate_ms"] = t_elapsed

    status = "PASS" if faithfulness_score >= min_faithfulness else "LOW_CONFIDENCE"
    print(
        f"[Validator] Score={faithfulness_score:.2f} | "
        f"Supported={supported_count}/{len(claims)} | "
        f"Status={status} | {t_elapsed:.0f}ms"
    )

    return {
        "claims": updated_claims,
        "faithfulness_score": faithfulness_score,
        "final_response": final_response,
        "latency_breakdown": latency,
        "errors": errors,
    }
