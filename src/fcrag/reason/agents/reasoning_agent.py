"""
fcrag/reason/agents/reasoning_agent.py -- FCRAG 2.0 Reasoning Agent
====================================================================
Builds a detailed prompt from anomaly event + retrieved context,
calls the LLM (via FCRAGLLMClient), and parses the response into
structured causal chains, claims, citations, and corrective actions.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.reason.state import (
    FCRAGState,
    CausalNode,
    Claim,
    Citation,
    CorrectiveAction,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RCA_PROMPT_TEMPLATE = """You are a 5G telecom network fault analysis expert. Analyze the following network anomaly and provide a structured Root Cause Analysis.

## Anomaly Event
- Cell ID: {cell_id}
- Fault Type: {fault_type}
- Severity: {severity}
- KPI Deviations: {kpi_summary}
- Anomaly Score: {anomaly_score}

## Retrieved Context
{context_block}

## Instructions
Based on the above context, provide:
1. A concise RCA summary (2-3 sentences)
2. A causal chain showing: Symptom -> Trigger -> Root Cause
3. Specific corrective actions with 3GPP spec references
4. Key factual claims that can be verified

Format your response as follows:

### RCA Summary
[Your summary here]

### Causal Chain
- Symptom: [observed symptom]
- Trigger: [triggering condition]
- Root Cause: [fundamental cause with spec reference]

### Corrective Actions
1. [Action with spec reference]
2. [Action with spec reference]

### Key Claims
- [Verifiable factual claim 1]
- [Verifiable factual claim 2]
"""


def _build_context_block(contexts: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    if not contexts:
        return "(No context retrieved)"

    lines = []
    for i, ctx in enumerate(contexts[:10], 1):  # Limit to 10 chunks
        source = ctx.get("source_file", "unknown")
        clause = ctx.get("clause_id", "")
        collection = ctx.get("collection", "")
        text = ctx.get("text", "")[:500]  # Truncate long chunks

        header = f"[{i}] {collection}/{source}"
        if clause:
            header += f" clause {clause}"
        lines.append(f"{header}\n{text}\n")

    return "\n".join(lines)


def _build_kpi_summary(kpi_deltas: dict) -> str:
    """Format KPI deltas into a human-readable summary."""
    parts = []
    for key, value in kpi_deltas.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "No KPI data"


def _parse_llm_response(
    response: str,
    contexts: list[dict],
    fault_type: str,
) -> dict[str, Any]:
    """
    Parse the LLM response into structured data.
    Falls back to basic extraction if the response doesn't follow the template.
    """
    rca_summary = ""
    causal_chain = []
    corrective_actions = []
    claims = []
    citations = []

    # --- Extract RCA Summary ---
    summary_match = re.search(
        r'###?\s*RCA\s*Summary\s*\n(.*?)(?=\n###|\Z)',
        response, re.DOTALL | re.IGNORECASE,
    )
    if summary_match:
        rca_summary = summary_match.group(1).strip()
    else:
        # Fallback: use first 2 sentences
        sentences = re.split(r'[.!?]\s+', response)
        rca_summary = ". ".join(sentences[:2]).strip()
        if rca_summary and not rca_summary.endswith("."):
            rca_summary += "."

    # --- Extract Causal Chain ---
    chain_match = re.search(
        r'###?\s*Causal\s*Chain\s*\n(.*?)(?=\n###|\Z)',
        response, re.DOTALL | re.IGNORECASE,
    )
    if chain_match:
        chain_text = chain_match.group(1)
        # Parse "- Symptom: ...", "- Trigger: ...", "- Root Cause: ..."
        for label in ["Symptom", "Trigger", "Root Cause"]:
            match = re.search(rf'-\s*{label}\s*:\s*(.*?)(?=\n-|\Z)', chain_text, re.DOTALL)
            if match:
                cause_text = match.group(1).strip()
                # Try to extract spec reference
                spec_ref = ""
                spec_match = re.search(r'(TS\s*[\d.]+(?:\s*(?:section|clause)?\s*[\d.]+)*)', cause_text, re.IGNORECASE)
                if spec_match:
                    spec_ref = spec_match.group(1)

                causal_chain.append(CausalNode(
                    node=label.upper().replace(" ", "_"),
                    cause=cause_text[:200],
                    evidence=spec_ref,
                ).to_dict())

    if not causal_chain:
        # Fallback: create a basic chain
        causal_chain = [
            CausalNode(node="SYMPTOM", cause=f"{fault_type} detected", evidence="").to_dict(),
            CausalNode(node="ROOT_CAUSE", cause=rca_summary[:100] if rca_summary else fault_type, evidence="").to_dict(),
        ]

    # --- Extract Corrective Actions ---
    actions_match = re.search(
        r'###?\s*Corrective\s*Actions?\s*\n(.*?)(?=\n###|\Z)',
        response, re.DOTALL | re.IGNORECASE,
    )
    if actions_match:
        action_lines = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', actions_match.group(1), re.DOTALL)
        for i, action_text in enumerate(action_lines, 1):
            action_text = action_text.strip()
            spec_ref = ""
            spec_match = re.search(r'(TS\s*[\d.]+(?:\s*(?:section|clause)?\s*[\d.]+)*)', action_text, re.IGNORECASE)
            if spec_match:
                spec_ref = spec_match.group(1)
            corrective_actions.append(CorrectiveAction(
                priority=i,
                action=action_text[:200],
                spec_reference=spec_ref,
            ).to_dict())

    if not corrective_actions:
        corrective_actions = [
            CorrectiveAction(
                priority=1,
                action=f"Review {fault_type} configuration parameters",
                spec_reference="",
            ).to_dict()
        ]

    # --- Extract Claims ---
    claims_match = re.search(
        r'###?\s*Key\s*Claims?\s*\n(.*?)(?=\n###|\Z)',
        response, re.DOTALL | re.IGNORECASE,
    )
    if claims_match:
        claim_lines = re.findall(r'-\s*(.*?)(?=\n-|\Z)', claims_match.group(1), re.DOTALL)
        for claim_text in claim_lines:
            claim_text = claim_text.strip()
            if len(claim_text.split()) >= 3:  # At least 3 words
                claims.append(Claim(text=claim_text[:300]).to_dict())

    if not claims:
        # Auto-extract claims from RCA summary
        if rca_summary:
            for sentence in re.split(r'[.!?]\s+', rca_summary):
                sentence = sentence.strip()
                if len(sentence.split()) >= 5:
                    claims.append(Claim(text=sentence).to_dict())

    # --- Extract Citations from context ---
    for ctx in contexts[:5]:
        source = ctx.get("source_file", "")
        clause = ctx.get("clause_id", "")
        if source or clause:
            ref = source
            if clause:
                ref += f" section {clause}"
            citations.append(Citation(
                spec_reference=ref,
                chunk_text=ctx.get("text", "")[:100],
                collection=ctx.get("collection", ""),
            ).to_dict())

    return {
        "rca_summary": rca_summary,
        "causal_chain": causal_chain,
        "corrective_actions": corrective_actions,
        "claims": claims,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def reason(state: FCRAGState) -> dict[str, Any]:
    """
    LangGraph node: Generate Root Cause Analysis using LLM.

    Reads:  state["anomaly_event"], state["fault_type"],
            state["retrieved_contexts"]
    Writes: rca_summary, causal_chain, claims, citations,
            corrective_actions, latency_breakdown
    """
    t_start = time.perf_counter()

    anomaly_event = state.get("anomaly_event", {})
    fault_type = state.get("fault_type", "UNKNOWN")
    contexts = state.get("retrieved_contexts", [])
    errors = list(state.get("errors", []))

    # Build the prompt
    prompt = RCA_PROMPT_TEMPLATE.format(
        cell_id=anomaly_event.get("cell_id", "unknown"),
        fault_type=fault_type,
        severity=anomaly_event.get("severity", "UNKNOWN"),
        kpi_summary=_build_kpi_summary(anomaly_event.get("kpi_deltas", {})),
        anomaly_score=anomaly_event.get("anomaly_score", 0.0),
        context_block=_build_context_block(contexts),
    )

    # Generate response via LLM
    from fcrag.reason.llm_client import FCRAGLLMClient

    try:
        client = FCRAGLLMClient()
        llm_response = client.generate(prompt)
    except Exception as exc:
        errors.append(f"ReasoningAgent: LLM generation failed: {exc}")
        llm_response = ""

    # Parse structured output
    parsed = _parse_llm_response(llm_response, contexts, fault_type)

    t_elapsed = (time.perf_counter() - t_start) * 1000
    latency = dict(state.get("latency_breakdown", {}))
    latency["reason_ms"] = t_elapsed

    print(
        f"[ReasoningAgent] Fault={fault_type} | "
        f"Claims={len(parsed['claims'])} | "
        f"Actions={len(parsed['corrective_actions'])} | "
        f"{t_elapsed:.0f}ms"
    )

    return {
        **parsed,
        "latency_breakdown": latency,
        "errors": errors,
    }
