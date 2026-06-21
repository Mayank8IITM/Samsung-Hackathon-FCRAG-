"""
fcrag/eval/faithfulness_eval.py — FCRAG 2.0 Faithfulness Evaluator
====================================================================
Measures how grounded/faithful the LLM-generated RCA output is to
the retrieved context chunks.

Method: Word-level Jaccard overlap between LLM response and retrieved
context. This is a lightweight proxy that doesn't need an NLI model.

  faithfulness_score = |response_words ∩ context_words| / |response_words|

A score >= 0.5 means the model is mostly using words from the context.
A score < 0.3 is flagged as "hallucination risk".

Runs on all 20 custom fault scenarios and samples the LLM for each.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fcrag.retrieve.retriever import HybridRetriever
from fcrag.reason.llm_client import FCRAGLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _words(text: str, min_len: int = 3) -> set:
    """Extract unique meaningful words (len >= min_len)."""
    tokens = re.findall(r"[a-zA-Z0-9\-\.]+", text.lower())
    return {t for t in tokens if len(t) >= min_len}


def _faithfulness_score(response: str, context: str) -> float | None:
    """
    Jaccard-based faithfulness:
        score = |response_words ∩ context_words| / |response_words|
        
    Returns None if the response is a refusal (e.g. "insufficient data").
    """
    if "insufficient data" in response.lower():
        return None
        
    resp_words = _words(response)
    ctx_words = _words(context)
    if not resp_words:
        return 0.0
    overlap = resp_words & ctx_words
    return len(overlap) / len(resp_words)


def _build_rag_prompt(query: str, context_chunks: list) -> str:
    """Build the guardrail RAG prompt that instructs the model to use only context."""
    context_str = "\n\n".join(
        f"[Chunk {i+1}] (Source: {c.clause_id})\n{c.text}"
        for i, c in enumerate(context_chunks)
    )
    return (
        "You are a Telecom Network Fault RCA Assistant.\n"
        "Answer STRICTLY using the provided context. "
        "Do NOT use outside knowledge. "
        "If context is insufficient, say 'Insufficient data in the knowledge base. This prototype currently only supports TS 38.331, TS 38.300, TS 23.501, TS 23.502, TR 21.916, and TR 21.918.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Fault Query: {query}\n"
        "RCA Response:"
    )


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

def run_faithfulness_eval(top_k: int = 5, max_tokens: int = 200) -> Dict[str, Any]:
    """
    Run faithfulness evaluation on all 20 custom fault scenarios.

    For each: retrieve top-k chunks, generate RCA, compute faithfulness.
    Saves results to results/faithfulness_results.json.
    """
    mapping_path = ROOT / "data" / "custom_scenarios" / "fault_clause_mapping.json"
    with open(mapping_path) as f:
        scenarios: List[Dict] = json.load(f)

    retriever = HybridRetriever()
    llm = FCRAGLLMClient()

    print(f"\n{'='*60}")
    print(f"Faithfulness Evaluator  (n={len(scenarios)}, top_k={top_k})")
    print(f"{'='*60}")

    results = []
    scores: List[float] = []
    llm_latencies: List[float] = []
    retrieval_latencies: List[float] = []

    for idx, scenario in enumerate(scenarios, 1):
        sid = scenario["scenario_id"]
        query = scenario["fault_description"]
        fault_type = scenario["fault_type"]

        # --- Retrieve ---
        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, use_reranker=True)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        retrieval_latencies.append(retrieval_ms)

        context_str = " ".join(c.text for c in chunks)

        # --- Generate ---
        prompt = _build_rag_prompt(query, chunks)
        t1 = time.perf_counter()
        response = llm.generate(prompt, max_tokens=max_tokens)
        llm_ms = (time.perf_counter() - t1) * 1000
        llm_latencies.append(llm_ms)

        # --- Score ---
        score = _faithfulness_score(response, context_str)
        if score is not None:
            scores.append(score)

        risk = ""
        if score is None:
            risk = " ℹ️  GUARDRAIL ENGAGED (Ignored from score)"
        elif score < 0.3:
            risk = " ⚠️  HALLUCINATION RISK"
        elif score < 0.5:
            risk = " ⚠️  LOW FAITHFULNESS"

        score_str = f"{score:.3f}" if score is not None else "N/A"
        print(
            f"[{idx:02d}/{len(scenarios)}] {sid} ({fault_type})\n"
            f"       Query     : {query[:70]}\n"
            f"       Response  : {response[:100].strip()}...\n"
            f"       Faithfulness={score_str}{risk}\n"
            f"       Retrieval={retrieval_ms:.0f}ms  LLM={llm_ms:.0f}ms\n"
        )

        results.append({
            "scenario_id": sid,
            "fault_type": fault_type,
            "query": query,
            "response_excerpt": response[:300],
            "faithfulness_score": round(score, 4) if score is not None else None,
            "hallucination_risk": score is not None and score < 0.3,
            "low_faithfulness": score is not None and score < 0.5,
            "guardrail_engaged": score is None,
            "retrieval_latency_ms": round(retrieval_ms, 1),
            "llm_latency_ms": round(llm_ms, 1),
            "total_latency_ms": round(retrieval_ms + llm_ms, 1),
        })

    avg_score = sum(scores) / len(scores) if scores else 0.0
    hallu_count = sum(1 for s in scores if s < 0.3)
    low_faith_count = sum(1 for s in scores if 0.3 <= s < 0.5)
    pass_count = sum(1 for s in scores if s >= 0.5)
    guardrail_count = len(scenarios) - len(scores)

    sorted_scores = sorted(scores) if scores else [0.0]
    p50 = sorted_scores[len(sorted_scores) // 2]
    p10 = sorted_scores[max(0, int(0.10 * len(sorted_scores)))]

    avg_ret_ms = sum(retrieval_latencies) / len(retrieval_latencies)
    avg_llm_ms = sum(llm_latencies) / len(llm_latencies)

    summary = {
        "evaluator": "faithfulness",
        "n_scenarios": len(scenarios),
        "n_scored": len(scores),
        "n_guardrail": guardrail_count,
        "top_k": top_k,
        "avg_faithfulness_score": round(avg_score, 4),
        "median_faithfulness_score": round(p50, 4),
        "p10_faithfulness_score": round(p10, 4),
        "target_faithfulness": 0.90,
        "faithfulness_pass": avg_score >= 0.90,
        "pass_count": pass_count,
        "low_faithfulness_count": low_faith_count,
        "hallucination_risk_count": hallu_count,
        "avg_retrieval_latency_ms": round(avg_ret_ms, 1),
        "avg_llm_latency_ms": round(avg_llm_ms, 1),
        "avg_total_latency_ms": round(avg_ret_ms + avg_llm_ms, 1),
        "per_scenario": results,
    }

    out_path = ROOT / "results" / "faithfulness_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    faith_icon = "✅" if avg_score >= 0.90 else "❌"
    print(f"\n{'='*60}")
    print(f"FAITHFULNESS RESULTS")
    print(f"{'='*60}")
    print(f"  {faith_icon} Avg Faithfulness : {avg_score:.4f}  (target > 0.90)")
    print(f"     Median          : {p50:.4f}")
    print(f"     P10 (worst 10%) : {p10:.4f}")
    print(f"  ✅ Pass (>= 0.50)  : {pass_count}/{len(scores)} scored")
    print(f"  ⚠️  Low (<  0.50)  : {low_faith_count}/{len(scores)} scored")
    print(f"  ⚠️  Halluc (<0.30) : {hallu_count}/{len(scores)} scored")
    print(f"  ℹ️  Guardrail used : {guardrail_count}/{len(scenarios)} total")
    print(f"  ⏱  Avg Retrieval  : {avg_ret_ms:.0f}ms")
    print(f"  ⏱  Avg LLM        : {avg_llm_ms:.0f}ms")
    print(f"  ⏱  Avg Total      : {avg_ret_ms + avg_llm_ms:.0f}ms")
    print(f"\nSaved → {out_path}")

    return summary


if __name__ == "__main__":
    run_faithfulness_eval()
