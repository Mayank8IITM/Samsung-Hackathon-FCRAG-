"""
fcrag/eval/latency_bench.py — FCRAG 2.0 End-to-End Latency Benchmark
======================================================================
Runs 20 end-to-end fault analysis cycles (one per custom scenario) and
records the wall-clock time for each stage:

  Stage 1 — Retrieval  : HybridRetriever (dense + sparse + rerank)
  Stage 2 — LLM        : FCRAGLLMClient.generate()
  Stage 3 — Total      : retrieval + LLM (the critical path)

Reports: P50, P95, P99 per stage and overall pass/fail vs 4s target.

Latency is measured on the same machine the model runs — if you are
using HF Inference API (Tier 2), network round-trip is included in
the LLM latency figure.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fcrag.retrieve.retriever import HybridRetriever
from fcrag.reason.llm_client import FCRAGLLMClient


# ---------------------------------------------------------------------------
# Guardrail prompt builder (same as faithfulness_eval)
# ---------------------------------------------------------------------------

def _build_prompt(query: str, chunks: list) -> str:
    ctx = "\n\n".join(
        f"[Chunk {i+1}] (Source: {c.clause_id})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return (
        "You are a Telecom Network Fault RCA Assistant.\n"
        "Answer STRICTLY using the provided context. "
        "Do NOT use outside knowledge. "
        "If context is insufficient, say 'Insufficient data in the knowledge base. This prototype currently only supports TS 38.331, TS 38.300, TS 23.501, TS 23.502, TR 21.916, and TR 21.918.'\n\n"
        f"Context:\n{ctx}\n\n"
        f"Fault Query: {query}\n"
        "RCA Response:"
    )


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentile(data: List[float], pct: float) -> float:
    """Return the p-th percentile (0-100) of sorted data."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(pct / 100 * len(s)), len(s) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------

def run_latency_bench(top_k: int = 5, max_tokens: int = 200) -> Dict[str, Any]:
    """
    Run end-to-end latency benchmark across all 20 custom scenarios.
    Saves results to results/latency_results.json.
    """
    mapping_path = ROOT / "data" / "custom_scenarios" / "fault_clause_mapping.json"
    with open(mapping_path) as f:
        scenarios: List[Dict] = json.load(f)

    retriever = HybridRetriever()
    llm = FCRAGLLMClient()

    n = len(scenarios)
    print(f"\n{'='*60}")
    print(f"End-to-End Latency Benchmark  (n={n} runs, top_k={top_k})")
    print(f"LLM Tier: {llm._tier_name()}")
    print(f"{'='*60}")
    print(f"{'Run':>4}  {'Fault Type':25}  {'Retrieval':>12}  {'LLM':>10}  {'Total':>10}")
    print(f"{'-'*4}  {'-'*25}  {'-'*12}  {'-'*10}  {'-'*10}")

    run_results: List[Dict] = []
    retrieval_times: List[float] = []
    llm_times: List[float] = []
    total_times: List[float] = []

    for idx, scenario in enumerate(scenarios, 1):
        query = scenario["fault_description"]
        fault_type = scenario["fault_type"]
        sid = scenario["scenario_id"]

        # --- Stage 1: Retrieval ---
        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, use_reranker=True)
        retrieval_s = time.perf_counter() - t0

        # --- Stage 2: LLM generation ---
        prompt = _build_prompt(query, chunks)
        t1 = time.perf_counter()
        _response = llm.generate(prompt, max_tokens=max_tokens)
        llm_s = time.perf_counter() - t1

        total_s = retrieval_s + llm_s

        retrieval_ms = retrieval_s * 1000
        llm_ms = llm_s * 1000
        total_ms = total_s * 1000

        retrieval_times.append(retrieval_ms)
        llm_times.append(llm_ms)
        total_times.append(total_ms)

        target_flag = "✅" if total_s < 4.0 else "❌"
        print(
            f"{idx:>4}  {fault_type:25}  "
            f"{retrieval_ms:>9.0f}ms  "
            f"{llm_ms:>7.0f}ms  "
            f"{total_ms:>7.0f}ms  {target_flag}"
        )

        run_results.append({
            "run": idx,
            "scenario_id": sid,
            "fault_type": fault_type,
            "query": query,
            "retrieval_ms": round(retrieval_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "total_ms": round(total_ms, 1),
            "meets_4s_target": total_s < 4.0,
        })

    # --- Compute percentiles ---
    def _stats(times: List[float]) -> Dict:
        return {
            "mean_ms": round(statistics.mean(times), 1),
            "median_ms": round(statistics.median(times), 1),
            "p50_ms": round(_percentile(times, 50), 1),
            "p95_ms": round(_percentile(times, 95), 1),
            "p99_ms": round(_percentile(times, 99), 1),
            "min_ms": round(min(times), 1),
            "max_ms": round(max(times), 1),
        }

    ret_stats = _stats(retrieval_times)
    llm_stats = _stats(llm_times)
    total_stats = _stats(total_times)

    passes_target = sum(1 for r in run_results if r["meets_4s_target"])
    overall_pass = total_stats["p50_ms"] < 4000

    summary = {
        "evaluator": "latency_benchmark",
        "n_runs": n,
        "top_k": top_k,
        "llm_tier": llm._tier_name(),
        "target_latency_ms": 4000,
        "runs_meeting_target": passes_target,
        "runs_failing_target": n - passes_target,
        "overall_pass": overall_pass,
        "retrieval_latency": ret_stats,
        "llm_latency": llm_stats,
        "total_latency": total_stats,
        "per_run": run_results,
    }

    out_path = ROOT / "results" / "latency_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    lat_icon = "✅" if overall_pass else "❌"
    print(f"\n{'='*60}")
    print(f"LATENCY BENCHMARK RESULTS  (LLM: {llm._tier_name()})")
    print(f"{'='*60}")
    print(f"  Stage          |  Mean    |  P50     |  P95     |  P99")
    print(f"  {'-'*60}")
    print(f"  Retrieval      | {ret_stats['mean_ms']:>7.0f}ms | {ret_stats['p50_ms']:>7.0f}ms | {ret_stats['p95_ms']:>7.0f}ms | {ret_stats['p99_ms']:>7.0f}ms")
    print(f"  LLM Generate   | {llm_stats['mean_ms']:>7.0f}ms | {llm_stats['p50_ms']:>7.0f}ms | {llm_stats['p95_ms']:>7.0f}ms | {llm_stats['p99_ms']:>7.0f}ms")
    print(f"  Total (E2E)    | {total_stats['mean_ms']:>7.0f}ms | {total_stats['p50_ms']:>7.0f}ms | {total_stats['p95_ms']:>7.0f}ms | {total_stats['p99_ms']:>7.0f}ms")
    print(f"\n  {lat_icon} E2E P50 vs 4s target: {total_stats['p50_ms']:.0f}ms  ({'PASS' if overall_pass else 'FAIL'})")
    print(f"  Runs meeting < 4s: {passes_target}/{n}")
    print(f"\nSaved → {out_path}")

    return summary


if __name__ == "__main__":
    run_latency_bench()
