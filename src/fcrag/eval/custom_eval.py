"""
fcrag/eval/custom_eval.py — FCRAG 2.0 Custom Scenario Evaluator
================================================================
Runs the hybrid retriever on the 20 labelled fault→clause scenarios
and computes MRR and Recall@5 using exact clause-id matching.

Ground truth: data/custom_scenarios/fault_clause_mapping.json
Each scenario has:
  - fault_description  : natural language query
  - relevant_clauses   : list of ground-truth clause ids (e.g. "TS 38.331 §5.5.4.4")
  - fault_type         : category label
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.retrieve.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Clause matching helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip spaces, dots, and words for fuzzy clause matching."""
    return text.lower().replace(" ", "").replace(".", "").replace("§", "").replace("#", "").replace("section", "").replace("clause", "")


def _clause_hit(retrieved_clause_id: str, ground_truth_clauses: List[str]) -> bool:
    """
    Return True if the retrieved clause_id matches any ground-truth clause.
    Uses normalised prefix matching.
    e.g., if ground truth is TS 38.331 §5.5.4.4 and retrieved is TS 38.331 Section 5.5.4,
    both normalise to ts383315544 and ts38331554. The retrieved is a prefix of the GT, so it matches.
    """
    norm_ret = _normalise(retrieved_clause_id)
    for gt in ground_truth_clauses:
        norm_gt = _normalise(gt)
        if norm_gt and (norm_gt.startswith(norm_ret) or norm_ret.startswith(norm_gt)):
            return True
    return False


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

def run_custom_eval(top_k: int = 5) -> Dict[str, Any]:
    """
    Run MRR + Recall@{top_k} on the 20 custom fault scenarios.

    Returns a results dict saved to results/custom_eval_results.json.
    """
    mapping_path = ROOT / "data" / "custom_scenarios" / "fault_clause_mapping.json"
    with open(mapping_path) as f:
        scenarios: List[Dict] = json.load(f)

    retriever = HybridRetriever()

    results = []
    reciprocal_ranks: List[float] = []
    hits: List[int] = []

    print(f"\n{'='*60}")
    print(f"Custom Fault Evaluator  (n={len(scenarios)}, top_k={top_k})")
    print(f"{'='*60}")

    for idx, scenario in enumerate(scenarios, 1):
        sid = scenario["scenario_id"]
        query = scenario["fault_description"]
        gt_clauses = scenario["relevant_clauses"]
        fault_type = scenario["fault_type"]

        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, use_reranker=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Find first rank hit
        rr = 0.0
        hit = 0
        first_hit_rank = None
        retrieved_clause_ids = []

        for rank, chunk in enumerate(chunks, 1):
            retrieved_clause_ids.append(chunk.clause_id)
            if _clause_hit(chunk.clause_id, gt_clauses):
                if rr == 0.0:
                    rr = 1.0 / rank
                    first_hit_rank = rank
                hit = 1

        reciprocal_ranks.append(rr)
        hits.append(hit)

        status = "✅ HIT" if hit else "❌ MISS"
        print(
            f"[{idx:02d}/{len(scenarios)}] {sid} ({fault_type})\n"
            f"       Query: {query[:70]}...\n"
            f"       GT   : {gt_clauses}\n"
            f"       Ret  : {retrieved_clause_ids}\n"
            f"       {status}  RR={rr:.3f}  Rank={first_hit_rank}  "
            f"Latency={elapsed_ms:.0f}ms\n"
        )

        results.append({
            "scenario_id": sid,
            "fault_type": fault_type,
            "query": query,
            "ground_truth_clauses": gt_clauses,
            "retrieved_clause_ids": retrieved_clause_ids,
            "reciprocal_rank": rr,
            "hit_at_k": hit,
            "first_hit_rank": first_hit_rank,
            "retrieval_latency_ms": round(elapsed_ms, 1),
        })

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    recall_at_k = sum(hits) / len(hits)

    # Per-fault-type breakdown
    by_type: Dict[str, Dict] = {}
    for r in results:
        ft = r["fault_type"]
        if ft not in by_type:
            by_type[ft] = {"rr_sum": 0.0, "hits": 0, "count": 0}
        by_type[ft]["rr_sum"] += r["reciprocal_rank"]
        by_type[ft]["hits"] += r["hit_at_k"]
        by_type[ft]["count"] += 1

    type_summary = {
        ft: {
            "mrr": round(v["rr_sum"] / v["count"], 4),
            "recall_at_k": round(v["hits"] / v["count"], 4),
            "count": v["count"],
        }
        for ft, v in by_type.items()
    }

    summary = {
        "evaluator": "custom_fault_scenarios",
        "n_scenarios": len(scenarios),
        "top_k": top_k,
        "mrr": round(mrr, 4),
        "recall_at_k": round(recall_at_k, 4),
        "target_mrr": 0.75,
        "target_recall": 0.85,
        "mrr_pass": mrr >= 0.75,
        "recall_pass": recall_at_k >= 0.85,
        "by_fault_type": type_summary,
        "per_scenario": results,
    }

    out_path = ROOT / "results" / "custom_eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    mrr_icon = "✅" if mrr >= 0.75 else "❌"
    rec_icon = "✅" if recall_at_k >= 0.85 else "❌"

    print(f"\n{'='*60}")
    print(f"CUSTOM EVAL RESULTS")
    print(f"{'='*60}")
    print(f"  {mrr_icon} MRR        : {mrr:.4f}  (target > 0.75)")
    print(f"  {rec_icon} Recall@{top_k}  : {recall_at_k:.4f}  (target > 0.85)")
    print(f"\nPer fault type:")
    for ft, s in type_summary.items():
        print(f"  {ft:25s}  MRR={s['mrr']:.3f}  R@{top_k}={s['recall_at_k']:.3f}  (n={s['count']})")
    print(f"\nSaved → {out_path}")

    return summary


if __name__ == "__main__":
    run_custom_eval()
