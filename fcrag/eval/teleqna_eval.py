"""
fcrag/eval/teleqna_eval.py — FCRAG 2.0 TeleQnA Evaluator
==========================================================
Evaluates retrieval quality using the TeleQnA benchmark dataset.

Filters to "Standards specifications" and "Standards overview" categories
(as per settings.yaml). Samples 500 questions for practical runtime.

Metric:
  - MRR      : Mean Reciprocal Rank (is the answer in top-k? at what rank?)
  - Recall@5 : Was ANY top-5 chunk useful for answering the question?

"Useful" is determined by keyword overlap: if the correct answer's key
terms appear in any retrieved chunk, that chunk counts as a hit.
This is a proven proxy for retrieval quality without needing manual labels.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fcrag.retrieve.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

TARGET_CATEGORIES = {"standards specifications", "standards overview"}


def load_teleqna(
    path: Path,
    categories: set | None = None,
    sample: int = 500,
    seed: int = 42,
) -> List[Dict]:
    """
    Load TeleQnA from the .txt (JSON) file and filter by category.

    Returns a list of dicts:
      {question, options, answer, answer_text, category}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    cats = categories or TARGET_CATEGORIES
    filtered = []

    for key, item in raw.items():
        cat = item.get("category", "").strip().lower()
        if cat not in cats:
            continue

        # Parse options (option 1 ... option 5)
        options = {}
        for k, v in item.items():
            if k.startswith("option"):
                options[k] = v

        # Parse correct answer text (strip "option N: " prefix)
        raw_answer = item.get("answer", "")
        answer_text = re.sub(r"^option\s*\d+:\s*", "", raw_answer, flags=re.IGNORECASE).strip()

        filtered.append({
            "id": key,
            "question": item.get("question", ""),
            "options": options,
            "answer": raw_answer,
            "answer_text": answer_text,
            "category": item.get("category", ""),
        })

    random.seed(seed)
    if len(filtered) > sample:
        filtered = random.sample(filtered, sample)

    return filtered


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

def _keywords(text: str, min_len: int = 4) -> set:
    """Extract meaningful words (len >= min_len) from text."""
    words = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return {w for w in words if len(w) >= min_len}


def _chunk_relevant(chunk_text: str, answer_text: str, threshold: float = 0.15) -> bool:
    """
    Return True if the chunk text has sufficient keyword overlap with
    the correct answer.

    Jaccard(answer_keywords ∩ chunk_keywords) >= threshold
    """
    ans_kws = _keywords(answer_text)
    if not ans_kws:
        return False
    chunk_kws = _keywords(chunk_text)
    overlap = ans_kws & chunk_kws
    jaccard = len(overlap) / len(ans_kws)
    return jaccard >= threshold


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

def run_teleqna_eval(
    sample: int = 500,
    top_k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run MRR + Recall@{top_k} on sampled TeleQnA standards questions.

    Returns results dict saved to results/teleqna_eval_results.json.
    """
    data_path = ROOT / "data" / "teleqna" / "TeleQnA.txt"
    questions = load_teleqna(data_path, sample=sample, seed=seed)

    print(f"\n{'='*60}")
    print(f"TeleQnA Evaluator  (n={len(questions)}, top_k={top_k})")
    print(f"Categories: {TARGET_CATEGORIES}")
    print(f"{'='*60}")

    retriever = HybridRetriever()

    results = []
    reciprocal_ranks: List[float] = []
    hits: List[int] = []
    latencies: List[float] = []
    category_stats: Dict[str, Dict] = {}

    for idx, q in enumerate(questions, 1):
        query = q["question"]
        answer_text = q["answer_text"]
        category = q["category"]

        t0 = time.perf_counter()
        # Use no reranker for speed across 500 questions;
        # RRF score is sufficient for retrieval evaluation
        chunks = retriever.retrieve(query, top_k=top_k, use_reranker=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        rr = 0.0
        hit = 0
        first_hit_rank: Optional[int] = None

        for rank, chunk in enumerate(chunks, 1):
            if _chunk_relevant(chunk.text, answer_text):
                if rr == 0.0:
                    rr = 1.0 / rank
                    first_hit_rank = rank
                hit = 1

        reciprocal_ranks.append(rr)
        hits.append(hit)

        # Per-category tracking
        if category not in category_stats:
            category_stats[category] = {"rr_sum": 0.0, "hits": 0, "count": 0}
        category_stats[category]["rr_sum"] += rr
        category_stats[category]["hits"] += hit
        category_stats[category]["count"] += 1

        if idx % 50 == 0 or idx == len(questions):
            running_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            running_rec = sum(hits) / len(hits)
            print(
                f"  [{idx:3d}/{len(questions)}]  "
                f"Running MRR={running_mrr:.4f}  "
                f"Recall@{top_k}={running_rec:.4f}  "
                f"Latency={elapsed_ms:.0f}ms"
            )

        results.append({
            "id": q["id"],
            "category": category,
            "question": query[:120],
            "answer_text": answer_text[:120],
            "reciprocal_rank": rr,
            "hit_at_k": hit,
            "first_hit_rank": first_hit_rank,
            "retrieval_latency_ms": round(elapsed_ms, 1),
        })

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    recall_at_k = sum(hits) / len(hits)
    avg_latency = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[int(0.50 * len(sorted_lat))]
    p95 = sorted_lat[int(0.95 * len(sorted_lat))]

    cat_summary = {
        cat: {
            "mrr": round(v["rr_sum"] / v["count"], 4),
            "recall_at_k": round(v["hits"] / v["count"], 4),
            "count": v["count"],
        }
        for cat, v in category_stats.items()
    }

    summary = {
        "evaluator": "teleqna",
        "n_questions": len(questions),
        "top_k": top_k,
        "categories": list(TARGET_CATEGORIES),
        "mrr": round(mrr, 4),
        "recall_at_k": round(recall_at_k, 4),
        "target_mrr": 0.75,
        "target_recall": 0.85,
        "mrr_pass": mrr >= 0.75,
        "recall_pass": recall_at_k >= 0.85,
        "avg_retrieval_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "by_category": cat_summary,
        "per_question": results,
    }

    out_path = ROOT / "results" / "teleqna_eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    mrr_icon = "✅" if mrr >= 0.75 else "❌"
    rec_icon = "✅" if recall_at_k >= 0.85 else "❌"

    print(f"\n{'='*60}")
    print(f"TELEQNA EVAL RESULTS")
    print(f"{'='*60}")
    print(f"  {mrr_icon} MRR        : {mrr:.4f}  (target > 0.75)")
    print(f"  {rec_icon} Recall@{top_k}  : {recall_at_k:.4f}  (target > 0.85)")
    print(f"  ⏱  Avg Retrieval Latency : {avg_latency:.0f}ms")
    print(f"  ⏱  P50={p50:.0f}ms  P95={p95:.0f}ms")
    print(f"\nPer category:")
    for cat, s in cat_summary.items():
        print(f"  {cat:30s}  MRR={s['mrr']:.3f}  R@{top_k}={s['recall_at_k']:.3f}  (n={s['count']})")
    print(f"\nSaved → {out_path}")

    return summary


if __name__ == "__main__":
    run_teleqna_eval()
