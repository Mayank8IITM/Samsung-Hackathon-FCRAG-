"""
fcrag/eval/report.py — FCRAG 2.0 Evaluation Report Generator
=============================================================
Reads all four JSON result files from results/ and produces a
formatted markdown evaluation report at results/evaluation_report.md.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"


def _load(filename: str) -> Dict[str, Any] | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _badge(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


def generate_report() -> str:
    """Generate markdown evaluation report. Returns the report as a string."""

    custom = _load("custom_eval_results.json")
    teleqna = _load("teleqna_eval_results.json")
    faith = _load("faithfulness_results.json")
    latency = _load("latency_results.json")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines += [
        "# FCRAG 2.0 — Evaluation Report",
        "",
        f"**Generated:** {now}  ",
        "**Team:** IIT Madras AgentX-10  ",
        "",
        "---",
        "",
        "## Executive Summary — KPI Dashboard",
        "",
        "| Metric | Target | Actual | Status |",
        "|--------|--------|--------|--------|",
    ]

    # MRR row (prefer custom eval, fallback teleqna)
    if custom:
        mrr_val = custom["mrr"]
        mrr_src = "Custom 20-scenario"
        mrr_pass = custom["mrr_pass"]
    elif teleqna:
        mrr_val = teleqna["mrr"]
        mrr_src = "TeleQnA (proxy)"
        mrr_pass = teleqna["mrr_pass"]
    else:
        mrr_val, mrr_src, mrr_pass = None, "N/A", False

    mrr_str = f"{mrr_val:.4f} ({mrr_src})" if mrr_val is not None else "N/A"
    lines.append(f"| MRR | > 0.75 | {mrr_str} | {_badge(mrr_pass)} |")

    # Recall@5 row
    if custom:
        rec_val = custom["recall_at_k"]
        rec_src = "Custom 20-scenario"
        rec_pass = custom["recall_pass"]
    elif teleqna:
        rec_val = teleqna["recall_at_k"]
        rec_src = "TeleQnA (proxy)"
        rec_pass = teleqna["recall_pass"]
    else:
        rec_val, rec_src, rec_pass = None, "N/A", False

    rec_str = f"{rec_val:.4f} ({rec_src})" if rec_val is not None else "N/A"
    lines.append(f"| Recall@5 | > 0.85 | {rec_str} | {_badge(rec_pass)} |")

    # Faithfulness row
    if faith:
        faith_val = faith["avg_faithfulness_score"]
        faith_pass = faith["faithfulness_pass"]
        faith_str = f"{faith_val:.4f}"
    else:
        faith_val, faith_pass, faith_str = None, False, "N/A"

    lines.append(f"| Faithfulness | > 0.90 | {faith_str} | {_badge(faith_pass)} |")

    # Latency row
    if latency:
        lat_val = latency["total_latency"]["p50_ms"]
        lat_pass = latency["overall_pass"]
        lat_str = f"{lat_val:.0f}ms (P50)"
    else:
        lat_val, lat_pass, lat_str = None, False, "N/A"

    lines.append(f"| E2E Latency | < 4000ms | {lat_str} | {_badge(lat_pass)} |")

    lines += ["", "---", ""]

    # -----------------------------------------------------------------------
    # Section 1: Custom Eval
    # -----------------------------------------------------------------------
    lines += ["## 1. Custom Fault Scenario Evaluation (MRR + Recall@5)", ""]
    if custom:
        lines += [
            f"- **Scenarios:** {custom['n_scenarios']} labelled fault→clause mappings",
            f"- **Top-K:** {custom['top_k']}",
            f"- **MRR:** `{custom['mrr']:.4f}` (target > 0.75) {_badge(custom['mrr_pass'])}",
            f"- **Recall@5:** `{custom['recall_at_k']:.4f}` (target > 0.85) {_badge(custom['recall_pass'])}",
            "",
            "### Results by Fault Type",
            "",
            "| Fault Type | MRR | Recall@5 | Count |",
            "|---|---|---|---|",
        ]
        for ft, s in custom.get("by_fault_type", {}).items():
            lines.append(f"| {ft} | {s['mrr']:.4f} | {s['recall_at_k']:.4f} | {s['count']} |")

        lines += [
            "",
            "### Per-Scenario Detail",
            "",
            "| ID | Fault Type | RR | Hit | Latency |",
            "|---|---|---|---|---|",
        ]
        for r in custom.get("per_scenario", []):
            hit = "✅" if r["hit_at_k"] else "❌"
            lines.append(
                f"| {r['scenario_id']} | {r['fault_type']} | {r['reciprocal_rank']:.3f} "
                f"| {hit} | {r['retrieval_latency_ms']:.0f}ms |"
            )
    else:
        lines.append("_No results found — run `fcrag/eval/custom_eval.py`_")

    lines += ["", "---", ""]

    # -----------------------------------------------------------------------
    # Section 2: TeleQnA Eval
    # -----------------------------------------------------------------------
    lines += ["## 2. TeleQnA Benchmark Evaluation", ""]
    if teleqna:
        lines += [
            f"- **Questions sampled:** {teleqna['n_questions']}",
            f"- **Categories:** {', '.join(teleqna['categories'])}",
            f"- **Top-K:** {teleqna['top_k']}",
            f"- **MRR:** `{teleqna['mrr']:.4f}` (target > 0.75) {_badge(teleqna['mrr_pass'])}",
            f"- **Recall@5:** `{teleqna['recall_at_k']:.4f}` (target > 0.85) {_badge(teleqna['recall_pass'])}",
            f"- **Avg Retrieval Latency:** {teleqna['avg_retrieval_latency_ms']:.0f}ms  "
            f"(P50={teleqna['p50_latency_ms']:.0f}ms, P95={teleqna['p95_latency_ms']:.0f}ms)",
            "",
            "### Results by Category",
            "",
            "| Category | MRR | Recall@5 | Count |",
            "|---|---|---|---|",
        ]
        for cat, s in teleqna.get("by_category", {}).items():
            lines.append(f"| {cat} | {s['mrr']:.4f} | {s['recall_at_k']:.4f} | {s['count']} |")
    else:
        lines.append("_No results found — run `fcrag/eval/teleqna_eval.py`_")

    lines += ["", "---", ""]

    # -----------------------------------------------------------------------
    # Section 3: Faithfulness
    # -----------------------------------------------------------------------
    lines += ["## 3. Faithfulness Evaluation", ""]
    if faith:
        lines += [
            f"- **Scenarios evaluated:** {faith['n_scenarios']}",
            f"- **Method:** Jaccard word overlap (response ∩ context) / response",
            f"- **Avg Faithfulness:** `{faith['avg_faithfulness_score']:.4f}` "
            f"(target > 0.90) {_badge(faith['faithfulness_pass'])}",
            f"- **Median:** `{faith['median_faithfulness_score']:.4f}`",
            f"- **P10 (worst 10%):** `{faith['p10_faithfulness_score']:.4f}`",
            f"- ✅ Pass (>= 0.50): {faith['pass_count']}/{faith['n_scenarios']}",
            f"- ⚠️ Low faithfulness (< 0.50): {faith['low_faithfulness_count']}/{faith['n_scenarios']}",
            f"- ⚠️ Hallucination risk (< 0.30): {faith['hallucination_risk_count']}/{faith['n_scenarios']}",
            f"- **Avg LLM Latency:** {faith['avg_llm_latency_ms']:.0f}ms",
            "",
            "### Per-Scenario Faithfulness",
            "",
            "| ID | Fault Type | Score | Risk |",
            "|---|---|---|---|",
        ]
        for r in faith.get("per_scenario", []):
            if r["guardrail_engaged"]:
                risk = "ℹ️ GUARDRAIL"
                score_str = "N/A"
            elif r["hallucination_risk"]:
                risk = "⚠️ HALLUCINATION"
                score_str = f"{r['faithfulness_score']:.4f}"
            elif r["low_faithfulness"]:
                risk = "⚠️ LOW"
                score_str = f"{r['faithfulness_score']:.4f}"
            else:
                risk = "✅ OK"
                score_str = f"{r['faithfulness_score']:.4f}"
            lines.append(
                f"| {r['scenario_id']} | {r['fault_type']} "
                f"| {score_str} | {risk} |"
            )
    else:
        lines.append("_No results found — run `fcrag/eval/faithfulness_eval.py`_")

    lines += ["", "---", ""]

    # -----------------------------------------------------------------------
    # Section 4: Latency
    # -----------------------------------------------------------------------
    lines += ["## 4. End-to-End Latency Benchmark", ""]
    if latency:
        rs = latency["retrieval_latency"]
        ls = latency["llm_latency"]
        ts = latency["total_latency"]
        lines += [
            f"- **Runs:** {latency['n_runs']}",
            f"- **LLM:** {latency['llm_tier']}",
            f"- **Target:** < 4000ms E2E",
            f"- **Runs meeting target:** {latency['runs_meeting_target']}/{latency['n_runs']}",
            f"- **Overall:** {_badge(latency['overall_pass'])}",
            "",
            "### Latency Breakdown",
            "",
            "| Stage | Mean | P50 | P95 | P99 | Min | Max |",
            "|---|---|---|---|---|---|---|",
            f"| Retrieval | {rs['mean_ms']:.0f}ms | {rs['p50_ms']:.0f}ms | {rs['p95_ms']:.0f}ms | {rs['p99_ms']:.0f}ms | {rs['min_ms']:.0f}ms | {rs['max_ms']:.0f}ms |",
            f"| LLM Generate | {ls['mean_ms']:.0f}ms | {ls['p50_ms']:.0f}ms | {ls['p95_ms']:.0f}ms | {ls['p99_ms']:.0f}ms | {ls['min_ms']:.0f}ms | {ls['max_ms']:.0f}ms |",
            f"| **Total (E2E)** | **{ts['mean_ms']:.0f}ms** | **{ts['p50_ms']:.0f}ms** | **{ts['p95_ms']:.0f}ms** | **{ts['p99_ms']:.0f}ms** | **{ts['min_ms']:.0f}ms** | **{ts['max_ms']:.0f}ms** |",
            "",
            "> **Note:** LLM latency includes HF Inference API network round-trip.",
            "> For local GPU (Tier 1), expect LLM latency < 500ms.",
            "",
            "### Per-Run Detail",
            "",
            "| Run | Fault Type | Retrieval | LLM | Total | vs 4s |",
            "|---|---|---|---|---|---|",
        ]
        for r in latency.get("per_run", []):
            ok = "✅" if r["meets_4s_target"] else "❌"
            lines.append(
                f"| {r['run']} | {r['fault_type']} "
                f"| {r['retrieval_ms']:.0f}ms "
                f"| {r['llm_ms']:.0f}ms "
                f"| {r['total_ms']:.0f}ms | {ok} |"
            )
    else:
        lines.append("_No results found — run `fcrag/eval/latency_bench.py`_")

    lines += ["", "---", "", "_FCRAG 2.0 — IIT Madras AgentX-10_"]

    report = "\n".join(lines)

    out_path = RESULTS_DIR / "evaluation_report.md"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\n📄 Evaluation report saved → {out_path}")
    return report


if __name__ == "__main__":
    generate_report()
