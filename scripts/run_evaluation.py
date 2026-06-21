"""
scripts/run_evaluation.py — FCRAG 2.0 One-Command Evaluation Runner
====================================================================
Runs all four Phase 6 evaluators sequentially, then generates the
final markdown report.

Usage:
    python scripts/run_evaluation.py                 # full evaluation
    python scripts/run_evaluation.py --quick         # custom + latency only (skip TeleQnA)
    python scripts/run_evaluation.py --custom-only   # custom eval only
    python scripts/run_evaluation.py --report-only   # just regenerate report from existing results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def section(title: str):
    print(f"\n{'#' * 60}")
    print(f"#  {title}")
    print(f"{'#' * 60}")


def main():
    parser = argparse.ArgumentParser(description="FCRAG 2.0 Evaluation Runner")
    parser.add_argument("--quick", action="store_true",
                        help="Skip TeleQnA eval (faster — only custom + faithfulness + latency)")
    parser.add_argument("--custom-only", action="store_true",
                        help="Run only the custom 20-scenario evaluator")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip all evals — just regenerate report from existing results")
    parser.add_argument("--skip-faithfulness", action="store_true",
                        help="Skip faithfulness eval (saves ~2 min of LLM API calls)")
    args = parser.parse_args()

    wall_start = time.perf_counter()

    print("=" * 60)
    print("FCRAG 2.0 — Phase 6 Evaluation Harness")
    print("=" * 60)

    timings = {}

    if not args.report_only:

        # ── 1. Custom Fault Evaluation ───────────────────────────────────
        section("1/4  Custom Fault Evaluation  (MRR + Recall@5)")
        t0 = time.perf_counter()
        from fcrag.eval.custom_eval import run_custom_eval
        custom_results = run_custom_eval(top_k=5)
        timings["custom_eval"] = time.perf_counter() - t0
        print(f"\n⏱  Custom eval completed in {timings['custom_eval']:.1f}s")

        if args.custom_only:
            section("Report")
            from fcrag.eval.report import generate_report
            generate_report()
            print(f"\n✅ Done (custom only) — Total: {time.perf_counter()-wall_start:.1f}s")
            return

        # ── 2. TeleQnA Evaluation ────────────────────────────────────────
        if not args.quick:
            section("2/4  TeleQnA Evaluation  (MRR + Recall@5, n=500)")
            t0 = time.perf_counter()
            from fcrag.eval.teleqna_eval import run_teleqna_eval
            teleqna_results = run_teleqna_eval(sample=500, top_k=5)
            timings["teleqna_eval"] = time.perf_counter() - t0
            print(f"\n⏱  TeleQnA eval completed in {timings['teleqna_eval']:.1f}s")
        else:
            print("\n⏭️  TeleQnA eval skipped (--quick mode)")

        # ── 3. Faithfulness Evaluation ───────────────────────────────────
        if not args.skip_faithfulness:
            section("3/4  Faithfulness Evaluation  (20 RAG responses)")
            t0 = time.perf_counter()
            from fcrag.eval.faithfulness_eval import run_faithfulness_eval
            faith_results = run_faithfulness_eval(top_k=5, max_tokens=200)
            timings["faithfulness_eval"] = time.perf_counter() - t0
            print(f"\n⏱  Faithfulness eval completed in {timings['faithfulness_eval']:.1f}s")
        else:
            print("\n⏭️  Faithfulness eval skipped (--skip-faithfulness)")

        # ── 4. Latency Benchmark ─────────────────────────────────────────
        section("4/4  End-to-End Latency Benchmark  (20 runs)")
        t0 = time.perf_counter()
        from fcrag.eval.latency_bench import run_latency_bench
        latency_results = run_latency_bench(top_k=5, max_tokens=200)
        timings["latency_bench"] = time.perf_counter() - t0
        print(f"\n⏱  Latency bench completed in {timings['latency_bench']:.1f}s")

    # ── 5. Report ─────────────────────────────────────────────────────────
    section("📄 Generating Evaluation Report")
    from fcrag.eval.report import generate_report
    generate_report()

    # ── Final Summary ─────────────────────────────────────────────────────
    total_s = time.perf_counter() - wall_start

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    if timings:
        for name, s in timings.items():
            print(f"  {name:25s}: {s:.1f}s")
    print(f"  {'Total wall time':25s}: {total_s:.1f}s")
    print(f"\n📂 Results saved in: {ROOT / 'results'}/")
    print(f"   • custom_eval_results.json")
    print(f"   • teleqna_eval_results.json  (if run)")
    print(f"   • faithfulness_results.json  (if run)")
    print(f"   • latency_results.json")
    print(f"   • evaluation_report.md  ← final report")


if __name__ == "__main__":
    main()
