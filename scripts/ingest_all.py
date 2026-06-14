"""
scripts/ingest_all.py — FCRAG 2.0 Phase 1 Ingestion Pipeline
=============================================================
One-command runner for the full ingestion pipeline:
  Step 1: chunk   — Chunk all normalized data sources            (Phase 1.1)
  Step 2: embed   — Embed chunks with sentence-transformers      (Phase 1.2)
  Step 3: index   — Upload dense vectors to Qdrant               (Phase 1.3)
  Step 4: bm25    — Build BM25 sparse keyword index              (Phase 1.4)
  Step 5: simu5g  — Generate English narratives from KPI logs    (Phase 1.5)

Usage:
  python scripts/ingest_all.py                   # run all steps
  python scripts/ingest_all.py --step chunk      # chunking only
  python scripts/ingest_all.py --step embed      # embedding only
  python scripts/ingest_all.py --step index      # indexing only
  python scripts/ingest_all.py --step bm25       # BM25 build only
  python scripts/ingest_all.py --step simu5g     # Simu5G narrative generation only
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def step_chunk():
    print("\n" + "=" * 60)
    print("PHASE 1.1 — Chunking all data sources")
    print("=" * 60)
    from fcrag.ingest.chunker import run_chunker
    t0 = time.time()
    stats = run_chunker(verbose=True)
    elapsed = time.time() - t0

    print("\n  ─── Chunk Summary ───")
    total = 0
    for collection, count in stats.items():
        print(f"  {collection:<25} {count:>8,} chunks")
        total += count
    print(f"  {'TOTAL':<25} {total:>8,} chunks")
    print(f"\n  Chunking completed in {elapsed:.1f}s")
    print(f"  Output → data/processed/chunks/")
    return stats


def step_simu5g():
    """Phase 1.5: Translate raw Simu5G KPI CSVs into English JSONL narratives."""
    import time
    from fcrag.ingest.simu5g_generator import run_generation
    t0 = time.time()
    run_generation()
    elapsed = time.time() - t0
    print(f"\n  Simu5G narrative generation completed in {elapsed:.1f}s")


def step_embed():
    print("\n" + "=" * 60)
    print("PHASE 1.2 — Embedding all data sources")
    print("=" * 60)
    from fcrag.ingest.embedder import run_embedder
    import time
    t0 = time.time()
    run_embedder()
    elapsed = time.time() - t0
    print(f"\n  Embedding completed in {elapsed:.1f}s")


def step_index():
    from fcrag.ingest.indexer import run_indexer
    run_indexer()


def step_bm25():
    from fcrag.ingest.bm25_builder import run_bm25_build
    run_bm25_build()


def main():
    parser = argparse.ArgumentParser(
        description="FCRAG 2.0 — Phase 1 Ingestion Pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["chunk", "embed", "index", "bm25", "simu5g"],
        help="Run only a specific step. Default: run all steps.",
    )
    args = parser.parse_args()

    print("\n>> FCRAG 2.0 -- Ingestion Pipeline")

    if args.step == "chunk" or args.step is None:
        step_chunk()
    if args.step == "embed" or args.step is None:
        step_embed()
    if args.step == "index" or args.step is None:
        step_index()
    if args.step == "bm25" or args.step is None:
        step_bm25()
    if args.step == "simu5g" or args.step is None:
        step_simu5g()

    print("\n[OK] Ingestion pipeline done.")


if __name__ == "__main__":
    main()
