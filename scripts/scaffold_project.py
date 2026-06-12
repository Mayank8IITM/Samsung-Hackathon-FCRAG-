"""
FCRAG 2.0 — Project Directory Scaffolding
==========================================
Run this once after download_data.py to create all module directories and __init__.py files.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRECTORIES = [
    # Core FCRAG package
    "fcrag",
    "fcrag/ingest",
    "fcrag/sense",
    "fcrag/encode",
    "fcrag/retrieve",
    "fcrag/reason",
    "fcrag/reason/agents",
    "fcrag/explain",
    "fcrag/feedback",
    "fcrag/eval",
    "fcrag/api",
    # Config
    "config",
    # Data directories
    "data/3gpp",
    "data/oran",
    "data/simu5g",
    "data/teleqna",
    "data/tele_eval",
    "data/custom_scenarios",
    "data/feedback",
    # Model artifacts
    "models",
    # Scripts
    "scripts",
    # Demo
    "demo",
    # Tests
    "tests",
    # Results
    "results",
]

INIT_PACKAGES = [
    "fcrag",
    "fcrag/ingest",
    "fcrag/sense",
    "fcrag/encode",
    "fcrag/retrieve",
    "fcrag/reason",
    "fcrag/reason/agents",
    "fcrag/explain",
    "fcrag/feedback",
    "fcrag/eval",
    "fcrag/api",
    "tests",
]

PLACEHOLDER_FILES = {
    # Ingest
    "fcrag/ingest/chunker.py":         "# Phase 1: 125-token chunker + PaddleOCR\n",
    "fcrag/ingest/embedder.py":        "# Phase 1: Gemma-2-2B-Tele embedding pipeline\n",
    "fcrag/ingest/indexer.py":         "# Phase 1: Qdrant collection creation & population\n",
    "fcrag/ingest/bm25_builder.py":    "# Phase 1: BM25 index construction\n",
    "fcrag/ingest/simu5g_generator.py":"# Phase 1: Simu5G narrative embedding\n",
    # Sense
    "fcrag/sense/kpi_stream.py":       "# Phase 2: KPI ingestion (CSV / streaming)\n",
    "fcrag/sense/detector.py":         "# Phase 2: IsolationForest + EWMA anomaly detection\n",
    "fcrag/sense/correlator.py":       "# Phase 5.5 Add4: Multi-cell correlation\n",
    "fcrag/sense/event_schema.py":     "# Phase 2: AnomalyEvent dataclass\n",
    # Encode
    "fcrag/encode/fse.py":             "# Phase 3: Fault Signature Encoder (PyTorch)\n",
    "fcrag/encode/fse_trainer.py":     "# Phase 3: Contrastive training loop\n",
    "fcrag/encode/taae.py":            "# Phase 3: Telecom Acronym & Augmentation Engine\n",
    # Retrieve
    "fcrag/retrieve/retriever.py":     "# Phase 4: Hybrid retrieval orchestrator\n",
    "fcrag/retrieve/dense.py":         "# Phase 4: Qdrant dense search\n",
    "fcrag/retrieve/sparse.py":        "# Phase 4: BM25 retrieval\n",
    "fcrag/retrieve/reranker.py":      "# Phase 4: Cross-encoder reranking (ONNX)\n",
    "fcrag/retrieve/explainer.py":     "# Phase 5.5 Add3: Retrieval explanation layer\n",
    # Reason
    "fcrag/reason/graph.py":           "# Phase 5: LangGraph StateGraph definition\n",
    "fcrag/reason/llm_client.py":      "# Phase 5: vLLM client wrapper (primary + fallback)\n",
    "fcrag/reason/tiers.py":           "# Phase 5.5 Add1: Confidence-calibrated tiered response\n",
    "fcrag/reason/agents/decomposer.py":       "# Phase 5: Decomposer Agent\n",
    "fcrag/reason/agents/retriever_agent.py":  "# Phase 5: Retriever Agent\n",
    "fcrag/reason/agents/reasoning_agent.py":  "# Phase 5: Reasoning Agent\n",
    "fcrag/reason/agents/validator.py":        "# Phase 5: Validator Agent\n",
    # Explain
    "fcrag/explain/reporter.py":       "# Phase 5: Output package assembly\n",
    "fcrag/explain/causal_graph.py":   "# Phase 5: NetworkX causal graph builder\n",
    "fcrag/explain/formatter.py":      "# Phase 5: JSON + human-readable formatter\n",
    # Feedback
    "fcrag/feedback/collector.py":     "# Phase 5.5 Add2: Feedback log writer\n",
    "fcrag/feedback/retrain_trigger.py":"# Phase 5.5 Add2: Trigger FSE re-train\n",
    # Eval
    "fcrag/eval/teleqna_eval.py":      "# Phase 6: MRR / Recall@5 on TeleQnA\n",
    "fcrag/eval/faithfulness_eval.py": "# Phase 6: Faithfulness evaluation\n",
    "fcrag/eval/latency_bench.py":     "# Phase 6: 50-run latency benchmark\n",
    "fcrag/eval/custom_eval.py":       "# Phase 6: 20-fault custom scenario evaluation\n",
    # API
    "fcrag/api/main.py":               "# Phase 7: FastAPI application\n",
    "fcrag/api/schemas.py":            "# Phase 7: Pydantic request/response schemas\n",
    # Scripts
    "scripts/ingest_all.py":          "# Phase 1: One-command ingestion pipeline\n",
    "scripts/run_evaluation.py":       "# Phase 6: One-command full benchmark\n",
    "scripts/retrain_fse.py":          "# Phase 5.5 Add2: FSE retraining script\n",
    # Demo
    "demo/app.py":                     "# Phase 5.5 Add5: Streamlit demo dashboard\n",
    "demo/run_demo.py":                "# Phase 7: CLI demo script (5 fault scenarios)\n",
    # Tests
    "tests/test_detector.py":          "# Phase 2 tests\n",
    "tests/test_retrieval.py":         "# Phase 4 tests\n",
    "tests/test_agents.py":            "# Phase 5 tests\n",
    "tests/test_e2e.py":               "# Phase 6 end-to-end tests\n",
    "tests/conftest.py":               "# pytest fixtures\n",
}


def scaffold():
    print("🏗️  Creating FCRAG project structure...\n")

    # Create directories
    for d in DIRECTORIES:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)

    # Create __init__.py for all packages
    for pkg in INIT_PACKAGES:
        init = ROOT / pkg / "__init__.py"
        if not init.exists():
            init.write_text(f'"""FCRAG — {pkg.replace("/", ".")} package"""\n')

    # Create placeholder files
    for rel_path, content in PLACEHOLDER_FILES.items():
        full_path = ROOT / rel_path
        if not full_path.exists():
            full_path.write_text(content)

    print("✅ Directory structure created:")
    for d in DIRECTORIES:
        print(f"   📁 {d}/")

    print(f"\n✅ {len(PLACEHOLDER_FILES)} module files created (placeholders)")
    print("\n🎯 Project scaffold ready. Next steps:")
    print("   1. uv add -r requirements.txt")
    print("   2. python scripts/download_data.py")
    print("   3. Place 3GPP PDFs in data/3gpp/")
    print("   4. python scripts/ingest_all.py")


if __name__ == "__main__":
    scaffold()
