# FCRAG 2.0 Repository Folder Structure

The project has been heavily refactored into a modern, standardized Python `src/` layout. This ensures proper package resolution during `pip install -e .` and isolates the application code from configuration and data artifacts.

## Root Directory Tree

```text
Samsung-Hackathon-FCRAG-/
├── docs/                        # Comprehensive documentation (Architecture, Installation, etc.)
├── data/                        # Local databases, 3GPP specs, and Simu5G telemetry
├── config/                      # YAML configuration files (settings.yaml, acronyms.yaml)
├── src/                         # Core Python codebase
│   ├── app.py                   # Streamlit NOC Dashboard Entry Point
│   ├── fcrag/                   # Primary FCRAG Python Package
│   ├── scripts/                 # Utility scripts for ingestion and evaluation
│   └── tests/                   # Pytest automated test suite
├── .env                         # Local environment variables (HuggingFace API Key)
├── pyproject.toml               # Python project configuration and dependency listing
├── README.md                    # Overview and Quickstart Guide
└── FCRAG_PRD.md / FCRAG_System_Design.md
```

## Detailed Directory Breakdown

### `src/fcrag/` - The Core Engine
This directory houses the primary logic for the FCRAG system.

- **`reason/`:** The Agentic RAG core.
  - `graph.py`: Defines the LangGraph execution DAG (State Machine).
  - `llm_client.py`: The interface to the `Llama-3.2-3B-Tele-it` Hugging Face model.
  - `agents/`: Contains the specific LangGraph node logic (`decomposer.py`, `retriever_agent.py`, `reasoning_agent.py`, `validator.py`).
- **`retrieve/`:** The Hybrid Search Engine.
  - `retriever.py`: The `HybridRetriever` class that fuses Sparse and Dense search.
  - `dense.py`: Handles Qdrant vector database queries.
  - `sparse.py`: Handles `rank_bm25` lexical queries.
  - `reranker.py`: The Cross-Encoder implementation to strictly rank the fused results.
- **`ingest/`:** Data Processing.
  - `chunker.py`: Splits raw 3GPP `.txt` files into optimal semantic chunks.
  - `embedder.py`: Interacts with `sentence-transformers` to generate vectors.
  - `indexer.py`: Loads the vectors into Qdrant.
- **`eval/`:** Benchmarking and validation utilities (used by the test scripts).

### `src/scripts/` - Utilities & Execution
Scripts intended to be run directly from the terminal.

- `ingest_all.py`: The master script to chunk 3GPP specs, generate embeddings, and build both Qdrant and BM25 indexes.
- `test_rag.py`: A highly detailed benchmark script that evaluates the system's Latency, Recall, and Faithfulness outside of the Streamlit UI.
- `download_data.py`: Helper script to fetch 3GPP resources.

### `src/tests/` - Testing Suite
Standard `pytest` directory ensuring module stability.
- `test_retrieval.py`: Asserts that dense and sparse searches return valid clause structures.
- `test_agents.py`: Asserts the LangGraph nodes pass state correctly.

### `data/` - Storage (Ignored via `.gitignore` where appropriate)
- **`qdrant_db/`:** Persistent local disk storage for the Qdrant vector database.
- **`processed/`:**
  - `3gpp_text/`: The raw TS 38.331, TS 38.300, TS 23.501 texts.
  - `chunks/`: `.jsonl` files containing the split and metadata-tagged text.
  - `indexes/`: `.pkl` files containing the pre-built `rank_bm25` sparse indexes.
- **`custom_scenarios/`:** Holds `fault_clause_mapping.json`, which dictates the preset anomaly injections used in the NOC Dashboard.
- **`simu5g/`:** Legacy and supplementary KPI log data.
