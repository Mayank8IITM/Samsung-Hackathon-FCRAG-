# FCRAG 2.0 Technical Stack

This document outlines the Open-Source Software (OSS) libraries and frameworks selected for FCRAG 2.0. The stack was deliberately chosen to support a local, private, and agentic architecture without depending on closed-source APIs for its core logic.

## Core Libraries

| Component | Library / Framework | Version | Purpose | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | Python | `>=3.11` | Primary language for data manipulation, AI inference, and backend logic. | [python.org](https://www.python.org/) |
| **Frontend/UI** | Streamlit | `>=1.38.0` | Creates the interactive Network Operations Center (NOC) dashboard (`src/app.py`). Chosen for its rapid prototyping capabilities in Python. | [streamlit.io](https://streamlit.io/) |
| **Visualizations** | Plotly | `>=5.24.0` | Generates the real-time telemetry sparklines, bar charts for reranker scores, and source breakdown pie charts within the NOC dashboard. | [plotly.com](https://plotly.com/) |
| **Data Processing** | Pandas | `>=2.2.0` | Handles CSV/JSON parsing for KPI streams, baseline extraction, and generating mock timeseries data for the UI. | [pandas.pydata.org](https://pandas.pydata.org/) |

## Agentic AI & RAG Pipeline

| Component | Library / Framework | Version | Purpose | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Graph Orchestration** | LangGraph | `>=0.2.0` | Constructs the Directed Acyclic Graph (DAG) for multi-agent reasoning, managing state transitions between the Decomposer, Retriever, Reasoner, and Validator nodes. | [langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/) |
| **Vector Database** | Qdrant Client | `>=1.11.0` | Stores and retrieves dense vector representations of the 3GPP specifications locally on disk without requiring a heavy Docker deployment. | [qdrant.tech](https://qdrant.tech/) |
| **Sparse Retrieval** | Rank-BM25 | `>=0.2.2` | Provides traditional lexical (keyword) search capabilities. Crucial for matching exact telecom identifiers (e.g., "TS 38.331", "RRCSetup") which dense embeddings sometimes miss. | [pypi.org/project/rank-bm25/](https://pypi.org/project/rank-bm25/) |
| **Embeddings Generator** | Sentence-Transformers | `>=3.0.0` | Wraps Hugging Face models to generate semantic vector embeddings from 3GPP text chunks (`all-MiniLM-L6-v2`) and executes Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`). | [sbert.net](https://www.sbert.net/) |
| **LLM Inference Client** | Hugging Face Hub | `>=0.25.0` | Connects to Hugging Face Inference API for communicating with `Llama-3.2-3B-Tele-it`. Enables seamless API switching depending on tier constraints. | [huggingface.co](https://huggingface.co/) |

## Infrastructure & Testing

| Component | Library / Framework | Version | Purpose | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Logging** | Loguru | `>=0.7.2` | Structured, color-coded terminal logging for tracking agent execution boundaries and retrieval latencies in the backend. | [github.com/Delgan/loguru](https://github.com/Delgan/loguru) |
| **Testing** | Pytest | `>=8.0.0` | Powers the `src/tests/` suite to ensure system stability during codebase reorganizations. | [pytest.org](https://docs.pytest.org/) |
| **Environment Management** | python-dotenv | `>=1.0.0` | securely loads local API keys (e.g. `HUGGINGFACE_API_KEY`) from `.env` files preventing accidental credential leakage. | [pypi.org/project/python-dotenv/](https://pypi.org/project/python-dotenv/) |

## Why this Stack?

- **Local Execution Capability:** Qdrant Client (running in persistent disk mode) and Sentence-Transformers allow the entire retrieval pipeline to execute locally without cloud vendor lock-in.
- **Explainability:** LangGraph's explicit state machine provides full visibility into exactly what context is retrieved before passing it to the LLM.
- **Performance:** Relying on lightweight models like `MiniLM-L6-v2` guarantees sub-second retrieval times, which is essential for real-time anomaly analysis in a telecom setting.
