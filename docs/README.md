# FCRAG  (Fault Cause Root Analysis Graph) 📡

An autonomous, agentic system designed for Root Cause Analysis (RCA) in 3GPP Telecommunications Networks using open-weight Large Language Models and Hybrid Retrieval.

##  Salient Features
- **Autonomous RCA:** Generates diagnostic reports detailing the Problem Statement, Root Cause, and Actionable Recommendations for injected network faults.
- **Agentic Workflow:** Utilizes specialized agents (Decomposer, Hybrid Retriever, Reasoner) built on an autonomous execution graph.
- **Hybrid Retrieval System:** Fuses BM25 (Sparse) and Qdrant (Dense Vector) search, refined by a highly accurate Cross-Encoder, to precisely fetch relevant 3GPP specification clauses.
- **NOC Interactive Dashboard:** A highly modern, interactive Streamlit UI built with a Network Operations Center aesthetic, complete with live telemetry visualizations and performance dials.
- **100% Open Weight:** Exclusively relies on open-source, local-capable models for data privacy.

##  Technical Architecture
The system architecture operates as a Directed Acyclic Graph (DAG) for agent execution:
1. **Anomaly Injection:** Telemetry data or manual queries are ingested.
2. **KPI Extraction:** The query is decomposed into search directives.
3. **Hybrid Search:** Queries the dense vector database (`sentence-transformers`) and a local BM25 index.
4. **Cross-Encoder Reranking:** Filters and strictly ranks the fused results.
5. **LLM Synthesis:** `Llama-3.2-3B-Tele-it` generates a highly structured diagnostic report grounded *only* in the retrieved context.

##  Technical Stack
| Component | OSS Library / Project Used | Link |
| :--- | :--- | :--- |
| **Frontend Framework** | Streamlit | [streamlit.io](https://streamlit.io/) |
| **Data Visualization** | Plotly | [plotly.com](https://plotly.com/) |
| **Vector Database** | Qdrant | [qdrant.tech](https://qdrant.tech/) |
| **LLM Inference** | HuggingFace Hub API | [huggingface.co](https://huggingface.co/) |
| **Sparse Retrieval** | Rank-BM25 | [pypi.org/project/rank-bm25/](https://pypi.org/project/rank-bm25/) |
| **Orchestration** | LangChain / LangGraph (concept) | [langchain.com](https://langchain.com/) |

##  Models Used (Hugging Face)
- **Language Model:** [AliMaatouk/Llama-3.2-3B-Tele-it](https://huggingface.co/AliMaatouk/Llama-3.2-3B-Tele-it)
- **Embedding Model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Reranker:** [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

##  Datasets Used
- [3GPP Specifications (Rel 15-18)](https://www.3gpp.org/)
- Internal Synthetic Fault Clause Mapping Scenarios (Proprietary / Generated).

##  Installation Instructions
1. **Clone the Repository** and navigate to the root directory.
2. **Install Dependencies:**
   Ensure you have a Python 3.11 environment active. We recommend using `uv` or `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Plotly, Pandas, and Numpy are required for the new interactive UI)*
3. **Environment Setup:**
   Create a `.env` file in the root directory and add your Hugging Face API key:
   ```env
   HUGGINGFACE_API_KEY=your_api_key_here
   ```

##  User Guide
### Running the Dashboard
Since the codebase is organized in the `src/` directory, start the interactive monitor by running:
```bash
streamlit run src/app.py
```
1. Wait for the engine to initialize the local Embeddings and Cross-Encoder (this may take a few seconds on the first run as it downloads weights).
2. The UI will display a **Live Telemetry Feed**.
3. Click **"Inject Network Anomaly"** to simulate a fault and trigger the agentic RCA pipeline.
4. Review the generated Diagnostic Report and the Evidence retrieved.

### Running Backend KPI Tests
To execute the autonomous KPI benchmarks, run:
```bash
python src/scripts/test_rag.py
```
