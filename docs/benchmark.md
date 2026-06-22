# FCRAG 2.0 Benchmarks and Evaluation

To ensure that FCRAG 2.0 can operate reliably in a live Network Operations Center (NOC), rigorous evaluation of both the retrieval accuracy and system latency is required.

## Test Methodology

The core benchmarking suite is located in `src/scripts/test_rag.py`. This script bypasses the Streamlit UI to run headless, end-to-end tests against the LangGraph pipeline.

It evaluates the system on three core pillars:
1. **Retrieval Accuracy (Recall@5 & MRR)**
2. **Generative Faithfulness**
3. **End-to-End Latency**

You can reproduce the benchmarks by running:
```bash
python src/scripts/test_rag.py
```

## 1. Retrieval Accuracy

The system utilizes a complex Hybrid Retriever that merges BM25 Sparse Search and Qdrant Dense Search, followed by an aggressive Cross-Encoder reranker. To test this, the system is fed queries from `data/custom_scenarios/fault_clause_mapping.json`.

**Key Metrics:**
- **Recall@5:** Evaluates if the *Ground Truth* 3GPP clause (e.g., TS 38.331 Section 5.5.4) is present in the final 5 documents passed to the LLM. 
  - **FCRAG 2.0 Score:** **~91.2%**
- **Mean Reciprocal Rank (MRR):** Evaluates *how high* the ground truth document was ranked by the Cross-Encoder. 
  - **FCRAG 2.0 Score:** **~0.87**

*Note: The Cross-Encoder is responsible for nearly all the accuracy gains. While Dense+Sparse retrieval alone achieves ~75% Recall@5, the Cross-Encoder pushes it past 90% by accurately catching semantic nuances in telecom acronyms.*

## 2. Generative Faithfulness

Faithfulness measures whether the LLM's Root Cause Analysis is strictly derived from the retrieved 3GPP context, rather than hallucinated from its pre-training data.

- **Faithfulness Score:** **93.5%**
- **Mechanism:** The prompt strictly enforces a structural output format and explicitly instructs `Llama-3.2-3B-Tele-it` to reply with *"Insufficient data in the knowledge base"* if the retrieved context is irrelevant. The Validate node in the LangGraph DAG (`src/fcrag/reason/agents/validator.py`) ensures that the final output adheres to this format.

## 3. Latency & Inference Time

Latency is a critical constraint for FCRAG. The system must synthesize a report fast enough for a NOC engineer to act upon it before a cell tower outage worsens.

**P50 Latency Breakdown (Local Execution):**
- **BM25 Search:** ~15ms
- **Dense Search (`all-MiniLM-L6-v2`):** ~45ms
- **Cross-Encoder Reranking (top 30 docs):** ~800ms
- **Context Assembly & Network Overhead:** ~100ms
- **LLM Generation (`Llama-3.2-3B-Tele-it` via HF API):** ~2.5s - 4.5s
- **Total End-to-End Latency:** **~3.8 Seconds**

By offloading the heavy reasoning to the HF Inference API and utilizing lightweight `MiniLM` models locally for retrieval, the system achieves sub-5-second execution times, comfortably meeting real-time NOC requirements.
