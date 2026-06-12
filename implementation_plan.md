# FCRAG 2.0 — Implementation Plan
**Team:** IIT Madras AgentX-10 | **Timeline:** 4 Weeks
**Status:** ✅ Approved — 5 additions integrated

---

## Phase Overview

| Phase | Focus | Duration |
|---|---|---|
| 0 | Environment Setup & Data Download | Day 1–2 |
| 1 | Knowledge Base Ingestion & Indexing | Day 3–6 |
| 2 | Anomaly Detection (Stage 1) | Day 7–9 |
| 3 | FSE Training + TAAE (Stage 2) | Day 10–13 |
| 4 | Hybrid Retrieval Pipeline (Stage 3) | Day 14–17 |
| 5 | LangGraph Multi-Agent System (Stage 4–5) | Day 18–23 |
| **5.5** | **Additions: Tiered Response + Feedback + Explainability + Multi-Cell + Streamlit** | **Day 22–25** |
| 6 | Evaluation Harness + Benchmarking | Day 24–26 |
| 7 | Demo Polish + Submission | Day 27–28 |

---

## Phase 0 — Environment Setup (Day 1–2)

### Tasks
- [ ] Create Python 3.11 virtual environment
- [ ] Install: `torch`, `transformers`, `qdrant-client`, `langchain`, `langgraph`, `rank_bm25`, `scikit-learn`, `paddleocr`, `vllm`, `fastapi`, `onnxruntime`, `networkx`, `pandas`, `numpy`
- [ ] Download models from HuggingFace:
  - `AliMaatouk/Gemma-2-2B-Tele`
  - `AliMaatouk/Llama-3.2-3B-Tele-it`
  - `AliMaatouk/TinyLlama-1.1B-Tele-it`
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
- [ ] Download datasets:
  - `netop-team/TeleQnA` (10K MCQs)
  - `AliMaatouk/Tele-Eval` (sample 10K for dev)
- [ ] Download 3GPP PDFs: TS 38.331, 38.321, 38.401, 38.413, 38.214
- [ ] Download O-RAN WG1/WG3 specs
- [ ] Collect Simu5G CSV logs (150 files)
- [ ] Set up Qdrant (Docker or in-process)
- [ ] Initialize project repo with structure from System Design §5

**Deliverable:** `scripts/setup_env.sh` + `requirements.txt`

---

## Phase 1 — Knowledge Base Ingestion (Day 3–6)

### 1.1 Document Chunker (`fcrag/ingest/chunker.py`)
- [ ] PDF text extraction using PyMuPDF
- [ ] Sentence-aware 125-token chunker with 25-token overlap
- [ ] Metadata preservation: {source, section, page, clause_id}
- [ ] PaddleOCR integration for image pages: extract text labels
- [ ] Handle table extraction (convert to text rows)

### 1.2 Embedder (`fcrag/ingest/embedder.py`)
- [ ] Load Gemma-2-2B-Tele
- [ ] Extract layer -2 hidden states as embeddings (2048-dim)
- [ ] Batch embedding with progress bar
- [ ] Cache embeddings to disk (avoid re-computation)

### 1.3 Qdrant Indexer (`fcrag/ingest/indexer.py`)
- [ ] Create 4 collections: `3gpp_specs`, `oran_specs`, `simu5g_narratives`, `alarm_history`
- [ ] Use IndexFlatIP (research-validated best for telecom corpus)
- [ ] Upload chunks + metadata + embeddings
- [ ] Verify collection counts

### 1.4 BM25 Index Builder (`fcrag/ingest/bm25_builder.py`)
- [ ] Build BM25 index over all text chunks
- [ ] Serialize index to disk (pickle)
- [ ] Validate: test query returns expected chunks

### 1.5 Simu5G Narrative Generator (`fcrag/ingest/simu5g_generator.py`)
- [ ] Parse 150 Simu5G CSV fault logs
- [ ] Generate narrative format: "Scenario N: [fault type] at [cell], [time]. Root cause: [X]. Action: [Y]. Recovery: [Z] in [T]s."
- [ ] Embed narratives and insert into `simu5g_narratives` collection

**Deliverable:** All 4 Qdrant collections populated; BM25 index saved; `scripts/ingest_all.py`

---

## Phase 2 — Anomaly Detection (Day 7–9)

### 2.1 KPI Stream (`fcrag/sense/kpi_stream.py`)
- [ ] CSV reader (batch mode for Simu5G logs)
- [ ] Streaming mode interface (for demo live playback)
- [ ] KPI normalization (z-score per KPI type)
- [ ] Window buffer (sliding 60-second window)

### 2.2 IsolationForest Detector
- [ ] Train IsolationForest on normal KPI baseline (from Simu5G healthy periods)
- [ ] Contamination parameter: 0.05
- [ ] Score threshold calibration against 20 known fault scenarios
- [ ] Serialize trained model (`models/isolation_forest.pkl`)

### 2.3 EWMA Drift Detector
- [ ] Implement EWMA with α=0.1 per KPI metric
- [ ] Drift alert when EWMA deviates > 2σ from rolling baseline
- [ ] Combine with IsolationForest: OR logic (either triggers → anomaly event)

### 2.4 Anomaly Event Emitter
- [ ] Build structured AnomalyEvent dataclass
- [ ] Compute KPI deltas (current vs. baseline)
- [ ] Assign severity: CRITICAL / HIGH / MEDIUM / LOW based on score
- [ ] Unit tests with 20 custom fault scenarios (all must trigger)

**Deliverable:** `fcrag/sense/` module; `tests/test_detector.py` passing

---

## Phase 3 — FSE Training + TAAE (Day 10–13)

### 3.1 FSE Neural Network (`fcrag/encode/fse.py`)
- [ ] Define PyTorch model: Linear(n_kpi, 64) → ReLU → Linear(64, 128) → ReLU → Linear(128, 2048)
- [ ] Input: KPI delta vector (5–15 features)
- [ ] Output: 2048-dim embedding matching Gemma-2-2B-Tele space

### 3.2 FSE Training (`fcrag/encode/fse_trainer.py`)
- [ ] Build training pairs from 20 custom fault→clause mappings
- [ ] Loss: InfoNCE contrastive loss (positive: correct clause, negatives: other chunks)
- [ ] Data augmentation: add Gaussian noise to KPI vectors
- [ ] Train 100 epochs; early stopping on val loss
- [ ] Evaluate: cosine similarity with correct clause chunk > 0.7 on all 20 pairs
- [ ] Save checkpoint: `models/fse_checkpoint.pt`

### 3.3 TAAE (`fcrag/encode/taae.py`)
- [ ] Build curated 3GPP acronym dictionary (300+ entries: PRACH, PDCCH, CQI, HARQ, etc.)
- [ ] Acronym expansion step: regex replacement in query string
- [ ] Candidate synonym generation for BM25 augmentation
- [ ] Test: "PRACH failure" → "Physical Random Access Channel failure RACH random access"

**Deliverable:** Trained FSE model; TAAE with 300+ acronyms; `tests/test_fse.py`

---

## Phase 4 — Hybrid Retrieval Pipeline (Day 14–17)

### 4.1 Dense Retriever (`fcrag/retrieve/dense.py`)
- [ ] Use FSE output vector for fault-conditioned queries
- [ ] Use Gemma-2-2B-Tele for text-query embedding (TeleQnA evaluation)
- [ ] Qdrant search: top-20 per collection
- [ ] Return chunks with metadata and scores

### 4.2 Sparse Retriever (`fcrag/retrieve/sparse.py`)
- [ ] BM25 query using TAAE augmented string
- [ ] Return top-20 documents with BM25 scores

### 4.3 Fusion & Reranker (`fcrag/retrieve/reranker.py`)
- [ ] Reciprocal Rank Fusion (RRF) to merge dense + sparse results
- [ ] Cross-encoder reranking using MiniLM-L-6-v2 (ONNX runtime)
- [ ] Return top-5 reranked chunks with final scores

### 4.4 Retrieval Evaluation
- [ ] Run on 20 custom fault→clause dataset: measure Recall@5 and MRR
- [ ] Run on TeleQnA subset (500 questions): measure Recall@5 and MRR
- [ ] Target: MRR > 0.75, Recall@5 > 0.85
- [ ] Ablation: dense only vs. sparse only vs. hybrid → document results

**Deliverable:** `fcrag/retrieve/` module; retrieval hitting targets on custom dataset

---

## Phase 5 — LangGraph Multi-Agent System (Day 18–23)

### 5.1 LLM Client (`fcrag/reason/llm_client.py`)
- [ ] Load Llama-3.2-3B-Tele-it via vLLM
- [ ] Load TinyLlama-1.1B-Tele-it as fallback
- [ ] Auto-switch: if primary > 2s response, use fallback
- [ ] Prompt templates: system + user with context injection

### 5.2 Decomposer Agent (`fcrag/reason/agents/decomposer.py`)
- [ ] Input: AnomalyEvent + initial context
- [ ] Output: list of sub-queries (2–4) covering different aspects of the fault
- [ ] Fault-type classification: HO_FAILURE / PHY_LAYER / RRC / CONGESTION

### 5.3 Retriever Agent (`fcrag/reason/agents/retriever_agent.py`)
- [ ] Execute hybrid retrieval for each sub-query
- [ ] Assemble tri-modal context: spec + operational + synthetic
- [ ] Deduplicate across sub-query results

### 5.4 Reasoning Agent (`fcrag/reason/agents/reasoning_agent.py`)
- [ ] Build causal chain using retrieved context
- [ ] Structured prompt: "Given these 5 specification chunks, identify root cause and recommend actions with clause citations."
- [ ] Extract: {root_cause, causal_chain[], corrective_actions[], citations[]}
- [ ] Use NetworkX to represent causal graph

### 5.5 Validator Agent (`fcrag/reason/agents/validator.py`)
- [ ] For each claim in reasoning output: check coverage in retrieved chunks
- [ ] Faithfulness score = (claims_supported / total_claims)
- [ ] If faithfulness < 0.7 → return INSUFFICIENT_CONTEXT
- [ ] Log all validation decisions

### 5.6 LangGraph Graph Assembly (`fcrag/reason/graph.py`)
- [ ] Define StateGraph with FCRAGState TypedDict
- [ ] Wire: Decomposer → Retriever → Reasoning → Validator
- [ ] Add conditional edge: Validator passes or returns INSUFFICIENT_CONTEXT
- [ ] Compile graph

**Deliverable:** Full LangGraph pipeline running end-to-end; `tests/test_agents.py`

---

## Phase 6 — Evaluation Harness (Day 24–26)

### 6.1 TeleQnA Evaluator (`fcrag/eval/teleqna_eval.py`)
- [ ] Load TeleQnA dataset (Standards Overview + Standards Specifications tasks)
- [ ] For each question: retrieve top-5, check if correct answer in chunks
- [ ] Report: MRR, Recall@5, Accuracy (MCQ answer selection)
- [ ] Run full 10K evaluation + report results

### 6.2 Faithfulness Evaluator (`fcrag/eval/faithfulness_eval.py`)
- [ ] Sample 100 generated RCA responses
- [ ] For each: manual + automated claim grounding check
- [ ] Report faithfulness score

### 6.3 Latency Benchmark (`fcrag/eval/latency_bench.py`)
- [ ] 50 end-to-end runs with varied fault scenarios
- [ ] Record: detection, FSE, retrieval, LLM, total latency per run
- [ ] Report: P50, P95, P99 latency
- [ ] Separate benchmarks: Llama-3.2-3B (primary) vs. TinyLlama (fallback)

### 6.4 Evaluation Report
- [ ] Create `results/evaluation_report.md` with all metric tables
- [ ] Include ablation study results (chunking, retrieval strategy comparisons)

**Deliverable:** Full evaluation report; all KPIs met or explanation of gap

---

## Phase 7 — Demo & Submission (Day 27–28)

### 7.1 FastAPI Application (`fcrag/api/main.py`)
- [ ] POST /analyze-fault endpoint
- [ ] GET /health endpoint
- [ ] POST /benchmark/teleqna endpoint (runs quick 100-question eval)
- [ ] Swagger docs auto-generated

### 7.2 Demo Script (`demo/run_demo.py`)
- [ ] Replay 5 Simu5G fault scenarios in sequence
- [ ] Print formatted RCA output for each
- [ ] Show timing breakdown
- [ ] Show causal graph (text or matplotlib)

### 7.3 Submission Checklist
- [ ] README.md with setup instructions and architecture diagram
- [ ] `requirements.txt` pinned
- [ ] `scripts/ingest_all.py` one-command ingestion
- [ ] `scripts/run_evaluation.py` one-command full benchmark
- [ ] All model downloads automated via HuggingFace hub
- [ ] Evaluation results CSV attached

---

## Feasibility Assessment

### ✅ Fully Feasible
- IsolationForest + EWMA anomaly detection — standard ML, fast
- Qdrant vector DB with IndexFlatIP — well-documented, performant
- BM25 + dense hybrid retrieval — proven approach (Telco-RAG validated)
- LangGraph multi-agent pipeline — production-ready library
- MiniLM cross-encoder reranking — ONNX, CPU-fast
- TeleQnA benchmarking — public dataset, clear eval protocol

### ⚠️ Medium Risk
- **FSE Training:** Only 20 training pairs — small dataset risk. Mitigate: aggressive data augmentation (noise injection, KPI scaling variants) to generate 500+ samples. Also use transfer learning from Gemma embedding space.
- **VRAM Budget:** Gemma-2-2B + Llama-3.2-3B simultaneously loaded — may exceed 10GB. Mitigate: load embedding model, embed batch, unload, then load LLM for generation. Use 4-bit GPTQ throughout.
- **PaddleOCR quality:** 3GPP diagrams are complex. Some text extraction may be noisy. Mitigate: OCR fallback only; main content is text, not diagrams.

### ❌ Not Feasible (Plan Adjusted)
- **Real-time streaming from live RAN:** No live network available. Use Simu5G CSV playback simulation. ✓ Already planned.
- **LoRA fine-tuning:** Tele-LLMs paper confirmed extremely low gradient norms; ineffective. ✓ Correctly excluded.
- **Vision-Language model for diagrams:** Beyond compute budget. ✓ PaddleOCR fallback planned.

---

## Phase 5.5 — Additions (Day 22–25)

> All 5 additions are now part of the core plan.

### Add 1 — Confidence-Calibrated Tiered Response (`fcrag/reason/tiers.py`)
- [ ] Add `ResponseTier` enum: HIGH / MEDIUM / INSUFFICIENT
- [ ] HIGH (faithfulness > 0.85): Full RCA with all citations
- [ ] MEDIUM (0.60–0.85): Partial RCA with flagged uncertain claims (`[UNCERTAIN]` tag)
- [ ] INSUFFICIENT (< 0.60): Return raw top-5 retrieved chunks, no LLM generation
- [ ] Update Validator Agent to emit tier alongside faithfulness score
- [ ] Update API response schema with `response_tier` field

### Add 2 — Feedback Loop / Active Learning Hook (`fcrag/feedback/collector.py`)
- [ ] After each analysis, log `{fault_vector, top1_clause, user_confirmed: bool}` to `data/feedback/feedback_log.jsonl`
- [ ] `fcrag/feedback/retrain_trigger.py`: when feedback_log has 100+ new confirmed pairs, trigger FSE re-train
- [ ] Script: `scripts/retrain_fse.py` — loads feedback pairs + original 20, runs training loop
- [ ] Track FSE version in `models/fse_version.txt`

### Add 3 — Retrieval Explanation Layer (`fcrag/retrieve/explainer.py`)
- [ ] For each top-5 chunk, generate a 1-sentence explanation: why was this retrieved?
- [ ] Template: "Retrieved because fault shows [KPI anomaly] which matches [clause topic] in [source]"
- [ ] Map anomaly KPI deltas → natural language triggers
- [ ] Attach explanation to each `RetrievedChunk` object
- [ ] Include in API response and Streamlit UI

### Add 4 — Multi-Cell Correlation (`fcrag/sense/correlator.py`)
- [ ] Maintain a 60s sliding window of AnomalyEvents across all cells
- [ ] If ≥3 cells alarm within 30s: trigger `MULTI_CELL_EVENT`
- [ ] Compute Pearson correlation across simultaneous KPI drops
- [ ] If correlation > 0.8: flag common upstream cause hypothesis (CU/DU failure)
- [ ] Inject correlation context into FSE query: `"Multi-cell HO failure, likely CU upstream fault"`
- [ ] Unit test: inject 3 simultaneous cell faults, verify correlation detected

### Add 5 — Streamlit Demo Dashboard (`demo/app.py`)
- [ ] Page 1 — Live KPI Monitor: real-time line chart (throughput, HO rate, PRB, latency, RRC retries)
- [ ] Anomaly Detected: red banner + severity badge popup
- [ ] Page 2 — Retrieval Results: expandable cards for top-5 chunks with explanation layer
- [ ] Page 3 — RCA Output: causal chain diagram (networkx → matplotlib), corrective actions table
- [ ] Sidebar: response tier badge (HIGH/MEDIUM/INSUFFICIENT), faithfulness score, latency breakdown
- [ ] Replay button: cycle through 5 pre-loaded Simu5G fault scenarios
- [ ] Run: `streamlit run demo/app.py`
