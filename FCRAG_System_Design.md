# FCRAG 2.0 — System Design Document
**Project:** Fault-Conditioned Retrieval-Augmented Generation  
**Team:** IIT Madras AgentX-10  
**Version:** 1.0 | Date: June 2026

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FCRAG 2.0 SYSTEM                                │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌────────────────────────────────┐    │
│  │KPI Stream│───▶│ Stage 1  │    │         Stage 3                │    │
│  │Simulator │    │  SENSE   │    │         RETRIEVE               │    │
│  │(Simu5G)  │    │          │    │  ┌──────────────────────────┐  │    │
│  └──────────┘    │IsoForest │    │  │ Dense: Gemma-2-2B-Tele   │  │    │
│                  │  EWMA    │    │  │ Sparse: BM25             │  │    │
│                  └────┬─────┘    │  │ Rerank: MiniLM-L-6-v2   │  │    │
│                       │          │  └──────────────────────────┘  │    │
│                  ┌────▼─────┐    │         ▲         ▲            │    │
│                  │ Stage 2  │    │  ┌──────┘         └──────┐     │    │
│                  │  ENCODE  │───▶│  │Qdrant Collections     │     │    │
│                  │   FSE    │    │  │ • 3GPP specs          │     │    │
│                  │  TAAE    │    │  │ • O-RAN specs         │     │    │
│                  └──────────┘    │  │ • Simu5G narratives   │     │    │
│                                  │  │ • Historical alarms   │     │    │
│                                  │  └───────────────────────┘     │    │
│                                  └──────────┬─────────────────────┘    │
│                                             │                           │
│                            ┌────────────────▼──────────────────────┐   │
│                            │           Stage 4: REASON             │   │
│                            │         LangGraph Pipeline            │   │
│                            │  Decomposer → Retriever → Reasoning   │   │
│                            │          → Validator Agent            │   │
│                            └────────────────┬──────────────────────┘   │
│                                             │                           │
│                            ┌────────────────▼──────────────────────┐   │
│                            │         Stage 5: EXPLAIN & ACT        │   │
│                            │  • Cited RCA Report                   │   │
│                            │  • Causal Graph (NetworkX)            │   │
│                            │  • Corrective Actions                 │   │
│                            │  • Confidence Scores                  │   │
│                            └───────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Stage 1 — SENSE (Anomaly Detection)

**Module:** `fcrag/sense/detector.py`

```
KPI Stream → Feature Extraction → Dual Detector → Anomaly Event
```

| Sub-component | Implementation | Latency Budget |
|---|---|---|
| KPI Ingestion | CSV/streaming parser | < 10ms |
| IsolationForest | scikit-learn, n_estimators=100 | < 80ms |
| EWMA Drift Detector | Custom sliding window | < 20ms |
| Anomaly Event Emitter | Structured JSON event | < 5ms |

**Anomaly Event Schema:**
```json
{
  "event_id": "uuid4",
  "timestamp": "ISO-8601",
  "cell_id": "string",
  "severity": "CRITICAL|HIGH|MEDIUM|LOW",
  "kpi_deltas": {
    "throughput_drop_pct": float,
    "ho_success_rate_drop": float,
    "prb_utilization_spike": float,
    "rrc_retry_increase": float,
    "latency_increase_ms": float
  },
  "anomaly_score": float,
  "drift_detected": bool,
  "raw_kpi_vector": [float, ...]
}
```

---

### 2.2 Stage 2 — ENCODE (Fault Signature Encoder)

**Module:** `fcrag/encode/fse.py` + `fcrag/encode/taae.py`

#### Fault Signature Encoder (FSE) — Custom Neural Network
```
Input:  Anomaly vector (5–15 KPI features)
        → Linear(64) → ReLU → Dropout(0.1)
        → Linear(128) → ReLU
        → Linear(embedding_dim=2048)  ← matches Gemma-2-2B-Tele hidden size
Output: Retrieval embedding vector
```

**Training Strategy:**
- Train FSE using contrastive loss on (anomaly_vector, relevant_3GPP_chunk) pairs
- 20 manually-mapped fault scenarios used as training + validation set
- Cosine similarity as learning objective

#### TAAE (Telecom Acronym & Augmentation Engine)
```
Input:  FSE embedding + anomaly metadata
Step 1: Acronym expansion via curated dict (PRACH→Physical Random Access Channel, etc.)
Step 2: Candidate answer synonym generation
Step 3: Query string assembly for BM25
Output: {dense_query_vector, bm25_query_string, augmented_terms[]}
```

---

### 2.3 Stage 3 — RETRIEVE (Hybrid Retrieval Pipeline)

**Module:** `fcrag/retrieve/retriever.py`

#### Qdrant Collections Design

| Collection Name | Source | Chunk Size | Index Type | # Vectors |
|---|---|---|---|---|
| `3gpp_specs` | TS 38.331/321/401/413/214 | 125 tokens + 25 overlap | IndexFlatIP | ~60,000 |
| `oran_specs` | WG1/WG3 docs | 125 tokens + 25 overlap | IndexFlatIP | ~15,000 |
| `simu5g_narratives` | Synthetic fault logs | Full narrative | IndexFlatIP | ~150 |
| `alarm_history` | Historical alarm patterns | Per-alarm record | IndexFlatIP | variable |

#### Retrieval Flow
```
Query → Dense Search (top-20) + BM25 (top-20)
     → Merge & Deduplicate
     → Cross-Encoder Rerank (top-5)
     → Tri-Modal Context Assembly
     → Context Package (spec + operational + synthetic)
```

#### Chunking Pipeline (`fcrag/ingest/chunker.py`)
- Sentence-aware splitting at 125 tokens
- 25-token sliding overlap between chunks
- Metadata preserved: {source_doc, section, page_num, clause_id}
- PaddleOCR applied on images: text labels extracted and appended as text chunk
- Gemma-2-2B-Tele (layer -2 hidden states) used for embedding generation

---

### 2.4 Stage 4 — REASON (LangGraph Multi-Agent Pipeline)

**Module:** `fcrag/reason/graph.py`

#### Agent Graph (LangGraph StateGraph)
```
START
  │
  ▼
[Decomposer Agent]
  • Breaks fault context into sub-queries
  • Identifies fault category (HO failure / congestion / RRC / PHY layer)
  │
  ▼
[Retriever Agent]
  • Executes hybrid retrieval for each sub-query
  • Assembles tri-modal context package
  │
  ▼
[Reasoning Agent] ← Llama-3.2-3B-Tele-it (primary) / TinyLlama-1.1B (fallback)
  • Graph-based root cause analysis
  • Builds causal chain: Symptom → Trigger → Root Cause → Recommended Action
  • Produces structured output with clause citations
  │
  ▼
[Validator Agent]
  • Checks every factual claim against retrieved context
  • Computes faithfulness score (claim overlap with source chunks)
  • If score < 0.7: returns INSUFFICIENT_CONTEXT
  • If score ≥ 0.7: approves and emits final response
  │
  ▼
END
```

#### Shared State Schema (LangGraph TypedDict)
```python
class FCRAGState(TypedDict):
    anomaly_event: dict
    sub_queries: list[str]
    retrieved_contexts: list[RetrievedChunk]
    causal_chain: list[CausalNode]
    claims: list[Claim]
    citations: list[Citation]
    faithfulness_score: float
    final_response: str | Literal["INSUFFICIENT_CONTEXT"]
    latency_breakdown: dict[str, float]
```

---

### 2.5 Stage 5 — EXPLAIN & ACT (Output Generation)

**Module:** `fcrag/explain/reporter.py`

**Output Package:**
```json
{
  "rca_summary": "Root cause: A3 handover offset too aggressive (-3dB). Ref: TS 38.331 §5.5.4.4",
  "causal_chain": [
    {"node": "HO_FAILURE", "cause": "A3_OFFSET_TOO_AGGRESSIVE", "evidence": "TS 38.331 §5.5.4.4"},
    {"node": "A3_OFFSET_TOO_AGGRESSIVE", "cause": "PARAMETER_MISCONFIGURATION", "evidence": "O-RAN WG3 §6.2"}
  ],
  "corrective_actions": [
    {"priority": 1, "action": "Increase A3 offset from -3dB to -1dB", "spec_reference": "TS 38.331 §5.5.4.4"},
    {"priority": 2, "action": "Enable HO history logging for Cell-42", "spec_reference": "TS 38.401 §8.3"}
  ],
  "citations": ["TS 38.331 §5.5.4.4", "O-RAN WG3 §6.2", "TS 38.401 §8.3"],
  "faithfulness_score": 0.94,
  "confidence": 0.87,
  "latency_ms": 3240,
  "synthetic_precedent": "Simu5G Scenario 7: HO failure at Cell-42, similar A3 offset issue resolved in 120s"
}
```

---

## 3. Data Flow (End-to-End)

```
[KPI CSV / Stream]
        │
        ▼
[IsolationForest + EWMA]  ──── anomaly_event (JSON)
        │
        ▼
[FSE Neural Network]  ──── dense_query_vector (2048-dim)
[TAAE]                ──── bm25_query_string, augmented_terms
        │
        ▼
[Qdrant]     ←── IndexFlatIP search (top-20 dense + top-20 sparse)
[BM25 Index] ←── Inverted index search
        │
        ▼
[Cross-Encoder Reranker (MiniLM ONNX)]  ──── reranked top-5
        │
        ▼
[Tri-Modal Context Package]
  ├── Spec chunks (3GPP/O-RAN)
  ├── Alarm history matches
  └── Simu5G synthetic precedents
        │
        ▼
[LangGraph: Decomposer → Retriever → Reasoning → Validator]
        │
        ▼
[Structured RCA Output + Causal Graph + Cited Actions]
```

---

## 4. Infrastructure & Hardware

### Single-GPU Deployment (Target)
| Component | Resource | Notes |
|---|---|---|
| Gemma-2-2B-Tele (embedding) | GPU, ~6GB VRAM | 4-bit GPTQ quantization |
| Llama-3.2-3B-Tele-it (LLM) | GPU, ~8GB VRAM | 4-bit GPTQ; vLLM + PagedAttention |
| TinyLlama-1.1B-Tele-it | GPU, ~3GB VRAM | Fallback, fast |
| MiniLM-L-6-v2 (reranker) | CPU (ONNX) | ~100MB, CPU-efficient |
| Qdrant (vector DB) | RAM | In-process or local Docker |
| IsolationForest | CPU | scikit-learn |
| BM25 | CPU | rank_bm25 library |

### VRAM Budget (Single GPU)
```
Llama-3.2-3B (4-bit)  ≈  2GB active
Gemma-2-2B (4-bit)    ≈  1.5GB active
KV cache (vLLM)       ≈  2–3GB
Overhead              ≈  1GB
─────────────────────────────────
Total                 ≈  7–8GB  → fits RTX 3080/A10G (10GB+)
```

---

## 5. File Structure

```
fcrag/
├── ingest/
│   ├── chunker.py          # 125-token chunker + PaddleOCR
│   ├── embedder.py         # Gemma-2-2B-Tele embedding pipeline
│   ├── indexer.py          # Qdrant collection creation & population
│   └── bm25_builder.py     # BM25 index construction
├── sense/
│   ├── kpi_stream.py       # KPI ingestion (CSV / streaming)
│   ├── detector.py         # IsolationForest + EWMA
│   └── event_schema.py     # AnomalyEvent dataclass
├── encode/
│   ├── fse.py              # Fault Signature Encoder (PyTorch)
│   ├── fse_trainer.py      # Contrastive training loop
│   └── taae.py             # Telecom Acronym & Augmentation Engine
├── retrieve/
│   ├── retriever.py        # Hybrid retrieval orchestrator
│   ├── dense.py            # Qdrant dense search
│   ├── sparse.py           # BM25 retrieval
│   └── reranker.py         # Cross-encoder reranking (ONNX)
├── reason/
│   ├── graph.py            # LangGraph StateGraph definition
│   ├── agents/
│   │   ├── decomposer.py
│   │   ├── retriever_agent.py
│   │   ├── reasoning_agent.py
│   │   └── validator.py
│   └── llm_client.py       # vLLM client wrapper (primary + fallback)
├── explain/
│   ├── reporter.py         # Output package assembly
│   ├── causal_graph.py     # NetworkX causal graph builder
│   └── formatter.py        # JSON + human-readable formatter
├── eval/
│   ├── teleqna_eval.py     # MRR / Recall@5 on TeleQnA benchmark
│   ├── faithfulness_eval.py
│   ├── latency_bench.py    # 50-run timing benchmark
│   └── custom_eval.py      # 20-fault custom scenario evaluation
├── api/
│   ├── main.py             # FastAPI app
│   └── schemas.py
├── config/
│   └── settings.yaml       # All tunable parameters
├── data/
│   ├── 3gpp/               # Raw 3GPP PDFs
│   ├── oran/               # O-RAN PDFs
│   ├── simu5g/             # Simu5G CSV logs
│   └── custom_scenarios/   # 20 fault→clause mappings
└── tests/
    ├── test_detector.py
    ├── test_retrieval.py
    ├── test_agents.py
    └── test_e2e.py
```

---

## 6. API Contract

### POST /analyze-fault
**Request:**
```json
{
  "cell_id": "Cell-42",
  "kpi_snapshot": {
    "throughput_mbps": 12.3,
    "ho_success_rate": 0.71,
    "prb_utilization": 0.94,
    "rrc_retries": 18,
    "latency_ms": 87
  },
  "mode": "auto"  // "auto" | "manual_query"
}
```

**Response:**
```json
{
  "status": "RCA_COMPLETE" | "INSUFFICIENT_CONTEXT",
  "rca_summary": "string",
  "causal_chain": [...],
  "corrective_actions": [...],
  "citations": [...],
  "faithfulness_score": float,
  "latency_ms": int
}
```

### GET /health
Returns system status, model load state, Qdrant collection stats.

### POST /benchmark/teleqna
Runs TeleQnA evaluation subset; returns MRR, Recall@5, Accuracy.

---

## 7. Evaluation Architecture

```
TeleQnA (10K MCQs)
    │
    ▼
[FCRAG Retriever] → top-5 chunks
    │
    ▼
[MRR Calculator] → MRR > 0.75 ✓
[Recall@5 Calculator] → Recall > 0.85 ✓
    │
[FCRAG Reasoning Agent] → generated answer
    │
    ▼
[Faithfulness Scorer] → faithfulness > 0.90 ✓

Custom 20-fault Dataset
    │
    ▼
[Retriever] → verify correct 3GPP clause in top-5 → Recall@5 per fault type
```
