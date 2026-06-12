# FCRAG 2.0 — Product Requirements Document (PRD)
**Project:** Fault-Conditioned Retrieval-Augmented Generation  
**Team:** IIT Madras AgentX-10 (Mayank Singh · Ali Jawad)  
**Version:** 1.0 | Date: June 2026

---

## 1. Executive Summary

FCRAG is an **autonomous telecom network intelligence system** that eliminates manual fault diagnosis. It monitors 5G RAN KPI streams, auto-detects anomalies, autonomously generates retrieval queries (no human prompt), fetches relevant 3GPP/O-RAN specification evidence, and produces cited, explainable root-cause analyses with corrective action recommendations — all within **< 4 seconds end-to-end**.

### The Core Value Proposition
| Before FCRAG | After FCRAG |
|---|---|
| 35–55 min manual diagnosis | < 4s autonomous detection-to-recommendation |
| SME dependency for every fault | Zero SME needed for Level-1 diagnosis |
| 30% engineer time on PDF search | Automated retrieval with exact clause citations |
| Manual monitoring of select cells | All cells monitored simultaneously |
| Generic LLM hallucinations | Validator guardrail with INSUFFICIENT_CONTEXT fallback |

---

## 2. Problem Statement

Telecom RAN networks generate petabytes of operational data annually (KPI streams, alarm logs, config states, trace files). Network operations remain **heavily dependent on manual SME intervention**.

**Root Cause of the Gap:**
- Generic LLMs hallucinate 3GPP technical concepts
- Misinterpret standardized abbreviations (PRACH, PDCCH, CQI)
- Cannot anchor outputs to exact specification clause numbers
- All existing RAG systems are **reactive** (wait for human prompt)

**Hackathon KPI Targets:**
- MRR > 75%
- Recall@5 > 85%  
- Faithfulness > 90%
- End-to-End Latency < 4s

---

## 3. Goals & Non-Goals

### Goals
- G1: Build an event-driven fault detection system (no human prompt required)
- G2: Hybrid retrieval over 3GPP + O-RAN + synthetic Simu5G corpus
- G3: Multi-agent causal reasoning with full citation traceability
- G4: Meet all 4 hackathon KPIs simultaneously
- G5: Demonstrate live on single-GPU hardware

### Non-Goals
- Not a production-grade telecom OSS/BSS integration
- Not a real-time network controller (read-only recommendations)
- Not fine-tuning any base model from scratch
- Not a vision-language multimodal system (OCR fallback used instead)

---

## 4. User Personas

### Primary: Network Operations Engineer (NOC)
- Monitors 100s of cells simultaneously
- Currently: opens dashboards, correlates alarms, searches 3GPP PDFs
- **Need:** Instant root cause + recommended action without manual search

### Secondary: RAN Performance Analyst
- Reviews KPI trends, conducts post-mortems
- **Need:** Cited evidence from standards documents for remediation reports

### Tertiary: Hackathon Evaluator / Demo Reviewer
- Evaluates MRR, Recall@5, Faithfulness, and latency on benchmark
- **Need:** Reproducible, measurable, explainable system

---

## 5. Functional Requirements

### FR-1: KPI Ingestion & Anomaly Detection
- FR-1.1: Continuously ingest KPI streams: throughput, HO success rate, latency, PRB utilization, RRC retry metrics
- FR-1.2: Detect multivariate outliers using IsolationForest (< 100ms)
- FR-1.3: Detect temporal drift using EWMA detector (slow degradation)
- FR-1.4: Emit structured anomaly events with severity, KPI deltas, cell ID, timestamp

### FR-2: Fault Signature Encoding (FSE)
- FR-2.1: Map anomaly vectors → telecom retrieval embeddings using custom FSE neural network
- FR-2.2: Expand telecom acronyms via curated 3GPP dictionary before retrieval
- FR-2.3: Generate structured retrieval queries autonomously (no human input)
- FR-2.4: Rule-based fallback if anomaly data is insufficient

### FR-3: Hybrid Retrieval Pipeline
- FR-3.1: Dense retrieval using Gemma-2-2B-Tele embeddings (layer -2 hidden states)
- FR-3.2: Sparse retrieval using BM25 for exact technical term matching
- FR-3.3: Cross-encoder reranking using ms-marco-MiniLM-L-6-v2 (ONNX accelerated)
- FR-3.4: Retrieve from 3 simultaneous sources:
  - Specification Context (3GPP Release 16/18 clauses)
  - Operational Context (historical alarms & KPI deviations)
  - Synthetic Context (Simu5G failure scenarios)
- FR-3.5: Query augmentation with candidate answer synonyms (2–3.5% accuracy gain)
- FR-3.6: Achieve Recall@5 > 85%, MRR > 75% on TeleQnA benchmark

### FR-4: Multi-Agent Reasoning (LangGraph Pipeline)
- FR-4.1: Decomposer Agent — breaks complex fault queries into sub-questions
- FR-4.2: Retriever Agent — executes hybrid search, assembles tri-modal context
- FR-4.3: Reasoning Agent — performs graph-based RCA with exact citations
- FR-4.4: Validator Agent — checks every claim is grounded in retrieved context
- FR-4.5: Return INSUFFICIENT_CONTEXT if confidence < threshold (no hallucination)

### FR-5: Explainable Output
- FR-5.1: Exact 3GPP clause citations for every recommendation
- FR-5.2: Confidence scores per claim
- FR-5.3: Visual causal graph of root-cause chain
- FR-5.4: Prioritized corrective action list
- FR-5.5: Faithfulness > 90% verified by Validator Agent

### FR-6: Knowledge Base Construction
- FR-6.1: Ingest and chunk 3GPP TS 38.331, 38.321, 38.401, 38.413, 38.214 (~2000 pages)
- FR-6.2: Ingest O-RAN WG1/WG3 specifications (~500 pages)
- FR-6.3: Use 125-token chunks with overlap (research-validated optimal size)
- FR-6.4: OCR fallback (PaddleOCR) for diagrams/state machine images
- FR-6.5: Index in Qdrant using IndexFlatIP
- FR-6.6: Generate and embed 150 Simu5G synthetic fault resolution narratives

### FR-7: Benchmarking & Evaluation
- FR-7.1: Evaluate on TeleQnA (10,000 MCQs) — Standards Overview + Specifications tasks
- FR-7.2: Evaluate generation quality on Tele-Eval (750K Q&A pairs)
- FR-7.3: Validate retrieval quality on 20 custom fault→clause mappings (MRR/Recall@5)
- FR-7.4: Report end-to-end latency across 50 benchmark runs

---

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Latency | End-to-end < 4s; Anomaly detection < 100ms; Retrieval < 500ms; LLM < 2.5s |
| Accuracy | MRR > 75%, Recall@5 > 85%, Faithfulness > 90% |
| Hardware | Single GPU deployment (fits within VRAM using 4-bit GPTQ quantization) |
| Fallback | TinyLlama-1.1B-Tele-it fallback if Llama-3.2-3B-Tele-it is too slow |
| Hallucination | INSUFFICIENT_CONTEXT returned, never a fabricated answer |
| Reproducibility | All 50 benchmark runs logged with seed, version, and timing |
| Openness | All models from HuggingFace; all datasets publicly available |

---

## 7. KPIs & Success Metrics

### Retrieval Quality
- MRR (Mean Reciprocal Rank) > 0.75
- Recall@5 > 85%
- Precision@1 > 70%

### Generation Quality
- Faithfulness Score > 90% (Validator Agent)
- Answer Relevance > 80%
- Citation Coverage: every factual claim has ≥1 cited clause

### System Performance
- P50 End-to-End Latency < 4s
- P95 End-to-End Latency < 6s
- Anomaly Detection Latency < 100ms

### Business Impact (Demo Target)
- MTTR reduction: 55 min → < 4s
- SME escalations avoided: 100% of Level-1 faults
- Hallucination rate: < 10%

---

## 8. Constraints & Risks

| Constraint | Risk | Mitigation |
|---|---|---|
| Single-GPU demo environment | LLM too slow | TinyLlama fallback; report both metrics |
| No real operator data | Synthetic fault coverage gaps | Simu5G + 20 custom fault scenarios |
| OCR-only for diagrams | Diagram comprehension limited | PaddleOCR for text labels in diagrams |
| No LoRA fine-tuning | Model not telecom-specialized | Use pre-trained Tele-LLMs from HuggingFace |
| < 4s latency budget | Pipeline too slow | Component-level timing budgets enforced |
