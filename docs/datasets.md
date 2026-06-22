# FCRAG 2.0 Datasets and Knowledge Base

The accuracy of a Retrieval-Augmented Generation (RAG) system is entirely dependent on the quality of its knowledge base. FCRAG uses a combination of official 3GPP specifications and simulated network fault data to ground the LLM's reasoning.

## 1. Primary Knowledge Base: 3GPP Specifications

The core intelligence of the system relies on the official 3GPP Technical Specifications (Rel 15-18).

### Sources Ingested
- **TS 38.331:** Radio Resource Control (RRC) protocol specifications.
- **TS 38.300:** NR; Overall description; Stage-2.
- **TS 23.501:** System architecture for the 5G System (5GS).
- **TS 23.502:** Procedures for the 5G System (5GS).
- **TR 21.916 / TR 21.918:** Supplementary technical reports.

### Ingestion & Chunking Strategy
Raw 3GPP PDFs are notoriously difficult for LLMs to parse due to massive tables, footnotes, and multi-page clause spanning. FCRAG solves this via `src/fcrag/ingest/chunker.py`:
1. The text is parsed from PDF to `.txt` (`data/processed/3gpp_text/`).
2. A sliding-window semantic chunker splits the text into ~500-token chunks with 50-token overlaps.
3. Crucially, the chunker attempts to preserve `Section X.Y.Z` headers as metadata. This ensures that when the LLM generates an RCA report, it can accurately cite the specific 3GPP clause (e.g., "According to TS 38.331 Section 5.5.4...").

## 2. Telemetry & Fault Data: Simu5G / OAI KPM

To test the system, we need simulated faults to query against the specifications. 

### Data Sources
- **OAI RAN KPM Dataset:** Provides the baseline "healthy" metrics for throughput and latency.
- **Simu5G Fault Logs:** `data/simu5g/kpi_logs/` contains `.csv` timeseries data simulating various failure states:
  - `HO_FAILURE`
  - `PRB_CONGESTION`
  - `PRACH_CONGESTION`
  - `INTERFERENCE`

### Scenario Generation
Because the LLM requires natural language to reason over, the raw numeric KPIs must be translated. `src/fcrag/ingest/simu5g_generator.py` acts as a data bridge, taking the raw `.csv` KPI drops and generating "Fault Narratives" (e.g., *“Cell 42 experienced a 29% drop in HO Success Rate alongside a latency spike.”*).

These narratives are stored in `data/custom_scenarios/fault_clause_mapping.json` and are directly linked to the "Ground Truth" 3GPP clause that theoretically solves the problem. This allows us to rigorously benchmark the system's retrieval accuracy.

## 3. Storage and Indexing

### Persistent Storage
- **Sparse Indexes:** Stored as `pickle` files (`.pkl`) in `data/processed/indexes/` for instant memory loading via `rank_bm25`.
- **Dense Indexes:** Stored natively in `data/qdrant_db/` using Qdrant's local persistent storage mode. This removes the need for Docker containers and allows instant booting of the `src/app.py` dashboard.
- **Raw Text Chunks:** Stored as `.jsonl` files in `data/processed/chunks/` to maintain the mapping between the Qdrant Vector ID and the actual human-readable 3GPP text.
