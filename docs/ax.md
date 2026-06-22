# Samsung EnnovateX AX Hackathon: FCRAG 2.0 Technical Report

This document details how FCRAG 2.0 leverages **open-weight models** and an **agentic development workflow** to execute autonomous Root Cause Analysis (RCA) in 3GPP network environments. It is written specifically to satisfy the Phase 2 requirements for the EnnovateX AX Hackathon.

---

## 1. Open-Weight Models Strategy

To ensure data privacy, transparency, and compliance with the hackathon rules, FCRAG exclusively uses open-source models executed locally or via Hugging Face inference.

### Reasoning Engine: `Llama-3.2-3B-Tele-it`
- **Why selected:** Telecommunication RCA requires an understanding of dense acronyms (PRACH, RRC, PUSCH). Generic models fail here. This specific fine-tune (`Tele-it`) was trained on telecom specifications. At 3B parameters, it provides sub-4-second inference times, crucial for real-time Network Operations Center (NOC) dashboards.

### Dense Retrieval Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- **Why selected:** We needed a model that could run 100% locally on CPU without Docker. This 22M parameter model embeds 3GPP text chunks in milliseconds and stores them directly into a local Qdrant database.

### Strict Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Why selected:** Bi-encoders (like the one above) often retrieve semantically similar but technically incorrect clauses. Cross-Encoders pass the query and document through the transformer attention layers simultaneously. This drastically reduced LLM hallucination in our tests by filtering out false positives before they reached the context window.

---

## 2. Agentic Architecture & Multi-Agent Orchestration

FCRAG 2.0 abandons traditional linear scripts in favor of an **Agentic Directed Acyclic Graph (DAG)** built using LangGraph (`src/fcrag/reason/graph.py`).

The system passes a unified `FCRAGState` object (defined in `src/fcrag/reason/state.py`) between four distinct agent nodes:

1. **`Decomposer` (`agents/decomposer.py`):** Translates raw numeric anomalies (e.g., `-29% HO_Success_Rate`) into natural language queries for the vector database.
2. **`RetrieveContext` (`agents/retriever_agent.py`):** Interfaces with the `HybridRetriever`. It doesn't just blindly query; it merges dense vectors and BM25 sparse lists, applying the Cross-Encoder.
3. **`Reason` (`agents/reasoning_agent.py`):** Assembles the prompt and calls the Hugging Face `Llama-3.2-3B-Tele-it` model.
4. **`Validate` (`agents/validator.py`):** Parses the output, checking if the LLM followed structural instructions (Problem, Root Cause, Recommendations) and scores faithfulness.

### Communication
Agents do not call each other directly. They mutate the `FCRAGState` dictionary and return it to the LangGraph executor, which handles edge routing. If the `Decomposer` fails, the graph halts before wasting expensive LLM tokens.

---

## 3. Tool Chaining & The Hybrid RAG Pipeline

The retrieval pipeline (`src/fcrag/retrieve/retriever.py`) relies heavily on tool chaining:

1. **Sparse Retrieval (Rank-BM25):** We implemented this because Dense models struggle with exact keyword matching (e.g., finding exactly "TS 38.331 Section 5.5.4" instead of "Section 5.5.5").
2. **Dense Retrieval (Qdrant):** Captures the semantic intent (e.g., mapping "call drops" to "Radio Link Failure").
3. **Reciprocal Rank Fusion:** The results from tools #1 and #2 are merged.
4. **CE Reranker:** The merged list is trimmed to the Top 30, scored by the Cross-Encoder, and strictly truncated to the Top 5.

### Context Optimization and Grounding
To prevent the "Lost-in-the-Middle" phenomenon, we strictly limit the context window to the Top 5 reranked clauses. `src/fcrag/ingest/chunker.py` uses a sliding window (500 tokens, 50 overlap) that explicitly injects the 3GPP Section Header into the metadata.

Because of this, the LLM is explicitly grounded. The prompt (`src/fcrag/reason/agents/reasoning_agent.py`) demands: *"Answer STRICTLY based on the provided context... If context is insufficient, reply 'Insufficient data in the knowledge base.'"* This prevents the model from guessing.

---

## 4. Agentic Development Setup

The development of Phase 2 was heavily accelerated using autonomous agentic tools.

- **Google Antigravity (AGY) & Gemini 3.1 Pro:** We utilized an AGY-powered IDE that had direct terminal access. 
- **Autonomous Refactoring:** The entire codebase was reorganized from a flat structure to a modern `src/` layout. An agent wrote a custom Python regex script (`fix_paths_safe.py`) to crawl 35+ files and dynamically rewrite `Path(__file__)` logic to prevent imports from breaking.
- **Iterative UI Generation:** We instructed the coding assistant to build a "futuristic NOC dashboard" in Streamlit. The agent iteratively designed the layout, added glassmorphism CSS, embedded Plotly sparklines, and wrote the `hex_to_rgba` logic to fix a Plotly color rendering bug—all through autonomous debugging loops inside the IDE.

---

## 5. Retrospective

### What Worked Exceptionally Well
- **Tele-IT Fine-tunes:** Using `Llama-3.2-3B-Tele-it` was a massive breakthrough. Generic models simply hallucinate when faced with dense 3GPP acronyms.
- **Hybrid Fusion:** Adding BM25 was mandatory. Dense vectors alone failed to match specific protocol messages like `RRCReconfigurationComplete`.
- **Agentic IDEs:** Allowing an AI agent to run `pytest` in the terminal and self-correct its own Python bugs reduced UI development time by roughly 70%.

### What Didn't Work (Bottlenecks & Failed Experiments)
- **Massive Rerankers:** We initially tried using a large 1B+ parameter reranker. It took 15 seconds to evaluate 50 documents, which destroyed the "real-time" requirement of a NOC dashboard. We dropped down to `ms-marco-MiniLM-L-6-v2` and limited the reranking pool to 30 documents.
- **Context Overflow:** Feeding 20 chunks to the 3B model caused it to lose track of the actual anomaly and just summarize the 3GPP spec. Restricting it to a hyper-relevant Top-5 fixed this.

---

## 6. Future Work
- **Dedicated Validator Agent:** Currently, the validator is a python parsing script. Future versions will use a smaller secondary LLM to judge the primary LLM's faithfulness before showing it to the user.
- **Model Context Protocol (MCP):** Implementing an MCP server to allow the LLM to autonomously query live network database APIs (e.g., Prometheus) instead of relying on static CSV files.
- **Adaptive Retrieval:** Teaching the LangGraph to loop back and re-query the vector DB if the `Reason` node determines the initial context was insufficient.
