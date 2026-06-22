# FCRAG 2.0 Open-Weight Models

FCRAG 2.0 exclusively uses open-weight models. This architectural constraint ensures that the telecom RCA pipeline can be deployed in highly secure, air-gapped NOC environments without transmitting sensitive proprietary telemetry or network architectures to closed APIs (like OpenAI).

## 1. The Reasoning Engine (LLM)

### [`AliMaatouk/Llama-3.2-3B-Tele-it`](https://huggingface.co/AliMaatouk/Llama-3.2-3B-Tele-it)
- **Role:** Synthesis and Root Cause Analysis.
- **Architecture:** Transformer-based Large Language Model (Llama-3 architecture).
- **Parameters:** 3 Billion.
- **Why Selected:** 
  - **Domain Adaptation:** This specific variant has been fine-tuned on telecommunications data (`Tele-it`). It possesses an intrinsic understanding of 3GPP acronyms (RRC, PRACH, PRB, HO) that generic models often struggle with.
  - **Latency:** At only 3B parameters, it provides exceptionally fast inference speeds, which is critical for real-time anomaly analysis.
  - **Context Limit:** 8K tokens. By aggressively filtering the context using our Cross-Encoder, we comfortably fit the most highly-relevant 3GPP clauses into this window without truncating.
- **Integration:** Implemented via the Hugging Face Inference API in `src/fcrag/reason/llm_client.py`.

## 2. The Embedding Engine (Dense Retrieval)

### [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Role:** Generates dense vector representations of both the 3GPP text chunks and the incoming anomaly queries.
- **Architecture:** Miniature BERT-style transformer.
- **Parameters:** ~22 Million.
- **Why Selected:**
  - **Speed:** It is optimized for maximum throughput. It maps sentences and paragraphs to a 384-dimensional dense vector space in milliseconds.
  - **Resource Efficiency:** Can easily be loaded locally via PyTorch/Sentence-Transformers without needing dedicated GPU hardware (`src/fcrag/ingest/embedder.py`).
- **Tradeoffs:** Because it is a smaller, generic model, it occasionally struggles with highly specialized telecom acronyms (e.g., treating "HO" as an English word rather than "Handover"). This weakness is completely mitigated by pairing it with the Sparse BM25 retriever.

## 3. The Reranking Engine (Cross-Encoder)

### [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- **Role:** Acts as the strict final judge of relevance. It takes the fused top-30 results from the Dense and Sparse retrievers and re-evaluates them.
- **Architecture:** Cross-Encoder (passes the query and the document simultaneously through the transformer layers).
- **Why Selected:**
  - **Precision:** Unlike Bi-Encoders (like the embedding model above) which compare two distinct vectors via cosine similarity, a Cross-Encoder analyzes the *attention* between the query tokens and the document tokens. This leads to drastically higher accuracy.
  - **Hallucination Prevention:** By establishing a strict relevance threshold, it ensures that irrelevant documents fetched by the weaker initial retrievers are discarded before they reach the LLM context window (`src/fcrag/retrieve/reranker.py`).
- **Latency Considerations:** Cross-Encoders are computationally heavy. To mitigate latency, FCRAG only feeds the top 30 pre-filtered documents to this model, rather than the entire 3GPP database.
