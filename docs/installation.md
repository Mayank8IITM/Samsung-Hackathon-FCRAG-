# Installation & Execution Guide

This guide provides step-by-step instructions to set up the FCRAG 2.0 environment, build the local vector indices, and launch the interactive dashboard.

## 1. Prerequisites
- **Python Version:** Python 3.11+ is strictly required due to dependency constraints with modern deep learning and LangChain libraries.
- **System Requirements:** A minimum of 8GB RAM is recommended to load the embedding and cross-encoder models into memory. 

## 2. Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Samsung-Hackathon-FCRAG-
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   Because the codebase utilizes a modern `src/` layout, install the package in editable mode:
   ```bash
   pip install -e .
   ```
   *(This ensures all internal imports like `from fcrag.retrieve import...` resolve correctly across the project.)*

## 3. Configuration

FCRAG relies on the Hugging Face Inference API for the core `Llama-3.2-3B-Tele-it` model.

1. **Create an environment file:**
   Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```
2. **Add your Hugging Face API key:**
   ```env
   HUGGINGFACE_API_KEY=hf_your_api_key_here
   ```

## 4. Local Database Initialization (Qdrant & BM25)

Before analyzing anomalies, you must populate the local databases with the 3GPP specifications. FCRAG uses Qdrant in persistent disk mode, meaning it stores vectors directly in the `data/qdrant_db/` folder without requiring a Docker container.

Run the ingestion script:
```bash
python src/scripts/ingest_all.py
```

**What this script does:**
- Parses raw 3GPP `.txt` files from `data/processed/3gpp_text/`.
- Chunks them into optimal sizes for retrieval.
- Downloads `all-MiniLM-L6-v2` locally and computes dense vectors, storing them in Qdrant.
- Builds the sparse lexical indexes using `rank_bm25` and saves them to `data/processed/indexes/`.

*(Note: The first run will download approximately 1GB of model weights to your local Hugging Face cache.)*

## 5. Running the Application

### The NOC Interactive Dashboard
To launch the Streamlit dashboard:
```bash
streamlit run src/app.py
```
This will open the web interface (usually at `http://localhost:8501`). On the first run, it will initialize the Cross-Encoder model.

### Command-Line Backend Benchmark
To run the automated test suite that benchmarks latency, MRR, and Recall metrics without the UI:
```bash
python src/scripts/test_rag.py
```

## Troubleshooting

- **Qdrant Lock Errors (`Storage folder is already accessed`)**: Qdrant local disk mode only permits one active process at a time. Ensure you are not running `test_rag.py` at the exact same time as the `streamlit run src/app.py` server. Stop the Streamlit server before running background ingestion scripts.
- **Missing Module `fcrag`**: This means the package was not installed in editable mode. Run `pip install -e .` from the project root.
- **API Key Errors**: Ensure `.env` is loaded. The application uses `python-dotenv` to inject the key into the environment securely.
