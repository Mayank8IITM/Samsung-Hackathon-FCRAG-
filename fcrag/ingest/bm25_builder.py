"""
fcrag/ingest/bm25_builder.py — FCRAG 2.0 BM25 Index Builder
===========================================================
Builds an exact-keyword (sparse) search index using BM25Okapi.
Maintains a tokenization scheme optimized for telecom standards
(preserving internal periods and hyphens like "38.331" or "Cell-46").
"""

import json
import pickle
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

# Find project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fcrag.ingest.chunker import CHUNK_OUTPUT

# Directory to save built indexes
INDEX_OUTPUT = ROOT / "data" / "processed" / "indexes"
INDEX_OUTPUT.mkdir(parents=True, exist_ok=True)

def tokenize(text: str) -> List[str]:
    """
    Tokenizes text for BM25.
    Lowercase everything, and extract words.
    Crucially, preserves internal dots and hyphens for telecom terms (e.g. '38.331', '5G-110').
    """
    if not text:
        return []
    
    # Matches words, including those with internal dots or hyphens
    # Example: "TS 38.331, Cell-46!" -> ["ts", "38.331", "cell-46"]
    tokens = re.findall(r'\b\w+(?:[.-]\w+)*\b', text.lower())
    return tokens


class BM25Builder:
    """Builds and serializes BM25 keyword indexes for collections."""

    def __init__(self, collections: List[str] = None):
        self.collections = collections or ["3gpp_specs", "oran_specs", "simu5g_narratives", "alarm_history"]

    def build_collection(self, collection_name: str) -> int:
        """
        Reads a collection's chunks, builds the BM25 index,
        and saves it along with payload mapping to disk.
        """
        jsonl_path = CHUNK_OUTPUT / f"{collection_name}.jsonl"
        if not jsonl_path.exists():
            print(f"[WARNING] Chunk file not found: {jsonl_path}. Skipping BM25 build.")
            return 0

        # Load chunks
        docs = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))

        if not docs:
            print(f"No documents found for {collection_name}.")
            return 0

        print(f"Tokenizing {len(docs):,} documents for '{collection_name}' BM25 Index...")
        tokenized_corpus = []
        payloads = []
        
        for i, doc in enumerate(docs):
            # Tokenize the core text
            tokens = tokenize(doc.get("text", ""))
            
            # Combine text and metadata into payload for retrieval mapping
            payload = {
                "id": i,
                "text": doc.get("text", ""),
                "source_type": doc.get("source_type", ""),
                "source_file": doc.get("source_file", ""),
                "clause_id": doc.get("clause_id", ""),
            }
            # Add extra metadata if present
            if isinstance(doc.get("metadata"), dict):
                payload.update(doc["metadata"])
                
            tokenized_corpus.append(tokens)
            payloads.append(payload)

        print(f"Fitting BM25Okapi model for '{collection_name}'...")
        bm25_model = BM25Okapi(tokenized_corpus)

        # Save model and payloads
        save_path = INDEX_OUTPUT / f"bm25_{collection_name}.pkl"
        
        bundle = {
            "model": bm25_model,
            "payloads": payloads,
            "collection_name": collection_name
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)
            
        print(f"Successfully saved BM25 index to {save_path.name}")
        return len(docs)

    def search(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Convenience method to load an index and run a test search.
        (Primarily used for testing; the main hybrid retriever will handle actual serving).
        """
        save_path = INDEX_OUTPUT / f"bm25_{collection_name}.pkl"
        if not save_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {save_path}")
            
        with open(save_path, "rb") as f:
            bundle = pickle.load(f)
            
        model: BM25Okapi = bundle["model"]
        payloads = bundle["payloads"]
        
        query_tokens = tokenize(query)
        # get_top_n returns the raw documents (which we didn't store in the corpus directly, 
        # rank_bm25 expects the corpus elements).
        # We stored tokens in rank_bm25, so we should get scores instead.
        scores = model.get_scores(query_tokens)
        
        # Sort by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                result = payloads[idx].copy()
                result["bm25_score"] = float(scores[idx])
                results.append(result)
                
        return results


def run_bm25_build():
    """Main orchestrator for building all BM25 indexes."""
    print("\n" + "=" * 60)
    print("PHASE 1.4 — BM25 Keyword Indexing")
    print("=" * 60)
    
    builder = BM25Builder()
    total_indexed = 0
    
    for coll in builder.collections:
        count = builder.build_collection(coll)
        total_indexed += count
        
    print("\n" + "-" * 60)
    print(f"✅ BM25 indexing complete! Total documents indexed: {total_indexed:,}")
    print("-" * 60)

if __name__ == "__main__":
    run_bm25_build()
