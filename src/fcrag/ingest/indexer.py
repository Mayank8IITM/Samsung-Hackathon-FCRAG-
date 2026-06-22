"""
fcrag/ingest/indexer.py — FCRAG 2.0 Qdrant Vector Indexer
===========================================================
Reads the parsed text chunks and cached embeddings, then uploads
them into the specified Qdrant collections.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# Find project root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.ingest.embedder import load_config, FCRAGEmbeddings
from fcrag.ingest.chunker import CHUNK_OUTPUT


class Indexer:
    """Orchestrates the creation and population of Qdrant vector collections."""

    def __init__(self):
        self.config = load_config()
        self.qdrant_cfg = self.config.get("qdrant", {})
        self.embedder = FCRAGEmbeddings()
        self.client = self._init_client()

    def _init_client(self) -> QdrantClient:
        """Initialize the Qdrant client based on configuration."""
        in_memory = self.qdrant_cfg.get("in_memory", True)
        
        if in_memory:
            print("Initializing QdrantClient (In-Memory mode)")
            return QdrantClient(location=":memory:")
        else:
            persist_dir = self.qdrant_cfg.get("persist_directory", "data/qdrant_db")
            db_path = ROOT / persist_dir
            db_path.mkdir(parents=True, exist_ok=True)
            print(f"Initializing QdrantClient (Persistent disk mode: {db_path})")
            return QdrantClient(path=str(db_path))

    def init_collections(self):
        """Delete and recreate the 4 core vector collections."""
        collections_cfg = self.qdrant_cfg.get("collections", {})
        
        for key, coll_info in collections_cfg.items():
            name = coll_info["name"]
            vector_size = coll_info["vector_size"]
            
            distance_str = coll_info.get("distance", "Cosine").upper()
            distance = getattr(models.Distance, distance_str, models.Distance.COSINE)

            print(f"Creating collection '{name}' with size {vector_size}...")
            
            # Recreate collection to ensure a clean slate
            if self.client.collection_exists(collection_name=name):
                self.client.delete_collection(collection_name=name)
                
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )

    def index_collection(self, collection_name: str, batch_size: int = 100) -> int:
        """Read chunks, retrieve embeddings, and upload to the given collection."""
        jsonl_path = CHUNK_OUTPUT / f"{collection_name}.jsonl"
        if not jsonl_path.exists():
            print(f"[WARNING] Chunk file not found: {jsonl_path}. Skipping.")
            return 0

        # Load all chunks
        docs = []
        keywords = ["handover", "ho", "a3", "prb", "congestion", "utilization", "latency", "delay", "throughput"]
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    text_lower = doc.get("text", "").lower()
                    if any(kw in text_lower for kw in keywords):
                        docs.append(doc)
        
        if not docs:
            print(f"No documents found for {collection_name} matching demo keywords.")
            return 0
            
        docs = docs[:100]

        print(f"Retrieving embeddings for {len(docs):,} chunks in {collection_name}...")
        texts = [doc["text"] for doc in docs]
        
        # This will load instantly if cache is already built
        embeddings = self.embedder.embed_documents(texts)
        
        # Upload points in batches
        print(f"Uploading {len(docs):,} points to Qdrant collection '{collection_name}'...")
        points = []
        
        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            # Build payload
            payload = {
                "source_type": doc.get("source_type", ""),
                "source_file": doc.get("source_file", ""),
                "clause_id": doc.get("clause_id", ""),
                "text": doc.get("text", "")
            }
            # Merge additional metadata
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                payload.update(metadata)
                
            # Use UUIDs or just integer IDs. Qdrant accepts ints or UUID strings.
            # Using chunk_index if possible, else just loop index.
            # Qdrant client handles automatic UUID generation if id isn't specified,
            # but we can use simple ints for ease.
            points.append(
                models.PointStruct(
                    id=i,
                    vector=emb,
                    payload=payload
                )
            )

        # Batch upload
        for i in tqdm(range(0, len(points), batch_size), desc=f"Uploading {collection_name}"):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
            
        print(f"Successfully indexed {len(points):,} documents into {collection_name}.")
        return len(points)


def run_indexer():
    """Main execution function to index all collections."""
    print("\n" + "=" * 60)
    print("PHASE 1.3 — Qdrant Indexing")
    print("=" * 60)
    
    indexer = Indexer()
    
    print("\n1. Initializing collections...")
    indexer.init_collections()
    
    print("\n2. Populating collections...")
    collections_to_index = ["3gpp_specs", "oran_specs", "simu5g_narratives", "alarm_history"]
    
    total_indexed = 0
    for coll in collections_to_index:
        count = indexer.index_collection(coll)
        total_indexed += count
        
    print("\n" + "-" * 60)
    print(f"✅ Qdrant indexing complete! Total documents indexed: {total_indexed:,}")
    print("-" * 60)


if __name__ == "__main__":
    run_indexer()
