"""
fcrag/ingest/embedder.py — FCRAG 2.0 Embedding Pipeline (LangChain Integration)
================================================================================
Generates vector embeddings for text chunks using LangChain embedding classes.
Supports:
  1. local_sentence_transformer (via HuggingFaceEmbeddings)
  2. hf_inference_api (via HuggingFaceEndpointEmbeddings)
  3. gemma_tele (Custom LangChain Embeddings wrapper for Gemma hidden states)

Maintains a pickled disk cache for fast re-runs.
"""

import os
import sys
import pickle
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

# Find project root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

# Load environment variables
load_dotenv(ROOT / ".env")


def load_config() -> Dict[str, Any]:
    """Load the master configuration from config/settings.yaml."""
    config_path = ROOT / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class GemmaTeleEmbeddings(Embeddings):
    """Custom LangChain Embeddings wrapper for Gemma-2-2B-Tele hidden state extraction."""
    
    def __init__(self, model_name: str, hidden_layer_index: int, batch_size: int, device: str, dtype_str: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        self.batch_size = batch_size
        self.device = device
        self.hidden_layer_index = hidden_layer_index
        
        print(f"Loading local Gemma model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        torch_dtype = torch.float16 if dtype_str == "float16" and torch.cuda.is_available() else torch.float32
        device_map = "auto" if self.device == "cuda" and torch.cuda.is_available() else self.device
        
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import torch
        from tqdm import tqdm
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Gemma Embedding"):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            
            # Move inputs to correct device
            device = "cuda" if "cuda" in self.device and torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                target_state = hidden_states[self.hidden_layer_index]
                
                # Attention mask mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                input_mask_expanded = attention_mask.expand(target_state.size()).float()
                sum_embeddings = torch.sum(target_state * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled = sum_embeddings / sum_mask
                
                embeddings.extend(pooled.cpu().float().numpy().tolist())
                
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class FCRAGEmbeddings(Embeddings):
    """
    LangChain compatible embedding provider that orchestrates underlying models
    and adds robust, single-file disk caching for the FCRAG ingestion pipeline.
    """

    def __init__(self):
        self.config = load_config()
        self.embed_cfg = self.config["models"]["embedding"]
        self.provider = self.embed_cfg.get("provider", "local_sentence_transformer")
        self.model_name = self.embed_cfg.get("name", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = self.embed_cfg.get("device", "cpu")
        self.dim = self.embed_cfg.get("embedding_dim", 384)
        self.batch_size = self.embed_cfg.get("batch_size", 32)

        # Setup Cache path
        data_dir = ROOT / self.config["paths"].get("data_dir", "data")
        self.cache_dir = data_dir / "processed" / "chunks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_safe_name = f"{self.provider}_{self.model_name.replace('/', '_')}"
        self.cache_path = self.cache_dir / f"embeddings_cache_{model_safe_name}.pkl"
        
        self.cache: Dict[str, List[float]] = {}
        self._load_cache()

        self.core_embeddings: Embeddings = self._init_model()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache):,} cached embeddings from {self.cache_path.name}")
            except Exception as e:
                print(f"Error loading embedding cache: {e}. Starting fresh.")
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"Saved {len(self.cache):,} cached embeddings to {self.cache_path.name}")
        except Exception as e:
            print(f"Error saving embedding cache: {e}")

    def _init_model(self) -> Embeddings:
        """Initialize the core LangChain Embeddings instance based on provider."""
        print(f"Initializing LangChain embedding provider: {self.provider} ({self.model_name})")
        
        if self.provider == "local_sentence_transformer":
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'batch_size': self.batch_size}
            )
            
        elif self.provider == "hf_inference_api":
            token = os.environ.get("HF_TOKEN")
            if not token:
                print("[WARNING] HF_TOKEN is not set. Inference API may fail or be rate-limited.")
            return HuggingFaceEndpointEmbeddings(
                model=self.model_name,
                huggingfacehub_api_token=token
            )
            
        elif self.provider == "gemma_tele":
            return GemmaTeleEmbeddings(
                model_name=self.model_name,
                hidden_layer_index=self.embed_cfg.get("hidden_layer_index", -2),
                batch_size=self.batch_size,
                device=self.device,
                dtype_str=self.embed_cfg.get("dtype", "float16")
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents, using the disk cache to avoid redundant computation."""
        if not texts:
            return []

        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for idx, txt in enumerate(texts):
            norm_txt = " ".join(txt.split())
            if norm_txt in self.cache:
                results[idx] = self.cache[norm_txt]
            else:
                missing_indices.append(idx)
                missing_texts.append(norm_txt)

        if missing_texts:
            print(f"Computing embeddings for {len(missing_texts):,} missing texts...")
            computed = self.core_embeddings.embed_documents(missing_texts)
            
            for idx, emb in zip(missing_indices, computed):
                # Ensure it's a standard python list of floats
                emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
                
                norm_txt = missing_texts[missing_texts.index(" ".join(texts[idx].split()))]
                self.cache[norm_txt] = emb_list
                results[idx] = emb_list
                
            self._save_cache()

        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]
        
    # Alias to maintain backward compatibility with old script just in case
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


def run_embedder(verbose: bool = True):
    """Pre-compute embeddings for all chunks in the data directory."""
    print("\nStarting pre-computation of embeddings via LangChain for all collections...")
    import json
    from fcrag.ingest.chunker import CHUNK_OUTPUT
    
    embedder = FCRAGEmbeddings()
    collections = ["3gpp_specs", "simu5g_narratives", "alarm_history"]
    
    for coll in collections:
        jsonl_path = CHUNK_OUTPUT / f"{coll}.jsonl"
        if not jsonl_path.exists():
            print(f"[WARNING] Chunk file not found: {jsonl_path}. Skip.")
            continue
            
        print(f"\nProcessing collection: {coll}")
        texts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    texts.append(chunk["text"])
                    
        if texts:
            embedder.embed_documents(texts)
            print(f"Completed {coll} embeddings.")
        else:
            print(f"No texts found in {coll}.")


if __name__ == "__main__":
    run_embedder()
