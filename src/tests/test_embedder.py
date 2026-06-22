import os
import shutil
import pytest
from pathlib import Path
from fcrag.ingest.embedder import FCRAGEmbeddings, load_config

# Setup temporary cache directory for testing
TEST_CACHE_DIR = Path(__file__).resolve().parent.parent / "temp_cache"


@pytest.fixture(autouse=True)
def setup_temp_cache(monkeypatch):
    """Fixture to mock cache directory and isolate cache testing."""
    TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Mock self.cache_dir and self.cache_path in FCRAGEmbeddings.__init__
    original_init = FCRAGEmbeddings.__init__
    
    def mock_init(self):
        self.config = load_config()
        self.embed_cfg = self.config["models"]["embedding"]
        self.provider = "local_sentence_transformer"  # Force local for testing
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = "cpu"
        self.dim = 384
        self.batch_size = 4
        
        self.cache_dir = TEST_CACHE_DIR
        self.cache_path = self.cache_dir / "test_cache.pkl"
        
        self.cache = {}
        self._load_cache()
        self.core_embeddings = self._init_model()
        
    monkeypatch.setattr(FCRAGEmbeddings, "__init__", mock_init)
    
    yield
    
    # Cleanup temp cache dir
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)


def test_embedder_local_init():
    """Test that the embedder can be initialized with local_sentence_transformer."""
    embedder = FCRAGEmbeddings()
    assert embedder.provider == "local_sentence_transformer"
    assert embedder.core_embeddings is not None
    assert embedder.dim == 384


def test_embedder_generation():
    """Test generating embeddings for sample texts."""
    embedder = FCRAGEmbeddings()
    test_texts = ["Hello, this is a test.", "FCRAG telecom network intelligence."]
    
    embeddings = embedder.embed_documents(test_texts)
    
    # Check shape & type
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) == 384
    assert isinstance(embeddings[0][0], float)


def test_embedder_caching():
    """Test that embedding cache functions correctly."""
    embedder = FCRAGEmbeddings()
    text = "Unique test sentence for cache validation."
    
    # 1. First run: should compute and cache
    assert text not in embedder.cache
    embs_1 = embedder.embed_documents([text])
    assert text in embedder.cache
    assert len(embs_1) == 1
    
    # 2. Check cache file was saved
    assert embedder.cache_path.exists()
    
    # 3. Reload embedder to check load from cache
    embedder_2 = FCRAGEmbeddings()
    assert text in embedder_2.cache
    
    embs_2 = embedder_2.embed_documents([text])
    
    assert embs_1 == embs_2
