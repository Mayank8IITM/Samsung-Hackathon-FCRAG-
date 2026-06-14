import pytest
from pathlib import Path
from fcrag.ingest.bm25_builder import BM25Builder, tokenize

def test_tokenize():
    """Verify that tokenization handles telecom standards properly."""
    text = "The TS 38.331 spec states Cell-46 has an anomaly!"
    tokens = tokenize(text)
    
    assert "ts" in tokens
    assert "38.331" in tokens
    assert "cell-46" in tokens
    assert "!" not in tokens
    assert "anomaly" in tokens

def test_bm25_mock_build_and_search(tmp_path, monkeypatch):
    """Test building and searching an index using a mock collection."""
    # Redirect INDEX_OUTPUT to temporary path
    import fcrag.ingest.bm25_builder
    monkeypatch.setattr(fcrag.ingest.bm25_builder, "INDEX_OUTPUT", tmp_path)
    
    # Mock CHUNK_OUTPUT and create a fake jsonl
    test_coll = "test_collection"
    test_jsonl = tmp_path / f"{test_coll}.jsonl"
    
    import json
    with open(test_jsonl, "w") as f:
        f.write(json.dumps({"text": "This is a document about TS 38.331", "clause_id": "TS_1"}) + "\n")
        f.write(json.dumps({"text": "Another document about completely unrelated stuff.", "clause_id": "TS_2"}) + "\n")
        f.write(json.dumps({"text": "Cell-46 is having a massive HO_FAILURE.", "clause_id": "TS_3"}) + "\n")
        
    monkeypatch.setattr(fcrag.ingest.bm25_builder, "CHUNK_OUTPUT", tmp_path)
    
    builder = BM25Builder([test_coll])
    count = builder.build_collection(test_coll)
    assert count == 3
    
    # Test searching
    results = builder.search(test_coll, "TS 38.331 failure")
    
    # Top result should be the first document
    assert len(results) > 0
    assert results[0]["clause_id"] == "TS_1"
    assert "bm25_score" in results[0]
