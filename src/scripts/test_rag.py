import sys
import os
from pathlib import Path

# Add project root to python path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.reason.llm_client import FCRAGLLMClient
from fcrag.retrieve.retriever import HybridRetriever

def main():
    print("="*60)
    print("FCRAG 2.0 - End-to-End Test")
    print("="*60)
    
    # 1. Test LLM Connection (Hugging Face API)
    print("\n[1] Testing LLM Connection...")
    try:
        llm = FCRAGLLMClient()
        test_prompt = "What does 3GPP say about HO_FAILURE? Give a one-sentence summary."
        print(f"Querying LLM: '{test_prompt}'")
        response = llm.generate(test_prompt, max_tokens=50)
        print(f"LLM Response:\n{response}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")

    # 2. Test Retrieval (Qdrant + BM25)
    print("\n[2] Testing Vector Retrieval...")
    try:
        retriever = HybridRetriever()
        query = "A3 offset too aggressive causing handover failure"
        print(f"Searching knowledge base for: '{query}'")
        
        results = retriever.retrieve(query, top_k=3)
        for i, res in enumerate(results):
            print(f"\nResult {i+1} (Score: {res.rerank_score:.2f})")
            print(f"Source: {res.clause_id} ({res.source_type})")
            print(f"Text: {res.text[:100]}...")
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        
    print("\nTest completed!")

if __name__ == "__main__":
    main()
