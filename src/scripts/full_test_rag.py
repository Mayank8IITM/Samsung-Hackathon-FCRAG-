import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.reason.llm_client import FCRAGLLMClient
from fcrag.retrieve.retriever import HybridRetriever

def main():
    print("=" * 60)
    print("FCRAG 2.0 - End-to-End Test")
    print("=" * 60)
    
    # 1. Initialize Clients
    try:
        llm = FCRAGLLMClient()
        retriever = HybridRetriever()
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    # 2. Test Retrieval
    print("\n[1] Retrieving Knowledge Base Data...")
    query = "A3 offset too aggressive causing handover failure"
    print(f"User Query: '{query}'")
    
    try:
        results = retriever.retrieve(query, top_k=2)
        retrieved_context = ""
        for i, res in enumerate(results):
            print(f" -> Found Document {i+1} (Score: {res.rerank_score:.2f})")
            retrieved_context += f"Document {i+1}:\n{res.text}\n\n"
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        return

    # 3. Test LLM Generation with Guardrails
    print("\n[2] Generating RAG Answer with Guardrails...")
    
    # This is the GUARDRAIL PROMPT to prevent hallucination
    rag_prompt = (
        f"You are an expert telecom network analyst. Answer STRICTLY based on the provided context.\n"
        f"Do not use outside knowledge. If the context does not contain enough information to answer the query, reply with \"Insufficient data in the knowledge base. This prototype currently only supports TS 38.331, TS 38.300, TS 23.501, TS 23.502, TR 21.916, and TR 21.918.\"\n\n"
        f"Query: {query}\n\nContext:\n{retrieved_context}\n\nReasoned RCA:"
    )

    try:
        response = llm.generate(rag_prompt, max_tokens=150)
        print(f"\nFinal RAG Response:\n{response}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        
    print("\nTest completed!")

if __name__ == "__main__":
    main()
