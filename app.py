import streamlit as st
import json
import time
import random
import sys
from pathlib import Path

# Add project root to path so we can import fcrag
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fcrag.retrieve.retriever import HybridRetriever
from fcrag.reason.llm_client import FCRAGLLMClient

# --- Page Config ---
st.set_page_config(
    page_title="FCRAG 2.0 NOC Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Dark mode NOC aesthetic */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Primary button (Anomaly Simulator) */
    .stButton>button[kind="primary"] {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(45deg, #ff4b4b, #ff904b);
        color: white;
        font-weight: bold;
        border: None;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
        transform: translateY(-1px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }
    
    /* Retrieved Clause Cards */
    .clause-card {
        background: rgba(0, 200, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00c8ff;
        margin-bottom: 15px;
        font-size: 0.9em;
    }
    .clause-header {
        color: #00c8ff;
        font-weight: bold;
        margin-bottom: 5px;
        font-size: 1.1em;
    }
    
    /* Title Gradient */
    .gradient-text {
        font-weight: 800;
        font-size: 2.5em;
        background: -webkit-linear-gradient(45deg, #ff4b4b, #ff904b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
@st.cache_resource(show_spinner=False)
def init_system():
    # Load the core FCRAG components (only once per session)
    retriever = HybridRetriever()
    llm = FCRAGLLMClient()
    
    # Load custom scenarios for the Anomaly Simulator
    scenarios_path = ROOT / "data" / "custom_scenarios" / "fault_clause_mapping.json"
    with open(scenarios_path) as f:
        scenarios = json.load(f)
        
    return retriever, llm, scenarios

with st.spinner("Initializing FCRAG 2.0 Engine (Loading Vector DB & Models)..."):
    retriever, llm, scenarios = init_system()

# --- Sidebar: Anomaly Simulator & System Status ---
with st.sidebar:
    st.title("📡 NOC Controls")
    
    st.markdown("### Anomaly Simulator")
    st.caption("Simulate an incoming network fault alert from the downstream monitoring pipeline.")
    
    # Randomly pick a scenario and trigger the run flag
    if st.button("🚨 Detect Network Anomaly", type="primary"):
        scenario = random.choice(scenarios)
        st.session_state.current_query = scenario["fault_description"]
        st.session_state.scenario_data = scenario
        st.session_state.run_analysis = True
    
    st.markdown("---")
    st.markdown("### System Status")
    
    tier_name = llm._tier_name()
    db_status = "Qdrant (Local) - Online 🟢"
    
    st.markdown(f'<div class="metric-card"><b>LLM Engine:</b><br>{tier_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><b>Vector DB:</b><br>{db_status}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><b>Retrieval Mode:</b><br>Hybrid + CrossEncoder</div>', unsafe_allow_html=True)

# --- Main Dashboard ---
st.markdown('<p class="gradient-text">FCRAG 2.0: Automated RCA Dashboard</p>', unsafe_allow_html=True)
st.markdown("Intelligent Root Cause Analysis via 3GPP Retrieval-Augmented Generation.")

# Initialize chat history array
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "System Online. Waiting for anomaly detection or manual query."}
    ]

# Display all previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Catch manual input
user_query = st.chat_input("Enter manual fault query...")
if user_query:
    st.session_state.current_query = user_query
    st.session_state.scenario_data = None  # Reset scenario data if manual
    st.session_state.run_analysis = True

# Process query if flagged
if st.session_state.get("run_analysis", False):
    query = st.session_state.current_query
    st.session_state.run_analysis = False # Reset flag
    
    # 1. Echo the query into the chat
    query_msg = f"**[ANOMALY ALERT]** {query}"
    st.session_state.messages.append({"role": "user", "content": query_msg})
    with st.chat_message("user"):
        st.markdown(query_msg)
        
    # 2. Begin Assistant processing
    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        
        # UI Status Container to show steps
        status = st.status("Analyzing Fault Sequence...", expanded=True)
        
        # --- Step 1: Retrieval ---
        status.write("🔍 Extracting KPIs and matching against 3GPP specifications...")
        t_ret_start = time.perf_counter()
        results = retriever.retrieve(query, top_k=5)
        ret_time = time.perf_counter() - t_ret_start
        
        if not results:
            status.update(label="Analysis Failed", state="error", expanded=False)
            st.error("No relevant 3GPP specifications found in the database.")
            st.stop()
            
        status.write(f"✅ Retrieved {len(results)} relevant clauses via Hybrid Search ({ret_time*1000:.0f}ms).")
        
        # --- Step 2: Context Assembly ---
        status.write("🧠 Contextualizing data for LLM Reasoning...")
        context_str = "\\n\\n".join([f"[Source: {res.clause_id}]\\n{res.text}" for res in results])
        
        # --- Step 3: LLM Generation ---
        status.write("⚙️ Generating Root Cause Analysis...")
        prompt = (
            f"You are an expert telecom network analyst. Answer STRICTLY based on the provided context.\\n"
            f"Do not use outside knowledge. If the context does not contain enough information to answer the query, reply with \\\"Insufficient data in the knowledge base. This prototype currently only supports TS 38.331, TS 38.300, TS 23.501, TS 23.502, TR 21.916, and TR 21.918.\\\"\\n\\n"
            f"Query: {query}\\n\\nContext:\\n{context_str}\\n\\nReasoned RCA:"
        )
        
        t_llm_start = time.perf_counter()
        rca_report = llm.generate(prompt, max_tokens=256)
        llm_time = time.perf_counter() - t_llm_start
        
        total_time = time.perf_counter() - t0
        status.update(label=f"Analysis Complete ({total_time:.2f}s)", state="complete", expanded=False)
        
        # --- Display LLM Result ---
        st.markdown("### 📋 Automated Root Cause Analysis")
        st.markdown(rca_report)
        st.session_state.messages.append({"role": "assistant", "content": f"### 📋 Automated Root Cause Analysis\\n{rca_report}"})
        
        # --- Explainability Panel ---
        st.markdown("---")
        with st.expander("🔍 Under the Hood (Explainability Dashboard)", expanded=False):
            col1, col2 = st.columns(2)
            col1.metric("Retrieval Latency", f"{ret_time*1000:.0f} ms")
            col2.metric("LLM Latency", f"{llm_time:.2f} s")
            
            # If this was a simulated scenario, show what we expected
            if st.session_state.get("scenario_data"):
                sc_data = st.session_state.scenario_data
                st.markdown(f"**Target Fault Type:** `{sc_data['fault_type']}`")
                st.markdown(f"**Ground Truth Clauses:** {', '.join(sc_data['relevant_clauses'])}")
            
            st.markdown("#### Retrieved 3GPP Context")
            for i, res in enumerate(results):
                # Clean up text for HTML display
                safe_text = res.text[:300].replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f'''
                <div class="clause-card">
                    <div class="clause-header">Top-{i+1} Match: {res.clause_id}</div>
                    <b>Reranker Score:</b> {res.rerank_score:.3f} | <b>Type:</b> {res.source_type}<br><br>
                    <i>{safe_text}...</i>
                </div>
                ''', unsafe_allow_html=True)
