import streamlit as st
import requests
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_agraph import agraph, Node, Edge, Config

# ==============================================================================
# UI Configuration
# ==============================================================================
st.set_page_config(
    page_title="FCRAG 2.0 | Telecom RCA",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    /* Gradient Header */
    .main-header {
        background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #8892b0;
        margin-bottom: 2rem;
    }
    /* Metric Cards styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #64ffda;
    }
    /* Action cards */
    .action-card {
        background-color: #112240;
        border-left: 4px solid #64ffda;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Simulation Scenarios
# ==============================================================================
SCENARIOS = {
    "HO_FAILURE (Handover Drop)": {
        "cell_id": "Cell-42",
        "severity": "HIGH",
        "kpi_snapshot": {
            "ho_success_rate_drop": 0.35,
            "throughput_drop_pct": 15.0,
            "latency_increase_ms": 12.0
        },
        "anomaly_score": 0.88,
        "kpi_name": "Handover Success Rate",
        "baseline": 98.0,
        "drop_val": 63.0
    },
    "PRB_CONGESTION (Capacity Spike)": {
        "cell_id": "Cell-99",
        "severity": "CRITICAL",
        "kpi_snapshot": {
            "prb_utilization_spike": 0.40,
            "latency_increase_ms": 85.0
        },
        "anomaly_score": 0.95,
        "kpi_name": "PRB Utilization (%)",
        "baseline": 45.0,
        "drop_val": 95.0
    },
    "LATENCY_SPIKE (Delay)": {
        "cell_id": "Cell-11",
        "severity": "MEDIUM",
        "kpi_snapshot": {
            "latency_increase_ms": 120.0,
            "throughput_drop_pct": 5.0
        },
        "anomaly_score": 0.75,
        "kpi_name": "User Plane Latency (ms)",
        "baseline": 15.0,
        "drop_val": 135.0
    }
}


# ==============================================================================
# Sidebar - Simulation Controls
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Samsung_Logo.svg/512px-Samsung_Logo.svg.png", width=150)
    st.markdown("### FCRAG 2.0 Simulator")
    st.markdown("Fault-Conditioned Retrieval-Augmented Generation for 5G/6G Networks.")
    st.markdown("---")
    
    st.markdown("#### Scenario Configuration")
    scenario_name = st.selectbox("Select Network Anomaly", list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_name]
    
    st.markdown("#### KPI Snapshot Payload")
    st.json(scenario["kpi_snapshot"])
    
    trigger = st.button("🚨 TRIGGER ANOMALY", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("Team: **IIT Madras AgentX-10**")


# ==============================================================================
# Main Content
# ==============================================================================
st.markdown('<h1 class="main-header">FCRAG 2.0 Network RCA</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-driven Root Cause Analysis and Remediation for Telecom Networks</p>', unsafe_allow_html=True)

# Placeholder for Live Chart
chart_placeholder = st.empty()

if not trigger:
    # Display baseline chart
    df = pd.DataFrame({
        "Time": pd.date_range(start="now", periods=20, freq="1min"),
        scenario["kpi_name"]: np.random.normal(scenario["baseline"], 2.0, 20)
    })
    fig = px.line(df, x="Time", y=scenario["kpi_name"], title=f"Live KPI Stream: {scenario['cell_id']}")
    fig.update_layout(template="plotly_dark", height=300)
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    st.info("👈 Select a scenario from the sidebar and click **Trigger Anomaly** to initiate the AI Reasoning pipeline.")

else:
    # Anomaly Animation
    with st.spinner("Simulating network stream..."):
        df = pd.DataFrame({
            "Time": pd.date_range(start="now", periods=30, freq="1min"),
            scenario["kpi_name"]: np.concatenate([
                np.random.normal(scenario["baseline"], 2.0, 20),
                np.random.normal(scenario["drop_val"], 5.0, 10)  # The spike/drop
            ])
        })
        fig = px.line(df, x="Time", y=scenario["kpi_name"], title=f"🚨 ANOMALY DETECTED: {scenario['cell_id']}")
        fig.add_vline(x=df["Time"].iloc[20], line_width=3, line_dash="dash", line_color="red")
        fig.update_layout(template="plotly_dark", height=300)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    time.sleep(1) # Dramatic pause

    # --------------------------------------------------------------------------
    # API Call
    # --------------------------------------------------------------------------
    st.markdown("### Agent Reasoning Pipeline")
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("1/4: Decomposer Agent classifying fault...")
    progress_bar.progress(25)
    time.sleep(0.5)
    
    status_text.text("2/4: Retriever Agent fetching tri-modal context (3GPP, Simu5G, Alarms)...")
    progress_bar.progress(50)
    
    # Actually call the API
    api_url = "http://127.0.0.1:8000/analyze-fault"
    payload = {
        "cell_id": scenario["cell_id"],
        "severity": scenario["severity"],
        "kpi_snapshot": scenario["kpi_snapshot"],
        "mode": "auto",
        "anomaly_score": scenario["anomaly_score"]
    }
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        status_text.text("3/4: Reasoning Agent generating causal chain...")
        progress_bar.progress(75)
        time.sleep(0.5)
        
        status_text.text("4/4: Validator Agent checking faithfulness against context...")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        status_text.empty()
        progress_bar.empty()
        
        # ----------------------------------------------------------------------
        # Results Dashboard
        # ----------------------------------------------------------------------
        st.success(f"✅ RCA Complete in {data.get('latency_ms', 0)}ms")
        
        # Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", data.get("status", "N/A"))
        c2.metric("Confidence", f"{data.get('confidence', 0.0):.1%}")
        c3.metric("Faithfulness", f"{data.get('faithfulness_score', 0.0):.1%}")
        c4.metric("Latency", f"{data.get('latency_ms', 0)} ms")
        
        st.markdown("---")
        
        # Two-column layout for Graph and Text
        col_left, col_right = st.columns([1.5, 1])
        
        with col_left:
            st.markdown("### 🕸️ Causal Graph")
            
            # Parse NetworkX JSON to agraph format
            nx_data = data.get("causal_graph", {})
            nodes = []
            edges = []
            
            for n in nx_data.get("nodes", []):
                # Handle ID correctly
                node_id = str(n.get("id", n.get("label", "Unknown")))
                label = str(n.get("label", node_id))
                desc = str(n.get("description", ""))
                
                # Assign distinct colors based on node type
                color = "#26D0CE" # default cyan
                if "SYMPTOM" in label:
                    color = "#FF4B4B" # red
                elif "ROOT_CAUSE" in label:
                    color = "#64ffda" # green
                    
                nodes.append(Node(
                    id=node_id,
                    label=label,
                    size=25,
                    title=desc,
                    color=color,
                    shape="dot"
                ))
            
            # NetworkX exports edges as 'links' or 'edges'
            link_key = "links" if "links" in nx_data else "edges"
            for idx, e in enumerate(nx_data.get(link_key, [])):
                src = str(e.get("source", ""))
                tgt = str(e.get("target", ""))
                edges.append(Edge(source=src, target=tgt, type="STRAIGHT", color="#8892b0"))
                
            config = Config(width=600, height=400, directed=True, 
                            physics=True, hierarchical=False)
            
            if nodes:
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.warning("No causal graph data returned.")
                
            with st.expander("Raw Output Package (JSON)"):
                st.json(data)

        with col_right:
            st.markdown("### 📝 Executive Summary")
            st.info(data.get("rca_summary", "No summary available."))
            
            st.markdown("### 🛠️ Corrective Actions")
            for action in sorted(data.get("corrective_actions", []), key=lambda x: x.get("priority", 99)):
                st.markdown(f"""
                <div class="action-card">
                    <b>Priority {action.get('priority')}:</b> {action.get('action')}<br>
                    <span style="color:#8892b0; font-size:0.9em">📖 Ref: {action.get('spec_reference', 'N/A')}</span>
                </div>
                """, unsafe_allow_html=True)
                
            if data.get("citations"):
                st.markdown("### 📚 Source Citations")
                for cit in data.get("citations", []):
                    st.markdown(f"- `{cit}`")
                    
    except requests.exceptions.ConnectionError:
        st.error("🚨 Could not connect to FastAPI server. Is it running at `http://127.0.0.1:8000`?")
        st.info("Run `uv run uvicorn fcrag.api.main:app --reload` in another terminal.")
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
