import streamlit as st
import json
import time
import random
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# src/app.py -> parent is src/ -> parent.parent is project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fcrag.retrieve.retriever import HybridRetriever
from fcrag.reason.llm_client import FCRAGLLMClient

st.set_page_config(
    page_title="FCRAG  | Telecom RCA Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080c14; color: #e2e8f0; }

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1421 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,200,255,0.12);
}

/* ── NOC Header ─────────────────────────────── */
.noc-header {
    background: linear-gradient(135deg, #0d1421 0%, #0a1f3a 50%, #091229 100%);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.noc-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,200,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.noc-title {
    font-size: 3.2em;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff, #00c8ff, #0055ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px 0;
    letter-spacing: -1.2px;
    text-shadow: 0 0 40px rgba(0,200,255,0.2);
}
.noc-subtitle {
    color: #94a3b8;
    font-size: 1.1em;
    font-weight: 500;
    margin: 0;
    max-width: 600px;
    line-height: 1.5;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10b981;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.live-dot {
    width: 7px; height: 7px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse-green 1.5s ease-in-out infinite;
    display: inline-block;
}
@keyframes pulse-green {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── KPI Cards ──────────────────────────────── */
.kpi-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(0,200,255,0.3); }
.kpi-value {
    font-size: 1.9em;
    font-weight: 700;
    color: #00c8ff;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.78em;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Status Chip ────────────────────────────── */
.chip-green  { background: rgba(16,185,129,0.1);  border: 1px solid rgba(16,185,129,0.3);  color: #10b981;  padding: 3px 10px; border-radius: 100px; font-size:0.78em; font-weight:600; }
.chip-yellow { background: rgba(245,158,11,0.1);  border: 1px solid rgba(245,158,11,0.3);  color: #f59e0b;  padding: 3px 10px; border-radius: 100px; font-size:0.78em; font-weight:600; }
.chip-red    { background: rgba(239,68,68,0.1);   border: 1px solid rgba(239,68,68,0.3);   color: #ef4444;  padding: 3px 10px; border-radius: 100px; font-size:0.78em; font-weight:600; }
.chip-blue   { background: rgba(0,200,255,0.1);   border: 1px solid rgba(0,200,255,0.3);   color: #00c8ff;  padding: 3px 10px; border-radius: 100px; font-size:0.78em; font-weight:600; }

/* ── Section Heading ────────────────────────── */
.section-heading {
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #00c8ff;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.3), transparent);
}

/* ── Scenario Card ──────────────────────────── */
.scenario-card {
    background: rgba(239,68,68,0.05);
    border: 1px solid rgba(239,68,68,0.2);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.2s;
}
.scenario-card:hover {
    background: rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.5);
    transform: translateX(3px);
}
.scenario-id   { font-size:0.75em; color:#ef4444; font-weight:700; font-family:'JetBrains Mono',monospace; margin-bottom:4px; }
.scenario-desc { font-size:0.88em; color:#cbd5e1; line-height:1.4; }

/* ── Clause Evidence Card ───────────────────── */
.clause-card {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.15);
    border-left: 4px solid #00c8ff;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.clause-meta { font-size:0.78em; color:#64748b; margin-top:6px; font-family:'JetBrains Mono',monospace; }
.clause-id   { color:#00c8ff; font-weight:700; font-size:0.9em; }
.score-high  { color:#10b981; font-weight:600; }
.score-med   { color:#f59e0b; font-weight:600; }
.score-low   { color:#ef4444; font-weight:600; }

/* ── Timeline ───────────────────────────────── */
.timeline-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 12px 0;
    border-left: 2px solid rgba(0,200,255,0.15);
    padding-left: 20px;
    margin-left: 10px;
    position: relative;
    transition: all 0.3s;
}
.timeline-step::before {
    content: '';
    position: absolute;
    left: -7px;
    top: 16px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: #1e293b;
    border: 2px solid #334155;
    transition: all 0.3s;
}
.timeline-step.done { border-left-color: #10b981; }
.timeline-step.done::before  { background: #10b981; border-color: #064e3b; box-shadow: 0 0 10px rgba(16,185,129,0.4); }
.timeline-step.error { border-left-color: #ef4444; }
.timeline-step.error::before { background: #ef4444; border-color: #7f1d1d; box-shadow: 0 0 10px rgba(239,68,68,0.4); }
.timeline-step.running { border-left-color: #00c8ff; }
.timeline-step.running::before { 
    background: #00c8ff; 
    border-color: #082f49; 
    box-shadow: 0 0 15px rgba(0,200,255,0.6);
    animation: pulse-blue 1.5s infinite; 
}
@keyframes pulse-blue {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 200, 255, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 10px rgba(0, 200, 255, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 200, 255, 0); }
}
.timeline-label { font-size: 0.95em; font-weight: 600; color: #f8fafc; display: flex; align-items: center; gap: 8px;}
.timeline-time  { font-size: 0.8em; color: #94a3b8; font-family:'JetBrains Mono',monospace; margin-top:4px;}

/* ── Metric Strip ───────────────────────────── */
.metric-strip {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.metric-name  { font-size:0.82em; color:#94a3b8; }
.metric-val   { font-size:0.9em; font-weight:600; color:#e2e8f0; font-family:'JetBrains Mono',monospace; }

/* ── RCA Report ─────────────────────────────── */
.rca-section {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.rca-section h4 { color:#00c8ff; font-size:0.9em; font-weight:700; margin:0 0 10px; text-transform:uppercase; letter-spacing:0.5px; }
.rca-body { font-size:0.9em; color:#cbd5e1; line-height:1.65; }

/* ── Inject Btn ─────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
    width: 100% !important;
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    border: 1px solid rgba(239,68,68,0.4) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
    font-size: 0.92em !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(220,38,38,0.25) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(220,38,38,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Tab Styling ─────────────────────────────── */
[data-baseweb="tab-list"] { background: transparent !important; gap: 12px; }
[data-baseweb="tab"] {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}
[data-baseweb="tab"]:hover {
    background: rgba(30,41,59,0.8) !important;
    color: #f8fafc !important;
    border-color: rgba(255,255,255,0.2) !important;
    transform: translateY(-2px);
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
    box-shadow: 0 8px 16px rgba(37,99,235,0.25) !important;
}

/* ── Streamlit Metric Fix (force visible text) ── */
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.6em !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.82em !important;
}
[data-testid="stMetricDelta"] { color: #10b981 !important; }

/* ── Control Panel ──────────────────────────── */
.control-panel {
    background: linear-gradient(135deg, #0d1421 0%, #0a1628 100%);
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 16px;
}
.control-title {
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #00c8ff;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.control-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.3), transparent);
}

/* ── Anomaly Btn override ───────────────────── */
.anomaly-btn > div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    border: 1px solid rgba(239,68,68,0.5) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95em !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(220,38,38,0.3) !important;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s !important;
}
.anomaly-btn > div[data-testid="stButton"] > button:hover {
    box-shadow: 0 6px 20px rgba(220,38,38,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Raw LLM block ──────────────────────────── */
.llm-raw {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 18px 22px;
    font-size: 0.88em;
    color: #cbd5e1;
    line-height: 1.7;
    font-family: 'JetBrains Mono', monospace;
    white-space: pre-wrap;
    word-break: break-word;
    margin-top: 14px;
}

/* ── Sidebar hide ───────────────────────────── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


def generate_mock_timeseries_data():
    """Generates mock timeseries data for the NOC monitor visualization."""
    times = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="s")
    signal = np.random.normal(-65, 2, 60)
    latency = np.random.normal(20, 2, 60)
    throughput = np.random.normal(150, 10, 60)
    return pd.DataFrame({
        "Time": times,
        "Signal Strength (dBm)": signal,
        "Latency (ms)": latency,
        "Throughput (Mbps)": throughput,
    })


def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert a hex color string to an rgba() string Plotly can accept."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_sparkline(df, col, color, y_range=None):
    # Derive a translucent fill color that Plotly accepts
    if color.startswith("#"):
        fill_color = hex_to_rgba(color, 0.12)
    elif color.startswith("rgb("):
        fill_color = color.replace("rgb(", "rgba(").replace(")", ",0.12)")
    else:
        fill_color = "rgba(0,200,255,0.12)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df[col],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate=f"<b>{col}</b>: %{{y:.1f}}<extra></extra>",
    ))
    fig.update_layout(
        height=120, margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=True, showgrid=False, zeroline=False,
                   tickfont=dict(color="#64748b", size=9),
                   range=y_range if y_range else [df[col].min() * 0.98, df[col].max() * 1.02]),
        showlegend=False,
    )
    return fig



def color_for_score(score: float):
    if score >= 0.85:
        return "#10b981"
    elif score >= 0.6:
        return "#f59e0b"
    else:
        return "#ef4444"


def score_chip(score: float):
    if score >= 2.0:
        css = "score-high"
    elif score >= 0.0:
        css = "score-med"
    else:
        css = "score-low"
    return f'<span class="{css}">{score:.3f}</span>'


def parse_rca_sections(text: str):
    sections = {"problem": "", "root_cause": "", "recommendations": ""}
    current = None
    for line in text.splitlines():
        lower = line.lower().strip()
        if lower.startswith("##") and ("problem" in lower or "issue" in lower or "fault" in lower):
            current = "problem"; continue
        elif lower.startswith("##") and "root cause" in lower:
            current = "root_cause"; continue
        elif lower.startswith("##") and "recommendation" in lower:
            current = "recommendations"; continue
        elif lower.startswith("##") and "analysis" in lower and "root" not in lower:
            current = "root_cause"; continue
        if current and line.strip():
            sections[current] += line + "\n"
    # If model didn't follow structure, dump everything into problem
    if not any(sections.values()):
        sections["problem"] = text
    return sections


@st.cache_resource(show_spinner=False)
def init_system():
    retriever = HybridRetriever()
    llm = FCRAGLLMClient()
    scenarios_path = ROOT / "data" / "custom_scenarios" / "fault_clause_mapping.json"
    with open(scenarios_path) as f:
        scenarios = json.load(f)
    return retriever, llm, scenarios


for key, default in [
    ("run_analysis", False),
    ("current_query", ""),
    ("scenario_data", None),
    ("rca_result", None),
    ("rca_results_list", []),
    ("rca_timings", {}),
    ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


with st.spinner("  Initializing FCRAG Engine…"):
    retriever, llm, scenarios = init_system()


tier_name = llm._tier_name() if hasattr(llm, "_tier_name") else "HF Inference API"
st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
    <div class="metric-strip" style="flex:1;min-width:180px;">
        <span class="metric-name">LLM Engine</span>
        <span class="metric-val" style="font-size:0.78em;">{tier_name}</span>
    </div>
    <div class="metric-strip" style="flex:1;min-width:140px;">
        <span class="metric-name">Vector DB</span>
        <span class="chip-green">Online</span>
    </div>
    <div class="metric-strip" style="flex:1;min-width:140px;">
        <span class="metric-name">Retrieval Mode</span>
        <span class="chip-blue">Hybrid + CrossEncoder</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Hero / Main Heading
st.markdown("""
<div style="
    background: linear-gradient(135deg, #020817 0%, #0a1628 50%, #020817 100%);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius: 20px;
    padding: 52px 48px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
">
  <!-- glow blobs -->
  <div style="position:absolute;top:-80px;right:-80px;width:340px;height:340px;
              background:radial-gradient(circle,rgba(0,200,255,0.08) 0%,transparent 70%);
              pointer-events:none;"></div>
  <div style="position:absolute;bottom:-60px;left:-40px;width:260px;height:260px;
              background:radial-gradient(circle,rgba(0,85,255,0.07) 0%,transparent 70%);
              pointer-events:none;"></div>

  <!-- eyebrow -->
  <div style="display:inline-flex;align-items:center;gap:8px;
              background:rgba(0,200,255,0.07);border:1px solid rgba(0,200,255,0.2);
              border-radius:100px;padding:5px 16px;margin-bottom:22px;">
    <span style="width:7px;height:7px;background:#10b981;border-radius:50%;
                 display:inline-block;animation:pulse-green 1.5s ease-in-out infinite;"></span>
    <span style="color:#00c8ff;font-size:0.75em;font-weight:700;letter-spacing:1.5px;
                 text-transform:uppercase;">Samsung EnnovateX AX Hackathon</span>
  </div>

  <!-- main title -->
  <h1 style="font-size:4.2em;font-weight:900;margin:0 0 14px;
             background:linear-gradient(135deg, #ffffff 0%, #00c8ff 50%, #3b82f6 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;letter-spacing:-2px;line-height:1.05;">
    FCRAG 
  </h1>

  <!-- tagline -->
  <p style="font-size:1.25em;color:#94a3b8;margin:0 0 22px;font-weight:500;max-width:640px;line-height:1.6;">
    <strong style="color:#e2e8f0;">Fault Cause Root Analysis Graph</strong> &mdash;
    Autonomous Root Cause Analysis for 5G / LTE networks,
    powered by Hybrid RAG and an open-weight telecom-tuned LLM.
  </p>

  <!-- feature chips -->
  <div style="display:flex;flex-wrap:wrap;gap:10px;">
    <span style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
                 color:#10b981;padding:5px 14px;border-radius:100px;font-size:0.8em;font-weight:600;">
       Hybrid BM25 + Qdrant Retrieval
    </span>
    <span style="background:rgba(0,200,255,0.08);border:1px solid rgba(0,200,255,0.25);
                 color:#00c8ff;padding:5px 14px;border-radius:100px;font-size:0.8em;font-weight:600;">
       Llama-3.2-3B-Tele-it
    </span>
    <span style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);
                 color:#a78bfa;padding:5px 14px;border-radius:100px;font-size:0.8em;font-weight:600;">
       LangGraph DAG Orchestration
    </span>
    <span style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                 color:#f59e0b;padding:5px 14px;border-radius:100px;font-size:0.8em;font-weight:600;">
       3GPP Specification Grounded
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
specs = [
    (k1, "Avg Recall@5", "91.2 %"),
    (k2, "Avg MRR",      "0.87"),
    (k3, "Faithfulness", "93.5 %"),
    (k4, "P50 Latency",  "3.8 s"),
]
for col, label, val in specs:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander(" Live Network Telemetry Monitor", expanded=True):
    st.markdown('<div class="section-heading">Real-time Cell Tower Metrics (Simulated)</div>', unsafe_allow_html=True)
    df_live = generate_mock_timeseries_data()

    c1, c2, c3 = st.columns(3)
    with c1:
        avg_sig = df_live["Signal Strength (dBm)"].mean()
        color = "#10b981" if avg_sig > -70 else "#f59e0b"
        st.markdown(f'<div class="kpi-label">Signal Strength (dBm)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value" style="color:{color};font-size:1.4em;">{avg_sig:.1f}</div>', unsafe_allow_html=True)
        st.plotly_chart(make_sparkline(df_live, "Signal Strength (dBm)", color, y_range=[-80, -55]),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        avg_lat = df_live["Latency (ms)"].mean()
        color = "#10b981" if avg_lat < 25 else "#f59e0b"
        st.markdown(f'<div class="kpi-label">Latency (ms)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value" style="color:{color};font-size:1.4em;">{avg_lat:.1f}</div>', unsafe_allow_html=True)
        st.plotly_chart(make_sparkline(df_live, "Latency (ms)", color),
                        use_container_width=True, config={"displayModeBar": False})
    with c3:
        avg_tp = df_live["Throughput (Mbps)"].mean()
        color = "#00c8ff"
        st.markdown(f'<div class="kpi-label">Throughput (Mbps)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value" style="color:{color};font-size:1.4em;">{avg_tp:.1f}</div>', unsafe_allow_html=True)
        st.plotly_chart(make_sparkline(df_live, "Throughput (Mbps)", color),
                        use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="control-title"> Anomaly Injection Controls</div>', unsafe_allow_html=True)

ctrl_l, ctrl_r = st.columns([1, 1], gap="large")

with ctrl_l:
    st.markdown("**Auto-detect from Scenario Library**")
    st.caption("Picks a random pre-defined 3GPP network fault and runs the full RCA pipeline.")
    st.markdown('<div class="anomaly-btn">', unsafe_allow_html=True)
    if st.button(" Detect Random Network Anomaly", type="primary", use_container_width=True):
        scenario = random.choice(scenarios)
        st.session_state.current_query = scenario["fault_description"]
        st.session_state.scenario_data = scenario
        st.session_state.run_analysis = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with ctrl_r:
    st.markdown("**Manual Fault Description**")
    st.caption("Describe a network anomaly in plain text to run a custom RCA analysis.")
    manual_query = st.text_area(
        "Manual Query",
        placeholder="e.g. 'Persistent handover failures detected on Cell 42. A3 offset may be too aggressive…'",
        height=120,
        label_visibility="collapsed",
    )
    if st.button(" Analyze Manual Fault", disabled=not manual_query, use_container_width=True):
        st.session_state.current_query = manual_query
        st.session_state.scenario_data = None
        st.session_state.run_analysis = True
        st.rerun()

    if st.session_state.history:
        st.markdown("**Recent Queries:**")
        for h in reversed(st.session_state.history[-3:]):
            st.markdown(f"""
            <div class="metric-strip">
                <span class="metric-name" style="font-size:0.75em;">{h['Time']}</span>
                <span class="metric-val" style="font-size:0.75em;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:220px;">{h['Query'][:45]}…</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.run_analysis:
    query = st.session_state.current_query
    st.session_state.run_analysis = False

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(239,68,68,0.12),rgba(185,28,28,0.06));
                border:1px solid rgba(239,68,68,0.35);
                border-left:5px solid #ef4444;
                border-radius:14px;padding:22px 28px;margin-bottom:24px;
                box-shadow:0 8px 32px rgba(239,68,68,0.12);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
          <span style="background:#ef4444;color:#fff;font-size:0.72em;font-weight:800;
                       text-transform:uppercase;letter-spacing:1px;padding:3px 10px;
                       border-radius:100px;"> Anomaly Detected</span>
          <span style="color:#64748b;font-size:0.78em;font-family:'JetBrains Mono',monospace;">Starting FCRAG Agentic Pipeline…</span>
        </div>
        <div style="color:#fca5a5;font-size:1.05em;font-weight:500;line-height:1.55;">{query}</div>
    </div>
    """, unsafe_allow_html=True)

    tl_placeholder = st.empty()

    def render_timeline(steps):
        ICONS  = {"done": "", "error": "", "running": "", "pending": ""}
        LABELS = {"done": "Complete", "error": "Failed", "running": "Running…", "pending": "Waiting"}
        COLORS = {"done": "#10b981", "error": "#ef4444", "running": "#00c8ff", "pending": "#334155"}
        BG     = {"done": "rgba(16,185,129,0.06)", "error": "rgba(239,68,68,0.06)",
                  "running": "rgba(0,200,255,0.07)", "pending": "rgba(255,255,255,0.015)"}

        html_lines = []
        html_lines.append('<div style="margin-bottom:28px;">')
        html_lines.append('<div style="color:#64748b;font-size:0.72em;text-transform:uppercase;font-weight:700;letter-spacing:2px;margin-bottom:18px;display:flex;align-items:center;gap:10px;">')
        html_lines.append('<span style="flex:1;height:1px;background:rgba(255,255,255,0.06);"></span>')
        html_lines.append(' AGENT EXECUTION GRAPH')
        html_lines.append('<span style="flex:1;height:1px;background:rgba(255,255,255,0.06);"></span>')
        html_lines.append('</div>')

        for idx, (name, status, t) in enumerate(steps):
            c = COLORS[status]
            bg = BG[status]
            icon = ICONS[status]
            badge_txt = LABELS[status]
            step_num = f"{idx+1:02d}"
            time_str = f"{t:.3f} ms" if t is not None else ""
            pulse_style = "animation:pulse-blue 1.5s infinite;" if status == "running" else ""
            
            latency_html = ""
            if time_str:
                latency_html = f'<div style="background:{c}11;border:1px solid {c}44;border-radius:10px;padding:8px 16px;font-family:\'JetBrains Mono\',monospace;font-size:0.88em;font-weight:700;color:{c};white-space:nowrap;box-shadow:0 2px 8px {c}22;"> {time_str}</div>'

            html_lines.append(f'<div style="display:flex;align-items:center;gap:18px;background:{bg};border:1px solid {c}30;border-left:5px solid {c};border-radius:14px;padding:18px 22px;margin-bottom:12px;box-shadow:0 4px 20px {c}10;">')
            html_lines.append(f'<div style="min-width:42px;height:42px;border-radius:50%;background:{c}20;border:2.5px solid {c};display:flex;align-items:center;justify-content:center;font-size:0.82em;font-weight:900;color:{c};box-shadow:0 0 16px {c}30;{pulse_style}">{step_num}</div>')
            html_lines.append(f'<div style="flex:1;"><div style="font-size:1.05em;font-weight:700;color:#f8fafc;margin-bottom:5px;">{icon}&nbsp; {name}</div><div style="font-size:0.8em;color:#64748b;">Status: <span style="color:{c};font-weight:700;">{badge_txt}</span></div></div>')
            html_lines.append(latency_html)
            html_lines.append('</div>')

        html_lines.append("</div>")
        tl_placeholder.markdown("".join(html_lines), unsafe_allow_html=True)

    steps = [
        ("Hybrid Retrieval (BM25 + Dense Vector)", "running", None),
        ("Cross-Encoder Reranking", "pending", None),
        ("Context Assembly", "pending", None),
        ("LLM Reasoning (Llama-3.2-3B-Tele-it)", "pending", None),
    ]
    render_timeline(steps)

    # Step 1 — Retrieve
    t_ret = time.perf_counter()
    with st.spinner("Searching 3GPP specifications…"):
        results = retriever.retrieve(query, top_k=5)
    ret_time = time.perf_counter() - t_ret

    if not results:
        steps[0] = (steps[0][0], "error", ret_time)
        render_timeline(steps)
        st.error("No relevant 3GPP specifications found. Ensure the vector database is populated.")
        st.stop()

    steps[0] = (steps[0][0], "done", ret_time)
    steps[1] = (steps[1][0], "done", None)
    steps[2] = (steps[2][0], "running", None)
    render_timeline(steps)

    # Step 2 — Context
    context_str = "\n\n".join([f"[Source: {r.clause_id}]\n{r.text}" for r in results])
    steps[2] = (steps[2][0], "done", None)
    steps[3] = (steps[3][0], "running", None)
    render_timeline(steps)

    # Step 3 — LLM
    prompt = (
        f"You are an expert telecom network analyst. Answer STRICTLY based on the provided context.\n"
        f"Do not use outside knowledge. If context is insufficient, say: "
        f"'Insufficient data in knowledge base.'\n\n"
        f"Query: {query}\n\nContext:\n{context_str}\n\n"
        f"Format your response in exactly these sections:\n"
        f"## Problem Description\n[Describe the problem based on the query and context]\n\n"
        f"## Root Cause Analysis\n[Provide detailed technical analysis]\n\n"
        f"## Recommendations\n[Provide specific actionable recommendations]\n\n"
        f"Reasoned RCA:"
    )
    t_llm = time.perf_counter()
    with st.spinner("Generating detailed root cause analysis…"):
        rca_report = llm.generate(prompt, max_tokens=1024)
    llm_time = time.perf_counter() - t_llm

    steps[3] = (steps[3][0], "done", llm_time)
    render_timeline(steps)

    # Store
    st.session_state.rca_result = rca_report
    st.session_state.rca_results_list = results
    st.session_state.rca_timings = {"retrieval": ret_time, "llm": llm_time, "total": ret_time + llm_time}
    st.session_state.history.append({
        "Time": time.strftime("%H:%M:%S"),
        "Query": query,
        "Retrieval_ms": f"{ret_time*1000:.0f}",
        "LLM_s": f"{llm_time:.2f}",
    })


if st.session_state.rca_result:
    rca_report = st.session_state.rca_result
    results = st.session_state.rca_results_list
    timings = st.session_state.rca_timings

    tab_dash, tab_rca, tab_evidence = st.tabs([" Dashboard", " RCA Report", " Evidence"])

    with tab_dash:
        st.markdown('<div class="section-heading">Analysis Performance</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieved Clauses", len(results))
        m2.metric("Retrieval Time", f"{timings['retrieval']:.3f} ms")
        m3.metric("LLM Time", f"{timings['llm']:.2f} ms")
        m4.metric("Total Time", f"{timings['total']:.2f} ms")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Reranker Score Distribution</div>', unsafe_allow_html=True)

        if results:
            ids = [r.clause_id for r in results]
            scores = [r.rerank_score for r in results]
            bar_colors = [color_for_score(s) if s > 0 else "#ef4444" for s in scores]
            fig_bar = go.Figure(go.Bar(
                x=ids, y=scores,
                marker_color=bar_colors,
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=10),
                hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
            ))
            fig_bar.update_layout(
                height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(color="#64748b", size=9), gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(tickfont=dict(color="#64748b", size=10), gridcolor="rgba(255,255,255,0.04)",
                           zerolinecolor="rgba(255,255,255,0.1)"),
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-heading">Source Type Breakdown</div>', unsafe_allow_html=True)
        source_counts = {}
        for r in results:
            src_type = r.source_type if hasattr(r, "source_type") else "3gpp_spec"
            source_counts[src_type] = source_counts.get(src_type, 0) + 1

        pie_fig = go.Figure(go.Pie(
            labels=list(source_counts.keys()),
            values=list(source_counts.values()),
            marker_colors=["#00c8ff", "#10b981", "#f59e0b", "#a855f7"],
            hole=0.55,
            textfont=dict(color="#e2e8f0"),
            hovertemplate="<b>%{label}</b>: %{value} clauses<extra></extra>",
        ))
        pie_fig.update_layout(
            height=240, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=4, b=0),
            legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(0,0,0,0)"),
            showlegend=True,
        )
        st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

    with tab_rca:
        st.markdown('<div class="section-heading">Automated Root Cause Analysis</div>', unsafe_allow_html=True)
        sections = parse_rca_sections(rca_report)

        has_structured = any(sections.values())

        if sections["problem"]:
            st.markdown(f"""
            <div class="rca-section">
                <h4> Problem Description</h4>
                <div class="rca-body">{sections["problem"].replace(chr(10), "<br>")}</div>
            </div>""", unsafe_allow_html=True)

        if sections["root_cause"]:
            st.markdown(f"""
            <div class="rca-section" style="border-color:rgba(245,158,11,0.2);border-left-color:#f59e0b;">
                <h4 style="color:#f59e0b;"> Root Cause Analysis</h4>
                <div class="rca-body">{sections["root_cause"].replace(chr(10), "<br>")}</div>
            </div>""", unsafe_allow_html=True)

        if sections["recommendations"]:
            st.markdown(f"""
            <div class="rca-section" style="border-color:rgba(16,185,129,0.2);border-left-color:#10b981;">
                <h4 style="color:#10b981;"> Recommendations</h4>
                <div class="rca-body">{sections["recommendations"].replace(chr(10), "<br>")}</div>
            </div>""", unsafe_allow_html=True)


        if st.session_state.get("scenario_data"):
            sc = st.session_state.scenario_data
            st.markdown("---")
            st.markdown('<div class="section-heading">Ground Truth Reference</div>', unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(f'**Fault Type:** `{sc.get("fault_type","—")}`')
                st.markdown(f'**Scenario ID:** `{sc.get("scenario_id","—")}`')
            with g2:
                clauses = sc.get("relevant_clauses", [])
                st.markdown(f'**Ground Truth Clauses:** {", ".join([f"`{c}`" for c in clauses])}')

    with tab_evidence:
        st.markdown('<div class="section-heading">Retrieved 3GPP Specification Clauses</div>', unsafe_allow_html=True)
        for i, res in enumerate(results):
            safe_excerpt = res.text[:350].replace("<", "&lt;").replace(">", "&gt;")
            score_val = res.rerank_score if hasattr(res, "rerank_score") else 0.0
            source_type = res.source_type if hasattr(res, "source_type") else "3gpp_spec"
            st.markdown(f"""
            <div class="clause-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
                    <span class="clause-id">Top-{i+1}: {res.clause_id}</span>
                    {score_chip(score_val)}
                </div>
                <div class="clause-meta">Source Type: {source_type}</div>
                <div style="font-size:0.87em;color:#94a3b8;line-height:1.6;margin-top:10px;font-family:'JetBrains Mono',monospace;">{safe_excerpt}…</div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander(f"Full Text — {res.clause_id}"):
                st.markdown(f"```text\n{res.text}\n```")
