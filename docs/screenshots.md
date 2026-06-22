# FCRAG 2.0 Dashboard Screenshots

This document serves as a visual guide to the FCRAG 2.0 Streamlit NOC Dashboard. Due to repository constraints, raw image binaries are not directly checked into `docs/` by default, but the layout below provides placeholders for where screenshot evidence of the UI should be inserted prior to final judging.

---

## 1. The Live Telemetry Monitor
*This view shows the baseline state of the dashboard before an anomaly is detected.*

> **[PLACEHOLDER: Insert `screenshot_telemetry.png` here]**
> 
> *Caption:* The main dashboard features three real-time sparkline charts (Signal Strength, Latency, and Throughput) built with Plotly. This simulates a live OAI KPM cell tower feed. The glassmorphism CSS styling gives it a premium Network Operations Center aesthetic.

---

## 2. Anomaly Injection & Agent Timeline
*This view shows the system reacting to a simulated fault.*

> **[PLACEHOLDER: Insert `screenshot_timeline.png` here]**
> 
> *Caption:* Upon clicking "Detect Random Network Anomaly", the LangGraph DAG execution is visualized as a pulsing timeline. Users can watch in real-time as the Hybrid Retriever searches the Qdrant DB, the Cross-Encoder filters results, and the Llama-3.2 model synthesizes the report.

---

## 3. The RCA Report (Root Cause Analysis)
*This view shows the final synthesized output.*

> **[PLACEHOLDER: Insert `screenshot_rca_report.png` here]**
> 
> *Caption:* The `🧠 RCA Report` tab cleanly separates the LLM's response into Problem Description, Root Cause Analysis, and Actionable Recommendations. Below the formatted text, the raw unedited output from the `Llama-3.2-3B-Tele-it` model is displayed for transparency.

---

## 4. Ground Truth & Evidence Transparency
*This view proves the system isn't hallucinating.*

> **[PLACEHOLDER: Insert `screenshot_evidence.png` here]**
> 
> *Caption:* The `🔍 Evidence` tab provides total transparency. It displays the exact 3GPP clauses retrieved (e.g., TS 38.331) alongside their Cross-Encoder confidence scores. The `📊 Dashboard` tab (not pictured) contains a pie chart breaking down the source types (3GPP vs Simu5G) and a bar chart of the reranker distribution.
