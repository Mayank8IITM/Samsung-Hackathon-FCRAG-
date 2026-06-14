"""
fcrag/ingest/simu5g_generator.py — FCRAG 2.0 Simu5G Narrative Generator
========================================================================
Translates raw Simu5G KPI trace files into normalized English narratives.
Computes deviations during anomaly windows and outputs JSONL chunks.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Find project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SIMU5G_DIR = ROOT / "data" / "simu5g"
KPI_LOGS_DIR = SIMU5G_DIR / "kpi_logs"
OUTPUT_FILE = SIMU5G_DIR / "fault_narratives_normalized.jsonl"


def generate_narratives():
    """
    Parses all CSVs in kpi_logs, computes standard deviations during faults,
    and constructs a natural language narrative for the RAG chunker.
    """
    if not KPI_LOGS_DIR.exists():
        print(f"[ERROR] KPI logs directory not found: {KPI_LOGS_DIR}")
        return

    csv_files = list(KPI_LOGS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {KPI_LOGS_DIR}")
        return

    print(f"Parsing {len(csv_files)} KPI logs to generate narratives...")
    narratives = []
    
    # Simple mapping for root causes and actions to simulate expert system
    expert_rules = {
        "HO_FAILURE": ("A3 offset too aggressive, handover failure rate increased.", "Increase A3 offset hysteresis."),
        "PRB_CONGESTION": ("PRB utilization exceeded 90%, throughput severely degraded.", "Triggered PRB reallocation and admission control."),
        "PRACH_CONGESTION": ("RRC connection setup failure spike, RACH preamble collision rate high.", "Reconfigured PRACH preamble format and back-off timer."),
        "INTERFERENCE": ("High interference detected, SINR dropped significantly.", "Initiated interference mitigation and power control."),
        "RLF_SPIKE": ("Radio Link Failure spike due to poor coverage.", "Adjusted antenna tilt and TX power."),
        "BEAM_FAILURE": ("Beam failure recovery triggered frequently.", "Optimized beam sweeping and tracking loops."),
        "PDCP_DELAY": ("PDCP queue overflow leading to high latency.", "Increased PDCP discard timer and buffer size."),
        "UE_MASS_DETACH": ("Massive UE detach event detected.", "Reset MME context and cleared stale RRC connections."),
        "TIMING_DRIFT": ("Uplink timing drift out of bounds.", "Sent Timing Advance Commands (TAC) to affected UEs."),
        "CAPACITY_OVERFLOW": ("Cell capacity overflow.", "Activated load balancing to neighbor cells."),
        "NORMAL": ("Network operating normally.", "No action required.")
    }

    # KPIs to track
    metrics = [
        "ho_sr", "rsrp", "rsrq", "sinr", "throughput", "prb", "latency", 
        "rrc_sr", "rrc_reest", "rlf", "prach_fail", "connected_ues", "ul_timing_err"
    ]

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if "fault_active" not in df.columns:
                continue

            # Identify the overall fault for this run
            fault_id = df["fault_type"].iloc[0] if "fault_type" in df.columns else "NORMAL"
            if fault_id == "NORMAL":
                continue  # Skip strictly normal files to focus on anomalies

            cell_id = df["cell_id"].iloc[0] if "cell_id" in df.columns else "Unknown"
            timestamp = df["timestamp"].iloc[0] if "timestamp" in df.columns else "2024-01-01 00:00 UTC"

            # Compute deviations (simplified: just difference of means)
            normal_df = df[df["fault_active"] == 0]
            fault_df = df[df["fault_active"] == 1]
            
            anomaly_vector = {}
            if not normal_df.empty and not fault_df.empty:
                for metric in metrics:
                    if metric in df.columns:
                        n_mean = normal_df[metric].mean()
                        n_std = normal_df[metric].std()
                        f_mean = fault_df[metric].mean()
                        
                        # Avoid div by zero, calculate crude z-score deviation
                        dev = (f_mean - n_mean) / (n_std if n_std > 1e-5 else 1.0)
                        anomaly_vector[f"{metric}_dev"] = round(dev, 3)

            desc, action = expert_rules.get(fault_id, ("Unknown anomaly detected.", "Investigate logs."))

            # Build narrative text
            kpi_text = ", ".join([f"{k}={v}σ" for k, v in anomaly_vector.items() if abs(v) > 1.0])
            if not kpi_text:
                kpi_text = "Minor fluctuations"

            text = (
                f"Fault: {fault_id} at {cell_id} {timestamp}. "
                f"Description: {desc} "
                f"KPI deviations: {kpi_text}. "
                f"Root cause: {desc} "
                f"Action taken: {action}"
            )

            record = {
                "fault_id": fault_id,
                "cell": cell_id,
                "timestamp": timestamp,
                "text": text,
                "action": action,
                "anomaly_vector": anomaly_vector
            }
            narratives.append(record)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Write to JSONL
    if narratives:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
            for n in narratives:
                f_out.write(json.dumps(n) + "\n")
        print(f"[OK] Generated {len(narratives)} narratives and saved to {OUTPUT_FILE.name}")
    else:
        print("No narratives generated.")


def run_generation():
    print("\n" + "=" * 60)
    print("PHASE 1.5 -- Simu5G Narrative Generator")
    print("=" * 60)
    generate_narratives()
    print("-" * 60)


if __name__ == "__main__":
    run_generation()
