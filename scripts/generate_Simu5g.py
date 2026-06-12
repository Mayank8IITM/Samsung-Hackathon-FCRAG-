"""
gen_simu5g.py — Complete Simu5G Synthetic KPI + Fault Log Generator for FCRAG
Generates:
  - 150 labelled KPI time-series CSV files (one per scenario run)
  - 1 fault_narratives.jsonl  (synthetic memory for Qdrant)
  - 1 fse_training_pairs.json (AnomalyVector ↔ fault_type pairs for FSE)
  - 1 eval_fault_scenarios.json (20 eval scenarios with expected 3GPP clauses)
"""

import json, random, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

OUT_DIR        = "data/simu5g"
CSV_DIR        = f"{OUT_DIR}/kpi_logs"
NARR_FILE      = f"{OUT_DIR}/fault_narratives.jsonl"
FSE_FILE       = f"{OUT_DIR}/fse_training_pairs.json"
EVAL_FILE      = f"{OUT_DIR}/eval_fault_scenarios.json"

os.makedirs(CSV_DIR, exist_ok=True)

# ── 1. FAULT CATALOGUE ─────────────────────────────────────────────────────────
# Each entry: (fault_id, description, kpi_signature, 3gpp_clauses, action_template)
FAULT_CATALOGUE = [
    {
        "fault_id":    "HO_FAILURE",
        "description": "A3 offset too aggressive, handover failure rate increased",
        "kpi_deltas":  {"ho_sr": -0.25, "rsrp": -12, "sinr": -6,  "prb": +0.15, "throughput": -0.30, "rrc_reest": +0.20, "rlf": +0.15},
        "noise_scale": {"ho_sr": 0.04, "rsrp": 3, "sinr": 2, "prb": 0.05, "throughput": 0.08, "rrc_reest": 0.04, "rlf": 0.03},
        "clauses":     ["TS 38.331 Section 5.5.4", "TS 38.331 Section 5.3.3", "TS 38.321 Section 5.1"],
        "action":      "Increase A3 offset hysteresis from {old}dB to {new}dB. HSR restored from {before}% to {after}% within {time}s.",
        "params":      {"param": "A3 offset", "old_range": (-3, -1), "new_range": (0, 2), "metric": "Handover Success Rate"},
    },
    {
        "fault_id":    "PRB_CONGESTION",
        "description": "PRB utilization exceeded 90%, throughput severely degraded",
        "kpi_deltas":  {"prb": +0.45, "throughput": -0.50, "latency": +80, "ho_sr": -0.10, "sinr": -3, "rsrq": -4},
        "noise_scale": {"prb": 0.05, "throughput": 0.08, "latency": 15, "ho_sr": 0.03, "sinr": 1.5, "rsrq": 1},
        "clauses":     ["TS 38.214 Section 5.1", "TS 38.321 Section 5.4", "TS 38.213 Section 9"],
        "action":      "Triggered PRB reallocation and admission control. PRB util reduced from {before}% to {after}% in {time}s.",
        "params":      {"param": "PRB scheduler weight", "old_range": (0.85, 0.95), "new_range": (0.50, 0.65), "metric": "PRB Utilisation"},
    },
    {
        "fault_id":    "PRACH_CONGESTION",
        "description": "RRC connection setup failure spike, RACH preamble collision rate high",
        "kpi_deltas":  {"rrc_sr": -0.30, "prach_fail": +0.35, "latency": +45, "throughput": -0.20, "ho_sr": -0.05},
        "noise_scale": {"rrc_sr": 0.05, "prach_fail": 0.06, "latency": 10, "throughput": 0.05, "ho_sr": 0.02},
        "clauses":     ["TS 38.321 Section 5.1", "TS 38.331 Section 5.3.3", "TS 38.213 Section 8"],
        "action":      "Reconfigured PRACH preamble format and back-off timer. RRC SR restored from {before}% to {after}% in {time}s.",
        "params":      {"param": "PRACH back-off", "old_range": (0, 5), "new_range": (20, 40), "metric": "RRC Setup Success Rate"},
    },
    {
        "fault_id":    "INTERFERENCE",
        "description": "High inter-cell interference from neighbour, SINR and RSRQ degraded",
        "kpi_deltas":  {"sinr": -10, "rsrq": -6, "rsrp": -8, "throughput": -0.35, "rlf": +0.12, "prb": +0.20},
        "noise_scale": {"sinr": 2.5, "rsrq": 1.5, "rsrp": 2, "throughput": 0.07, "rlf": 0.03, "prb": 0.04},
        "clauses":     ["TS 38.214 Section 5.2", "TS 38.401 Section 8.3", "TS 38.423 Section 8"],
        "action":      "Applied ICIC via X2 interface. SINR recovered from {before}dB to {after}dB in {time}s.",
        "params":      {"param": "Tx power ICIC", "old_range": (41, 43), "new_range": (36, 38), "metric": "SINR"},
    },
    {
        "fault_id":    "RLF_SPIKE",
        "description": "Radio link failure spike, frequent RLF events due to coverage hole",
        "kpi_deltas":  {"rlf": +0.30, "rsrp": -15, "sinr": -8, "ho_sr": -0.20, "rrc_reest": +0.25, "throughput": -0.40},
        "noise_scale": {"rlf": 0.05, "rsrp": 3.5, "sinr": 2, "ho_sr": 0.04, "rrc_reest": 0.05, "throughput": 0.08},
        "clauses":     ["TS 38.331 Section 5.3.10", "TS 38.321 Section 5.3", "TS 38.331 Section 5.3.7"],
        "action":      "Triggered antenna tilt adjustment and power boost. RLF rate reduced from {before}% to {after}% in {time}s.",
        "params":      {"param": "antenna tilt", "old_range": (-2, 0), "new_range": (3, 6), "metric": "RLF Rate"},
    },
    {
        "fault_id":    "BEAM_FAILURE",
        "description": "Beam failure recovery triggered repeatedly, UE-gNB beam alignment lost",
        "kpi_deltas":  {"rsrp": -14, "sinr": -9, "throughput": -0.45, "ho_sr": -0.15, "rlf": +0.10},
        "noise_scale": {"rsrp": 3, "sinr": 2.5, "throughput": 0.09, "ho_sr": 0.03, "rlf": 0.02},
        "clauses":     ["TS 38.213 Section 6", "TS 38.331 Section 5.17", "TS 38.213 Section 9.1"],
        "action":      "Reconfigured beam management periodicity and BFR threshold. Beam stability restored in {time}s.",
        "params":      {"param": "BFR threshold", "old_range": (3, 5), "new_range": (1, 2), "metric": "Beam Success Rate"},
    },
    {
        "fault_id":    "PDCP_DELAY",
        "description": "PDCP reordering buffer overflow causing latency spike",
        "kpi_deltas":  {"latency": +120, "throughput": -0.25, "prb": +0.10, "sinr": -2},
        "noise_scale": {"latency": 25, "throughput": 0.06, "prb": 0.03, "sinr": 1},
        "clauses":     ["TS 38.323 Section 4.2", "TS 38.323 Section 5.4", "TS 38.331 Section 9.3"],
        "action":      "Adjusted PDCP reordering timer t-Reordering from {old}ms to {new}ms. Latency normalized in {time}s.",
        "params":      {"param": "t-Reordering", "old_range": (35, 50), "new_range": (10, 20), "metric": "PDCP Latency"},
    },
    {
        "fault_id":    "UE_MASS_DETACH",
        "description": "Mass UE detachment event, MME/AMF overload suspected",
        "kpi_deltas":  {"connected_ues": -0.60, "rrc_sr": -0.40, "throughput": -0.55, "prb": -0.30},
        "noise_scale": {"connected_ues": 0.08, "rrc_sr": 0.06, "throughput": 0.09, "prb": 0.05},
        "clauses":     ["TS 38.413 Section 8.6", "TS 38.401 Section 8.2", "TS 38.331 Section 5.3.3"],
        "action":      "Triggered AMF load rebalancing. UE reconnection rate restored from {before}% to {after}% in {time}s.",
        "params":      {"param": "AMF load threshold", "old_range": (95, 100), "new_range": (70, 80), "metric": "UE Connection Rate"},
    },
    {
        "fault_id":    "TIMING_DRIFT",
        "description": "Timing advance drift detected, uplink synchronization degraded",
        "kpi_deltas":  {"ul_timing_err": +0.40, "ho_sr": -0.12, "sinr": -4, "throughput": -0.20, "rlf": +0.08},
        "noise_scale": {"ul_timing_err": 0.06, "ho_sr": 0.03, "sinr": 1.5, "throughput": 0.05, "rlf": 0.02},
        "clauses":     ["TS 38.213 Section 4.2", "TS 38.321 Section 5.2", "TS 38.331 Section 5.3.3"],
        "action":      "Recalibrated TA loop and UL sync timer. Timing error reduced from {before}us to {after}us in {time}s.",
        "params":      {"param": "TA correction period", "old_range": (100, 200), "new_range": (20, 40), "metric": "UL Timing Error"},
    },
    {
        "fault_id":    "CAPACITY_OVERFLOW",
        "description": "Cell capacity limit reached, new UE admission blocked by ACB",
        "kpi_deltas":  {"connected_ues": +0.35, "prb": +0.40, "throughput": -0.35, "latency": +60, "rrc_sr": -0.25},
        "noise_scale": {"connected_ues": 0.05, "prb": 0.05, "throughput": 0.07, "latency": 12, "rrc_sr": 0.04},
        "clauses":     ["TS 38.331 Section 5.3.14", "TS 38.321 Section 5.4.5", "TS 38.213 Section 9"],
        "action":      "Enabled ACB with barring factor {old} → {new}. Admission rate normalized in {time}s.",
        "params":      {"param": "ACB barring factor", "old_range": (0.0, 0.1), "new_range": (0.5, 0.7), "metric": "Cell Admission Rate"},
    },
]

# ── 2. BASELINE KPI DISTRIBUTIONS (normal operation) ──────────────────────────
BASELINE = {
    "ho_sr":         (0.965, 0.008),   # Handover Success Rate (fraction)
    "rsrp":          (-75,   5.0),     # dBm
    "rsrq":          (-8,    2.0),     # dB
    "sinr":          (18,    3.0),     # dB
    "throughput":    (0.78,  0.08),    # fraction of peak
    "prb":           (0.42,  0.07),    # PRB utilisation fraction
    "latency":       (12,    3.0),     # ms
    "rrc_sr":        (0.985, 0.005),   # RRC Setup Success Rate
    "rrc_reest":     (0.02,  0.005),   # RRC Re-establishment rate
    "rlf":           (0.008, 0.003),   # Radio Link Failure rate
    "prach_fail":    (0.015, 0.005),   # PRACH failure rate
    "connected_ues": (0.65,  0.08),    # fraction of capacity
    "ul_timing_err": (0.05,  0.01),    # UL timing error (normalised)
}

CELLS = [f"Cell-{i:02d}" for i in range(1, 51)]

# ── 3. GENERATE KPI TIME-SERIES CSVS ──────────────────────────────────────────
def generate_kpi_window(baseline, fault=None, n_samples=20, fault_inject_at=12):
    """
    Generates a rolling window of KPI samples.
    Normal for first `fault_inject_at` steps, then fault signature.
    """
    rows = []
    t = datetime(2024, 1, 15, 8, 0, 0)
    for i in range(n_samples):
        row = {"timestamp": t.isoformat(), "sample_idx": i}
        is_fault = (fault is not None) and (i >= fault_inject_at)
        row["label"] = fault["fault_id"] if is_fault else "NORMAL"
        row["fault_active"] = int(is_fault)
        for kpi, (base_mean, base_std) in BASELINE.items():
            val = np.random.normal(base_mean, base_std)
            if is_fault and kpi in fault["kpi_deltas"]:
                delta = fault["kpi_deltas"][kpi]
                noise = np.random.normal(0, fault["noise_scale"].get(kpi, base_std * 0.5))
                val = val + delta + noise
            row[kpi] = round(float(np.clip(val, -150, 200)), 4)
        rows.append(row)
        t += timedelta(milliseconds=50)
    return pd.DataFrame(rows)


def make_anomaly_vector(df, fault_inject_at=12):
    """Extract the normalised deviation vector from a window."""
    pre  = df[df["sample_idx"] < fault_inject_at]
    post = df[df["sample_idx"] >= fault_inject_at]
    if len(post) == 0:
        return {k: 0.0 for k in BASELINE}
    vec = {}
    for kpi in BASELINE:
        pre_mean  = pre[kpi].mean()
        post_mean = post[kpi].mean()
        baseline_std = BASELINE[kpi][1]
        vec[f"{kpi}_dev"] = round((post_mean - pre_mean) / (baseline_std + 1e-9), 4)
    return vec


print("Generating KPI CSV files...")
run_id = 0
csv_manifest = []

for fault in FAULT_CATALOGUE:
    for run in range(15):          # 15 runs × 10 faults = 150 CSVs
        cell = random.choice(CELLS)
        fault_at = random.randint(8, 14)
        df = generate_kpi_window(BASELINE, fault=fault, n_samples=20, fault_inject_at=fault_at)
        df["cell_id"]   = cell
        df["run_id"]    = run_id
        df["fault_type"] = fault["fault_id"]

        fname = f"{CSV_DIR}/run_{run_id:04d}_{fault['fault_id']}_{cell.replace('-','')}.csv"
        df.to_csv(fname, index=False)

        # compute anomaly vector for this run
        av = make_anomaly_vector(df, fault_inject_at=fault_at)
        csv_manifest.append({
            "run_id":        run_id,
            "cell":          cell,
            "fault_type":    fault["fault_id"],
            "fault_at":      fault_at,
            "csv_file":      fname,
            "anomaly_vector": av,
            "clauses":       fault["clauses"],
        })
        run_id += 1

# Also generate 30 normal-only runs for negative examples
for run in range(30):
    cell = random.choice(CELLS)
    df = generate_kpi_window(BASELINE, fault=None, n_samples=20)
    df["cell_id"]    = cell
    df["run_id"]     = run_id
    df["fault_type"] = "NORMAL"
    fname = f"{CSV_DIR}/run_{run_id:04d}_NORMAL_{cell.replace('-','')}.csv"
    df.to_csv(fname, index=False)
    av = make_anomaly_vector(df)
    csv_manifest.append({
        "run_id":        run_id,
        "cell":          cell,
        "fault_type":    "NORMAL",
        "fault_at":      None,
        "csv_file":      fname,
        "anomaly_vector": av,
        "clauses":       [],
    })
    run_id += 1

print(f"  Generated {run_id} CSV files → {CSV_DIR}/")

# ── 4. FAULT NARRATIVES (Synthetic Memory for Qdrant) ─────────────────────────
print("Generating fault narratives for Qdrant synthetic memory...")

def make_narrative(fault, cell, ts, av):
    p       = fault["params"]
    old_val = round(random.uniform(*p["old_range"]), 2)
    new_val = round(random.uniform(*p["new_range"]), 2)
    before  = round(random.uniform(60, 78), 1)
    after   = round(random.uniform(90, 99), 1)
    time_s  = random.randint(45, 180)
    action  = fault["action"].format(
        param=p["param"], old=old_val, new=new_val,
        before=before, after=after, time=time_s
    )
    kv_str = ", ".join(f"{k}={v:+.2f}σ" for k, v in list(av.items())[:5])
    text = (
        f"Fault: {fault['fault_id']} at {cell} {ts}. "
        f"Description: {fault['description']}. "
        f"KPI deviations: {kv_str}. "
        f"Root cause: {fault['description']}. "
        f"Action taken: {action} "
        f"Relevant 3GPP specs: {', '.join(fault['clauses'])}."
    )
    return {
        "fault_id":    fault["fault_id"],
        "cell":        cell,
        "timestamp":   ts,
        "text":        text,
        "clauses":     fault["clauses"],
        "action":      action,
        "anomaly_vector": av,
    }

narratives = []
base_ts = datetime(2024, 1, 1, 0, 0, 0)
for i, fault in enumerate(FAULT_CATALOGUE):
    for j in range(15):           # 15 narratives per fault = 150 total
        cell = random.choice(CELLS)
        ts   = (base_ts + timedelta(hours=i*24+j*6)).strftime("%Y-%m-%d %H:%M UTC")
        av   = {f"{k}_dev": round(fault["kpi_deltas"].get(k, 0) /
                                   (BASELINE[k][1] + 1e-9) +
                                   np.random.normal(0, 0.15), 3)
                for k in BASELINE}
        narratives.append(make_narrative(fault, cell, ts, av))

with open(NARR_FILE, "w") as f:
    for n in narratives:
        f.write(json.dumps(n) + "\n")
print(f"  Generated {len(narratives)} narratives → {NARR_FILE}")

# ── 5. FSE TRAINING PAIRS ──────────────────────────────────────────────────────
# Positive pairs: (anomaly_vector, fault_type)
# Negative pairs: (anomaly_vector_from_different_fault, same_fault_type) ← hard negatives
print("Generating FSE training pairs...")

fse_pairs = []

# Positives from CSV manifest
for entry in csv_manifest:
    if entry["fault_type"] == "NORMAL":
        continue
    fse_pairs.append({
        "anomaly_vector": entry["anomaly_vector"],
        "fault_type":     entry["fault_type"],
        "clauses":        entry["clauses"],
        "label":          1,
        "pair_type":      "positive",
    })

# Hard negatives: pair anomaly vector from fault A with label of fault B
fault_ids = [f["fault_id"] for f in FAULT_CATALOGUE]
positive_entries = [e for e in csv_manifest if e["fault_type"] != "NORMAL"]

for entry in positive_entries:
    wrong_fault = random.choice([f for f in fault_ids if f != entry["fault_type"]])
    fse_pairs.append({
        "anomaly_vector": entry["anomaly_vector"],
        "fault_type":     wrong_fault,
        "clauses":        [],
        "label":          0,
        "pair_type":      "hard_negative",
    })

# Augmented positives: add ±10% Gaussian noise to existing anomaly vectors
aug_pairs = []
for pair in [p for p in fse_pairs if p["pair_type"] == "positive"]:
    av = pair["anomaly_vector"].copy()
    noisy_av = {k: round(v + np.random.normal(0, abs(v) * 0.10 + 0.05), 4) for k, v in av.items()}
    aug_pairs.append({
        "anomaly_vector": noisy_av,
        "fault_type":     pair["fault_type"],
        "clauses":        pair["clauses"],
        "label":          1,
        "pair_type":      "augmented_positive",
    })

fse_pairs.extend(aug_pairs)
random.shuffle(fse_pairs)

with open(FSE_FILE, "w") as f:
    json.dump(fse_pairs, f, indent=2)
print(f"  Generated {len(fse_pairs)} FSE training pairs → {FSE_FILE}")
print(f"    Positive: {sum(1 for p in fse_pairs if p['label']==1)}")
print(f"    Hard negative: {sum(1 for p in fse_pairs if p['pair_type']=='hard_negative')}")
print(f"    Augmented: {sum(1 for p in fse_pairs if p['pair_type']=='augmented_positive')}")

# ── 6. EVAL FAULT SCENARIOS (20 ground-truth scenarios for MRR/Recall eval) ───
print("Generating eval fault scenarios...")

eval_scenarios = []
for i, fault in enumerate(FAULT_CATALOGUE):
    cell = random.choice(CELLS)
    ts   = (datetime(2024, 3, 1) + timedelta(hours=i*12)).strftime("%Y-%m-%d %H:%M UTC")
    av   = {f"{k}_dev": round(fault["kpi_deltas"].get(k, 0) /
                               (BASELINE[k][1] + 1e-9), 3)
            for k in BASELINE}
    eval_scenarios.append({
        "scenario_id":   f"EVAL_{i+1:02d}",
        "fault_type":    fault["fault_id"],
        "cell":          cell,
        "timestamp":     ts,
        "description":   fault["description"],
        "anomaly_vector": av,
        "expected_clauses": fault["clauses"],
        "query":         f"Why is {fault['fault_id'].replace('_',' ').lower()} occurring at {cell}? "
                         f"KPI deviations: {', '.join(f'{k}={v:+.2f}' for k,v in list(av.items())[:4])}",
    })

# Add 10 more mixed/edge-case scenarios
for i in range(10):
    f1, f2 = random.sample(FAULT_CATALOGUE, 2)
    cell   = random.choice(CELLS)
    combined_av = {}
    for k in BASELINE:
        d1 = f1["kpi_deltas"].get(k, 0)
        d2 = f2["kpi_deltas"].get(k, 0) * 0.5   # secondary fault at 50% intensity
        combined_av[f"{k}_dev"] = round((d1 + d2) / (BASELINE[k][1] + 1e-9), 3)
    eval_scenarios.append({
        "scenario_id":   f"EVAL_{i+11:02d}",
        "fault_type":    f"{f1['fault_id']}+{f2['fault_id']}",
        "cell":          cell,
        "timestamp":     (datetime(2024, 3, 15) + timedelta(hours=i*8)).strftime("%Y-%m-%d %H:%M UTC"),
        "description":   f"Co-occurring: {f1['description']} AND {f2['description']}",
        "anomaly_vector": combined_av,
        "expected_clauses": list(set(f1["clauses"] + f2["clauses"]))[:4],
        "query":         f"Co-occurring {f1['fault_id']} and {f2['fault_id']} at {cell}.",
    })

with open(EVAL_FILE, "w") as f:
    json.dump(eval_scenarios, f, indent=2)
print(f"  Generated {len(eval_scenarios)} eval scenarios → {EVAL_FILE}")

# ── 7. SUMMARY ─────────────────────────────────────────────────────────────────
print("\n─── Generation Complete ───────────────────────────────────────")
print(f"  KPI CSVs:           {run_id} files in {CSV_DIR}/")
print(f"  Fault narratives:   {len(narratives)} entries in {NARR_FILE}")
print(f"  FSE training pairs: {len(fse_pairs)} pairs in {FSE_FILE}")
print(f"  Eval scenarios:     {len(eval_scenarios)} scenarios in {EVAL_FILE}")
print(f"  KPI columns:        {list(BASELINE.keys())}")
print(f"  Fault types:        {[f['fault_id'] for f in FAULT_CATALOGUE]}")
print("────────────────────────────────────────────────────────────────")