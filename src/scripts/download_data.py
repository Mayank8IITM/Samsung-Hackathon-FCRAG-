"""
FCRAG 2.0 — Dataset & Model Download Script
============================================
Run this after `uv add -r requirements.txt`

What this script downloads (via code):
  ✅ TeleQnA benchmark           → data/teleqna/
  ✅ Tele-Eval (10K sample)      → data/tele_eval/
  ✅ HuggingFace models          → models/  (via huggingface_hub)

What requires MANUAL download:
  ⚠️  3GPP PDFs (TS 38.xxx)     → data/3gpp/      (links provided below)
  ⚠️  O-RAN WG1/WG3 specs       → data/oran/      (links provided below)
  ✅  Simu5G fault logs          → data/simu5g/    (auto-generated synthetically)
"""

import os
import json
import random
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

dirs = [
    DATA_DIR / "teleqna",
    DATA_DIR / "tele_eval",
    DATA_DIR / "3gpp",        # manual PDFs go here
    DATA_DIR / "oran",         # manual PDFs go here
    DATA_DIR / "simu5g",
    DATA_DIR / "custom_scenarios",
    DATA_DIR / "feedback",
    MODELS_DIR,
]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Directory structure created.\n")


# ── 1. TeleQnA Dataset ────────────────────────────────────────────────────────
# The dataset lives on GitHub (not public HuggingFace Hub).
# We download the JSON files directly from the GitHub raw URL.
TELEQNA_FILES = {
    "TeleQnA_testing1.json":  "https://raw.githubusercontent.com/netop-team/TeleQnA/main/TeleQnA_testing1.json",
    "TeleQnA_testing2.json":  "https://raw.githubusercontent.com/netop-team/TeleQnA/main/TeleQnA_testing2.json",
    "TeleQnA_training.json":  "https://raw.githubusercontent.com/netop-team/TeleQnA/main/TeleQnA_training.json",
}


def download_teleqna():
    import urllib.request
    print("📥 Downloading TeleQnA benchmark from GitHub...")
    out_dir = DATA_DIR / "teleqna"
    all_ok = True
    for filename, url in TELEQNA_FILES.items():
        dest = out_dir / filename
        if dest.exists():
            print(f"   ⏭️  Already exists: {filename}")
            continue
        try:
            print(f"   → Fetching {filename} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size // 1024
            print(f"✅  ({size_kb} KB)")
        except Exception as e:
            print(f"❌  {e}")
            all_ok = False

    if all_ok:
        # Merge into a single unified JSONL for easy loading
        merged = []
        for filename in TELEQNA_FILES:
            with open(out_dir / filename) as f:
                data = json.load(f)
            # TeleQnA JSON is a dict: {"Q1": {...}, "Q2": {...}}
            if isinstance(data, dict):
                for qid, qdata in data.items():
                    qdata["id"] = qid
                    qdata["source_file"] = filename
                    merged.append(qdata)
            elif isinstance(data, list):
                merged.extend(data)
        merged_path = out_dir / "teleqna_all.jsonl"
        with open(merged_path, "w") as f:
            for item in merged:
                f.write(json.dumps(item) + "\n")
        print(f"   ✅ TeleQnA merged: {len(merged)} questions → {merged_path}")
    else:
        print("   ⚠️  Some files failed. Check your internet connection.")


# ── 2. Tele-Eval Dataset (sample 10K) ─────────────────────────────────────────
def download_tele_eval():
    print("\n📥 Downloading Tele-Eval (10K sample for dev)...")
    try:
        from datasets import load_dataset
        # Correct split name is 'data' (not 'train') — verified from HF dataset card
        ds = load_dataset("AliMaatouk/Tele-Eval", split="data[:10000]")
        ds.save_to_disk(str(DATA_DIR / "tele_eval"))
        print(f"   ✅ Tele-Eval (10K) saved → {DATA_DIR / 'tele_eval'}")
        print(f"   📊 Features: {ds.features}")
    except Exception as e:
        print(f"   ❌ Tele-Eval download failed: {e}")
        print("   → Manual fix: the split is named 'data', not 'train'")
        print("   → Or download full dataset: https://huggingface.co/datasets/AliMaatouk/Tele-Eval")


# ── 3. HuggingFace Models ─────────────────────────────────────────────────────
MODELS = [
    {
        "repo_id": "AliMaatouk/Gemma-2-2B-Tele",
        "local_dir": MODELS_DIR / "gemma-2-2b-tele",
        "role": "Dense Embedding Extractor",
    },
    {
        "repo_id": "AliMaatouk/Llama-3.2-3B-Tele-it",
        "local_dir": MODELS_DIR / "llama-3.2-3b-tele-it",
        "role": "Reasoning LLM (Primary)",
    },
    {
        "repo_id": "AliMaatouk/TinyLlama-1.1B-Tele-it",
        "local_dir": MODELS_DIR / "tinyllama-1.1b-tele-it",
        "role": "Reasoning LLM (Fallback)",
    },
    {
        "repo_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "local_dir": MODELS_DIR / "ms-marco-minilm-l6-v2",
        "role": "Cross-Encoder Reranker",
    },
]


def download_models(only: str = None):
    """
    Download models one at a time. Models are large (1–6 GB each).
    Downloads are resumable — safe to Ctrl+C and re-run.

    Args:
        only: short name to download just one model.
              Options: 'gemma' | 'llama' | 'tinyllama' | 'reranker'
    Example:
        python scripts/download_data.py --models-only reranker
    """
    short_names = {
        "gemma": "AliMaatouk/Gemma-2-2B-Tele",
        "llama": "AliMaatouk/Llama-3.2-3B-Tele-it",
        "tinyllama": "AliMaatouk/TinyLlama-1.1B-Tele-it",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }

    targets = [m for m in MODELS if (
        only is None or m["repo_id"] == short_names.get(only, only)
    )]

    if not targets:
        print(f"   ❌ Unknown model shortname '{only}'. Choose: {list(short_names.keys())}")
        return

    print("\n📥 Downloading HuggingFace models (resumable — safe to Ctrl+C and retry)...")
    for m in targets:
        dest = Path(m["local_dir"])
        # Quick check: if model already fully downloaded (has config.json)
        if (dest / "config.json").exists():
            print(f"   ⏭️  Already downloaded: {m['repo_id']}")
            continue
        print(f"\n   → {m['repo_id']} ({m['role']})")
        print(f"      Saving to: {dest}")
        try:
            snapshot_download(
                repo_id=m["repo_id"],
                local_dir=str(dest),
                # Skip non-PyTorch formats to save disk space
                ignore_patterns=[
                    "*.msgpack", "flax_model*", "tf_model*",
                    "rust_model*", "onnx/*",
                ],
                # Use local_dir_use_symlinks=False so files are real copies
                local_dir_use_symlinks=False,
            )
            print(f"   ✅ Done → {dest}")
        except KeyboardInterrupt:
            print(f"\n   ⏸️  Interrupted. Re-run to resume (already downloaded files are kept).")
            raise
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print(f"   → Re-run with: python scripts/download_data.py --models-only {only or m['repo_id'].split('/')[1].lower()}")


# ── 4. Simu5G Synthetic Fault Logs ────────────────────────────────────────────
FAULT_TYPES = [
    ("HO_FAILURE", "A3 offset too aggressive, handover failure rate increased"),
    ("CONGESTION", "PRB utilization exceeded 90%, throughput degraded"),
    ("RRC_FAILURE", "RRC connection setup failure spike, likely RACH congestion"),
    ("INTERFERENCE", "High interference from neighboring cell, SINR dropped"),
    ("LINK_FAILURE", "Radio link failure, frequent RLF events detected"),
    ("BEAM_FAILURE", "Beam failure recovery triggered, UE-gNB alignment lost"),
    ("PDCP_DELAY", "PDCP layer delay spike due to reordering buffer overflow"),
    ("UE_DETACH", "Mass UE detachment event, MME overload suspected"),
    ("TIMING_DRIFT", "Timing advance drift detected, uplink sync degraded"),
    ("CAPACITY_OVERFLOW", "Cell capacity limit reached, new UE admission blocked"),
]

CELLS = [f"Cell-{i}" for i in range(1, 51)]
RESOLUTION_TEMPLATES = [
    "Adjusted {param} from {old} to {new}. Recovery: {metric} restored from {before}% to {after}% within {time}s.",
    "Applied {param} reconfiguration. {metric} improved from {before}% to {after}% in {time}s.",
    "Triggered {param} optimization. Fault resolved, {metric} normalized in {time}s.",
]


def _random_scenario(idx: int, fault_type: str, description: str) -> dict:
    cell = random.choice(CELLS)
    param_map = {
        "HO_FAILURE": ("A3 offset", "-3dB", "-1dB", "HO success rate", 72, 96),
        "CONGESTION": ("PRB reservation limit", "95%", "80%", "throughput", 40, 88),
        "RRC_FAILURE": ("RACH preamble power", "-110dBm", "-105dBm", "RRC success rate", 65, 92),
        "INTERFERENCE": ("inter-cell PCI offset", "0", "3", "SINR", 55, 87),
        "LINK_FAILURE": ("RLF timer T310", "1000ms", "2000ms", "link stability", 60, 94),
        "BEAM_FAILURE": ("beam failure recovery threshold", "3", "1", "beam alignment", 58, 90),
        "PDCP_DELAY": ("reorder timer t-Reordering", "35ms", "50ms", "PDCP latency SLA", 50, 91),
        "UE_DETACH": ("paging load threshold", "80%", "60%", "UE attachment rate", 45, 95),
        "TIMING_DRIFT": ("TA loop gain", "0.5", "0.25", "uplink timing sync", 62, 93),
        "CAPACITY_OVERFLOW": ("admission control threshold", "95%", "85%", "new UE acceptance", 30, 85),
    }
    param, old, new, metric, before, after = param_map.get(
        fault_type, ("parameter", "old_val", "new_val", "metric", 60, 90)
    )
    time_to_recover = random.randint(60, 300)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    resolution = random.choice(RESOLUTION_TEMPLATES).format(
        param=param, old=old, new=new, metric=metric,
        before=before, after=after, time=time_to_recover
    )
    kpi_deltas = {
        "throughput_drop_pct": round(random.uniform(20, 60), 1),
        "ho_success_rate_drop": round(random.uniform(0.1, 0.4), 2),
        "prb_utilization": round(random.uniform(0.7, 0.99), 2),
        "rrc_retries": random.randint(5, 30),
        "latency_increase_ms": random.randint(20, 120),
    }
    return {
        "scenario_id": idx,
        "fault_type": fault_type,
        "cell_id": cell,
        "timestamp": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{hour:02d}:{minute:02d}:00Z",
        "description": description,
        "kpi_deltas": kpi_deltas,
        "root_cause": f"{fault_type}: {description}",
        "resolution": resolution,
        "narrative": (
            f"Scenario {idx}: {fault_type} at {cell}, "
            f"{hour:02d}:{minute:02d} UTC. "
            f"Root cause: {description}. "
            f"Action: {resolution}"
        ),
    }


def generate_simu5g_logs():
    print("\n⚙️  Generating Simu5G synthetic fault logs...")
    random.seed(42)
    scenarios = []
    idx = 1
    # Generate 15 scenarios per fault type = 150 total
    for fault_type, description in FAULT_TYPES:
        for _ in range(15):
            scenarios.append(_random_scenario(idx, fault_type, description))
            idx += 1

    # Save as JSONL
    out_path = DATA_DIR / "simu5g" / "fault_scenarios.jsonl"
    with open(out_path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s) + "\n")

    print(f"   ✅ {len(scenarios)} fault scenarios saved → {out_path}")

    # Also save individual CSVs (one per fault type) to mimic OMNeT++ output
    import csv
    for fault_type, _ in FAULT_TYPES:
        subset = [s for s in scenarios if s["fault_type"] == fault_type]
        csv_path = DATA_DIR / "simu5g" / f"{fault_type.lower()}_scenarios.csv"
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=[
                "scenario_id", "fault_type", "cell_id", "timestamp",
                "throughput_drop_pct", "ho_success_rate_drop",
                "prb_utilization", "rrc_retries", "latency_increase_ms",
                "root_cause", "resolution"
            ])
            writer.writeheader()
            for s in subset:
                row = {**{"scenario_id": s["scenario_id"], "fault_type": s["fault_type"],
                           "cell_id": s["cell_id"], "timestamp": s["timestamp"]},
                       **s["kpi_deltas"],
                       "root_cause": s["root_cause"], "resolution": s["resolution"]}
                writer.writerow(row)
    print(f"   ✅ 10 CSV files (15 scenarios each) saved → {DATA_DIR / 'simu5g'}/")


# ── 5. 3GPP Manual Download Instructions ──────────────────────────────────────
def print_3gpp_instructions():
    print("\n" + "=" * 60)
    print("⚠️  MANUAL DOWNLOAD REQUIRED: 3GPP Specifications")
    print("=" * 60)
    specs = [
    ("TS 38.300", "NR Overall Description",
     "https://www.3gpp.org/ftp/Specs/archive/38_series/38.300/"),

    ("TS 38.321", "MAC Protocol",
     "https://www.3gpp.org/ftp/Specs/archive/38_series/38.321/"),

    ("TS 38.331", "RRC Protocol",
     "https://www.3gpp.org/ftp/Specs/archive/38_series/38.331/"),

    ("TS 38.214", "Physical Layer Procedures",
     "https://www.3gpp.org/ftp/Specs/archive/38_series/38.214/"),

    ("TS 38.401", "NG-RAN Architecture",
     "https://www.3gpp.org/ftp/Specs/archive/38_series/38.401/"),

    ("TS 38.413", "NG Application Protocol (NGAP)",
     "https://www.3gpp.org/ftp/Specs/archive/38_series/38.413/"),
]
    print(f"\n   Download the latest Release 16/18 ZIP from each URL,")
    print(f"   extract the .doc/.pdf, and place the PDFs in: {DATA_DIR / '3gpp'}/\n")
    for spec, name, url in specs:
        print(f"   📄 {spec} ({name})")
        print(f"      URL: {url}\n")

    print("=" * 60)
print("⚠️  MANUAL DOWNLOAD REQUIRED: O-RAN Specifications")
print("=" * 60)

print("\nDownload ONLY these PDFs:")

print("\nWG1")
print("  • O-RAN Architecture Description")

print("\nWG3")
print("  • Near-RT RIC Architecture")
print("  • E2 Interface Specification")
print("  • E2SM-KPM Specification")

print("\nPortal:")
print("https://specifications.o-ran.org/")

print(f"\nPlace PDFs in: {DATA_DIR / 'oran'}/\n")


# ── 6. Custom 20-Fault Scenario Dataset ───────────────────────────────────────
CUSTOM_SCENARIOS = [
    {
        "scenario_id": f"CS-{i+1:02d}",
        "fault_description": fd,
        "relevant_clauses": clauses,
        "fault_type": ft,
    }
    for i, (fd, clauses, ft) in enumerate([
        ("Handover failure due to A3 offset misconfiguration",
         ["TS 38.331 §5.5.4.4", "TS 38.331 §5.5.3.2"], "HO_FAILURE"),
        ("Random access failure causing RRC setup rejection",
         ["TS 38.321 §5.1", "TS 38.331 §5.3.3"], "RRC_FAILURE"),
        ("PRB utilization exceeds 95% causing scheduling starvation",
         ["TS 38.214 §5.1.3", "TS 38.214 §6.1.2"], "CONGESTION"),
        ("Radio link failure timer T310 expiry",
         ["TS 38.331 §5.3.10", "TS 38.331 §5.3.11"], "LINK_FAILURE"),
        ("PDCP reordering timer expiry causing UL delay",
         ["TS 38.323 §5.2.2.4", "TS 38.323 §7.3"], "PDCP_DELAY"),
        ("Beam failure recovery procedure triggered",
         ["TS 38.331 §5.3.14", "TS 38.214 §6.1.1"], "BEAM_FAILURE"),
        ("Inter-cell interference degrading SINR on PDCCH",
         ["TS 38.214 §5.1.6", "TS 38.213 §10"], "INTERFERENCE"),
        ("UE detachment due to paging channel overflow",
         ["TS 38.401 §8.3", "TS 38.413 §8.6"], "UE_DETACH"),
        ("Timing advance drift causing uplink synchronization loss",
         ["TS 38.213 §4.2", "TS 38.321 §5.2"], "TIMING_DRIFT"),
        ("Cell capacity limit blocking new UE admissions",
         ["TS 38.401 §8.7", "O-RAN WG3 §6.2"], "CAPACITY_OVERFLOW"),
        ("CQI mismatch causing excessive HARQ retransmissions",
         ["TS 38.214 §5.2.2", "TS 38.321 §5.4.2"], "LINK_FAILURE"),
        ("PRACH preamble collision causing access failure",
         ["TS 38.321 §5.1.2", "TS 38.213 §8.1"], "RRC_FAILURE"),
        ("SRS misconfiguration causing UL CSI feedback errors",
         ["TS 38.331 §6.3.2", "TS 38.214 §6.1.1"], "INTERFERENCE"),
        ("RLC AM mode buffer stall causing throughput drop",
         ["TS 38.322 §5.3.3", "TS 38.322 §5.3.4"], "PDCP_DELAY"),
        ("X2 handover failure due to Xn interface congestion",
         ["TS 38.401 §8.8", "TS 38.413 §8.4"], "HO_FAILURE"),
        ("CSI-RS beam sweeping failure in massive MIMO",
         ["TS 38.214 §5.2.2.3", "TS 38.331 §5.3.14.2"], "BEAM_FAILURE"),
        ("Secondary cell addition failure in NR-DC",
         ["TS 38.331 §5.3.5.6", "TS 38.401 §8.9"], "HO_FAILURE"),
        ("PDCCH decoding failure causing DL scheduling loss",
         ["TS 38.213 §10", "TS 38.214 §5.1.6"], "INTERFERENCE"),
        ("RRC reconfiguration failure during HO execution",
         ["TS 38.331 §5.3.5.5", "TS 38.331 §5.5.4.4"], "HO_FAILURE"),
        ("gNB-CU PDCP entity setup failure in F1 split",
         ["TS 38.401 §6.1.3", "TS 38.473 §8.2"], "LINK_FAILURE"),
    ])
]


def generate_custom_scenarios():
    print("\n⚙️  Generating custom 20-fault→clause benchmark dataset...")
    out_path = DATA_DIR / "custom_scenarios" / "fault_clause_mapping.json"
    with open(out_path, "w") as f:
        json.dump(CUSTOM_SCENARIOS, f, indent=2)
    print(f"   ✅ 20 custom fault scenarios saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  FCRAG 2.0 — Data & Model Download")
    print("=" * 60)

    skip_models = "--skip-models" in sys.argv
    skip_hf = "--skip-hf" in sys.argv

    if not skip_hf:
        download_teleqna()
        download_tele_eval()
    else:
        print("\n⏭️  Skipping HuggingFace dataset downloads (--skip-hf)")

    if not skip_models:
        download_models()
    else:
        print("\n⏭️  Skipping model downloads (--skip-models)")

    generate_simu5g_logs()
    generate_custom_scenarios()
    print_3gpp_instructions()

    print("\n" + "=" * 60)
    print("✅ Phase 0 data setup complete!")
    print("   Next: Place 3GPP PDFs in data/3gpp/")
    print("         Place O-RAN PDFs in data/oran/")
    print("         Then run: python scripts/ingest_all.py")
    print("=" * 60)
