"""
tests/test_simu5g.py — FCRAG 2.0 Phase 1.5 Tests
==================================================
Tests for the Simu5G Narrative Generator.
"""
import json
import pytest
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kpi_csv(path: Path, fault_type: str, cell_id: str = "Cell-01") -> None:
    """Write a minimal Simu5G KPI CSV to *path*."""
    rows = []
    # 5 normal rows, then 5 fault rows
    for i in range(5):
        rows.append(
            {
                "timestamp": f"2024-01-01 00:0{i}:00 UTC",
                "cell_id": cell_id,
                "fault_type": fault_type,
                "fault_active": 0,
                "ho_sr": 95.0,
                "rsrp": -80.0,
                "rsrq": -10.0,
                "sinr": 15.0,
                "throughput": 100.0,
                "prb": 30.0,
                "latency": 5.0,
                "rrc_sr": 99.0,
                "rrc_reest": 0.1,
                "rlf": 0.0,
                "prach_fail": 0.0,
                "connected_ues": 50.0,
                "ul_timing_err": 0.0,
            }
        )
    for i in range(5, 10):
        rows.append(
            {
                "timestamp": f"2024-01-01 00:0{i}:00 UTC",
                "cell_id": cell_id,
                "fault_type": fault_type,
                "fault_active": 1,
                "ho_sr": 30.0,     # big drop — easy to detect
                "rsrp": -100.0,
                "rsrq": -20.0,
                "sinr": 2.0,
                "throughput": 10.0,
                "prb": 95.0,
                "latency": 50.0,
                "rrc_sr": 60.0,
                "rrc_reest": 5.0,
                "rlf": 3.0,
                "prach_fail": 8.0,
                "connected_ues": 10.0,
                "ul_timing_err": 20.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_narratives_produces_output(tmp_path, monkeypatch):
    """Narrative generator should create JSONL output for fault CSVs."""
    import fcrag.ingest.simu5g_generator as sg

    # Create a fake kpi_logs directory with one fault CSV
    kpi_dir = tmp_path / "kpi_logs"
    kpi_dir.mkdir()
    _make_kpi_csv(kpi_dir / "run_0000_HO_FAILURE_Cell01.csv", "HO_FAILURE")

    # Redirect paths inside the module
    monkeypatch.setattr(sg, "KPI_LOGS_DIR", kpi_dir)
    output_path = tmp_path / "fault_narratives_normalized.jsonl"
    monkeypatch.setattr(sg, "OUTPUT_FILE", output_path)

    sg.generate_narratives()

    assert output_path.exists(), "Output JSONL should be created"
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "One narrative per fault file"

    record = json.loads(lines[0])
    assert record["fault_id"] == "HO_FAILURE"
    assert "Cell-01" in record["cell"]
    assert "text" in record and len(record["text"]) > 20
    assert "anomaly_vector" in record


def test_generate_narratives_skips_normal(tmp_path, monkeypatch):
    """NORMAL runs should not produce narratives."""
    import fcrag.ingest.simu5g_generator as sg

    kpi_dir = tmp_path / "kpi_logs"
    kpi_dir.mkdir()
    _make_kpi_csv(kpi_dir / "run_0150_NORMAL_Cell43.csv", "NORMAL", cell_id="Cell43")

    output_path = tmp_path / "fault_narratives_normalized.jsonl"
    monkeypatch.setattr(sg, "KPI_LOGS_DIR", kpi_dir)
    monkeypatch.setattr(sg, "OUTPUT_FILE", output_path)

    sg.generate_narratives()

    # Output file should NOT be created (no narratives = nothing written)
    assert not output_path.exists() or output_path.stat().st_size == 0


def test_narrative_text_contains_fault_id_and_kpis(tmp_path, monkeypatch):
    """Narrative text must mention the fault type and at least one KPI deviation."""
    import fcrag.ingest.simu5g_generator as sg

    kpi_dir = tmp_path / "kpi_logs"
    kpi_dir.mkdir()
    _make_kpi_csv(kpi_dir / "run_0000_PRB_CONGESTION_Cell27.csv", "PRB_CONGESTION", cell_id="Cell27")

    output_path = tmp_path / "fault_narratives_normalized.jsonl"
    monkeypatch.setattr(sg, "KPI_LOGS_DIR", kpi_dir)
    monkeypatch.setattr(sg, "OUTPUT_FILE", output_path)

    sg.generate_narratives()

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    text = record["text"]
    assert "PRB_CONGESTION" in text, "Fault ID should appear in narrative text"
    # At least one KPI deviation (sigma value) should appear in the text
    assert "σ" in text, "KPI deviation in sigma notation should appear in text"


def test_multiple_fault_files(tmp_path, monkeypatch):
    """Multiple fault CSVs should each produce one narrative."""
    import fcrag.ingest.simu5g_generator as sg

    kpi_dir = tmp_path / "kpi_logs"
    kpi_dir.mkdir()
    faults = [
        ("run_0000_HO_FAILURE_Cell01.csv", "HO_FAILURE", "Cell01"),
        ("run_0001_INTERFERENCE_Cell24.csv", "INTERFERENCE", "Cell24"),
        ("run_0002_RLF_SPIKE_Cell03.csv", "RLF_SPIKE", "Cell03"),
    ]
    for fname, ftype, cell in faults:
        _make_kpi_csv(kpi_dir / fname, ftype, cell_id=cell)

    output_path = tmp_path / "fault_narratives_normalized.jsonl"
    monkeypatch.setattr(sg, "KPI_LOGS_DIR", kpi_dir)
    monkeypatch.setattr(sg, "OUTPUT_FILE", output_path)

    sg.generate_narratives()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "Three fault files → three narratives"
    fault_ids = {json.loads(l)["fault_id"] for l in lines}
    assert fault_ids == {"HO_FAILURE", "INTERFERENCE", "RLF_SPIKE"}
