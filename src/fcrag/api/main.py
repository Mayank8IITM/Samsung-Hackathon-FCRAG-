"""
fcrag/api/main.py -- FCRAG 2.0 FastAPI Application
===================================================
Main entrypoint for the REST API. Wraps the LangGraph reasoning
pipeline and the explain module into HTTP endpoints.
"""

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from fcrag.api.schemas import AnalyzeFaultRequest, AnalyzeFaultResponse
from fcrag.reason.graph import run_pipeline
from fcrag.explain.reporter import build_output_package

app = FastAPI(
    title="FCRAG 2.0 API",
    description="Fault-Conditioned Retrieval-Augmented Generation for 5G/6G Networks",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/analyze-fault", response_model=AnalyzeFaultResponse)
def analyze_fault(request: AnalyzeFaultRequest):
    """
    End-to-end RCA pipeline endpoint.
    Takes KPI deltas, runs retrieval & reasoning, returns structured output.
    """
    try:
        # Convert request to anomaly event format expected by the pipeline
        anomaly_event = {
            "event_id": f"api-{int(time.time())}",
            "cell_id": request.cell_id,
            "severity": request.severity,
            "kpi_deltas": request.kpi_snapshot,
            "anomaly_score": request.anomaly_score,
        }

        # Run the LangGraph pipeline
        state = run_pipeline(anomaly_event, verbose=False)

        # Package the state into the final output JSON
        output_pkg = build_output_package(state)

        return output_pkg

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
