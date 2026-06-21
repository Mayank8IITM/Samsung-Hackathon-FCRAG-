"""
fcrag/api/schemas.py -- FastAPI Request/Response Schemas
=========================================================
Pydantic models for the FCRAG API endpoints.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field

class AnalyzeFaultRequest(BaseModel):
    cell_id: str = Field(..., example="Cell-42")
    severity: str = Field(default="UNKNOWN", example="HIGH")
    kpi_snapshot: dict[str, float] = Field(
        ...,
        example={"ho_success_rate_drop": 0.29, "throughput_drop_pct": 15.0}
    )
    mode: Literal["auto", "manual"] = Field(default="auto")
    anomaly_score: float = Field(default=0.0)

class CorrectiveActionModel(BaseModel):
    priority: int
    action: str
    spec_reference: str = ""

class AnalyzeFaultResponse(BaseModel):
    status: str
    rca_summary: str
    causal_chain: list[dict[str, Any]]
    causal_graph: dict[str, Any]
    corrective_actions: list[CorrectiveActionModel]
    citations: list[str]
    faithfulness_score: float
    confidence: float
    latency_ms: int
    errors: list[str] = []
