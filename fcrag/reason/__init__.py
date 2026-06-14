"""FCRAG 2.0 -- fcrag.reason package

Public API
----------
from fcrag.reason import run_pipeline, build_graph, FCRAGState
"""

from fcrag.reason.state import FCRAGState, CausalNode, Claim, Citation, CorrectiveAction
from fcrag.reason.graph import build_graph, run_pipeline
from fcrag.reason.llm_client import FCRAGLLMClient

__all__ = [
    "FCRAGState",
    "CausalNode",
    "Claim",
    "Citation",
    "CorrectiveAction",
    "build_graph",
    "run_pipeline",
    "FCRAGLLMClient",
]
