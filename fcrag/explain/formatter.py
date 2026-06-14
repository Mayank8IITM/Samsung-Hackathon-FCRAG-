"""
fcrag/explain/formatter.py -- FCRAG 2.0 Output Formatter
========================================================
Converts the final output package into readable formats like Markdown.
"""

from __future__ import annotations
from typing import Any

def format_markdown(output_pkg: dict[str, Any]) -> str:
    """
    Format the RCA output package as a Markdown report.
    """
    status = output_pkg.get("status", "UNKNOWN")
    summary = output_pkg.get("rca_summary", "")
    latency = output_pkg.get("latency_ms", 0)
    faithfulness = output_pkg.get("faithfulness_score", 0.0)
    confidence = output_pkg.get("confidence", 0.0)

    lines = [
        f"# Root Cause Analysis Report",
        f"**Status:** {status}  ",
        f"**Confidence:** {confidence:.2f} (Faithfulness: {faithfulness:.2f})  ",
        f"**Processing Time:** {latency}ms\n",
    ]

    if status == "INSUFFICIENT_CONTEXT" or status == "ERROR":
        lines.append(f"> [!WARNING]\n> {summary}")
        return "\n".join(lines)

    lines.append(f"## Summary\n{summary}\n")

    causal_chain = output_pkg.get("causal_chain", [])
    if causal_chain:
        lines.append("## Causal Chain")
        for node in causal_chain:
            n_type = node.get("node", "")
            cause = node.get("cause", "")
            ev = node.get("evidence", "")
            lines.append(f"- **{n_type}:** {cause}")
            if ev:
                lines.append(f"  - *Evidence:* {ev}")
        lines.append("")

    actions = output_pkg.get("corrective_actions", [])
    if actions:
        lines.append("## Corrective Actions")
        for act in sorted(actions, key=lambda x: x.get("priority", 99)):
            pri = act.get("priority", "")
            action_text = act.get("action", "")
            spec = act.get("spec_reference", "")
            text = f"{pri}. {action_text}"
            if spec:
                text += f" (Ref: {spec})"
            lines.append(text)
        lines.append("")

    citations = output_pkg.get("citations", [])
    if citations:
        lines.append("## Citations")
        for c in citations:
            lines.append(f"- {c}")

    errors = output_pkg.get("errors", [])
    if errors:
        lines.append("\n## Warnings")
        for e in errors:
            lines.append(f"- {e}")

    return "\n".join(lines)
