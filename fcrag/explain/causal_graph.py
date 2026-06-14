"""
fcrag/explain/causal_graph.py -- NetworkX Causal Graph Builder
================================================================
Converts a list of CausalNode dicts into a NetworkX directed graph,
and exports it to standard node-link data format for UI visualization.
"""

from __future__ import annotations

import networkx as nx
from typing import Any


def build_causal_graph(causal_chain: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a NetworkX directed graph from a causal chain.

    Parameters
    ----------
    causal_chain : List of dicts representing CausalNode objects
                   (keys: node, cause, evidence)

    Returns
    -------
    dict : NetworkX node-link data suitable for JSON serialization
           and UI rendering (e.g. D3.js, Cytoscape, Streamlit agraph)
    """
    G = nx.DiGraph()

    for idx, item in enumerate(causal_chain):
        # The 'node' is typically the effect (e.g. SYMPTOM, ROOT_CAUSE)
        # The 'cause' is the text describing it.
        # We will map this as: [Cause Text] -> [Node Label]
        # Or, if there's a sequence, [Previous Item] -> [Current Item]
        
        node_id = item.get("node", f"NODE_{idx}")
        cause_text = item.get("cause", "")
        evidence = item.get("evidence", "")

        G.add_node(
            node_id,
            label=node_id,
            description=cause_text,
            evidence=evidence,
        )

        # If it's a sequence (Symptom -> Trigger -> Cause), link them in reverse order of discovery?
        # Typically the chain is ordered: Symptom, Trigger, Root Cause
        # But causality flows: Root Cause -> Trigger -> Symptom
        # We'll just link them chronologically as they appear in the list for simplicity:
        if idx > 0:
            prev_node_id = causal_chain[idx - 1].get("node", f"NODE_{idx - 1}")
            # Edge from previous to current
            G.add_edge(prev_node_id, node_id)

    return nx.node_link_data(G)
