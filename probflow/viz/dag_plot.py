"""DAG (Directed Acyclic Graph) plotting for probabilistic networks.

Provides :func:`plot_network` which renders a Bayesian or causal network
as a publication-quality graph using Graphviz.

Node types
----------
* **observed** – rendered as a gray filled box (rectangle).
* **query** – rendered with a bold outline.
* **latent** – rendered as an ellipse (default Graphviz shape).

Edge labels can optionally display conditional probability values.
The *critical path* (longest dependency chain) can be highlighted in red.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import graphviz


# ---------------------------------------------------------------------------
# Public data helpers
# ---------------------------------------------------------------------------

def _node_id(name: str) -> str:
    """Return a sanitised Graphviz node identifier."""
    return name.replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Critical-path computation
# ---------------------------------------------------------------------------

def _build_adjacency(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Return (adjacency list, in-degree map) for the network."""
    adj: Dict[str, List[str]] = {n["name"]: [] for n in nodes}
    indeg: Dict[str, int] = {n["name"]: 0 for n in nodes}
    for e in edges:
        src, dst = e["from"], e["to"]
        adj.setdefault(src, []).append(dst)
        indeg.setdefault(dst, 0)
        indeg[dst] += 1
        # Ensure src is present
        indeg.setdefault(src, 0)
    return adj, indeg


def _longest_path(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
) -> List[str]:
    """Return the *critical path* — the longest dependency chain.

    Uses a topological-sort based dynamic-programming approach.
    If the graph is empty or has no edges the result is an empty list.
    """
    if not nodes or not edges:
        return []

    adj, indeg = _build_adjacency(nodes, edges)
    all_names = [n["name"] for n in nodes]

    # Kahn's algorithm for topological order
    topo: List[str] = []
    queue = [n for n in all_names if indeg.get(n, 0) == 0]
    while queue:
        node = queue.pop(0)
        topo.append(node)
        for child in adj.get(node, []):
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)

    # DP for longest path
    dist: Dict[str, int] = {n: 0 for n in all_names}
    pred: Dict[str, Optional[str]] = {n: None for n in all_names}

    for node in topo:
        for child in adj.get(node, []):
            if dist[node] + 1 > dist[child]:
                dist[child] = dist[node] + 1
                pred[child] = node

    # Reconstruct path ending at the node with largest distance
    if not dist:
        return []
    end = max(dist, key=lambda n: dist[n])
    if dist[end] == 0:
        return []

    path: List[str] = []
    cur: Optional[str] = end
    while cur is not None:
        path.append(cur)
        cur = pred[cur]
    path.reverse()
    return path


def _critical_edges(path: List[str]) -> set:
    """Return a set of (src, dst) tuples along the critical path."""
    return {(path[i], path[i + 1]) for i in range(len(path) - 1)}


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def _add_legend(dot: graphviz.Digraph) -> None:
    """Append a legend sub-graph explaining node types."""
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Legend", style="dashed", fontsize="10")
        legend.node(
            "_legend_observed",
            "Observed",
            shape="box",
            style="filled",
            fillcolor="lightgray",
        )
        legend.node(
            "_legend_query",
            "Query",
            shape="ellipse",
            style="bold",
            penwidth="3",
        )
        legend.node(
            "_legend_latent",
            "Latent",
            shape="ellipse",
        )
        # Invisible edges to stack legend items vertically
        legend.edge("_legend_observed", "_legend_query", style="invis")
        legend.edge("_legend_query", "_legend_latent", style="invis")


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def plot_network(
    network: Dict[str, Any],
    output: str = "graph.png",
    highlight_critical_path: bool = False,
    show_probabilities: bool = True,
) -> graphviz.Digraph:
    """Render a probabilistic network as a Graphviz DAG.

    Parameters
    ----------
    network : dict
        A dictionary describing the network with the following keys:

        * ``"nodes"`` – list of dicts, each with:
            - ``"name"`` (str): unique node identifier.
            - ``"type"`` (str): one of ``"observed"``, ``"query"``,
              or ``"latent"`` (default ``"latent"``).
        * ``"edges"`` – list of dicts, each with:
            - ``"from"`` (str): source node name.
            - ``"to"`` (str): target node name.
            - ``"probability"`` (float, optional): conditional probability
              value shown as an edge label when *show_probabilities* is
              ``True``.

    output : str, default ``"graph.png"``
        File path for the rendered output.  The extension determines the
        format: ``.png``, ``.svg``, or ``.dot`` / ``.gv``.

    highlight_critical_path : bool, default ``False``
        When ``True``, the longest dependency chain is drawn in red.

    show_probabilities : bool, default ``True``
        When ``True``, edge labels display the ``"probability"`` value
        (if provided) on each edge.

    Returns
    -------
    graphviz.Digraph
        The Graphviz Digraph object (also saved to *output*).
    """
    nodes: List[Dict[str, Any]] = network.get("nodes", [])
    edges_list: List[Dict[str, Any]] = network.get("edges", [])

    # Determine output format from file extension
    _, ext = os.path.splitext(output)
    ext = ext.lstrip(".").lower()

    is_dot_format = ext in ("dot", "gv")
    render_format = ext if ext in ("png", "svg") else "png"

    dot = graphviz.Digraph(
        format=render_format,
        engine="dot",
    )
    dot.attr(rankdir="TB")  # top-to-bottom hierarchical layout
    dot.attr("graph", nodesep="0.5", ranksep="0.75")  # prevent overlaps

    # Critical path
    crit_edges: set = set()
    crit_nodes: set = set()
    if highlight_critical_path:
        path = _longest_path(nodes, edges_list)
        crit_edges = _critical_edges(path)
        crit_nodes = set(path)

    # --- Nodes ---
    for node in nodes:
        name = node["name"]
        nid = _node_id(name)
        ntype = node.get("type", "latent")

        attrs: Dict[str, str] = {"label": name}

        if ntype == "observed":
            attrs["shape"] = "box"
            attrs["style"] = "filled"
            attrs["fillcolor"] = "lightgray"
        elif ntype == "query":
            attrs["shape"] = "ellipse"
            attrs["style"] = "bold"
            attrs["penwidth"] = "3"
        else:  # latent
            attrs["shape"] = "ellipse"

        # Critical-path node highlighting
        if highlight_critical_path and name in crit_nodes:
            attrs["color"] = "red"
            if "style" in attrs:
                attrs["style"] += ",bold"
            else:
                attrs["style"] = "bold"

        dot.node(nid, **attrs)

    # --- Edges ---
    for edge in edges_list:
        src = _node_id(edge["from"])
        dst = _node_id(edge["to"])
        edge_attrs: Dict[str, str] = {}

        if show_probabilities and "probability" in edge:
            edge_attrs["label"] = f" {edge['probability']:.2f} "

        if highlight_critical_path and (edge["from"], edge["to"]) in crit_edges:
            edge_attrs["color"] = "red"
            edge_attrs["penwidth"] = "2.5"

        dot.edge(src, dst, **edge_attrs)

    # --- Legend ---
    _add_legend(dot)

    # --- Render / save ---
    if is_dot_format:
        # Write raw DOT source
        with open(output, "w") as fh:
            fh.write(dot.source)
    else:
        # graphviz.render writes <filename>.png or <filename>.svg
        # We strip the extension for the filename argument because
        # graphviz appends the format extension automatically.
        base = output
        if output.endswith(f".{render_format}"):
            base = output[: -len(render_format) - 1]
        dot.render(filename=base, cleanup=True)

    return dot
