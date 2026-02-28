"""Model serialization for ProbFlow belief networks.

Provides :func:`save_model` and :func:`load_model` for exporting and
importing :class:`~probflow.networks.dag.BeliefNetwork` instances to
and from JSON files.

Features:

* **JSON export** with distribution parameters, state labels, and
  graph structure.
* **Versioning** via a ``format_version`` field for backward
  compatibility.
* **Pickle fallback** for complex (non-array) CPDs such as callables.
* **Validation on load**: DAG acyclicity, probability constraints
  (0 <= p <= 1, sum(p) == 1 for priors).
"""

from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np

from probflow.networks.dag import BeliefNetwork

# Current serialization format version
FORMAT_VERSION = "1.0"

# Tolerance for probability sum validation
_PROB_SUM_TOL = 1e-6


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #


def save_model(
    network: BeliefNetwork,
    filepath: Union[str, Path],
) -> None:
    """Export a :class:`BeliefNetwork` to a JSON file.

    Parameters
    ----------
    network : BeliefNetwork
        The network to serialize.
    filepath : str or Path
        Destination file path.  Parent directories must exist.

    Raises
    ------
    TypeError
        If *network* is not a :class:`BeliefNetwork`.
    """
    if not isinstance(network, BeliefNetwork):
        raise TypeError(
            f"Expected BeliefNetwork, got {type(network).__name__}"
        )

    data = _network_to_dict(network)
    filepath = Path(filepath)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_model(
    filepath: Union[str, Path],
    *,
    allow_pickle: bool = False,
) -> BeliefNetwork:
    """Reconstruct a :class:`BeliefNetwork` from a JSON file.

    Validates:
    * DAG acyclicity (no directed cycles).
    * Probability constraints: all values in [0, 1] and root priors
      sum to 1 (within tolerance).

    Parameters
    ----------
    filepath : str or Path
        Path to a JSON file previously written by :func:`save_model`.
    allow_pickle : bool, optional
        If *True*, allow loading pickle-serialized objects (e.g.,
        callable CPDs).  **Warning**: unpickling data from untrusted
        sources can execute arbitrary code.  Default is *False*.

    Returns
    -------
    BeliefNetwork
        The reconstructed network.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file is corrupted, contains cycles, violates
        probability constraints, or contains pickle data when
        *allow_pickle* is *False*.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted model file: {exc}") from exc

    return _dict_to_network(data, allow_pickle=allow_pickle)


# ------------------------------------------------------------------ #
#  Internal: serialization helpers
# ------------------------------------------------------------------ #


def _network_to_dict(network: BeliefNetwork) -> Dict[str, Any]:
    """Convert a BeliefNetwork to a JSON-serializable dictionary."""
    nodes_data: List[Dict[str, Any]] = []

    for name in network.nodes:
        parents = [
            p for p, c in network.edges if c == name
        ]
        states = network.get_states(name)
        cpd = network._cpds[name]

        node_entry: Dict[str, Any] = {
            "name": name,
            "states": states,
            "parents": parents,
        }

        if isinstance(cpd, np.ndarray):
            node_entry["distribution"] = cpd.tolist()
            node_entry["distribution_type"] = "array"
        elif callable(cpd):
            # Pickle fallback for callable CPDs
            pickled = pickle.dumps(cpd)
            node_entry["distribution"] = base64.b64encode(pickled).decode(
                "ascii"
            )
            node_entry["distribution_type"] = "pickle"
        else:
            # Generic pickle fallback
            pickled = pickle.dumps(cpd)
            node_entry["distribution"] = base64.b64encode(pickled).decode(
                "ascii"
            )
            node_entry["distribution_type"] = "pickle"

        nodes_data.append(node_entry)

    return {
        "format_version": FORMAT_VERSION,
        "nodes": nodes_data,
        "edges": [
            {"parent": p, "child": c} for p, c in network.edges
        ],
    }


def _dict_to_network(
    data: Dict[str, Any],
    *,
    allow_pickle: bool = False,
) -> BeliefNetwork:
    """Reconstruct a BeliefNetwork from a deserialized dictionary."""
    # --- version handling ---
    version = data.get("format_version")
    if version is None:
        raise ValueError(
            "Missing 'format_version' in model file"
        )

    _migrate_if_needed(data, version)

    # --- validate structure ---
    if "nodes" not in data:
        raise ValueError("Missing 'nodes' in model file")

    nodes_data = data["nodes"]
    if not isinstance(nodes_data, list):
        raise ValueError("'nodes' must be a list")

    # --- validate DAG acyclicity ---
    _validate_acyclicity(data)

    # --- reconstruct network ---
    network = BeliefNetwork()

    for node_data in nodes_data:
        name = node_data.get("name")
        if name is None:
            raise ValueError("Node entry missing 'name'")

        states = node_data.get("states")
        parents = node_data.get("parents", [])

        dist_type = node_data.get("distribution_type", "array")
        raw_dist = node_data.get("distribution")

        if raw_dist is None:
            raise ValueError(
                f"Node '{name}' missing 'distribution'"
            )

        if dist_type == "array":
            distribution = np.array(raw_dist, dtype=np.float64)
        elif dist_type == "pickle":
            if not allow_pickle:
                raise ValueError(
                    f"Node '{name}' uses pickle serialization. "
                    f"Set allow_pickle=True to load pickle data. "
                    f"Warning: only load pickle data from trusted sources."
                )
            pickled_bytes = base64.b64decode(raw_dist)
            distribution = pickle.loads(pickled_bytes)  # noqa: S301
        else:
            raise ValueError(
                f"Unknown distribution_type '{dist_type}' for node '{name}'"
            )

        # --- probability constraint validation ---
        if isinstance(distribution, np.ndarray):
            _validate_probabilities(name, distribution, parents)

        if callable(distribution):
            # Callable CPDs cannot go through add_node() which calls
            # np.asarray; inject directly into the network internals.
            num_states = len(states) if states else 2
            if states is None:
                states = [f"s{i}" for i in range(num_states)]
            placeholder = np.ones(num_states) / num_states
            if parents:
                # Build a placeholder CPT with the right shape
                parent_cards = [
                    len(network.get_states(p)) for p in parents
                ]
                shape = parent_cards + [num_states]
                placeholder = np.ones(shape) / num_states
            network.add_node(
                name,
                placeholder,
                parents=parents if parents else None,
                states=states,
            )
            # Replace with the actual callable
            network._cpds[name] = distribution
        else:
            network.add_node(
                name,
                distribution,
                parents=parents if parents else None,
                states=states,
            )

    return network


# ------------------------------------------------------------------ #
#  Validation helpers
# ------------------------------------------------------------------ #


def _validate_acyclicity(data: Dict[str, Any]) -> None:
    """Check that the graph defined in *data* is a DAG (no cycles)."""
    g = nx.DiGraph()
    for node_data in data["nodes"]:
        name = node_data.get("name")
        if name is not None:
            g.add_node(name)
    edges = data.get("edges", [])
    for edge in edges:
        g.add_edge(edge["parent"], edge["child"])

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(
            "Model file contains a cycle; the graph is not a valid DAG"
        )


def _validate_probabilities(
    name: str,
    distribution: np.ndarray,
    parents: List[str],
) -> None:
    """Validate probability constraints on a distribution array.

    Checks:
    * All values are in [0, 1].
    * For root nodes (no parents), the distribution sums to 1.
    * For child nodes, each row of the CPT (conditioned on parent
      states) sums to 1.
    """
    if np.any(distribution < 0) or np.any(distribution > 1):
        raise ValueError(
            f"Node '{name}': distribution values must be in [0, 1]"
        )

    if not parents:
        # Root node: prior must sum to 1
        total = float(distribution.sum())
        if abs(total - 1.0) > _PROB_SUM_TOL:
            raise ValueError(
                f"Node '{name}': prior distribution sums to {total}, "
                f"expected 1.0"
            )
    else:
        # Child node: each conditional slice (last axis) must sum to 1
        if distribution.ndim < 2:
            return
        # Sum along the last axis (child states)
        row_sums = distribution.sum(axis=-1)
        if np.any(np.abs(row_sums - 1.0) > _PROB_SUM_TOL):
            raise ValueError(
                f"Node '{name}': CPT rows do not all sum to 1.0"
            )


# ------------------------------------------------------------------ #
#  Version migration
# ------------------------------------------------------------------ #


def _migrate_if_needed(data: Dict[str, Any], version: str) -> None:
    """Apply migrations to bring *data* up to the current format.

    Currently supports:
    * ``"0.1"`` -> ``"1.0"``: adds ``distribution_type`` field to
      nodes that lack it.
    """
    if version == FORMAT_VERSION:
        return

    if version == "0.1":
        _migrate_v01_to_v10(data)
        return

    # Unknown future versions: attempt to load as-is but warn
    # (forward compatibility best-effort)


def _migrate_v01_to_v10(data: Dict[str, Any]) -> None:
    """Migrate from format version 0.1 to 1.0.

    Version 0.1 stored distributions without the ``distribution_type``
    field.  We add ``"array"`` as the default type.
    """
    for node_data in data.get("nodes", []):
        if "distribution_type" not in node_data:
            node_data["distribution_type"] = "array"
    data["format_version"] = FORMAT_VERSION
