"""Model serialization and deserialization for BeliefNetwork.

Provides :func:`save_model` and :func:`load_model` for persisting
:class:`~probflow.networks.dag.BeliefNetwork` instances to disk in
JSON format.  The format includes a version field for backward
compatibility.

Features:

* JSON export with distribution parameters (priors and CPTs).
* Format versioning for backward-compatible deserialization.
* Pickle fallback for complex objects (e.g. callable CPDs) that
  cannot be represented in JSON, stored as base64-encoded blobs.
* Validation on load: DAG acyclicity and probability constraints
  (0 <= p <= 1, sum(p) = 1 for root priors).
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
FORMAT_VERSION = 1

# Tolerance for probability constraint checks
_PROB_TOL = 1e-6


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

    payload = _serialize_network(network)
    filepath = Path(filepath)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_model(
    filepath: Union[str, Path],
) -> BeliefNetwork:
    """Reconstruct a :class:`BeliefNetwork` from a JSON file.

    Validates DAG acyclicity and probability constraints on load.

    Parameters
    ----------
    filepath : str or Path
        Path to a JSON file previously created by :func:`save_model`.

    Returns
    -------
    BeliefNetwork
        The reconstructed network.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file is corrupted, has an unsupported version, or
        fails validation (cycles, invalid probabilities).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted model file: {exc}") from exc

    return _deserialize_network(payload)


# ------------------------------------------------------------------ #
#  Serialization helpers
# ------------------------------------------------------------------ #


def _serialize_network(network: BeliefNetwork) -> Dict[str, Any]:
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

        # Try JSON-native serialization; fall back to pickle for
        # non-serializable objects (e.g. callable CPDs).
        if isinstance(cpd, np.ndarray):
            node_entry["distribution"] = cpd.tolist()
            node_entry["distribution_shape"] = list(cpd.shape)
            node_entry["encoding"] = "json"
        else:
            node_entry["distribution_pickle"] = _pickle_to_base64(cpd)
            node_entry["encoding"] = "pickle"

        nodes_data.append(node_entry)

    return {
        "format_version": FORMAT_VERSION,
        "nodes": nodes_data,
        "edges": [
            {"parent": p, "child": c} for p, c in network.edges
        ],
    }


def _deserialize_network(payload: Dict[str, Any]) -> BeliefNetwork:
    """Reconstruct a BeliefNetwork from a deserialized dictionary."""
    # ---- version check ------------------------------------------------ #
    version = payload.get("format_version")
    if version is None:
        raise ValueError("Missing 'format_version' in model file")

    if version > FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format version {version} "
            f"(max supported: {FORMAT_VERSION})"
        )

    # Migrate from older versions if necessary
    nodes_data = _migrate(payload, version)

    # ---- rebuild network ---------------------------------------------- #
    bn = BeliefNetwork()

    for node_entry in nodes_data:
        name = node_entry.get("name")
        if name is None:
            raise ValueError("Node entry missing 'name' field")

        states = node_entry.get("states")
        parents = node_entry.get("parents", [])
        encoding = node_entry.get("encoding", "json")

        if encoding == "pickle":
            blob = node_entry.get("distribution_pickle")
            if blob is None:
                raise ValueError(
                    f"Node '{name}': pickle encoding but no "
                    "'distribution_pickle' field"
                )
            distribution = _base64_to_unpickle(blob)

            # Pickle-encoded objects may not be numpy arrays, so
            # add the node with a dummy distribution then replace.
            n_states = len(states) if states else 2
            if parents:
                parent_cards = [
                    len(bn.get_states(p)) for p in parents
                ]
                dummy_shape = parent_cards + [n_states]
                dummy = np.ones(dummy_shape) / n_states
            else:
                dummy = np.ones(n_states) / n_states
            bn.add_node(
                name,
                dummy,
                parents=parents if parents else None,
                states=states,
            )
            bn._cpds[name] = distribution
        else:
            raw = node_entry.get("distribution")
            if raw is None:
                raise ValueError(
                    f"Node '{name}': missing 'distribution' field"
                )
            distribution = np.array(raw, dtype=np.float64)

            bn.add_node(
                name,
                distribution,
                parents=parents if parents else None,
                states=states,
            )

    # ---- validation --------------------------------------------------- #
    _validate_acyclicity(bn)
    _validate_probabilities(bn)

    return bn


# ------------------------------------------------------------------ #
#  Version migration
# ------------------------------------------------------------------ #


def _migrate(
    payload: Dict[str, Any], version: int
) -> List[Dict[str, Any]]:
    """Apply version migrations to bring *payload* up to current format.

    Parameters
    ----------
    payload : dict
        The raw deserialized JSON.
    version : int
        The format version found in the file.

    Returns
    -------
    list of dict
        The (possibly migrated) list of node entries.
    """
    nodes_data = payload.get("nodes")
    if nodes_data is None:
        raise ValueError("Missing 'nodes' in model file")

    # Version 1 is current -- no migration needed.
    # Future versions would add migration steps here:
    # if version < 2:
    #     nodes_data = _migrate_v1_to_v2(nodes_data)

    return nodes_data


# ------------------------------------------------------------------ #
#  Validation
# ------------------------------------------------------------------ #


def _validate_acyclicity(network: BeliefNetwork) -> None:
    """Verify the reconstructed graph is a DAG (no cycles).

    Raises
    ------
    ValueError
        If the graph contains a cycle.
    """
    if not nx.is_directed_acyclic_graph(network._graph):
        raise ValueError(
            "Loaded network contains a cycle and is not a valid DAG"
        )


def _validate_probabilities(network: BeliefNetwork) -> None:
    """Check probability constraints on all CPDs.

    * All values must be in [0, 1].
    * Root priors must sum to 1.
    * Each row of a CPT (conditioned on parent states) must sum to 1.

    Raises
    ------
    ValueError
        If any constraint is violated.
    """
    for name in network.nodes:
        cpd = network._cpds[name]

        if not isinstance(cpd, np.ndarray):
            # Pickle-encoded callable CPDs cannot be validated
            continue

        # Check range [0, 1]
        if np.any(cpd < -_PROB_TOL) or np.any(cpd > 1.0 + _PROB_TOL):
            raise ValueError(
                f"Node '{name}': distribution values outside [0, 1]"
            )

        # Check sum-to-one
        parents = [p for p, c in network.edges if c == name]

        if not parents:
            # Root prior: should sum to 1
            total = cpd.sum()
            if abs(total - 1.0) > _PROB_TOL:
                raise ValueError(
                    f"Node '{name}': prior does not sum to 1 "
                    f"(sum={total:.6f})"
                )
        else:
            # CPT: each conditional distribution (last axis) sums to 1
            # Reshape to (product_of_parent_states, self_states)
            n_self = cpd.shape[-1]
            flat = cpd.reshape(-1, n_self)
            row_sums = flat.sum(axis=1)
            bad = np.abs(row_sums - 1.0) > _PROB_TOL
            if np.any(bad):
                raise ValueError(
                    f"Node '{name}': CPT rows do not all sum to 1"
                )


# ------------------------------------------------------------------ #
#  Pickle helpers
# ------------------------------------------------------------------ #


def _pickle_to_base64(obj: Any) -> str:
    """Serialize *obj* via pickle and return a base64-encoded string."""
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(raw).decode("ascii")


def _base64_to_unpickle(b64: str) -> Any:
    """Decode a base64 string and unpickle the result."""
    raw = base64.b64decode(b64.encode("ascii"))
    return pickle.loads(raw)  # noqa: S301
