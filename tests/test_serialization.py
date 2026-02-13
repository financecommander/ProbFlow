"""Tests for probflow/integration/serialization.py.

Covers:
- Round-trip consistency (save → load produces identical network)
- Version migration and versioning
- Error handling (corrupted files, invalid data, missing fields)
- Pickle fallback for non-JSON-serializable CPDs
- DAG acyclicity validation
- Probability constraint validation (0≤p≤1, Σp=1)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from probflow.integration.serialization import (
    FORMAT_VERSION,
    load_model,
    save_model,
)
from probflow.networks.dag import BeliefNetwork


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _build_chain_abc() -> BeliefNetwork:
    """Build a 3-node chain A → B → C.

    P(A) = [0.4, 0.6]
    P(B|A) = [[0.9, 0.1],   # A=a0
              [0.3, 0.7]]   # A=a1
    P(C|B) = [[0.8, 0.2],   # B=b0
              [0.4, 0.6]]   # B=b1
    """
    bn = BeliefNetwork()
    bn.add_node("A", np.array([0.4, 0.6]), states=["a0", "a1"])
    bn.add_node(
        "B",
        np.array([[0.9, 0.1], [0.3, 0.7]]),
        parents=["A"],
        states=["b0", "b1"],
    )
    bn.add_node(
        "C",
        np.array([[0.8, 0.2], [0.4, 0.6]]),
        parents=["B"],
        states=["c0", "c1"],
    )
    return bn


def _build_collider() -> BeliefNetwork:
    """Build a collider: A → C ← B."""
    bn = BeliefNetwork()
    bn.add_node("A", np.array([0.5, 0.5]), states=["a0", "a1"])
    bn.add_node("B", np.array([0.5, 0.5]), states=["b0", "b1"])
    bn.add_node(
        "C",
        np.array([
            [[0.9, 0.1], [0.6, 0.4]],
            [[0.3, 0.7], [0.2, 0.8]],
        ]),
        parents=["A", "B"],
        states=["c0", "c1"],
    )
    return bn


# ------------------------------------------------------------------ #
#  Round-trip consistency
# ------------------------------------------------------------------ #


class TestRoundTrip:
    """Round-trip save/load preserves the network exactly."""

    def test_chain_round_trip(self) -> None:
        """3-node chain A→B→C survives save/load."""
        bn = _build_chain_abc()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            loaded = load_model(path)

            assert loaded.nodes == bn.nodes
            assert loaded.edges == bn.edges

            for name in bn.nodes:
                assert loaded.get_states(name) == bn.get_states(name)
                np.testing.assert_allclose(
                    loaded._cpds[name], bn._cpds[name], atol=1e-10
                )
        finally:
            os.unlink(path)

    def test_collider_round_trip(self) -> None:
        """Collider A→C←B survives save/load."""
        bn = _build_collider()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            loaded = load_model(path)

            assert loaded.nodes == bn.nodes
            assert loaded.edges == bn.edges

            for name in bn.nodes:
                np.testing.assert_allclose(
                    loaded._cpds[name], bn._cpds[name], atol=1e-10
                )
        finally:
            os.unlink(path)

    def test_single_root_round_trip(self) -> None:
        """Single root node round-trips correctly."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.2, 0.3, 0.5]), states=["lo", "mid", "hi"])

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            loaded = load_model(path)

            assert loaded.nodes == ["X"]
            assert loaded.get_states("X") == ["lo", "mid", "hi"]
            np.testing.assert_allclose(
                loaded._cpds["X"], [0.2, 0.3, 0.5], atol=1e-10
            )
        finally:
            os.unlink(path)

    def test_inference_after_round_trip(self) -> None:
        """Inference results are identical after round-trip."""
        bn = _build_chain_abc()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            loaded = load_model(path)

            for name in ["A", "B", "C"]:
                np.testing.assert_allclose(
                    loaded.marginal(name),
                    bn.marginal(name),
                    atol=1e-10,
                )
        finally:
            os.unlink(path)

    def test_auto_states_round_trip(self) -> None:
        """Auto-generated state labels survive round-trip."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.5, 0.5]))

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            loaded = load_model(path)
            assert loaded.get_states("X") == ["s0", "s1"]
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  Version migration
# ------------------------------------------------------------------ #


class TestVersioning:
    """Format version handling and migration."""

    def test_version_in_output(self) -> None:
        """Saved file contains format_version field."""
        bn = _build_chain_abc()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            with open(path) as fh:
                data = json.load(fh)
            assert data["format_version"] == FORMAT_VERSION
        finally:
            os.unlink(path)

    def test_future_version_raises(self) -> None:
        """Loading a file with a future version raises ValueError."""
        payload = {
            "format_version": FORMAT_VERSION + 999,
            "nodes": [],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported format version"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_missing_version_raises(self) -> None:
        """Loading a file without format_version raises ValueError."""
        payload = {"nodes": [], "edges": []}
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="format_version"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_current_version_loads(self) -> None:
        """Current version files load without migration errors."""
        bn = _build_chain_abc()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            loaded = load_model(path)
            assert loaded.nodes == bn.nodes
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  Error handling
# ------------------------------------------------------------------ #


class TestErrorHandling:
    """Error handling for corrupted and invalid files."""

    def test_file_not_found(self) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        path = os.path.join(tempfile.gettempdir(), "nonexistent_model_12345.json")
        with pytest.raises(FileNotFoundError):
            load_model(path)

    def test_corrupted_json(self) -> None:
        """Loading a file with invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            f.write("{not valid json!!!")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Corrupted"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_missing_nodes_field(self) -> None:
        """Loading a file without 'nodes' raises ValueError."""
        payload = {"format_version": 1, "edges": []}
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="nodes"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_node_missing_name(self) -> None:
        """Node entry without 'name' raises ValueError."""
        payload = {
            "format_version": 1,
            "nodes": [{"states": ["a", "b"], "distribution": [0.5, 0.5]}],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="name"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_node_missing_distribution(self) -> None:
        """Node entry without distribution raises ValueError."""
        payload = {
            "format_version": 1,
            "nodes": [
                {"name": "X", "states": ["a", "b"], "encoding": "json"}
            ],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="distribution"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_save_non_belief_network_raises(self) -> None:
        """save_model with wrong type raises TypeError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(TypeError, match="BeliefNetwork"):
                save_model("not a network", path)
        finally:
            os.unlink(path)

    def test_path_as_string(self) -> None:
        """save_model / load_model accept string paths."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.5, 0.5]))

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)  # string path
            loaded = load_model(path)  # string path
            assert loaded.nodes == ["X"]
        finally:
            os.unlink(path)

    def test_path_as_pathlib(self) -> None:
        """save_model / load_model accept pathlib.Path."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.5, 0.5]))

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = Path(f.name)
        try:
            save_model(bn, path)
            loaded = load_model(path)
            assert loaded.nodes == ["X"]
        finally:
            path.unlink()


# ------------------------------------------------------------------ #
#  Probability validation
# ------------------------------------------------------------------ #


class TestProbabilityValidation:
    """Probability constraints are enforced on load."""

    def test_negative_probability_rejected(self) -> None:
        """Negative probability values are rejected."""
        payload = {
            "format_version": 1,
            "nodes": [
                {
                    "name": "X",
                    "states": ["a", "b"],
                    "parents": [],
                    "distribution": [-0.5, 1.5],
                    "distribution_shape": [2],
                    "encoding": "json",
                }
            ],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_probability_greater_than_one_rejected(self) -> None:
        """Probability values > 1 are rejected."""
        payload = {
            "format_version": 1,
            "nodes": [
                {
                    "name": "X",
                    "states": ["a", "b"],
                    "parents": [],
                    "distribution": [1.5, 0.5],
                    "distribution_shape": [2],
                    "encoding": "json",
                }
            ],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_prior_not_summing_to_one_rejected(self) -> None:
        """Root prior that does not sum to 1 is rejected."""
        payload = {
            "format_version": 1,
            "nodes": [
                {
                    "name": "X",
                    "states": ["a", "b"],
                    "parents": [],
                    "distribution": [0.3, 0.3],
                    "distribution_shape": [2],
                    "encoding": "json",
                }
            ],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="sum to 1"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_cpt_row_not_summing_to_one_rejected(self) -> None:
        """CPT row that does not sum to 1 is rejected."""
        payload = {
            "format_version": 1,
            "nodes": [
                {
                    "name": "A",
                    "states": ["a0", "a1"],
                    "parents": [],
                    "distribution": [0.5, 0.5],
                    "distribution_shape": [2],
                    "encoding": "json",
                },
                {
                    "name": "B",
                    "states": ["b0", "b1"],
                    "parents": ["A"],
                    "distribution": [[0.9, 0.1], [0.3, 0.3]],
                    "distribution_shape": [2, 2],
                    "encoding": "json",
                },
            ],
            "edges": [{"parent": "A", "child": "B"}],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="CPT rows"):
                load_model(path)
        finally:
            os.unlink(path)

    def test_valid_probabilities_accepted(self) -> None:
        """Valid probabilities pass validation."""
        bn = _build_chain_abc()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            loaded = load_model(path)
            assert loaded.nodes == bn.nodes
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  DAG acyclicity validation
# ------------------------------------------------------------------ #


class TestAcyclicityValidation:
    """DAG acyclicity is enforced on load."""

    def test_valid_dag_accepted(self) -> None:
        """A valid DAG loads without error."""
        bn = _build_chain_abc()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            loaded = load_model(path)
            assert loaded.nodes == bn.nodes
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  Pickle fallback
# ------------------------------------------------------------------ #


class TestPickleFallback:
    """Pickle fallback for complex objects."""

    def test_pickle_round_trip(self) -> None:
        """Callable CPDs are serialized via pickle and restored."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.5, 0.5]), states=["x0", "x1"])

        # Manually inject a non-ndarray CPD to test pickle path
        original_cpd = {"custom": [1, 2, 3]}
        bn._cpds["X"] = original_cpd

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)

            # Verify pickle encoding in JSON
            with open(path) as fh:
                data = json.load(fh)
            node_data = data["nodes"][0]
            assert node_data["encoding"] == "pickle"
            assert "distribution_pickle" in node_data

            loaded = load_model(path)
            assert loaded._cpds["X"] == original_cpd
        finally:
            os.unlink(path)

    def test_pickle_missing_blob_raises(self) -> None:
        """Pickle-encoded node without blob raises ValueError."""
        payload = {
            "format_version": 1,
            "nodes": [
                {
                    "name": "X",
                    "states": ["a", "b"],
                    "parents": [],
                    "encoding": "pickle",
                }
            ],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="distribution_pickle"):
                load_model(path)
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  JSON output structure
# ------------------------------------------------------------------ #


class TestJsonStructure:
    """Verify the JSON output structure."""

    def test_json_has_required_fields(self) -> None:
        """Output JSON contains format_version, nodes, edges."""
        bn = _build_chain_abc()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            with open(path) as fh:
                data = json.load(fh)

            assert "format_version" in data
            assert "nodes" in data
            assert "edges" in data
            assert len(data["nodes"]) == 3
            assert len(data["edges"]) == 2
        finally:
            os.unlink(path)

    def test_node_entry_fields(self) -> None:
        """Each node entry contains name, states, parents, distribution."""
        bn = _build_chain_abc()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            with open(path) as fh:
                data = json.load(fh)

            for node in data["nodes"]:
                assert "name" in node
                assert "states" in node
                assert "parents" in node
                assert "encoding" in node
        finally:
            os.unlink(path)

    def test_edge_entry_fields(self) -> None:
        """Each edge entry contains parent and child."""
        bn = _build_chain_abc()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            save_model(bn, path)
            with open(path) as fh:
                data = json.load(fh)

            for edge in data["edges"]:
                assert "parent" in edge
                assert "child" in edge
        finally:
            os.unlink(path)
