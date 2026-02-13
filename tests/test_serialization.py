"""Tests for probflow/integration/serialization.py.

Covers:
- Round-trip consistency (save then load, compare)
- Version migration (v0.1 -> v1.0)
- Error handling (corrupted files, invalid probabilities, cycles)
- Pickle fallback for callable CPDs
"""

from __future__ import annotations

import json
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
    """Build a 3-node chain A -> B -> C."""
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


# ------------------------------------------------------------------ #
#  Round-trip consistency tests
# ------------------------------------------------------------------ #


class TestRoundTrip:
    """Round-trip save/load tests."""

    def test_round_trip_simple_chain(self, tmp_path: Path) -> None:
        """Save and reload a 3-node chain; structure must match."""
        bn = _build_chain_abc()
        filepath = tmp_path / "model.json"

        save_model(bn, filepath)
        loaded = load_model(filepath)

        assert loaded.nodes == bn.nodes
        assert loaded.edges == bn.edges

    def test_round_trip_preserves_states(self, tmp_path: Path) -> None:
        """State labels survive a round trip."""
        bn = _build_chain_abc()
        filepath = tmp_path / "model.json"

        save_model(bn, filepath)
        loaded = load_model(filepath)

        for name in bn.nodes:
            assert loaded.get_states(name) == bn.get_states(name)

    def test_round_trip_preserves_distributions(self, tmp_path: Path) -> None:
        """Distribution arrays survive a round trip."""
        bn = _build_chain_abc()
        filepath = tmp_path / "model.json"

        save_model(bn, filepath)
        loaded = load_model(filepath)

        for name in bn.nodes:
            np.testing.assert_allclose(
                loaded._cpds[name], bn._cpds[name], atol=1e-10
            )

    def test_round_trip_preserves_marginals(self, tmp_path: Path) -> None:
        """Marginals are the same before and after round trip."""
        bn = _build_chain_abc()
        filepath = tmp_path / "model.json"

        save_model(bn, filepath)
        loaded = load_model(filepath)

        for name in bn.nodes:
            np.testing.assert_allclose(
                loaded.marginal(name),
                bn.marginal(name),
                atol=1e-10,
            )

    def test_round_trip_single_root(self, tmp_path: Path) -> None:
        """A network with a single root node survives round trip."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.3, 0.7]), states=["lo", "hi"])
        filepath = tmp_path / "single.json"

        save_model(bn, filepath)
        loaded = load_model(filepath)

        assert loaded.nodes == ["X"]
        np.testing.assert_allclose(loaded._cpds["X"], [0.3, 0.7])

    def test_round_trip_collider(self, tmp_path: Path) -> None:
        """A collider (A->C<-B) survives round trip."""
        bn = BeliefNetwork()
        bn.add_node("A", np.array([0.5, 0.5]))
        bn.add_node("B", np.array([0.5, 0.5]))
        bn.add_node(
            "C",
            np.array([
                [[0.9, 0.1], [0.6, 0.4]],
                [[0.3, 0.7], [0.2, 0.8]],
            ]),
            parents=["A", "B"],
        )
        filepath = tmp_path / "collider.json"

        save_model(bn, filepath)
        loaded = load_model(filepath)

        assert set(loaded.nodes) == {"A", "B", "C"}
        np.testing.assert_allclose(
            loaded._cpds["C"], bn._cpds["C"], atol=1e-10
        )

    def test_format_version_in_json(self, tmp_path: Path) -> None:
        """The JSON file contains the format_version field."""
        bn = _build_chain_abc()
        filepath = tmp_path / "model.json"

        save_model(bn, filepath)

        with open(filepath) as fh:
            data = json.load(fh)

        assert data["format_version"] == FORMAT_VERSION


# ------------------------------------------------------------------ #
#  Version migration tests
# ------------------------------------------------------------------ #


class TestVersionMigration:
    """Tests for backward-compatible version migration."""

    def test_migrate_v01_adds_distribution_type(
        self, tmp_path: Path
    ) -> None:
        """A v0.1 file (no distribution_type) loads correctly."""
        data = {
            "format_version": "0.1",
            "nodes": [
                {
                    "name": "A",
                    "states": ["a0", "a1"],
                    "parents": [],
                    "distribution": [0.4, 0.6],
                },
                {
                    "name": "B",
                    "states": ["b0", "b1"],
                    "parents": ["A"],
                    "distribution": [[0.9, 0.1], [0.3, 0.7]],
                },
            ],
            "edges": [{"parent": "A", "child": "B"}],
        }

        filepath = tmp_path / "v01.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        loaded = load_model(filepath)
        assert loaded.nodes == ["A", "B"]
        np.testing.assert_allclose(loaded._cpds["A"], [0.4, 0.6])


# ------------------------------------------------------------------ #
#  Error handling tests
# ------------------------------------------------------------------ #


class TestErrorHandling:
    """Tests for error conditions."""

    def test_corrupted_json(self, tmp_path: Path) -> None:
        """A file with invalid JSON raises ValueError."""
        filepath = tmp_path / "bad.json"
        filepath.write_text("{not valid json!!!")

        with pytest.raises(ValueError, match="Corrupted model file"):
            load_model(filepath)

    def test_missing_file(self, tmp_path: Path) -> None:
        """A non-existent file raises FileNotFoundError."""
        filepath = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_model(filepath)

    def test_missing_format_version(self, tmp_path: Path) -> None:
        """A file without format_version raises ValueError."""
        data = {"nodes": [], "edges": []}
        filepath = tmp_path / "no_version.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="format_version"):
            load_model(filepath)

    def test_missing_nodes_key(self, tmp_path: Path) -> None:
        """A file without nodes key raises ValueError."""
        data = {"format_version": "1.0"}
        filepath = tmp_path / "no_nodes.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="nodes"):
            load_model(filepath)

    def test_cycle_detection(self, tmp_path: Path) -> None:
        """A file with a cycle in edges raises ValueError."""
        data = {
            "format_version": "1.0",
            "nodes": [
                {
                    "name": "A",
                    "states": ["a0", "a1"],
                    "parents": [],
                    "distribution": [0.5, 0.5],
                    "distribution_type": "array",
                },
                {
                    "name": "B",
                    "states": ["b0", "b1"],
                    "parents": ["A"],
                    "distribution": [[0.8, 0.2], [0.3, 0.7]],
                    "distribution_type": "array",
                },
            ],
            "edges": [
                {"parent": "A", "child": "B"},
                {"parent": "B", "child": "A"},
            ],
        }

        filepath = tmp_path / "cycle.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="cycle"):
            load_model(filepath)

    def test_negative_probability(self, tmp_path: Path) -> None:
        """Negative probability values raise ValueError."""
        data = {
            "format_version": "1.0",
            "nodes": [
                {
                    "name": "A",
                    "states": ["a0", "a1"],
                    "parents": [],
                    "distribution": [-0.1, 1.1],
                    "distribution_type": "array",
                },
            ],
            "edges": [],
        }

        filepath = tmp_path / "neg_prob.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            load_model(filepath)

    def test_probability_sum_not_one(self, tmp_path: Path) -> None:
        """Prior that does not sum to 1 raises ValueError."""
        data = {
            "format_version": "1.0",
            "nodes": [
                {
                    "name": "A",
                    "states": ["a0", "a1"],
                    "parents": [],
                    "distribution": [0.3, 0.3],
                    "distribution_type": "array",
                },
            ],
            "edges": [],
        }

        filepath = tmp_path / "bad_sum.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="sums to"):
            load_model(filepath)

    def test_cpt_rows_not_sum_to_one(self, tmp_path: Path) -> None:
        """CPT rows that do not sum to 1 raise ValueError."""
        data = {
            "format_version": "1.0",
            "nodes": [
                {
                    "name": "A",
                    "states": ["a0", "a1"],
                    "parents": [],
                    "distribution": [0.5, 0.5],
                    "distribution_type": "array",
                },
                {
                    "name": "B",
                    "states": ["b0", "b1"],
                    "parents": ["A"],
                    "distribution": [[0.9, 0.2], [0.3, 0.7]],
                    "distribution_type": "array",
                },
            ],
            "edges": [{"parent": "A", "child": "B"}],
        }

        filepath = tmp_path / "bad_cpt.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="CPT rows"):
            load_model(filepath)

    def test_save_wrong_type(self) -> None:
        """Passing a non-BeliefNetwork to save_model raises TypeError."""
        with pytest.raises(TypeError, match="Expected BeliefNetwork"):
            save_model("not a network", "/tmp/test.json")  # type: ignore[arg-type]

    def test_missing_node_name(self, tmp_path: Path) -> None:
        """Node without a 'name' field raises ValueError."""
        data = {
            "format_version": "1.0",
            "nodes": [
                {
                    "states": ["a0", "a1"],
                    "parents": [],
                    "distribution": [0.5, 0.5],
                    "distribution_type": "array",
                }
            ],
            "edges": [],
        }
        filepath = tmp_path / "no_name.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="missing 'name'"):
            load_model(filepath)

    def test_missing_distribution(self, tmp_path: Path) -> None:
        """Node without a 'distribution' field raises ValueError."""
        data = {
            "format_version": "1.0",
            "nodes": [
                {
                    "name": "A",
                    "states": ["a0", "a1"],
                    "parents": [],
                    "distribution_type": "array",
                }
            ],
            "edges": [],
        }
        filepath = tmp_path / "no_dist.json"
        with open(filepath, "w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="missing 'distribution'"):
            load_model(filepath)


# ------------------------------------------------------------------ #
#  Pickle fallback tests
# ------------------------------------------------------------------ #


def _sample_callable_cpd(parent_state: int) -> np.ndarray:
    """Module-level callable CPD for pickle testing."""
    if parent_state == 0:
        return np.array([0.8, 0.2])
    return np.array([0.3, 0.7])


class TestPickleFallback:
    """Tests for pickle fallback with callable CPDs."""

    def test_callable_cpd_round_trip(self, tmp_path: Path) -> None:
        """A network with a callable CPD uses pickle and round-trips."""
        bn = BeliefNetwork()
        bn.add_node("A", np.array([0.5, 0.5]), states=["a0", "a1"])
        bn.add_node(
            "B",
            np.array([[0.8, 0.2], [0.3, 0.7]]),
            parents=["A"],
            states=["b0", "b1"],
        )

        # Inject a module-level callable CPD for pickle fallback
        bn._cpds["B"] = _sample_callable_cpd  # type: ignore[assignment]

        filepath = tmp_path / "callable.json"
        save_model(bn, filepath)

        # Verify pickle type in JSON
        with open(filepath) as fh:
            data = json.load(fh)
        b_node = [n for n in data["nodes"] if n["name"] == "B"][0]
        assert b_node["distribution_type"] == "pickle"

        loaded = load_model(filepath)
        # The callable is restored
        restored_cpd = loaded._cpds["B"]
        assert callable(restored_cpd)
        np.testing.assert_allclose(restored_cpd(0), [0.8, 0.2])
        np.testing.assert_allclose(restored_cpd(1), [0.3, 0.7])
