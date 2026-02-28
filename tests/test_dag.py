"""Tests for probflow/networks/dag.py.

Covers:
- BeliefNetwork construction with add_node
- Evidence observation with observe / clear_evidence
- Unconditional marginal via variable_elimination
- Conditional inference via infer (belief propagation / fallback)
- D-separation on 3-node chain A→B→C
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

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


# ------------------------------------------------------------------ #
#  Construction tests
# ------------------------------------------------------------------ #

class TestBeliefNetworkConstruction:
    """Tests for graph construction."""

    def test_add_root_node(self) -> None:
        """Adding a root node stores its prior."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.3, 0.7]))
        assert "X" in bn.nodes

    def test_add_child_node(self) -> None:
        """Adding a child node creates an edge."""
        bn = BeliefNetwork()
        bn.add_node("A", np.array([0.5, 0.5]))
        bn.add_node("B", np.array([[0.8, 0.2], [0.1, 0.9]]), parents=["A"])
        assert ("A", "B") in bn.edges

    def test_duplicate_node_raises(self) -> None:
        """Adding a node with a duplicate name raises ValueError."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="already exists"):
            bn.add_node("X", np.array([0.3, 0.7]))

    def test_missing_parent_raises(self) -> None:
        """Referencing a non-existent parent raises ValueError."""
        bn = BeliefNetwork()
        with pytest.raises(ValueError, match="must be added before"):
            bn.add_node(
                "B",
                np.array([[0.5, 0.5], [0.5, 0.5]]),
                parents=["A"],
            )

    def test_auto_states(self) -> None:
        """States are auto-generated when not specified."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.2, 0.3, 0.5]))
        assert bn.get_states("X") == ["s0", "s1", "s2"]

    def test_custom_states(self) -> None:
        """Custom state labels are stored correctly."""
        bn = BeliefNetwork()
        bn.add_node("X", np.array([0.5, 0.5]), states=["low", "high"])
        assert bn.get_states("X") == ["low", "high"]

    def test_states_count_mismatch_raises(self) -> None:
        """Mismatched state count raises ValueError."""
        bn = BeliefNetwork()
        with pytest.raises(ValueError, match="does not match"):
            bn.add_node("X", np.array([0.5, 0.5]), states=["a", "b", "c"])

    def test_three_node_chain_structure(self) -> None:
        """3-node chain A→B→C has correct structure."""
        bn = _build_chain_abc()
        assert bn.nodes == ["A", "B", "C"]
        assert ("A", "B") in bn.edges
        assert ("B", "C") in bn.edges
        assert ("A", "C") not in bn.edges

    def test_repr(self) -> None:
        """repr includes nodes and edges."""
        bn = _build_chain_abc()
        r = repr(bn)
        assert "A" in r
        assert "B" in r
        assert "C" in r


# ------------------------------------------------------------------ #
#  Evidence tests
# ------------------------------------------------------------------ #

class TestEvidence:
    """Tests for observation and evidence management."""

    def test_observe_by_index(self) -> None:
        """observe() accepts integer state index."""
        bn = _build_chain_abc()
        bn.observe("A", 0)
        assert bn.evidence == {"A": 0}

    def test_observe_by_label(self) -> None:
        """observe() accepts string state label."""
        bn = _build_chain_abc()
        bn.observe("A", "a1")
        assert bn.evidence == {"A": 1}

    def test_observe_invalid_variable(self) -> None:
        """Observing a non-existent variable raises ValueError."""
        bn = _build_chain_abc()
        with pytest.raises(ValueError, match="not in network"):
            bn.observe("Z", 0)

    def test_observe_invalid_label(self) -> None:
        """Observing an invalid state label raises ValueError."""
        bn = _build_chain_abc()
        with pytest.raises(ValueError, match="not a valid state"):
            bn.observe("A", "nonexistent")

    def test_clear_evidence(self) -> None:
        """clear_evidence() removes all observations."""
        bn = _build_chain_abc()
        bn.observe("A", 0)
        bn.observe("B", 1)
        bn.clear_evidence()
        assert bn.evidence == {}


# ------------------------------------------------------------------ #
#  Marginal tests
# ------------------------------------------------------------------ #

class TestMarginal:
    """Tests for unconditional marginal computation."""

    def test_root_marginal(self) -> None:
        """Marginal of root node equals its prior."""
        bn = _build_chain_abc()
        m = bn.marginal("A")
        np.testing.assert_allclose(m, [0.4, 0.6], atol=1e-10)

    def test_child_marginal(self) -> None:
        """Marginal of B = sum_A P(B|A) * P(A)."""
        bn = _build_chain_abc()
        m = bn.marginal("B")
        # P(B=b0) = P(B=b0|A=a0)*P(A=a0) + P(B=b0|A=a1)*P(A=a1)
        #         = 0.9*0.4 + 0.3*0.6 = 0.36 + 0.18 = 0.54
        expected_b0 = 0.9 * 0.4 + 0.3 * 0.6
        expected_b1 = 0.1 * 0.4 + 0.7 * 0.6
        np.testing.assert_allclose(m, [expected_b0, expected_b1], atol=1e-10)

    def test_grandchild_marginal(self) -> None:
        """Marginal of C computed through the full chain."""
        bn = _build_chain_abc()
        m = bn.marginal("C")
        # P(B=b0) = 0.54, P(B=b1) = 0.46
        p_b0 = 0.54
        p_b1 = 0.46
        expected_c0 = 0.8 * p_b0 + 0.4 * p_b1
        expected_c1 = 0.2 * p_b0 + 0.6 * p_b1
        np.testing.assert_allclose(m, [expected_c0, expected_c1], atol=1e-10)

    def test_marginal_sums_to_one(self) -> None:
        """Marginals must sum to 1."""
        bn = _build_chain_abc()
        for name in ["A", "B", "C"]:
            m = bn.marginal(name)
            assert abs(m.sum() - 1.0) < 1e-10

    def test_marginal_nonexistent_variable(self) -> None:
        """Querying a non-existent variable raises ValueError."""
        bn = _build_chain_abc()
        with pytest.raises(ValueError, match="not in network"):
            bn.marginal("Z")


# ------------------------------------------------------------------ #
#  Conditional inference tests
# ------------------------------------------------------------------ #

class TestInfer:
    """Tests for conditional inference via infer()."""

    def test_infer_no_evidence(self) -> None:
        """infer() with no evidence equals marginal()."""
        bn = _build_chain_abc()
        m = bn.marginal("C")
        i = bn.infer("C")
        np.testing.assert_allclose(i, m, atol=1e-10)

    def test_infer_with_evidence_on_root(self) -> None:
        """P(C | A=a0) computed correctly."""
        bn = _build_chain_abc()
        bn.observe("A", "a0")
        result = bn.infer("C")
        # P(B=b0|A=a0) = 0.9, P(B=b1|A=a0) = 0.1
        # P(C=c0|A=a0) = 0.8*0.9 + 0.4*0.1 = 0.72 + 0.04 = 0.76
        expected_c0 = 0.8 * 0.9 + 0.4 * 0.1
        expected_c1 = 0.2 * 0.9 + 0.6 * 0.1
        np.testing.assert_allclose(
            result, [expected_c0, expected_c1], atol=1e-6
        )

    def test_infer_with_evidence_on_leaf(self) -> None:
        """P(A | C=c0) computed correctly via Bayes' theorem."""
        bn = _build_chain_abc()
        bn.observe("C", "c0")
        result = bn.infer("A")
        # P(C=c0|A=a0) = 0.8*0.9 + 0.4*0.1 = 0.76
        # P(C=c0|A=a1) = 0.8*0.3 + 0.4*0.7 = 0.52
        # P(A=a0|C=c0) = 0.4*0.76 / (0.4*0.76 + 0.6*0.52)
        p_c0_a0 = 0.76
        p_c0_a1 = 0.52
        p_a0_c0 = 0.4 * p_c0_a0 / (0.4 * p_c0_a0 + 0.6 * p_c0_a1)
        p_a1_c0 = 1.0 - p_a0_c0
        np.testing.assert_allclose(
            result, [p_a0_c0, p_a1_c0], atol=1e-6
        )

    def test_infer_deterministic(self) -> None:
        """Deterministic CPT: observing child pins parent."""
        bn = BeliefNetwork()
        bn.add_node("A", np.array([0.5, 0.5]), states=["a0", "a1"])
        bn.add_node(
            "B",
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            parents=["A"],
            states=["b0", "b1"],
        )
        bn.observe("B", "b0")
        result = bn.infer("A")
        np.testing.assert_allclose(result, [1.0, 0.0], atol=1e-10)

    def test_infer_nonexistent_variable(self) -> None:
        """Querying a non-existent variable raises ValueError."""
        bn = _build_chain_abc()
        with pytest.raises(ValueError, match="not in network"):
            bn.infer("Z")


# ------------------------------------------------------------------ #
#  D-separation tests (3-node chain A→B→C)
# ------------------------------------------------------------------ #

class TestDSeparation:
    """D-separation tests on the 3-node chain A→B→C."""

    def test_a_not_dsep_c_unconditional(self) -> None:
        """A and C are NOT d-separated unconditionally (connected via B)."""
        bn = _build_chain_abc()
        assert not bn.d_separated("A", "C")

    def test_a_dsep_c_given_b(self) -> None:
        """A ⊥ C | B in the chain A→B→C."""
        bn = _build_chain_abc()
        assert bn.d_separated("A", "C", {"B"})

    def test_a_not_dsep_b_unconditional(self) -> None:
        """A and B are NOT d-separated unconditionally."""
        bn = _build_chain_abc()
        assert not bn.d_separated("A", "B")

    def test_a_not_dsep_b_given_c(self) -> None:
        """A and B are NOT d-separated given C (B is not blocked)."""
        bn = _build_chain_abc()
        assert not bn.d_separated("A", "B", {"C"})

    def test_b_not_dsep_c_unconditional(self) -> None:
        """B and C are NOT d-separated unconditionally."""
        bn = _build_chain_abc()
        assert not bn.d_separated("B", "C")

    def test_collider_dsep(self) -> None:
        """In a collider A→C←B, A ⊥ B unconditionally."""
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
        assert bn.d_separated("A", "B")

    def test_collider_not_dsep_given_c(self) -> None:
        """In a collider A→C←B, A and B are NOT d-separated given C."""
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
        assert not bn.d_separated("A", "B", {"C"})

    def test_dsep_invalid_variable(self) -> None:
        """d_separated with a non-existent variable raises ValueError."""
        bn = _build_chain_abc()
        with pytest.raises(ValueError, match="not in network"):
            bn.d_separated("A", "Z")

    def test_dsep_invalid_conditioning_variable(self) -> None:
        """d_separated with invalid conditioning var raises ValueError."""
        bn = _build_chain_abc()
        with pytest.raises(ValueError, match="not in network"):
            bn.d_separated("A", "C", {"Z"})
