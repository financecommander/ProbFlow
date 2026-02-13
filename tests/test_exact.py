"""Tests for probflow/inference/exact.py.

Covers:
- FactorTable: multiply, marginalize, reduce, normalize
- Belief propagation vs brute-force enumeration on polytrees
- Variable elimination on arbitrary DAGs
- Performance (<1ms for 10-node tree)
- Numerical stability
- Loop detection
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from probflow.core.types import Node, Variable
from probflow.inference.exact import (
    FactorTable,
    _has_undirected_cycle,
    belief_propagation,
    variable_elimination,
)
from probflow.networks.graph import build_chain, build_tree


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _brute_force_marginals(
    root: Node,
    nodes: list[Node],
    evidence: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Compute exact marginals by enumerating all joint configurations."""
    evidence = evidence or {}
    n = len(nodes)
    name_to_idx = {nd.name: i for i, nd in enumerate(nodes)}
    cards = [nd.variable.num_states for nd in nodes]

    marginals = {nd.name: np.zeros(nd.variable.num_states) for nd in nodes}

    # Enumerate all combinations
    total_configs = 1
    for c in cards:
        total_configs *= c

    for config_id in range(total_configs):
        assignment = []
        tmp = config_id
        for c in cards:
            assignment.append(tmp % c)
            tmp //= c

        # Check evidence
        skip = False
        for var_name, state_idx in evidence.items():
            idx = name_to_idx[var_name]
            if assignment[idx] != state_idx:
                skip = True
                break
        if skip:
            continue

        # Compute joint probability P(X1=x1, ..., Xn=xn)
        prob = 1.0
        for i, nd in enumerate(nodes):
            if not nd.parents:
                prior = nd.prior if nd.prior is not None else (
                    np.ones(nd.variable.num_states) / nd.variable.num_states
                )
                prob *= prior[assignment[i]]
            else:
                parent = nd.parents[0]
                parent_idx = name_to_idx[parent.name]
                prob *= nd.cpt[assignment[parent_idx], assignment[i]]

        for i, nd in enumerate(nodes):
            marginals[nd.name][assignment[i]] += prob

    # Normalise
    for name in marginals:
        total = marginals[name].sum()
        if total > 0:
            marginals[name] /= total

    return marginals


# ------------------------------------------------------------------ #
#  FactorTable tests
# ------------------------------------------------------------------ #

class TestFactorTable:
    """Tests for the FactorTable class."""

    def test_create_factor(self):
        """FactorTable stores variables, cardinalities, and values."""
        f = FactorTable(["A"], [3], np.array([0.2, 0.5, 0.3]))
        assert f.variables == ["A"]
        assert f.cardinalities == [3]
        assert f.values.shape == (3,)

    def test_create_factor_shape_mismatch(self):
        """ValueError when shape doesn't match cardinalities."""
        with pytest.raises(ValueError, match="does not match"):
            FactorTable(["A", "B"], [2, 3], np.ones((2, 2)))

    def test_multiply_shared_variable(self):
        """Multiply two factors with a shared variable."""
        f1 = FactorTable(["A", "B"], [2, 2], np.array([[0.3, 0.7],
                                                         [0.6, 0.4]]))
        f2 = FactorTable(["B", "C"], [2, 2], np.array([[0.1, 0.9],
                                                         [0.8, 0.2]]))
        product = f1.multiply(f2)
        assert set(product.variables) == {"A", "B", "C"}
        assert product.values.shape == (2, 2, 2)

    def test_multiply_no_shared(self):
        """Multiply two factors with disjoint variables (outer product)."""
        f1 = FactorTable(["A"], [2], np.array([0.4, 0.6]))
        f2 = FactorTable(["B"], [3], np.array([0.1, 0.2, 0.7]))
        product = f1.multiply(f2)
        assert set(product.variables) == {"A", "B"}
        np.testing.assert_allclose(
            product.values.sum(),
            f1.values.sum() * f2.values.sum(),
            atol=1e-10,
        )

    def test_marginalize(self):
        """Marginalize a variable out of a factor."""
        vals = np.array([[0.3, 0.7], [0.6, 0.4]])
        f = FactorTable(["A", "B"], [2, 2], vals)
        m = f.marginalize("B")
        assert m.variables == ["A"]
        np.testing.assert_allclose(m.values, [1.0, 1.0])

    def test_marginalize_missing_variable(self):
        """ValueError when marginalising a variable not in the factor."""
        f = FactorTable(["A"], [2], np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="not in factor"):
            f.marginalize("Z")

    def test_reduce(self):
        """Reduce (condition) on a specific state."""
        vals = np.array([[0.3, 0.7], [0.6, 0.4]])
        f = FactorTable(["A", "B"], [2, 2], vals)
        r = f.reduce("A", 1)
        assert r.variables == ["B"]
        np.testing.assert_allclose(r.values, [0.6, 0.4])

    def test_reduce_missing_variable(self):
        """ValueError when reducing a variable not in the factor."""
        f = FactorTable(["A"], [2], np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="not in factor"):
            f.reduce("Z", 0)

    def test_normalize(self):
        """Normalise a factor so entries sum to 1."""
        f = FactorTable(["A"], [3], np.array([2.0, 3.0, 5.0]))
        n = f.normalize()
        np.testing.assert_allclose(n.values, [0.2, 0.3, 0.5])
        assert abs(n.values.sum() - 1.0) < 1e-10

    def test_normalize_zero(self):
        """Normalising an all-zero factor returns all zeros."""
        f = FactorTable(["A"], [2], np.array([0.0, 0.0]))
        n = f.normalize()
        np.testing.assert_allclose(n.values, [0.0, 0.0])

    def test_from_node_root(self):
        """FactorTable.from_node for a root node uses its prior."""
        var = Variable("X", ["a", "b"])
        node = Node(variable=var, prior=np.array([0.3, 0.7]))
        f = FactorTable.from_node(node)
        assert f.variables == ["X"]
        np.testing.assert_allclose(f.values, [0.3, 0.7])

    def test_from_node_child(self):
        """FactorTable.from_node for a child node uses its CPT."""
        parent_var = Variable("P", ["a", "b"])
        child_var = Variable("C", ["x", "y"])
        parent = Node(variable=parent_var, prior=np.array([0.5, 0.5]))
        child = Node(variable=child_var)
        cpt = np.array([[0.9, 0.1], [0.2, 0.8]])
        parent.add_child(child, cpt)
        f = FactorTable.from_node(child)
        assert f.variables == ["P", "C"]
        np.testing.assert_allclose(f.values, cpt)

    def test_multiply_commutative(self):
        """Multiplication result should give same total regardless of order."""
        f1 = FactorTable(["A", "B"], [2, 2],
                         np.array([[0.3, 0.7], [0.6, 0.4]]))
        f2 = FactorTable(["B"], [2], np.array([0.4, 0.6]))
        p1 = f1.multiply(f2)
        p2 = f2.multiply(f1)
        np.testing.assert_allclose(p1.values.sum(), p2.values.sum(), atol=1e-10)


# ------------------------------------------------------------------ #
#  Loop detection
# ------------------------------------------------------------------ #

class TestLoopDetection:
    """Tests for cycle detection in the undirected skeleton."""

    def test_tree_has_no_cycle(self):
        """A balanced tree should not be flagged as cyclic."""
        _, nodes = build_tree(7, num_states=2, seed=0)
        assert not _has_undirected_cycle(nodes)

    def test_chain_has_no_cycle(self):
        """A chain should not be flagged as cyclic."""
        _, nodes = build_chain(5, num_states=2, seed=0)
        assert not _has_undirected_cycle(nodes)

    def test_cycle_detected(self):
        """A graph with an extra edge forming a cycle should be detected."""
        _, nodes = build_chain(3, num_states=2, seed=0)
        # Add edge from node 2 back to node 0 → creates a cycle
        cpt = np.array([[0.5, 0.5], [0.5, 0.5]])
        nodes[2].add_child(nodes[0], cpt)
        assert _has_undirected_cycle(nodes)

    def test_bp_raises_on_loop(self):
        """belief_propagation raises ValueError on cyclic graphs."""
        _, nodes = build_chain(3, num_states=2, seed=0)
        cpt = np.array([[0.5, 0.5], [0.5, 0.5]])
        nodes[2].add_child(nodes[0], cpt)
        with pytest.raises(ValueError, match="cycles"):
            belief_propagation(nodes[0], nodes)


# ------------------------------------------------------------------ #
#  Belief propagation tests
# ------------------------------------------------------------------ #

class TestBeliefPropagation:
    """Tests for belief propagation on tree-structured networks."""

    def test_single_node(self):
        """BP on a single node returns its prior."""
        var = Variable("X", ["a", "b"])
        node = Node(variable=var, prior=np.array([0.3, 0.7]))
        result = belief_propagation(node, [node])
        np.testing.assert_allclose(result["X"], [0.3, 0.7], atol=1e-10)

    def test_two_nodes_marginals(self):
        """BP on a two-node tree matches brute-force computation."""
        parent_var = Variable("P", ["a", "b"])
        child_var = Variable("C", ["x", "y"])
        parent = Node(variable=parent_var, prior=np.array([0.4, 0.6]))
        child = Node(variable=child_var)
        cpt = np.array([[0.9, 0.1], [0.3, 0.7]])
        parent.add_child(child, cpt)

        bp = belief_propagation(parent, [parent, child])
        bf = _brute_force_marginals(parent, [parent, child])

        for name in bp:
            np.testing.assert_allclose(bp[name], bf[name], atol=1e-10)

    def test_chain_vs_brute_force(self):
        """BP on a chain matches brute-force enumeration."""
        root, nodes = build_chain(5, num_states=2, seed=42)
        bp = belief_propagation(root, nodes)
        bf = _brute_force_marginals(root, nodes)

        for name in bp:
            np.testing.assert_allclose(bp[name], bf[name], atol=1e-6)

    def test_tree_vs_brute_force(self):
        """BP on a balanced tree matches brute-force enumeration."""
        root, nodes = build_tree(7, num_states=2, seed=42)
        bp = belief_propagation(root, nodes)
        bf = _brute_force_marginals(root, nodes)

        for name in bp:
            np.testing.assert_allclose(bp[name], bf[name], atol=1e-6)

    def test_query_single_variable(self):
        """Query for a single variable returns only that variable."""
        root, nodes = build_chain(4, num_states=2, seed=10)
        bp = belief_propagation(root, nodes, query="X2")
        assert list(bp.keys()) == ["X2"]
        assert abs(bp["X2"].sum() - 1.0) < 1e-10

    def test_with_evidence(self):
        """BP with evidence conditions correctly."""
        parent_var = Variable("P", ["a", "b"])
        child_var = Variable("C", ["x", "y"])
        parent = Node(variable=parent_var, prior=np.array([0.4, 0.6]))
        child = Node(variable=child_var)
        cpt = np.array([[0.9, 0.1], [0.3, 0.7]])
        parent.add_child(child, cpt)

        # Evidence: P = 'a' (index 0)
        bp = belief_propagation(parent, [parent, child], evidence={"P": 0})
        # P should be [1, 0], C should be [0.9, 0.1]
        np.testing.assert_allclose(bp["P"], [1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(bp["C"], [0.9, 0.1], atol=1e-6)

    def test_marginals_sum_to_one(self):
        """All marginals should sum to 1."""
        root, nodes = build_tree(10, num_states=2, seed=99)
        bp = belief_propagation(root, nodes)
        for name, marginal in bp.items():
            assert abs(marginal.sum() - 1.0) < 1e-10, (
                f"{name} marginal sums to {marginal.sum()}"
            )

    def test_three_state_variables(self):
        """BP works with more than two states."""
        root, nodes = build_chain(4, num_states=3, seed=55)
        bp = belief_propagation(root, nodes)
        bf = _brute_force_marginals(root, nodes)

        for name in bp:
            np.testing.assert_allclose(bp[name], bf[name], atol=1e-6)


# ------------------------------------------------------------------ #
#  Variable elimination tests
# ------------------------------------------------------------------ #

class TestVariableElimination:
    """Tests for variable elimination on arbitrary DAGs."""

    def test_single_node(self):
        """VE on a single node returns its prior."""
        var = Variable("X", ["a", "b"])
        node = Node(variable=var, prior=np.array([0.3, 0.7]))
        result = variable_elimination(node, [node], query="X")
        np.testing.assert_allclose(result["X"], [0.3, 0.7], atol=1e-10)

    def test_two_nodes(self):
        """VE on a two-node DAG matches brute-force."""
        parent_var = Variable("P", ["a", "b"])
        child_var = Variable("C", ["x", "y"])
        parent = Node(variable=parent_var, prior=np.array([0.4, 0.6]))
        child = Node(variable=child_var)
        cpt = np.array([[0.9, 0.1], [0.3, 0.7]])
        parent.add_child(child, cpt)

        ve = variable_elimination(parent, [parent, child], query="C")
        bf = _brute_force_marginals(parent, [parent, child])
        np.testing.assert_allclose(ve["C"], bf["C"], atol=1e-10)

    def test_chain(self):
        """VE on a chain matches brute-force."""
        root, nodes = build_chain(5, num_states=2, seed=42)

        for node in nodes:
            ve = variable_elimination(root, nodes, query=node.name)
            bf = _brute_force_marginals(root, nodes)
            np.testing.assert_allclose(
                ve[node.name], bf[node.name], atol=1e-6
            )

    def test_tree(self):
        """VE on a tree matches brute-force."""
        root, nodes = build_tree(7, num_states=2, seed=42)

        for node in nodes:
            ve = variable_elimination(root, nodes, query=node.name)
            bf = _brute_force_marginals(root, nodes)
            np.testing.assert_allclose(
                ve[node.name], bf[node.name], atol=1e-6,
            )

    def test_with_evidence(self):
        """VE with evidence conditions correctly."""
        parent_var = Variable("P", ["a", "b"])
        child_var = Variable("C", ["x", "y"])
        parent = Node(variable=parent_var, prior=np.array([0.4, 0.6]))
        child = Node(variable=child_var)
        cpt = np.array([[0.9, 0.1], [0.3, 0.7]])
        parent.add_child(child, cpt)

        ve = variable_elimination(
            parent, [parent, child], query="C", evidence={"P": 0}
        )
        # Given P=a, C should be [0.9, 0.1]
        np.testing.assert_allclose(ve["C"], [0.9, 0.1], atol=1e-10)

    def test_cancer_bayes(self):
        """VE reproduces classic Bayes' theorem cancer screening result."""
        p_cancer = 0.01
        p_pos_given_cancer = 0.90
        p_pos_given_no_cancer = 0.05
        p_pos = (p_cancer * p_pos_given_cancer
                 + (1 - p_cancer) * p_pos_given_no_cancer)
        p_cancer_given_pos = p_cancer * p_pos_given_cancer / p_pos

        cancer_var = Variable("Cancer", ["no", "yes"])
        test_var = Variable("Test", ["negative", "positive"])
        cancer_node = Node(
            variable=cancer_var,
            prior=np.array([1 - p_cancer, p_cancer]),
        )
        test_node = Node(variable=test_var)
        cpt = np.array([
            [1 - p_pos_given_no_cancer, p_pos_given_no_cancer],
            [1 - p_pos_given_cancer, p_pos_given_cancer],
        ])
        cancer_node.add_child(test_node, cpt)

        ve = variable_elimination(
            cancer_node, [cancer_node, test_node],
            query="Cancer", evidence={"Test": 1},
        )

        np.testing.assert_allclose(
            ve["Cancer"][1], p_cancer_given_pos, atol=1e-10
        )

    def test_ve_agrees_with_bp_on_tree(self):
        """On a tree, VE and BP should produce the same marginals."""
        root, nodes = build_tree(7, num_states=2, seed=42)
        bp = belief_propagation(root, nodes)

        for node in nodes:
            ve = variable_elimination(root, nodes, query=node.name)
            np.testing.assert_allclose(
                ve[node.name], bp[node.name], atol=1e-6
            )

    def test_three_state_variables(self):
        """VE works with more than two states."""
        root, nodes = build_chain(4, num_states=3, seed=55)
        bf = _brute_force_marginals(root, nodes)
        for node in nodes:
            ve = variable_elimination(root, nodes, query=node.name)
            np.testing.assert_allclose(
                ve[node.name], bf[node.name], atol=1e-6
            )


# ------------------------------------------------------------------ #
#  Performance tests
# ------------------------------------------------------------------ #

class TestPerformance:
    """Performance tests for exact inference."""

    def test_bp_10_node_tree_under_1ms(self):
        """Belief propagation on a 10-node tree should take <1ms."""
        root, nodes = build_tree(10, num_states=2, seed=42)

        # Warm up
        belief_propagation(root, nodes)

        # Time multiple runs
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            belief_propagation(root, nodes)
        elapsed = (time.perf_counter() - start) / n_runs

        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 1.0, (
            f"BP on 10-node tree took {elapsed_ms:.3f}ms (limit: 1ms)"
        )

    def test_ve_10_node_tree_under_5ms(self):
        """Variable elimination on a 10-node tree should be fast."""
        root, nodes = build_tree(10, num_states=2, seed=42)

        # Warm up
        variable_elimination(root, nodes, query="X5")

        n_runs = 50
        start = time.perf_counter()
        for _ in range(n_runs):
            variable_elimination(root, nodes, query="X5")
        elapsed = (time.perf_counter() - start) / n_runs

        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 5.0, (
            f"VE on 10-node tree took {elapsed_ms:.3f}ms (limit: 5ms)"
        )


# ------------------------------------------------------------------ #
#  Numerical stability tests
# ------------------------------------------------------------------ #

class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_very_small_probabilities(self):
        """BP handles very small prior probabilities without NaN."""
        var_a = Variable("A", ["a0", "a1"])
        var_b = Variable("B", ["b0", "b1"])
        node_a = Node(variable=var_a, prior=np.array([1e-15, 1 - 1e-15]))
        node_b = Node(variable=var_b)
        cpt = np.array([[0.9, 0.1], [0.2, 0.8]])
        node_a.add_child(node_b, cpt)

        bp = belief_propagation(node_a, [node_a, node_b])
        for name, marginal in bp.items():
            assert not np.any(np.isnan(marginal)), f"NaN in {name}"
            assert abs(marginal.sum() - 1.0) < 1e-6

    def test_deterministic_cpt(self):
        """BP handles deterministic (0/1) CPTs correctly."""
        var_a = Variable("A", ["a0", "a1"])
        var_b = Variable("B", ["b0", "b1"])
        node_a = Node(variable=var_a, prior=np.array([0.5, 0.5]))
        node_b = Node(variable=var_b)
        # Deterministic: if A=a0 then B=b0; if A=a1 then B=b1
        cpt = np.array([[1.0, 0.0], [0.0, 1.0]])
        node_a.add_child(node_b, cpt)

        bp = belief_propagation(node_a, [node_a, node_b])
        np.testing.assert_allclose(bp["A"], [0.5, 0.5], atol=1e-10)
        np.testing.assert_allclose(bp["B"], [0.5, 0.5], atol=1e-10)

    def test_uniform_prior(self):
        """BP with uniform prior produces correct results."""
        root, nodes = build_tree(7, num_states=2, seed=42)
        # Override root prior to uniform
        nodes[0].prior = np.array([0.5, 0.5])
        bp = belief_propagation(root, nodes)
        bf = _brute_force_marginals(root, nodes)

        for name in bp:
            np.testing.assert_allclose(bp[name], bf[name], atol=1e-6)

    def test_ve_very_small_probabilities(self):
        """VE handles very small probabilities without NaN."""
        var_a = Variable("A", ["a0", "a1"])
        var_b = Variable("B", ["b0", "b1"])
        node_a = Node(variable=var_a, prior=np.array([1e-15, 1 - 1e-15]))
        node_b = Node(variable=var_b)
        cpt = np.array([[0.9, 0.1], [0.2, 0.8]])
        node_a.add_child(node_b, cpt)

        ve = variable_elimination(node_a, [node_a, node_b], query="B")
        assert not np.any(np.isnan(ve["B"])), "NaN in VE result"
        assert abs(ve["B"].sum() - 1.0) < 1e-6

    def test_nearly_deterministic_evidence(self):
        """VE with evidence on deterministic CPT produces valid results."""
        var_a = Variable("A", ["a0", "a1"])
        var_b = Variable("B", ["b0", "b1"])
        node_a = Node(variable=var_a, prior=np.array([0.5, 0.5]))
        node_b = Node(variable=var_b)
        cpt = np.array([[1.0, 0.0], [0.0, 1.0]])
        node_a.add_child(node_b, cpt)

        # Observe B=b0 → A must be a0
        ve = variable_elimination(
            node_a, [node_a, node_b], query="A", evidence={"B": 0}
        )
        np.testing.assert_allclose(ve["A"], [1.0, 0.0], atol=1e-10)
