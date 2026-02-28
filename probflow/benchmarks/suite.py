"""ProbFlow benchmark suite using pytest-benchmark.

Run:
    pytest probflow/benchmarks/suite.py --benchmark-only --benchmark-autosave

CI integration:
    pytest probflow/benchmarks/suite.py --benchmark-only --benchmark-autosave \
        --benchmark-compare --benchmark-compare-fail=mean:10%
"""

from __future__ import annotations

import tracemalloc
from typing import Dict

import numpy as np
import pytest

from probflow.core.types import Node, Variable
from probflow.inference.belief_propagation import belief_propagation
from probflow.inference.monte_carlo import forward_sample, monte_carlo_marginals
from probflow.networks.graph import build_chain, build_tree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tree_10():
    """10-node tree-structured Bayesian network."""
    return build_tree(num_nodes=10, num_states=2, seed=42)


@pytest.fixture
def tree_100():
    """100-node tree-structured Bayesian network."""
    return build_tree(num_nodes=100, num_states=2, seed=42)


# ---------------------------------------------------------------------------
# Benchmark: Belief Propagation Speed
# ---------------------------------------------------------------------------

def test_belief_propagation_speed(benchmark, tree_10):
    """Belief propagation on a 10-node tree must complete in <1ms."""
    root, nodes = tree_10

    result = benchmark(belief_propagation, root, nodes)

    # Verify output structure
    assert len(result) == 10
    for name, marginal in result.items():
        assert marginal.shape == (2,)
        assert abs(marginal.sum() - 1.0) < 1e-6

    # Performance assertion: median < 1ms
    median_ms = benchmark.stats.stats.median * 1000
    assert median_ms < 1.0, (
        f"Belief propagation took {median_ms:.3f}ms (limit: 1ms)"
    )


# ---------------------------------------------------------------------------
# Benchmark: Monte Carlo Throughput
# ---------------------------------------------------------------------------

def test_monte_carlo_throughput(benchmark, tree_10):
    """10K Monte Carlo samples must complete in <50ms."""
    root, nodes = tree_10

    result = benchmark(forward_sample, root, nodes, 10_000, seed=42)

    # Verify output structure
    assert result.shape == (10_000, 10)
    assert result.dtype == np.int32

    # Performance assertion: median < 50ms
    median_ms = benchmark.stats.stats.median * 1000
    assert median_ms < 50.0, (
        f"Monte Carlo sampling took {median_ms:.3f}ms (limit: 50ms)"
    )


# ---------------------------------------------------------------------------
# Benchmark: Memory Scaling
# ---------------------------------------------------------------------------

def test_memory_scaling(tree_100):
    """100-node network inference must use <10MB of memory."""
    root, nodes = tree_100

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # Run belief propagation on a 100-node network
    marginals = belief_propagation(root, nodes)

    # Also run Monte Carlo sampling
    samples = forward_sample(root, nodes, 10_000, seed=42)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Compute memory difference
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    total_mb = total_bytes / (1024 * 1024)

    assert total_mb < 10.0, (
        f"Memory usage: {total_mb:.2f}MB (limit: 10MB)"
    )

    # Verify correctness
    assert len(marginals) == 100
    assert samples.shape == (10_000, 100)


# ---------------------------------------------------------------------------
# Benchmark: Inference Accuracy
# ---------------------------------------------------------------------------

class TestInferenceAccuracy:
    """Validate inference results against known analytical solutions."""

    def test_dice_sum_distribution(self):
        """Verify the distribution of the sum of two dice.

        Two fair 6-sided dice: P(sum=k) has a known triangular distribution.
        We model each die as a node and compute marginals via Monte Carlo.
        """
        # Analytical solution: P(sum=k) for k=2..12
        analytical = np.zeros(13)  # index 0-12, only 2-12 are valid
        for i in range(1, 7):
            for j in range(1, 7):
                analytical[i + j] += 1
        analytical /= 36.0

        # Build two independent dice as a simple BN
        die1_var = Variable("Die1", [str(i) for i in range(1, 7)])
        die2_var = Variable("Die2", [str(i) for i in range(1, 7)])
        die1 = Node(variable=die1_var, prior=np.ones(6) / 6)
        die2 = Node(variable=die2_var, prior=np.ones(6) / 6)

        # Use CPT where die2 is independent of die1 (uniform for each parent)
        cpt = np.ones((6, 6)) / 6
        die1.add_child(die2, cpt)

        n_samples = 100_000
        samples = forward_sample(die1, [die1, die2], n_samples, seed=42)

        # Convert 0-indexed states (0-5) to die values (1-6), then sum both dice
        sums = (samples[:, 0] + 1) + (samples[:, 1] + 1)
        empirical = np.zeros(13)
        for k in range(2, 13):
            empirical[k] = np.sum(sums == k) / n_samples

        # Compare: max absolute error should be < 1%
        max_error = np.max(np.abs(analytical[2:] - empirical[2:]))
        assert max_error < 0.01, (
            f"Dice sum max error: {max_error:.4f} (limit: 0.01)"
        )

    def test_cancer_test_bayes(self):
        """Validate Bayes' theorem: cancer screening test.

        Classic problem:
        - P(cancer) = 0.01
        - P(positive | cancer) = 0.90 (sensitivity)
        - P(positive | no cancer) = 0.05 (false positive rate)

        Analytical: P(cancer | positive) = 0.01 * 0.90 / (0.01*0.90 + 0.99*0.05)
                                         ≈ 0.1538
        """
        # Analytical solution
        p_cancer = 0.01
        p_pos_given_cancer = 0.90
        p_pos_given_no_cancer = 0.05
        p_pos = p_cancer * p_pos_given_cancer + (1 - p_cancer) * p_pos_given_no_cancer
        p_cancer_given_pos = p_cancer * p_pos_given_cancer / p_pos

        # Build Bayesian network
        cancer_var = Variable("Cancer", ["no", "yes"])
        test_var = Variable("Test", ["negative", "positive"])

        cancer_node = Node(
            variable=cancer_var,
            prior=np.array([1 - p_cancer, p_cancer]),
        )
        test_node = Node(variable=test_var)

        # CPT: P(test | cancer)
        # Row 0 (no cancer): [P(neg|no_cancer), P(pos|no_cancer)]
        # Row 1 (cancer):    [P(neg|cancer),    P(pos|cancer)]
        cpt = np.array([
            [1 - p_pos_given_no_cancer, p_pos_given_no_cancer],
            [1 - p_pos_given_cancer, p_pos_given_cancer],
        ])
        cancer_node.add_child(test_node, cpt)

        # Monte Carlo estimation
        n_samples = 500_000
        samples = forward_sample(
            cancer_node, [cancer_node, test_node], n_samples, seed=42
        )

        # Filter for positive test results
        positive_mask = samples[:, 1] == 1  # "positive" is state index 1
        cancer_and_positive = np.sum((samples[:, 0] == 1) & positive_mask)
        total_positive = np.sum(positive_mask)

        if total_positive > 0:
            empirical_p = cancer_and_positive / total_positive
        else:
            empirical_p = 0.0

        # Allow 1% absolute tolerance for Monte Carlo estimation
        assert abs(empirical_p - p_cancer_given_pos) < 0.01, (
            f"P(cancer|positive): empirical={empirical_p:.4f}, "
            f"analytical={p_cancer_given_pos:.4f}"
        )

    def test_belief_propagation_chain_accuracy(self):
        """Verify BP on a chain matches Monte Carlo estimates."""
        root, nodes = build_chain(num_nodes=5, num_states=2, seed=123)

        bp_marginals = belief_propagation(root, nodes)
        mc_marginals = monte_carlo_marginals(root, nodes, 200_000, seed=123)

        for name in bp_marginals:
            bp = bp_marginals[name]
            mc = mc_marginals[name]
            max_diff = np.max(np.abs(bp - mc))
            assert max_diff < 0.02, (
                f"Node {name}: BP={bp}, MC={mc}, diff={max_diff:.4f}"
            )
