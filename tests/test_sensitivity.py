"""Tests for probflow.inference.sensitivity."""

import time

import numpy as np
import pytest

from probflow.inference.sensitivity import sensitivity_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_network(params):
    """y = 3*a + 2*b + 0.5*c"""
    return {"y": 3.0 * params["a"] + 2.0 * params["b"] + 0.5 * params["c"]}


def _quadratic_network(params):
    """y = a^2 + 3*b"""
    return {"y": params["a"] ** 2 + 3.0 * params["b"]}


def _distribution_network(params):
    """Simulate a Gaussian-like target: target = mu + 2*sigma."""
    return {"target": params["mu"] + 2.0 * params["sigma"]}


# ---------------------------------------------------------------------------
# Linear model – known sensitivities
# ---------------------------------------------------------------------------

class TestLinearModel:
    """For y = 3a + 2b + 0.5c the exact sensitivities are 3, 2, 0.5."""

    def test_exact_sensitivities(self):
        params = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = sensitivity_analysis(_linear_network, "y", params)

        assert abs(result["a"] - 3.0) < 1e-8
        assert abs(result["b"] - 2.0) < 1e-8
        assert abs(result["c"] - 0.5) < 1e-8

    def test_sensitivities_at_different_point(self):
        params = {"a": 5.0, "b": -3.0, "c": 10.0}
        result = sensitivity_analysis(_linear_network, "y", params)

        assert abs(result["a"] - 3.0) < 1e-8
        assert abs(result["b"] - 2.0) < 1e-8
        assert abs(result["c"] - 0.5) < 1e-8

    def test_parameter_at_zero(self):
        """When a parameter value is zero, h falls back to perturbation."""
        params = {"a": 0.0, "b": 1.0, "c": 1.0}
        result = sensitivity_analysis(_linear_network, "y", params)

        assert abs(result["a"] - 3.0) < 1e-8


# ---------------------------------------------------------------------------
# Rank ordering correctness
# ---------------------------------------------------------------------------

class TestRankOrdering:
    """Results must be sorted by descending absolute sensitivity."""

    def test_linear_rank_order(self):
        params = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = sensitivity_analysis(_linear_network, "y", params)

        keys = list(result.keys())
        assert keys == ["a", "b", "c"], f"Expected ['a', 'b', 'c'], got {keys}"

    def test_quadratic_rank_order(self):
        """For y = a^2 + 3b at a=1: dy/da = 2, dy/db = 3 → b first."""
        params = {"a": 1.0, "b": 1.0}
        result = sensitivity_analysis(_quadratic_network, "y", params)

        keys = list(result.keys())
        assert keys[0] == "b", f"Expected 'b' first, got {keys}"

    def test_quadratic_rank_at_large_a(self):
        """At a=10: dy/da = 20, dy/db = 3 → a first."""
        params = {"a": 10.0, "b": 1.0}
        result = sensitivity_analysis(_quadratic_network, "y", params)

        keys = list(result.keys())
        assert keys[0] == "a", f"Expected 'a' first, got {keys}"


# ---------------------------------------------------------------------------
# Distribution parameters (mu, sigma)
# ---------------------------------------------------------------------------

class TestDistributionParams:
    """Support distribution parameters such as mu and sigma."""

    def test_mu_sensitivity(self):
        params = {"mu": 0.0, "sigma": 1.0}
        result = sensitivity_analysis(_distribution_network, "target", params)

        assert abs(result["mu"] - 1.0) < 1e-8

    def test_sigma_sensitivity(self):
        params = {"mu": 0.0, "sigma": 1.0}
        result = sensitivity_analysis(_distribution_network, "target", params)

        assert abs(result["sigma"] - 2.0) < 1e-8

    def test_sigma_ranked_higher(self):
        params = {"mu": 0.0, "sigma": 1.0}
        result = sensitivity_analysis(_distribution_network, "target", params)

        keys = list(result.keys())
        assert keys[0] == "sigma"


# ---------------------------------------------------------------------------
# Computational cost
# ---------------------------------------------------------------------------

class TestComputationalCost:
    """Sensitivity analysis should call the network exactly 2*N times."""

    def test_call_count(self):
        call_count = {"n": 0}

        def counting_network(params):
            call_count["n"] += 1
            return _linear_network(params)

        params = {"a": 1.0, "b": 1.0, "c": 1.0}
        sensitivity_analysis(counting_network, "y", params)

        # 2 calls (plus/minus) per parameter → 6 total
        assert call_count["n"] == 6

    def test_reasonable_time(self):
        params = {"a": 1.0, "b": 1.0, "c": 1.0}
        start = time.perf_counter()
        sensitivity_analysis(_linear_network, "y", params)
        elapsed = time.perf_counter() - start

        # Should finish well under 1 second for a trivial network
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# Custom perturbation
# ---------------------------------------------------------------------------

class TestCustomPerturbation:
    def test_small_perturbation(self):
        params = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = sensitivity_analysis(
            _linear_network, "y", params, perturbation=0.001
        )
        assert abs(result["a"] - 3.0) < 1e-8

    def test_large_perturbation(self):
        params = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = sensitivity_analysis(
            _linear_network, "y", params, perturbation=0.5
        )
        assert abs(result["a"] - 3.0) < 1e-8


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_non_callable_network(self):
        with pytest.raises(TypeError, match="callable"):
            sensitivity_analysis("not_callable", "y", {"a": 1.0})

    def test_empty_parameters(self):
        with pytest.raises(ValueError, match="non-empty"):
            sensitivity_analysis(_linear_network, "y", {})

    def test_missing_target(self):
        def simple(params):
            return {"y": params["a"]}

        with pytest.raises(KeyError, match="not_a_target"):
            sensitivity_analysis(simple, "not_a_target", {"a": 1.0})
