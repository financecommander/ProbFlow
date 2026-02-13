"""Tests for probflow/inference/sampling.py.

Covers:
- SimulationResults: mean, std, quantile, histogram, len, repr
- MonteCarloSimulation: single-process and multi-process runs
- @simulate decorator
- Convergence (10K samples → true mean within 0.01)
- Performance (<50ms for 10K samples)
- Reproducibility (seed control)
"""

from __future__ import annotations

import time

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI

from probflow.inference.sampling import (
    MonteCarloSimulation,
    SimulationResults,
    simulate,
)


def _random_uniform() -> float:
    """Module-level function for multiprocessing tests (lambdas can't be pickled)."""
    return float(np.random.random())


# ------------------------------------------------------------------ #
#  SimulationResults tests
# ------------------------------------------------------------------ #


class TestSimulationResults:
    """Tests for the SimulationResults container."""

    def test_mean(self) -> None:
        """Mean of known data is correct."""
        r = SimulationResults(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert r.mean() == pytest.approx(3.0)

    def test_std(self) -> None:
        """Std with ddof=1 of known data is correct."""
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        r = SimulationResults(data)
        assert r.std() == pytest.approx(float(np.std(data, ddof=1)))

    def test_quantile(self) -> None:
        """Quantile returns correct values."""
        data = np.arange(1.0, 101.0)
        r = SimulationResults(data)
        assert r.quantile(0.5) == pytest.approx(50.5)
        assert r.quantile(0.0) == pytest.approx(1.0)
        assert r.quantile(1.0) == pytest.approx(100.0)

    def test_quantile_invalid(self) -> None:
        """Quantile raises ValueError for q outside [0, 1]."""
        r = SimulationResults(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="q must be in"):
            r.quantile(-0.1)
        with pytest.raises(ValueError, match="q must be in"):
            r.quantile(1.5)

    def test_len(self) -> None:
        """Length matches number of samples."""
        r = SimulationResults(np.ones(42))
        assert len(r) == 42

    def test_repr(self) -> None:
        """repr contains class name, n, mean, and std."""
        r = SimulationResults(np.array([1.0, 2.0, 3.0]))
        text = repr(r)
        assert "SimulationResults" in text
        assert "n=3" in text

    def test_samples_property(self) -> None:
        """Samples property returns the underlying array."""
        data = np.array([1.0, 2.0, 3.0])
        r = SimulationResults(data)
        np.testing.assert_array_equal(r.samples, data)

    def test_histogram_returns_axes(self) -> None:
        """Histogram returns a matplotlib Axes without showing."""
        r = SimulationResults(np.random.default_rng(0).normal(size=100))
        ax = r.histogram(show=False)
        assert ax is not None


# ------------------------------------------------------------------ #
#  MonteCarloSimulation tests
# ------------------------------------------------------------------ #


class TestMonteCarloSimulation:
    """Tests for the MonteCarloSimulation engine."""

    def test_basic_run(self) -> None:
        """Run produces SimulationResults with correct length."""
        sim = MonteCarloSimulation(lambda: 1.0, n_samples=100, seed=0)
        results = sim.run()
        assert len(results) == 100
        assert results.mean() == pytest.approx(1.0)

    def test_invalid_n_samples(self) -> None:
        """ValueError for non-positive n_samples."""
        with pytest.raises(ValueError, match="n_samples must be positive"):
            MonteCarloSimulation(lambda: 1.0, n_samples=0)
        with pytest.raises(ValueError, match="n_samples must be positive"):
            MonteCarloSimulation(lambda: 1.0, n_samples=-5)

    def test_multiprocess_run(self) -> None:
        """Multi-worker run produces correct number of samples."""
        sim = MonteCarloSimulation(
            _random_uniform, n_samples=200, seed=42, n_workers=2,
        )
        results = sim.run()
        assert len(results) == 200

    def test_multiprocess_convergence(self) -> None:
        """Multi-worker run converges to true mean."""
        sim = MonteCarloSimulation(
            _random_uniform, n_samples=10_000, seed=99, n_workers=2,
        )
        results = sim.run()
        assert results.mean() == pytest.approx(0.5, abs=0.02)


# ------------------------------------------------------------------ #
#  @simulate decorator tests
# ------------------------------------------------------------------ #


class TestSimulateDecorator:
    """Tests for the @simulate decorator."""

    def test_decorator_returns_results(self) -> None:
        """Decorated function returns SimulationResults."""
        @simulate(n_samples=50, seed=0)
        def constant():
            return 42.0

        results = constant()
        assert isinstance(results, SimulationResults)
        assert len(results) == 50
        assert results.mean() == pytest.approx(42.0)

    def test_decorator_preserves_name(self) -> None:
        """Decorated function preserves original __name__."""
        @simulate(n_samples=10, seed=0)
        def my_simulation():
            return 0.0

        assert my_simulation.__name__ == "my_simulation"


# ------------------------------------------------------------------ #
#  Convergence tests
# ------------------------------------------------------------------ #


class TestConvergence:
    """Convergence tests: 10K samples → true mean within tolerance."""

    def test_uniform_mean_convergence(self) -> None:
        """10K Uniform(0,1) samples → mean ≈ 0.5 within 0.01."""
        sim = MonteCarloSimulation(
            lambda: np.random.random(), n_samples=10_000, seed=42,
        )
        results = sim.run()
        assert results.mean() == pytest.approx(0.5, abs=0.01)

    def test_normal_mean_convergence(self) -> None:
        """10K Normal(5, 1) samples → mean ≈ 5.0 within 0.05."""
        rng_holder = {"rng": np.random.default_rng(123)}

        def sample_normal():
            return float(rng_holder["rng"].normal(5.0, 1.0))

        sim = MonteCarloSimulation(sample_normal, n_samples=10_000, seed=42)
        results = sim.run()
        assert results.mean() == pytest.approx(5.0, abs=0.05)

    def test_bernoulli_convergence(self) -> None:
        """10K Bernoulli(0.3) samples → mean ≈ 0.3 within 0.01."""
        def bernoulli():
            return 1.0 if np.random.random() < 0.3 else 0.0

        sim = MonteCarloSimulation(bernoulli, n_samples=10_000, seed=7)
        results = sim.run()
        assert results.mean() == pytest.approx(0.3, abs=0.01)

    def test_exponential_convergence(self) -> None:
        """10K Exponential(rate=2) samples → mean ≈ 0.5 within 0.02."""
        def exponential():
            return np.random.exponential(0.5)

        sim = MonteCarloSimulation(exponential, n_samples=10_000, seed=11)
        results = sim.run()
        assert results.mean() == pytest.approx(0.5, abs=0.02)


# ------------------------------------------------------------------ #
#  Performance tests
# ------------------------------------------------------------------ #


class TestPerformance:
    """Performance tests: 10K samples should complete quickly."""

    def test_10k_samples_under_50ms(self) -> None:
        """10K simple samples should take <50ms (single-process)."""
        sim = MonteCarloSimulation(
            lambda: np.random.random(), n_samples=10_000, seed=0,
        )

        # Warm up
        sim.run()

        n_runs = 5
        start = time.perf_counter()
        for _ in range(n_runs):
            sim.run()
        elapsed = (time.perf_counter() - start) / n_runs

        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 50.0, (
            f"10K samples took {elapsed_ms:.1f}ms (limit: 50ms)"
        )


# ------------------------------------------------------------------ #
#  Reproducibility tests
# ------------------------------------------------------------------ #


class TestReproducibility:
    """Reproducibility tests: same seed → same results."""

    def test_same_seed_same_results(self) -> None:
        """Two runs with the same seed produce identical samples."""
        def random_val():
            return np.random.random()

        sim1 = MonteCarloSimulation(random_val, n_samples=100, seed=42)
        sim2 = MonteCarloSimulation(random_val, n_samples=100, seed=42)

        r1 = sim1.run()
        r2 = sim2.run()

        np.testing.assert_array_equal(r1.samples, r2.samples)

    def test_different_seed_different_results(self) -> None:
        """Two runs with different seeds produce different samples."""
        def random_val():
            return np.random.random()

        sim1 = MonteCarloSimulation(random_val, n_samples=100, seed=1)
        sim2 = MonteCarloSimulation(random_val, n_samples=100, seed=2)

        r1 = sim1.run()
        r2 = sim2.run()

        assert not np.array_equal(r1.samples, r2.samples)

    def test_decorator_reproducibility(self) -> None:
        """Decorated function with seed gives identical results."""
        @simulate(n_samples=100, seed=99)
        def roll():
            return np.random.random()

        r1 = roll()
        r2 = roll()
        np.testing.assert_array_equal(r1.samples, r2.samples)

    def test_no_seed_varies(self) -> None:
        """Runs without seed should generally differ."""
        sim1 = MonteCarloSimulation(
            lambda: np.random.random(), n_samples=100,
        )
        sim2 = MonteCarloSimulation(
            lambda: np.random.random(), n_samples=100,
        )
        r1 = sim1.run()
        r2 = sim2.run()
        # Extremely unlikely to be equal without shared seed
        assert not np.array_equal(r1.samples, r2.samples)
