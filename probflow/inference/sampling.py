"""Monte Carlo simulation for ProbFlow.

Provides:
- SimulationResults: container for simulation samples with statistics
- MonteCarloSimulation: engine that runs n independent samples
- simulate: decorator wrapping user functions for Monte Carlo sampling
"""

from __future__ import annotations

import functools
import multiprocessing
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


class SimulationResults:
    """Container for Monte Carlo simulation samples with summary statistics.

    Args:
        samples: 1-D array of simulation outputs.
    """

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = np.asarray(samples, dtype=np.float64)

    @property
    def samples(self) -> np.ndarray:
        """Return the raw sample array."""
        return self._samples

    def mean(self) -> float:
        """Compute the sample mean."""
        return float(np.mean(self._samples))

    def std(self) -> float:
        """Compute the sample standard deviation."""
        return float(np.std(self._samples, ddof=1))

    def quantile(self, q: float) -> float:
        """Compute the *q*-th quantile (0 ≤ q ≤ 1).

        Args:
            q: Quantile to compute.

        Returns:
            The quantile value.

        Raises:
            ValueError: If *q* is outside [0, 1].
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")
        return float(np.quantile(self._samples, q))

    def histogram(
        self,
        bins: int = 50,
        *,
        show: bool = True,
        ax: Any | None = None,
    ) -> Any:
        """Plot a histogram of the samples.

        Args:
            bins: Number of histogram bins.
            show: If *True*, call ``plt.show()``.
            ax: Optional matplotlib axes to plot on.

        Returns:
            The matplotlib axes object.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.hist(self._samples, bins=bins, density=True, alpha=0.7,
                edgecolor="black")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title("Simulation Results")
        if show:
            plt.show()
        return ax

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return (
            f"SimulationResults(n={len(self)}, "
            f"mean={self.mean():.6g}, std={self.std():.6g})"
        )


def _run_chunk(args: tuple[Callable[[], float], int, int]) -> np.ndarray:
    """Execute a chunk of simulation samples in a worker process.

    Args:
        args: Tuple of (func, chunk_size, seed).

    Returns:
        Array of sample values.
    """
    func, chunk_size, seed = args
    rng = np.random.default_rng(seed)
    # Set the global numpy random state for the worker so user code
    # calling np.random.* gets reproducible results.
    np.random.seed(rng.integers(0, 2**31))
    return np.array([func() for _ in range(chunk_size)], dtype=np.float64)


class MonteCarloSimulation:
    """Monte Carlo simulation engine.

    Runs *n_samples* independent evaluations of a callable and collects
    the results into a :class:`SimulationResults` object.

    Args:
        func: A callable that takes no arguments and returns a numeric
            scalar.
        n_samples: Number of independent samples to draw.
        seed: Optional random seed for reproducibility.
        n_workers: Number of worker processes.  ``1`` disables
            multiprocessing.  When *None*, defaults to 1.
    """

    def __init__(
        self,
        func: Callable[[], float],
        n_samples: int = 10_000,
        seed: int | None = None,
        n_workers: int | None = None,
    ) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        self.func = func
        self.n_samples = n_samples
        self.seed = seed
        self.n_workers = n_workers if n_workers is not None else 1

    def run(self) -> SimulationResults:
        """Execute the simulation and return results.

        Returns:
            A :class:`SimulationResults` containing all samples.
        """
        rng = np.random.default_rng(self.seed)

        if self.n_workers <= 1:
            # Single-process: use vectorised loop with seed control
            np.random.seed(
                rng.integers(0, 2**31) if self.seed is not None else None
            )
            samples = np.array(
                [self.func() for _ in range(self.n_samples)],
                dtype=np.float64,
            )
        else:
            # Split work across processes
            chunk_sizes = _split_work(self.n_samples, self.n_workers)
            seeds = rng.integers(0, 2**31, size=len(chunk_sizes)).tolist()
            tasks = [
                (self.func, cs, s)
                for cs, s in zip(chunk_sizes, seeds)
            ]
            with multiprocessing.Pool(processes=self.n_workers) as pool:
                chunks = pool.map(_run_chunk, tasks)
            samples = np.concatenate(chunks)

        return SimulationResults(samples)


def _split_work(total: int, n_workers: int) -> list[int]:
    """Divide *total* items as evenly as possible across *n_workers*."""
    base, remainder = divmod(total, n_workers)
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


def simulate(
    n_samples: int = 10_000,
    seed: int | None = None,
    n_workers: int | None = None,
) -> Callable:
    """Decorator that wraps a function for Monte Carlo simulation.

    The decorated function, when called, returns a
    :class:`SimulationResults` object.

    Args:
        n_samples: Number of independent samples.
        seed: Optional random seed.
        n_workers: Number of worker processes (default 1).

    Returns:
        A decorator.

    Example::

        @simulate(n_samples=10_000, seed=42)
        def coin_flip():
            return 1.0 if np.random.random() < 0.5 else 0.0

        results = coin_flip()
        print(results.mean())  # ≈ 0.5
    """
    def decorator(func: Callable[[], float]) -> Callable[[], SimulationResults]:
        @functools.wraps(func)
        def wrapper() -> SimulationResults:
            sim = MonteCarloSimulation(
                func, n_samples=n_samples, seed=seed, n_workers=n_workers,
            )
            return sim.run()
        return wrapper
    return decorator
