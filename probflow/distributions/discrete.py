"""Discrete probability distributions."""

from __future__ import annotations

import numpy as np
from scipy import stats


class Categorical:
    """Categorical distribution (multinomial with n=1).

    Parameters
    ----------
    probs : array-like
        Probabilities for each category. Must be non-negative and sum to 1.
    labels : list of str, optional
        Human-readable labels for each category. If provided, samples
        return label strings instead of integer indices.
    """

    def __init__(self, probs, labels=None) -> None:
        probs = np.asarray(probs, dtype=float)
        if np.any(probs < 0):
            raise ValueError("All probabilities must be non-negative.")
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError(
                f"Probabilities must sum to 1, got {probs.sum()}"
            )
        self.probs = probs
        self._k = len(probs)
        self.labels = list(labels) if labels is not None else None
        if self.labels is not None and len(self.labels) != self._k:
            raise ValueError("Number of labels must match number of probabilities")

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw *n* random samples.

        Returns label strings if *labels* were provided, otherwise integers.
        """
        indices = np.random.choice(self._k, size=n, p=self.probs)
        if self.labels is not None:
            return np.array([self.labels[i] for i in indices])
        return indices

    def pmf(self, x) -> float | np.ndarray:
        """Probability mass function evaluated at *x*."""
        if self.labels is not None:
            label_to_idx = {l: i for i, l in enumerate(self.labels)}
            x_arr = np.atleast_1d(x)
            result = np.array([self.probs[label_to_idx[v]] if v in label_to_idx else 0.0
                               for v in x_arr])
            return float(result[0]) if np.ndim(x) == 0 or isinstance(x, str) else result
        x = np.asarray(x, dtype=int)
        in_range = (x >= 0) & (x < self._k)
        safe_x = np.where(in_range, x, 0)
        result = np.where(in_range, self.probs[safe_x], 0.0)
        return float(result) if result.ndim == 0 else result

    def cdf(self, x) -> float | np.ndarray:
        """Cumulative distribution function evaluated at *x*."""
        x = np.asarray(x)
        cumprobs = np.cumsum(self.probs)

        def _scalar_cdf(val):
            if val < 0:
                return 0.0
            idx = int(np.floor(val))
            if idx >= self._k:
                return 1.0
            return float(cumprobs[idx])

        if x.ndim == 0:
            return _scalar_cdf(x)
        return np.array([_scalar_cdf(v) for v in x.flat]).reshape(x.shape)

    def mode(self) -> int | str:
        """Return the most probable category."""
        idx = int(np.argmax(self.probs))
        if self.labels is not None:
            return self.labels[idx]
        return idx

    def __repr__(self) -> str:
        return f"Categorical(probs={self.probs})"
