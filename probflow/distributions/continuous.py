"""Continuous probability distributions backed by scipy.stats."""

from __future__ import annotations

import numpy as np
from scipy import stats


class Normal:
    """Gaussian distribution parameterised by *mu* (mean) and *sigma* (std dev).

    Supports closed-form convolution via ``+`` (sum of independent normals)
    and affine scaling via ``*``.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._degenerate = self.sigma == 0.0
        if not self._degenerate:
            self._dist = stats.norm(loc=self.mu, scale=self.sigma)

    # ---------- core API ----------

    def sample(self, n: int = 1) -> np.ndarray:
        if self._degenerate:
            return np.full(n, self.mu)
        return self._dist.rvs(size=n)

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        if self._degenerate:
            x = np.asarray(x, dtype=float)
            return np.where(x == self.mu, np.inf, 0.0)
        return self._dist.pdf(x)

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        if self._degenerate:
            x = np.asarray(x, dtype=float)
            return np.where(x >= self.mu, 1.0, 0.0)
        return self._dist.cdf(x)

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        if self._degenerate:
            return np.full_like(np.asarray(q, dtype=float), self.mu)
        return self._dist.ppf(q)

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma**2

    # ---------- operators ----------

    def __add__(self, other: Normal) -> Normal:
        """Convolution of two independent Normal distributions."""
        if not isinstance(other, Normal):
            return NotImplemented
        new_mu = self.mu + other.mu
        new_sigma = np.sqrt(self.sigma**2 + other.sigma**2)
        return Normal(new_mu, new_sigma)

    def __mul__(self, scalar: float) -> Normal:
        """Affine scaling: if X ~ N(mu, sigma), then c*X ~ N(c*mu, |c|*sigma)."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Normal(scalar * self.mu, abs(scalar) * self.sigma)

    def __rmul__(self, scalar: float) -> Normal:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return f"Normal(mu={self.mu}, sigma={self.sigma})"
