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


class LogNormal:
    """Log-normal distribution parameterised by *mu* and *sigma* of the
    underlying normal (i.e. ``ln(X) ~ N(mu, sigma)``).

    Supports scaling via ``*`` (multiplying a log-normal RV by a positive
    constant shifts *mu*).
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._degenerate = self.sigma == 0.0
        if not self._degenerate:
            # scipy's lognorm: s=sigma, scale=exp(mu)
            self._dist = stats.lognorm(s=self.sigma, scale=np.exp(self.mu))

    # ---------- core API ----------

    def sample(self, n: int = 1) -> np.ndarray:
        if self._degenerate:
            return np.full(n, np.exp(self.mu))
        return self._dist.rvs(size=n)

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        if self._degenerate:
            x = np.asarray(x, dtype=float)
            return np.where(x == np.exp(self.mu), np.inf, 0.0)
        return self._dist.pdf(x)

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        if self._degenerate:
            x = np.asarray(x, dtype=float)
            return np.where(x >= np.exp(self.mu), 1.0, 0.0)
        return self._dist.cdf(x)

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        if self._degenerate:
            return np.full_like(np.asarray(q, dtype=float), np.exp(self.mu))
        return self._dist.ppf(q)

    def mean(self) -> float:
        if self._degenerate:
            return float(np.exp(self.mu))
        return float(self._dist.mean())

    def variance(self) -> float:
        if self._degenerate:
            return 0.0
        return float(self._dist.var())

    # ---------- operators ----------

    def __mul__(self, scalar: float) -> LogNormal:
        """Scaling a log-normal RV by a positive constant *c*:
        if X ~ LogNormal(mu, sigma), then c*X ~ LogNormal(mu + ln(c), sigma).
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar <= 0:
            raise ValueError("LogNormal can only be scaled by a positive constant")
        return LogNormal(self.mu + np.log(scalar), self.sigma)

    def __rmul__(self, scalar: float) -> LogNormal:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return f"LogNormal(mu={self.mu}, sigma={self.sigma})"


class Beta:
    """Beta distribution parameterised by *alpha* and *beta* shape parameters.

    Supports scaling via ``*`` (result is a scaled Beta on ``[0, c]``
    rather than ``[0, 1]``).  Internally this is tracked via *loc* and *scale*
    of :func:`scipy.stats.beta`.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.loc = float(loc)
        self.scale = float(scale)
        self._dist = stats.beta(self.alpha, self.beta, loc=self.loc, scale=self.scale)

    # ---------- core API ----------

    def sample(self, n: int = 1) -> np.ndarray:
        return self._dist.rvs(size=n)

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        return self._dist.cdf(x)

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        return self._dist.ppf(q)

    def mean(self) -> float:
        return float(self._dist.mean())

    def variance(self) -> float:
        return float(self._dist.var())

    # ---------- operators ----------

    def __mul__(self, scalar: float) -> Beta:
        """Scale the support by a constant: if X ~ Beta(a, b) on [loc, loc+scale],
        then c*X ~ Beta(a, b) on [c*loc, c*loc + c*scale].
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Beta(
            self.alpha, self.beta, loc=scalar * self.loc, scale=scalar * self.scale
        )

    def __rmul__(self, scalar: float) -> Beta:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return f"Beta(alpha={self.alpha}, beta={self.beta})"
