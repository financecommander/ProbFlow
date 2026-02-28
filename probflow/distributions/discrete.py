"""Discrete probability distributions."""

from typing import Union

import numpy as np
from scipy import stats

from ..core.types import Dist


def probability(distribution, threshold, comparison="gt"):
    """Compute the probability of a distribution exceeding or falling below a threshold.

    Parameters
    ----------
    distribution : Bernoulli, Poisson, or Categorical
        A discrete distribution instance.
    threshold : float or int
        The threshold value.
    comparison : str
        One of "gt" (P(X > threshold)), "ge" (P(X >= threshold)),
        "lt" (P(X < threshold)), "le" (P(X <= threshold)),
        or "eq" (P(X == threshold)).

    Returns
    -------
    float
        The computed probability.
    """
    if comparison == "gt":
        return 1.0 - distribution.cdf(threshold)
    elif comparison == "ge":
        return 1.0 - distribution.cdf(threshold - 1)
    elif comparison == "lt":
        return distribution.cdf(threshold - 1)
    elif comparison == "le":
        return distribution.cdf(threshold)
    elif comparison == "eq":
        return distribution.pmf(threshold)
    else:
        raise ValueError(
            f"Unknown comparison '{comparison}'. Use 'gt', 'ge', 'lt', 'le', or 'eq'."
        )


class Bernoulli(Dist):
    """Bernoulli distribution for binary outcomes.

    Parameters
    ----------
    p : float
        Probability of success (1), must be in [0, 1].
    """

    def __init__(self, p=0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self._dist = stats.bernoulli(self.p)

    def sample(self, n=1):
        """Draw n random samples from the distribution."""
        return self._dist.rvs(size=n)

    def pmf(self, x):
        """Probability mass function evaluated at x."""
        return self._dist.pmf(x)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for pmf() to satisfy Dist ABC interface."""
        return self.pmf(x)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function evaluated at x."""
        return self._dist.cdf(x)

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function (inverse CDF) at q."""
        return self._dist.ppf(q)

    def mode(self):
        """Return the mode of the distribution."""
        if self.p >= 0.5:
            return 1
        return 0

    def __and__(self, other):
        """Joint probability assuming independence: P(A & B) = P(A) * P(B)."""
        if not isinstance(other, Bernoulli):
            return NotImplemented
        return Bernoulli(self.p * other.p)

    def __or__(self, other):
        """Union probability assuming independence: P(A or B) = P(A) + P(B) - P(A)*P(B)."""
        if not isinstance(other, Bernoulli):
            return NotImplemented
        return Bernoulli(self.p + other.p - self.p * other.p)

    def __repr__(self):
        return f"Bernoulli(p={self.p})"


class Poisson(Dist):
    """Poisson distribution for count data.

    Parameters
    ----------
    lambda_ : float
        Rate parameter (mean), must be > 0.
    """

    def __init__(self, lambda_=1.0):
        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be > 0, got {lambda_}")
        self.lambda_ = float(lambda_)
        self._dist = stats.poisson(self.lambda_)

    def sample(self, n=1):
        """Draw n random samples from the distribution."""
        return self._dist.rvs(size=n)

    def pmf(self, x):
        """Probability mass function evaluated at x."""
        return self._dist.pmf(x)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for pmf() to satisfy Dist ABC interface."""
        return self.pmf(x)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function evaluated at x."""
        return self._dist.cdf(x)

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function (inverse CDF) at q."""
        return self._dist.ppf(q)

    def mode(self):
        """Return the mode of the distribution."""
        return int(np.floor(self.lambda_))

    def __repr__(self):
        return f"Poisson(lambda_={self.lambda_})"


class Categorical(Dist):
    """Categorical distribution (multinomial with n=1).

    Parameters
    ----------
    probs : array-like
        Probabilities for each category. Must be non-negative and sum to 1.
    labels : list of str, optional
        Human-readable labels for each category. If provided, samples
        return label strings instead of integer indices.
    """

    def __init__(self, probs, labels=None):
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

    def sample(self, n=1):
        """Draw n random samples from the distribution.

        Returns label strings if labels were provided, otherwise integers.
        """
        indices = np.random.choice(self._k, size=n, p=self.probs)
        if self.labels is not None:
            return np.array([self.labels[i] for i in indices])
        return indices

    def pmf(self, x):
        """Probability mass function evaluated at x."""
        if self.labels is not None:
            label_to_idx = {label: i for i, label in enumerate(self.labels)}
            x_arr = np.atleast_1d(x)
            result = np.array([
                self.probs[label_to_idx[v]] if v in label_to_idx else 0.0
                for v in x_arr
            ])
            return float(result[0]) if np.ndim(x) == 0 or isinstance(x, str) else result
        x = np.asarray(x, dtype=int)
        in_range = (x >= 0) & (x < self._k)
        safe_x = np.where(in_range, x, 0)
        result = np.where(in_range, self.probs[safe_x], 0.0)
        return float(result) if result.ndim == 0 else result

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for pmf() to satisfy Dist ABC interface."""
        return self.pmf(x)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function evaluated at x."""
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

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function (inverse CDF) at q."""
        q = np.asarray(q, dtype=float)
        cumprobs = np.cumsum(self.probs)

        def _scalar_quantile(prob):
            for i, cp in enumerate(cumprobs):
                if prob <= cp:
                    return float(i)
            return float(self._k - 1)

        if q.ndim == 0:
            return _scalar_quantile(float(q))
        return np.array([_scalar_quantile(p) for p in q.flat]).reshape(q.shape)

    def mode(self):
        """Return the most probable category."""
        idx = int(np.argmax(self.probs))
        if self.labels is not None:
            return self.labels[idx]
        return idx

    def __repr__(self):
        return f"Categorical(probs={self.probs})"
