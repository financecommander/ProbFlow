"""Discrete probability distributions."""

import numpy as np
from scipy import stats


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


class Bernoulli:
    """Bernoulli distribution for binary outcomes.

    Parameters
    ----------
    p : float
        Probability of success (1), must be in [0, 1].
    """

    def __init__(self, p):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self._dist = stats.bernoulli(self.p)

    def sample(self, n=1):
        """Draw n random samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            Array of 0s and 1s.
        """
        return self._dist.rvs(size=n)

    def pmf(self, x):
        """Probability mass function evaluated at x.

        Parameters
        ----------
        x : int or array-like
            Point(s) at which to evaluate the PMF.

        Returns
        -------
        float or numpy.ndarray
        """
        return self._dist.pmf(x)

    def cdf(self, x):
        """Cumulative distribution function evaluated at x.

        Parameters
        ----------
        x : float or array-like
            Point(s) at which to evaluate the CDF.

        Returns
        -------
        float or numpy.ndarray
        """
        return self._dist.cdf(x)

    def mode(self):
        """Return the mode of the distribution.

        Returns
        -------
        int
            1 if p > 0.5, 0 if p < 0.5, 1 if p == 0.5 (convention).
        """
        if self.p >= 0.5:
            return 1
        return 0

    def __and__(self, other):
        """Joint probability assuming independence: P(A & B) = P(A) * P(B).

        Parameters
        ----------
        other : Bernoulli
            Another Bernoulli distribution.

        Returns
        -------
        Bernoulli
            A new Bernoulli with p = self.p * other.p.
        """
        if not isinstance(other, Bernoulli):
            return NotImplemented
        return Bernoulli(self.p * other.p)

    def __or__(self, other):
        """Union probability assuming independence: P(A or B) = P(A) + P(B) - P(A)*P(B).

        Parameters
        ----------
        other : Bernoulli
            Another Bernoulli distribution.

        Returns
        -------
        Bernoulli
            A new Bernoulli with p = P(A) + P(B) - P(A)*P(B) (inclusion-exclusion).
        """
        if not isinstance(other, Bernoulli):
            return NotImplemented
        return Bernoulli(self.p + other.p - self.p * other.p)

    def __repr__(self):
        return f"Bernoulli(p={self.p})"


class Poisson:
    """Poisson distribution for count data.

    Parameters
    ----------
    lambda_ : float
        Rate parameter (mean), must be > 0.
    """

    def __init__(self, lambda_):
        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be > 0, got {lambda_}")
        self.lambda_ = float(lambda_)
        self._dist = stats.poisson(self.lambda_)

    def sample(self, n=1):
        """Draw n random samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            Array of non-negative integers.
        """
        return self._dist.rvs(size=n)

    def pmf(self, x):
        """Probability mass function evaluated at x.

        Parameters
        ----------
        x : int or array-like
            Point(s) at which to evaluate the PMF.

        Returns
        -------
        float or numpy.ndarray
        """
        return self._dist.pmf(x)

    def cdf(self, x):
        """Cumulative distribution function evaluated at x.

        Parameters
        ----------
        x : float or array-like
            Point(s) at which to evaluate the CDF.

        Returns
        -------
        float or numpy.ndarray
        """
        return self._dist.cdf(x)

    def mode(self):
        """Return the mode of the distribution.

        For Poisson, the mode is floor(lambda_). When lambda_ is an integer,
        both lambda_ and lambda_ - 1 are modes; we return lambda_ by convention.

        Returns
        -------
        int
        """
        return int(np.floor(self.lambda_))

    def __repr__(self):
        return f"Poisson(lambda_={self.lambda_})"


class Categorical:
    """Categorical distribution (multinomial with n=1).

    Parameters
    ----------
    probs : array-like
        Probabilities for each category. Must be non-negative and sum to 1.
    """

    def __init__(self, probs):
        probs = np.asarray(probs, dtype=float)
        if np.any(probs < 0):
            raise ValueError("All probabilities must be non-negative.")
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError(
                f"Probabilities must sum to 1, got {probs.sum()}"
            )
        self.probs = probs
        self._k = len(probs)

    def sample(self, n=1):
        """Draw n random samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            Array of category indices (0 to k-1).
        """
        return np.random.choice(self._k, size=n, p=self.probs)

    def pmf(self, x):
        """Probability mass function evaluated at x.

        Parameters
        ----------
        x : int or array-like
            Category index/indices at which to evaluate the PMF.

        Returns
        -------
        float or numpy.ndarray
        """
        x = np.asarray(x, dtype=int)
        in_range = (x >= 0) & (x < self._k)
        safe_x = np.where(in_range, x, 0)
        result = np.where(in_range, self.probs[safe_x], 0.0)
        return float(result) if result.ndim == 0 else result

    def cdf(self, x):
        """Cumulative distribution function evaluated at x.

        Parameters
        ----------
        x : int or float or array-like
            Point(s) at which to evaluate the CDF.

        Returns
        -------
        float or numpy.ndarray
        """
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

    def mode(self):
        """Return the mode (most probable category index).

        Returns
        -------
        int
        """
        return int(np.argmax(self.probs))

    def __repr__(self):
        return f"Categorical(probs={self.probs})"
