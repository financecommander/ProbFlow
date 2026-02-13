"""Discrete probability distributions."""

import numpy as np
from scipy import stats


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

    @property
    def mean(self):
        """Mean of the distribution."""
        return self.p

    @property
    def variance(self):
        """Variance of the distribution."""
        return self.p * (1.0 - self.p)

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
