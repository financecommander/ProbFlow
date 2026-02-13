"""Discrete probability distributions."""

from typing import Union
import numpy as np
from scipy import stats
from ..core.types import Dist


class Bernoulli(Dist):
    """Bernoulli distribution.

    The Bernoulli distribution models a single trial with two possible outcomes:
    success (1) with probability p, or failure (0) with probability 1-p.

    Args:
        p: Probability of success (default: 0.5).

    Example:
        >>> bern = Bernoulli(0.7)
        >>> samples = bern.sample(1000)
    """

    def __init__(self, p: float = 0.5):
        """Initialize Bernoulli distribution.

        Args:
            p: Probability of success.

        Raises:
            ValueError: If p is not in [0, 1].
        """
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self._dist = stats.bernoulli(p=p)

    def sample(self, n: int) -> np.ndarray:
        """Draw random samples from the Bernoulli distribution.

        Args:
            n: Number of samples to draw.

        Returns:
            Array of shape ``(n,)`` containing 0s and 1s.
        """
        return self._dist.rvs(size=n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the probability mass function at *x*.

        Args:
            x: Point(s) at which to evaluate the PMF (should be 0 or 1).

        Returns:
            Probability mass at *x*.
        """
        return self._dist.pmf(x)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the cumulative distribution function at *x*.

        Args:
            x: Point(s) at which to evaluate the CDF.

        Returns:
            Cumulative probability at *x*.
        """
        return self._dist.cdf(x)

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the quantile function (inverse CDF) at *q*.

        Args:
            q: Probability value(s) in [0, 1] at which to compute quantiles.

        Returns:
            Quantile value(s) corresponding to *q*.
        """
        return self._dist.ppf(q)


class Poisson(Dist):
    """Poisson distribution.

    The Poisson distribution models the number of events occurring in a fixed
    interval of time or space, given a constant mean rate.

    Args:
        lam: Expected number of events (rate parameter, default: 1.0).

    Example:
        >>> poisson = Poisson(3.5)
        >>> samples = poisson.sample(1000)
    """

    def __init__(self, lam: float = 1.0):
        """Initialize Poisson distribution.

        Args:
            lam: Rate parameter (expected number of events).

        Raises:
            ValueError: If lam <= 0.
        """
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.lam = lam
        self._dist = stats.poisson(mu=lam)

    def sample(self, n: int) -> np.ndarray:
        """Draw random samples from the Poisson distribution.

        Args:
            n: Number of samples to draw.

        Returns:
            Array of shape ``(n,)`` containing non-negative integers.
        """
        return self._dist.rvs(size=n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the probability mass function at *x*.

        Args:
            x: Point(s) at which to evaluate the PMF (should be non-negative integers).

        Returns:
            Probability mass at *x*.
        """
        return self._dist.pmf(x)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the cumulative distribution function at *x*.

        Args:
            x: Point(s) at which to evaluate the CDF.

        Returns:
            Cumulative probability at *x*.
        """
        return self._dist.cdf(x)

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the quantile function (inverse CDF) at *q*.

        Args:
            q: Probability value(s) in [0, 1] at which to compute quantiles.

        Returns:
            Quantile value(s) corresponding to *q*.
        """
        return self._dist.ppf(q)


class Categorical(Dist):
    """Categorical distribution.

    The categorical distribution models a random variable that can take one of K
    different values, each with a specific probability.

    Args:
        probs: Array of probabilities for each category (must sum to 1).

    Example:
        >>> cat = Categorical([0.2, 0.3, 0.5])
        >>> samples = cat.sample(1000)
    """

    def __init__(self, probs: Union[list, np.ndarray]):
        """Initialize Categorical distribution.

        Args:
            probs: Probabilities for each category.

        Raises:
            ValueError: If probs don't sum to 1 or contain negative values.
        """
        probs = np.array(probs)
        if not np.allclose(probs.sum(), 1.0):
            raise ValueError("probabilities must sum to 1")
        if np.any(probs < 0):
            raise ValueError("probabilities must be non-negative")
        self.probs = probs
        self.k = len(probs)
        values = np.arange(self.k)
        self._dist = stats.rv_discrete(values=(values, probs))

    def sample(self, n: int) -> np.ndarray:
        """Draw random samples from the categorical distribution.

        Args:
            n: Number of samples to draw.

        Returns:
            Array of shape ``(n,)`` containing category indices (0 to k-1).
        """
        return self._dist.rvs(size=n)

    def pdf(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the probability mass function at *x*.

        Args:
            x: Category index/indices at which to evaluate the PMF.

        Returns:
            Probability mass at *x*.
        """
        return self._dist.pmf(x)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the cumulative distribution function at *x*.

        Args:
            x: Point(s) at which to evaluate the CDF.

        Returns:
            Cumulative probability at *x*.
        """
        return self._dist.cdf(x)

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the quantile function (inverse CDF) at *q*.

        Args:
            q: Probability value(s) in [0, 1] at which to compute quantiles.

        Returns:
            Quantile value(s) corresponding to *q*.
        """
        return self._dist.ppf(q)
