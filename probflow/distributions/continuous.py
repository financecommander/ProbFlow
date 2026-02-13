"""Continuous probability distributions."""

from typing import Union
import numpy as np
from scipy import stats
from ..core.types import Dist


class Normal(Dist):
    """Normal (Gaussian) distribution.

    The normal distribution is characterized by its mean (loc) and
    standard deviation (scale).

    Args:
        loc: Mean of the distribution (default: 0.0).
        scale: Standard deviation of the distribution (default: 1.0).

    Example:
        >>> norm = Normal(0, 1)
        >>> samples = norm.sample(1000)
        >>> prob = norm.pdf(0)
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        """Initialize Normal distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation of the distribution.

        Raises:
            ValueError: If scale <= 0.
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.loc = loc
        self.scale = scale
        self._dist = stats.norm(loc=loc, scale=scale)

    def sample(self, n: int) -> np.ndarray:
        """Draw random samples from the normal distribution.

        Args:
            n: Number of samples to draw.

        Returns:
            Array of shape ``(n,)`` containing random samples.
        """
        return self._dist.rvs(size=n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the probability density function at *x*.

        Args:
            x: Point(s) at which to evaluate the PDF.

        Returns:
            Probability density at *x*.
        """
        return self._dist.pdf(x)

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


class LogNormal(Dist):
    """Log-normal distribution.

    A log-normal distribution is the distribution of a random variable whose
    logarithm is normally distributed.

    Args:
        mu: Mean of the underlying normal distribution (default: 0.0).
        sigma: Standard deviation of the underlying normal distribution (default: 1.0).

    Example:
        >>> lognorm = LogNormal(0, 1)
        >>> samples = lognorm.sample(1000)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        """Initialize Log-normal distribution.

        Args:
            mu: Mean of the underlying normal distribution.
            sigma: Standard deviation of the underlying normal distribution.

        Raises:
            ValueError: If sigma <= 0.
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    def sample(self, n: int) -> np.ndarray:
        """Draw random samples from the log-normal distribution.

        Args:
            n: Number of samples to draw.

        Returns:
            Array of shape ``(n,)`` containing random samples.
        """
        return self._dist.rvs(size=n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the probability density function at *x*.

        Args:
            x: Point(s) at which to evaluate the PDF.

        Returns:
            Probability density at *x*.
        """
        return self._dist.pdf(x)

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


class Beta(Dist):
    """Beta distribution.

    The beta distribution is defined on the interval [0, 1] and is
    parameterized by two positive shape parameters alpha and beta.

    Args:
        alpha: First shape parameter (default: 1.0).
        beta: Second shape parameter (default: 1.0).

    Example:
        >>> beta_dist = Beta(2, 5)
        >>> samples = beta_dist.sample(1000)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """Initialize Beta distribution.

        Args:
            alpha: First shape parameter.
            beta: Second shape parameter.

        Raises:
            ValueError: If alpha <= 0 or beta <= 0.
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta
        self._dist = stats.beta(a=alpha, b=beta)

    def sample(self, n: int) -> np.ndarray:
        """Draw random samples from the beta distribution.

        Args:
            n: Number of samples to draw.

        Returns:
            Array of shape ``(n,)`` containing random samples.
        """
        return self._dist.rvs(size=n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the probability density function at *x*.

        Args:
            x: Point(s) at which to evaluate the PDF.

        Returns:
            Probability density at *x*.
        """
        return self._dist.pdf(x)

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
