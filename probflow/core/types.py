"""Core types for ProbFlow distributions."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class Dist(ABC):
    """Abstract base class for probability distributions.

    This class defines the interface that all probability distributions
    must implement, including sampling, probability density/mass functions,
    cumulative distribution functions, and quantile functions.

    Operator overloads are provided for distribution composition:

    - ``__add__``: Sum of distributions
    - ``__mul__``: Product of distributions
    - ``__and__``: Joint distribution (independence)
    """

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw random samples from the distribution.

        Args:
            n: Number of samples to draw.

        Returns:
            Array of shape ``(n,)`` containing random samples.
        """
        pass

    @abstractmethod
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the probability density/mass function at *x*.

        Args:
            x: Point(s) at which to evaluate the PDF/PMF.

        Returns:
            Probability density/mass at *x*.
        """
        pass

    @abstractmethod
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the cumulative distribution function at *x*.

        Args:
            x: Point(s) at which to evaluate the CDF.

        Returns:
            Cumulative probability at *x*.
        """
        pass

    @abstractmethod
    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the quantile function (inverse CDF) at *q*.

        Args:
            q: Probability value(s) in [0, 1] at which to compute quantiles.

        Returns:
            Quantile value(s) corresponding to *q*.
        """
        pass

    def __add__(self, other: 'Dist') -> 'Dist':
        """Create a distribution representing the sum of two distributions.

        Args:
            other: Another distribution to add.

        Returns:
            A new distribution representing the sum.
        """
        return SumDist(self, other)

    def __mul__(self, other: 'Dist') -> 'Dist':
        """Create a distribution representing the product of two distributions.

        Args:
            other: Another distribution to multiply.

        Returns:
            A new distribution representing the product.
        """
        return ProductDist(self, other)

    def __and__(self, other: 'Dist') -> 'Dist':
        """Create a joint distribution assuming independence.

        Args:
            other: Another distribution to combine.

        Returns:
            A new distribution representing the joint distribution.
        """
        return JointDist(self, other)


class SumDist(Dist):
    """Distribution representing the sum of two independent distributions."""

    def __init__(self, dist1: Dist, dist2: Dist):
        """Initialize sum distribution.

        Args:
            dist1: First distribution.
            dist2: Second distribution.
        """
        self.dist1 = dist1
        self.dist2 = dist2

    def sample(self, n: int) -> np.ndarray:
        """Sample by drawing from both distributions and summing."""
        return self.dist1.sample(n) + self.dist2.sample(n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF of sum via convolution (approximated by sampling)."""
        raise NotImplementedError("PDF of sum distribution not implemented")

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of sum via convolution (approximated by sampling)."""
        raise NotImplementedError("CDF of sum distribution not implemented")

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile of sum via empirical sampling."""
        samples = self.sample(10000)
        return np.quantile(samples, q)


class ProductDist(Dist):
    """Distribution representing the product of two independent distributions."""

    def __init__(self, dist1: Dist, dist2: Dist):
        """Initialize product distribution.

        Args:
            dist1: First distribution.
            dist2: Second distribution.
        """
        self.dist1 = dist1
        self.dist2 = dist2

    def sample(self, n: int) -> np.ndarray:
        """Sample by drawing from both distributions and multiplying."""
        return self.dist1.sample(n) * self.dist2.sample(n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF of product (approximated by sampling)."""
        raise NotImplementedError("PDF of product distribution not implemented")

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of product (approximated by sampling)."""
        raise NotImplementedError("CDF of product distribution not implemented")

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile of product via empirical sampling."""
        samples = self.sample(10000)
        return np.quantile(samples, q)


class JointDist(Dist):
    """Joint distribution of two independent distributions."""

    def __init__(self, dist1: Dist, dist2: Dist):
        """Initialize joint distribution.

        Args:
            dist1: First distribution.
            dist2: Second distribution.
        """
        self.dist1 = dist1
        self.dist2 = dist2

    def sample(self, n: int) -> np.ndarray:
        """Sample from both distributions and return as pairs."""
        samples1 = self.dist1.sample(n)
        samples2 = self.dist2.sample(n)
        return np.column_stack([samples1, samples2])

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Joint PDF as product of marginal PDFs (assuming independence)."""
        if isinstance(x, np.ndarray) and x.ndim == 2:
            return self.dist1.pdf(x[:, 0]) * self.dist2.pdf(x[:, 1])
        raise NotImplementedError("Joint PDF requires 2D input")

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Joint CDF as product of marginal CDFs (assuming independence)."""
        if isinstance(x, np.ndarray) and x.ndim == 2:
            return self.dist1.cdf(x[:, 0]) * self.dist2.cdf(x[:, 1])
        raise NotImplementedError("Joint CDF requires 2D input")

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile of joint distribution via empirical sampling."""
        samples = self.sample(10000)
        return np.quantile(samples, q, axis=0)
