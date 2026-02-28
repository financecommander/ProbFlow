"""Core types for ProbFlow probabilistic models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Distribution ABC
# ---------------------------------------------------------------------------

class Dist(ABC):
    """Abstract base class for probability distributions.

    This class defines the interface that all probability distributions
    must implement, including sampling, probability density/mass functions,
    cumulative distribution functions, and quantile functions.

    Operator overloads are provided for distribution composition:
    - __add__: Sum of distributions
    - __mul__: Product of distributions
    - __and__: Joint distribution (independence)
    """

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw random samples from the distribution."""
        pass

    @abstractmethod
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the probability density/mass function at x."""
        pass

    @abstractmethod
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the cumulative distribution function at x."""
        pass

    @abstractmethod
    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the quantile function (inverse CDF) at q."""
        pass

    def __add__(self, other: Dist) -> Dist:
        """Create a distribution representing the sum of two distributions."""
        return SumDist(self, other)

    def __mul__(self, other: Dist) -> Dist:
        """Create a distribution representing the product of two distributions."""
        return ProductDist(self, other)

    def __and__(self, other: Dist) -> Dist:
        """Create a joint distribution assuming independence."""
        return JointDist(self, other)


class SumDist(Dist):
    """Distribution representing the sum of two independent distributions."""

    def __init__(self, dist1: Dist, dist2: Dist):
        self.dist1 = dist1
        self.dist2 = dist2

    def sample(self, n: int) -> np.ndarray:
        return self.dist1.sample(n) + self.dist2.sample(n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError("PDF of sum distribution not implemented")

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError("CDF of sum distribution not implemented")

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        samples = self.sample(10000)
        return np.quantile(samples, q)


class ProductDist(Dist):
    """Distribution representing the product of two independent distributions."""

    def __init__(self, dist1: Dist, dist2: Dist):
        self.dist1 = dist1
        self.dist2 = dist2

    def sample(self, n: int) -> np.ndarray:
        return self.dist1.sample(n) * self.dist2.sample(n)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError("PDF of product distribution not implemented")

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError("CDF of product distribution not implemented")

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        samples = self.sample(10000)
        return np.quantile(samples, q)


class JointDist(Dist):
    """Joint distribution of two independent distributions."""

    def __init__(self, dist1: Dist, dist2: Dist):
        self.dist1 = dist1
        self.dist2 = dist2

    def sample(self, n: int) -> np.ndarray:
        samples1 = self.dist1.sample(n)
        samples2 = self.dist2.sample(n)
        return np.column_stack([samples1, samples2])

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(x, np.ndarray) and x.ndim == 2:
            return self.dist1.pdf(x[:, 0]) * self.dist2.pdf(x[:, 1])
        raise NotImplementedError("Joint PDF requires 2D input")

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(x, np.ndarray) and x.ndim == 2:
            return self.dist1.cdf(x[:, 0]) * self.dist2.cdf(x[:, 1])
        raise NotImplementedError("Joint CDF requires 2D input")

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        samples = self.sample(10000)
        return np.quantile(samples, q, axis=0)


# ---------------------------------------------------------------------------
# Factor Graph / Bayesian Network types
# ---------------------------------------------------------------------------

@dataclass
class Variable:
    """A discrete random variable with a finite set of states."""

    name: str
    states: List[str]

    @property
    def num_states(self) -> int:
        return len(self.states)


@dataclass
class Factor:
    """A factor (potential function) over a set of variables."""

    variables: List[Variable]
    values: np.ndarray  # shape: (|var1|, |var2|, ...)

    def __post_init__(self) -> None:
        expected = tuple(v.num_states for v in self.variables)
        if self.values.shape != expected:
            raise ValueError(
                f"Factor shape {self.values.shape} does not match "
                f"variable cardinalities {expected}"
            )


@dataclass
class Node:
    """A node in a factor graph / Bayesian network."""

    variable: Variable
    prior: Optional[np.ndarray] = None  # shape: (num_states,)
    children: List[Node] = field(default_factory=list)
    parents: List[Node] = field(default_factory=list)
    cpt: Optional[np.ndarray] = None  # conditional probability table

    @property
    def name(self) -> str:
        return self.variable.name

    def add_child(self, child: Node, cpt: np.ndarray) -> None:
        """Add a child node with its conditional probability table."""
        self.children.append(child)
        child.parents.append(self)
        child.cpt = cpt
