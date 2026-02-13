"""ProbFlow: A Python DSL for probabilistic programming.

This package provides tools for defining and working with probability
distributions, including continuous and discrete distributions, distribution
operations, and a context manager for probabilistic model definition.
"""

from .core.types import Dist
from .core.context import ProbFlow
from .distributions.continuous import Normal, LogNormal, Beta
from .distributions.discrete import Bernoulli, Poisson, Categorical

__version__ = "0.1.0"

__all__ = [
    "Dist",
    "ProbFlow",
    "Normal",
    "LogNormal",
    "Beta",
    "Bernoulli",
    "Poisson",
    "Categorical",
]
