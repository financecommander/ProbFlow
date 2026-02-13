"""Bridge between ternary (Triton-style) logic and probabilistic distributions.

Provides conversion utilities for mapping ternary truth values (TRUE, FALSE,
UNKNOWN) to and from Bernoulli probability distributions, enabling seamless
integration between Triton AST nodes and ProbFlow's probabilistic framework.

Example
-------
>>> from probflow.integration.triton_bridge import (
...     TernaryValue, triton_to_prob, prob_to_triton, BinaryDist,
... )
>>> dist = triton_to_prob(TernaryValue.TRUE)
>>> dist.p
0.95
>>> prob_to_triton(dist)
<TernaryValue.TRUE: 'TRUE'>
"""

from __future__ import annotations

import enum
from typing import Union

from probflow.distributions.discrete import Bernoulli


class TernaryValue(enum.Enum):
    """Ternary truth value used in Triton-style AST nodes.

    Members
    -------
    TRUE
        Definitely true.
    FALSE
        Definitely false.
    UNKNOWN
        Unknown / indeterminate.
    """

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


# Default probability mappings for ternary → Bernoulli conversion.
_TERNARY_TO_PROB = {
    TernaryValue.TRUE: 0.95,
    TernaryValue.FALSE: 0.05,
    TernaryValue.UNKNOWN: 0.5,
}


def triton_to_prob(ternary_value: TernaryValue) -> Bernoulli:
    """Convert a ternary truth value to a Bernoulli distribution.

    Parameters
    ----------
    ternary_value : TernaryValue
        The ternary value to convert.

    Returns
    -------
    Bernoulli
        A Bernoulli distribution encoding the ternary value as a probability:
        TRUE → Bernoulli(0.95), FALSE → Bernoulli(0.05),
        UNKNOWN → Bernoulli(0.5).

    Raises
    ------
    TypeError
        If *ternary_value* is not a :class:`TernaryValue`.
    """
    if not isinstance(ternary_value, TernaryValue):
        raise TypeError(
            f"Expected TernaryValue, got {type(ternary_value).__name__}"
        )
    return Bernoulli(_TERNARY_TO_PROB[ternary_value])


def prob_to_triton(
    dist: Bernoulli,
    true_threshold: float = 0.9,
    false_threshold: float = 0.1,
) -> TernaryValue:
    """Collapse a Bernoulli distribution to a ternary truth value.

    Parameters
    ----------
    dist : Bernoulli
        A Bernoulli distribution whose probability *p* will be compared
        against the thresholds.
    true_threshold : float, optional
        Minimum probability to classify as TRUE (default 0.9).
    false_threshold : float, optional
        Maximum probability to classify as FALSE (default 0.1).

    Returns
    -------
    TernaryValue
        TRUE if ``p >= true_threshold``, FALSE if ``p <= false_threshold``,
        UNKNOWN otherwise.

    Raises
    ------
    TypeError
        If *dist* is not a :class:`Bernoulli`.
    ValueError
        If *true_threshold* or *false_threshold* is outside [0, 1], or if
        *false_threshold* > *true_threshold*.
    """
    if not isinstance(dist, Bernoulli):
        raise TypeError(
            f"Expected Bernoulli distribution, got {type(dist).__name__}"
        )
    if not (0.0 <= true_threshold <= 1.0):
        raise ValueError(
            f"true_threshold must be in [0, 1], got {true_threshold}"
        )
    if not (0.0 <= false_threshold <= 1.0):
        raise ValueError(
            f"false_threshold must be in [0, 1], got {false_threshold}"
        )
    if false_threshold > true_threshold:
        raise ValueError(
            f"false_threshold ({false_threshold}) must not exceed "
            f"true_threshold ({true_threshold})"
        )

    if dist.p >= true_threshold:
        return TernaryValue.TRUE
    if dist.p <= false_threshold:
        return TernaryValue.FALSE
    return TernaryValue.UNKNOWN


class BinaryDist:
    """Wrapper for distributions that map cleanly to boolean logic.

    :class:`BinaryDist` wraps a :class:`Bernoulli` distribution and provides
    convenience properties for ternary conversion and boolean-like operations.

    Parameters
    ----------
    dist : Bernoulli
        The underlying Bernoulli distribution.
    true_threshold : float, optional
        Threshold for classifying as TRUE (default 0.9).
    false_threshold : float, optional
        Threshold for classifying as FALSE (default 0.1).

    Raises
    ------
    TypeError
        If *dist* is not a :class:`Bernoulli`.
    ValueError
        If thresholds are invalid.
    """

    def __init__(
        self,
        dist: Bernoulli,
        true_threshold: float = 0.9,
        false_threshold: float = 0.1,
    ) -> None:
        if not isinstance(dist, Bernoulli):
            raise TypeError(
                f"Expected Bernoulli distribution, got {type(dist).__name__}"
            )
        if not (0.0 <= true_threshold <= 1.0):
            raise ValueError(
                f"true_threshold must be in [0, 1], got {true_threshold}"
            )
        if not (0.0 <= false_threshold <= 1.0):
            raise ValueError(
                f"false_threshold must be in [0, 1], got {false_threshold}"
            )
        if false_threshold > true_threshold:
            raise ValueError(
                f"false_threshold ({false_threshold}) must not exceed "
                f"true_threshold ({true_threshold})"
            )
        self.dist = dist
        self.true_threshold = true_threshold
        self.false_threshold = false_threshold

    @classmethod
    def from_ternary(
        cls,
        ternary_value: TernaryValue,
        true_threshold: float = 0.9,
        false_threshold: float = 0.1,
    ) -> BinaryDist:
        """Create a :class:`BinaryDist` from a ternary truth value.

        Parameters
        ----------
        ternary_value : TernaryValue
            The ternary value to convert.
        true_threshold : float, optional
            Threshold for TRUE classification.
        false_threshold : float, optional
            Threshold for FALSE classification.

        Returns
        -------
        BinaryDist
        """
        dist = triton_to_prob(ternary_value)
        return cls(
            dist,
            true_threshold=true_threshold,
            false_threshold=false_threshold,
        )

    @property
    def p(self) -> float:
        """Probability of the underlying Bernoulli distribution."""
        return self.dist.p

    @property
    def ternary(self) -> TernaryValue:
        """Current ternary classification based on thresholds."""
        return prob_to_triton(
            self.dist,
            true_threshold=self.true_threshold,
            false_threshold=self.false_threshold,
        )

    def __and__(self, other: BinaryDist) -> BinaryDist:
        """Joint probability assuming independence.

        Parameters
        ----------
        other : BinaryDist
            Another BinaryDist.

        Returns
        -------
        BinaryDist
        """
        if not isinstance(other, BinaryDist):
            return NotImplemented
        return BinaryDist(
            self.dist & other.dist,
            true_threshold=self.true_threshold,
            false_threshold=self.false_threshold,
        )

    def __or__(self, other: BinaryDist) -> BinaryDist:
        """Union probability assuming independence.

        Parameters
        ----------
        other : BinaryDist
            Another BinaryDist.

        Returns
        -------
        BinaryDist
        """
        if not isinstance(other, BinaryDist):
            return NotImplemented
        return BinaryDist(
            self.dist | other.dist,
            true_threshold=self.true_threshold,
            false_threshold=self.false_threshold,
        )

    def __repr__(self) -> str:
        return (
            f"BinaryDist(p={self.dist.p}, ternary={self.ternary.value})"
        )
