"""Conditional probability distributions.

A ``ConditionalDist`` ties a child distribution to a parent variable
so that the child's distribution depends on the realised value of the parent.
Both discrete parents (exact mapping) and continuous parents
(binned/interpolated) are supported.

Example
-------
>>> from probflow.distributions.continuous import Normal
>>> from probflow.distributions.discrete import Categorical
>>> from probflow.distributions.conditional import ConditionalDist
>>>
>>> regime = Categorical([0.6, 0.4], labels=['bull', 'bear'])
>>> vol = ConditionalDist(
...     parent=regime,
...     mapping={'bull': Normal(1, 0.3), 'bear': Normal(2, 0.5)},
... )
>>> parent_samples = regime.sample(1000)
>>> child_samples = vol.sample(1000, parent_samples=parent_samples)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class ConditionalDist:
    """Distribution whose parameters are conditioned on a parent variable.

    Parameters
    ----------
    parent : object
        The parent distribution (e.g. a ``Categorical`` or ``Normal``).
    mapping : dict
        Maps parent values to child distributions.

        * **Discrete parent** – keys are the exact parent values (strings or
          ints) and values are distribution objects with a ``sample(n)``
          method.
        * **Continuous parent** – keys are ``float`` bin edges and values are
          the distributions for each bin.  The bins are defined by
          ``(-inf, k0], (k0, k1], …, (k_{n-1}, +inf)`` where ``k0 < k1 < …``
          are the sorted keys.

    Raises
    ------
    ValueError
        If *mapping* is empty.
    TypeError
        If *mapping* is not a dict.
    """

    def __init__(self, parent: Any, mapping: Dict[Any, Any]) -> None:
        if not isinstance(mapping, dict):
            raise TypeError("mapping must be a dict")
        if not mapping:
            raise ValueError("mapping must not be empty")

        self.parent = parent
        self.mapping = mapping

        # Determine parent type: continuous if all keys are numeric
        self._continuous_parent = all(
            isinstance(k, (int, float, np.integer, np.floating))
            for k in mapping
        )

        if self._continuous_parent:
            sorted_keys = sorted(mapping.keys())
            self._bin_edges: List[float] = [float(k) for k in sorted_keys]
            self._bin_dists: list = [mapping[k] for k in sorted_keys]
        else:
            self._bin_edges = []
            self._bin_dists = []

    # --------------------------------------------------------------------- #
    #  Sampling
    # --------------------------------------------------------------------- #

    def sample(
        self,
        n: int,
        parent_samples: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw samples conditioned on the parent variable.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        parent_samples : numpy.ndarray, optional
            Array of length *n* with realisations of the parent variable.
            If ``None``, samples are drawn from ``self.parent.sample(n)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n,)`` with samples from the conditional
            distribution corresponding to each parent value.

        Raises
        ------
        ValueError
            If ``parent_samples`` length does not match *n*, or if a
            discrete parent value is not found in the mapping.
        """
        if parent_samples is None:
            parent_samples = self.parent.sample(n)

        parent_samples = np.asarray(parent_samples)
        if parent_samples.shape[0] != n:
            raise ValueError(
                f"parent_samples length ({parent_samples.shape[0]}) "
                f"must equal n ({n})"
            )

        result = np.empty(n, dtype=float)

        if self._continuous_parent:
            self._sample_continuous(parent_samples, result)
        else:
            self._sample_discrete(parent_samples, result)

        return result

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #

    def _sample_discrete(
        self, parent_samples: np.ndarray, out: np.ndarray
    ) -> None:
        """Fill *out* with samples for a discrete parent."""
        unique_values = np.unique(parent_samples)
        for val in unique_values:
            key = val.item() if hasattr(val, 'item') else val
            if key not in self.mapping:
                raise ValueError(
                    f"Parent value {key!r} not found in mapping. "
                    f"Available keys: {list(self.mapping.keys())}"
                )
            mask = parent_samples == val
            count = int(mask.sum())
            dist = self.mapping[key]
            out[mask] = dist.sample(count)

    def _sample_continuous(
        self, parent_samples: np.ndarray, out: np.ndarray
    ) -> None:
        """Fill *out* with samples for a continuous (binned) parent."""
        bin_indices = np.digitize(parent_samples, self._bin_edges)
        # np.digitize returns index i such that bins[i-1] <= x < bins[i].
        # Clamp to valid range [0, len(bin_dists) - 1].
        bin_indices = np.clip(bin_indices, 0, len(self._bin_dists) - 1)

        for idx in range(len(self._bin_dists)):
            mask = bin_indices == idx
            count = int(mask.sum())
            if count == 0:
                continue
            out[mask] = self._bin_dists[idx].sample(count)

    # --------------------------------------------------------------------- #
    #  Accessors
    # --------------------------------------------------------------------- #

    def get_child_dist(self, parent_value: Any) -> Any:
        """Return the child distribution for a specific parent value.

        For continuous parents the value is binned first.
        """
        if self._continuous_parent:
            idx = int(np.digitize(float(parent_value), self._bin_edges))
            idx = min(max(idx, 0), len(self._bin_dists) - 1)
            return self._bin_dists[idx]
        if parent_value not in self.mapping:
            raise ValueError(
                f"Parent value {parent_value!r} not found in mapping."
            )
        return self.mapping[parent_value]

    def __repr__(self) -> str:
        keys = list(self.mapping.keys())
        return f"ConditionalDist(parent={self.parent!r}, keys={keys})"
