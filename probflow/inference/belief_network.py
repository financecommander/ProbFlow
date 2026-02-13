"""Belief network (Bayesian network) for conditional probability storage.

A :class:`BeliefNetwork` is a directed acyclic graph where each node
holds either a marginal distribution or a conditional distribution
conditioned on its parents.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np


class BeliefNetwork:
    """Directed acyclic graph of probability distributions.

    Nodes are added in topological order (parents before children).
    Each node stores either a marginal distribution or a conditional
    distribution.
    """

    def __init__(self) -> None:
        self._nodes: OrderedDict[str, Any] = OrderedDict()
        self._parents: Dict[str, List[str]] = {}

    def add_node(
        self,
        name: str,
        dist: Any,
        parents: Optional[List[str]] = None,
    ) -> None:
        """Add a node to the network.

        Parameters
        ----------
        name : str
            Unique identifier for this variable.
        dist : distribution object
            Either a marginal distribution (with a ``sample(n)`` method)
            or a conditional distribution.
        parents : list of str, optional
            Names of parent nodes. Must already exist in the network.
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        parents = parents or []
        for p in parents:
            if p not in self._nodes:
                raise ValueError(
                    f"Parent '{p}' must be added before child '{name}'"
                )
        self._nodes[name] = dist
        self._parents[name] = parents

    @property
    def nodes(self) -> List[str]:
        """Return node names in topological (insertion) order."""
        return list(self._nodes.keys())

    def get_dist(self, name: str) -> Any:
        """Return the distribution object for *name*."""
        return self._nodes[name]

    def get_parents(self, name: str) -> List[str]:
        """Return parent names for *name*."""
        return list(self._parents.get(name, []))

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        """Ancestral (forward) sampling from the full joint distribution.

        Parameters
        ----------
        n : int
            Number of joint samples to draw.

        Returns
        -------
        dict of str -> numpy.ndarray
            Maps each node name to an array of shape ``(n,)``.
        """
        samples: Dict[str, np.ndarray] = {}
        for name in self._nodes:
            dist = self._nodes[name]
            parents = self._parents[name]
            if not parents:
                samples[name] = np.asarray(dist.sample(n))
            else:
                parent_samples = samples[parents[0]]
                samples[name] = np.asarray(
                    dist.sample(n, parent_samples=parent_samples)
                )
        return samples

    def __repr__(self) -> str:
        return f"BeliefNetwork(nodes={self.nodes})"
