"""Belief network (Bayesian network) for conditional probability storage.

A :class:`BeliefNetwork` is a directed acyclic graph where each node
holds either a marginal distribution or a :class:`ConditionalDist`
conditioned on its parents.

Example
-------
>>> from probflow.distributions.continuous import Normal
>>> from probflow.distributions.discrete import Categorical
>>> from probflow.distributions.conditional import ConditionalDist
>>> from probflow.inference.belief_network import BeliefNetwork
>>>
>>> bn = BeliefNetwork()
>>> regime = Categorical([0.6, 0.4], labels=['bull', 'bear'])
>>> bn.add_node('regime', regime)
>>> vol = ConditionalDist(
...     parent=regime,
...     mapping={'bull': Normal(1, 0.3), 'bear': Normal(2, 0.5)},
... )
>>> bn.add_node('vol', vol, parents=['regime'])
>>> samples = bn.sample(1000)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np


class BeliefNetwork:
    """Directed acyclic graph of probability distributions.

    Nodes are added in topological order (parents before children).
    Each node stores either a marginal distribution or a
    :class:`~probflow.distributions.conditional.ConditionalDist`.
    """

    def __init__(self) -> None:
        # Insertion-ordered mapping: name -> distribution
        self._nodes: OrderedDict[str, Any] = OrderedDict()
        # name -> list of parent names
        self._parents: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ #
    #  Graph construction
    # ------------------------------------------------------------------ #

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
            or a :class:`ConditionalDist`.
        parents : list of str, optional
            Names of parent nodes.  Must already exist in the network.

        Raises
        ------
        ValueError
            If *name* already exists or a parent is missing.
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

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Ancestral sampling
    # ------------------------------------------------------------------ #

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        """Ancestral (forward) sampling from the full joint distribution.

        Iterates over nodes in topological order.  Root nodes are sampled
        from their marginal distribution; child nodes are sampled from
        their conditional distribution given the already-sampled parents.

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
                # Root node – sample from marginal
                samples[name] = np.asarray(dist.sample(n))
            else:
                # Child node – pass first parent's samples
                parent_samples = samples[parents[0]]
                samples[name] = np.asarray(
                    dist.sample(n, parent_samples=parent_samples)
                )
        return samples

    def __repr__(self) -> str:
        return f"BeliefNetwork(nodes={self.nodes})"
