"""Monte Carlo sampling for probabilistic inference."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from probflow.core.types import Node


def forward_sample(
    root: Node,
    nodes: List[Node],
    num_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Forward (ancestral) sampling from a Bayesian network.

    Samples are generated in topological order (root first).
    Returns an array of shape (num_samples, num_nodes) with state indices.
    """
    rng = np.random.default_rng(seed)
    num_nodes = len(nodes)
    node_idx = {id(n): i for i, n in enumerate(nodes)}

    samples = np.empty((num_samples, num_nodes), dtype=np.int32)

    # Topological order: BFS from root
    order: List[Node] = []
    queue = [root]
    visited = {id(root)}
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in node.children:
            if id(child) not in visited:
                visited.add(id(child))
                queue.append(child)

    for node in order:
        idx = node_idx[id(node)]
        ns = node.variable.num_states

        if not node.parents:
            # Root node: sample from prior
            prior = node.prior if node.prior is not None else np.ones(ns) / ns
            samples[:, idx] = rng.choice(ns, size=num_samples, p=prior)
        else:
            # Child node: sample from CPT conditioned on parent
            parent = node.parents[0]
            parent_idx = node_idx[id(parent)]
            parent_states = samples[:, parent_idx]

            # Vectorized sampling: for each parent state, sample child state
            for ps in range(parent.variable.num_states):
                mask = parent_states == ps
                count = mask.sum()
                if count > 0:
                    probs = node.cpt[ps]
                    samples[mask, idx] = rng.choice(ns, size=count, p=probs)

    return samples


def monte_carlo_marginals(
    root: Node,
    nodes: List[Node],
    num_samples: int,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Estimate marginal distributions using Monte Carlo sampling."""
    samples = forward_sample(root, nodes, num_samples, seed=seed)
    marginals: Dict[str, np.ndarray] = {}

    for i, node in enumerate(nodes):
        ns = node.variable.num_states
        counts = np.bincount(samples[:, i], minlength=ns).astype(float)
        marginals[node.name] = counts / num_samples

    return marginals
