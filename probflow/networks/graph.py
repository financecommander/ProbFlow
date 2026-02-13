"""Network construction utilities for ProbFlow."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from probflow.core.types import Node, Variable


def build_tree(
    num_nodes: int,
    num_states: int = 2,
    seed: Optional[int] = None,
) -> Tuple[Node, List[Node]]:
    """Build a tree-structured Bayesian network.

    Returns the root node and a list of all nodes.
    Each child has a random CPT conditioned on its parent.
    """
    rng = np.random.default_rng(seed)
    states = [f"s{i}" for i in range(num_states)]
    nodes: List[Node] = []

    for i in range(num_nodes):
        var = Variable(name=f"X{i}", states=states)
        prior = None
        if i == 0:
            prior = rng.dirichlet(np.ones(num_states))
        node = Node(variable=var, prior=prior)
        nodes.append(node)

    # Build a balanced tree: node i's children are 2i+1 and 2i+2
    for i in range(num_nodes):
        for child_idx in [2 * i + 1, 2 * i + 2]:
            if child_idx < num_nodes:
                # CPT shape: (parent_states, child_states)
                cpt = np.empty((num_states, num_states))
                for s in range(num_states):
                    cpt[s] = rng.dirichlet(np.ones(num_states))
                nodes[i].add_child(nodes[child_idx], cpt)

    return nodes[0], nodes


def build_chain(
    num_nodes: int,
    num_states: int = 2,
    seed: Optional[int] = None,
) -> Tuple[Node, List[Node]]:
    """Build a chain-structured Bayesian network (Markov chain)."""
    rng = np.random.default_rng(seed)
    states = [f"s{i}" for i in range(num_states)]
    nodes: List[Node] = []

    for i in range(num_nodes):
        var = Variable(name=f"X{i}", states=states)
        prior = None
        if i == 0:
            prior = rng.dirichlet(np.ones(num_states))
        node = Node(variable=var, prior=prior)
        nodes.append(node)

    for i in range(num_nodes - 1):
        cpt = np.empty((num_states, num_states))
        for s in range(num_states):
            cpt[s] = rng.dirichlet(np.ones(num_states))
        nodes[i].add_child(nodes[i + 1], cpt)

    return nodes[0], nodes
