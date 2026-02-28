"""Belief propagation inference for tree-structured networks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from probflow.core.types import Node


def belief_propagation(root: Node, nodes: List[Node]) -> Dict[str, np.ndarray]:
    """Run belief propagation on a tree-structured Bayesian network.

    Uses the standard two-pass algorithm:
    1. Collect messages from leaves to root (upward pass)
    2. Distribute messages from root to leaves (downward pass)

    Returns a dict mapping variable name -> marginal probability distribution.
    """
    num_nodes = len(nodes)
    # Index nodes for message storage
    node_idx = {id(n): i for i, n in enumerate(nodes)}

    # Messages: msg_up[child_idx] = message from child to parent
    # msg_down[child_idx] = message from parent to child
    msg_up: Dict[int, np.ndarray] = {}
    msg_down: Dict[int, np.ndarray] = {}

    # --- Upward pass (leaves to root) ---
    def collect(node: Node) -> np.ndarray:
        """Collect messages from children to this node."""
        idx = node_idx[id(node)]
        ns = node.variable.num_states

        if node.prior is not None:
            belief = node.prior.copy()
        else:
            belief = np.ones(ns)

        for child in node.children:
            child_msg = collect(child)
            # Marginalize: sum over child states
            # cpt[parent_state, child_state] * child_message[child_state]
            incoming = child.cpt @ child_msg  # shape: (parent_states,)
            belief *= incoming

        # Normalize to avoid underflow
        total = belief.sum()
        if total > 0:
            belief /= total

        msg_up[idx] = belief
        return belief

    collect(root)

    # --- Downward pass (root to leaves) ---
    def distribute(node: Node, parent_msg: np.ndarray) -> None:
        """Distribute messages from this node to children."""
        idx = node_idx[id(node)]
        ns = node.variable.num_states

        for child in node.children:
            child_idx = node_idx[id(child)]
            cs = child.variable.num_states

            # Message from parent to child:
            # For each child state, sum over parent states:
            # cpt[parent_state, child_state] * parent_belief[parent_state]
            # combined with incoming messages from siblings
            parent_belief = parent_msg.copy()
            for sibling in node.children:
                if id(sibling) != id(child):
                    sib_idx = node_idx[id(sibling)]
                    sib_msg = msg_up.get(sib_idx, np.ones(sibling.variable.num_states))
                    incoming = sibling.cpt @ sib_msg
                    parent_belief *= incoming

            # Compute message to child
            child_msg = child.cpt.T @ parent_belief  # shape: (child_states,)
            total = child_msg.sum()
            if total > 0:
                child_msg /= total
            msg_down[child_idx] = child_msg

            distribute(child, child_msg)

    root_belief = msg_up[node_idx[id(root)]]
    distribute(root, root_belief)

    # --- Compute marginals ---
    marginals: Dict[str, np.ndarray] = {}
    for node in nodes:
        idx = node_idx[id(node)]
        ns = node.variable.num_states

        if node is root:
            marginal = msg_up[idx].copy()
        else:
            # Combine upward and downward messages
            up = msg_up[idx]
            down = msg_down.get(idx, np.ones(ns))
            marginal = up * down

        total = marginal.sum()
        if total > 0:
            marginal /= total
        marginals[node.name] = marginal

    return marginals
