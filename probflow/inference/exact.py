"""Exact inference for discrete Bayesian networks.

Provides:

* :class:`FactorTable` – a factor (CPD table) with multiply and
  marginalize operations.
* :func:`belief_propagation` – exact inference for tree-structured
  (polytree) graphs via the two-pass collect/distribute algorithm.
* :func:`variable_elimination` – exact inference for arbitrary DAGs
  using the elimination algorithm.

Both functions accept *query* and *evidence* arguments and return a
normalized probability table over the query variable(s).

If the graph contains undirected loops, :func:`belief_propagation`
raises a :class:`ValueError`; callers should fall back to
:func:`variable_elimination` or sampling-based inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from probflow.core.types import Node, Variable

# Numerical stability floor for log-space operations
_LOG_EPS = 1e-300


# ------------------------------------------------------------------ #
#  FactorTable
# ------------------------------------------------------------------ #

class FactorTable:
    """A discrete factor (potential function) over a set of variables.

    Parameters
    ----------
    variables : list of str
        Variable names that index the axes of *values*.
    cardinalities : list of int
        Number of states for each variable (same order as *variables*).
    values : numpy.ndarray
        An N-dimensional array whose shape equals *cardinalities*.
    """

    def __init__(
        self,
        variables: List[str],
        cardinalities: List[int],
        values: np.ndarray,
    ) -> None:
        expected = tuple(cardinalities)
        if values.shape != expected:
            raise ValueError(
                f"FactorTable shape {values.shape} does not match "
                f"cardinalities {expected}"
            )
        self.variables: List[str] = list(variables)
        self.cardinalities: List[int] = list(cardinalities)
        self.values: np.ndarray = values.astype(np.float64, copy=False)

    # ----- factory helpers ------------------------------------------------

    @classmethod
    def from_node(cls, node: Node) -> "FactorTable":
        """Build a :class:`FactorTable` from a network :class:`Node`.

        * Root nodes (no parents) yield a 1-D factor from their prior.
        * Child nodes yield a factor from their CPT whose axes are
          ``[parent_0, ..., parent_k, self]``.
        """
        if not node.parents:
            prior = node.prior if node.prior is not None else (
                np.ones(node.variable.num_states) / node.variable.num_states
            )
            return cls(
                [node.name],
                [node.variable.num_states],
                prior.copy(),
            )

        var_names = [p.name for p in node.parents] + [node.name]
        cards = [p.variable.num_states for p in node.parents] + [
            node.variable.num_states
        ]
        return cls(var_names, cards, node.cpt.copy())

    # ----- core operations ------------------------------------------------

    def multiply(self, other: "FactorTable") -> "FactorTable":
        """Point-wise multiplication of two factors.

        Shared variables are aligned; non-shared variables are broadcast.
        Returns a new :class:`FactorTable`.
        """
        # Build the combined variable list
        combined_vars: List[str] = list(self.variables)
        combined_cards: List[int] = list(self.cardinalities)
        for v, c in zip(other.variables, other.cardinalities):
            if v not in combined_vars:
                combined_vars.append(v)
                combined_cards.append(c)

        # Reshape self for broadcasting
        a = self._broadcast_into(combined_vars)
        b = other._broadcast_into(combined_vars)
        return FactorTable(combined_vars, combined_cards, a * b)

    def marginalize(self, var: str) -> "FactorTable":
        """Sum out (marginalize) *var* from this factor.

        Returns a new :class:`FactorTable` with one fewer dimension.
        """
        if var not in self.variables:
            raise ValueError(f"Variable '{var}' not in factor")
        axis = self.variables.index(var)
        new_vars = [v for v in self.variables if v != var]
        new_cards = [c for v, c in zip(self.variables, self.cardinalities)
                     if v != var]
        new_values = self.values.sum(axis=axis)
        return FactorTable(new_vars, new_cards, new_values)

    def reduce(self, var: str, state_idx: int) -> "FactorTable":
        """Condition on *var* = *state_idx* (slice the factor).

        Returns a new :class:`FactorTable` without *var*.
        """
        if var not in self.variables:
            raise ValueError(f"Variable '{var}' not in factor")
        axis = self.variables.index(var)
        slices = [slice(None)] * len(self.variables)
        slices[axis] = state_idx
        new_values = self.values[tuple(slices)]
        new_vars = [v for v in self.variables if v != var]
        new_cards = [c for v, c in zip(self.variables, self.cardinalities)
                     if v != var]
        return FactorTable(new_vars, new_cards, np.asarray(new_values))

    def normalize(self) -> "FactorTable":
        """Return a copy normalized so that all entries sum to 1."""
        total = self.values.sum()
        if total > 0:
            return FactorTable(
                list(self.variables),
                list(self.cardinalities),
                self.values / total,
            )
        return FactorTable(
            list(self.variables),
            list(self.cardinalities),
            self.values.copy(),
        )

    # ----- helpers --------------------------------------------------------

    def _broadcast_into(self, target_vars: List[str]) -> np.ndarray:
        """Reshape values so axes align with *target_vars* (size-1 for missing)."""
        shape = []
        for tv in target_vars:
            if tv in self.variables:
                idx = self.variables.index(tv)
                shape.append(self.cardinalities[idx])
            else:
                shape.append(1)

        # Build index map: for each target axis, which self axis?
        # Then transpose self's data into the right order and reshape.
        src_axes = [self.variables.index(tv) for tv in target_vars
                    if tv in self.variables]
        extra_axes = [i for i, tv in enumerate(target_vars)
                      if tv not in self.variables]

        transposed = np.transpose(self.values, src_axes)
        # Expand dims for missing variables
        for ea in extra_axes:
            transposed = np.expand_dims(transposed, axis=ea)
        return transposed

    def __repr__(self) -> str:
        return (
            f"FactorTable(variables={self.variables}, "
            f"shape={self.values.shape})"
        )


# ------------------------------------------------------------------ #
#  Loop detection
# ------------------------------------------------------------------ #

def _has_undirected_cycle(nodes: List[Node]) -> bool:
    """Return True if the *undirected* skeleton of the DAG has a cycle.

    A tree with *n* nodes has exactly *n - 1* edges.  If the undirected
    edge count is >= *n*, the graph contains a cycle.
    """
    edges: Set[Tuple[str, str]] = set()
    for node in nodes:
        for child in node.children:
            a, b = node.name, child.name
            edge = (min(a, b), max(a, b))
            edges.add(edge)
    return len(edges) >= len(nodes)


# ------------------------------------------------------------------ #
#  Belief propagation (two-pass, tree-structured only)
# ------------------------------------------------------------------ #

def belief_propagation(
    root: Node,
    nodes: List[Node],
    query: Optional[str] = None,
    evidence: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Exact inference on a tree-structured Bayesian network.

    Uses the standard two-pass collect/distribute algorithm:

    1. **Collect** (leaves -> root): each node sends a message to its
       parent summarising its subtree.
    2. **Distribute** (root -> leaves): each node sends a message to
       its children incorporating the rest of the tree.

    Parameters
    ----------
    root : Node
        Root of the tree.
    nodes : list of Node
        All nodes in the network.
    query : str, optional
        If given, only the marginal for this variable is returned.
    evidence : dict mapping variable name -> state index, optional
        Hard evidence to condition on.

    Returns
    -------
    dict of str -> numpy.ndarray
        Marginal probability distributions for each (or just the query)
        variable.

    Raises
    ------
    ValueError
        If the undirected skeleton of the graph contains a cycle.
    """
    if _has_undirected_cycle(nodes):
        raise ValueError(
            "Graph contains undirected cycles; belief propagation "
            "requires a tree (polytree). Use variable_elimination() "
            "or sampling-based inference instead."
        )

    evidence = evidence or {}
    node_idx = {id(n): i for i, n in enumerate(nodes)}
    name_map = {n.name: n for n in nodes}

    msg_up: Dict[int, np.ndarray] = {}
    msg_down: Dict[int, np.ndarray] = {}

    # ---------- collect (upward pass) --------------------------------- #

    def _collect(node: Node) -> np.ndarray:
        idx = node_idx[id(node)]
        ns = node.variable.num_states

        if node.prior is not None:
            belief = node.prior.copy()
        else:
            belief = np.ones(ns)

        # Incorporate evidence
        if node.name in evidence:
            ev = np.zeros(ns)
            ev[evidence[node.name]] = 1.0
            belief *= ev

        for child in node.children:
            child_msg = _collect(child)
            incoming = child.cpt @ child_msg  # (parent_states,)
            belief *= incoming

        total = belief.sum()
        if total > 0:
            belief /= total

        msg_up[idx] = belief
        return belief

    _collect(root)

    # ---------- distribute (downward pass) ----------------------------- #

    def _distribute(node: Node, parent_msg: np.ndarray) -> None:
        idx = node_idx[id(node)]

        for child in node.children:
            child_idx = node_idx[id(child)]

            parent_belief = parent_msg.copy()
            for sibling in node.children:
                if id(sibling) != id(child):
                    sib_idx = node_idx[id(sibling)]
                    sib_msg = msg_up.get(
                        sib_idx,
                        np.ones(sibling.variable.num_states),
                    )
                    incoming = sibling.cpt @ sib_msg
                    parent_belief *= incoming

            child_msg = child.cpt.T @ parent_belief
            total = child_msg.sum()
            if total > 0:
                child_msg /= total
            msg_down[child_idx] = child_msg

            _distribute(child, child_msg)

    root_belief = msg_up[node_idx[id(root)]]
    _distribute(root, root_belief)

    # ---------- marginals --------------------------------------------- #

    marginals: Dict[str, np.ndarray] = {}
    for node in nodes:
        idx = node_idx[id(node)]
        ns = node.variable.num_states

        if node is root:
            marginal = msg_up[idx].copy()
        else:
            up = msg_up[idx]
            down = msg_down.get(idx, np.ones(ns))
            marginal = up * down

        total = marginal.sum()
        if total > 0:
            marginal /= total
        marginals[node.name] = marginal

    if query is not None:
        return {query: marginals[query]}

    return marginals


# ------------------------------------------------------------------ #
#  Variable elimination (arbitrary DAGs)
# ------------------------------------------------------------------ #

def variable_elimination(
    root: Node,
    nodes: List[Node],
    query: Optional[str] = None,
    evidence: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Exact inference on an arbitrary discrete Bayesian network.

    Eliminates hidden variables one at a time by multiplying all
    factors that mention the variable, then marginalising it out.

    Parameters
    ----------
    root : Node
        Root of the DAG (used for topological ordering).
    nodes : list of Node
        All nodes in the network.
    query : str, optional
        If given, only the marginal for this variable is returned.
        Other hidden variables are eliminated.
    evidence : dict mapping variable name -> state index, optional
        Hard evidence.

    Returns
    -------
    dict of str -> numpy.ndarray
        Marginal probability distributions.
    """
    evidence = evidence or {}
    query_vars: Set[str] = set()
    if query is not None:
        query_vars.add(query)
    else:
        query_vars = {n.name for n in nodes}

    # 1. Build initial factor list from CPTs
    factors: List[FactorTable] = []
    for node in nodes:
        factors.append(FactorTable.from_node(node))

    # 2. Apply evidence by reducing factors
    for var_name, state_idx in evidence.items():
        new_factors: List[FactorTable] = []
        for f in factors:
            if var_name in f.variables:
                reduced = f.reduce(var_name, state_idx)
                # Only keep factors that still have variables
                if reduced.variables:
                    new_factors.append(reduced)
                else:
                    # Scalar factor – wrap it as a 0-d placeholder
                    # so its weight is not lost
                    new_factors.append(reduced)
            else:
                new_factors.append(f)
        factors = new_factors

    # 3. Determine elimination order (all vars not in query and not evidence)
    all_vars = set()
    for f in factors:
        all_vars.update(f.variables)
    elim_vars = [v for v in all_vars if v not in query_vars and v not in evidence]

    # 4. Eliminate each hidden variable
    for var in elim_vars:
        # Collect factors that mention this variable
        involved: List[FactorTable] = []
        remaining: List[FactorTable] = []
        for f in factors:
            if var in f.variables:
                involved.append(f)
            else:
                remaining.append(f)

        if not involved:
            continue

        # Multiply all involved factors
        product = involved[0]
        for f in involved[1:]:
            product = product.multiply(f)

        # Marginalise out the variable
        new_factor = product.marginalize(var)
        remaining.append(new_factor)
        factors = remaining

    # 5. Multiply remaining factors
    if not factors:
        return {}

    result = factors[0]
    for f in factors[1:]:
        result = result.multiply(f)

    # 6. Normalise and extract marginals
    marginals: Dict[str, np.ndarray] = {}

    if query is not None and query in result.variables:
        # Marginalise out everything except query
        for v in list(result.variables):
            if v != query:
                result = result.marginalize(v)
        result = result.normalize()
        marginals[query] = result.values.ravel()
    else:
        # Return marginals for all remaining variables
        for var_name in list(result.variables):
            temp = result
            for v in list(temp.variables):
                if v != var_name:
                    temp = temp.marginalize(v)
            temp = temp.normalize()
            marginals[var_name] = temp.values.ravel()

    return marginals
