"""Directed acyclic graph (DAG) based Bayesian network.

Provides :class:`BeliefNetwork`, a high-level interface for building
and querying discrete Bayesian networks.  The graph structure is stored
in a :class:`networkx.DiGraph`; conditional probability distributions
(CPDs) are stored in a dictionary keyed by node name.

Inference methods:

* :meth:`BeliefNetwork.marginal` – unconditional marginal via variable
  elimination.
* :meth:`BeliefNetwork.infer` – conditional inference using belief
  propagation (tree-structured graphs) with automatic fallback to
  variable elimination for loopy graphs.
* :meth:`BeliefNetwork.d_separated` – d-separation test using
  networkx path algorithms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np

from probflow.core.types import Node, Variable
from probflow.inference.exact import (
    _has_undirected_cycle,
    belief_propagation,
    variable_elimination,
)


class BeliefNetwork:
    """Bayesian network backed by a :class:`networkx.DiGraph`.

    Each node has a name, a list of discrete states, and either a
    prior distribution (root nodes) or a conditional probability
    table (child nodes).  Evidence can be set on any node with
    :meth:`observe`, and queries answered with :meth:`marginal`
    and :meth:`infer`.

    Parameters
    ----------
    None

    Examples
    --------
    >>> bn = BeliefNetwork()
    >>> bn.add_node("A", np.array([0.4, 0.6]), states=["a0", "a1"])
    >>> bn.add_node("B", np.array([[0.9, 0.1], [0.3, 0.7]]),
    ...             parents=["A"], states=["b0", "b1"])
    >>> bn.marginal("B")
    array([...])
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        # name -> numpy array (prior for roots, CPT for children)
        self._cpds: Dict[str, np.ndarray] = {}
        # name -> list of state labels
        self._states: Dict[str, List[str]] = {}
        # name -> state index (observed evidence)
        self._evidence: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  Graph construction
    # ------------------------------------------------------------------ #

    def add_node(
        self,
        name: str,
        distribution: np.ndarray,
        parents: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
    ) -> None:
        """Add a node to the network.

        Parameters
        ----------
        name : str
            Unique identifier for this variable.
        distribution : numpy.ndarray
            For root nodes (no parents): a 1-D array of prior
            probabilities.  For child nodes: a CPT of shape
            ``(parent_states, self_states)`` (single parent) or
            ``(p1_states, ..., pk_states, self_states)`` for
            multiple parents.
        parents : list of str, optional
            Names of parent nodes.  Must already exist in the network.
        states : list of str, optional
            Labels for this variable's states.  If *None*, defaults to
            ``["s0", "s1", ...]``.

        Raises
        ------
        ValueError
            If *name* already exists or a parent is missing.
        """
        if name in self._graph:
            raise ValueError(f"Node '{name}' already exists")

        parents = parents or []
        for p in parents:
            if p not in self._graph:
                raise ValueError(
                    f"Parent '{p}' must be added before child '{name}'"
                )

        dist = np.asarray(distribution, dtype=np.float64)

        # Determine number of states
        if parents:
            num_states = dist.shape[-1]
        else:
            num_states = dist.shape[0]

        if states is None:
            states = [f"s{i}" for i in range(num_states)]

        if len(states) != num_states:
            raise ValueError(
                f"Number of states ({len(states)}) does not match "
                f"distribution shape ({num_states})"
            )

        self._graph.add_node(name)
        for p in parents:
            self._graph.add_edge(p, name)

        self._cpds[name] = dist
        self._states[name] = states

    # ------------------------------------------------------------------ #
    #  Evidence
    # ------------------------------------------------------------------ #

    def observe(self, variable: str, evidence: object) -> None:
        """Set observed evidence on a variable.

        Parameters
        ----------
        variable : str
            Name of the observed variable.
        evidence : str or int
            The observed state (label string) or state index (int).

        Raises
        ------
        ValueError
            If the variable does not exist or the evidence value is
            invalid.
        """
        if variable not in self._graph:
            raise ValueError(f"Variable '{variable}' not in network")

        if isinstance(evidence, (int, np.integer)):
            idx = int(evidence)
        else:
            states = self._states[variable]
            if evidence not in states:
                raise ValueError(
                    f"'{evidence}' is not a valid state of '{variable}'. "
                    f"Valid states: {states}"
                )
            idx = states.index(evidence)

        self._evidence[variable] = idx

    def clear_evidence(self) -> None:
        """Remove all observed evidence."""
        self._evidence.clear()

    # ------------------------------------------------------------------ #
    #  Internal: build Node objects for the inference engine
    # ------------------------------------------------------------------ #

    def _build_nodes(self) -> tuple[Node, list[Node]]:
        """Convert the networkx graph into :class:`Node` objects.

        Returns ``(root, nodes)`` where *root* is the first
        topologically-ordered root and *nodes* is the full list.
        """
        topo = list(nx.topological_sort(self._graph))
        node_map: Dict[str, Node] = {}

        for name in topo:
            var = Variable(name=name, states=list(self._states[name]))
            parents = list(self._graph.predecessors(name))

            if not parents:
                node = Node(variable=var, prior=self._cpds[name].copy())
            else:
                node = Node(variable=var)

            node_map[name] = node

        # Wire parent-child relationships and set CPTs
        for name in topo:
            parents = list(self._graph.predecessors(name))
            if parents:
                child = node_map[name]
                for p in parents:
                    parent_node = node_map[p]
                    parent_node.children.append(child)
                    child.parents.append(parent_node)
                child.cpt = self._cpds[name].copy()

        nodes = [node_map[name] for name in topo]
        root = nodes[0]
        return root, nodes

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def marginal(self, query: str) -> np.ndarray:
        """Return the unconditional marginal P(query).

        Uses the variable elimination algorithm.

        Parameters
        ----------
        query : str
            Name of the query variable.

        Returns
        -------
        numpy.ndarray
            Probability distribution over the states of *query*.
        """
        if query not in self._graph:
            raise ValueError(f"Variable '{query}' not in network")

        root, nodes = self._build_nodes()
        result = variable_elimination(root, nodes, query=query)
        return result[query]

    def infer(self, query: str) -> np.ndarray:
        """Return P(query | evidence) using current evidence.

        For tree-structured graphs, uses belief propagation.  If the
        graph contains undirected cycles (is *not* a polytree), falls
        back to variable elimination.

        Parameters
        ----------
        query : str
            Name of the query variable.

        Returns
        -------
        numpy.ndarray
            Posterior probability distribution over the states of *query*.
        """
        if query not in self._graph:
            raise ValueError(f"Variable '{query}' not in network")

        root, nodes = self._build_nodes()
        evidence = dict(self._evidence) if self._evidence else None

        # Try belief propagation first (works only on trees)
        if not _has_undirected_cycle(nodes):
            result = belief_propagation(
                root, nodes, query=query, evidence=evidence
            )
            return result[query]

        # Fallback to variable elimination for loopy graphs
        result = variable_elimination(
            root, nodes, query=query, evidence=evidence
        )
        return result[query]

    # ------------------------------------------------------------------ #
    #  D-separation
    # ------------------------------------------------------------------ #

    def d_separated(
        self,
        x: str,
        y: str,
        z: Optional[Set[str]] = None,
    ) -> bool:
        """Check whether *x* and *y* are d-separated given *z*.

        Uses the Bayes-Ball algorithm via networkx's
        :func:`~networkx.algorithms.d_separation.d_separated`.

        Parameters
        ----------
        x : str
            First variable.
        y : str
            Second variable.
        z : set of str, optional
            Conditioning set.  If *None*, tests unconditional
            independence.

        Returns
        -------
        bool
            True if *x* ⊥ *y* | *z* in the DAG.

        Raises
        ------
        ValueError
            If any variable is not in the network.
        """
        for v in [x, y]:
            if v not in self._graph:
                raise ValueError(f"Variable '{v}' not in network")
        z = z or set()
        for v in z:
            if v not in self._graph:
                raise ValueError(f"Variable '{v}' not in network")

        return nx.is_d_separator(self._graph, {x}, {y}, z)

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    @property
    def nodes(self) -> List[str]:
        """Return node names in topological order."""
        return list(nx.topological_sort(self._graph))

    @property
    def edges(self) -> List[tuple[str, str]]:
        """Return directed edges as (parent, child) tuples."""
        return list(self._graph.edges())

    @property
    def evidence(self) -> Dict[str, int]:
        """Return current evidence as {variable: state_index}."""
        return dict(self._evidence)

    def get_states(self, name: str) -> List[str]:
        """Return the state labels for a variable."""
        return list(self._states[name])

    def __repr__(self) -> str:
        return (
            f"BeliefNetwork(nodes={list(self._graph.nodes)}, "
            f"edges={list(self._graph.edges)})"
        )
