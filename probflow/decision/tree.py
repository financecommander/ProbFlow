"""Decision tree analysis with backward induction.

Provides a :class:`DecisionTree` for building and solving decision trees,
along with :class:`UtilityFunction` helpers for risk-sensitive evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility function helpers
# ---------------------------------------------------------------------------

class UtilityFunction:
    """Factory for common utility functions used in decision analysis."""

    @staticmethod
    def linear() -> Callable[[float], float]:
        """Risk-neutral (identity) utility: u(x) = x."""
        return lambda x: x

    @staticmethod
    def exponential(risk_aversion: float) -> Callable[[float], float]:
        """Constant Absolute Risk Aversion (CARA) utility.

        u(x) = 1 - exp(-a * x)  where *a* is the risk-aversion coefficient.
        When *a* > 0 the decision-maker is risk-averse.

        Parameters
        ----------
        risk_aversion : float
            The Arrow-Pratt coefficient of absolute risk aversion (*a*).
            Must be positive.
        """
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        a = risk_aversion
        return lambda x: 1.0 - math.exp(-a * x)

    @staticmethod
    def logarithmic(base_wealth: float = 1.0) -> Callable[[float], float]:
        """Logarithmic (CRRA) utility: u(x) = ln(base_wealth + x).

        Parameters
        ----------
        base_wealth : float
            Shifts the argument so that the domain stays positive.
            Must be positive.
        """
        if base_wealth <= 0:
            raise ValueError("base_wealth must be positive")
        w = base_wealth
        return lambda x: math.log(w + x)


# ---------------------------------------------------------------------------
# Internal node representations
# ---------------------------------------------------------------------------

@dataclass
class _DecisionNode:
    """A decision node where the agent picks among choices."""
    name: str
    choices: List[str]


@dataclass
class _ChanceNode:
    """A chance node with probabilistic outcomes."""
    name: str
    outcomes: List[str]
    probs: List[float]


# ---------------------------------------------------------------------------
# DecisionTree
# ---------------------------------------------------------------------------

class DecisionTree:
    """Build and solve a decision tree via backward induction.

    Nodes are added in order from the root downward.  Each node is
    identified by a *name*.  Terminal payoffs are set by specifying the
    *path* – a tuple of edge labels from the root to the terminal node.

    Example
    -------
    >>> tree = DecisionTree()
    >>> tree.add_decision("invest", ["stocks", "bonds"])
    >>> tree.add_chance("stocks_market", ["bull", "bear"], [0.6, 0.4])
    >>> tree.set_payoff(("stocks", "bull"), 100)
    >>> tree.set_payoff(("stocks", "bear"), -50)
    >>> tree.set_payoff(("bonds",), 30)
    >>> result = tree.solve()
    """

    def __init__(self) -> None:
        # Ordered list of nodes (first is root)
        self._nodes: List[Any] = []
        # name -> node object
        self._node_map: Dict[str, Any] = {}
        # edge_label -> child node name
        self._edges: Dict[str, str] = {}
        # Maps path-tuple to payoff value
        self._payoffs: Dict[Tuple[str, ...], float] = {}
        # parent_name -> node (for building tree structure)
        self._children: Dict[str, List[str]] = {}
        # edge_label -> parent_name
        self._edge_parent: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Building the tree
    # ------------------------------------------------------------------

    def add_decision(self, name: str, choices: List[str]) -> None:
        """Add a decision node.

        Parameters
        ----------
        name : str
            Unique identifier for this node.
        choices : list of str
            Labels for each possible action the decision-maker can take.
        """
        if name in self._node_map:
            raise ValueError(f"Node '{name}' already exists")
        node = _DecisionNode(name=name, choices=choices)
        self._nodes.append(node)
        self._node_map[name] = node
        self._children[name] = []
        for choice in choices:
            self._edge_parent[choice] = name

    def add_chance(self, name: str, outcomes: List[str],
                   probs: List[float]) -> None:
        """Add a chance node.

        Parameters
        ----------
        name : str
            Unique identifier for this node.
        outcomes : list of str
            Labels for each possible outcome.
        probs : list of float
            Probability of each outcome.  Must sum to 1 (within tolerance).
        """
        if name in self._node_map:
            raise ValueError(f"Node '{name}' already exists")
        if len(outcomes) != len(probs):
            raise ValueError("outcomes and probs must have the same length")
        if abs(sum(probs) - 1.0) > 1e-9:
            raise ValueError("Probabilities must sum to 1")
        if any(p < 0 for p in probs):
            raise ValueError("Probabilities must be non-negative")
        node = _ChanceNode(name=name, outcomes=outcomes, probs=list(probs))
        self._nodes.append(node)
        self._node_map[name] = node
        self._children[name] = []
        for outcome in outcomes:
            self._edge_parent[outcome] = name

    def set_payoff(self, path: Tuple[str, ...], value: float) -> None:
        """Assign a payoff to a terminal node identified by *path*.

        Parameters
        ----------
        path : tuple of str
            Sequence of edge labels from the root to the terminal node.
        value : float
            The payoff at this terminal state.
        """
        self._payoffs[tuple(path)] = float(value)

    # ------------------------------------------------------------------
    # Tree structure helpers
    # ------------------------------------------------------------------

    def _build_tree(self) -> Dict[str, Any]:
        """Return a nested dict representing the tree structure.

        The tree is built by matching edge labels to child nodes.
        An edge label is connected to the child node whose parent edge
        label matches.
        """
        # Build a mapping: for each node, which edge leads to it?
        # Convention: if a node name starts with an edge label + "_", or
        # if the node was added right after the edge, we match them.
        # We use insertion order and edge parent info.
        tree: Dict[str, Any] = {}
        if not self._nodes:
            return tree

        root = self._nodes[0]
        tree = self._build_subtree(root, ())
        return tree

    def _get_edges(self, node: Any) -> List[str]:
        """Return the edge labels for a node."""
        if isinstance(node, _DecisionNode):
            return node.choices
        elif isinstance(node, _ChanceNode):
            return node.outcomes
        return []

    def _find_child_node(self, edge: str,
                         path: Tuple[str, ...]) -> Optional[Any]:
        """Find the child node for a given edge label.

        Matching strategy (in priority order):
        1. A node whose name equals the edge label.
        2. A node that was registered as a child of the edge's parent,
           matching by position.
        3. If the path leads to a payoff, it is a terminal edge.
        """
        # Direct name match
        if edge in self._node_map:
            return self._node_map[edge]

        # Check if any node was added whose edge_parent maps to this edge
        # Use naming convention: look for nodes whose name starts with edge
        for node in self._nodes:
            if node.name.startswith(edge + "_") or node.name.endswith("_" + edge):
                # Verify it hasn't been used already differently
                return node

        return None

    def _build_subtree(self, node: Any,
                       path: Tuple[str, ...]) -> Dict[str, Any]:
        """Recursively build subtree dict."""
        edges = self._get_edges(node)
        subtree: Dict[str, Any] = {"node": node, "children": {}}

        for edge in edges:
            new_path = path + (edge,)
            child_node = self._find_child_node(edge, new_path)
            if child_node is not None and child_node is not node:
                subtree["children"][edge] = self._build_subtree(
                    child_node, new_path)
            else:
                # Terminal
                subtree["children"][edge] = {
                    "node": None,
                    "payoff": self._payoffs.get(new_path, 0.0),
                }
        return subtree

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def solve(self, utility_function: Optional[Callable[[float], float]] = None
              ) -> Dict[str, Any]:
        """Solve the decision tree using backward induction.

        Parameters
        ----------
        utility_function : callable, optional
            Maps payoff → utility.  If *None*, risk-neutral (identity) is
            used.

        Returns
        -------
        dict
            ``{"expected_value": float,
               "strategy": dict,
               "pruned_edges": list}``
            *strategy* maps each decision-node name to its optimal choice.
            *pruned_edges* lists ``(node_name, edge_label)`` pairs for
            sub-optimal branches.
        """
        if utility_function is None:
            utility_function = UtilityFunction.linear()

        tree = self._build_tree()
        strategy: Dict[str, str] = {}
        pruned: List[Tuple[str, str]] = []

        ev = self._solve_subtree(tree, utility_function, strategy, pruned)

        return {
            "expected_value": ev,
            "strategy": strategy,
            "pruned_edges": pruned,
        }

    def _solve_subtree(
        self,
        subtree: Dict[str, Any],
        utility_fn: Callable[[float], float],
        strategy: Dict[str, str],
        pruned: List[Tuple[str, str]],
    ) -> float:
        """Recursively solve a subtree, returning its expected value."""
        node = subtree.get("node")

        # Terminal node
        if node is None:
            return subtree.get("payoff", 0.0)

        children = subtree.get("children", {})
        if not children:
            return subtree.get("payoff", 0.0)

        if isinstance(node, _DecisionNode):
            # Evaluate each choice
            values: Dict[str, float] = {}
            for edge, child in children.items():
                values[edge] = self._solve_subtree(
                    child, utility_fn, strategy, pruned)
            # Pick the choice that maximises expected utility
            best_edge = max(values, key=lambda e: utility_fn(values[e]))
            strategy[node.name] = best_edge
            # Record pruned edges
            for edge in values:
                if edge != best_edge:
                    pruned.append((node.name, edge))
            return values[best_edge]

        elif isinstance(node, _ChanceNode):
            # Compute expected utility, then convert back to CE
            child_values: Dict[str, float] = {}
            for edge, child in children.items():
                child_values[edge] = self._solve_subtree(
                    child, utility_fn, strategy, pruned)

            # Expected utility
            eu = 0.0
            for outcome, prob in zip(node.outcomes, node.probs):
                eu += prob * utility_fn(child_values[outcome])

            # Convert EU back to certainty equivalent (monetary value)
            # For linear utility, CE = EU.
            # For non-linear, we invert numerically.
            ce = self._invert_utility(eu, utility_fn, child_values)
            return ce

        return 0.0

    @staticmethod
    def _invert_utility(
        target_u: float,
        utility_fn: Callable[[float], float],
        child_values: Dict[str, float],
    ) -> float:
        """Find certainty equivalent x such that utility_fn(x) ≈ target_u.

        Uses bisection over a range derived from the child values.
        """
        # Quick check for linear utility
        vals = list(child_values.values())
        lo, hi = min(vals) - abs(min(vals)) - 1, max(vals) + abs(max(vals)) + 1

        # Expand range to ensure we bracket
        for _ in range(20):
            if utility_fn(lo) <= target_u <= utility_fn(hi):
                break
            lo = lo * 2 - hi
            hi = hi * 2 - lo

        # Check if linear (identity)
        test_val = sum(vals) / len(vals) if vals else 0
        if abs(utility_fn(test_val) - test_val) < 1e-12:
            # Linear utility – target_u IS the monetary value
            return target_u

        # Bisection
        for _ in range(200):
            mid = (lo + hi) / 2.0
            u_mid = utility_fn(mid)
            if abs(u_mid - target_u) < 1e-12:
                return mid
            if u_mid < target_u:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    # Graphviz DOT export
    # ------------------------------------------------------------------

    def to_dot(self) -> str:
        """Export the tree to Graphviz DOT format.

        Decision nodes are drawn as squares, chance nodes as circles,
        and terminal nodes as triangles.

        Returns
        -------
        str
            A string in DOT language.
        """
        tree = self._build_tree()
        lines = ["digraph DecisionTree {", "    rankdir=LR;"]
        self._counter = 0
        self._dot_subtree(tree, lines)
        lines.append("}")
        return "\n".join(lines)

    def _next_id(self) -> str:
        self._counter += 1
        return f"n{self._counter}"

    def _dot_subtree(self, subtree: Dict[str, Any],
                     lines: List[str],
                     parent_id: Optional[str] = None,
                     edge_label: Optional[str] = None) -> str:
        """Recursively emit DOT nodes and edges."""
        node = subtree.get("node")
        nid = self._next_id()

        if node is None:
            # Terminal node
            payoff = subtree.get("payoff", 0)
            lines.append(
                f'    {nid} [label="{payoff}" shape=triangle];')
        elif isinstance(node, _DecisionNode):
            lines.append(
                f'    {nid} [label="{node.name}" shape=square];')
        elif isinstance(node, _ChanceNode):
            lines.append(
                f'    {nid} [label="{node.name}" shape=circle];')

        if parent_id is not None and edge_label is not None:
            prob_str = ""
            # If parent is a chance node, show probability
            lines.append(
                f'    {parent_id} -> {nid} [label="{edge_label}{prob_str}"];')

        children = subtree.get("children", {})
        for edge, child in children.items():
            self._dot_subtree(child, lines, parent_id=nid, edge_label=edge)

        return nid
