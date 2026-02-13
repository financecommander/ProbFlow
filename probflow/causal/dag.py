"""Causal directed acyclic graph with do-calculus support.

A :class:`CausalDAG` extends :class:`BeliefNetwork` to support causal
interventions (the ``do`` operator), counterfactual reasoning via twin
networks, and identifiability checks (backdoor and frontdoor criteria).

Example
-------
>>> from probflow.distributions.discrete import Categorical
>>> from probflow.distributions.conditional import ConditionalDist
>>> from probflow.causal.dag import CausalDAG
>>>
>>> dag = CausalDAG()
>>> dag.add_node('X', Categorical([0.5, 0.5], labels=['0', '1']))
>>> dag.add_node('Y', ConditionalDist(
...     parent=dag.get_dist('X'),
...     mapping={'0': Categorical([0.8, 0.2], labels=['0', '1']),
...              '1': Categorical([0.3, 0.7], labels=['0', '1'])},
... ), parents=['X'])
>>> intervened = dag.do('X', '1')
>>> samples = intervened.sample(1000)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from probflow.inference.belief_network import BeliefNetwork


class _FixedDist:
    """A degenerate distribution that always returns a fixed value."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def sample(self, n: int = 1) -> np.ndarray:
        return np.full(n, self.value)

    def __repr__(self) -> str:
        return f"_FixedDist(value={self.value!r})"


class CausalDAG(BeliefNetwork):
    """Directed acyclic graph with causal semantics.

    Extends :class:`BeliefNetwork` with the ``do`` operator for
    interventions, counterfactual reasoning, and identifiability checks.
    """

    # ------------------------------------------------------------------ #
    #  Graph helpers
    # ------------------------------------------------------------------ #

    def children_of(self, name: str) -> List[str]:
        """Return the names of all direct children of *name*."""
        result: List[str] = []
        for node_name, parents in self._parents.items():
            if name in parents:
                result.append(node_name)
        return result

    def ancestors_of(self, name: str) -> Set[str]:
        """Return all ancestors of *name* (excluding *name* itself)."""
        visited: Set[str] = set()
        stack = list(self._parents.get(name, []))
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(self._parents.get(current, []))
        return visited

    def descendants_of(self, name: str) -> Set[str]:
        """Return all descendants of *name* (excluding *name* itself)."""
        visited: Set[str] = set()
        stack = list(self.children_of(name))
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(self.children_of(current))
        return visited

    # ------------------------------------------------------------------ #
    #  do() operator – graph surgery
    # ------------------------------------------------------------------ #

    def do(
        self,
        variable: str,
        value: Any,
    ) -> "CausalDAG":
        """Perform a causal intervention: do(*variable* = *value*).

        Graph surgery removes all incoming edges to *variable* and fixes
        its value to *value*.  Returns a **new** :class:`CausalDAG`.

        Parameters
        ----------
        variable : str
            Name of the variable to intervene on.
        value : any
            The fixed value to assign.

        Returns
        -------
        CausalDAG
            A new DAG representing the post-intervention distribution.

        Raises
        ------
        ValueError
            If *variable* is not in the network.
        """
        if variable not in self._nodes:
            raise ValueError(f"Variable '{variable}' not in the network")

        new_dag = CausalDAG()
        for name in self._nodes:
            if name == variable:
                # Replace with a fixed (degenerate) distribution;
                # remove all incoming edges (no parents).
                new_dag.add_node(name, _FixedDist(value))
            else:
                parents = self._parents[name]
                dist = self._nodes[name]
                new_dag.add_node(name, dist, parents=parents)
        return new_dag

    # ------------------------------------------------------------------ #
    #  Counterfactual reasoning – twin network method
    # ------------------------------------------------------------------ #

    def counterfactual(
        self,
        intervention: Dict[str, Any],
        evidence: Dict[str, Any],
        n: int = 10_000,
    ) -> Dict[str, np.ndarray]:
        """Estimate counterfactual quantities via the twin network method.

        The procedure:

        1. **Abduction** – Sample from the factual (observational)
           network and keep only samples consistent with *evidence*.
        2. **Action** – Build the interventional (twin) network with
           ``do()`` for each variable in *intervention*.
        3. **Prediction** – Resample the non-intervened variables in
           the twin network using the abducted (evidence-consistent)
           samples as context.

        Parameters
        ----------
        intervention : dict
            Mapping of variable names to their interventional values.
        evidence : dict
            Mapping of variable names to their observed (factual) values.
        n : int, optional
            Number of Monte Carlo samples to draw (before filtering).

        Returns
        -------
        dict of str -> numpy.ndarray
            Counterfactual samples for every variable.
        """
        # Step 1: Abduction – sample and filter by evidence
        raw_samples = self.sample(n)
        mask = np.ones(n, dtype=bool)
        for var, val in evidence.items():
            mask &= raw_samples[var] == val
        if not mask.any():
            raise ValueError(
                "No samples matched the evidence. "
                "Increase n or check evidence values."
            )

        # Build interventional network
        twin = self._copy()
        for var, val in intervention.items():
            twin = twin.do(var, val)

        # Step 3: Prediction – resample with abducted context
        # For exogenous / root nodes that are NOT intervened on,
        # reuse the abducted samples.
        count = int(mask.sum())
        cf_samples: Dict[str, np.ndarray] = {}
        for name in twin._nodes:
            if name in intervention:
                cf_samples[name] = np.full(count, intervention[name])
            else:
                parents = twin._parents[name]
                dist = twin._nodes[name]
                if not parents:
                    # Root node: reuse abducted samples
                    cf_samples[name] = raw_samples[name][mask]
                else:
                    parent_samples = cf_samples[parents[0]]
                    cf_samples[name] = np.asarray(
                        dist.sample(count, parent_samples=parent_samples)
                    )
        return cf_samples

    # ------------------------------------------------------------------ #
    #  Identifiability checks
    # ------------------------------------------------------------------ #

    def identify_effect(
        self,
        treatment: str,
        outcome: str,
    ) -> Dict[str, Any]:
        """Check whether the causal effect of *treatment* on *outcome*
        is identifiable via the backdoor or frontdoor criterion.

        Parameters
        ----------
        treatment : str
            Name of the treatment variable.
        outcome : str
            Name of the outcome variable.

        Returns
        -------
        dict
            ``identifiable`` (bool), ``method`` (str or None), and
            ``adjustment_set`` (set of str or None).

        Raises
        ------
        ValueError
            If *treatment* or *outcome* is not in the network.
        """
        if treatment not in self._nodes:
            raise ValueError(
                f"Treatment '{treatment}' not in the network"
            )
        if outcome not in self._nodes:
            raise ValueError(
                f"Outcome '{outcome}' not in the network"
            )

        # Try backdoor criterion first
        bd = self._find_backdoor_set(treatment, outcome)
        if bd is not None:
            return {
                "identifiable": True,
                "method": "backdoor",
                "adjustment_set": bd,
            }

        # Try frontdoor criterion
        fd = self._find_frontdoor_set(treatment, outcome)
        if fd is not None:
            return {
                "identifiable": True,
                "method": "frontdoor",
                "adjustment_set": fd,
            }

        return {
            "identifiable": False,
            "method": None,
            "adjustment_set": None,
        }

    # ------------------------------------------------------------------ #
    #  Confounding detection
    # ------------------------------------------------------------------ #

    def find_confounders(
        self,
        treatment: str,
        outcome: str,
    ) -> Set[str]:
        """Return the set of confounders between *treatment* and *outcome*.

        A confounder is a variable that is an ancestor of both the
        treatment and the outcome (i.e. a common cause).

        Parameters
        ----------
        treatment : str
            Name of the treatment variable.
        outcome : str
            Name of the outcome variable.

        Returns
        -------
        set of str
            Names of confounding variables.
        """
        treatment_ancestors = self.ancestors_of(treatment)
        outcome_ancestors = self.ancestors_of(outcome)
        # A confounder is an ancestor of both, OR a parent of both
        # that creates a non-causal path.
        # Standard definition: common ancestor that is not on the
        # causal path from treatment to outcome.
        causal_path = self.descendants_of(treatment)
        common_ancestors = treatment_ancestors & outcome_ancestors
        # Also include direct common parents
        treatment_parents = set(self._parents.get(treatment, []))
        outcome_parents = set(self._parents.get(outcome, []))
        direct_common = treatment_parents & outcome_parents
        confounders = (common_ancestors | direct_common) - causal_path
        # Also check: a parent of treatment that is also an ancestor
        # of outcome (but not through treatment)
        for p in treatment_parents:
            p_descendants = self.descendants_of(p)
            if outcome in p_descendants and p not in causal_path:
                confounders.add(p)
        return confounders

    # ------------------------------------------------------------------ #
    #  Interventional sampling helpers
    # ------------------------------------------------------------------ #

    def interventional_sample(
        self,
        interventions: Dict[str, Any],
        n: int = 10_000,
    ) -> Dict[str, np.ndarray]:
        """Sample from the interventional distribution.

        Convenience method equivalent to chaining ``do()`` calls.

        Parameters
        ----------
        interventions : dict
            Mapping of variable names to their interventional values.
        n : int
            Number of samples.

        Returns
        -------
        dict of str -> numpy.ndarray
        """
        dag = self._copy()
        for var, val in interventions.items():
            dag = dag.do(var, val)
        return dag.sample(n)

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _copy(self) -> "CausalDAG":
        """Create a shallow copy of this DAG."""
        new_dag = CausalDAG()
        for name in self._nodes:
            new_dag._nodes[name] = self._nodes[name]
            new_dag._parents[name] = list(self._parents[name])
        return new_dag

    def _find_backdoor_set(
        self,
        treatment: str,
        outcome: str,
    ) -> Optional[Set[str]]:
        """Find a valid backdoor adjustment set, or None.

        A set Z satisfies the backdoor criterion relative to
        (treatment, outcome) if:
        1. No node in Z is a descendant of treatment.
        2. Z blocks every path between treatment and outcome that
           contains an arrow into treatment (i.e. "backdoor paths").

        We use the simple strategy of checking whether the set of
        all non-descendants of treatment that are parents of treatment
        (and their ancestors) constitutes a valid adjustment set.
        """
        descendants_of_treatment = self.descendants_of(treatment)
        # Candidate set: all parents of treatment (and their ancestors)
        # that are not descendants of treatment.
        treatment_parents = set(self._parents.get(treatment, []))
        candidate = set()
        for p in treatment_parents:
            if p not in descendants_of_treatment:
                candidate.add(p)
                # Add ancestors of p that are also not descendants
                for a in self.ancestors_of(p):
                    if a not in descendants_of_treatment:
                        candidate.add(a)

        # Verify: the candidate set must block all backdoor paths.
        # A simple check: if there are parents of treatment that are
        # also ancestors/parents of outcome, they must be in the set.
        if not candidate and not treatment_parents:
            # No parents of treatment → no backdoor paths → identifiable
            return set()

        if not candidate and treatment_parents:
            # Treatment has parents but none qualify → check if they
            # are related to outcome at all
            any_confounding = False
            for p in treatment_parents:
                p_descendants = self.descendants_of(p)
                if outcome in p_descendants or p == outcome:
                    any_confounding = True
                    break
                p_ancestors = self.ancestors_of(outcome)
                if p in p_ancestors:
                    any_confounding = True
                    break
            if not any_confounding:
                return set()
            return None

        # Verify candidate blocks backdoor paths using d-separation
        # approximation: check that conditioning on the candidate set
        # blocks all non-causal paths from treatment to outcome.
        if self._blocks_backdoor_paths(treatment, outcome, candidate):
            return candidate

        return None

    def _blocks_backdoor_paths(
        self,
        treatment: str,
        outcome: str,
        conditioning_set: Set[str],
    ) -> bool:
        """Check if conditioning_set blocks all backdoor paths.

        Uses a simplified reachability check: in the moral graph
        (after removing outgoing edges from treatment), check if
        treatment and outcome are d-separated given the conditioning set.
        """
        # Build adjacency (undirected) excluding edges FROM treatment
        adj: Dict[str, Set[str]] = {n: set() for n in self._nodes}
        for name, parents in self._parents.items():
            for p in parents:
                if name == treatment and p in self._parents.get(treatment, []):
                    # This is a backdoor edge – keep it
                    adj[p].add(name)
                    adj[name].add(p)
                elif p == treatment:
                    # Outgoing edge from treatment – skip for backdoor
                    continue
                else:
                    adj[p].add(name)
                    adj[name].add(p)

        # BFS from treatment to outcome, avoiding conditioning set
        visited: Set[str] = set()
        queue = [treatment]
        visited.add(treatment)
        while queue:
            current = queue.pop(0)
            if current == outcome:
                return False  # Path found → not blocked
            for neighbor in adj.get(current, set()):
                if neighbor not in visited and neighbor not in conditioning_set:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return True  # No path found → blocked

    def _find_frontdoor_set(
        self,
        treatment: str,
        outcome: str,
    ) -> Optional[Set[str]]:
        """Find a valid frontdoor adjustment set, or None.

        A set Z satisfies the frontdoor criterion relative to
        (treatment, outcome) if:
        1. Treatment blocks all paths from Z to outcome that go
           through treatment.  (Z is "intercepted" by treatment.)
        2. There is no unblocked backdoor path from treatment to Z.
        3. All backdoor paths from Z to outcome are blocked by treatment.

        We look for mediator variables that lie on all directed paths
        from treatment to outcome.
        """
        # Find all directed paths from treatment to outcome
        all_paths = self._find_all_directed_paths(treatment, outcome)
        if not all_paths:
            return None

        # Candidate mediators: nodes on every path from treatment to outcome
        # (excluding treatment and outcome themselves)
        mediators: Optional[Set[str]] = None
        for path in all_paths:
            path_nodes = set(path[1:-1])  # exclude endpoints
            if mediators is None:
                mediators = path_nodes
            else:
                mediators &= path_nodes

        if mediators is None or not mediators:
            return None

        # Verify frontdoor conditions:
        # 1. All directed paths from treatment to outcome go through
        #    at least one mediator
        for path in all_paths:
            path_interior = set(path[1:-1])
            if not path_interior & mediators:
                return None

        # 2. No backdoor path from treatment to mediators
        # (i.e., no common cause of treatment and mediator
        #  that is not blocked)
        for m in mediators:
            # Mediator should only be reachable from treatment
            m_ancestors = self.ancestors_of(m)
            treatment_ancestors = self.ancestors_of(treatment)
            # Check no unblocked backdoor between treatment and mediator:
            # a common ancestor of both that is NOT treatment itself
            common = (m_ancestors & treatment_ancestors) - {treatment}
            if common:
                return None

        return mediators

    def _find_all_directed_paths(
        self,
        start: str,
        end: str,
    ) -> List[List[str]]:
        """Find all directed paths from *start* to *end* via DFS."""
        paths: List[List[str]] = []

        def _dfs(current: str, path: List[str]) -> None:
            if current == end:
                paths.append(list(path))
                return
            for child in self.children_of(current):
                if child not in path:  # avoid cycles
                    path.append(child)
                    _dfs(child, path)
                    path.pop()

        _dfs(start, [start])
        return paths

    def __repr__(self) -> str:
        return f"CausalDAG(nodes={self.nodes})"
