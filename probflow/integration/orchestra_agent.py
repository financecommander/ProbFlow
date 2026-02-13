"""Uncertainty-aware agent integration for Orchestra task scheduler.

Provides :class:`UncertaintyAwareAgent`, a mixin that enables any agent to
plan under uncertainty by generating competing hypotheses, scoring plans by
success probability, and adaptively replanning when the value of gathering
more information exceeds its cost.

Example
-------
>>> class MyAgent(UncertaintyAwareAgent):
...     def generate_hypotheses(self, goal, environment):
...         return [
...             Hypothesis("optimistic", {"demand": 100}, probability=0.3),
...             Hypothesis("moderate", {"demand": 60}, probability=0.5),
...             Hypothesis("pessimistic", {"demand": 20}, probability=0.2),
...         ]
...     def generate_plans(self, goal, environment, hypothesis):
...         return [Plan("plan_A", actions=["act1"], cost=10.0)]
...     def estimate_success_probability(self, plan, hypothesis, environment):
...         return 0.8
...     def compute_utility(self, plan, hypothesis, environment):
...         return 50.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A hypothesis about the state of the world.

    Parameters
    ----------
    name : str
        Human-readable label for this hypothesis.
    state : dict
        Key-value pairs describing assumed world state.
    probability : float
        Prior probability assigned to this hypothesis (0–1).
    """
    name: str
    state: Dict[str, Any] = field(default_factory=dict)
    probability: float = 0.0


@dataclass
class Plan:
    """A candidate plan consisting of ordered actions.

    Parameters
    ----------
    name : str
        Human-readable label.
    actions : list
        Ordered sequence of action descriptions.
    cost : float
        Estimated resource cost of executing the plan.
    """
    name: str
    actions: List[Any] = field(default_factory=list)
    cost: float = 0.0


@dataclass
class RankedPlan:
    """A plan annotated with its probability and expected utility.

    This is the element type returned by
    :meth:`UncertaintyAwareAgent.plan_under_uncertainty`.
    """
    plan: Plan
    probability: float
    expected_utility: float


# ---------------------------------------------------------------------------
# UncertaintyAwareAgent mixin
# ---------------------------------------------------------------------------

class UncertaintyAwareAgent:
    """Mixin that adds uncertainty-aware planning to any agent class.

    Subclasses must override the four hook methods:

    * :meth:`generate_hypotheses` – produce competing world-state hypotheses
    * :meth:`generate_plans` – produce candidate plans for a hypothesis
    * :meth:`estimate_success_probability` – P(success | plan, hypothesis)
    * :meth:`compute_utility` – utility of executing *plan* under *hypothesis*

    The mixin then exposes:

    * :meth:`plan_under_uncertainty` – full planning pipeline
    * :meth:`adaptive_replanning_threshold` – value-of-information test
    """

    # ------------------------------------------------------------------
    # Hooks (override in subclasses)
    # ------------------------------------------------------------------

    def generate_hypotheses(
        self, goal: Any, environment: Any
    ) -> List[Hypothesis]:
        """Return competing hypotheses about the environment.

        Parameters
        ----------
        goal : Any
            The objective the agent is trying to achieve.
        environment : Any
            Observable state of the world.

        Returns
        -------
        list of Hypothesis
            At least two hypotheses with probabilities summing to 1.
        """
        raise NotImplementedError

    def generate_plans(
        self, goal: Any, environment: Any, hypothesis: Hypothesis
    ) -> List[Plan]:
        """Return candidate plans given a hypothesis.

        Parameters
        ----------
        goal : Any
            The objective the agent is trying to achieve.
        environment : Any
            Observable state of the world.
        hypothesis : Hypothesis
            The assumed world state.

        Returns
        -------
        list of Plan
        """
        raise NotImplementedError

    def estimate_success_probability(
        self, plan: Plan, hypothesis: Hypothesis, environment: Any
    ) -> float:
        """Return P(plan succeeds | hypothesis, environment).

        Must return a value in [0, 1].
        """
        raise NotImplementedError

    def compute_utility(
        self, plan: Plan, hypothesis: Hypothesis, environment: Any
    ) -> float:
        """Return the utility of executing *plan* when *hypothesis* is true."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Core planning pipeline
    # ------------------------------------------------------------------

    def plan_under_uncertainty(
        self, goal: Any, environment: Any
    ) -> List[Dict[str, Any]]:
        """Generate and rank plans under uncertainty.

        The pipeline:

        1. Generate competing hypotheses.
        2. For every hypothesis, generate candidate plans.
        3. Score each unique plan across all hypotheses, weighting by
           hypothesis probability and success probability.
        4. Return plans ranked by expected utility (descending).

        Parameters
        ----------
        goal : Any
            The objective to achieve.
        environment : Any
            Observable world state.

        Returns
        -------
        list of dict
            Each element is ``{'plan': Plan, 'probability': float,
            'expected_utility': float}``.  The list is sorted by
            ``expected_utility`` in descending order.
        """
        hypotheses = self.generate_hypotheses(goal, environment)
        self._validate_hypotheses(hypotheses)

        # Collect unique plans and their per-hypothesis scores
        plan_scores: Dict[str, _PlanAccumulator] = {}

        for hyp in hypotheses:
            plans = self.generate_plans(goal, environment, hyp)
            for plan in plans:
                p_success = self.estimate_success_probability(
                    plan, hyp, environment
                )
                utility = self.compute_utility(plan, hyp, environment)

                if plan.name not in plan_scores:
                    plan_scores[plan.name] = _PlanAccumulator(plan)

                acc = plan_scores[plan.name]
                acc.add(
                    hyp_prob=hyp.probability,
                    success_prob=p_success,
                    utility=utility,
                )

        # Build ranked result list
        ranked: List[Dict[str, Any]] = []
        for acc in plan_scores.values():
            ranked.append({
                "plan": acc.plan,
                "probability": acc.weighted_success_probability(),
                "expected_utility": acc.expected_utility(),
            })

        ranked.sort(key=lambda r: r["expected_utility"], reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Adaptive replanning
    # ------------------------------------------------------------------

    def adaptive_replanning_threshold(
        self, plan: Dict[str, Any], *, information_cost: float = 0.0
    ) -> bool:
        """Decide whether to replan using the value of information.

        The *value of perfect information* (VPI) is estimated as:

            VPI = (1 - probability) × expected_utility

        If VPI exceeds *information_cost*, replanning is worthwhile.

        Parameters
        ----------
        plan : dict
            A ranked-plan dict as returned by :meth:`plan_under_uncertainty`.
        information_cost : float
            Cost of gathering additional information before replanning.

        Returns
        -------
        bool
            ``True`` if the agent should replan (VPI > information_cost).
        """
        probability = plan["probability"]
        expected_utility = plan["expected_utility"]

        # Value of perfect information: potential gain from resolving
        # uncertainty, proportional to remaining uncertainty and the
        # expected payoff at stake.
        vpi = (1.0 - probability) * abs(expected_utility)

        return vpi > information_cost

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_hypotheses(hypotheses: List[Hypothesis]) -> None:
        """Check that hypotheses are well-formed."""
        if len(hypotheses) < 2:
            raise ValueError(
                "At least two competing hypotheses are required"
            )
        total = sum(h.probability for h in hypotheses)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Hypothesis probabilities must sum to 1 (got {total})"
            )
        if any(h.probability < 0 for h in hypotheses):
            raise ValueError("Hypothesis probabilities must be non-negative")


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------

class _PlanAccumulator:
    """Accumulates per-hypothesis scores for a single plan."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self._entries: List[Dict[str, float]] = []

    def add(self, *, hyp_prob: float, success_prob: float,
            utility: float) -> None:
        self._entries.append({
            "hyp_prob": hyp_prob,
            "success_prob": success_prob,
            "utility": utility,
        })

    def weighted_success_probability(self) -> float:
        """Hypothesis-weighted success probability."""
        return sum(
            e["hyp_prob"] * e["success_prob"] for e in self._entries
        )

    def expected_utility(self) -> float:
        """Expected utility weighted by hypothesis probability and success."""
        return sum(
            e["hyp_prob"] * e["success_prob"] * e["utility"]
            for e in self._entries
        )


# ---------------------------------------------------------------------------
# Integration example: Orchestra task scheduler
# ---------------------------------------------------------------------------

class OrchestraSchedulerAgent(UncertaintyAwareAgent):
    """Example agent integrating with an Orchestra-style task scheduler.

    This demonstrates how to subclass :class:`UncertaintyAwareAgent` and
    wire it into a task-scheduling workflow.

    Parameters
    ----------
    task_registry : dict, optional
        Mapping of task names to callables.  Defaults to an empty dict.

    Example
    -------
    >>> agent = OrchestraSchedulerAgent()
    >>> results = agent.plan_under_uncertainty(
    ...     goal="deploy_service",
    ...     environment={"cpu_load": 0.4, "memory_free_gb": 8},
    ... )
    >>> best = results[0]
    >>> print(best["plan"].name, best["expected_utility"])
    """

    def __init__(self, task_registry: Optional[Dict[str, Any]] = None):
        self.task_registry = task_registry or {}

    def generate_hypotheses(self, goal, environment):
        """Generate hypotheses about resource availability."""
        return [
            Hypothesis(
                "high_capacity",
                {"available_resources": "high"},
                probability=0.4,
            ),
            Hypothesis(
                "medium_capacity",
                {"available_resources": "medium"},
                probability=0.4,
            ),
            Hypothesis(
                "low_capacity",
                {"available_resources": "low"},
                probability=0.2,
            ),
        ]

    def generate_plans(self, goal, environment, hypothesis):
        """Generate plans based on resource hypothesis."""
        capacity = hypothesis.state.get("available_resources", "medium")
        if capacity == "high":
            return [
                Plan("parallel_deploy", actions=["deploy_all"], cost=5.0),
                Plan("sequential_deploy", actions=["deploy_one_by_one"], cost=3.0),
            ]
        elif capacity == "medium":
            return [
                Plan("parallel_deploy", actions=["deploy_all"], cost=5.0),
                Plan("sequential_deploy", actions=["deploy_one_by_one"], cost=3.0),
            ]
        else:
            return [
                Plan("sequential_deploy", actions=["deploy_one_by_one"], cost=3.0),
                Plan("minimal_deploy", actions=["deploy_critical_only"], cost=1.0),
            ]

    def estimate_success_probability(self, plan, hypothesis, environment):
        """Estimate success based on plan type and capacity."""
        capacity = hypothesis.state.get("available_resources", "medium")
        lookup = {
            ("parallel_deploy", "high"): 0.95,
            ("parallel_deploy", "medium"): 0.70,
            ("parallel_deploy", "low"): 0.30,
            ("sequential_deploy", "high"): 0.90,
            ("sequential_deploy", "medium"): 0.85,
            ("sequential_deploy", "low"): 0.70,
            ("minimal_deploy", "high"): 0.99,
            ("minimal_deploy", "medium"): 0.95,
            ("minimal_deploy", "low"): 0.90,
        }
        return lookup.get((plan.name, capacity), 0.5)

    def compute_utility(self, plan, hypothesis, environment):
        """Compute utility as benefit minus cost."""
        benefit_map = {
            "parallel_deploy": 100.0,
            "sequential_deploy": 80.0,
            "minimal_deploy": 40.0,
        }
        benefit = benefit_map.get(plan.name, 50.0)
        return benefit - plan.cost
