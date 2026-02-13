"""Tests for probflow.integration.orchestra_agent module."""

import math
import pytest

from probflow.integration.orchestra_agent import (
    Hypothesis,
    Plan,
    UncertaintyAwareAgent,
    OrchestraSchedulerAgent,
)


# -----------------------------------------------------------------------
# Helpers: concrete agent for testing
# -----------------------------------------------------------------------

class SimpleTestAgent(UncertaintyAwareAgent):
    """Minimal agent for deterministic testing."""

    def __init__(self, hypotheses, plans_per_hyp, success_probs, utilities):
        self._hypotheses = hypotheses
        self._plans_per_hyp = plans_per_hyp
        self._success_probs = success_probs
        self._utilities = utilities

    def generate_hypotheses(self, goal, environment):
        return self._hypotheses

    def generate_plans(self, goal, environment, hypothesis):
        return self._plans_per_hyp.get(hypothesis.name, [])

    def estimate_success_probability(self, plan, hypothesis, environment):
        return self._success_probs.get((plan.name, hypothesis.name), 0.5)

    def compute_utility(self, plan, hypothesis, environment):
        return self._utilities.get((plan.name, hypothesis.name), 0.0)


# -----------------------------------------------------------------------
# Multi-hypothesis generation
# -----------------------------------------------------------------------

class TestMultiHypothesisGeneration:
    def test_requires_at_least_two_hypotheses(self):
        """Planning must reject fewer than two hypotheses."""
        class BadAgent(UncertaintyAwareAgent):
            def generate_hypotheses(self, goal, env):
                return [Hypothesis("only_one", {}, probability=1.0)]
            def generate_plans(self, goal, env, hyp):
                return []
            def estimate_success_probability(self, plan, hyp, env):
                return 1.0
            def compute_utility(self, plan, hyp, env):
                return 0.0

        agent = BadAgent()
        with pytest.raises(ValueError, match="At least two"):
            agent.plan_under_uncertainty("goal", {})

    def test_hypotheses_must_sum_to_one(self):
        """Hypothesis probabilities must sum to 1."""
        class BadAgent(UncertaintyAwareAgent):
            def generate_hypotheses(self, goal, env):
                return [
                    Hypothesis("a", {}, probability=0.3),
                    Hypothesis("b", {}, probability=0.3),
                ]
            def generate_plans(self, goal, env, hyp):
                return []
            def estimate_success_probability(self, plan, hyp, env):
                return 1.0
            def compute_utility(self, plan, hyp, env):
                return 0.0

        agent = BadAgent()
        with pytest.raises(ValueError, match="sum to 1"):
            agent.plan_under_uncertainty("goal", {})

    def test_negative_probability_rejected(self):
        """Negative hypothesis probabilities must be rejected."""
        class BadAgent(UncertaintyAwareAgent):
            def generate_hypotheses(self, goal, env):
                return [
                    Hypothesis("a", {}, probability=-0.5),
                    Hypothesis("b", {}, probability=1.5),
                ]
            def generate_plans(self, goal, env, hyp):
                return []
            def estimate_success_probability(self, plan, hyp, env):
                return 1.0
            def compute_utility(self, plan, hyp, env):
                return 0.0

        agent = BadAgent()
        with pytest.raises(ValueError, match="non-negative"):
            agent.plan_under_uncertainty("goal", {})

    def test_multiple_hypotheses_propagated(self):
        """All hypotheses contribute to plan scoring."""
        hypotheses = [
            Hypothesis("optimistic", {"demand": 100}, probability=0.6),
            Hypothesis("pessimistic", {"demand": 20}, probability=0.4),
        ]
        plan = Plan("sell", actions=["sell_widgets"], cost=5.0)
        agent = SimpleTestAgent(
            hypotheses=hypotheses,
            plans_per_hyp={"optimistic": [plan], "pessimistic": [plan]},
            success_probs={
                ("sell", "optimistic"): 0.9,
                ("sell", "pessimistic"): 0.3,
            },
            utilities={
                ("sell", "optimistic"): 80.0,
                ("sell", "pessimistic"): 10.0,
            },
        )
        results = agent.plan_under_uncertainty("sell_stuff", {})
        assert len(results) == 1
        # Weighted probability: 0.6*0.9 + 0.4*0.3 = 0.54 + 0.12 = 0.66
        assert results[0]["probability"] == pytest.approx(0.66)

    def test_three_hypotheses(self):
        """Three-hypothesis scenario works correctly."""
        hypotheses = [
            Hypothesis("high", {}, probability=0.3),
            Hypothesis("mid", {}, probability=0.5),
            Hypothesis("low", {}, probability=0.2),
        ]
        plan = Plan("act")
        agent = SimpleTestAgent(
            hypotheses=hypotheses,
            plans_per_hyp={
                "high": [plan], "mid": [plan], "low": [plan],
            },
            success_probs={
                ("act", "high"): 1.0,
                ("act", "mid"): 0.5,
                ("act", "low"): 0.0,
            },
            utilities={
                ("act", "high"): 100.0,
                ("act", "mid"): 50.0,
                ("act", "low"): 0.0,
            },
        )
        results = agent.plan_under_uncertainty("goal", {})
        assert len(results) == 1
        # Weighted prob: 0.3*1.0 + 0.5*0.5 + 0.2*0.0 = 0.55
        assert results[0]["probability"] == pytest.approx(0.55)


# -----------------------------------------------------------------------
# Utility calculation
# -----------------------------------------------------------------------

class TestUtilityCalculation:
    def test_expected_utility_calculation(self):
        """Expected utility = sum(hyp_prob * success_prob * utility)."""
        hypotheses = [
            Hypothesis("good", {}, probability=0.7),
            Hypothesis("bad", {}, probability=0.3),
        ]
        plan = Plan("invest")
        agent = SimpleTestAgent(
            hypotheses=hypotheses,
            plans_per_hyp={"good": [plan], "bad": [plan]},
            success_probs={
                ("invest", "good"): 0.8,
                ("invest", "bad"): 0.4,
            },
            utilities={
                ("invest", "good"): 100.0,
                ("invest", "bad"): -20.0,
            },
        )
        results = agent.plan_under_uncertainty("goal", {})
        # EU = 0.7*0.8*100 + 0.3*0.4*(-20) = 56 + (-2.4) = 53.6
        assert results[0]["expected_utility"] == pytest.approx(53.6)

    def test_plans_ranked_by_expected_utility(self):
        """Plans are returned in descending expected-utility order."""
        hypotheses = [
            Hypothesis("h1", {}, probability=0.5),
            Hypothesis("h2", {}, probability=0.5),
        ]
        plan_a = Plan("plan_a", cost=0)
        plan_b = Plan("plan_b", cost=0)
        agent = SimpleTestAgent(
            hypotheses=hypotheses,
            plans_per_hyp={
                "h1": [plan_a, plan_b],
                "h2": [plan_a, plan_b],
            },
            success_probs={
                ("plan_a", "h1"): 0.6,
                ("plan_a", "h2"): 0.4,
                ("plan_b", "h1"): 0.9,
                ("plan_b", "h2"): 0.8,
            },
            utilities={
                ("plan_a", "h1"): 50.0,
                ("plan_a", "h2"): 50.0,
                ("plan_b", "h1"): 30.0,
                ("plan_b", "h2"): 30.0,
            },
        )
        results = agent.plan_under_uncertainty("goal", {})
        assert len(results) == 2
        # plan_a EU = 0.5*0.6*50 + 0.5*0.4*50 = 15 + 10 = 25
        # plan_b EU = 0.5*0.9*30 + 0.5*0.8*30 = 13.5 + 12 = 25.5
        assert results[0]["plan"].name == "plan_b"
        assert results[1]["plan"].name == "plan_a"
        assert results[0]["expected_utility"] == pytest.approx(25.5)
        assert results[1]["expected_utility"] == pytest.approx(25.0)

    def test_zero_utility(self):
        """Plans with zero utility are handled correctly."""
        hypotheses = [
            Hypothesis("h1", {}, probability=0.5),
            Hypothesis("h2", {}, probability=0.5),
        ]
        plan = Plan("noop")
        agent = SimpleTestAgent(
            hypotheses=hypotheses,
            plans_per_hyp={"h1": [plan], "h2": [plan]},
            success_probs={("noop", "h1"): 1.0, ("noop", "h2"): 1.0},
            utilities={("noop", "h1"): 0.0, ("noop", "h2"): 0.0},
        )
        results = agent.plan_under_uncertainty("goal", {})
        assert results[0]["expected_utility"] == pytest.approx(0.0)

    def test_result_dict_keys(self):
        """Each result must have 'plan', 'probability', 'expected_utility'."""
        hypotheses = [
            Hypothesis("a", {}, probability=0.5),
            Hypothesis("b", {}, probability=0.5),
        ]
        plan = Plan("p")
        agent = SimpleTestAgent(
            hypotheses=hypotheses,
            plans_per_hyp={"a": [plan], "b": [plan]},
            success_probs={("p", "a"): 0.5, ("p", "b"): 0.5},
            utilities={("p", "a"): 10.0, ("p", "b"): 10.0},
        )
        results = agent.plan_under_uncertainty("goal", {})
        for r in results:
            assert "plan" in r
            assert "probability" in r
            assert "expected_utility" in r
            assert isinstance(r["plan"], Plan)
            assert isinstance(r["probability"], float)
            assert isinstance(r["expected_utility"], float)


# -----------------------------------------------------------------------
# Adaptive replanning threshold
# -----------------------------------------------------------------------

class TestThresholdTriggering:
    def test_low_probability_triggers_replan(self):
        """Low success probability means high VPI → replan."""
        plan = {"plan": Plan("risky"), "probability": 0.3, "expected_utility": 100.0}
        agent = SimpleTestAgent([], {}, {}, {})
        # VPI = (1 - 0.3) * 100 = 70 > 0 → True
        assert agent.adaptive_replanning_threshold(plan) is True

    def test_high_probability_no_replan(self):
        """High success probability means low VPI → no replan when cost high."""
        plan = {"plan": Plan("safe"), "probability": 0.95, "expected_utility": 100.0}
        agent = SimpleTestAgent([], {}, {}, {})
        # VPI = (1 - 0.95) * 100 = 5
        assert agent.adaptive_replanning_threshold(
            plan, information_cost=10.0
        ) is False

    def test_threshold_with_zero_utility(self):
        """Zero expected utility → VPI = 0 → no replan."""
        plan = {"plan": Plan("noop"), "probability": 0.5, "expected_utility": 0.0}
        agent = SimpleTestAgent([], {}, {}, {})
        assert agent.adaptive_replanning_threshold(plan) is False

    def test_threshold_exact_boundary(self):
        """VPI exactly equals cost → no replan (strict inequality)."""
        plan = {"plan": Plan("border"), "probability": 0.5, "expected_utility": 20.0}
        agent = SimpleTestAgent([], {}, {}, {})
        # VPI = (1 - 0.5) * 20 = 10
        assert agent.adaptive_replanning_threshold(
            plan, information_cost=10.0
        ) is False

    def test_threshold_just_above_cost(self):
        """VPI just above cost → replan."""
        plan = {"plan": Plan("edge"), "probability": 0.5, "expected_utility": 20.0}
        agent = SimpleTestAgent([], {}, {}, {})
        # VPI = 10
        assert agent.adaptive_replanning_threshold(
            plan, information_cost=9.99
        ) is True

    def test_threshold_negative_utility(self):
        """Negative utility: VPI uses abs(expected_utility)."""
        plan = {"plan": Plan("loss"), "probability": 0.4, "expected_utility": -50.0}
        agent = SimpleTestAgent([], {}, {}, {})
        # VPI = (1 - 0.4) * 50 = 30 > 0
        assert agent.adaptive_replanning_threshold(plan) is True

    def test_certain_plan_no_replan(self):
        """Probability=1.0 → VPI=0 → no replan."""
        plan = {"plan": Plan("certain"), "probability": 1.0, "expected_utility": 100.0}
        agent = SimpleTestAgent([], {}, {}, {})
        assert agent.adaptive_replanning_threshold(plan) is False


# -----------------------------------------------------------------------
# Orchestra scheduler integration example
# -----------------------------------------------------------------------

class TestOrchestraSchedulerAgent:
    def test_generates_three_hypotheses(self):
        agent = OrchestraSchedulerAgent()
        hyps = agent.generate_hypotheses("deploy", {})
        assert len(hyps) == 3
        total = sum(h.probability for h in hyps)
        assert total == pytest.approx(1.0)

    def test_plan_under_uncertainty_returns_ranked_list(self):
        agent = OrchestraSchedulerAgent()
        results = agent.plan_under_uncertainty("deploy_service", {})
        assert len(results) >= 2
        # Verify descending order by expected utility
        for i in range(len(results) - 1):
            assert results[i]["expected_utility"] >= results[i + 1]["expected_utility"]

    def test_all_plans_have_positive_probability(self):
        agent = OrchestraSchedulerAgent()
        results = agent.plan_under_uncertainty("deploy_service", {})
        for r in results:
            assert r["probability"] > 0

    def test_replanning_threshold_on_best_plan(self):
        agent = OrchestraSchedulerAgent()
        results = agent.plan_under_uncertainty("deploy_service", {})
        best = results[0]
        # Best plan should still have some uncertainty
        assert best["probability"] < 1.0
        # With zero cost, VPI > 0 → should replan
        assert agent.adaptive_replanning_threshold(best) is True

    def test_replanning_threshold_with_high_cost(self):
        agent = OrchestraSchedulerAgent()
        results = agent.plan_under_uncertainty("deploy_service", {})
        best = results[0]
        # With extremely high information cost, should not replan
        assert agent.adaptive_replanning_threshold(
            best, information_cost=1e6
        ) is False
