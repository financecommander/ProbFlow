"""Tests for probflow.decision.tree module."""

import math
import pytest

from probflow.decision.tree import DecisionTree, UtilityFunction


# -----------------------------------------------------------------------
# Utility function tests
# -----------------------------------------------------------------------

class TestUtilityFunction:
    def test_linear(self):
        u = UtilityFunction.linear()
        assert u(0) == 0
        assert u(100) == 100
        assert u(-50) == -50

    def test_exponential(self):
        u = UtilityFunction.exponential(risk_aversion=0.01)
        assert u(0) == pytest.approx(0.0)
        # Monotonically increasing
        assert u(100) > u(50) > u(0) > u(-50)
        # Known formula: 1 - exp(-a*x)
        assert u(100) == pytest.approx(1 - math.exp(-0.01 * 100))

    def test_exponential_invalid(self):
        with pytest.raises(ValueError):
            UtilityFunction.exponential(risk_aversion=0)
        with pytest.raises(ValueError):
            UtilityFunction.exponential(risk_aversion=-1)

    def test_logarithmic(self):
        u = UtilityFunction.logarithmic(base_wealth=100)
        assert u(0) == pytest.approx(math.log(100))
        assert u(50) == pytest.approx(math.log(150))

    def test_logarithmic_invalid(self):
        with pytest.raises(ValueError):
            UtilityFunction.logarithmic(base_wealth=0)
        with pytest.raises(ValueError):
            UtilityFunction.logarithmic(base_wealth=-10)


# -----------------------------------------------------------------------
# Simple tree construction and solving
# -----------------------------------------------------------------------

class TestSimpleTree:
    def test_single_decision_with_payoffs(self):
        """Decision between two deterministic outcomes."""
        tree = DecisionTree()
        tree.add_decision("choice", ["A", "B"])
        tree.set_payoff(("A",), 100)
        tree.set_payoff(("B",), 50)
        result = tree.solve()
        assert result["expected_value"] == pytest.approx(100)
        assert result["strategy"]["choice"] == "A"

    def test_decision_with_chance(self):
        """Decision followed by a chance node on one branch."""
        tree = DecisionTree()
        tree.add_decision("invest", ["stocks", "bonds"])
        tree.add_chance("stocks", ["bull", "bear"], [0.6, 0.4])
        tree.set_payoff(("stocks", "bull"), 100)
        tree.set_payoff(("stocks", "bear"), -50)
        tree.set_payoff(("bonds",), 30)
        result = tree.solve()
        # EV(stocks) = 0.6*100 + 0.4*(-50) = 60 - 20 = 40
        assert result["expected_value"] == pytest.approx(40)
        assert result["strategy"]["invest"] == "stocks"

    def test_chance_node_probabilities_validation(self):
        tree = DecisionTree()
        with pytest.raises(ValueError):
            tree.add_chance("bad", ["a", "b"], [0.5, 0.6])
        with pytest.raises(ValueError):
            tree.add_chance("bad2", ["a", "b"], [0.5])
        with pytest.raises(ValueError):
            tree.add_chance("bad3", ["a", "b"], [-0.1, 1.1])

    def test_duplicate_node_name(self):
        tree = DecisionTree()
        tree.add_decision("d1", ["a", "b"])
        with pytest.raises(ValueError):
            tree.add_decision("d1", ["c", "d"])


# -----------------------------------------------------------------------
# Canonical example: newsvendor
# -----------------------------------------------------------------------

class TestNewsvendor:
    """Classic newsvendor problem as a decision tree.

    A newsvendor must decide how many papers to stock.
    - Cost per paper: $0.50, Selling price: $1.00
    - Demand: Low (30) w/ p=0.3, Medium (50) w/ p=0.5, High (70) w/ p=0.2
    - Stock options: 30, 50, 70
    """

    def _build(self):
        tree = DecisionTree()
        tree.add_decision("stock", ["stock_30", "stock_50", "stock_70"])

        # Chance nodes for each stocking level
        probs = [0.3, 0.5, 0.2]
        outcomes = ["low", "medium", "high"]
        demands = {"low": 30, "medium": 50, "high": 70}
        cost = 0.50
        price = 1.00

        for stock_label, stock_qty in [("stock_30", 30), ("stock_50", 50),
                                       ("stock_70", 70)]:
            tree.add_chance(stock_label, outcomes, probs)
            for outcome in outcomes:
                demand = demands[outcome]
                sold = min(stock_qty, demand)
                payoff = sold * price - stock_qty * cost
                tree.set_payoff((stock_label, outcome), payoff)

        return tree

    def test_newsvendor_risk_neutral(self):
        tree = self._build()
        result = tree.solve()

        # Calculate expected payoffs:
        # stock_30: sold=min(30,d) for d in {30,50,70} → always 30
        #   payoff = 30*1 - 30*0.5 = 15 for all demands → EV = 15
        # stock_50: sold=min(50,d)
        #   low:  30*1 - 50*0.5 = 5
        #   med:  50*1 - 50*0.5 = 25
        #   high: 50*1 - 50*0.5 = 25
        #   EV = 0.3*5 + 0.5*25 + 0.2*25 = 1.5 + 12.5 + 5.0 = 19.0
        # stock_70: sold=min(70,d)
        #   low:  30*1 - 70*0.5 = -5
        #   med:  50*1 - 70*0.5 = 15
        #   high: 70*1 - 70*0.5 = 35
        #   EV = 0.3*(-5) + 0.5*15 + 0.2*35 = -1.5 + 7.5 + 7.0 = 13.0
        assert result["strategy"]["stock"] == "stock_50"
        assert result["expected_value"] == pytest.approx(19.0)

    def test_newsvendor_risk_averse(self):
        """With high risk aversion, the safe option (stock_30) is chosen."""
        tree = self._build()
        # Use very high risk aversion to prefer the safe stock_30 (guaranteed 15)
        u = UtilityFunction.exponential(risk_aversion=0.5)
        result = tree.solve(utility_function=u)
        # With enough risk aversion, the guaranteed 15 beats risky 19 EV
        assert result["strategy"]["stock"] == "stock_30"


# -----------------------------------------------------------------------
# Canonical example: oil drilling
# -----------------------------------------------------------------------

class TestOilDrilling:
    """Oil drilling decision with an optional seismic test.

    Simplified version:
    - Decision: drill or don't drill
    - If drill: chance of oil (p=0.5, payoff=500) or dry (p=0.5, payoff=-200)
    - Don't drill: payoff=0
    """

    def test_basic_drill_decision(self):
        tree = DecisionTree()
        tree.add_decision("drill_decision", ["drill", "no_drill"])
        tree.add_chance("drill", ["oil", "dry"], [0.5, 0.5])
        tree.set_payoff(("drill", "oil"), 500)
        tree.set_payoff(("drill", "dry"), -200)
        tree.set_payoff(("no_drill",), 0)

        result = tree.solve()
        # EV(drill) = 0.5*500 + 0.5*(-200) = 150
        assert result["expected_value"] == pytest.approx(150)
        assert result["strategy"]["drill_decision"] == "drill"

    def test_drill_risk_averse(self):
        """Risk-averse decision-maker may choose not to drill."""
        tree = DecisionTree()
        tree.add_decision("drill_decision", ["drill", "no_drill"])
        tree.add_chance("drill", ["oil", "dry"], [0.5, 0.5])
        tree.set_payoff(("drill", "oil"), 500)
        tree.set_payoff(("drill", "dry"), -200)
        tree.set_payoff(("no_drill",), 0)

        # With very high risk aversion, don't drill is safer
        u = UtilityFunction.exponential(risk_aversion=0.02)
        result = tree.solve(utility_function=u)
        assert result["strategy"]["drill_decision"] == "no_drill"


# -----------------------------------------------------------------------
# Risk aversion impact
# -----------------------------------------------------------------------

class TestRiskAversionImpact:
    def test_increasing_risk_aversion_decreases_risky_choice(self):
        """As risk aversion increases, the certainty equivalent of a
        risky gamble decreases."""
        tree = DecisionTree()
        tree.add_decision("choose", ["gamble", "safe"])
        tree.add_chance("gamble", ["win", "lose"], [0.5, 0.5])
        tree.set_payoff(("gamble", "win"), 200)
        tree.set_payoff(("gamble", "lose"), 0)
        tree.set_payoff(("safe",), 80)

        # Risk neutral: EV(gamble) = 100 > 80 → gamble
        result_neutral = tree.solve()
        assert result_neutral["strategy"]["choose"] == "gamble"

        # Moderate risk aversion: might still gamble or switch
        # High risk aversion: switch to safe
        u_high = UtilityFunction.exponential(risk_aversion=0.03)
        result_averse = tree.solve(utility_function=u_high)
        assert result_averse["strategy"]["choose"] == "safe"

    def test_log_utility_risk_aversion(self):
        """Logarithmic utility is risk-averse."""
        tree = DecisionTree()
        tree.add_decision("choose", ["gamble", "safe"])
        tree.add_chance("gamble", ["win", "lose"], [0.5, 0.5])
        tree.set_payoff(("gamble", "win"), 200)
        tree.set_payoff(("gamble", "lose"), 0)
        tree.set_payoff(("safe",), 80)

        u_log = UtilityFunction.logarithmic(base_wealth=50)
        result = tree.solve(utility_function=u_log)
        # Log utility is risk-averse; CE of gamble < 100
        # With base_wealth=50: E[ln(50+x)] = 0.5*ln(250) + 0.5*ln(50)
        # ln(250)≈5.52, ln(50)≈3.91 → E[u] ≈ 4.715
        # CE: ln(50+CE) = 4.715 → CE = exp(4.715)-50 ≈ 61.8
        # 61.8 < 80 → safe is chosen
        assert result["strategy"]["choose"] == "safe"


# -----------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------

class TestPruning:
    def test_pruned_edges_reported(self):
        """Pruned (sub-optimal) edges should be in the result."""
        tree = DecisionTree()
        tree.add_decision("root", ["A", "B", "C"])
        tree.set_payoff(("A",), 100)
        tree.set_payoff(("B",), 50)
        tree.set_payoff(("C",), 75)

        result = tree.solve()
        assert result["strategy"]["root"] == "A"
        pruned = result["pruned_edges"]
        assert ("root", "B") in pruned
        assert ("root", "C") in pruned
        assert len(pruned) == 2

    def test_no_pruning_at_chance_nodes(self):
        """Chance nodes never prune — all outcomes are considered."""
        tree = DecisionTree()
        tree.add_decision("d", ["go"])
        tree.add_chance("go", ["good", "bad"], [0.7, 0.3])
        tree.set_payoff(("go", "good"), 100)
        tree.set_payoff(("go", "bad"), -10)

        result = tree.solve()
        # Only one decision choice, so nothing pruned at decision
        assert result["pruned_edges"] == []
        # EV = 0.7*100 + 0.3*(-10) = 67
        assert result["expected_value"] == pytest.approx(67.0)


# -----------------------------------------------------------------------
# DOT export
# -----------------------------------------------------------------------

class TestDotExport:
    def test_dot_contains_structure(self):
        tree = DecisionTree()
        tree.add_decision("invest", ["stocks", "bonds"])
        tree.add_chance("stocks", ["bull", "bear"], [0.6, 0.4])
        tree.set_payoff(("stocks", "bull"), 100)
        tree.set_payoff(("stocks", "bear"), -50)
        tree.set_payoff(("bonds",), 30)

        dot = tree.to_dot()
        assert "digraph DecisionTree" in dot
        assert "invest" in dot
        assert "stocks" in dot
        assert "shape=square" in dot  # decision node
        assert "shape=circle" in dot  # chance node
        assert "shape=triangle" in dot  # terminal node
        assert "->" in dot  # edges

    def test_dot_empty_tree(self):
        tree = DecisionTree()
        dot = tree.to_dot()
        assert "digraph DecisionTree" in dot
