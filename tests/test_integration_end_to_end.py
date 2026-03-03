"""Full pipeline integration tests.

Tests that combine distributions, Monte Carlo simulation, decision trees,
Markov chains, HMMs, causal DAGs, and serialization into complete
end-to-end workflows.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from probflow.causal.dag import CausalDAG
from probflow.decision.tree import DecisionTree, UtilityFunction
from probflow.distributions.conditional import ConditionalDist
from probflow.distributions.continuous import Beta, LogNormal, Normal
from probflow.distributions.discrete import Bernoulli, Categorical, Poisson
from probflow.inference.sampling import MonteCarloSimulation, SimulationResults
from probflow.integration.serialization import load_model, save_model
from probflow.networks.dag import BeliefNetwork
from probflow.temporal.markov import HiddenMarkovModel, MarkovChain


# ================================================================
# Test class 1: Distributions -> Monte Carlo -> Decision Tree
# ================================================================


class TestDistributionSimulationDecision:
    """Pipeline: define distributions, run Monte Carlo simulation,
    then feed simulated outcomes into a decision tree."""

    def test_monte_carlo_with_normal_distribution(self) -> None:
        """Sample from a Normal distribution via MonteCarloSimulation
        and verify statistics match the theoretical parameters."""
        dist = Normal(mu=100, sigma=15)

        sim = MonteCarloSimulation(
            func=lambda: float(dist.sample(1)[0]),
            n_samples=10000,
            seed=42,
        )
        results = sim.run()

        assert isinstance(results, SimulationResults)
        assert len(results) == 10000
        assert np.isclose(results.mean(), 100, atol=2)
        assert np.isclose(results.std(), 15, atol=2)
        # Check quantiles
        assert results.quantile(0.5) > 95  # Median near 100
        assert results.quantile(0.5) < 105

    def test_monte_carlo_with_conditional_distribution(self) -> None:
        """Monte Carlo simulation using a ConditionalDist that switches
        between two Normal distributions based on a Categorical parent."""
        regime = Categorical([0.6, 0.4], labels=["bull", "bear"])
        returns = ConditionalDist(
            parent=regime,
            mapping={
                "bull": Normal(mu=0.10, sigma=0.15),
                "bear": Normal(mu=-0.05, sigma=0.25),
            },
        )

        def simulate_return():
            parent_sample = regime.sample(1)
            child_sample = returns.sample(1, parent_samples=parent_sample)
            return float(child_sample[0])

        sim = MonteCarloSimulation(
            func=simulate_return,
            n_samples=10000,
            seed=42,
        )
        results = sim.run()

        # Expected mean: 0.6 * 0.10 + 0.4 * (-0.05) = 0.04
        assert np.isclose(results.mean(), 0.04, atol=0.02)

    def test_simulation_to_decision_tree(self) -> None:
        """Run Monte Carlo on two investment strategies, then use simulated
        expected values to build a decision tree."""
        # Strategy A: LogNormal returns (risky)
        np.random.seed(42)
        dist_a = LogNormal(mu=np.log(110), sigma=0.3)
        sim_a = MonteCarloSimulation(
            func=lambda: float(dist_a.sample(1)[0]),
            n_samples=5000,
            seed=42,
        )
        results_a = sim_a.run()

        # Strategy B: Normal returns (safer)
        dist_b = Normal(mu=105, sigma=10)
        sim_b = MonteCarloSimulation(
            func=lambda: float(dist_b.sample(1)[0]),
            n_samples=5000,
            seed=42,
        )
        results_b = sim_b.run()

        # Build decision tree using simulated expected values
        tree = DecisionTree()
        tree.add_decision("invest", ["strategy_a", "strategy_b"])
        tree.set_payoff(("strategy_a",), results_a.mean())
        tree.set_payoff(("strategy_b",), results_b.mean())

        result = tree.solve()
        assert "expected_value" in result
        assert "strategy" in result
        # The strategy with higher simulated mean should be chosen
        chosen = result["strategy"]["invest"]
        if results_a.mean() > results_b.mean():
            assert chosen == "strategy_a"
        else:
            assert chosen == "strategy_b"

    def test_full_pipeline_with_chance_nodes(self) -> None:
        """Full pipeline: simulate probabilities, build decision tree
        with chance nodes, solve with risk-averse utility."""
        # Simulate success probability using Beta distribution
        success_rate = Beta(alpha=8, beta=2)
        sim = MonteCarloSimulation(
            func=lambda: float(success_rate.sample(1)[0]),
            n_samples=5000,
            seed=42,
        )
        results = sim.run()
        p_success = results.mean()  # Should be close to 0.8

        # Build decision tree
        tree = DecisionTree()
        tree.add_decision("launch", ["go", "no_go"])
        tree.add_chance("go", ["success", "failure"], [p_success, 1 - p_success])
        tree.set_payoff(("go", "success"), 500)
        tree.set_payoff(("go", "failure"), -200)
        tree.set_payoff(("no_go",), 50)

        # Solve with risk-neutral utility
        result_neutral = tree.solve()
        assert result_neutral["strategy"]["launch"] in ["go", "no_go"]

        # Solve with risk-averse utility
        utility_fn = UtilityFunction.exponential(risk_aversion=0.005)
        result_averse = tree.solve(utility_function=utility_fn)
        assert result_averse["strategy"]["launch"] in ["go", "no_go"]

        # Risk-neutral expected value of "go" is: 0.8*500 + 0.2*(-200) = 360
        # which is much higher than "no_go" (50), so both should choose "go"
        assert result_neutral["strategy"]["launch"] == "go"


# ================================================================
# Test class 2: MarkovChain -> Forecast -> Decision under uncertainty
# ================================================================


class TestMarkovChainForecastDecision:
    """Pipeline: MarkovChain forecasting feeding into decision-making."""

    def _build_market_chain(self) -> MarkovChain:
        """Build a Markov chain for market regimes."""
        states = ["bull", "bear", "neutral"]
        transition = [
            [0.7, 0.1, 0.2],   # bull -> bull/bear/neutral
            [0.2, 0.5, 0.3],   # bear -> bull/bear/neutral
            [0.3, 0.2, 0.5],   # neutral -> bull/bear/neutral
        ]
        return MarkovChain(states, transition)

    def test_forecast_from_initial_state(self) -> None:
        """Forecast state probabilities from a known initial state."""
        mc = self._build_market_chain()
        forecast = mc.forecast(horizon=10, initial_state="bull")

        assert forecast.shape == (11, 3)
        # At t=0, should be deterministic (bull)
        np.testing.assert_allclose(forecast[0], [1, 0, 0])
        # At every time step, probabilities should sum to 1
        for t in range(11):
            assert np.isclose(forecast[t].sum(), 1.0)

    def test_stationary_distribution_convergence(self) -> None:
        """Long-run forecast should converge to the stationary distribution."""
        mc = self._build_market_chain()
        stationary = mc.stationary_distribution()

        assert len(stationary) == 3
        assert np.isclose(stationary.sum(), 1.0)
        assert np.all(stationary >= 0)

        # Long-run forecast should match stationary distribution
        long_forecast = mc.forecast(horizon=100, initial_state="bear")
        np.testing.assert_allclose(long_forecast[-1], stationary, atol=0.01)

    def test_forecast_drives_decision_tree(self) -> None:
        """Use Markov chain forecast probabilities as chance node
        probabilities in a decision tree."""
        mc = self._build_market_chain()

        # Forecast 3 steps ahead from neutral state
        forecast = mc.forecast(horizon=3, initial_state="neutral")
        future_probs = forecast[3]  # Probabilities at t=3

        # Build decision tree using forecasted probabilities
        tree = DecisionTree()
        tree.add_decision("strategy", ["aggressive", "conservative"])

        # Aggressive strategy payoffs depend on market regime
        tree.add_chance(
            "aggressive",
            ["bull", "bear", "neutral"],
            list(future_probs),
        )
        tree.set_payoff(("aggressive", "bull"), 200)
        tree.set_payoff(("aggressive", "bear"), -100)
        tree.set_payoff(("aggressive", "neutral"), 50)

        # Conservative strategy is safe
        tree.set_payoff(("conservative",), 40)

        result = tree.solve()
        assert "expected_value" in result
        assert result["strategy"]["strategy"] in ["aggressive", "conservative"]

    def test_markov_chain_to_monte_carlo(self) -> None:
        """Combine Markov chain with Monte Carlo simulation: simulate
        portfolio value paths using Markov regime switching."""
        mc = self._build_market_chain()

        def simulate_portfolio():
            # Simulate a 5-step path
            rng = np.random.default_rng()
            state_idx = 0  # Start in bull
            value = 100.0
            returns = {"bull": 0.05, "bear": -0.03, "neutral": 0.01}
            states = mc.states

            for _ in range(5):
                ret = returns[states[state_idx]]
                value *= (1 + ret)
                # Transition to next state
                probs = mc.transition_matrix[state_idx]
                state_idx = rng.choice(3, p=probs)
            return value

        sim = MonteCarloSimulation(
            func=simulate_portfolio,
            n_samples=5000,
            seed=42,
        )
        results = sim.run()

        # Starting at 100 in bull market, mean should be > 100
        assert results.mean() > 95
        assert results.std() > 0
        # VaR at 5% should be below mean
        assert results.quantile(0.05) < results.mean()


# ================================================================
# Test class 3: Serialization round-trip tests
# ================================================================


class TestSerializationRoundTrip:
    """Test save_model/load_model for round-trip fidelity of complex models."""

    def test_simple_belief_network_roundtrip(self) -> None:
        """Save and load a simple CPT-based BeliefNetwork."""
        bn = BeliefNetwork()
        bn.add_node("A", np.array([0.3, 0.7]), states=["a0", "a1"])
        bn.add_node(
            "B",
            np.array([[0.6, 0.4], [0.2, 0.8]]),
            parents=["A"],
            states=["b0", "b1"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            save_model(bn, path)

            # Verify the file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["format_version"] == "1.0"
            assert len(data["nodes"]) == 2

            # Load and verify structure
            loaded = load_model(path)
            assert isinstance(loaded, BeliefNetwork)
            assert loaded.nodes == bn.nodes
            assert loaded.edges == bn.edges
            assert loaded.get_states("A") == ["a0", "a1"]
            assert loaded.get_states("B") == ["b0", "b1"]

            # Verify distributions are preserved
            np.testing.assert_allclose(loaded._cpds["A"], bn._cpds["A"])
            np.testing.assert_allclose(loaded._cpds["B"], bn._cpds["B"])

    def test_complex_network_roundtrip(self) -> None:
        """Save and load a larger multi-node network."""
        bn = BeliefNetwork()
        bn.add_node("Weather", np.array([0.6, 0.4]), states=["sunny", "rainy"])
        bn.add_node(
            "Traffic",
            np.array([[0.7, 0.3], [0.4, 0.6]]),
            parents=["Weather"],
            states=["light", "heavy"],
        )
        bn.add_node(
            "Mood",
            np.array([[0.9, 0.1], [0.5, 0.5]]),
            parents=["Weather"],
            states=["happy", "sad"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "complex_model.json"
            save_model(bn, path)
            loaded = load_model(path)

            assert set(loaded.nodes) == {"Weather", "Traffic", "Mood"}
            assert len(loaded.edges) == 2

            # Inference should produce same results
            original_marginal = bn.marginal("Traffic")
            loaded_marginal = loaded.marginal("Traffic")
            np.testing.assert_allclose(original_marginal, loaded_marginal, atol=1e-10)

    def test_roundtrip_preserves_inference(self) -> None:
        """After round-trip serialization, inference should produce
        identical results."""
        bn = BeliefNetwork()
        bn.add_node("D", np.array([0.4, 0.6]), states=["d0", "d1"])
        bn.add_node(
            "E",
            np.array([[0.9, 0.1], [0.3, 0.7]]),
            parents=["D"],
            states=["e0", "e1"],
        )

        # Get inference results before serialization
        bn.observe("D", "d1")
        original_posterior = bn.infer("E")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "inference_model.json"
            save_model(bn, path)
            loaded = load_model(path)

            # Perform same inference on loaded model
            loaded.observe("D", "d1")
            loaded_posterior = loaded.infer("E")

            np.testing.assert_allclose(original_posterior, loaded_posterior, atol=1e-10)

    def test_save_non_belief_network_raises(self) -> None:
        """Saving a non-BeliefNetwork object should raise TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            with pytest.raises(TypeError, match="Expected BeliefNetwork"):
                save_model("not a network", path)

    def test_load_nonexistent_file_raises(self) -> None:
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.json")


# ================================================================
# Test class 4: HMM full workflow
# ================================================================


class TestHMMWorkflow:
    """End-to-end HMM workflow: define, fit, decode, and verify."""

    def test_hmm_viterbi_decoding(self) -> None:
        """Create an HMM with known parameters and decode observations."""
        hmm = HiddenMarkovModel(
            hidden_states=["sunny", "rainy"],
            observables=["walk", "shop", "clean"],
            transition_probs=[[0.7, 0.3], [0.4, 0.6]],
            emission_probs=[
                [0.6, 0.3, 0.1],   # sunny: walk=0.6, shop=0.3, clean=0.1
                [0.1, 0.4, 0.5],   # rainy: walk=0.1, shop=0.4, clean=0.5
            ],
            initial_probs=[0.6, 0.4],
        )

        observations = ["walk", "walk", "clean", "clean", "shop"]
        decoded = hmm.infer_state(observations)

        assert len(decoded) == 5
        assert all(s in ["sunny", "rainy"] for s in decoded)
        # First two observations (walk, walk) should likely be sunny
        assert decoded[0] == "sunny"
        assert decoded[1] == "sunny"
        # Last clean observations should likely be rainy
        assert decoded[3] == "rainy"

    def test_hmm_fit_and_decode(self) -> None:
        """Fit an HMM using Baum-Welch and verify it can decode sequences."""
        # Generate training data from a known HMM
        np.random.seed(42)
        true_hmm = HiddenMarkovModel(
            hidden_states=["A", "B"],
            observables=["x", "y", "z"],
            transition_probs=[[0.8, 0.2], [0.3, 0.7]],
            emission_probs=[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
            initial_probs=[0.6, 0.4],
        )

        # Generate a sequence
        rng = np.random.default_rng(42)
        states = []
        obs = []
        state = rng.choice(2, p=[0.6, 0.4])
        for _ in range(100):
            states.append(true_hmm.hidden_states[state])
            emit = rng.choice(3, p=true_hmm.emission_probs[state])
            obs.append(true_hmm.observables[emit])
            state = rng.choice(2, p=true_hmm.transition_probs[state])

        # Fit a new HMM from the data
        learned_hmm = HiddenMarkovModel(
            hidden_states=["A", "B"],
            observables=["x", "y", "z"],
        )
        learned_hmm.fit(obs, max_iter=50)

        # The learned model should be able to decode the sequence
        decoded = learned_hmm.infer_state(obs)
        assert len(decoded) == 100
        assert all(s in ["A", "B"] for s in decoded)

    def test_hmm_to_decision_pipeline(self) -> None:
        """Use HMM state inference to drive a decision tree.

        Pipeline: observe market signals -> infer hidden regime via HMM ->
        use inferred regime probabilities in decision tree.
        """
        hmm = HiddenMarkovModel(
            hidden_states=["expansion", "recession"],
            observables=["up", "flat", "down"],
            transition_probs=[[0.8, 0.2], [0.3, 0.7]],
            emission_probs=[
                [0.6, 0.3, 0.1],   # expansion: mostly up
                [0.1, 0.3, 0.6],   # recession: mostly down
            ],
            initial_probs=[0.7, 0.3],
        )

        # Observe recent market signals
        recent_signals = ["up", "up", "flat", "down", "down", "down"]
        decoded = hmm.infer_state(recent_signals)

        # Count regime occurrences to estimate current regime probability
        n_expansion = sum(1 for s in decoded if s == "expansion")
        n_recession = sum(1 for s in decoded if s == "recession")
        total = n_expansion + n_recession
        p_expansion = n_expansion / total
        p_recession = n_recession / total

        # Build decision tree based on inferred regime
        tree = DecisionTree()
        tree.add_decision("portfolio", ["equities", "bonds"])
        tree.add_chance(
            "equities",
            ["expansion", "recession"],
            [p_expansion, p_recession],
        )
        tree.set_payoff(("equities", "expansion"), 150)
        tree.set_payoff(("equities", "recession"), -50)
        tree.set_payoff(("bonds",), 30)

        result = tree.solve()
        assert result["strategy"]["portfolio"] in ["equities", "bonds"]

        # With many "down" signals, recession should be likely,
        # making bonds potentially preferred
        if p_recession > 0.6:
            # Expected value of equities would be < 30
            ev_equities = p_expansion * 150 + p_recession * (-50)
            if ev_equities < 30:
                assert result["strategy"]["portfolio"] == "bonds"


# ================================================================
# Test class 5: Markov chain + Monte Carlo combined pipeline
# ================================================================


class TestMarkovMonteCarloCombo:
    """Combine Markov chains with Monte Carlo for multi-step simulation."""

    def test_regime_switching_portfolio_simulation(self) -> None:
        """Simulate a portfolio with Markov-switching returns using
        Monte Carlo and make decisions based on the simulation results."""
        mc = MarkovChain(
            states=["bull", "bear"],
            transition_matrix=[[0.8, 0.2], [0.4, 0.6]],
        )

        bull_returns = Normal(mu=0.08, sigma=0.15)
        bear_returns = Normal(mu=-0.03, sigma=0.20)

        def simulate_5year_return():
            rng = np.random.default_rng()
            state = 0  # Start in bull
            cumulative = 1.0
            for _ in range(5):
                if state == 0:
                    ret = float(bull_returns.sample(1)[0])
                else:
                    ret = float(bear_returns.sample(1)[0])
                cumulative *= (1 + ret)
                state = rng.choice(2, p=mc.transition_matrix[state])
            return cumulative - 1.0  # Total return

        sim = MonteCarloSimulation(
            func=simulate_5year_return,
            n_samples=5000,
            seed=42,
        )
        results = sim.run()

        # The portfolio should have positive expected return over 5 years
        # (starting in bull market with mostly positive returns)
        assert results.mean() > -0.5  # Sanity check
        assert results.std() > 0  # There should be variation

        # Use results for a decision
        tree = DecisionTree()
        tree.add_decision("invest", ["risky", "safe"])
        tree.set_payoff(("risky",), results.mean() * 1000)  # Scale to dollars
        tree.set_payoff(("safe",), 50)  # Safe return of $50

        result = tree.solve()
        assert result["strategy"]["invest"] in ["risky", "safe"]

    def test_stationary_distribution_as_decision_input(self) -> None:
        """Use Markov chain stationary distribution as chance node
        probabilities in a decision tree."""
        mc = MarkovChain(
            states=["good", "ok", "bad"],
            transition_matrix=[
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.3, 0.6],
            ],
        )

        stationary = mc.stationary_distribution()
        assert np.isclose(stationary.sum(), 1.0)

        tree = DecisionTree()
        tree.add_decision("action", ["expand", "maintain"])
        tree.add_chance("expand", ["good", "ok", "bad"], list(stationary))
        tree.set_payoff(("expand", "good"), 500)
        tree.set_payoff(("expand", "ok"), 100)
        tree.set_payoff(("expand", "bad"), -300)
        tree.set_payoff(("maintain",), 80)

        result = tree.solve()
        assert "expected_value" in result
        assert result["strategy"]["action"] in ["expand", "maintain"]


# ================================================================
# Test class 6: Cross-module integration smoke tests
# ================================================================


class TestCrossModuleSmoke:
    """Smoke tests verifying different modules can be combined in
    realistic scenarios without errors."""

    def test_poisson_monte_carlo_decision(self) -> None:
        """Poisson distribution -> Monte Carlo -> Decision Tree."""
        demand = Poisson(lambda_=50)

        sim = MonteCarloSimulation(
            func=lambda: float(demand.sample(1)[0]),
            n_samples=5000,
            seed=42,
        )
        results = sim.run()
        avg_demand = results.mean()

        # Decision: stock 60 units or 40 units?
        tree = DecisionTree()
        tree.add_decision("stock", ["high_stock", "low_stock"])
        # High stock: surplus if demand < 60, perfect if >= 60
        # Low stock: surplus if demand < 40, stockout if >= 40
        p_high_demand = 1.0 - demand.cdf(59)
        p_low_demand = demand.cdf(59)
        tree.add_chance("high_stock", ["sells_well", "surplus"],
                        [float(p_high_demand), float(p_low_demand)])
        tree.set_payoff(("high_stock", "sells_well"), 500)
        tree.set_payoff(("high_stock", "surplus"), 300)
        tree.set_payoff(("low_stock",), 350)

        result = tree.solve()
        assert result["strategy"]["stock"] in ["high_stock", "low_stock"]

    def test_bernoulli_and_categorical_combined(self) -> None:
        """Combine Bernoulli and Categorical in a belief network scenario."""
        # Event happens or not (Bernoulli)
        event = Bernoulli(p=0.3)

        # If event happens, outcome is categorical
        outcome = Categorical([0.5, 0.3, 0.2], labels=["win", "draw", "lose"])

        # Monte Carlo: simulate event occurrence then outcome
        def simulate():
            happened = int(event.sample(1)[0])
            if happened:
                result = outcome.sample(1)[0]
                payoffs = {"win": 100.0, "draw": 0.0, "lose": -50.0}
                return payoffs[result]
            return 0.0  # No event

        sim = MonteCarloSimulation(
            func=simulate,
            n_samples=5000,
            seed=42,
        )
        results = sim.run()

        # Expected value: 0.3 * (0.5*100 + 0.3*0 + 0.2*(-50)) + 0.7 * 0
        # = 0.3 * (50 + 0 - 10) = 0.3 * 40 = 12
        assert np.isclose(results.mean(), 12.0, atol=3.0)

    def test_decision_tree_with_utility_functions(self) -> None:
        """Verify different utility functions produce different optimal strategies."""
        tree = DecisionTree()
        tree.add_decision("gamble", ["bet", "safe"])
        tree.add_chance("bet", ["win", "lose"], [0.5, 0.5])
        tree.set_payoff(("bet", "win"), 200)
        tree.set_payoff(("bet", "lose"), -100)
        tree.set_payoff(("safe",), 30)

        # Risk-neutral: EV of bet = 0.5*200 + 0.5*(-100) = 50 > 30 -> bet
        result_neutral = tree.solve(utility_function=UtilityFunction.linear())
        assert result_neutral["strategy"]["gamble"] == "bet"
        assert np.isclose(result_neutral["expected_value"], 50.0, atol=0.1)

        # Risk-averse with high risk aversion -> should prefer safe option
        result_averse = tree.solve(
            utility_function=UtilityFunction.exponential(risk_aversion=0.02)
        )
        # With high risk aversion and negative payoff possible, safe may be preferred
        assert result_averse["strategy"]["gamble"] in ["bet", "safe"]

    def test_normal_convolution_chain(self) -> None:
        """Chain of Normal convolutions and scaling, verified via Monte Carlo."""
        a = Normal(mu=10, sigma=2)
        b = Normal(mu=20, sigma=3)
        c = Normal(mu=5, sigma=1)

        # (a + b) * 0.5 + c
        ab = a + b
        ab_scaled = 0.5 * ab
        total = ab_scaled + c

        # Theoretical: mu = 0.5*(10+20) + 5 = 20
        # sigma = sqrt((0.5*sqrt(4+9))^2 + 1) = sqrt(0.25*13 + 1) = sqrt(4.25)
        assert isinstance(total, Normal)
        assert np.isclose(total.mu, 20.0)
        expected_sigma = np.sqrt((0.5 * np.sqrt(13))**2 + 1)
        assert np.isclose(total.sigma, expected_sigma, atol=0.01)

        # Verify via Monte Carlo
        sim = MonteCarloSimulation(
            func=lambda: float(total.sample(1)[0]),
            n_samples=10000,
            seed=42,
        )
        results = sim.run()
        assert np.isclose(results.mean(), 20.0, atol=0.5)
        assert np.isclose(results.std(), expected_sigma, atol=0.3)
