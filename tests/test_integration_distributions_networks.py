"""Integration tests combining distributions with Bayesian networks.

Tests that verify distributions (Normal, Categorical, ConditionalDist)
integrate correctly with the BeliefNetwork (networks/dag.py) and
the belief-network-based inference pipeline (inference/belief_network.py).
"""

import numpy as np
import pytest

from probflow.distributions.conditional import ConditionalDist
from probflow.distributions.continuous import Beta, LogNormal, Normal
from probflow.distributions.discrete import Bernoulli, Categorical, Poisson
from probflow.inference.belief_network import BeliefNetwork as SamplingBeliefNetwork
from probflow.networks.dag import BeliefNetwork as CPTBeliefNetwork


# ================================================================
# Test class 1: CPT-based BeliefNetwork with inference
# ================================================================


class TestCPTBeliefNetworkInference:
    """Build a CPT-based BeliefNetwork and run inference after observing evidence."""

    def _build_weather_network(self) -> CPTBeliefNetwork:
        """Classic weather-sprinkler-grass-wet example.

        Structure:
            Cloudy -> Sprinkler
            Cloudy -> Rain
            Sprinkler -> WetGrass
            Rain -> WetGrass
        """
        bn = CPTBeliefNetwork()

        # P(Cloudy)
        bn.add_node("Cloudy", np.array([0.5, 0.5]), states=["F", "T"])

        # P(Sprinkler | Cloudy)
        bn.add_node(
            "Sprinkler",
            np.array([[0.5, 0.5], [0.9, 0.1]]),
            parents=["Cloudy"],
            states=["F", "T"],
        )

        # P(Rain | Cloudy)
        bn.add_node(
            "Rain",
            np.array([[0.8, 0.2], [0.2, 0.8]]),
            parents=["Cloudy"],
            states=["F", "T"],
        )

        # P(WetGrass | Sprinkler, Rain)
        bn.add_node(
            "WetGrass",
            np.array([
                [[1.0, 0.0], [0.1, 0.9]],
                [[0.1, 0.9], [0.01, 0.99]],
            ]),
            parents=["Sprinkler", "Rain"],
            states=["F", "T"],
        )
        return bn

    def test_marginal_without_evidence(self) -> None:
        """Marginal distribution should be a valid probability vector."""
        bn = self._build_weather_network()
        marginal = bn.marginal("Rain")
        assert marginal.shape == (2,)
        assert np.isclose(marginal.sum(), 1.0)
        # P(Rain) = P(R|C=F)*P(C=F) + P(R|C=T)*P(C=T)
        # = 0.2*0.5 + 0.8*0.5 = 0.5
        assert np.isclose(marginal[1], 0.5, atol=0.05)

    def test_infer_with_evidence_changes_posterior(self) -> None:
        """Observing evidence should shift the posterior vs. the marginal."""
        bn = self._build_weather_network()
        prior = bn.marginal("Rain")

        # Observe that it is cloudy
        bn.observe("Cloudy", "T")
        posterior = bn.infer("Rain")

        assert posterior.shape == (2,)
        assert np.isclose(posterior.sum(), 1.0)
        # P(Rain | Cloudy=T) should be ~0.8 (higher than marginal ~0.5)
        assert posterior[1] > prior[1]

    def test_infer_multiple_evidence(self) -> None:
        """Inference with multiple evidence variables should be consistent."""
        bn = self._build_weather_network()
        bn.observe("Cloudy", "T")
        bn.observe("Sprinkler", "F")
        posterior = bn.infer("Rain")

        assert np.isclose(posterior.sum(), 1.0)
        # With Cloudy=T and Sprinkler=F, Rain should still be likely
        assert posterior[1] > 0.5

    def test_clear_evidence_resets(self) -> None:
        """After clearing evidence, inference should match the marginal."""
        bn = self._build_weather_network()
        marginal = bn.marginal("Rain")

        bn.observe("Cloudy", "T")
        bn.clear_evidence()
        posterior_after_clear = bn.infer("Rain")

        np.testing.assert_allclose(posterior_after_clear, marginal, atol=0.05)

    def test_d_separation_in_network(self) -> None:
        """Sprinkler and Rain should be d-separated given Cloudy."""
        bn = self._build_weather_network()
        # Sprinkler _|_ Rain | Cloudy
        assert bn.d_separated("Sprinkler", "Rain", {"Cloudy"})
        # Without conditioning, they are NOT d-separated (they share parent Cloudy)
        assert not bn.d_separated("Sprinkler", "Rain")


# ================================================================
# Test class 2: ConditionalDist with SamplingBeliefNetwork
# ================================================================


class TestConditionalDistWithBeliefNetwork:
    """Test ConditionalDist distributions integrated into a SamplingBeliefNetwork."""

    def test_categorical_parent_conditional_child(self) -> None:
        """A Categorical parent driving a ConditionalDist Normal child."""
        bn = SamplingBeliefNetwork()

        regime = Categorical([0.6, 0.4], labels=["bull", "bear"])
        bn.add_node("regime", regime)

        returns = ConditionalDist(
            parent=regime,
            mapping={
                "bull": Normal(mu=0.08, sigma=0.15),
                "bear": Normal(mu=-0.05, sigma=0.25),
            },
        )
        bn.add_node("returns", returns, parents=["regime"])

        samples = bn.sample(5000)
        assert "regime" in samples
        assert "returns" in samples
        assert samples["regime"].shape == (5000,)
        assert samples["returns"].shape == (5000,)

        # Bull regime has positive mean, bear has negative.
        # With 60% bull and 40% bear, overall mean should be slightly positive.
        bull_mask = samples["regime"] == "bull"
        bear_mask = samples["regime"] == "bear"

        bull_returns = samples["returns"][bull_mask]
        bear_returns = samples["returns"][bear_mask]

        # Bull returns should have positive mean, bear negative
        assert bull_returns.mean() > 0, "Bull regime mean should be positive"
        assert bear_returns.mean() < 0, "Bear regime mean should be negative"

    def test_multi_level_conditional_chain(self) -> None:
        """Chain of conditionals: Economy -> Regime -> Volatility."""
        bn = SamplingBeliefNetwork()

        economy = Categorical([0.3, 0.5, 0.2], labels=["recession", "normal", "boom"])
        bn.add_node("economy", economy)

        regime = ConditionalDist(
            parent=economy,
            mapping={
                "recession": Categorical([0.7, 0.3], labels=["bear", "bull"]),
                "normal": Categorical([0.4, 0.6], labels=["bear", "bull"]),
                "boom": Categorical([0.1, 0.9], labels=["bear", "bull"]),
            },
        )
        bn.add_node("regime", regime, parents=["economy"])

        volatility = ConditionalDist(
            parent=regime,
            mapping={
                "bear": Normal(mu=0.3, sigma=0.1),
                "bull": Normal(mu=0.15, sigma=0.05),
            },
        )
        bn.add_node("volatility", volatility, parents=["regime"])

        samples = bn.sample(10000)
        assert len(samples) == 3
        assert all(k in samples for k in ["economy", "regime", "volatility"])

        # Volatility during bear regime should be higher than during bull
        bear_mask = samples["regime"] == "bear"
        bull_mask = samples["regime"] == "bull"
        assert samples["volatility"][bear_mask].mean() > samples["volatility"][bull_mask].mean()

    def test_continuous_parent_binned_conditional(self) -> None:
        """ConditionalDist with a continuous Normal parent using bin edges."""
        bn = SamplingBeliefNetwork()

        temperature = Normal(mu=20.0, sigma=5.0)
        bn.add_node("temperature", temperature)

        # Bin edges: cold (<15), moderate (15-25), hot (>25)
        energy_use = ConditionalDist(
            parent=temperature,
            mapping={
                15.0: Normal(mu=100, sigma=10),   # cold: high energy use
                25.0: Normal(mu=50, sigma=5),      # moderate: low energy use
            },
        )
        bn.add_node("energy_use", energy_use, parents=["temperature"])

        samples = bn.sample(5000)
        assert samples["temperature"].shape == (5000,)
        assert samples["energy_use"].shape == (5000,)

        # Cold temperatures should correlate with higher energy use
        cold_mask = samples["temperature"] < 15.0
        warm_mask = samples["temperature"] >= 15.0
        if cold_mask.any() and warm_mask.any():
            assert samples["energy_use"][cold_mask].mean() > samples["energy_use"][warm_mask].mean()


# ================================================================
# Test class 3: Distribution operations feeding into network decisions
# ================================================================


class TestDistributionOperationsInNetwork:
    """Test distribution arithmetic (Normal convolution, scaling) combined
    with network-level constructs."""

    def test_normal_convolution_in_belief_network(self) -> None:
        """Sum of two Normal distributions used as a node distribution."""
        bn = SamplingBeliefNetwork()

        # Portfolio return = stock_return + bond_return
        stock_return = Normal(mu=0.10, sigma=0.20)
        bond_return = Normal(mu=0.04, sigma=0.05)

        # Normal + Normal = Normal (convolution) with closed-form params
        portfolio_return = stock_return + bond_return
        assert isinstance(portfolio_return, Normal)
        assert np.isclose(portfolio_return.mu, 0.14)
        expected_sigma = np.sqrt(0.20**2 + 0.05**2)
        assert np.isclose(portfolio_return.sigma, expected_sigma)

        bn.add_node("portfolio_return", portfolio_return)
        samples = bn.sample(10000)

        # Sample statistics should match theoretical parameters
        assert np.isclose(samples["portfolio_return"].mean(), 0.14, atol=0.02)
        assert np.isclose(samples["portfolio_return"].std(), expected_sigma, atol=0.02)

    def test_normal_scaling_in_belief_network(self) -> None:
        """Scaling a Normal distribution and using it in a network."""
        bn = SamplingBeliefNetwork()

        base_dist = Normal(mu=100, sigma=15)
        # Applying leverage of 2x
        leveraged = 2.0 * base_dist
        assert isinstance(leveraged, Normal)
        assert np.isclose(leveraged.mu, 200)
        assert np.isclose(leveraged.sigma, 30)

        bn.add_node("leveraged_return", leveraged)
        samples = bn.sample(5000)
        assert np.isclose(samples["leveraged_return"].mean(), 200, atol=5)

    def test_combined_convolution_then_conditional(self) -> None:
        """Convolve two Normals, then use the result as a parent for
        a ConditionalDist in a belief network."""
        bn = SamplingBeliefNetwork()

        # Revenue = base_revenue + bonus
        base_revenue = Normal(mu=1000, sigma=100)
        bonus = Normal(mu=200, sigma=50)
        total_revenue = base_revenue + bonus
        bn.add_node("total_revenue", total_revenue)

        # Decision depends on whether revenue exceeds a threshold
        # Use binned conditional: revenue < 1100 => low tier, >= 1100 => high tier
        tier = ConditionalDist(
            parent=total_revenue,
            mapping={
                1100.0: Categorical([0.8, 0.2], labels=["low", "high"]),  # below threshold
            },
        )
        bn.add_node("tier", tier, parents=["total_revenue"])

        samples = bn.sample(5000)
        assert "total_revenue" in samples
        assert "tier" in samples
        # Revenue mean is 1200, so most samples should be above 1100
        # therefore the "high" tier (from the second bin) should dominate
        assert samples["tier"].shape == (5000,)

    def test_beta_distribution_in_network(self) -> None:
        """Beta distribution as a probability parameter feeding into Bernoulli sampling."""
        bn = SamplingBeliefNetwork()

        # Conversion rate follows a Beta distribution
        conversion_rate = Beta(alpha=5, beta=20)
        bn.add_node("conversion_rate", conversion_rate)

        samples = bn.sample(3000)
        # Beta(5, 20) has mean = 5/25 = 0.2
        assert np.isclose(samples["conversion_rate"].mean(), 0.2, atol=0.03)
        # All samples should be in [0, 1]
        assert np.all(samples["conversion_rate"] >= 0)
        assert np.all(samples["conversion_rate"] <= 1)

    def test_lognormal_distribution_in_network(self) -> None:
        """LogNormal distribution in a belief network for modeling asset prices."""
        bn = SamplingBeliefNetwork()

        price = LogNormal(mu=4.0, sigma=0.5)
        bn.add_node("price", price)

        # Scale by a factor
        scaled_price = 1.1 * price  # 10% premium
        bn.add_node("premium_price", scaled_price)

        samples = bn.sample(5000)
        # LogNormal is always positive
        assert np.all(samples["price"] > 0)
        assert np.all(samples["premium_price"] > 0)
        # Scaled price should have higher mean
        assert samples["premium_price"].mean() > samples["price"].mean()


# ================================================================
# Test class 4: CPT BeliefNetwork combined with distribution queries
# ================================================================


class TestCPTNetworkWithDistributionQueries:
    """Test combining CPT-based network inference with distribution operations."""

    def test_inference_result_drives_categorical(self) -> None:
        """Use network inference output as parameters for a Categorical distribution."""
        bn = CPTBeliefNetwork()
        bn.add_node("Weather", np.array([0.7, 0.3]), states=["sunny", "rainy"])
        bn.add_node(
            "Activity",
            np.array([[0.8, 0.2], [0.3, 0.7]]),
            parents=["Weather"],
            states=["outdoor", "indoor"],
        )

        # Get marginal for Activity (no evidence)
        activity_probs = bn.marginal("Activity")
        assert np.isclose(activity_probs.sum(), 1.0)

        # Use marginal as parameters for a Categorical distribution
        activity_dist = Categorical(activity_probs, labels=["outdoor", "indoor"])
        samples = activity_dist.sample(1000)
        assert all(s in ["outdoor", "indoor"] for s in samples)

        # Now observe evidence and re-query
        bn.observe("Weather", "rainy")
        posterior_probs = bn.infer("Activity")
        posterior_dist = Categorical(posterior_probs, labels=["outdoor", "indoor"])

        # When rainy, indoor should be more likely
        assert posterior_dist.pmf("indoor") > posterior_dist.pmf("outdoor")

    def test_network_structure_queries(self) -> None:
        """Verify network structure properties are accessible and consistent."""
        bn = CPTBeliefNetwork()
        bn.add_node("A", np.array([0.3, 0.7]), states=["a0", "a1"])
        bn.add_node(
            "B",
            np.array([[0.6, 0.4], [0.2, 0.8]]),
            parents=["A"],
            states=["b0", "b1"],
        )
        bn.add_node(
            "C",
            np.array([[0.9, 0.1], [0.4, 0.6]]),
            parents=["A"],
            states=["c0", "c1"],
        )

        assert set(bn.nodes) == {"A", "B", "C"}
        assert ("A", "B") in bn.edges
        assert ("A", "C") in bn.edges
        assert bn.get_states("A") == ["a0", "a1"]

        # B and C should be d-separated given A
        assert bn.d_separated("B", "C", {"A"})

    def test_bernoulli_operations_with_network_probs(self) -> None:
        """Use inferred probabilities to construct Bernoulli distributions
        and test independence operations."""
        bn = CPTBeliefNetwork()
        bn.add_node("Test", np.array([0.4, 0.6]), states=["neg", "pos"])
        bn.add_node(
            "Disease",
            np.array([[0.95, 0.05], [0.1, 0.9]]),
            parents=["Test"],
            states=["healthy", "sick"],
        )

        # Get P(Disease=sick) marginally
        disease_marginal = bn.marginal("Disease")
        p_sick = float(disease_marginal[1])

        # Get P(Disease=sick | Test=pos)
        bn.observe("Test", "pos")
        disease_posterior = bn.infer("Disease")
        p_sick_given_pos = float(disease_posterior[1])

        # Construct Bernoulli distributions from these probabilities
        prior_sick = Bernoulli(p_sick)
        posterior_sick = Bernoulli(p_sick_given_pos)

        # Posterior probability of sickness given positive test should be higher
        assert posterior_sick.p > prior_sick.p

        # Test Bernoulli AND/OR operators
        joint = prior_sick & posterior_sick
        assert isinstance(joint, Bernoulli)
        assert np.isclose(joint.p, prior_sick.p * posterior_sick.p)

        union = prior_sick | posterior_sick
        assert isinstance(union, Bernoulli)
        expected_union = prior_sick.p + posterior_sick.p - prior_sick.p * posterior_sick.p
        assert np.isclose(union.p, expected_union)
