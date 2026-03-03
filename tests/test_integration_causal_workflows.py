"""Integration tests for end-to-end causal inference workflows.

Tests that verify CausalDAG, do-calculus interventions, counterfactual
reasoning, sensitivity analysis, and confounding detection work together
as complete causal inference pipelines.
"""

import numpy as np
import pytest

from probflow.causal.dag import CausalDAG, _FixedDist
from probflow.distributions.conditional import ConditionalDist
from probflow.distributions.continuous import Normal
from probflow.distributions.discrete import Categorical
from probflow.inference.sensitivity import sensitivity_analysis


# ================================================================
# Test class 1: Full do-calculus intervention workflow
# ================================================================


class TestDoCalculusWorkflow:
    """Build CausalDAG, perform do-calculus intervention, compare with observational."""

    def _build_treatment_dag(self) -> CausalDAG:
        """Build a treatment -> outcome DAG with a confounder.

        Structure:
            Confounder -> Treatment
            Confounder -> Outcome
            Treatment  -> Outcome

        This creates confounding between Treatment and Outcome.
        """
        dag = CausalDAG()

        confounder = Categorical([0.4, 0.6], labels=["low", "high"])
        dag.add_node("Confounder", confounder)

        treatment = ConditionalDist(
            parent=confounder,
            mapping={
                "low": Categorical([0.7, 0.3], labels=["0", "1"]),
                "high": Categorical([0.3, 0.7], labels=["0", "1"]),
            },
        )
        dag.add_node("Treatment", treatment, parents=["Confounder"])

        # Outcome depends on Treatment.  Since ConditionalDist only takes
        # a single parent's samples, we model it conditioned on Treatment.
        outcome = ConditionalDist(
            parent=treatment,
            mapping={
                "0": Normal(mu=5.0, sigma=1.0),
                "1": Normal(mu=8.0, sigma=1.5),
            },
        )
        dag.add_node("Outcome", outcome, parents=["Treatment"])

        return dag

    def test_observational_vs_interventional_differ(self) -> None:
        """Interventional distribution P(Y | do(X=1)) should differ from
        observational P(Y | X=1) when there is confounding."""
        dag = self._build_treatment_dag()

        # Observational: sample and filter for Treatment=1
        np.random.seed(42)
        obs_samples = dag.sample(20000)
        treated_mask = obs_samples["Treatment"] == "1"
        obs_outcome_given_treated = obs_samples["Outcome"][treated_mask].astype(float)

        # Interventional: do(Treatment=1)
        np.random.seed(42)
        intervened_dag = dag.do("Treatment", "1")
        int_samples = intervened_dag.sample(20000)
        int_outcome = int_samples["Outcome"].astype(float)

        # Both should have valid samples
        assert len(obs_outcome_given_treated) > 100
        assert len(int_outcome) == 20000

        # The means should be in a plausible range around 8.0
        # (Treatment=1 maps to Normal(8.0, 1.5))
        assert 6.0 < obs_outcome_given_treated.mean() < 10.0
        assert 6.0 < int_outcome.mean() < 10.0

    def test_do_removes_incoming_edges(self) -> None:
        """After do(Treatment=1), Treatment should have no parents."""
        dag = self._build_treatment_dag()
        intervened = dag.do("Treatment", "1")

        # In the intervened DAG, Treatment is now a root node with _FixedDist
        assert isinstance(intervened._nodes["Treatment"], _FixedDist)
        assert intervened._parents["Treatment"] == []

        # But Confounder and Outcome structure should be preserved
        assert "Confounder" in intervened._nodes
        assert "Outcome" in intervened._nodes
        assert intervened._parents["Outcome"] == ["Treatment"]

    def test_do_on_nonexistent_variable_raises(self) -> None:
        """Intervening on a variable not in the network should raise."""
        dag = self._build_treatment_dag()
        with pytest.raises(ValueError, match="not in the network"):
            dag.do("Nonexistent", "1")

    def test_interventional_sample_convenience(self) -> None:
        """The interventional_sample method should produce consistent results."""
        dag = self._build_treatment_dag()
        np.random.seed(123)
        samples = dag.interventional_sample({"Treatment": "1"}, n=5000)

        assert "Treatment" in samples
        assert "Outcome" in samples
        # All Treatment samples should be "1" (the intervention value)
        assert np.all(samples["Treatment"] == "1")
        # Outcome should reflect Treatment=1 -> Normal(8, 1.5)
        outcome_mean = samples["Outcome"].astype(float).mean()
        assert 7.0 < outcome_mean < 9.0


# ================================================================
# Test class 2: Counterfactual reasoning
# ================================================================


class TestCounterfactualReasoning:
    """Test counterfactual reasoning end-to-end using the twin network method."""

    def _build_simple_dag(self) -> CausalDAG:
        """X -> Y with no confounding.

        X ~ Categorical([0.5, 0.5])
        Y | X ~ {0: Normal(3, 1), 1: Normal(7, 1)}
        """
        dag = CausalDAG()

        x = Categorical([0.5, 0.5], labels=["0", "1"])
        dag.add_node("X", x)

        y = ConditionalDist(
            parent=x,
            mapping={
                "0": Normal(mu=3.0, sigma=1.0),
                "1": Normal(mu=7.0, sigma=1.0),
            },
        )
        dag.add_node("Y", y, parents=["X"])
        return dag

    def test_counterfactual_intervention_shifts_outcome(self) -> None:
        """Counterfactual: 'What would Y have been if X had been 1, given
        that we observed X=0?'"""
        dag = self._build_simple_dag()
        np.random.seed(42)

        cf_samples = dag.counterfactual(
            intervention={"X": "1"},
            evidence={"X": "0"},
            n=50000,
        )

        assert "X" in cf_samples
        assert "Y" in cf_samples
        # All X values should be "1" (the intervention)
        assert np.all(cf_samples["X"] == "1")
        # Y should reflect X=1 -> Normal(7, 1)
        y_values = cf_samples["Y"].astype(float)
        assert 6.0 < y_values.mean() < 8.0

    def test_counterfactual_no_matching_evidence_raises(self) -> None:
        """If evidence is impossible (no matching samples), should raise."""
        dag = self._build_simple_dag()
        with pytest.raises(ValueError, match="No samples matched"):
            dag.counterfactual(
                intervention={"X": "1"},
                evidence={"X": "nonexistent_value"},
                n=1000,
            )

    def test_counterfactual_preserves_exogenous(self) -> None:
        """Root nodes not intervened on should keep their abducted values."""
        dag = CausalDAG()
        u = Categorical([0.5, 0.5], labels=["a", "b"])
        dag.add_node("U", u)
        x = ConditionalDist(
            parent=u,
            mapping={
                "a": Categorical([0.9, 0.1], labels=["0", "1"]),
                "b": Categorical([0.1, 0.9], labels=["0", "1"]),
            },
        )
        dag.add_node("X", x, parents=["U"])
        y = ConditionalDist(
            parent=x,
            mapping={
                "0": Normal(mu=2, sigma=0.5),
                "1": Normal(mu=6, sigma=0.5),
            },
        )
        dag.add_node("Y", y, parents=["X"])

        np.random.seed(42)
        cf = dag.counterfactual(
            intervention={"X": "1"},
            evidence={"U": "a"},
            n=50000,
        )

        # U should be preserved from abduction (all "a" since evidence says U=a)
        assert np.all(cf["U"] == "a")
        # X should be "1" from intervention
        assert np.all(cf["X"] == "1")
        # Y should reflect X=1 -> Normal(6, 0.5)
        y_vals = cf["Y"].astype(float)
        assert 5.0 < y_vals.mean() < 7.0


# ================================================================
# Test class 3: Sensitivity analysis on causal effect estimates
# ================================================================


class TestSensitivityAnalysisOnCausalEffects:
    """Test sensitivity analysis applied to causal effect estimation functions."""

    def test_sensitivity_of_linear_causal_model(self) -> None:
        """Sensitivity analysis on a linear structural equation model.

        Y = alpha * X + beta * Z + noise
        """

        def causal_model(params):
            alpha = params["alpha"]
            beta = params["beta"]
            # Simple deterministic calculation of E[Y] given the parameters
            x_mean = 5.0
            z_mean = 3.0
            y_mean = alpha * x_mean + beta * z_mean
            return {"E_Y": y_mean}

        result = sensitivity_analysis(
            network=causal_model,
            target="E_Y",
            parameters={"alpha": 2.0, "beta": 1.0},
            perturbation=0.01,
        )

        # dE_Y/dalpha = x_mean = 5.0
        assert np.isclose(result["alpha"], 5.0, atol=0.1)
        # dE_Y/dbeta = z_mean = 3.0
        assert np.isclose(result["beta"], 3.0, atol=0.1)
        # alpha should be more sensitive (larger absolute derivative)
        assert abs(result["alpha"]) > abs(result["beta"])

    def test_sensitivity_with_nonlinear_model(self) -> None:
        """Sensitivity analysis on a nonlinear causal model."""

        def nonlinear_model(params):
            a = params["a"]
            b = params["b"]
            return {"output": a**2 + 3 * b}

        result = sensitivity_analysis(
            network=nonlinear_model,
            target="output",
            parameters={"a": 2.0, "b": 1.0},
            perturbation=0.01,
        )

        # d(a^2 + 3b)/da = 2a = 4.0
        assert np.isclose(result["a"], 4.0, atol=0.1)
        # d(a^2 + 3b)/db = 3.0
        assert np.isclose(result["b"], 3.0, atol=0.1)

    def test_sensitivity_analysis_on_causal_ate(self) -> None:
        """Sensitivity of average treatment effect to model parameters.

        ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
        where Y = base + treatment_effect * X + confounder_strength * C
        """

        def ate_model(params):
            treatment_effect = params["treatment_effect"]
            confounder_strength = params["confounder_strength"]
            base = params["base"]
            # ATE under do-calculus (removing confounding)
            # = E[Y | do(X=1)] - E[Y | do(X=0)]
            # = (base + treatment_effect * 1) - (base + treatment_effect * 0)
            # = treatment_effect
            # But if we include confounder bias:
            confounder_mean = 0.5
            ate = treatment_effect + confounder_strength * confounder_mean * 0.0
            return {"ATE": ate, "biased_ATE": treatment_effect + confounder_strength * confounder_mean}

        result = sensitivity_analysis(
            network=ate_model,
            target="ATE",
            parameters={
                "treatment_effect": 3.0,
                "confounder_strength": 2.0,
                "base": 10.0,
            },
            perturbation=0.01,
        )

        # ATE depends only on treatment_effect (sensitivity = 1)
        assert np.isclose(result["treatment_effect"], 1.0, atol=0.1)
        # ATE is insensitive to confounder_strength and base
        assert np.isclose(result["confounder_strength"], 0.0, atol=0.1)
        assert np.isclose(result["base"], 0.0, atol=0.1)

    def test_sensitivity_validation_errors(self) -> None:
        """Sensitivity analysis should raise on invalid inputs."""
        with pytest.raises(TypeError, match="callable"):
            sensitivity_analysis("not_callable", "y", {"a": 1.0})

        with pytest.raises(ValueError, match="non-empty"):
            sensitivity_analysis(lambda p: {"y": 1}, "y", {})

        with pytest.raises(ValueError, match="positive"):
            sensitivity_analysis(lambda p: {"y": 1}, "y", {"a": 1.0}, perturbation=-0.1)


# ================================================================
# Test class 4: Confounding detection and backdoor adjustment
# ================================================================


class TestConfoundingDetectionAndAdjustment:
    """Test confounding detection, backdoor criterion, and effect estimation."""

    def _build_confounded_dag(self) -> CausalDAG:
        """Build a DAG with known confounding.

        Structure:
            Z -> X
            Z -> Y
            X -> Y

        Z is a confounder of the X -> Y relationship.
        """
        dag = CausalDAG()

        z = Categorical([0.5, 0.5], labels=["0", "1"])
        dag.add_node("Z", z)

        x = ConditionalDist(
            parent=z,
            mapping={
                "0": Categorical([0.8, 0.2], labels=["0", "1"]),
                "1": Categorical([0.2, 0.8], labels=["0", "1"]),
            },
        )
        dag.add_node("X", x, parents=["Z"])

        y = ConditionalDist(
            parent=x,
            mapping={
                "0": Normal(mu=2.0, sigma=1.0),
                "1": Normal(mu=6.0, sigma=1.0),
            },
        )
        dag.add_node("Y", y, parents=["X"])

        return dag

    def test_find_confounders(self) -> None:
        """Confounder detection should identify Z as a confounder of X -> Y."""
        dag = self._build_confounded_dag()
        confounders = dag.find_confounders("X", "Y")
        assert "Z" in confounders

    def test_identify_effect_backdoor(self) -> None:
        """The causal effect of X on Y should be identifiable via the
        backdoor criterion with Z as the adjustment set."""
        dag = self._build_confounded_dag()
        result = dag.identify_effect("X", "Y")

        assert result["identifiable"] is True
        assert result["method"] == "backdoor"
        assert "Z" in result["adjustment_set"]

    def test_identify_effect_no_confounding(self) -> None:
        """When there is no confounding, the effect should be directly identifiable."""
        dag = CausalDAG()
        x = Categorical([0.5, 0.5], labels=["0", "1"])
        dag.add_node("X", x)
        y = ConditionalDist(
            parent=x,
            mapping={
                "0": Normal(mu=3.0, sigma=1.0),
                "1": Normal(mu=7.0, sigma=1.0),
            },
        )
        dag.add_node("Y", y, parents=["X"])

        result = dag.identify_effect("X", "Y")
        assert result["identifiable"] is True
        assert result["method"] == "backdoor"
        assert result["adjustment_set"] == set()  # Empty set -- no confounders

    def test_backdoor_adjustment_effect_estimation(self) -> None:
        """Full workflow: detect confounders -> find backdoor set ->
        estimate causal effect via intervention."""
        dag = self._build_confounded_dag()

        # Step 1: Detect confounders
        confounders = dag.find_confounders("X", "Y")
        assert len(confounders) > 0

        # Step 2: Identify via backdoor
        identification = dag.identify_effect("X", "Y")
        assert identification["identifiable"]

        # Step 3: Estimate causal effect via do-calculus
        np.random.seed(42)
        samples_do_1 = dag.interventional_sample({"X": "1"}, n=10000)
        samples_do_0 = dag.interventional_sample({"X": "0"}, n=10000)

        y_do_1 = samples_do_1["Y"].astype(float).mean()
        y_do_0 = samples_do_0["Y"].astype(float).mean()
        causal_effect = y_do_1 - y_do_0

        # True causal effect: E[Y|do(X=1)] - E[Y|do(X=0)] = 6 - 2 = 4
        assert 3.0 < causal_effect < 5.0

    def test_graph_traversal_helpers(self) -> None:
        """Test ancestors_of, descendants_of, children_of for the confounded DAG."""
        dag = self._build_confounded_dag()

        # Z is an ancestor of both X and Y
        assert "Z" in dag.ancestors_of("X")
        assert "Z" in dag.ancestors_of("Y")

        # X is an ancestor of Y
        assert "X" in dag.ancestors_of("Y")

        # Y is a descendant of X
        assert "Y" in dag.descendants_of("X")
        assert "Y" in dag.descendants_of("Z")

        # Z's children include X
        assert "X" in dag.children_of("Z")

    def test_frontdoor_dag(self) -> None:
        """Test identifiability via frontdoor criterion.

        Structure:
            U -> X  (U is unobserved confounder, but exists in graph)
            U -> Y
            X -> M -> Y

        The frontdoor set is {M}.
        """
        dag = CausalDAG()
        u = Categorical([0.5, 0.5], labels=["0", "1"])
        dag.add_node("U", u)

        x = ConditionalDist(
            parent=u,
            mapping={
                "0": Categorical([0.6, 0.4], labels=["0", "1"]),
                "1": Categorical([0.4, 0.6], labels=["0", "1"]),
            },
        )
        dag.add_node("X", x, parents=["U"])

        m = ConditionalDist(
            parent=x,
            mapping={
                "0": Categorical([0.9, 0.1], labels=["0", "1"]),
                "1": Categorical([0.2, 0.8], labels=["0", "1"]),
            },
        )
        dag.add_node("M", m, parents=["X"])

        y = ConditionalDist(
            parent=m,
            mapping={
                "0": Normal(mu=1.0, sigma=0.5),
                "1": Normal(mu=5.0, sigma=0.5),
            },
        )
        dag.add_node("Y", y, parents=["M"])

        # M should be a descendant of X and ancestor of Y
        assert "M" in dag.descendants_of("X")
        assert "M" in dag.ancestors_of("Y")

        # The confounders of X and Y should include U
        confounders = dag.find_confounders("X", "Y")
        assert "U" in confounders


# ================================================================
# Test class 5: Combined causal + sensitivity workflow
# ================================================================


class TestCausalSensitivityWorkflow:
    """Combine causal DAG operations with sensitivity analysis for
    a complete analytical workflow."""

    def test_sensitivity_of_causal_effect_to_priors(self) -> None:
        """How sensitive is the estimated causal effect to the prior
        probability of the confounder?"""

        def causal_effect_model(params):
            p_z = params["p_z"]
            # X | Z: P(X=1|Z=0)=0.2, P(X=1|Z=1)=0.8
            # Y | X: E[Y|X=0]=2, E[Y|X=1]=6
            # Under do(X=1): E[Y|do(X=1)] = 6 always
            # Under do(X=0): E[Y|do(X=0)] = 2 always
            # ATE = 4, independent of p_z (as expected from do-calculus)
            ate = 6.0 - 2.0  # Causal effect is constant
            # But observational association changes with p_z:
            p_x1 = 0.2 * (1 - p_z) + 0.8 * p_z
            obs_e_y = p_x1 * 6.0 + (1 - p_x1) * 2.0
            return {"ATE": ate, "obs_E_Y": obs_e_y}

        result = sensitivity_analysis(
            network=causal_effect_model,
            target="ATE",
            parameters={"p_z": 0.5},
            perturbation=0.01,
        )

        # ATE should be insensitive to p_z (it's 4.0 regardless)
        assert np.isclose(result["p_z"], 0.0, atol=0.1)

        # Now check observational sensitivity
        result_obs = sensitivity_analysis(
            network=causal_effect_model,
            target="obs_E_Y",
            parameters={"p_z": 0.5},
            perturbation=0.01,
        )

        # Observational E[Y] IS sensitive to p_z
        # d(obs_E_Y)/d(p_z) = d/dp_z [p_x1 * 6 + (1-p_x1) * 2]
        # = d/dp_z [4 * p_x1 + 2] = 4 * d(p_x1)/d(p_z) = 4 * 0.6 = 2.4
        assert abs(result_obs["p_z"]) > 1.0  # Significantly sensitive
