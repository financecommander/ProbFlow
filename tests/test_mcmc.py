"""Tests for probflow.inference.mcmc."""

import numpy as np
import pytest

from probflow.distributions.continuous import Normal
from probflow.inference.belief_network import BeliefNetwork

# Skip entire module if PyMC is not installed
pymc = pytest.importorskip("pymc")
arviz = pytest.importorskip("arviz")

from probflow.inference.mcmc import MCMCSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_normal_network(mu=0.0, sigma=1.0):
    """Single-node BeliefNetwork: X ~ Normal(mu, sigma)."""
    bn = BeliefNetwork()
    bn.add_node("x", Normal(mu, sigma))
    return bn


def _two_node_network():
    """Two independent normals: mu ~ N(0, 10), sigma_proxy ~ N(1, 0.5)."""
    bn = BeliefNetwork()
    bn.add_node("mu", Normal(0, 10))
    bn.add_node("sigma_proxy", Normal(1, 0.5))
    return bn


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMCMCSamplerConstruction:
    """Tests for creating an MCMCSampler from a BeliefNetwork."""

    def test_from_network_creates_sampler(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        assert isinstance(sampler, MCMCSampler)

    def test_from_network_var_names(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        assert sampler._var_names == ["x"]

    def test_from_network_multi_node(self):
        bn = _two_node_network()
        sampler = MCMCSampler.from_network(bn)
        assert set(sampler._var_names) == {"mu", "sigma_proxy"}

    def test_from_network_unsupported_dist_raises(self):
        """A node with an unsupported distribution type should raise."""

        class CustomDist:
            pass

        bn = BeliefNetwork()
        bn.add_node("x", CustomDist())
        with pytest.raises(ValueError, match="Unsupported"):
            MCMCSampler.from_network(bn)

    def test_repr_before_sampling(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        r = repr(sampler)
        assert "sampled=False" in r
        assert "x" in r

    def test_model_property(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        assert sampler.model is not None

    def test_inference_data_none_before_sample(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        assert sampler.inference_data is None


# ---------------------------------------------------------------------------
# Sampling – simple Normal model convergence
# ---------------------------------------------------------------------------

class TestNormalConvergence:
    """MCMC on a single Normal(5, 2) should converge to the prior."""

    @pytest.fixture(scope="class")
    def sampler_and_idata(self):
        bn = _simple_normal_network(mu=5.0, sigma=2.0)
        sampler = MCMCSampler.from_network(bn)
        idata = sampler.sample(
            n_samples=1000,
            tune=500,
            chains=2,
            random_seed=42,
        )
        return sampler, idata

    def test_returns_inference_data(self, sampler_and_idata):
        _, idata = sampler_and_idata
        assert hasattr(idata, "posterior")

    def test_posterior_shape(self, sampler_and_idata):
        _, idata = sampler_and_idata
        # 2 chains × 1000 draws
        assert idata.posterior["x"].shape == (2, 1000)

    def test_r_hat_near_one(self, sampler_and_idata):
        sampler, _ = sampler_and_idata
        r_hat = sampler.r_hat()
        assert abs(r_hat["x"] - 1.0) < 0.05

    def test_ess_reasonable(self, sampler_and_idata):
        sampler, _ = sampler_and_idata
        ess = sampler.ess()
        # ESS should be meaningful (> 100 out of 2000 total draws)
        assert ess["x"] > 100

    def test_diagnostics_keys(self, sampler_and_idata):
        sampler, _ = sampler_and_idata
        diag = sampler.diagnostics()
        assert "x" in diag
        assert "r_hat" in diag["x"]
        assert "ess_bulk" in diag["x"]
        assert "ess_tail" in diag["x"]

    def test_repr_after_sampling(self, sampler_and_idata):
        sampler, _ = sampler_and_idata
        r = repr(sampler)
        assert "sampled=True" in r


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------

class TestParameterRecovery:
    """Posterior mean should recover the prior mean for a simple model."""

    @pytest.fixture(scope="class")
    def sampler_and_idata(self):
        bn = _simple_normal_network(mu=3.0, sigma=1.0)
        sampler = MCMCSampler.from_network(bn)
        idata = sampler.sample(
            n_samples=2000,
            tune=500,
            chains=2,
            random_seed=123,
        )
        return sampler, idata

    def test_posterior_mean_near_prior_mean(self, sampler_and_idata):
        _, idata = sampler_and_idata
        post_mean = float(idata.posterior["x"].mean())
        # For a Normal prior with no data, posterior == prior
        assert abs(post_mean - 3.0) < 0.5

    def test_posterior_std_near_prior_std(self, sampler_and_idata):
        _, idata = sampler_and_idata
        post_std = float(idata.posterior["x"].std())
        assert abs(post_std - 1.0) < 0.5


# ---------------------------------------------------------------------------
# Compare to exact inference
# ---------------------------------------------------------------------------

class TestCompareToExact:
    """For a Normal prior with no likelihood, the posterior equals the prior.

    We can compare MCMC samples to exact prior moments.
    """

    @pytest.fixture(scope="class")
    def idata(self):
        bn = _simple_normal_network(mu=0.0, sigma=1.0)
        sampler = MCMCSampler.from_network(bn)
        return sampler.sample(
            n_samples=2000,
            tune=500,
            chains=2,
            random_seed=99,
        )

    def test_mean_matches_exact(self, idata):
        post_mean = float(idata.posterior["x"].mean())
        assert abs(post_mean - 0.0) < 0.2

    def test_std_matches_exact(self, idata):
        post_std = float(idata.posterior["x"].std())
        assert abs(post_std - 1.0) < 0.3

    def test_quantiles_match_exact(self, idata):
        """10th and 90th percentile should be near ±1.28."""
        samples = idata.posterior["x"].values.flatten()
        q10 = np.percentile(samples, 10)
        q90 = np.percentile(samples, 90)
        # Exact: N(0,1) → q10 ≈ -1.28, q90 ≈ 1.28
        assert abs(q10 - (-1.28)) < 0.3
        assert abs(q90 - 1.28) < 0.3


# ---------------------------------------------------------------------------
# Multi-node sampling
# ---------------------------------------------------------------------------

class TestMultiNodeSampling:
    """Test sampling from a network with multiple independent nodes."""

    @pytest.fixture(scope="class")
    def sampler_and_idata(self):
        bn = _two_node_network()
        sampler = MCMCSampler.from_network(bn)
        idata = sampler.sample(
            n_samples=1000,
            tune=500,
            chains=2,
            random_seed=77,
        )
        return sampler, idata

    def test_both_vars_present(self, sampler_and_idata):
        _, idata = sampler_and_idata
        assert "mu" in idata.posterior
        assert "sigma_proxy" in idata.posterior

    def test_mu_mean(self, sampler_and_idata):
        _, idata = sampler_and_idata
        post_mean = float(idata.posterior["mu"].mean())
        assert abs(post_mean - 0.0) < 2.0  # wide prior

    def test_sigma_proxy_mean(self, sampler_and_idata):
        _, idata = sampler_and_idata
        post_mean = float(idata.posterior["sigma_proxy"].mean())
        assert abs(post_mean - 1.0) < 0.5

    def test_diagnostics_both_vars(self, sampler_and_idata):
        sampler, _ = sampler_and_idata
        diag = sampler.diagnostics()
        assert "mu" in diag
        assert "sigma_proxy" in diag
        for var in ("mu", "sigma_proxy"):
            assert abs(diag[var]["r_hat"] - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error paths."""

    def test_diagnostics_before_sample_raises(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        with pytest.raises(RuntimeError, match="No samples"):
            sampler.diagnostics()

    def test_r_hat_before_sample_raises(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        with pytest.raises(RuntimeError, match="No samples"):
            sampler.r_hat()

    def test_ess_before_sample_raises(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        with pytest.raises(RuntimeError, match="No samples"):
            sampler.ess()

    def test_trace_plot_before_sample_raises(self):
        bn = _simple_normal_network()
        sampler = MCMCSampler.from_network(bn)
        with pytest.raises(RuntimeError, match="No samples"):
            sampler.trace_plot()
