"""Tests for probflow.distributions.conditional and BeliefNetwork integration."""

import numpy as np
import pytest

from probflow.distributions.conditional import ConditionalDist
from probflow.distributions.continuous import Normal
from probflow.distributions.discrete import Categorical
from probflow.inference.belief_network import BeliefNetwork


# ============================================================ ConditionalDist


class TestConditionalDistDiscrete:
    """ConditionalDist with a discrete (Categorical) parent."""

    def _make_regime_vol(self):
        regime = Categorical([0.6, 0.4], labels=["bull", "bear"])
        vol = ConditionalDist(
            parent=regime,
            mapping={
                "bull": Normal(1.0, 0.3),
                "bear": Normal(2.0, 0.5),
            },
        )
        return regime, vol

    # -- construction --

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ConditionalDist(parent=None, mapping={})

    def test_non_dict_mapping_raises(self):
        with pytest.raises(TypeError, match="dict"):
            ConditionalDist(parent=None, mapping=[(1, Normal())])

    # -- sampling shape --

    def test_sample_shape(self):
        regime, vol = self._make_regime_vol()
        parent_samples = regime.sample(500)
        child_samples = vol.sample(500, parent_samples=parent_samples)
        assert child_samples.shape == (500,)

    def test_sample_auto_parent(self):
        """When parent_samples is None, samples are drawn from parent."""
        regime, vol = self._make_regime_vol()
        child_samples = vol.sample(500)
        assert child_samples.shape == (500,)

    def test_sample_length_mismatch_raises(self):
        regime, vol = self._make_regime_vol()
        with pytest.raises(ValueError, match="length"):
            vol.sample(10, parent_samples=regime.sample(5))

    # -- sampling consistency: conditioned means --

    def test_conditioned_mean_bull(self):
        """Samples conditioned on 'bull' should cluster around mu=1."""
        _, vol = self._make_regime_vol()
        bull_parents = np.array(["bull"] * 10_000)
        samples = vol.sample(10_000, parent_samples=bull_parents)
        assert abs(samples.mean() - 1.0) < 0.05

    def test_conditioned_mean_bear(self):
        """Samples conditioned on 'bear' should cluster around mu=2."""
        _, vol = self._make_regime_vol()
        bear_parents = np.array(["bear"] * 10_000)
        samples = vol.sample(10_000, parent_samples=bear_parents)
        assert abs(samples.mean() - 2.0) < 0.05

    def test_conditioned_std_bull(self):
        """Std dev of bull-conditioned samples ≈ 0.3."""
        _, vol = self._make_regime_vol()
        bull_parents = np.array(["bull"] * 10_000)
        samples = vol.sample(10_000, parent_samples=bull_parents)
        assert abs(samples.std() - 0.3) < 0.05

    # -- parent-child correlation --

    def test_parent_child_correlation(self):
        """Bear samples should be systematically higher than bull samples."""
        regime, vol = self._make_regime_vol()
        n = 20_000
        parent_samples = regime.sample(n)
        child_samples = vol.sample(n, parent_samples=parent_samples)

        bull_mask = parent_samples == "bull"
        bear_mask = parent_samples == "bear"

        bull_mean = child_samples[bull_mask].mean()
        bear_mean = child_samples[bear_mask].mean()
        assert bear_mean > bull_mean + 0.5

    # -- unknown parent value --

    def test_unknown_parent_value_raises(self):
        _, vol = self._make_regime_vol()
        unknown = np.array(["sideways"] * 5)
        with pytest.raises(ValueError, match="not found"):
            vol.sample(5, parent_samples=unknown)

    # -- get_child_dist --

    def test_get_child_dist(self):
        _, vol = self._make_regime_vol()
        d = vol.get_child_dist("bull")
        assert isinstance(d, Normal)
        assert d.mu == 1.0


class TestConditionalDistContinuous:
    """ConditionalDist with a continuous (binned) parent."""

    def _make_binned(self):
        parent = Normal(0.0, 1.0)
        mapping = {
            0.0: Normal(10.0, 1.0),   # parent <= 0
            1.0: Normal(20.0, 1.0),   # 0 < parent <= 1
        }
        # bin edges at [0.0, 1.0]
        # digitize: <=0 → bin 0, (0,1] → bin 1, >1 → clamped to bin 1
        return parent, ConditionalDist(parent=parent, mapping=mapping)

    def test_sample_shape(self):
        parent, cond = self._make_binned()
        samples = cond.sample(500, parent_samples=parent.sample(500))
        assert samples.shape == (500,)

    def test_low_bin_mean(self):
        """Parent values ≤ 0 should produce child mean ≈ 10."""
        _, cond = self._make_binned()
        low_parents = np.full(10_000, -1.0)
        samples = cond.sample(10_000, parent_samples=low_parents)
        assert abs(samples.mean() - 10.0) < 0.1

    def test_high_bin_mean(self):
        """Parent values in (0, 1] should produce child mean ≈ 20."""
        _, cond = self._make_binned()
        mid_parents = np.full(10_000, 0.5)
        samples = cond.sample(10_000, parent_samples=mid_parents)
        assert abs(samples.mean() - 20.0) < 0.1

    def test_get_child_dist_continuous(self):
        _, cond = self._make_binned()
        d = cond.get_child_dist(-5.0)
        assert d.mu == 10.0


class TestConditionalDistIntegerKeys:
    """ConditionalDist with integer parent keys (discrete mapping)."""

    def test_integer_keys(self):
        """Integer keys are continuous (binned); values below first edge use bin 0."""
        parent = Normal(0.0, 1.0)
        cond = ConditionalDist(
            parent=parent,
            mapping={0: Normal(0.0, 1.0), 1: Normal(10.0, 1.0)},
        )
        # All parent values < 0 → bin 0 → Normal(0, 1)
        parents = np.full(5000, -1.0)
        samples = cond.sample(5000, parent_samples=parents)
        assert abs(samples.mean() - 0.0) < 0.1

    def test_integer_keys_are_continuous(self):
        """Integer keys are treated as continuous (binned) parent."""
        cond = ConditionalDist(
            parent=Normal(),
            mapping={0: Normal(0.0, 1.0), 1: Normal(10.0, 1.0)},
        )
        assert cond._continuous_parent is True


# ============================================================ BeliefNetwork


class TestBeliefNetwork:
    """Tests for BeliefNetwork CPD storage and ancestral sampling."""

    def _build_simple_network(self):
        bn = BeliefNetwork()
        regime = Categorical([0.6, 0.4], labels=["bull", "bear"])
        bn.add_node("regime", regime)
        vol = ConditionalDist(
            parent=regime,
            mapping={
                "bull": Normal(1.0, 0.3),
                "bear": Normal(2.0, 0.5),
            },
        )
        bn.add_node("vol", vol, parents=["regime"])
        return bn

    def test_nodes(self):
        bn = self._build_simple_network()
        assert bn.nodes == ["regime", "vol"]

    def test_get_parents(self):
        bn = self._build_simple_network()
        assert bn.get_parents("regime") == []
        assert bn.get_parents("vol") == ["regime"]

    def test_duplicate_node_raises(self):
        bn = BeliefNetwork()
        bn.add_node("x", Normal())
        with pytest.raises(ValueError, match="already exists"):
            bn.add_node("x", Normal())

    def test_missing_parent_raises(self):
        bn = BeliefNetwork()
        with pytest.raises(ValueError, match="must be added before"):
            bn.add_node("child", Normal(), parents=["nonexistent"])

    def test_sample_shape(self):
        bn = self._build_simple_network()
        samples = bn.sample(1000)
        assert set(samples.keys()) == {"regime", "vol"}
        assert samples["regime"].shape == (1000,)
        assert samples["vol"].shape == (1000,)

    def test_sample_parent_child_correlation(self):
        """Ancestral samples should preserve the CPD relationship."""
        bn = self._build_simple_network()
        samples = bn.sample(20_000)

        bull_mask = samples["regime"] == "bull"
        bear_mask = samples["regime"] == "bear"

        bull_vol_mean = samples["vol"][bull_mask].mean()
        bear_vol_mean = samples["vol"][bear_mask].mean()

        assert abs(bull_vol_mean - 1.0) < 0.05
        assert abs(bear_vol_mean - 2.0) < 0.05

    def test_sample_marginal_proportions(self):
        """Check that the regime proportions are roughly 60/40."""
        bn = self._build_simple_network()
        samples = bn.sample(50_000)
        bull_frac = (samples["regime"] == "bull").mean()
        assert abs(bull_frac - 0.6) < 0.02

    def test_repr(self):
        bn = self._build_simple_network()
        assert "regime" in repr(bn)
        assert "vol" in repr(bn)

    def test_three_level_network(self):
        """Three-level chain: regime → vol → spread."""
        bn = BeliefNetwork()
        regime = Categorical([0.5, 0.5], labels=["bull", "bear"])
        bn.add_node("regime", regime)

        vol = ConditionalDist(
            parent=regime,
            mapping={
                "bull": Normal(1.0, 0.1),
                "bear": Normal(3.0, 0.1),
            },
        )
        bn.add_node("vol", vol, parents=["regime"])

        # spread conditioned on vol (continuous parent, binned)
        spread = ConditionalDist(
            parent=vol,
            mapping={
                2.0: Normal(100.0, 5.0),   # vol <= 2  (bull regime)
            },
        )
        bn.add_node("spread", spread, parents=["vol"])

        samples = bn.sample(5000)
        assert samples["spread"].shape == (5000,)
