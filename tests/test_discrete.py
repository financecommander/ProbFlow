"""Tests for discrete probability distributions."""

import numpy as np
import pytest

from probflow.distributions.discrete import (
    Bernoulli,
    Categorical,
    Poisson,
    probability,
)


# ---------------------------------------------------------------------------
# Bernoulli tests
# ---------------------------------------------------------------------------


class TestBernoulli:
    def test_sample_shape(self):
        b = Bernoulli(0.5)
        samples = b.sample(1000)
        assert samples.shape == (1000,)
        assert set(np.unique(samples)).issubset({0, 1})

    def test_pmf(self):
        b = Bernoulli(0.3)
        assert pytest.approx(b.pmf(1)) == 0.3
        assert pytest.approx(b.pmf(0)) == 0.7

    def test_cdf(self):
        b = Bernoulli(0.4)
        assert pytest.approx(b.cdf(0)) == 0.6
        assert pytest.approx(b.cdf(1)) == 1.0

    def test_mode(self):
        assert Bernoulli(0.8).mode() == 1
        assert Bernoulli(0.2).mode() == 0
        assert Bernoulli(0.5).mode() == 1  # convention

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            Bernoulli(-0.1)
        with pytest.raises(ValueError):
            Bernoulli(1.1)

    # Bernoulli algebra: P(A & B) = P(A) * P(B) under independence
    def test_and_joint_probability(self):
        a = Bernoulli(0.3)
        b = Bernoulli(0.5)
        joint = a & b
        assert isinstance(joint, Bernoulli)
        assert pytest.approx(joint.p) == 0.3 * 0.5

    def test_and_commutativity(self):
        a = Bernoulli(0.4)
        b = Bernoulli(0.7)
        assert pytest.approx((a & b).p) == pytest.approx((b & a).p)

    # P(A or B) = P(A) + P(B) - P(A)*P(B) under independence
    def test_or_union_probability(self):
        a = Bernoulli(0.3)
        b = Bernoulli(0.5)
        union = a | b
        assert isinstance(union, Bernoulli)
        assert pytest.approx(union.p) == 0.3 + 0.5 - 0.3 * 0.5

    def test_or_commutativity(self):
        a = Bernoulli(0.4)
        b = Bernoulli(0.7)
        assert pytest.approx((a | b).p) == pytest.approx((b | a).p)

    def test_repr(self):
        assert "Bernoulli" in repr(Bernoulli(0.5))


# ---------------------------------------------------------------------------
# Poisson tests
# ---------------------------------------------------------------------------


class TestPoisson:
    def test_sample_shape(self):
        p = Poisson(5.0)
        samples = p.sample(1000)
        assert samples.shape == (1000,)
        assert np.all(samples >= 0)

    def test_pmf(self):
        p = Poisson(3.0)
        # PMF at 0 should be e^{-3}
        assert pytest.approx(p.pmf(0), rel=1e-6) == np.exp(-3.0)

    def test_cdf(self):
        p = Poisson(2.0)
        # CDF at a large value should approach 1
        assert pytest.approx(p.cdf(100), abs=1e-10) == 1.0
        # CDF at -1 should be 0
        assert pytest.approx(p.cdf(-1)) == 0.0

    def test_mode(self):
        assert Poisson(3.7).mode() == 3
        assert Poisson(5.0).mode() == 5

    # Poisson mean == variance == lambda
    def test_mean_equals_variance(self):
        lam = 7.0
        p = Poisson(lam)
        samples = p.sample(100_000)
        sample_mean = np.mean(samples)
        sample_var = np.var(samples, ddof=1)
        assert pytest.approx(sample_mean, rel=0.05) == lam
        assert pytest.approx(sample_var, rel=0.05) == lam

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            Poisson(0)
        with pytest.raises(ValueError):
            Poisson(-1)

    def test_repr(self):
        assert "Poisson" in repr(Poisson(3.0))


# ---------------------------------------------------------------------------
# Categorical tests
# ---------------------------------------------------------------------------


class TestCategorical:
    def test_sample_shape(self):
        c = Categorical([0.2, 0.3, 0.5])
        samples = c.sample(1000)
        assert samples.shape == (1000,)
        assert set(np.unique(samples)).issubset({0, 1, 2})

    def test_pmf(self):
        probs = [0.1, 0.4, 0.5]
        c = Categorical(probs)
        for i, p in enumerate(probs):
            assert pytest.approx(c.pmf(i)) == p
        # Out-of-range returns 0
        assert c.pmf(-1) == 0.0
        assert c.pmf(3) == 0.0

    def test_cdf(self):
        c = Categorical([0.2, 0.3, 0.5])
        assert pytest.approx(c.cdf(-1)) == 0.0
        assert pytest.approx(c.cdf(0)) == 0.2
        assert pytest.approx(c.cdf(1)) == 0.5
        assert pytest.approx(c.cdf(2)) == 1.0

    def test_mode(self):
        assert Categorical([0.1, 0.6, 0.3]).mode() == 1

    # sum(probs) must equal 1
    def test_probs_sum_to_one(self):
        probs = [0.25, 0.25, 0.25, 0.25]
        c = Categorical(probs)
        assert pytest.approx(np.sum(c.probs)) == 1.0

    def test_invalid_probs_not_sum_one(self):
        with pytest.raises(ValueError):
            Categorical([0.3, 0.3])

    def test_invalid_probs_negative(self):
        with pytest.raises(ValueError):
            Categorical([-0.1, 0.6, 0.5])

    def test_repr(self):
        assert "Categorical" in repr(Categorical([0.5, 0.5]))


# ---------------------------------------------------------------------------
# probability() helper tests
# ---------------------------------------------------------------------------


class TestProbabilityHelper:
    def test_gt(self):
        p = Poisson(3.0)
        # P(X > 0) = 1 - P(X <= 0) = 1 - CDF(0)
        assert pytest.approx(probability(p, 0, "gt")) == 1.0 - p.cdf(0)

    def test_ge(self):
        p = Poisson(3.0)
        # P(X >= 1) = 1 - P(X <= 0) = 1 - CDF(0)
        assert pytest.approx(probability(p, 1, "ge")) == 1.0 - p.cdf(0)

    def test_lt(self):
        p = Poisson(3.0)
        # P(X < 2) = P(X <= 1) = CDF(1)
        assert pytest.approx(probability(p, 2, "lt")) == p.cdf(1)

    def test_le(self):
        p = Poisson(3.0)
        # P(X <= 2) = CDF(2)
        assert pytest.approx(probability(p, 2, "le")) == p.cdf(2)

    def test_eq(self):
        p = Poisson(3.0)
        assert pytest.approx(probability(p, 3, "eq")) == p.pmf(3)

    def test_invalid_comparison(self):
        with pytest.raises(ValueError):
            probability(Poisson(1.0), 0, "invalid")

    def test_with_bernoulli(self):
        b = Bernoulli(0.7)
        assert pytest.approx(probability(b, 1, "eq")) == 0.7
        assert pytest.approx(probability(b, 0, "eq")) == 0.3
