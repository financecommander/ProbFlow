"""Tests for probflow.distributions.continuous."""

import math

import numpy as np
import pytest
from scipy import stats

from probflow.distributions.continuous import Beta, LogNormal, Normal


# ------------------------------------------------------------------ Normal --


class TestNormal:
    """Tests for the Normal distribution."""

    def test_mean_variance(self):
        d = Normal(3.0, 2.0)
        assert d.mean() == 3.0
        assert d.variance() == 4.0

    def test_pdf_matches_scipy(self):
        d = Normal(1.0, 2.0)
        xs = np.linspace(-5, 7, 50)
        np.testing.assert_allclose(d.pdf(xs), stats.norm.pdf(xs, 1.0, 2.0))

    def test_cdf_matches_scipy(self):
        d = Normal(1.0, 2.0)
        xs = np.linspace(-5, 7, 50)
        np.testing.assert_allclose(d.cdf(xs), stats.norm.cdf(xs, 1.0, 2.0))

    def test_quantile_matches_scipy(self):
        d = Normal(0.0, 1.0)
        qs = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        np.testing.assert_allclose(d.quantile(qs), stats.norm.ppf(qs, 0.0, 1.0))

    def test_sample_shape(self):
        d = Normal(0.0, 1.0)
        assert d.sample(100).shape == (100,)

    def test_sample_mean_converges(self):
        d = Normal(5.0, 1.0)
        samples = d.sample(50_000)
        assert abs(samples.mean() - 5.0) < 0.05

    # -- edge case: sigma == 0 (degenerate) --

    def test_sigma_zero_mean_variance(self):
        d = Normal(7.0, 0.0)
        assert d.mean() == 7.0
        assert d.variance() == 0.0

    def test_sigma_zero_cdf(self):
        d = Normal(3.0, 0.0)
        # CDF is a step function at mu for the degenerate distribution
        assert d.cdf(2.9) == 0.0
        assert d.cdf(3.0) == 1.0

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Normal(0, -1)

    # -- operator algebra --

    def test_add_normal(self):
        a = Normal(1.0, 2.0)
        b = Normal(3.0, 4.0)
        c = a + b
        assert isinstance(c, Normal)
        assert c.mean() == pytest.approx(4.0)
        assert c.variance() == pytest.approx(20.0)

    def test_add_returns_notimplemented(self):
        assert Normal().__add__("not a normal") is NotImplemented

    def test_mul_scalar(self):
        d = Normal(2.0, 3.0)
        scaled = d * 5
        assert isinstance(scaled, Normal)
        assert scaled.mean() == pytest.approx(10.0)
        assert scaled.variance() == pytest.approx(225.0)

    def test_rmul_scalar(self):
        d = Normal(2.0, 3.0)
        scaled = 5 * d
        assert scaled.mean() == pytest.approx(10.0)

    def test_mul_negative_scalar(self):
        d = Normal(2.0, 3.0)
        scaled = d * (-2)
        assert scaled.mean() == pytest.approx(-4.0)
        assert scaled.sigma == pytest.approx(6.0)  # |c| * sigma

    def test_mul_returns_notimplemented(self):
        assert Normal().__mul__("bad") is NotImplemented

    def test_repr(self):
        assert "Normal" in repr(Normal(1, 2))


# --------------------------------------------------------------- LogNormal --


class TestLogNormal:
    """Tests for the LogNormal distribution."""

    def test_mean_variance(self):
        d = LogNormal(0.0, 1.0)
        expected_mean = np.exp(0.0 + 0.5)
        expected_var = (np.exp(1.0) - 1) * np.exp(2 * 0.0 + 1.0)
        assert d.mean() == pytest.approx(expected_mean)
        assert d.variance() == pytest.approx(expected_var)

    def test_pdf_matches_scipy(self):
        d = LogNormal(1.0, 0.5)
        xs = np.linspace(0.01, 10, 50)
        expected = stats.lognorm.pdf(xs, s=0.5, scale=np.exp(1.0))
        np.testing.assert_allclose(d.pdf(xs), expected)

    def test_cdf_matches_scipy(self):
        d = LogNormal(1.0, 0.5)
        xs = np.linspace(0.01, 10, 50)
        expected = stats.lognorm.cdf(xs, s=0.5, scale=np.exp(1.0))
        np.testing.assert_allclose(d.cdf(xs), expected)

    def test_quantile_matches_scipy(self):
        d = LogNormal(0.0, 1.0)
        qs = np.array([0.1, 0.5, 0.9])
        expected = stats.lognorm.ppf(qs, s=1.0, scale=np.exp(0.0))
        np.testing.assert_allclose(d.quantile(qs), expected)

    def test_sample_shape(self):
        d = LogNormal(0.0, 1.0)
        assert d.sample(200).shape == (200,)

    def test_sigma_zero(self):
        d = LogNormal(2.0, 0.0)
        assert d.mean() == pytest.approx(np.exp(2.0))
        assert d.variance() == pytest.approx(0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            LogNormal(0, -1)

    # -- operator algebra --

    def test_mul_positive_scalar(self):
        d = LogNormal(1.0, 0.5)
        scaled = d * 3.0
        assert isinstance(scaled, LogNormal)
        assert scaled.mu == pytest.approx(1.0 + np.log(3.0))
        assert scaled.sigma == pytest.approx(0.5)

    def test_rmul(self):
        d = LogNormal(0.0, 1.0)
        scaled = 2.0 * d
        assert scaled.mu == pytest.approx(np.log(2.0))

    def test_mul_nonpositive_raises(self):
        with pytest.raises(ValueError, match="positive constant"):
            LogNormal() * (-1)

    def test_mul_returns_notimplemented(self):
        assert LogNormal().__mul__("bad") is NotImplemented

    def test_repr(self):
        assert "LogNormal" in repr(LogNormal(1, 2))


# ------------------------------------------------------------------- Beta --


class TestBeta:
    """Tests for the Beta distribution."""

    def test_mean_variance(self):
        d = Beta(2.0, 5.0)
        expected_mean = 2.0 / 7.0
        expected_var = (2.0 * 5.0) / (49.0 * 8.0)
        assert d.mean() == pytest.approx(expected_mean)
        assert d.variance() == pytest.approx(expected_var)

    def test_pdf_matches_scipy(self):
        d = Beta(2.0, 5.0)
        xs = np.linspace(0.01, 0.99, 50)
        np.testing.assert_allclose(d.pdf(xs), stats.beta.pdf(xs, 2.0, 5.0))

    def test_cdf_matches_scipy(self):
        d = Beta(2.0, 5.0)
        xs = np.linspace(0.01, 0.99, 50)
        np.testing.assert_allclose(d.cdf(xs), stats.beta.cdf(xs, 2.0, 5.0))

    def test_quantile_matches_scipy(self):
        d = Beta(2.0, 5.0)
        qs = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(d.quantile(qs), stats.beta.ppf(qs, 2.0, 5.0))

    def test_sample_shape(self):
        d = Beta(1.0, 1.0)
        assert d.sample(300).shape == (300,)

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="positive"):
            Beta(0, 1)
        with pytest.raises(ValueError, match="positive"):
            Beta(1, -1)

    # -- operator algebra --

    def test_mul_scalar(self):
        d = Beta(2.0, 3.0)
        scaled = d * 10
        assert isinstance(scaled, Beta)
        assert scaled.mean() == pytest.approx(10 * 2.0 / 5.0)

    def test_rmul(self):
        d = Beta(2.0, 3.0)
        scaled = 10 * d
        assert scaled.mean() == pytest.approx(10 * 2.0 / 5.0)

    def test_mul_returns_notimplemented(self):
        assert Beta().__mul__("bad") is NotImplemented

    def test_repr(self):
        assert "Beta" in repr(Beta(1, 2))
