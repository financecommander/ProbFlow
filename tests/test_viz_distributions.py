"""Tests for probflow.viz.distributions."""

import os
import tempfile

import numpy as np
import pytest

from probflow.distributions.continuous import Beta, LogNormal, Normal
from probflow.viz.distributions import (
    COLORBLIND_SAFE_PALETTE,
    plot_distribution,
    tornado_chart,
)


# ------------------------------------------------------------------ helpers --

def _normal():
    return Normal(0.0, 1.0)


def _lognormal():
    return LogNormal(0.0, 0.5)


def _beta():
    return Beta(2.0, 5.0)


# ----------------------------------------------------------- smoke tests --


class TestPlotDistributionSmoke:
    """Smoke tests: calls must not crash."""

    def test_single_pdf(self):
        fig = plot_distribution(_normal(), labels="std normal")
        assert fig is not None

    def test_single_cdf(self):
        fig = plot_distribution(_normal(), kind="cdf")
        assert fig is not None

    def test_both_panels(self):
        fig = plot_distribution(_normal(), kind="both")
        assert fig is not None

    def test_comparison_mode(self):
        fig = plot_distribution(
            [_normal(), Normal(2.0, 0.5), _beta()],
            labels=["N(0,1)", "N(2,0.5)", "Beta(2,5)"],
        )
        assert fig is not None

    def test_comparison_cdf(self):
        fig = plot_distribution(
            [_normal(), _lognormal()],
            labels=["Normal", "LogNormal"],
            kind="cdf",
        )
        assert fig is not None

    def test_comparison_both(self):
        fig = plot_distribution(
            [_normal(), _lognormal()],
            kind="both",
        )
        assert fig is not None

    def test_custom_quantiles(self):
        fig = plot_distribution(_normal(), quantiles=[0.1, 0.5, 0.9])
        assert fig is not None

    def test_even_quantiles(self):
        fig = plot_distribution(_normal(), quantiles=[0.25, 0.75])
        assert fig is not None

    def test_no_labels_uses_repr(self):
        fig = plot_distribution(_normal())
        assert fig is not None

    def test_title_and_labels(self):
        fig = plot_distribution(
            _normal(),
            title="My Plot",
            xlabel="Value",
            ylabel="Prob Density",
        )
        assert fig is not None

    def test_existing_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_distribution(_normal(), ax=ax)
        # When existing axes are passed, returns the axes
        assert result is ax
        plt.close(fig)

    def test_lognormal_pdf(self):
        fig = plot_distribution(_lognormal(), labels="LN(0,0.5)")
        assert fig is not None

    def test_beta_pdf(self):
        fig = plot_distribution(_beta(), labels="Beta(2,5)")
        assert fig is not None


class TestPlotDistributionInteractive:
    """Smoke tests for plotly interactive mode."""

    def test_interactive_single_pdf(self):
        fig = plot_distribution(_normal(), interactive=True)
        assert fig is not None

    def test_interactive_cdf(self):
        fig = plot_distribution(_normal(), kind="cdf", interactive=True)
        assert fig is not None

    def test_interactive_both(self):
        fig = plot_distribution(_normal(), kind="both", interactive=True)
        assert fig is not None

    def test_interactive_comparison(self):
        fig = plot_distribution(
            [_normal(), _lognormal()],
            labels=["N(0,1)", "LN(0,0.5)"],
            interactive=True,
        )
        assert fig is not None


# --------------------------------------------------------- tornado chart --


class TestTornadoChartSmoke:
    """Smoke tests for tornado_chart."""

    def test_basic(self):
        data = {"alpha": 3.0, "beta": -1.5, "gamma": 0.8}
        fig = tornado_chart(data)
        assert fig is not None

    def test_threshold_filters(self):
        data = {"a": 5.0, "b": 0.05, "c": -2.0}
        fig = tornado_chart(data, threshold=0.1)
        assert fig is not None

    def test_threshold_too_high_shows_all(self):
        data = {"a": 0.01, "b": 0.02}
        fig = tornado_chart(data, threshold=100.0)
        assert fig is not None

    def test_custom_title(self):
        data = {"x": 1.0}
        fig = tornado_chart(data, title="My Tornado", xlabel="Impact")
        assert fig is not None

    def test_existing_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = {"a": 2.0, "b": -1.0}
        result = tornado_chart(data, ax=ax)
        assert result is ax
        plt.close(fig)


class TestTornadoChartInteractive:
    """Smoke tests for plotly tornado chart."""

    def test_interactive_basic(self):
        data = {"alpha": 3.0, "beta": -1.5}
        fig = tornado_chart(data, interactive=True)
        assert fig is not None

    def test_interactive_title(self):
        data = {"x": 1.0, "y": -0.5}
        fig = tornado_chart(data, interactive=True, title="Sensitivity")
        assert fig is not None


# ----------------------------------------------------- visual regression --


class TestVisualRegression:
    """Save baseline PNGs and verify they are created without errors."""

    def test_save_pdf_baseline(self, tmp_path):
        path = str(tmp_path / "baseline_pdf.png")
        fig = plot_distribution(_normal(), labels="N(0,1)", save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_save_cdf_baseline(self, tmp_path):
        path = str(tmp_path / "baseline_cdf.png")
        fig = plot_distribution(_normal(), kind="cdf", save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_save_both_baseline(self, tmp_path):
        path = str(tmp_path / "baseline_both.png")
        fig = plot_distribution(_normal(), kind="both", save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_save_comparison_baseline(self, tmp_path):
        path = str(tmp_path / "baseline_comparison.png")
        fig = plot_distribution(
            [_normal(), Normal(2.0, 0.5), _beta()],
            labels=["N(0,1)", "N(2,0.5)", "Beta(2,5)"],
            save_path=path,
        )
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_save_tornado_baseline(self, tmp_path):
        path = str(tmp_path / "baseline_tornado.png")
        data = {"revenue": 3.5, "costs": -2.1, "discount": 0.8, "volume": 1.2}
        fig = tornado_chart(data, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


# ------------------------------------------------------- accessibility --


class TestAccessibility:
    """Verify colorblind-safe palette properties."""

    def test_palette_length(self):
        assert len(COLORBLIND_SAFE_PALETTE) >= 8

    def test_palette_all_hex(self):
        import re
        for color in COLORBLIND_SAFE_PALETTE:
            assert re.match(r"^#[0-9A-Fa-f]{6}$", color), f"Invalid hex: {color}"

    def test_palette_unique(self):
        assert len(set(COLORBLIND_SAFE_PALETTE)) == len(COLORBLIND_SAFE_PALETTE)

    def test_wong_palette_colors(self):
        """Verify the palette matches the Wong 2011 colorblind-safe palette."""
        # Key colours from the Wong palette
        assert "#0072B2" in COLORBLIND_SAFE_PALETTE  # blue
        assert "#E69F00" in COLORBLIND_SAFE_PALETTE  # orange
        assert "#009E73" in COLORBLIND_SAFE_PALETTE  # green
        assert "#CC79A7" in COLORBLIND_SAFE_PALETTE  # pink

    def test_plot_uses_distinct_colors(self):
        """Multiple distributions should get different palette colours."""
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_distribution(
            [_normal(), Normal(2.0, 0.5), _lognormal()],
            labels=["A", "B", "C"],
        )
        ax = fig.axes[0]
        line_colors = [line.get_color() for line in ax.lines
                       if not line.get_linestyle().startswith("--")]
        # All visible lines should have distinct colours
        assert len(set(line_colors)) >= 3
