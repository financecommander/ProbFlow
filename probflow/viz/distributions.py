"""Distribution visualization utilities.

Provides ``plot_distribution`` for PDF/CDF overlay plots with shaded quantile
regions and ``tornado_chart`` for horizontal-bar sensitivity displays.

Both functions support a static *matplotlib* backend (default) and an
interactive *plotly* backend for web dashboards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong 2011, widely recommended for accessibility)
# ---------------------------------------------------------------------------
COLORBLIND_SAFE_PALETTE: List[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
]


def _get_color(index: int) -> str:
    """Return a color from the colorblind-safe palette (wraps around)."""
    return COLORBLIND_SAFE_PALETTE[index % len(COLORBLIND_SAFE_PALETTE)]


# ---------------------------------------------------------------------------
# Internal helpers for x-range computation
# ---------------------------------------------------------------------------


def _x_range_for_dist(dist: Any, margin: float = 4.0) -> Tuple[float, float]:
    """Compute a sensible x-range by querying the distribution's quantile."""
    try:
        lo = float(dist.quantile(0.001))
        hi = float(dist.quantile(0.999))
    except Exception:
        # Fallback: use mean ± margin * sqrt(variance)
        mu = float(dist.mean())
        sd = max(float(np.sqrt(dist.variance())), 1e-6)
        lo, hi = mu - margin * sd, mu + margin * sd
    pad = max((hi - lo) * 0.05, 1e-6)
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# plot_distribution  (matplotlib + plotly)
# ---------------------------------------------------------------------------


def plot_distribution(
    dists: Any,
    labels: Optional[Union[str, Sequence[str]]] = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    *,
    kind: str = "pdf",
    interactive: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    n_points: int = 500,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
) -> Any:
    """Plot one or more distributions with shaded quantile regions.

    Parameters
    ----------
    dists : distribution or list of distributions
        Each distribution must expose ``pdf(x)``, ``cdf(x)``, and
        ``quantile(q)`` methods (the standard ProbFlow interface).
    labels : str or list of str, optional
        Legend labels for each distribution.  Defaults to ``repr(dist)``.
    quantiles : sequence of float
        Quantile levels to mark.  Pairs of quantiles (sorted) define
        shaded regions; the middle quantile (if odd count) is drawn as a
        vertical line.
    kind : ``"pdf"`` | ``"cdf"`` | ``"both"``
        Which curve(s) to draw.
    interactive : bool
        If *True*, produce a Plotly figure instead of matplotlib.
    title, xlabel, ylabel : str, optional
        Axis labels / title.
    n_points : int
        Number of evaluation points for the curve.
    ax : matplotlib Axes, optional
        Pre-existing axes to draw on (ignored in interactive mode).
    figsize : tuple
        Figure size when creating a new matplotlib figure.
    save_path : str, optional
        If given, save the figure to this path (matplotlib only).

    Returns
    -------
    matplotlib Figure **or** plotly Figure, depending on *interactive*.
    """
    # Normalise inputs to lists
    if not isinstance(dists, (list, tuple)):
        dists = [dists]
    if labels is None:
        labels = [repr(d) for d in dists]
    elif isinstance(labels, str):
        labels = [labels]
    if len(labels) < len(dists):
        labels = list(labels) + [repr(d) for d in dists[len(labels):]]

    if interactive:
        return _plot_distribution_plotly(
            dists, labels, quantiles, kind, title, xlabel, ylabel, n_points
        )
    return _plot_distribution_mpl(
        dists, labels, quantiles, kind, title, xlabel, ylabel,
        n_points, ax, figsize, save_path,
    )


# ---------------------------------------------------------------------------
# matplotlib backend
# ---------------------------------------------------------------------------


def _plot_distribution_mpl(
    dists, labels, quantiles, kind, title, xlabel, ylabel,
    n_points, ax, figsize, save_path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created_fig = False
    if kind == "both":
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        ax_pdf, ax_cdf = axes
        created_fig = True
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        else:
            fig = ax.figure

    for idx, (dist, label) in enumerate(zip(dists, labels)):
        color = _get_color(idx)
        x_lo, x_hi = _x_range_for_dist(dist)
        xs = np.linspace(x_lo, x_hi, n_points)

        if kind in ("pdf", "both"):
            target_ax = ax_pdf if kind == "both" else ax
            ys = dist.pdf(xs)
            target_ax.plot(xs, ys, color=color, label=label, linewidth=1.5)
            _shade_quantiles_mpl(target_ax, dist, xs, ys, quantiles, color)

        if kind in ("cdf", "both"):
            target_ax = ax_cdf if kind == "both" else ax
            ys = dist.cdf(xs)
            target_ax.plot(xs, ys, color=color, label=label, linewidth=1.5)
            _mark_quantile_lines_mpl(target_ax, dist, quantiles, color)

    # Decorate axes
    for a in ([ax_pdf, ax_cdf] if kind == "both" else [ax]):
        a.legend(frameon=False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    if kind == "both":
        ax_pdf.set_ylabel(ylabel or "Density")
        ax_cdf.set_ylabel("Cumulative probability")
        for a in [ax_pdf, ax_cdf]:
            a.set_xlabel(xlabel or "x")
        if title:
            fig.suptitle(title)
    else:
        default_ylabel = "Density" if kind == "pdf" else "Cumulative probability"
        ax.set_ylabel(ylabel or default_ylabel)
        ax.set_xlabel(xlabel or "x")
        if title:
            ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if created_fig:
        return fig
    return ax


def _shade_quantiles_mpl(ax, dist, xs, ys, quantiles, color):
    """Shade inter-quantile regions and mark median-like lines on a PDF axis."""
    sorted_q = sorted(quantiles)
    # Shade pairs of quantiles (outer to inner)
    alpha_step = 0.10
    i = 0
    while i + 1 < len(sorted_q):
        q_lo = float(dist.quantile(sorted_q[i]))
        q_hi = float(dist.quantile(sorted_q[-(i + 1)]))
        if q_lo < q_hi:
            mask = (xs >= q_lo) & (xs <= q_hi)
            ax.fill_between(xs[mask], ys[mask], alpha=alpha_step * (i + 1) + 0.08,
                            color=color, linewidth=0)
        i += 1
    # If there's a middle quantile, draw a vertical line
    if len(sorted_q) % 2 == 1:
        mid_q = sorted_q[len(sorted_q) // 2]
        x_mid = float(dist.quantile(mid_q))
        ax.axvline(x_mid, color=color, linestyle="--", linewidth=1, alpha=0.8)


def _mark_quantile_lines_mpl(ax, dist, quantiles, color):
    """Draw horizontal guide lines at quantile levels on a CDF axis."""
    for q in quantiles:
        x_q = float(dist.quantile(q))
        ax.axhline(q, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.plot(x_q, q, "o", color=color, markersize=4)


# ---------------------------------------------------------------------------
# plotly backend
# ---------------------------------------------------------------------------


def _plot_distribution_plotly(
    dists, labels, quantiles, kind, title, xlabel, ylabel, n_points,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive mode. "
            "Install it with: pip install probflow[interactive]"
        ) from exc

    if kind == "both":
        fig = make_subplots(rows=1, cols=2, subplot_titles=("PDF", "CDF"))
    else:
        fig = go.Figure()

    for idx, (dist, label) in enumerate(zip(dists, labels)):
        color = _get_color(idx)
        x_lo, x_hi = _x_range_for_dist(dist)
        xs = np.linspace(x_lo, x_hi, n_points)

        if kind in ("pdf", "both"):
            ys = dist.pdf(xs)
            col = 1 if kind == "both" else None
            row = 1 if kind == "both" else None
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", name=label,
                           line=dict(color=color, width=2)),
                row=row, col=col,
            )
            _shade_quantiles_plotly(fig, dist, xs, ys, quantiles, color, label,
                                   row=row, col=col)

        if kind in ("cdf", "both"):
            ys = dist.cdf(xs)
            col = 2 if kind == "both" else None
            row = 1 if kind == "both" else None
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", name=label,
                           line=dict(color=color, width=2),
                           showlegend=(kind != "both")),
                row=row, col=col,
            )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or "x",
        yaxis_title=ylabel or ("Density" if kind == "pdf" else "Cumulative probability"),
        template="plotly_white",
    )
    return fig


def _shade_quantiles_plotly(fig, dist, xs, ys, quantiles, color, label,
                            row=None, col=None):
    """Add shaded quantile bands to a plotly figure."""
    import plotly.graph_objects as go

    sorted_q = sorted(quantiles)
    i = 0
    while i + 1 < len(sorted_q):
        q_lo_val = float(dist.quantile(sorted_q[i]))
        q_hi_val = float(dist.quantile(sorted_q[-(i + 1)]))
        if q_lo_val < q_hi_val:
            mask = (xs >= q_lo_val) & (xs <= q_hi_val)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xs[mask], xs[mask][::-1]]),
                    y=np.concatenate([ys[mask], np.zeros_like(ys[mask])]),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15 * (i + 1),
                    line=dict(width=0),
                    showlegend=False,
                    name=f"{label} q{sorted_q[i]:.0%}-{sorted_q[-(i+1)]:.0%}",
                ),
                row=row, col=col,
            )
        i += 1


# ---------------------------------------------------------------------------
# tornado_chart
# ---------------------------------------------------------------------------


def tornado_chart(
    sensitivity_dict: Dict[str, float],
    threshold: float = 0.1,
    *,
    interactive: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """Horizontal bar chart of sensitivity coefficients.

    Parameters
    ----------
    sensitivity_dict : dict
        ``{parameter_name: sensitivity_value}`` as returned by
        ``probflow.inference.sensitivity.sensitivity_analysis``.
    threshold : float
        Only show parameters whose absolute sensitivity exceeds this value.
    interactive : bool
        If *True*, produce a Plotly figure.
    title : str, optional
        Chart title.
    xlabel : str, optional
        X-axis label.
    figsize : tuple
        Figure size (matplotlib only).
    save_path : str, optional
        Save to file (matplotlib only).
    ax : matplotlib Axes, optional
        Pre-existing axes (matplotlib only).

    Returns
    -------
    matplotlib Figure **or** plotly Figure.
    """
    # Filter and sort
    filtered = {
        k: v for k, v in sensitivity_dict.items()
        if abs(float(v)) >= threshold
    }
    if not filtered:
        filtered = dict(
            sorted(sensitivity_dict.items(),
                   key=lambda kv: abs(float(kv[1])), reverse=True)
        )

    sorted_items = sorted(filtered.items(), key=lambda kv: abs(float(kv[1])))
    names = [item[0] for item in sorted_items]
    values = [float(item[1]) for item in sorted_items]

    if interactive:
        return _tornado_plotly(names, values, title, xlabel)
    return _tornado_mpl(names, values, title, xlabel, figsize, save_path, ax)


def _tornado_mpl(names, values, title, xlabel, figsize, save_path, ax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    colors = [_get_color(0) if v >= 0 else _get_color(5) for v in values]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel or "Sensitivity")
    ax.set_title(title or "Tornado Chart")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if created_fig:
        return fig
    return ax


def _tornado_plotly(names, values, title, xlabel):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive mode. "
            "Install it with: pip install probflow[interactive]"
        ) from exc

    colors = [_get_color(0) if v >= 0 else _get_color(5) for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=title or "Tornado Chart",
        xaxis_title=xlabel or "Sensitivity",
        template="plotly_white",
    )
    return fig
