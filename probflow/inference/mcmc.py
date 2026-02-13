"""MCMC sampling via PyMC NUTS sampler.

Provides :class:`MCMCSampler` which converts a
:class:`~probflow.inference.belief_network.BeliefNetwork` (or a plain
callable log-density) into a PyMC model and draws posterior samples
using the NUTS algorithm.

PyMC and ArviZ are **optional** dependencies and are imported lazily
so that the rest of ProbFlow remains usable without them.

Example
-------
>>> from probflow.inference.mcmc import MCMCSampler
>>> from probflow.inference.belief_network import BeliefNetwork
>>> from probflow.distributions.continuous import Normal
>>>
>>> bn = BeliefNetwork()
>>> bn.add_node("mu", Normal(0, 10))
>>> bn.add_node("sigma", Normal(1, 1))
>>>
>>> sampler = MCMCSampler.from_network(bn)
>>> idata = sampler.sample(1000, tune=500, chains=2)
>>> diagnostics = sampler.diagnostics()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _import_pymc():
    """Lazily import PyMC, raising a clear error if not installed."""
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "PyMC is required for MCMC sampling. "
            "Install it with:  pip install 'probflow[mcmc]'"
        ) from exc
    return pm


def _import_arviz():
    """Lazily import ArviZ, raising a clear error if not installed."""
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "ArviZ is required for MCMC diagnostics. "
            "Install it with:  pip install 'probflow[mcmc]'"
        ) from exc
    return az


# Mapping from ProbFlow distribution class names to PyMC distribution
# constructors. Extend as new ProbFlow distributions are added.
_DIST_MAP = {
    "Normal": lambda pm, name, dist: pm.Normal(
        name, mu=dist.mu, sigma=dist.sigma
    ),
    "Beta": lambda pm, name, dist: pm.Beta(
        name, alpha=dist.alpha, beta=dist.beta
    ),
    "Categorical": lambda pm, name, dist: pm.Categorical(
        name, p=dist.probs
    ),
}


class MCMCSampler:
    """Wrapper around the PyMC NUTS sampler.

    Use :meth:`from_network` to build an ``MCMCSampler`` from a
    :class:`~probflow.inference.belief_network.BeliefNetwork`, or pass
    a pre-built ``pymc.Model`` to the constructor.

    Parameters
    ----------
    model : pymc.Model
        A compiled PyMC model ready for sampling.
    var_names : list of str, optional
        Variable names to track. Defaults to all free RVs.
    """

    def __init__(self, model: Any, var_names: Optional[List[str]] = None):
        self._model = model
        self._var_names = var_names
        self._idata: Optional[Any] = None  # arviz.InferenceData after sample()

    # ------------------------------------------------------------------ #
    #  Construction from a BeliefNetwork
    # ------------------------------------------------------------------ #

    @classmethod
    def from_network(cls, network) -> MCMCSampler:
        """Convert a :class:`BeliefNetwork` to a PyMC model.

        Each node whose distribution class appears in the internal
        mapping is translated to the corresponding PyMC distribution.
        Nodes with unsupported types raise ``ValueError``.

        Parameters
        ----------
        network : BeliefNetwork
            The belief network to convert.

        Returns
        -------
        MCMCSampler
            A sampler wrapping the generated PyMC model.

        Raises
        ------
        ValueError
            If a node's distribution type is not supported.
        """
        pm = _import_pymc()

        var_names: List[str] = []

        with pm.Model() as model:
            for name in network.nodes:
                dist = network.get_dist(name)
                dist_type = type(dist).__name__
                builder = _DIST_MAP.get(dist_type)
                if builder is None:
                    raise ValueError(
                        f"Unsupported distribution type '{dist_type}' "
                        f"for node '{name}'. Supported types: "
                        f"{list(_DIST_MAP.keys())}"
                    )
                builder(pm, name, dist)
                var_names.append(name)

        return cls(model, var_names=var_names)

    # ------------------------------------------------------------------ #
    #  Sampling
    # ------------------------------------------------------------------ #

    def sample(
        self,
        n_samples: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Run NUTS sampling and return an ArviZ ``InferenceData`` object.

        Parameters
        ----------
        n_samples : int
            Number of posterior draws per chain (after tuning).
        tune : int
            Number of tuning (warm-up) steps per chain.
        chains : int
            Number of independent chains to run.
        random_seed : int, optional
            Seed for reproducibility.
        **kwargs
            Extra keyword arguments forwarded to ``pymc.sample()``.

        Returns
        -------
        arviz.InferenceData
            The sampling results including posterior draws and sample
            statistics.
        """
        pm = _import_pymc()

        with self._model:
            self._idata = pm.sample(
                draws=n_samples,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                **kwargs,
            )

        return self._idata

    # ------------------------------------------------------------------ #
    #  Convergence diagnostics
    # ------------------------------------------------------------------ #

    def diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Compute convergence diagnostics for the most recent sample.

        Returns
        -------
        dict
            ``{var_name: {"r_hat": …, "ess_bulk": …, "ess_tail": …}}``
            for each variable.

        Raises
        ------
        RuntimeError
            If :meth:`sample` has not been called yet.
        """
        if self._idata is None:
            raise RuntimeError(
                "No samples available. Call sample() first."
            )

        az = _import_arviz()

        results: Dict[str, Dict[str, float]] = {}
        var_names = self._var_names or list(
            self._idata.posterior.data_vars
        )

        for var in var_names:
            r_hat = float(az.rhat(self._idata, var_names=[var])[var].values)
            ess_bulk = float(
                az.ess(self._idata, var_names=[var], method="bulk")[var].values
            )
            ess_tail = float(
                az.ess(self._idata, var_names=[var], method="tail")[var].values
            )
            results[var] = {
                "r_hat": r_hat,
                "ess_bulk": ess_bulk,
                "ess_tail": ess_tail,
            }

        return results

    def r_hat(self) -> Dict[str, float]:
        """Return R-hat (potential scale reduction factor) per variable.

        Values close to 1.0 indicate convergence.

        Returns
        -------
        dict
            ``{var_name: r_hat_value}``
        """
        diag = self.diagnostics()
        return {var: d["r_hat"] for var, d in diag.items()}

    def ess(self, method: str = "bulk") -> Dict[str, float]:
        """Return effective sample size per variable.

        Parameters
        ----------
        method : str
            One of ``"bulk"`` (default) or ``"tail"``.

        Returns
        -------
        dict
            ``{var_name: ess_value}``
        """
        if self._idata is None:
            raise RuntimeError(
                "No samples available. Call sample() first."
            )

        az = _import_arviz()

        var_names = self._var_names or list(
            self._idata.posterior.data_vars
        )
        result: Dict[str, float] = {}
        for var in var_names:
            ess_val = float(
                az.ess(self._idata, var_names=[var], method=method)[var].values
            )
            result[var] = ess_val
        return result

    def trace_plot(self, var_names: Optional[List[str]] = None, **kwargs):
        """Generate trace plots for the posterior samples.

        Parameters
        ----------
        var_names : list of str, optional
            Variables to plot. Defaults to all tracked variables.
        **kwargs
            Extra keyword arguments forwarded to ``arviz.plot_trace()``.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.

        Raises
        ------
        RuntimeError
            If :meth:`sample` has not been called yet.
        """
        if self._idata is None:
            raise RuntimeError(
                "No samples available. Call sample() first."
            )

        az = _import_arviz()

        plot_vars = var_names or self._var_names
        return az.plot_trace(self._idata, var_names=plot_vars, **kwargs)

    # ------------------------------------------------------------------ #
    #  Accessors
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> Any:
        """Return the underlying PyMC model."""
        return self._model

    @property
    def inference_data(self) -> Any:
        """Return the ArviZ InferenceData from the last sample() call."""
        return self._idata

    def __repr__(self) -> str:
        vars_str = ", ".join(self._var_names) if self._var_names else "all"
        sampled = self._idata is not None
        return f"MCMCSampler(vars=[{vars_str}], sampled={sampled})"
