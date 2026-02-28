"""Sensitivity analysis for probabilistic flow networks.

Provides functions to compute how sensitive a target variable is
to perturbations in model parameters using central finite differences.
"""

import numpy as np


def sensitivity_analysis(network, target, parameters, perturbation=0.1):
    """Compute sensitivity of a target variable to each parameter.

    Uses central finite differences to estimate the partial derivative
    of the target with respect to each parameter:

        sensitivity ≈ (f(x+h) - f(x-h)) / (2h)

    where h = perturbation * |x| (or perturbation when x == 0).

    Parameters
    ----------
    network : callable
        A callable (function or object with ``__call__``) that accepts a
        dict of parameter values and returns a dict of computed variables.
        Both *network parameters* and *distribution parameters* (e.g.
        ``mu``, ``sigma``) are supported as long as they appear in the
        ``parameters`` dict passed to this function.
    target : str
        Name of the output variable whose sensitivity is measured.
    parameters : dict
        Mapping of parameter names to their current (baseline) values.
        Values must be numeric (int or float).
    perturbation : float, optional
        Fractional perturbation size (default 0.1, i.e. 10 %).
        For a parameter value *x* the step size is
        ``h = perturbation * abs(x)``; when *x* is zero, ``h = perturbation``.

    Returns
    -------
    dict
        ``{param_name: sensitivity_coefficient}`` sorted by descending
        absolute sensitivity.

    Raises
    ------
    KeyError
        If *target* is not found in the dict returned by *network*.
    TypeError
        If *network* is not callable.
    ValueError
        If *parameters* is empty.

    Examples
    --------
    >>> def linear(params):
    ...     return {"y": 3 * params["a"] + 2 * params["b"]}
    >>> result = sensitivity_analysis(linear, "y", {"a": 1.0, "b": 1.0})
    >>> abs(result["a"] - 3.0) < 1e-6
    True
    """
    if not callable(network):
        raise TypeError("network must be callable")
    if not parameters:
        raise ValueError("parameters must be a non-empty dict")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive")

    sensitivities = {}

    for param_name, param_value in parameters.items():
        h = perturbation * abs(param_value) if param_value != 0 else perturbation

        params_plus = dict(parameters)
        params_plus[param_name] = param_value + h

        params_minus = dict(parameters)
        params_minus[param_name] = param_value - h

        output_plus = network(params_plus)
        output_minus = network(params_minus)

        if target not in output_plus:
            raise KeyError(
                f"target '{target}' not found in network output; "
                f"available keys: {list(output_plus.keys())}"
            )

        target_plus = np.asarray(output_plus[target], dtype=float)
        target_minus = np.asarray(output_minus[target], dtype=float)

        derivative = (target_plus - target_minus) / (2.0 * h)

        # Collapse to scalar when possible for cleaner output
        if derivative.ndim == 0:
            derivative = float(derivative)

        sensitivities[param_name] = derivative

    # Sort by descending absolute sensitivity
    def _sort_key(item):
        val = item[1]
        return float(np.max(np.abs(val)))

    sorted_items = sorted(sensitivities.items(), key=_sort_key, reverse=True)
    return dict(sorted_items)
