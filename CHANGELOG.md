# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-03

Initial release of ProbFlow -- a Python DSL for probabilistic programming.

### Added

#### Distributions
- Continuous: `Normal`, `LogNormal`, `Beta` with full pdf/cdf/quantile/sample
- Discrete: `Bernoulli`, `Poisson`, `Categorical` with labels support
- Conditional: `ConditionalDist` for CPD modeling with discrete and continuous parents
- Distribution algebra: operator overloading (`+`, `*`, `&`, `|`) for composing distributions
- `Dist` abstract base class for unified distribution interface

#### Bayesian Networks
- CPT-based `BeliefNetwork` with `networkx.DiGraph` backend
- Distribution-based `BeliefNetwork` with ancestral sampling
- Exact inference: belief propagation (polytrees), variable elimination (general DAGs)
- Evidence observation, marginal queries, d-separation testing

#### Causal Inference
- `CausalDAG` with Pearl's do-calculus (`do()` operator)
- Counterfactual reasoning (`counterfactual()`)
- Backdoor/frontdoor criterion identification
- Confounding detection and adjustment
- Interventional sampling

#### Temporal Models
- `MarkovChain`: transition matrices, forecast, stationary distribution
- `HiddenMarkovModel`: Baum-Welch EM fitting, Viterbi decoding

#### Decision Analysis
- `DecisionTree` with backward induction
- `UtilityFunction` factory: linear, exponential (CARA), logarithmic (CRRA)
- DOT export for tree visualization

#### Inference Engines
- `MonteCarloSimulation` with multiprocessing parallelism
- `@simulate` decorator for function-based simulations
- `MCMCSampler` wrapping PyMC NUTS with lazy imports
- `sensitivity_analysis()` via central finite differences

#### Visualization
- `plot_distribution()` and `tornado_chart()` with matplotlib + Plotly backends
- `plot_network()` via Graphviz with critical path highlighting and legend

#### Integrations
- `save_model`/`load_model` with JSON serialization and `allow_pickle` security flag
- `UncertaintyAwareAgent` mixin with VPI-based replanning
- Triton ternary logic bridge (`TernaryValue`, `BinaryDist`)

#### Infrastructure
- GitHub Actions CI (lint, typecheck, test on Python 3.9-3.12)
- Sphinx documentation with Read the Docs theme and GitHub Pages deploy
- pytest-benchmark performance suite
- PyPI publishing via trusted OIDC
