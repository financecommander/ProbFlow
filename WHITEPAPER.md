# ProbFlow: A Compositional Python DSL for Probabilistic Programming, Causal Inference, and Decision-Making Under Uncertainty

**Version 0.1.0**
**February 2026**

---

## Abstract

ProbFlow is a Python domain-specific language (DSL) for probabilistic programming that unifies probability distributions, Bayesian networks, causal inference, temporal models, and decision analysis under a single compositional API. Unlike existing probabilistic programming frameworks that focus narrowly on either model specification or inference, ProbFlow provides an integrated toolkit spanning the full pipeline from distribution definition through causal reasoning to decision-theoretic agent planning. The framework features a unified distribution interface via an abstract base class with operator overloading for natural model composition, dual Bayesian network architectures serving complementary inference paradigms, Pearl-style do-calculus for causal reasoning, Hidden Markov Models for temporal analysis, and an uncertainty-aware agent mixin for AI planning under epistemic uncertainty. ProbFlow comprises approximately 4,800 lines of core library code across 16 modules, backed by NumPy, SciPy, NetworkX, and optional PyMC integration for MCMC sampling.

---

## 1. Introduction

### 1.1 Motivation

Probabilistic reasoning pervades scientific computing, engineering, finance, and artificial intelligence. Practitioners routinely need to define probability distributions, compose them into structured models, perform inference under evidence, reason about causation versus correlation, and make decisions under uncertainty. However, the existing ecosystem fragments these concerns across disparate tools: SciPy provides standalone distributions, PyMC and Stan focus on Bayesian inference, DoWhy handles causal reasoning, and decision-theoretic frameworks exist separately.

ProbFlow addresses this fragmentation by providing a unified Python DSL where distributions, networks, causal models, temporal processes, and decision trees share a common type system and compose naturally through operator overloading and consistent interfaces.

### 1.2 Design Principles

ProbFlow is built on four design principles:

1. **Compositionality**: Distributions compose via operators (`+`, `*`, `&`). Networks compose via node addition. Causal models extend networks. Agents compose over hypotheses and plans.

2. **Dual Abstraction**: Two Bayesian network representations serve different needs — a lightweight distribution-based network for flexible sampling and causal reasoning, and a CPT-based network backed by NetworkX for exact inference algorithms.

3. **Progressive Disclosure**: Users start with simple distribution objects and gradually adopt networks, causal models, or decision trees as their problems require, without framework switches.

4. **Modular Inference**: Inference algorithms (exact, Monte Carlo, MCMC) are separate from model specification, allowing the same model to be analyzed through multiple inference strategies.

---

## 2. Architecture

### 2.1 Package Structure

```
probflow/
├── core/           # Type system and context management
│   ├── types.py    # Dist ABC, Variable, Factor, Node
│   └── context.py  # ProbFlow context manager
├── distributions/  # Probability distributions
│   ├── continuous.py   # Normal, LogNormal, Beta
│   ├── discrete.py     # Bernoulli, Poisson, Categorical
│   └── conditional.py  # ConditionalDist
├── networks/       # Bayesian network representations
│   ├── dag.py      # CPT-based BeliefNetwork (exact inference)
│   └── graph.py    # Network construction utilities
├── causal/         # Causal inference
│   └── dag.py      # CausalDAG with do-calculus
├── inference/      # Inference engines
│   ├── exact.py    # FactorTable, belief propagation, variable elimination
│   ├── sampling.py # Monte Carlo simulation
│   ├── mcmc.py     # PyMC NUTS integration
│   ├── sensitivity.py       # Finite-difference sensitivity analysis
│   ├── belief_network.py    # Distribution-based BeliefNetwork
│   ├── belief_propagation.py # Standalone BP implementation
│   └── monte_carlo.py       # Standalone MC implementation
├── temporal/       # Temporal probabilistic models
│   └── markov.py   # MarkovChain, HiddenMarkovModel
├── decision/       # Decision analysis
│   └── tree.py     # DecisionTree, UtilityFunction
├── integration/    # External system bridges
│   ├── orchestra_agent.py  # UncertaintyAwareAgent mixin
│   ├── triton_bridge.py    # Ternary logic bridge
│   └── serialization.py    # Model persistence (JSON)
├── viz/            # Visualization
│   ├── distributions.py    # Distribution plots (matplotlib/Plotly)
│   └── dag_plot.py         # Network visualization (Graphviz)
└── benchmarks/     # Performance testing
    └── suite.py    # pytest-benchmark suite
```

### 2.2 Dependency Architecture

ProbFlow maintains a minimal required dependency set:

| Dependency | Role | Required |
|------------|------|----------|
| NumPy ≥1.20 | Array computation, sampling | Yes |
| SciPy ≥1.7 | Statistical distributions | Yes |
| NetworkX ≥2.6 | Graph algorithms, d-separation | Yes |
| Matplotlib ≥3.4 | Static visualization | Yes |
| PyMC ≥5.0 | MCMC sampling | Optional |
| Plotly | Interactive visualization | Optional |
| Graphviz | Network diagram rendering | Optional |

---

## 3. Core Type System

### 3.1 The Dist Abstract Base Class

At ProbFlow's foundation is the `Dist` abstract base class, which defines a four-method interface that all probability distributions must implement:

```python
class Dist(ABC):
    def sample(self, n: int) -> np.ndarray: ...
    def pdf(self, x) -> Union[float, np.ndarray]: ...
    def cdf(self, x) -> Union[float, np.ndarray]: ...
    def quantile(self, q) -> Union[float, np.ndarray]: ...
```

This interface supports both continuous and discrete distributions. Discrete distributions implement `pmf()` as their primary mass function and expose `pdf()` as an alias to satisfy the ABC contract. This design enables polymorphic code that operates on any distribution type without branching on discrete versus continuous.

### 3.2 Operator Overloading for Composition

The `Dist` class defines three composition operators:

- **Addition** (`dist1 + dist2`): Creates a `SumDist` representing the distribution of X + Y for independent X, Y. Sampling is exact; PDF computation falls back to empirical quantile estimation.

- **Multiplication** (`dist1 * dist2`): Creates a `ProductDist` representing X × Y. Specific distribution subclasses may override this — for example, `Normal.__mul__(scalar)` returns a new Normal with closed-form parameters, and `LogNormal.__mul__(scalar)` shifts the location parameter.

- **Joint** (`dist1 & dist2`): Creates a `JointDist` representing the joint distribution under independence, with column-stacked samples.

### 3.3 Factor Graph Types

For discrete Bayesian network inference, ProbFlow defines three dataclass types:

- **Variable**: A discrete random variable with named states (e.g., `Variable("Weather", ["sunny", "rainy"])`).

- **Factor**: A potential function over variables, stored as an N-dimensional NumPy array with shape validation against variable cardinalities.

- **Node**: A factor graph node linking a variable to its parents, children, prior distribution, and conditional probability table.

These types serve as the low-level representation consumed by exact inference algorithms.

---

## 4. Probability Distributions

### 4.1 Continuous Distributions

ProbFlow provides three continuous distributions backed by SciPy:

| Distribution | Parameters | Special Features |
|-------------|------------|------------------|
| **Normal** | μ (mean), σ (std dev) | Closed-form convolution via `+`, affine scaling via `*`, degenerate case (σ=0) |
| **LogNormal** | μ, σ (of underlying normal) | Positive-scalar scaling shifts μ by ln(c) |
| **Beta** | α, β (shape), loc, scale | Support scaling via `*` adjusts loc and scale |

All continuous distributions implement `mean()` and `variance()` in addition to the core Dist interface. Degenerate cases (zero variance) are handled explicitly, returning point-mass distributions.

### 4.2 Discrete Distributions

| Distribution | Parameters | Special Features |
|-------------|------------|------------------|
| **Bernoulli** | p (success probability) | `&` (joint), `\|` (union) operators, `mode()` |
| **Poisson** | λ (rate) | `mode()` returns floor(λ) |
| **Categorical** | probs, labels (optional) | Label-based indexing, label-aware sampling and PMF |

The `Categorical` distribution supports an optional `labels` parameter, enabling string-valued samples and label-based PMF evaluation — a feature critical for integration with the causal inference module where variables take named states.

A standalone `probability()` helper function computes threshold probabilities (P(X > t), P(X ≥ t), etc.) for any discrete distribution.

### 4.3 Conditional Distributions

The `ConditionalDist` class parameterizes a child distribution on a parent variable:

```python
regime = Categorical([0.6, 0.4], labels=['bull', 'bear'])
volatility = ConditionalDist(
    parent=regime,
    mapping={'bull': Normal(1, 0.3), 'bear': Normal(2, 0.5)},
)
```

`ConditionalDist` automatically detects parent type:
- **Discrete parents**: Exact key lookup in the mapping dictionary.
- **Continuous parents**: Binning via `np.digitize()` with sorted numeric keys as bin edges.

Sampling draws from the parent distribution, then dispatches each sample to the appropriate child distribution based on the parent's realized value.

---

## 5. Bayesian Networks

ProbFlow provides two complementary Bayesian network implementations, each optimized for different inference paradigms.

### 5.1 Distribution-Based Network (Sampling & Causal Inference)

Located in `probflow.inference.belief_network`, this lightweight `BeliefNetwork` stores distribution objects directly:

```python
bn = BeliefNetwork()
bn.add_node('regime', Categorical([0.6, 0.4], labels=['bull', 'bear']))
bn.add_node('volatility', ConditionalDist(...), parents=['regime'])
samples = bn.sample(10000)  # Ancestral (forward) sampling
```

Nodes are stored in an `OrderedDict` enforcing topological insertion order. Ancestral sampling iterates nodes in order, sampling root nodes marginally and child nodes conditionally on their parents.

This architecture supports continuous and discrete variables, arbitrary distribution types, and serves as the base class for `CausalDAG`.

### 5.2 CPT-Based Network (Exact Inference)

Located in `probflow.networks.dag`, this `BeliefNetwork` uses NetworkX's `DiGraph` and stores conditional probability tables as NumPy arrays:

```python
bn = BeliefNetwork()
bn.add_node('Rain', distribution=np.array([0.2, 0.8]), states=['yes', 'no'])
bn.add_node('Sprinkler', distribution=np.array([[0.01, 0.99], [0.4, 0.6]]),
            parents=['Rain'], states=['on', 'off'])
```

This representation enables:

- **Evidence observation**: `bn.observe('Rain', 'yes')` conditions the network.
- **Marginal queries**: `bn.marginal('Sprinkler')` via variable elimination.
- **Conditional inference**: `bn.infer('Grass')` computes posterior given evidence, automatically selecting belief propagation for trees and falling back to variable elimination for general DAGs.
- **D-separation testing**: `bn.d_separated('X', 'Y', {'Z'})` via NetworkX.

---

## 6. Inference Engines

### 6.1 Exact Inference

The `FactorTable` class implements discrete factor operations — multiplication (with automatic axis alignment), marginalization (sum over variable), reduction (evidence conditioning), and normalization. Two exact inference algorithms operate on these factors:

**Belief Propagation**: A two-pass collect-distribute algorithm for tree-structured (polytree) networks. The upward pass collects messages from leaves to root; the downward pass distributes information back. Runtime is O(n·k²) for n nodes with at most k states.

**Variable Elimination**: A general-purpose algorithm that eliminates hidden variables by multiplying involved factors and summing out. Works on arbitrary DAGs, including those with undirected cycles. Variable ordering is determined by the topological structure.

The CPT-based `BeliefNetwork` automatically detects whether the moral graph contains cycles and selects the appropriate algorithm.

### 6.2 Monte Carlo Simulation

The `MonteCarloSimulation` class provides parallel Monte Carlo sampling:

```python
@simulate(n_samples=10000, seed=42, n_workers=4)
def portfolio_return():
    market = Normal(0.08, 0.15).sample(1)[0]
    alpha = Normal(0.02, 0.05).sample(1)[0]
    return market + alpha

results = portfolio_return()
print(results.mean(), results.quantile(0.05))
```

The `@simulate` decorator transforms any callable returning a float into a Monte Carlo estimator. Multi-process parallelism via `multiprocessing.Pool` with deterministic seed splitting ensures reproducibility.

`SimulationResults` provides `mean()`, `std()`, `quantile()`, and `histogram()` for result analysis.

### 6.3 MCMC via PyMC

The `MCMCSampler` adapts ProbFlow's distribution-based networks to PyMC's NUTS sampler:

```python
sampler = MCMCSampler.from_network(belief_network)
trace = sampler.sample(n_samples=2000, tune=1000, chains=4)
diagnostics = sampler.diagnostics()  # R-hat, ESS
```

An internal dispatch map translates ProbFlow distribution types (Normal, Beta, Categorical) to their PyMC equivalents. Convergence diagnostics (R-hat, effective sample size) and trace visualization are provided through ArviZ integration.

PyMC is lazily imported — the module raises a descriptive `ImportError` if PyMC is not installed, keeping it as a true optional dependency.

### 6.4 Sensitivity Analysis

The `sensitivity_analysis()` function computes partial derivatives of model outputs with respect to parameters using central finite differences:

∂f/∂x ≈ (f(x + h) − f(x − h)) / 2h

where h = perturbation × |x|. Results are returned sorted by descending absolute sensitivity, enabling rapid identification of the most influential parameters.

---

## 7. Causal Inference

### 7.1 The CausalDAG

`CausalDAG` extends the distribution-based `BeliefNetwork` with Pearl's causal hierarchy:

**Do-Operator (Interventions)**: `dag.do('Treatment', 'active')` performs graph surgery — removing all incoming edges to the intervened variable and fixing it to a degenerate distribution. The method returns a new `CausalDAG`, leaving the original unchanged (immutability).

**Counterfactual Reasoning**: `dag.counterfactual(intervention, evidence, n)` implements the three-step twin network procedure:
1. **Abduction**: Filter samples from the observational distribution that match the evidence.
2. **Action**: Construct the interventional network via `do()`.
3. **Prediction**: Resample non-intervened variables conditioned on the abducted context.

**Effect Identification**: `dag.identify_effect(treatment, outcome)` checks whether a causal effect is identifiable from observational data via:
- **Backdoor criterion**: Finds a set of variables that blocks all backdoor paths between treatment and outcome.
- **Frontdoor criterion**: Identifies mediator variables when backdoor adjustment is not possible.

**Confounding Detection**: `dag.find_confounders(treatment, outcome)` identifies common ancestors that could confound the treatment-outcome relationship.

### 7.2 Causal Example: Simpson's Paradox

```python
dag = CausalDAG()
dag.add_node('Gender', Categorical([0.5, 0.5], labels=['M', 'F']))
dag.add_node('Department', ConditionalDist(parent=..., mapping=...),
             parents=['Gender'])
dag.add_node('Admitted', ConditionalDist(parent=..., mapping=...),
             parents=['Gender', 'Department'])

# Observational: P(Admitted | Gender)
obs = dag.sample(10000)

# Interventional: P(Admitted | do(Gender))
intv = dag.do('Gender', 'F').sample(10000)

# These may differ — revealing Simpson's paradox
```

---

## 8. Temporal Models

### 8.1 Markov Chains

The `MarkovChain` class models discrete-time stochastic processes with the Markov property:

```python
mc = MarkovChain(
    states=['bull', 'bear', 'stagnant'],
    transition_matrix=np.array([[0.8, 0.1, 0.1],
                                 [0.2, 0.6, 0.2],
                                 [0.3, 0.2, 0.5]])
)
forecast = mc.forecast(horizon=10, initial_state='bull')
equilibrium = mc.stationary_distribution()
```

`forecast()` projects state distributions forward via matrix exponentiation. `stationary_distribution()` finds the equilibrium via left eigenvector decomposition of the transition matrix at eigenvalue 1.

### 8.2 Hidden Markov Models

The `HiddenMarkovModel` implements the three classical HMM algorithms:

- **Forward-Backward** (internal): Scaled forward/backward probabilities with log-space stability for observation likelihood computation.

- **Baum-Welch** (`fit()`): Expectation-Maximization algorithm that iteratively refines initial, transition, and emission probabilities until log-likelihood convergence within a specified tolerance.

- **Viterbi** (`infer_state()`): Dynamic programming in log-space with backtracking for most-likely hidden state sequence decoding.

---

## 9. Decision Analysis

### 9.1 Decision Trees

The `DecisionTree` class supports construction and backward-induction solution of decision trees:

```python
tree = DecisionTree()
tree.add_decision('invest', choices=['stocks', 'bonds'])
tree.add_chance('market', outcomes=['up', 'down'], probs=[0.6, 0.4])
tree.set_payoff(('invest', 'stocks', 'market', 'up'), 150)
tree.set_payoff(('invest', 'stocks', 'market', 'down'), -50)

result = tree.solve(utility_function=UtilityFunction.exponential(0.01))
```

Backward induction maximizes expected utility at decision nodes and computes probability-weighted expected values at chance nodes. For non-linear utility functions, certainty equivalents are computed via numerical bisection.

### 9.2 Utility Functions

The `UtilityFunction` factory provides three standard utility functions:

| Type | Formula | Risk Profile |
|------|---------|-------------|
| **Linear** | u(x) = x | Risk-neutral |
| **Exponential** (CARA) | u(x) = 1 − e^(−ax) | Constant absolute risk aversion |
| **Logarithmic** (CRRA) | u(x) = ln(w + x) | Constant relative risk aversion |

### 9.3 Visualization

`DecisionTree.to_dot()` exports the solved tree to Graphviz DOT format with visual differentiation: squares for decision nodes, circles for chance nodes, and triangles for terminal payoffs.

---

## 10. Agent Integration

### 10.1 UncertaintyAwareAgent Mixin

The `UncertaintyAwareAgent` abstract mixin enables any agent class to incorporate probabilistic reasoning into its planning:

```python
class MyAgent(UncertaintyAwareAgent):
    def generate_hypotheses(self, goal, env): ...
    def generate_plans(self, goal, env, hypothesis): ...
    def estimate_success_probability(self, plan, hyp, env): ...
    def compute_utility(self, plan, hyp, env): ...
```

The core `plan_under_uncertainty()` method:
1. Generates competing world-state hypotheses with prior probabilities.
2. For each hypothesis, generates candidate plans.
3. Scores each plan by hypothesis-weighted expected utility.
4. Returns ranked plans sorted by expected utility.

The `adaptive_replanning_threshold()` method implements value-of-perfect-information (VPI) analysis: if VPI exceeds the cost of gathering more information, the agent should replan rather than execute.

### 10.2 Triton Bridge

The `triton_bridge` module bridges ternary (three-valued) logic systems with probabilistic distributions:

- `TernaryValue.TRUE` maps to `Bernoulli(0.95)` (high confidence)
- `TernaryValue.FALSE` maps to `Bernoulli(0.05)` (high negative confidence)
- `TernaryValue.UNKNOWN` maps to `Bernoulli(0.5)` (maximum uncertainty)

The `BinaryDist` wrapper enables compositional boolean operations (`&`, `|`) on uncertain binary propositions while maintaining probability semantics.

---

## 11. Visualization

### 11.1 Distribution Plotting

`plot_distribution()` renders PDF, CDF, or both for one or more distributions with:
- Colorblind-safe 8-color palette (Wong 2011).
- Inter-quantile shaded regions for uncertainty visualization.
- Dual backends: Matplotlib (static, publication-quality) and Plotly (interactive, web-embeddable).

### 11.2 Sensitivity Visualization

`tornado_chart()` produces horizontal bar charts of sensitivity coefficients, sorted by absolute magnitude with optional threshold filtering. Useful for communicating which parameters most influence model outputs.

### 11.3 Network Visualization

`plot_network()` renders probabilistic networks as DAGs via Graphviz with:
- Visual node typing: observed (gray box), query (bold), latent (ellipse).
- Edge labels showing conditional probabilities.
- Critical path highlighting (longest dependency chain) in red.
- Auto-generated legend subgraph.

---

## 12. Model Persistence

The serialization module provides JSON-based model persistence:

```python
save_model(network, 'model.json')
loaded = load_model('model.json', allow_pickle=False)
```

The format stores node definitions (states, parents, distribution data) and edges. Array-based CPTs serialize as nested JSON lists; callable CPTs fall back to base64-encoded pickle with an explicit security gate (`allow_pickle=True` required).

Multi-stage validation on load checks DAG acyclicity, probability range constraints, and row-sum normalization. Format versioning supports backward-compatible migration from earlier schema versions.

---

## 13. Testing and Benchmarks

ProbFlow includes comprehensive testing:

- **428 unit tests** covering all modules (distributions, networks, inference, causal, temporal, decision, integration, visualization, serialization).
- **Statistical convergence tests**: Monte Carlo estimators verified against analytical values within tolerance bounds.
- **Performance benchmarks** via pytest-benchmark:
  - Belief propagation on 10-node tree: < 1ms target.
  - Monte Carlo 10K samples: < 50ms target.
  - 100-node network inference: < 10MB memory target.
- **Cross-validation**: Belief propagation vs. Monte Carlo vs. variable elimination consistency checks.
- **CI/CD**: GitHub Actions testing across Python 3.9, 3.10, and 3.11 with linting (Ruff), formatting (Black), and strict type checking (MyPy).

---

## 14. Related Work

| Framework | Focus | ProbFlow Differentiator |
|-----------|-------|------------------------|
| **PyMC** | Bayesian inference via MCMC/VI | ProbFlow integrates PyMC as an optional inference backend while adding causal reasoning, decision analysis, and agent planning |
| **Stan** | High-performance Bayesian inference | ProbFlow prioritizes Pythonic composability over raw sampling performance |
| **DoWhy** | Causal inference | ProbFlow embeds causal DAGs within a broader probabilistic framework including temporal models and decision trees |
| **pgmpy** | Probabilistic graphical models | ProbFlow provides a higher-level DSL with operator overloading and dual network architectures |
| **TensorFlow Probability** | Deep probabilistic models | ProbFlow targets interpretable probabilistic reasoning over neural integration |

---

## 15. Future Directions

Several extensions are planned:

1. **Continuous Bayesian networks**: Extending the CPT-based network to support Gaussian conditional distributions with exact inference via moment matching.

2. **Variational inference**: Adding ELBO-based approximate inference as an alternative to exact and MCMC methods.

3. **Causal discovery**: Implementing constraint-based (PC algorithm) and score-based (GES) structure learning from observational data.

4. **Multi-agent decision models**: Extending decision trees to game-theoretic settings with multiple agents and Nash equilibrium computation.

5. **GPU acceleration**: Optional PyTorch backend for large-scale Monte Carlo simulation and gradient-based optimization.

6. **Interactive model builder**: Web-based visual interface for constructing and querying probabilistic models, built on the Plotly visualization backend.

---

## 16. Conclusion

ProbFlow provides a unified Python DSL that bridges probability distributions, Bayesian networks, causal inference, temporal models, and decision analysis. By grounding all components in a common type system with operator overloading and consistent interfaces, ProbFlow enables practitioners to move fluidly between modeling, inference, causal reasoning, and decision-making within a single framework. The dual network architecture — lightweight distribution-based networks for flexibility and CPT-based networks for exact inference — ensures that the framework scales from simple distribution arithmetic to complex causal counterfactual analysis without sacrificing usability or performance.

---

## Appendix A: Installation

```bash
# Core installation
pip install numpy scipy networkx matplotlib

# Optional: MCMC support
pip install pymc arviz

# Optional: Interactive visualization
pip install plotly

# Optional: Network diagrams
pip install graphviz  # Python package
# Also requires Graphviz system binaries on PATH

# Development
pip install pytest pytest-benchmark black mypy ruff pre-commit
```

## Appendix B: Quick Reference

```python
import probflow as pf

# Distributions
x = pf.Normal(0, 1)
y = pf.Beta(2, 5)
z = x + pf.Normal(1, 1)       # Sum distribution
joint = x & y                   # Joint distribution

# Sampling & Statistics
samples = x.sample(10000)
print(x.pdf(0), x.cdf(0), x.quantile(0.95))

# Bayesian Network (lightweight)
from probflow.inference.belief_network import BeliefNetwork
from probflow.distributions.conditional import ConditionalDist

bn = BeliefNetwork()
bn.add_node('cause', pf.Categorical([0.3, 0.7], labels=['A', 'B']))
bn.add_node('effect', ConditionalDist(parent=..., mapping=...), parents=['cause'])

# Causal Inference
from probflow.causal.dag import CausalDAG
dag = CausalDAG()
# ... add nodes ...
interventional = dag.do('Treatment', 'active')
counterfactual = dag.counterfactual({'Treatment': 'active'}, {'Outcome': 'bad'})

# Decision Analysis
from probflow.decision.tree import DecisionTree, UtilityFunction
tree = DecisionTree()
# ... build tree ...
result = tree.solve(UtilityFunction.exponential(risk_aversion=0.01))

# Monte Carlo
from probflow.inference.sampling import simulate
@simulate(n_samples=10000, seed=42)
def model():
    return pf.Normal(0.08, 0.15).sample(1)[0]

# Visualization
from probflow.viz.distributions import plot_distribution
plot_distribution([pf.Normal(0,1), pf.Normal(2,1)], labels=['Prior','Posterior'])
```
