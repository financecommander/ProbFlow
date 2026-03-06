# ProbFlow

A compositional Python DSL for probabilistic programming, causal inference, and decision-making under uncertainty.

## Overview

ProbFlow unifies probability distributions, Bayesian networks, causal inference, temporal models, and decision analysis under a single compositional API. Instead of switching between disparate tools for each concern, ProbFlow lets you move fluidly from distribution arithmetic to causal counterfactuals to decision-theoretic planning.

**Key capabilities:**

- **Distributions** -- Normal, LogNormal, Beta, Bernoulli, Poisson, Categorical with a unified `Dist` ABC
- **Composition** -- operator overloads (`+`, `*`, `&`) for building complex distributions
- **Bayesian networks** -- dual architecture: CPT-based (exact inference) and distribution-based (sampling/causal)
- **Causal inference** -- Pearl's do-calculus, counterfactuals, backdoor/frontdoor identification
- **Temporal models** -- Markov chains and Hidden Markov Models (Baum-Welch, Viterbi)
- **Decision analysis** -- decision trees with backward induction and utility functions
- **Inference engines** -- exact (belief propagation, variable elimination), Monte Carlo, MCMC (PyMC)
- **Agent integration** -- uncertainty-aware planning mixin with value-of-information
- **Visualization** -- distribution plots, tornado charts, network DAG diagrams

## Installation

```bash
# Core
pip install numpy scipy networkx matplotlib

# Optional: MCMC sampling
pip install pymc arviz

# Optional: Interactive plots
pip install plotly

# Optional: Network diagrams
pip install graphviz

# Development
pip install -e ".[dev]"
```

## Quick Start

```python
from probflow import Normal, Beta, Bernoulli, Categorical

# Create and sample distributions
x = Normal(mu=0, sigma=1)
samples = x.sample(10000)
print(x.pdf(0), x.cdf(0), x.quantile(0.95))

# Compose distributions with operators
z = Normal(0, 1) + Normal(2, 0.5)   # Sum (closed-form convolution)
joint = Normal(0, 1) & Beta(2, 5)    # Joint distribution
scaled = 3 * Normal(0, 1)            # Affine scaling

# Conditional distributions
from probflow.distributions.conditional import ConditionalDist

regime = Categorical([0.6, 0.4], labels=['bull', 'bear'])
vol = ConditionalDist(
    parent=regime,
    mapping={'bull': Normal(1, 0.3), 'bear': Normal(2, 0.5)},
)

# Bayesian network with ancestral sampling
from probflow.inference.belief_network import BeliefNetwork

bn = BeliefNetwork()
bn.add_node('regime', regime)
bn.add_node('volatility', vol, parents=['regime'])
joint_samples = bn.sample(10000)
```

## Causal Inference

```python
from probflow.causal.dag import CausalDAG

dag = CausalDAG()
dag.add_node('Treatment', Categorical([0.5, 0.5], labels=['drug', 'placebo']))
dag.add_node('Outcome', ConditionalDist(...), parents=['Treatment'])

# Interventional query: P(Outcome | do(Treatment = drug))
intervened = dag.do('Treatment', 'drug')
samples = intervened.sample(5000)

# Counterfactual: What if treatment had been different?
cf = dag.counterfactual(
    intervention={'Treatment': 'drug'},
    evidence={'Outcome': 'bad'},
    n=5000,
)

# Identifiability check
result = dag.identify_effect('Treatment', 'Outcome')
print(result['method'], result['adjustment_set'])
```

## Decision Analysis

```python
from probflow.decision.tree import DecisionTree, UtilityFunction

tree = DecisionTree()
tree.add_decision('invest', choices=['stocks', 'bonds'])
tree.add_chance('market', outcomes=['up', 'down'], probs=[0.6, 0.4])
tree.set_payoff(('invest', 'stocks', 'market', 'up'), 150)
tree.set_payoff(('invest', 'stocks', 'market', 'down'), -50)
tree.set_payoff(('invest', 'bonds', 'market', 'up'), 40)
tree.set_payoff(('invest', 'bonds', 'market', 'down'), 40)

result = tree.solve(UtilityFunction.exponential(risk_aversion=0.01))
print(result['strategy'])      # Optimal policy
print(result['expected_value']) # Expected payoff
```

## Monte Carlo Simulation

```python
from probflow.inference.sampling import simulate

@simulate(n_samples=10000, seed=42, n_workers=4)
def portfolio_return():
    market = Normal(0.08, 0.15).sample(1)[0]
    alpha = Normal(0.02, 0.05).sample(1)[0]
    return market + alpha

results = portfolio_return()
print(f"Mean: {results.mean():.4f}, VaR 5%: {results.quantile(0.05):.4f}")
```

## Package Structure

```
probflow/
├── core/               # Type system: Dist ABC, Variable, Factor, Node
├── distributions/      # Normal, LogNormal, Beta, Bernoulli, Poisson, Categorical, ConditionalDist
├── networks/           # CPT-based BeliefNetwork (exact inference)
├── causal/             # CausalDAG with do-calculus and counterfactuals
├── inference/          # Exact (BP, VE), Monte Carlo, MCMC, sensitivity analysis
├── temporal/           # MarkovChain, HiddenMarkovModel
├── decision/           # DecisionTree, UtilityFunction
├── integration/        # UncertaintyAwareAgent, Triton bridge, serialization
├── viz/                # Distribution plots, tornado charts, DAG visualization
└── benchmarks/         # Performance test suite
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_continuous.py -v
python -m pytest tests/test_causal_dag.py -v
```

## Dependencies

| Package | Version | Required |
|---------|---------|----------|
| numpy | >= 1.20 | Yes |
| scipy | >= 1.7 | Yes |
| networkx | >= 2.6 | Yes |
| matplotlib | >= 3.4 | Yes |
| pymc | >= 5.0 | Optional (MCMC) |
| plotly | -- | Optional (interactive viz) |
| graphviz | -- | Optional (DAG rendering) |

## Documentation

See `WHITEPAPER.md` for a comprehensive technical overview covering architecture, design principles, and all modules.

See `docs/` for Sphinx API reference, theory guides, and Jupyter notebook tutorials.

---

## Shapeshifter Architecture — Three-Layer Model

This repository is part of the **Shapeshifter Orchestration Model**, a three-layer architecture for adaptive multi-agent workflows, distributed execution, and compression-aware model routing.

```
Adaptive Layer        → Shapeshifter (system-wide orchestration model)
Control Plane         → super-duper-spork + Orchestra + AI-PORTAL
Execution Plane       → Triton + BUNNY + distributed workers
```

### Role of `ProbFlow`

**Layer:** Adaptive Layer

ProbFlow provides the **routing intelligence** for the Adaptive Layer.

Responsibilities within the Shapeshifter architecture:

- uncertainty scoring for task classification
- routing optimization (workflow shape, compression profile, escalation policy)
- confidence scoring for validation routing
- expected value estimation for cost-optimal model selection
- telemetry regression for auto-tuning routing weights

### Repository Responsibility Matrix

| Layer     | Component        | Repository          | Role                            |
| --------- | ---------------- | ------------------- | ------------------------------- |
| Adaptive  | Routing & Policy | `ProbFlow`          | uncertainty scoring and routing |
| Control   | Workflow DSL     | `Orchestra`         | task graph definition           |
| Control   | Swarm Runtime    | `super-duper-spork` | scheduling and orchestration    |
| Execution | Model Runtime    | `Triton`            | AI inference and compression    |
| Execution | Worker Runtime   | `BUNNY`             | distributed execution           |
| Interface | UI & Telemetry   | `AI-PORTAL`         | monitoring and user interaction |

### Execution Lifecycle

```
Task Input → Task Classification → Workflow Selection → Task Graph Compilation
    → Worker Dispatch → Model Execution → Validation → Result Synthesis
```

### Telemetry Feedback Loop

```
Triton runtime metrics → AI-PORTAL dashboards → ProbFlow routing models → Shapeshifter policy updates
```

### Goals & Roadmap

See the [Orchestra README](https://github.com/financecommander/Orchestra#architecture-overview-shapeshifter-orchestration-model) for the full architecture specification, goals, and 7-phase roadmap.

> **Engineering principle:** Use the smallest reliable model and workflow capable of completing the task safely.

## Competitive Projection: System-Level Intelligence

The Shapeshifter architecture competes with frontier LLM systems on **system-level capability**, not single-model capability.

**ProbFlow role:** Adaptive Layer — Routing Intelligence

ProbFlow's uncertainty scoring and routing optimization determine whether the system correctly decomposes tasks, selects compression profiles, and sizes worker pools. Routing accuracy is the first condition for competitiveness — if routing fails, the system cannot compete.

### System Benchmark Matrix

| Category | Frontier LLM | Shapeshifter Projection |
|---|---|---|
| Coding | very strong | equal or better |
| Large repo analysis | weak | stronger |
| Structured reasoning | strong | equal |
| Operational automation | weak | stronger |
| Research synthesis | strong | equal |
| Cost efficiency | weak | stronger |
| Scalability | limited | far stronger |

### Advantage Mechanisms

| Mechanism | Frontier LLM | Shapeshifter |
|---|---|---|
| Reasoning | sequential | parallel workers |
| Task decomposition | single-pass | explicit planner |
| Validation | self-consistency | layered (static → tests → review → human) |
| Compute | massive GPUs | compressed + ternary + edge workers |

### Simulation Projection

| Metric | Frontier | Shapeshifter |
|---|---|---|
| Reasoning depth | 95 | 85 |
| Parallel execution | 30 | 95 |
| Validation reliability | 60 | 90 |
| Cost efficiency | 40 | 90 |
| Scalability | 35 | 95 |

Frontier models win on deep reasoning. Shapeshifter wins on system-level capability.

> **Shapeshifter is not a model competitor. It is an orchestration architecture that multiplies model capability.**

See the [Orchestra README](https://github.com/financecommander/Orchestra#competitive-projection-system-level-intelligence) for the full competitive projection, benchmark categories, demonstration strategy, and long-term advantage analysis.



---

## License

See LICENSE file for details.
