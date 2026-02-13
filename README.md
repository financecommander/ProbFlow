# ProbFlow

ProbFlow: A Python DSL for probabilistic programming

## Overview

ProbFlow is a Python package that provides a clean and intuitive API for working with probability distributions. It features:

- **Abstract base class** for probability distributions with a consistent interface
- **Continuous distributions**: Normal, LogNormal, Beta
- **Discrete distributions**: Bernoulli, Poisson, Categorical
- **Distribution composition** via operator overloads (`+`, `*`, `&`)
- **Context manager** for probabilistic model definition
- **NumPy backend** via scipy.stats for efficiency
- **Type hints** and comprehensive docstrings

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from probflow import Normal, Beta, ProbFlow

# Create distributions
norm = Normal(loc=0, scale=1)
beta = Beta(alpha=2, beta=5)

# Sample from distributions
samples = norm.sample(1000)

# Compute PDF, CDF, and quantiles
pdf_val = norm.pdf(0)
cdf_val = norm.cdf(0)
median = norm.quantile(0.5)

# Compose distributions
sum_dist = norm + Normal(1, 1)  # Sum of distributions
prod_dist = norm * beta         # Product of distributions
joint_dist = norm & beta        # Joint distribution

# Use context manager for model definition
with ProbFlow() as model:
    prior = Normal(0, 1)
    likelihood = Beta(2, 5)
    model.add_distribution(prior, 'prior')
    model.add_distribution(likelihood, 'likelihood')
```

## Package Structure

```
probflow/
├── __init__.py                 # Main package exports
├── core/
│   ├── __init__.py
│   ├── types.py               # Abstract Dist base class
│   └── context.py             # ProbFlow context manager
└── distributions/
    ├── __init__.py
    ├── continuous.py          # Normal, LogNormal, Beta
    └── discrete.py            # Bernoulli, Poisson, Categorical
```

## Core API

### Distribution Interface

All distributions inherit from the `Dist` abstract base class and implement:

- `sample(n)`: Draw n random samples
- `pdf(x)`: Probability density/mass function at x
- `cdf(x)`: Cumulative distribution function at x
- `quantile(q)`: Quantile function (inverse CDF) at q

### Operator Overloads

- `dist1 + dist2`: Sum of two distributions
- `dist1 * dist2`: Product of two distributions
- `dist1 & dist2`: Joint distribution (assumes independence)

### Context Manager

The `ProbFlow` context manager provides a scope for defining probabilistic models:

```python
with ProbFlow() as model:
    x = Normal(0, 1)
    model.add_distribution(x, 'x')
    # Access: model.get_distribution('x')
```

## Examples

See `examples.py` for comprehensive usage examples including:
- Continuous and discrete distributions
- Distribution composition with operators
- Context manager usage
- Vectorized operations

Run examples:
```bash
python examples.py
```

## Requirements

- numpy>=1.20.0
- scipy>=1.7.0

## License

See LICENSE file for details.
