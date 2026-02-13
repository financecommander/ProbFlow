"""Example usage of the ProbFlow package.

This example demonstrates the core features of the ProbFlow package including:
- Creating continuous and discrete distributions
- Sampling, PDF/CDF, and quantile operations
- Distribution composition with operators
- Using the ProbFlow context manager
"""

from probflow import (
    Normal, LogNormal, Beta,
    Bernoulli, Poisson, Categorical,
    ProbFlow
)
import numpy as np


def continuous_distributions_example():
    """Demonstrate continuous distributions."""
    print("=" * 60)
    print("Continuous Distributions Example")
    print("=" * 60)
    
    # Normal distribution
    print("\n1. Normal Distribution")
    norm = Normal(loc=0, scale=1)
    samples = norm.sample(1000)
    print(f"   Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")
    print(f"   PDF at 0: {norm.pdf(0):.4f}")
    print(f"   CDF at 0: {norm.cdf(0):.4f}")
    print(f"   Median: {norm.quantile(0.5):.4f}")
    
    # LogNormal distribution
    print("\n2. LogNormal Distribution")
    lognorm = LogNormal(mu=0, sigma=1)
    samples = lognorm.sample(1000)
    print(f"   Mean: {samples.mean():.3f}")
    print(f"   Median: {lognorm.quantile(0.5):.4f}")
    
    # Beta distribution
    print("\n3. Beta Distribution")
    beta = Beta(alpha=2, beta=5)
    samples = beta.sample(1000)
    print(f"   Mean: {samples.mean():.3f}")
    print(f"   All samples in [0, 1]: {(samples >= 0).all() and (samples <= 1).all()}")


def discrete_distributions_example():
    """Demonstrate discrete distributions."""
    print("\n" + "=" * 60)
    print("Discrete Distributions Example")
    print("=" * 60)
    
    # Bernoulli distribution
    print("\n1. Bernoulli Distribution")
    bern = Bernoulli(p=0.7)
    samples = bern.sample(1000)
    print(f"   Success rate: {samples.mean():.3f}")
    print(f"   P(X=1): {bern.pdf(1):.4f}")
    
    # Poisson distribution
    print("\n2. Poisson Distribution")
    poisson = Poisson(lam=3.5)
    samples = poisson.sample(1000)
    print(f"   Mean: {samples.mean():.3f}")
    print(f"   P(X=3): {poisson.pdf(3):.4f}")
    
    # Categorical distribution
    print("\n3. Categorical Distribution")
    cat = Categorical(probs=[0.2, 0.3, 0.5])
    samples = cat.sample(1000)
    print(f"   Category counts: {[int((samples == i).sum()) for i in range(3)]}")


def operator_overloads_example():
    """Demonstrate operator overloads for distribution composition."""
    print("\n" + "=" * 60)
    print("Distribution Composition Example")
    print("=" * 60)
    
    # Sum of distributions
    print("\n1. Sum of Distributions (X + Y)")
    norm1 = Normal(0, 1)
    norm2 = Normal(1, 1)
    sum_dist = norm1 + norm2
    samples = sum_dist.sample(1000)
    print(f"   X ~ Normal(0, 1), Y ~ Normal(1, 1)")
    print(f"   Mean of X+Y: {samples.mean():.3f} (expected ~1.0)")
    print(f"   Std of X+Y: {samples.std():.3f} (expected ~√2 ≈ 1.41)")
    
    # Product of distributions
    print("\n2. Product of Distributions (X * Y)")
    norm1 = Normal(1, 0.5)
    norm2 = Normal(2, 0.3)
    prod_dist = norm1 * norm2
    samples = prod_dist.sample(1000)
    print(f"   X ~ Normal(1, 0.5), Y ~ Normal(2, 0.3)")
    print(f"   Mean of X*Y: {samples.mean():.3f}")
    print(f"   Std of X*Y: {samples.std():.3f}")
    
    # Joint distribution
    print("\n3. Joint Distribution (X & Y)")
    norm = Normal(0, 1)
    beta = Beta(2, 5)
    joint = norm & beta
    samples = joint.sample(100)
    print(f"   X ~ Normal(0, 1), Y ~ Beta(2, 5)")
    print(f"   Joint samples shape: {samples.shape}")
    print(f"   Mean of X: {samples[:, 0].mean():.3f}")
    print(f"   Mean of Y: {samples[:, 1].mean():.3f}")


def context_manager_example():
    """Demonstrate ProbFlow context manager."""
    print("\n" + "=" * 60)
    print("ProbFlow Context Manager Example")
    print("=" * 60)
    
    print("\nDefining a probabilistic model:")
    with ProbFlow() as model:
        # Define distributions
        prior = Normal(0, 1)
        likelihood = Normal(prior.sample(1)[0], 0.5)
        
        # Register distributions
        model.add_distribution(prior, 'prior')
        model.add_distribution(likelihood, 'likelihood')
        
        print(f"   Context active: {ProbFlow.is_active()}")
        print(f"   Distributions registered: {len(model.distributions)}")
        print(f"   Named variables: {list(model.variables.keys())}")
    
    print(f"\nAfter exiting context:")
    print(f"   Context active: {ProbFlow.is_active()}")


def vectorized_operations_example():
    """Demonstrate vectorized operations."""
    print("\n" + "=" * 60)
    print("Vectorized Operations Example")
    print("=" * 60)
    
    norm = Normal(0, 1)
    
    # Vectorized PDF
    print("\n1. Vectorized PDF")
    x = np.array([-2, -1, 0, 1, 2])
    pdf_vals = norm.pdf(x)
    print(f"   x: {x}")
    print(f"   PDF(x): {pdf_vals}")
    
    # Vectorized CDF
    print("\n2. Vectorized CDF")
    cdf_vals = norm.cdf(x)
    print(f"   x: {x}")
    print(f"   CDF(x): {cdf_vals}")
    
    # Vectorized quantiles
    print("\n3. Vectorized Quantiles")
    q = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    quantiles = norm.quantile(q)
    print(f"   q: {q}")
    print(f"   Quantile(q): {quantiles}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ProbFlow Package Examples")
    print("=" * 60)
    
    continuous_distributions_example()
    discrete_distributions_example()
    operator_overloads_example()
    context_manager_example()
    vectorized_operations_example()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60 + "\n")
