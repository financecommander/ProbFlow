import pytest
import numpy as np
from probflow.distributions import Normal, LogNormal, Beta, Dist

def test_normal_distribution():
    dist = Normal(mu=0, sigma=1)
    samples = dist.sample(1000)
    assert len(samples) == 1000
    assert abs(np.mean(samples)) < 0.1
    assert abs(np.std(samples) - 1.0) < 0.1

def test_lognormal_distribution():
    dist = LogNormal(mu=0, sigma=0.5)
    samples = dist.sample(1000)
    assert len(samples) == 1000
    assert all(samples > 0)

def test_beta_distribution():
    dist = Beta(alpha=2, beta=2)
    samples = dist.sample(1000)
    assert len(samples) == 1000
    assert all(0 <= s <= 1 for s in samples)

def test_operator_overloads():
    dist1 = Normal(0, 1)
    dist2 = Normal(1, 1)
    combined = dist1 + dist2
    samples = combined.sample(100)
    assert len(samples) == 100
    joint = dist1 & dist2
    joint_samples = joint.sample(100)
    assert joint_samples.shape == (100, 2)
