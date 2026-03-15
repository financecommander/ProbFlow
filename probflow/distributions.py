from abc import ABC, abstractmethod
import numpy as np
from scipy import stats

class Dist(ABC):
    """Abstract base class for probability distributions."""
    @abstractmethod
    def sample(self, n=1):
        pass

    @abstractmethod
    def log_prob(self, x):
        pass

    def __add__(self, other):
        if isinstance(other, Dist):
            return CombinedDist(self, other, lambda x, y: x + y)
        return CombinedDist(self, ConstantDist(other), lambda x, y: x + y)

    def __mul__(self, other):
        if isinstance(other, Dist):
            return CombinedDist(self, other, lambda x, y: x * y)
        return CombinedDist(self, ConstantDist(other), lambda x, y: x * y)

    def __and__(self, other):
        if isinstance(other, Dist):
            return JointDist(self, other)
        raise ValueError("& operator only supported between distributions")

class ConstantDist(Dist):
    def __init__(self, value):
        self.value = value

    def sample(self, n=1):
        return np.full(n, self.value)

    def log_prob(self, x):
        return np.log(x == self.value).astype(float)

class CombinedDist(Dist):
    def __init__(self, dist1, dist2, operation):
        self.dist1 = dist1
        self.dist2 = dist2
        self.operation = operation

    def sample(self, n=1):
        return self.operation(self.dist1.sample(n), self.dist2.sample(n))

    def log_prob(self, x):
        raise NotImplementedError("log_prob not implemented for combined distributions")

class JointDist(Dist):
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2

    def sample(self, n=1):
        return np.stack([self.dist1.sample(n), self.dist2.sample(n)], axis=-1)

    def log_prob(self, x):
        return self.dist1.log_prob(x[..., 0]) + self.dist2.log_prob(x[..., 1])

class Normal(Dist):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.norm(mu, sigma)

    def sample(self, n=1):
        return self.dist.rvs(n)

    def log_prob(self, x):
        return self.dist.logpdf(x)

class LogNormal(Dist):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    def sample(self, n=1):
        return self.dist.rvs(n)

    def log_prob(self, x):
        return self.dist.logpdf(x)
