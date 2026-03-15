from abc import ABC, abstractmethod
import numpy as np
from scipy import stats

class Dist(ABC):
    @abstractmethod
    def sample(self, n=1):
        pass
    
    @abstractmethod
    def log_prob(self, x):
        pass
    
    def __add__(self, other):
        if isinstance(other, Dist):
            return CompositeDist(lambda x, y: x + y, self, other)
        return CompositeDist(lambda x, y: x + y, self, ConstantDist(other))
    
    def __mul__(self, other):
        if isinstance(other, Dist):
            return CompositeDist(lambda x, y: x * y, self, other)
        return CompositeDist(lambda x, y: x * y, self, ConstantDist(other))
    
    def __and__(self, other):
        return JointDist(self, other)

class ConstantDist(Dist):
    def __init__(self, value):
        self.value = value
    def sample(self, n=1):
        return np.full(n, self.value)
    def log_prob(self, x):
        return np.log(x == self.value).astype(float)

class CompositeDist(Dist):
    def __init__(self, op, dist1, dist2):
        self.op = op
        self.dist1 = dist1
        self.dist2 = dist2
    def sample(self, n=1):
        return self.op(self.dist1.sample(n), self.dist2.sample(n))
    def log_prob(self, x):
        raise NotImplementedError("Composite log_prob not implemented")

class JointDist(Dist):
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2
    def sample(self, n=1):
        return np.stack((self.dist1.sample(n), self.dist2.sample(n)), axis=-1)
    def log_prob(self, x):
        return self.dist1.log_prob(x[..., 0]) + self.dist2.log_prob(x[..., 1])

class Normal(Dist):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
    def sample(self, n=1):
        return np.random.normal(self.mu, self.sigma, n)
    def log_prob(self, x):
        return stats.norm.logpdf(x, self.mu, self.sigma)

class LogNormal(Dist):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
    def sample(self, n=1):
        return np.random.lognormal(self.mu, self.sigma, n)
    def log_prob(self, x):
        return stats.lognorm.logpdf(x, self.sigma, scale=np.exp(self.mu))

class Beta(Dist):
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
    def sample(self, n=1):
        return np.random.beta(self.alpha, self.beta, n)
    def log_prob(self, x):
        return stats.beta.logpdf(x, self.alpha, self.beta)
