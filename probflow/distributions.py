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
        return CompositeDist(lambda x, y: x + y, self, other)
    
    def __mul__(self, other):
        if isinstance(other, Dist):
            return CompositeDist(lambda x, y: x * y, self, other)
        return CompositeDist(lambda x, y: x * y, self, other)
    
    def __and__(self, other):
        return JointDist(self, other)

class CompositeDist(Dist):
    def __init__(self, op, dist1, dist2):
        self.op = op
        self.dist1 = dist1
        self.dist2 = dist2
    
    def sample(self, n=1):
        s1 = self.dist1.sample(n)
        s2 = self.dist2.sample(n) if isinstance(self.dist2, Dist) else self.dist2
        return self.op(s1, s2)
    
    def log_prob(self, x):
        raise NotImplementedError('Log prob not implemented for composite distributions')

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
        self.dist = stats.norm(loc=mu, scale=sigma)
    
    def sample(self, n=1):
        return self.dist.rvs(size=n)
    
    def log_prob(self, x):
        return self.dist.logpdf(x)

class LogNormal(Dist):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    
    def sample(self, n=1):
        return self.dist.rvs(size=n)
    
    def log_prob(self, x):
        return self.dist.logpdf(x)

class Beta(Dist):
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.dist = stats.beta(a=alpha, b=beta)
    
    def sample(self, n=1):
        return self.dist.rvs(size=n)
    
    def log_prob(self, x):
        return self.dist.logpdf(x)
