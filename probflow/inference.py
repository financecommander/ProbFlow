from typing import Dict, List
import numpy as np
from .bayesian import BayesianNetwork

class ExactInference:
    def __init__(self, model: BayesianNetwork):
        self.model = model
    
    def query(self, query: str, evidence: Dict[str, float]) -> np.ndarray:
        # TODO: Implement belief propagation or variable elimination
        return self.model.exact_inference(query, evidence)

class MonteCarloInference:
    def __init__(self, model: BayesianNetwork, n_samples: int = 1000):
        self.model = model
        self.n_samples = n_samples
    
    def query(self, query: str, evidence: Dict[str, float]) -> np.ndarray:
        samples = self.model.sample(self.n_samples)
        # TODO: Filter samples based on evidence and compute query distribution
        return np.histogram(samples[query], density=True)[0]

class MCMCInference:
    def __init__(self, model: BayesianNetwork, n_samples: int = 1000, burn_in: int = 100):
        self.model = model
        self.n_samples = n_samples
        self.burn_in = burn_in
    
    def query(self, query: str, evidence: Dict[str, float]) -> np.ndarray:
        # TODO: Implement Gibbs sampling or Metropolis-Hastings
        samples = self.model.sample(self.n_samples + self.burn_in)
        return np.histogram(samples[query][self.burn_in:], density=True)[0]
