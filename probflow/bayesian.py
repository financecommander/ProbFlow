import numpy as np
from typing import Dict, List, Tuple

class BayesianNetwork:
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str]], cpts: Dict[str, np.ndarray]):
        self.nodes = nodes
        self.edges = edges
        self.cpts = cpts  # Conditional Probability Tables
        self.parents = {n: [p for p, c in edges if c == n] for n in nodes}

    def infer(self, evidence: Dict[str, float]) -> Dict[str, np.ndarray]:
        # Simple exact inference placeholder using joint probability factorization
        result = {}
        for node in self.nodes:
            if node in evidence:
                result[node] = np.array([1.0 if v == evidence[node] else 0.0 for v in range(2)])
            else:
                result[node] = self._compute_marginal(node, evidence)
        return result

    def _compute_marginal(self, node: str, evidence: Dict[str, float]) -> np.ndarray:
        # Placeholder for marginal computation
        return np.array([0.5, 0.5])

    def sample(self, n: int = 1) -> Dict[str, np.ndarray]:
        samples = {node: np.zeros(n) for node in self.nodes}
        for i in range(n):
            for node in self._topological_sort():
                parents = self.parents[node]
                parent_vals = tuple(samples[p][i] for p in parents)
                cpt = self.cpts[node]
                prob = cpt[parent_vals] if parents else cpt
                samples[node][i] = np.random.binomial(1, prob)
        return samples

    def _topological_sort(self) -> List[str]:
        return self.nodes  # Simplified, assumes nodes are already sorted
