import numpy as np
from typing import Dict, List, Tuple

class BayesianNetwork:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: Dict[str, List[str]] = {}

    def add_node(self, name: str, cpt: np.ndarray, parents: List[str] = None):
        """Add a node with its conditional probability table (CPT)."""
        self.nodes[name] = {"cpt": cpt}
        self.edges[name] = parents if parents else []

    def infer_exact(self, evidence: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Exact inference using variable elimination."""
        # TODO: Implement full variable elimination algorithm
        result = {}
        for node in self.nodes:
            if node not in evidence:
                result[node] = np.ones(self.nodes[node]['cpt'].shape[0]) / self.nodes[node]['cpt'].shape[0]
        return result

    def sample(self, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Sample from the network using ancestral sampling."""
        samples = {node: np.zeros(n_samples) for node in self.nodes}
        ordered_nodes = self._topological_sort()
        for node in ordered_nodes:
            parents = self.edges[node]
            if parents:
                parent_samples = [samples[p] for p in parents]
                # TODO: Implement conditional sampling based on CPT
                samples[node] = np.random.choice(len(self.nodes[node]['cpt']), size=n_samples)
            else:
                probs = self.nodes[node]['cpt']
                samples[node] = np.random.choice(len(probs), p=probs, size=n_samples)
        return samples

    def _topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        # TODO: Implement proper topological sort
        return list(self.nodes.keys())
