import numpy as np
from typing import Dict, List, Any

class BayesianNetwork:
    def __init__(self):
        self.nodes: Dict[str, Any] = {}
        self.parents: Dict[str, List[str]] = {}
        self.cpts: Dict[str, np.ndarray] = {}
    
    def add_node(self, name: str, parents: List[str], cpt: np.ndarray):
        self.nodes[name] = len(self.nodes)
        self.parents[name] = parents
        self.cpts[name] = cpt
    
    def exact_inference(self, query: str, evidence: Dict[str, int]) -> np.ndarray:
        # Simplified variable elimination for inference
        # TODO: Implement full variable elimination algorithm
        return self.cpts[query][tuple(evidence.get(p, 0) for p in self.parents[query])]
    
    def sample(self, n_samples: int = 1) -> Dict[str, np.ndarray]:
        samples = {n: np.zeros(n_samples) for n in self.nodes}
        topo_order = list(self.nodes.keys())  # Assume topological order for simplicity
        for node in topo_order:
            parents_vals = [samples[p] for p in self.parents[node]]
            if parents_vals:
                idx = tuple(parents_vals)
                probs = self.cpts[node][idx]
            else:
                probs = self.cpts[node]
            samples[node] = np.random.choice(len(probs), size=n_samples, p=probs)
        return samples
